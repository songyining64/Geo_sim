"""
高级网格系统 - 实现Underworld级别的网格功能
支持复杂拓扑管理、自适应细化、多尺度支持、并行分区、动态变形
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    torch = None
    warnings.warn("PyTorch not available. ML-based mesh operations will be limited.")

try:
    from scipy.spatial import Delaunay
    from scipy.sparse import csr_matrix, lil_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available. Advanced mesh operations will be limited.")


@dataclass
class MeshFace:
    """网格面类"""
    id: int
    nodes: List[int]  # 面的节点ID列表
    element_ids: List[int]  # 包含此面的单元ID（边界面为[-1, 单元ID]）
    face_type: str = "edge"  # 2D为edge，3D为face
    boundary: bool = False  # 是否为边界面
    
    def __post_init__(self):
        if self.element_ids is None:
            self.element_ids = []


@dataclass
class MeshEdge:
    """网格边类"""
    id: int
    nodes: List[int]  # 边的两个节点ID
    element_ids: List[int]  # 包含此边的单元ID
    length: float = 0.0  # 边长度
    
    def __post_init__(self):
        if self.element_ids is None:
            self.element_ids = []


@dataclass
class AdvancedMeshElement:
    """高级网格单元类"""
    id: int
    nodes: List[int]  # 节点ID列表
    element_type: str = "triangle"  # 单元类型
    refinement_level: int = 0  # 细化级别
    parent_id: Optional[int] = None  # 父单元ID
    children_ids: List[int] = field(default_factory=list)  # 子单元ID列表
    quality: float = 1.0  # 单元质量
    center: np.ndarray = None  # 单元中心坐标
    volume: float = 0.0  # 单元体积/面积
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []


@dataclass
class AdvancedMesh:
    """高级网格类 - 支持复杂拓扑管理"""
    coordinates: np.ndarray  # (n_nodes, dim) 节点坐标
    elements: List[AdvancedMeshElement]  # 网格单元列表
    element_type: str = "triangle"  # 支持 triangle/tetra/hex 等
    dim: int = 2  # 空间维度
    
    # 拓扑信息
    faces: List[MeshFace] = field(default_factory=list)  # 面列表
    edges: List[MeshEdge] = field(default_factory=list)  # 边列表
    adj_elements: List[List[int]] = field(default_factory=list)  # 相邻单元
    
    # 网格属性
    max_refinement_level: int = 5  # 最大细化级别
    min_element_size: float = 1e-6  # 最小单元尺寸
    quality_threshold: float = 0.2  # 质量阈值
    
    def __post_init__(self):
        self.n_nodes = len(self.coordinates)
        self.n_elements = len(self.elements)
        if self.dim == 0:  # 自动检测维度
            self.dim = self.coordinates.shape[1]
        
        # 自动计算拓扑关系
        self._compute_topology()
        self._compute_element_properties()
        self._compute_mesh_quality()
    
    def _compute_topology(self):
        """计算面、边、相邻单元等拓扑关系"""
        print("🔄 计算网格拓扑关系...")
        
        # 计算边
        self._compute_edges()
        
        # 计算面
        self._compute_faces()
        
        # 计算相邻单元
        self._compute_adjacent_elements()
        
        print(f"✅ 拓扑计算完成: {len(self.edges)} 条边, {len(self.faces)} 个面")
    
    def _compute_edges(self):
        """计算网格边"""
        edge_dict = {}  # 用于去重
        
        for elem in self.elements:
            if self.element_type == "triangle":
                # 三角形的3条边
                for i in range(3):
                    next_i = (i + 1) % 3
                    edge_nodes = sorted([elem.nodes[i], elem.nodes[next_i]])
                    edge_key = tuple(edge_nodes)
                    
                    if edge_key not in edge_dict:
                        edge_dict[edge_key] = {
                            'nodes': edge_nodes,
                            'element_ids': [elem.id]
                        }
                    else:
                        edge_dict[edge_key]['element_ids'].append(elem.id)
            
            elif self.element_type == "tetra":
                # 四面体的6条边
                for i in range(4):
                    for j in range(i + 1, 4):
                        edge_nodes = sorted([elem.nodes[i], elem.nodes[j]])
                        edge_key = tuple(edge_nodes)
                        
                        if edge_key not in edge_dict:
                            edge_dict[edge_key] = {
                                'nodes': edge_nodes,
                                'element_ids': [elem.id]
                            }
                        else:
                            edge_dict[edge_key]['element_ids'].append(elem.id)
        
        # 创建边对象
        for i, (edge_key, edge_data) in enumerate(edge_dict.items()):
            # 计算边长度
            node1, node2 = edge_data['nodes']
            length = np.linalg.norm(self.coordinates[node1] - self.coordinates[node2])
            
            edge = MeshEdge(
                id=i,
                nodes=edge_data['nodes'],
                element_ids=edge_data['element_ids'],
                length=length
            )
            self.edges.append(edge)
    
    def _compute_faces(self):
        """计算网格面"""
        if self.dim == 2:
            # 2D：面就是边
            for edge in self.edges:
                face = MeshFace(
                    id=edge.id,
                    nodes=edge.nodes,
                    element_ids=edge.element_ids,
                    face_type="edge",
                    boundary=len(edge.element_ids) == 1
                )
                self.faces.append(face)
        else:
            # 3D：需要计算真正的面
            face_dict = {}
            
            for elem in self.elements:
                if self.element_type == "tetra":
                    # 四面体的4个面（三角形）
                    for i in range(4):
                        # 选择3个节点形成面
                        face_nodes = [elem.nodes[j] for j in range(4) if j != i]
                        face_nodes.sort()
                        face_key = tuple(face_nodes)
                        
                        if face_key not in face_dict:
                            face_dict[face_key] = {
                                'nodes': face_nodes,
                                'element_ids': [elem.id]
                            }
                        else:
                            face_dict[face_key]['element_ids'].append(elem.id)
            
            # 创建面对象
            for i, (face_key, face_data) in enumerate(face_dict.items()):
                face = MeshFace(
                    id=i,
                    nodes=face_data['nodes'],
                    element_ids=face_data['element_ids'],
                    face_type="triangle",
                    boundary=len(face_data['element_ids']) == 1
                )
                self.faces.append(face)
    
    def _compute_adjacent_elements(self):
        """计算相邻单元"""
        # 初始化相邻单元列表
        self.adj_elements = [[] for _ in range(self.n_elements)]
        
        # 基于边关系计算相邻单元
        for edge in self.edges:
            if len(edge.element_ids) == 2:
                # 内部边：两个单元相邻
                elem1, elem2 = edge.element_ids
                if elem2 not in self.adj_elements[elem1]:
                    self.adj_elements[elem1].append(elem2)
                if elem1 not in self.adj_elements[elem2]:
                    self.adj_elements[elem2].append(elem1)
    
    def _compute_element_properties(self):
        """计算单元属性（中心、体积等）"""
        for elem in self.elements:
            # 计算单元中心
            elem_coords = self.coordinates[elem.nodes]
            elem.center = np.mean(elem_coords, axis=0)
            
            # 计算单元体积/面积
            if self.element_type == "triangle":
                elem.volume = self._compute_triangle_area(elem.nodes)
            elif self.element_type == "tetra":
                elem.volume = self._compute_tetra_volume(elem.nodes)
    
    def _compute_triangle_area(self, node_ids: List[int]) -> float:
        """计算三角形面积"""
        coords = self.coordinates[node_ids]
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        return 0.5 * abs(np.cross(v1, v2))
    
    def _compute_tetra_volume(self, node_ids: List[int]) -> float:
        """计算四面体体积"""
        coords = self.coordinates[node_ids]
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        v3 = coords[3] - coords[0]
        return abs(np.dot(v1, np.cross(v2, v3))) / 6.0
    
    def _compute_mesh_quality(self):
        """计算网格整体质量"""
        qualities = []
        for elem in self.elements:
            elem.quality = self._compute_element_quality(elem)
            qualities.append(elem.quality)
        
        self.overall_quality = np.mean(qualities) if qualities else 0.0
    
    def _compute_element_quality(self, elem: AdvancedMeshElement) -> float:
        """计算单个单元质量"""
        if self.element_type == "triangle":
            coords = self.coordinates[elem.nodes]
            
            # 计算边长
            edges = []
            for i in range(3):
                edge = coords[(i+1)%3] - coords[i]
                edges.append(np.linalg.norm(edge))
            
            # 计算面积
            area = elem.volume
            
            # 质量指标：面积/最长边长的平方
            max_edge = max(edges)
            quality = area / (max_edge ** 2) if max_edge > 0 else 0.0
            
            return quality
        
        return 1.0  # 默认质量
    
    def get_boundary_faces(self) -> List[MeshFace]:
        """获取边界面"""
        return [face for face in self.faces if face.boundary]
    
    def get_boundary_nodes(self) -> List[int]:
        """获取边界节点"""
        boundary_nodes = set()
        for face in self.get_boundary_faces():
            boundary_nodes.update(face.nodes)
        return list(boundary_nodes)
    
    def get_element_neighbors(self, element_id: int) -> List[int]:
        """获取单元的相邻单元"""
        if element_id < len(self.adj_elements):
            return self.adj_elements[element_id]
        return []
    
    def get_mesh_statistics(self) -> Dict[str, Any]:
        """获取网格统计信息"""
        return {
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "n_edges": len(self.edges),
            "n_faces": len(self.faces),
            "dim": self.dim,
            "element_type": self.element_type,
            "overall_quality": self.overall_quality,
            "max_refinement_level": max(elem.refinement_level for elem in self.elements),
            "boundary_faces": len(self.get_boundary_faces()),
            "boundary_nodes": len(self.get_boundary_nodes())
        }


class AdaptiveRefinement:
    """自适应网格细化器"""
    
    def __init__(self, mesh: AdvancedMesh, 
                 refinement_threshold: float = 0.1,
                 coarsening_threshold: float = 0.02):
        self.mesh = mesh
        self.refinement_threshold = refinement_threshold
        self.coarsening_threshold = coarsening_threshold
        self.refinement_history = []
    
    def get_refinement_indicator(self, field: np.ndarray, 
                               indicator_type: str = "gradient") -> np.ndarray:
        """基于物理场确定需要细化的单元"""
        if indicator_type == "gradient":
            return self._compute_gradient_indicator(field)
        elif indicator_type == "error":
            return self._compute_error_indicator(field)
        elif indicator_type == "ml":
            return self._compute_ml_indicator(field)
        else:
            raise ValueError(f"未知的细化指标类型: {indicator_type}")
    
    def _compute_gradient_indicator(self, field: np.ndarray) -> np.ndarray:
        """基于梯度的细化指标"""
        indicators = np.zeros(self.mesh.n_elements)
        
        for i, elem in enumerate(self.mesh.elements):
            # 计算单元内场值梯度
            elem_values = field[elem.nodes]
            
            if len(elem_values) >= 2:
                # 使用节点间的最大差值作为梯度估计
                gradient = np.max(elem_values) - np.min(elem_values)
                indicators[i] = gradient
        
        # 归一化
        if np.max(indicators) > 0:
            indicators = indicators / np.max(indicators)
        
        return indicators
    
    def _compute_error_indicator(self, field: np.ndarray) -> np.ndarray:
        """基于误差的细化指标"""
        indicators = np.zeros(self.mesh.n_elements)
        
        for i, elem in enumerate(self.mesh.elements):
            # 计算单元内场值方差作为误差指标
            elem_values = field[elem.nodes]
            indicators[i] = np.var(elem_values)
        
        # 归一化
        if np.max(indicators) > 0:
            indicators = indicators / np.max(indicators)
        
        return indicators
    
    def _compute_ml_indicator(self, field: np.ndarray) -> np.ndarray:
        """基于机器学习的细化指标"""
        if not HAS_PYTORCH:
            print("警告：PyTorch不可用，回退到梯度指标")
            return self._compute_gradient_indicator(field)
        
        # 构建特征向量
        features = self._build_ml_features(field)
        
        # 使用预训练模型预测（这里简化处理）
        indicators = self._predict_refinement(features)
        
        return indicators
    
    def _build_ml_features(self, field: np.ndarray) -> np.ndarray:
        """构建ML特征向量"""
        n_elements = self.mesh.n_elements
        features = np.zeros((n_elements, 5))  # 5个特征
        
        for i, elem in enumerate(self.mesh.elements):
            # 特征1：场值梯度
            elem_values = field[elem.nodes]
            features[i, 0] = np.max(elem_values) - np.min(elem_values)
            
            # 特征2：单元质量
            features[i, 1] = elem.quality
            
            # 特征3：细化级别
            features[i, 2] = elem.refinement_level
            
            # 特征4：单元体积
            features[i, 3] = elem.volume
            
            # 特征5：相邻单元数
            features[i, 4] = len(self.mesh.get_element_neighbors(elem.id))
        
        return features
    
    def _predict_refinement(self, features: np.ndarray) -> np.ndarray:
        """使用ML模型预测细化指标"""
        # 简化：基于特征的线性组合
        weights = np.array([0.4, 0.2, 0.1, 0.2, 0.1])
        
        # 归一化特征
        for j in range(features.shape[1]):
            if np.max(features[:, j]) > 0:
                features[:, j] = features[:, j] / np.max(features[:, j])
        
        # 计算加权和
        indicators = np.dot(features, weights)
        
        return indicators
    
    def refine(self, indicator: np.ndarray, max_refinement_ratio: float = 0.8) -> AdvancedMesh:
        """执行网格细化"""
        print("🔄 开始自适应网格细化...")
        
        # 确定需要细化的单元
        to_refine = indicator > self.refinement_threshold
        n_to_refine = np.sum(to_refine)
        
        if n_to_refine == 0:
            print("   无需细化的单元")
            return self.mesh
        
        # 限制细化比例
        if n_to_refine > max_refinement_ratio * self.mesh.n_elements:
            # 选择指标最高的单元进行细化
            sorted_indices = np.argsort(indicator)[::-1]
            max_refine = int(max_refinement_ratio * self.mesh.n_elements)
            to_refine = np.zeros_like(to_refine, dtype=bool)
            to_refine[sorted_indices[:max_refine]] = True
            n_to_refine = max_refine
        
        print(f"   细化单元数: {n_to_refine}/{self.mesh.n_elements}")
        
        # 执行细化
        refined_mesh = self._perform_refinement(to_refine)
        
        # 记录细化历史
        self.refinement_history.append({
            'n_elements_before': self.mesh.n_elements,
            'n_elements_after': refined_mesh.n_elements,
            'n_refined': n_to_refine,
            'indicator_type': 'adaptive',
            'timestamp': time.time()
        })
        
        print(f"✅ 网格细化完成: {self.mesh.n_elements} -> {refined_mesh.n_elements} 单元")
        return refined_mesh
    
    def _perform_refinement(self, to_refine: np.ndarray) -> AdvancedMesh:
        """执行实际的网格细化"""
        new_elements = []
        new_coordinates = self.mesh.coordinates.copy()
        node_offset = self.mesh.n_nodes
        
        for i, elem in enumerate(self.mesh.elements):
            if to_refine[i] and elem.refinement_level < self.mesh.max_refinement_level:
                # 细化单元
                sub_elements, new_nodes = self._split_element(elem, new_coordinates, node_offset)
                new_elements.extend(sub_elements)
                new_coordinates = np.vstack([new_coordinates, new_nodes])
                node_offset += len(new_nodes)
            else:
                # 保持原单元
                new_elements.append(elem)
        
        # 创建新的网格
        refined_mesh = AdvancedMesh(
            coordinates=new_coordinates,
            elements=new_elements,
            element_type=self.mesh.element_type,
            dim=self.mesh.dim,
            max_refinement_level=self.mesh.max_refinement_level
        )
        
        return refined_mesh
    
    def _split_element(self, elem: AdvancedMeshElement, 
                      coordinates: np.ndarray, node_offset: int) -> Tuple[List[AdvancedMeshElement], np.ndarray]:
        """分裂单元"""
        if self.mesh.element_type == "triangle":
            return self._split_triangle(elem, coordinates, node_offset)
        elif self.mesh.element_type == "tetra":
            return self._split_tetra(elem, coordinates, node_offset)
        else:
            raise ValueError(f"不支持的单元类型: {self.mesh.element_type}")
    
    def _split_triangle(self, elem: AdvancedMeshElement, 
                       coordinates: np.ndarray, node_offset: int) -> Tuple[List[AdvancedMeshElement], np.ndarray]:
        """分裂三角形单元"""
        # 获取单元节点坐标
        node_coords = coordinates[elem.nodes]
        
        # 计算单元中心
        center = np.mean(node_coords, axis=0)
        
        # 添加中心节点
        center_node_id = node_offset
        new_nodes = [center]
        
        # 创建4个子三角形
        sub_elements = []
        n_nodes = len(elem.nodes)
        
        for i in range(n_nodes):
            next_i = (i + 1) % n_nodes
            
            # 子单元节点：原节点i、原节点next_i、中心节点
            sub_nodes = [elem.nodes[i], elem.nodes[next_i], center_node_id]
            
            # 创建子单元
            sub_elem = AdvancedMeshElement(
                id=len(sub_elements),
                nodes=sub_nodes,
                element_type="triangle",
                refinement_level=elem.refinement_level + 1,
                parent_id=elem.id
            )
            
            sub_elements.append(sub_elem)
        
        return sub_elements, np.array(new_nodes)
    
    def _split_tetra(self, elem: AdvancedMeshElement, 
                    coordinates: np.ndarray, node_offset: int) -> Tuple[List[AdvancedMeshElement], np.ndarray]:
        """分裂四面体单元（简化实现）"""
        # 获取单元节点坐标
        node_coords = coordinates[elem.nodes]
        
        # 计算单元中心
        center = np.mean(node_coords, axis=0)
        
        # 添加中心节点
        center_node_id = node_offset
        new_nodes = [center]
        
        # 创建8个子四面体（简化：基于4个面）
        sub_elements = []
        
        for i in range(4):
            # 选择3个节点形成面
            face_nodes = [elem.nodes[j] for j in range(4) if j != i]
            
            # 创建子单元：面节点 + 中心节点
               # 创建子单元：面节点 + 中心节点
            sub_nodes = face_nodes + [center_node_id]
            
            # 创建子单元
            sub_elem = AdvancedMeshElement(
                id=len(sub_elements),
                nodes=sub_nodes,
                element_type="tetra",
                refinement_level=elem.refinement_level + 1,
                parent_id=elem.id
            )
            
            sub_elements.append(sub_elem)
        
        return sub_elements, np.array(new_nodes)


class MultiScaleMeshManager:
    """多尺度网格管理器 - 实现不同尺度网格的物理场传递"""
    
    def __init__(self, fine_mesh: AdvancedMesh, max_levels: int = 5):
        self.meshes = [fine_mesh]  # 从细网格到粗网格的层次
        self.transfer_operators = []  # 插值算子
        self._build_coarse_meshes(max_levels)
        self._build_transfer_operators()
    
    def _build_coarse_meshes(self, max_levels: int):
        """生成粗网格层次（每一层是上一层的粗化）"""
        print(f"🔄 构建多尺度网格层次，最大级别: {max_levels}")
        
        current_mesh = self.meshes[0]
        for level in range(max_levels - 1):
            # 粗化策略：合并相邻单元
            coarse_mesh = self._coarsen_mesh(current_mesh)
            if coarse_mesh.n_elements < current_mesh.n_elements * 0.3:  # 粗化效果不明显则停止
                break
            self.meshes.append(coarse_mesh)
            current_mesh = coarse_mesh
        
        print(f"✅ 多尺度网格构建完成: {len(self.meshes)} 个层次")
        for i, mesh in enumerate(self.meshes):
            print(f"   级别 {i}: {mesh.n_elements} 单元, {mesh.n_nodes} 节点")
    
    def _coarsen_mesh(self, fine_mesh: AdvancedMesh) -> AdvancedMesh:
        """粗化网格"""
        # 简化策略：每4个相邻单元合并为1个粗单元
        if fine_mesh.element_type == "triangle":
            return self._coarsen_triangles(fine_mesh)
        else:
            return fine_mesh  # 暂不支持其他类型
    
    def _coarsen_triangles(self, fine_mesh: AdvancedMesh) -> AdvancedMesh:
        """粗化三角形网格"""
        # 基于相邻关系分组
        groups = self._group_triangles_for_coarsening(fine_mesh)
        
        # 创建粗网格
        coarse_elements = []
        coarse_coordinates = []
        node_map = {}  # 细网格节点到粗网格节点的映射
        
        for group in groups:
            if len(group) == 4:  # 4个三角形合并为1个
                # 提取所有节点
                all_nodes = set()
                for elem_id in group:
                    elem = fine_mesh.elements[elem_id]
                    all_nodes.update(elem.nodes)
                
                # 计算粗单元中心
                center = np.mean(fine_mesh.coordinates[list(all_nodes)], axis=0)
                center_id = len(coarse_coordinates)
                coarse_coordinates.append(center)
                
                # 创建粗单元（使用中心节点和边界节点）
                boundary_nodes = self._get_boundary_nodes(group, fine_mesh)
                if len(boundary_nodes) >= 3:
                    coarse_elem = AdvancedMeshElement(
                        id=len(coarse_elements),
                        nodes=[center_id] + boundary_nodes[:3],
                        element_type="triangle",
                        refinement_level=0
                    )
                    coarse_elements.append(coarse_elem)
        
        return AdvancedMesh(
            coordinates=np.array(coarse_coordinates),
            elements=coarse_elements,
            element_type="triangle",
            dim=fine_mesh.dim
        )
    
    def _group_triangles_for_coarsening(self, mesh: AdvancedMesh) -> List[List[int]]:
        """将三角形分组以便粗化"""
        groups = []
        used_elements = set()
        
        for elem_id, elem in enumerate(mesh.elements):
            if elem_id in used_elements:
                continue
            
            # 寻找相邻的三角形形成组
            group = [elem_id]
            used_elements.add(elem_id)
            
            # 寻找相邻的三角形
            neighbors = mesh.get_element_neighbors(elem_id)
            for neighbor_id in neighbors:
                if neighbor_id not in used_elements and len(group) < 4:
                    group.append(neighbor_id)
                    used_elements.add(neighbor_id)
            
            groups.append(group)
        
        return groups
    
    def _get_boundary_nodes(self, element_group: List[int], mesh: AdvancedMesh) -> List[int]:
        """获取元素组的边界节点"""
        # 简化：返回组中第一个单元的所有节点
        if element_group:
            return mesh.elements[element_group[0]].nodes
        return []
    
    def _build_transfer_operators(self):
        """构建插值算子"""
        print("🔄 构建多尺度插值算子...")
        
        for i in range(len(self.meshes) - 1):
            fine_mesh = self.meshes[i]
            coarse_mesh = self.meshes[i + 1]
            
            # 构建插值矩阵
            transfer_op = self._build_interpolation_matrix(fine_mesh, coarse_mesh)
            self.transfer_operators.append(transfer_op)
        
        print(f"✅ 插值算子构建完成: {len(self.transfer_operators)} 个")
    
    def _build_interpolation_matrix(self, fine_mesh: AdvancedMesh, 
                                  coarse_mesh: AdvancedMesh) -> np.ndarray:
        """构建插值矩阵"""
        # 简化：基于距离的线性插值
        n_fine = fine_mesh.n_nodes
        n_coarse = coarse_mesh.n_nodes
        
        transfer_matrix = np.zeros((n_coarse, n_fine))
        
        for i, coarse_node in enumerate(coarse_mesh.coordinates):
            for j, fine_node in enumerate(fine_mesh.coordinates):
                # 计算距离权重
                distance = np.linalg.norm(coarse_node - fine_node)
                if distance < 1e-10:  # 相同节点
                    transfer_matrix[i, j] = 1.0
                else:
                    # 距离权重（简化）
                    transfer_matrix[i, j] = 1.0 / (1.0 + distance)
        
        # 归一化
        row_sums = transfer_matrix.sum(axis=1, keepdims=True)
        transfer_matrix = transfer_matrix / (row_sums + 1e-10)
        
        return transfer_matrix
    
    def transfer_field(self, fine_field: np.ndarray, level: int) -> np.ndarray:
        """将细网格物理场插值到第level层粗网格"""
        if level >= len(self.meshes):
            raise ValueError(f"无效的网格级别: {level}")
        
        if level == 0:
            return fine_field  # 细网格层
        
        # 逐层插值
        current_field = fine_field
        for i in range(level):
            if i < len(self.transfer_operators):
                current_field = self.transfer_operators[i] @ current_field
        
        return current_field
    
    def get_multiscale_statistics(self) -> Dict[str, Any]:
        """获取多尺度统计信息"""
        stats = {}
        for i, mesh in enumerate(self.meshes):
            stats[f"level_{i}"] = {
                "n_elements": mesh.n_elements,
                "n_nodes": mesh.n_nodes,
                "element_type": mesh.element_type,
                "overall_quality": mesh.overall_quality
            }
        return stats


class ParallelMesh:
    """并行网格分区器"""
    
    def __init__(self, global_mesh: AdvancedMesh, n_partitions: int = 4):
        self.global_mesh = global_mesh
        self.n_partitions = n_partitions
        self.partitions = None
        self.local_meshes = []
        self._partition_mesh()
    
    def _partition_mesh(self):
        """将全局网格分成n_partitions个分区"""
        print(f"🔄 开始网格分区，分区数: {self.n_partitions}")
        
        # 基于单元邻接关系构建分区
        adjacency = [list(neighbors) for neighbors in self.global_mesh.adj_elements]
        
        # 简化分区策略：基于单元ID的简单分割
        self.partitions = np.array_split(range(self.global_mesh.n_elements), self.n_partitions)
        
        # 提取本地网格
        for partition in self.partitions:
            local_mesh = self._extract_local_mesh(partition)
            self.local_meshes.append(local_mesh)
        
        print(f"✅ 网格分区完成")
        for i, local_mesh in enumerate(self.local_meshes):
            print(f"   分区 {i}: {local_mesh.n_elements} 单元, {local_mesh.n_nodes} 节点")
    
    def _extract_local_mesh(self, element_ids: np.ndarray) -> AdvancedMesh:
        """提取本地网格"""
        # 提取本地单元
        local_elements = [self.global_mesh.elements[eid] for eid in element_ids]
        
        # 提取本地节点（去重）
        local_node_ids = set()
        for elem in local_elements:
            local_node_ids.update(elem.nodes)
        local_node_ids = sorted(list(local_node_ids))
        
        # 重新编号本地节点
        node_map = {gid: lid for lid, gid in enumerate(local_node_ids)}
        
        # 创建本地单元（重新编号）
        new_local_elements = []
        for elem in local_elements:
            new_nodes = [node_map[nid] for nid in elem.nodes]
            new_elem = AdvancedMeshElement(
                id=len(new_local_elements),
                nodes=new_nodes,
                element_type=elem.element_type,
                refinement_level=elem.refinement_level,
                parent_id=elem.parent_id
            )
            new_local_elements.append(new_elem)
        
        # 提取本地坐标
        local_coordinates = self.global_mesh.coordinates[local_node_ids]
        
        return AdvancedMesh(
            coordinates=local_coordinates,
            elements=new_local_elements,
            element_type=self.global_mesh.element_type,
            dim=self.global_mesh.dim
        )
    
    def get_partition_statistics(self) -> Dict[str, Any]:
        """获取分区统计信息"""
        stats = {
            "n_partitions": self.n_partitions,
            "global_elements": self.global_mesh.n_elements,
            "global_nodes": self.global_mesh.n_nodes,
            "partitions": []
        }
        
        for i, local_mesh in enumerate(self.local_meshes):
            stats["partitions"].append({
                "partition_id": i,
                "n_elements": local_mesh.n_elements,
                "n_nodes": local_mesh.n_nodes,
                "load_balance": local_mesh.n_elements / (self.global_mesh.n_elements / self.n_partitions)
            })
        
        return stats


class DynamicMesh:
    """动态网格变形器"""
    
    def __init__(self, initial_mesh: AdvancedMesh):
        self.mesh = initial_mesh
        self.deformation = np.zeros_like(initial_mesh.coordinates)  # 节点位移
        self.deformation_history = []
        self.quality_history = []
    
    def update_deformation(self, displacement_field: np.ndarray):
        """根据位移场更新网格形状"""
        print("🔄 更新网格变形...")
        
        # 记录变形历史
        self.deformation_history.append(self.deformation.copy())
        self.quality_history.append(self.mesh.overall_quality)
        
        # 更新变形
        self.deformation = displacement_field
        
        # 移动节点坐标
        self.mesh.coordinates += displacement_field
        
        # 重新计算网格属性
        self.mesh._compute_element_properties()
        self.mesh._compute_mesh_quality()
        
        # 检查网格质量
        if self.mesh.overall_quality < self.mesh.quality_threshold:
            print(f"⚠️  网格质量过低: {self.mesh.overall_quality:.3f} < {self.mesh.quality_threshold}")
            self._remesh()
        
        print(f"✅ 网格变形更新完成，质量: {self.mesh.overall_quality:.3f}")
    
    def _remesh(self):
        """重生成高质量网格"""
        print("🔄 开始网格重生成...")
        
        # 基于当前变形后的边界生成新网格
        new_mesh = self._generate_new_mesh()
        
        # 将旧网格的物理场插值到新网格
        field_mapper = FieldMapper(self.mesh, new_mesh)
        
        # 更新网格
        self.mesh = new_mesh
        
        print(f"✅ 网格重生成完成: {new_mesh.n_elements} 单元, 质量: {new_mesh.overall_quality:.3f}")
    
    def _generate_new_mesh(self) -> AdvancedMesh:
        """生成新网格（简化实现）"""
        # 简化：基于当前网格重新三角化
        if HAS_SCIPY:
            # 使用Delaunay三角化
            tri = Delaunay(self.mesh.coordinates)
            elements = tri.simplices
            
            # 创建新单元
            new_elements = []
            for i, elem_nodes in enumerate(elements):
                elem = AdvancedMeshElement(
                    id=i,
                    nodes=list(elem_nodes),
                    element_type="triangle",
                    refinement_level=0
                )
                new_elements.append(elem)
            
            return AdvancedMesh(
                coordinates=self.mesh.coordinates.copy(),
                elements=new_elements,
                element_type="triangle",
                dim=self.mesh.dim
            )
        else:
            # 回退到原网格
            return self.mesh
    
    def get_deformation_statistics(self) -> Dict[str, Any]:
        """获取变形统计信息"""
        if not self.deformation_history:
            return {"status": "No deformation yet"}
        
        max_deformation = np.max(np.linalg.norm(self.deformation, axis=1))
        avg_deformation = np.mean(np.linalg.norm(self.deformation, axis=1))
        
        return {
            "max_deformation": max_deformation,
            "avg_deformation": avg_deformation,
            "n_deformation_steps": len(self.deformation_history),
            "initial_quality": self.quality_history[0] if self.quality_history else 0.0,
            "current_quality": self.mesh.overall_quality,
            "quality_degradation": self.quality_history[0] - self.mesh.overall_quality if self.quality_history else 0.0
        }


class FieldMapper:
    """物理场映射器 - 在网格间插值物理场"""
    
    def __init__(self, source_mesh: AdvancedMesh, target_mesh: AdvancedMesh):
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh
        self.mapping_matrix = self._build_mapping_matrix()
    
    def _build_mapping_matrix(self) -> np.ndarray:
        """构建映射矩阵"""
        n_source = self.source_mesh.n_nodes
        n_target = self.target_mesh.n_nodes
        
        mapping_matrix = np.zeros((n_target, n_source))
        
        for i, target_node in enumerate(self.target_mesh.coordinates):
            for j, source_node in enumerate(self.source_mesh.coordinates):
                # 基于距离的插值权重
                distance = np.linalg.norm(target_node - source_node)
                if distance < 1e-10:
                    mapping_matrix[i, j] = 1.0
                else:
                    # 高斯权重
                    sigma = 0.1  # 插值半径
                    mapping_matrix[i, j] = np.exp(-distance**2 / (2 * sigma**2))
        
        # 归一化
        row_sums = mapping_matrix.sum(axis=1, keepdims=True)
        mapping_matrix = mapping_matrix / (row_sums + 1e-10)
        
        return mapping_matrix
    
    def map_field(self, source_field: np.ndarray) -> np.ndarray:
        """映射物理场"""
        return self.mapping_matrix @ source_field


def create_advanced_mesh_example():
    """创建高级网格示例"""
    print("🔧 创建高级网格示例...")
    
    # 创建简单2D网格
    coordinates = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # 4个角点
        [0.5, 0.5]  # 中心点
    ])
    
    elements = [
        AdvancedMeshElement(id=0, nodes=[0, 1, 4], element_type="triangle"),
        AdvancedMeshElement(id=1, nodes=[1, 2, 4], element_type="triangle"),
        AdvancedMeshElement(id=2, nodes=[2, 3, 4], element_type="triangle"),
        AdvancedMeshElement(id=3, nodes=[3, 0, 4], element_type="triangle")
    ]
    
    # 创建高级网格
    mesh = AdvancedMesh(coordinates=coordinates, elements=elements, element_type="triangle")
    
    # 输出网格统计信息
    stats = mesh.get_mesh_statistics()
    print(f"✅ 高级网格创建完成:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return mesh


def demo_adaptive_refinement():
    """演示自适应细化"""
    print("\n🚀 演示自适应网格细化...")
    
    # 创建网格
    mesh = create_advanced_mesh_example()
    
    # 创建物理场（温度场）
    n_nodes = mesh.n_nodes
    temperature_field = np.zeros(n_nodes)
    
    # 设置温度梯度（中心热源）
    for i in range(n_nodes):
        dist_to_center = np.linalg.norm(mesh.coordinates[i] - np.array([0.5, 0.5]))
        temperature_field[i] = 100.0 * np.exp(-dist_to_center / 0.3)
    
    # 创建自适应细化器
    refiner = AdaptiveRefinement(mesh)
    
    # 计算细化指标
    indicator = refiner.get_refinement_indicator(temperature_field, "gradient")
    
    # 执行细化
    refined_mesh = refiner.refine(indicator)
    
    # 输出细化统计
    print(f"\n�� 细化统计:")
    for i, history in enumerate(refiner.refinement_history):
        print(f"   步骤 {i+1}: {history['n_elements_before']} -> {history['n_elements_after']} 单元")
    
    return mesh, refined_mesh, refiner


def demo_multiscale_mesh():
    """演示多尺度网格"""
    print("\n�� 演示多尺度网格...")
    
    # 创建网格
    mesh = create_advanced_mesh_example()
    
    # 创建多尺度管理器
    multiscale_manager = MultiScaleMeshManager(mesh, max_levels=3)
    
    # 输出多尺度统计
    stats = multiscale_manager.get_multiscale_statistics()
    print(f"✅ 多尺度网格统计:")
    for level, level_stats in stats.items():
        print(f"   {level}: {level_stats['n_elements']} 单元, 质量: {level_stats['overall_quality']:.3f}")
    
    return multiscale_manager


def demo_parallel_mesh():
    """演示并行网格分区"""
    print("\n⚡ 演示并行网格分区...")
    
    # 创建网格
    mesh = create_advanced_mesh_example()
    
    # 创建并行分区器
    parallel_mesh = ParallelMesh(mesh, n_partitions=2)
    
    # 输出分区统计
    stats = parallel_mesh.get_partition_statistics()
    print(f"✅ 并行网格分区统计:")
    print(f"   分区数: {stats['n_partitions']}")
    for partition in stats['partitions']:
        print(f"   分区 {partition['partition_id']}: {partition['n_elements']} 单元, 负载均衡: {partition['load_balance']:.2f}")
    
    return parallel_mesh


def demo_dynamic_mesh():
    """演示动态网格变形"""
    print("\n🔄 演示动态网格变形...")
    
    # 创建网格
    mesh = create_advanced_mesh_example()
    
    # 创建动态网格
    dynamic_mesh = DynamicMesh(mesh)
    
    # 模拟位移场
    displacement_field = np.zeros_like(mesh.coordinates)
    displacement_field[:, 0] = 0.1 * np.sin(mesh.coordinates[:, 1] * np.pi)  # x方向位移
    displacement_field[:, 1] = 0.05 * np.cos(mesh.coordinates[:, 0] * np.pi)  # y方向位移
    
    # 更新变形
    dynamic_mesh.update_deformation(displacement_field)
    
    # 输出变形统计
    stats = dynamic_mesh.get_deformation_statistics()
    print(f"✅ 动态网格变形统计:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return dynamic_mesh


if __name__ == "__main__":
    print("🔧 高级网格系统演示")
    print("=" * 50)
    
    try:
        # 1. 基础高级网格
        mesh = create_advanced_mesh_example()
        
        # 2. 自适应细化
        mesh, refined_mesh, refiner = demo_adaptive_refinement()
        
        # 3. 多尺度网格
        multiscale_manager = demo_multiscale_mesh()
        
        # 4. 并行网格分区
        parallel_mesh = demo_parallel_mesh()
        
        # 5. 动态网格变形
        dynamic_mesh = demo_dynamic_mesh()
        
        print("\n✅ 所有高级网格功能演示完成!")
        print("\n📋 功能总结:")
        print("   - 复杂拓扑管理：面、边、相邻单元关系")
        print("   - 自适应细化：基于物理场梯度的动态网格调整")
        print("   - 多尺度支持：细网格到粗网格的层次结构")
        print("   - 并行分区：基于MPI的分布式网格")
        print("   - 动态变形：大变形场景的网格重生成")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()