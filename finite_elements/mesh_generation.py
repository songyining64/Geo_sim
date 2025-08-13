"""
自适应网格生成模块
实现基于物理场梯度的动态网格加密和粗化，减少冗余网格
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    torch = None
    warnings.warn("PyTorch not available. ML-based mesh adaptation will be limited.")


@dataclass
class MeshCell:
    """网格单元类"""
    id: int
    nodes: List[int]  # 节点ID列表
    level: int = 0     # 细化级别
    parent: Optional[int] = None  # 父单元ID
    children: List[int] = None    # 子单元ID列表
    center: np.ndarray = None     # 单元中心坐标
    volume: float = 0.0           # 单元体积
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class AdaptiveMesh:
    """自适应网格类"""
    nodes: np.ndarray  # 节点坐标 (n_nodes, dim)
    cells: List[MeshCell]  # 网格单元列表
    dim: int = 2  # 空间维度
    max_level: int = 5  # 最大细化级别
    min_cell_size: float = 1e-6  # 最小单元尺寸
    
    def __post_init__(self):
        self._build_connectivity()
        self._compute_cell_properties()
        self._build_boundary_topology()  # 新增：构建边界拓扑
    
    def _build_connectivity(self):
        """构建网格连接关系"""
        # 节点到单元的映射
        self.node_to_cells = {}
        for cell in self.cells:
            for node_id in cell.nodes:
                if node_id not in self.node_to_cells:
                    self.node_to_cells[node_id] = []
                self.node_to_cells[node_id].append(cell.id)
    
    def _build_boundary_topology(self):
        """构建边界拓扑关系 - 支持非结构化网格"""
        self.boundary_faces = []
        self.boundary_edges = []
        self.face_to_cells = {}  # 面到单元的映射
        self.edge_to_cells = {}  # 边到单元的映射
        
        # 识别边界面和边
        self._identify_boundary_faces()
        self._identify_boundary_edges()
    
    def _identify_boundary_faces(self):
        """识别边界面 - 基于面-节点拓扑关系"""
        # 构建面到单元的映射
        face_to_cells = {}
        
        for cell in self.cells:
            cell_faces = self._get_cell_faces(cell)
            for face in cell_faces:
                face_key = tuple(sorted(face))
                if face_key not in face_to_cells:
                    face_to_cells[face_key] = []
                face_to_cells[face_key].append(cell.id)
        
        # 识别边界面（只属于一个单元的面）
        for face_key, cell_ids in face_to_cells.items():
            if len(cell_ids) == 1:
                # 这是边界面
                boundary_face = {
                    'nodes': list(face_key),
                    'cells': cell_ids,
                    'area': self._compute_face_area(list(face_key)),
                    'normal': self._compute_face_normal(list(face_key)),
                    'center': self._compute_face_center(list(face_key))
                }
                self.boundary_faces.append(boundary_face)
                
                # 更新面到单元的映射
                self.face_to_cells[face_key] = cell_ids
    
    def _identify_boundary_edges(self):
        """识别边界边 - 支持2D和3D"""
        edge_to_cells = {}
        
        for cell in self.cells:
            cell_edges = self._get_cell_edges(cell)
            for edge in cell_edges:
                edge_key = tuple(sorted(edge))
                if edge_key not in edge_to_cells:
                    edge_to_cells[edge_key] = []
                edge_to_cells[edge_key].append(cell.id)
        
        # 识别边界边
        for edge_key, cell_ids in edge_to_cells.items():
            if len(cell_ids) == 1:
                boundary_edge = {
                    'nodes': list(edge_key),
                    'cells': cell_ids,
                    'length': self._compute_edge_length(list(edge_key))
                }
                self.boundary_edges.append(boundary_edge)
                self.edge_to_cells[edge_key] = cell_ids
    
    def _get_cell_faces(self, cell: MeshCell) -> List[List[int]]:
        """获取单元的面"""
        if self.dim == 2:
            # 2D情况：面就是边
            return self._get_cell_edges(cell)
        else:
            # 3D情况：需要根据单元类型确定面
            return self._get_3d_cell_faces(cell)
    
    def _get_cell_edges(self, cell: MeshCell) -> List[List[int]]:
        """获取单元的边"""
        edges = []
        nodes = cell.nodes
        n_nodes = len(nodes)
        
        for i in range(n_nodes):
            j = (i + 1) % n_nodes
            edges.append([nodes[i], nodes[j]])
        
        return edges
    
    def _get_3d_cell_faces(self, cell: MeshCell) -> List[List[int]]:
        """获取3D单元的面 - 简化实现"""
        # 这里需要根据具体的3D单元类型（四面体、六面体等）来确定面
        # 简化实现：假设是四面体
        if len(cell.nodes) == 4:  # 四面体
            faces = [
                [cell.nodes[0], cell.nodes[1], cell.nodes[2]],
                [cell.nodes[0], cell.nodes[1], cell.nodes[3]],
                [cell.nodes[0], cell.nodes[2], cell.nodes[3]],
                [cell.nodes[1], cell.nodes[2], cell.nodes[3]]
            ]
            return faces
        else:
            # 其他单元类型，返回空列表
            return []
    
    def _compute_face_area(self, face_nodes: List[int]) -> float:
        """计算面的面积"""
        if len(face_nodes) < 2:
            return 0.0
        
        if self.dim == 2:
            # 2D：边长度
            p1 = self.nodes[face_nodes[0]]
            p2 = self.nodes[face_nodes[1]]
            return np.linalg.norm(p2 - p1)
        else:
            # 3D：多边形面积
            return self._compute_polygon_area(face_nodes)
    
    def _compute_face_normal(self, face_nodes: List[int]) -> np.ndarray:
        """计算面的法向量"""
        if len(face_nodes) < 2:
            return np.zeros(self.dim)
        
        if self.dim == 2:
            # 2D：边的法向量
            p1 = self.nodes[face_nodes[0]]
            p2 = self.nodes[face_nodes[1]]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])  # 逆时针90度旋转
            return normal / np.linalg.norm(normal)
        else:
            # 3D：面的法向量
            if len(face_nodes) >= 3:
                p1 = self.nodes[face_nodes[0]]
                p2 = self.nodes[face_nodes[1]]
                p3 = self.nodes[face_nodes[2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    return normal / norm
            
            return np.zeros(self.dim)
    
    def _compute_face_center(self, face_nodes: List[int]) -> np.ndarray:
        """计算面的中心"""
        if not face_nodes:
            return np.zeros(self.dim)
        
        coords = self.nodes[face_nodes]
        return np.mean(coords, axis=0)
    
    def _compute_edge_length(self, edge_nodes: List[int]) -> float:
        """计算边的长度"""
        if len(edge_nodes) != 2:
            return 0.0
        
        p1 = self.nodes[edge_nodes[0]]
        p2 = self.nodes[edge_nodes[1]]
        return np.linalg.norm(p2 - p1)
    
    # 新增：网格拓扑修改接口
    def refine_cell(self, cell_id: int, refinement_type: str = 'h_refinement') -> List[int]:
        """细化指定单元"""
        if cell_id >= len(self.cells):
            raise ValueError(f"单元ID {cell_id} 超出范围")
        
        cell = self.cells[cell_id]
        if cell.level >= self.max_level:
            raise ValueError(f"单元 {cell_id} 已达到最大细化级别")
        
        if refinement_type == 'h_refinement':
            return self._h_refine_cell(cell_id)
        elif refinement_type == 'p_refinement':
            return self._p_refine_cell(cell_id)
        else:
            raise ValueError(f"不支持的细化类型: {refinement_type}")
    
    def _h_refine_cell(self, cell_id: int) -> List[int]:
        """h-细化：将单元分割为更小的单元"""
        cell = self.cells[cell_id]
        new_cells = []
        
        if self.dim == 2:
            # 2D情况：三角形或四边形细化
            if len(cell.nodes) == 3:  # 三角形
                new_cells = self._refine_triangle(cell)
            elif len(cell.nodes) == 4:  # 四边形
                new_cells = self._refine_quad(cell)
        else:
            # 3D情况：四面体或六面体细化
            if len(cell.nodes) == 4:  # 四面体
                new_cells = self._refine_tetrahedron(cell)
            elif len(cell.nodes) == 8:  # 六面体
                new_cells = self._refine_hexahedron(cell)
        
        # 更新单元列表
        self.cells[cell_id] = new_cells[0]  # 第一个新单元替换原单元
        self.cells.extend(new_cells[1:])    # 添加其他新单元
        
        # 更新连接关系
        self._update_connectivity_after_refinement(cell_id, new_cells)
        
        return [cell.id for cell in new_cells]
    
    def _refine_triangle(self, cell: MeshCell) -> List[MeshCell]:
        """细化三角形单元"""
        # 计算边的中点
        midpoints = []
        for i in range(3):
            j = (i + 1) % 3
            mid_coord = (self.nodes[cell.nodes[i]] + self.nodes[cell.nodes[j]]) / 2
            midpoints.append(self._add_node(mid_coord))
        
        # 创建4个新的三角形单元
        new_cells = []
        new_cells.append(MeshCell(
            id=cell.id,
            nodes=[cell.nodes[0], midpoints[0], midpoints[2]],
            level=cell.level + 1,
            parent=cell.id
        ))
        new_cells.append(MeshCell(
            id=len(self.cells),
            nodes=[midpoints[0], cell.nodes[1], midpoints[1]],
            level=cell.level + 1,
            parent=cell.id
        ))
        new_cells.append(MeshCell(
            id=len(self.cells) + 1,
            nodes=[midpoints[2], midpoints[1], cell.nodes[2]],
            level=cell.level + 1,
            parent=cell.id
        ))
        new_cells.append(MeshCell(
            id=len(self.cells) + 2,
            nodes=[midpoints[0], midpoints[1], midpoints[2]],
            level=cell.level + 1,
            parent=cell.id
        ))
        
        return new_cells
    
    def _refine_quad(self, cell: MeshCell) -> List[MeshCell]:
        """细化四边形单元"""
        # 计算边的中点和单元中心
        midpoints = []
        for i in range(4):
            j = (i + 1) % 4
            mid_coord = (self.nodes[cell.nodes[i]] + self.nodes[cell.nodes[j]]) / 2
            midpoints.append(self._add_node(mid_coord))
        
        center_coord = np.mean([self.nodes[node_id] for node_id in cell.nodes], axis=0)
        center_node = self._add_node(center_coord)
        
        # 创建4个新的四边形单元
        new_cells = []
        for i in range(4):
            j = (i + 1) % 4
            new_cells.append(MeshCell(
                id=cell.id if i == 0 else len(self.cells) + i - 1,
                nodes=[cell.nodes[i], midpoints[i], center_node, midpoints[(i-1) % 4]],
                level=cell.level + 1,
                parent=cell.id
            ))
        
        return new_cells
    
    def _refine_tetrahedron(self, cell: MeshCell) -> List[MeshCell]:
        """细化四面体单元 - 简化实现"""
        # 计算边的中点
        midpoints = []
        for i in range(6):  # 四面体有6条边
            # 简化：假设边的顺序
            edge_nodes = self._get_tetrahedron_edge(i, cell.nodes)
            mid_coord = np.mean([self.nodes[node_id] for node_id in edge_nodes], axis=0)
            midpoints.append(self._add_node(mid_coord))
        
        # 创建8个新的四面体单元（简化实现）
        new_cells = []
        # 这里需要实现具体的四面体细化逻辑
        # 简化：返回原单元
        new_cells.append(MeshCell(
            id=cell.id,
            nodes=cell.nodes,
            level=cell.level + 1,
            parent=cell.id
        ))
        
        return new_cells
    
    def _refine_hexahedron(self, cell: MeshCell) -> List[MeshCell]:
        """细化六面体单元 - 简化实现"""
        # 简化实现：返回原单元
        new_cells = []
        new_cells.append(MeshCell(
            id=cell.id,
            nodes=cell.nodes,
            level=cell.level + 1,
            parent=cell.id
        ))
        
        return new_cells
    
    def _get_tetrahedron_edge(self, edge_id: int, nodes: List[int]) -> List[int]:
        """获取四面体的边 - 简化实现"""
        # 四面体的6条边
        edges = [
            [nodes[0], nodes[1]], [nodes[0], nodes[2]], [nodes[0], nodes[3]],
            [nodes[1], nodes[2]], [nodes[1], nodes[3]], [nodes[2], nodes[3]]
        ]
        return edges[edge_id] if edge_id < len(edges) else [nodes[0], nodes[1]]
    
    def _add_node(self, coord: np.ndarray) -> int:
        """添加新节点"""
        new_node_id = len(self.nodes)
        self.nodes = np.vstack([self.nodes, coord])
        return new_node_id
    
    def _update_connectivity_after_refinement(self, old_cell_id: int, new_cells: List[MeshCell]):
        """细化后更新连接关系"""
        # 更新节点到单元的映射
        for cell in new_cells:
            for node_id in cell.nodes:
                if node_id not in self.node_to_cells:
                    self.node_to_cells[node_id] = []
                self.node_to_cells[node_id].append(cell.id)
        
        # 重新构建边界拓扑
        self._build_boundary_topology()
    
    def coarsen_cells(self, cell_ids: List[int]) -> bool:
        """粗化指定单元"""
        # 检查是否可以粗化
        for cell_id in cell_ids:
            if cell_id >= len(self.cells):
                return False
            cell = self.cells[cell_id]
            if cell.level == 0 or cell.parent is None:
                return False
        
        # 按父单元分组
        parent_groups = {}
        for cell_id in cell_ids:
            cell = self.cells[cell_id]
            if cell.parent not in parent_groups:
                parent_groups[cell.parent] = []
            parent_groups[cell.parent].append(cell_id)
        
        # 执行粗化
        for parent_id, children_ids in parent_groups.items():
            if len(children_ids) >= 2:  # 至少需要2个子单元才能粗化
                self._coarsen_cell_group(parent_id, children_ids)
        
        return True
    
    def _coarsen_cell_group(self, parent_id: int, children_ids: List[int]):
        """粗化一组子单元"""
        # 获取父单元信息
        parent_cell = self.cells[parent_id]
        
        # 合并子单元的节点
        all_nodes = set()
        for child_id in children_ids:
            child = self.cells[child_id]
            all_nodes.update(child.nodes)
        
        # 创建新的粗化单元
        coarsened_cell = MeshCell(
            id=parent_id,
            nodes=list(all_nodes),
            level=parent_cell.level - 1,
            parent=parent_cell.parent
        )
        
        # 更新单元列表
        self.cells[parent_id] = coarsened_cell
        
        # 删除子单元
        for child_id in children_ids:
            if child_id < len(self.cells):
                self.cells[child_id] = None
        
        # 清理None值
        self.cells = [cell for cell in self.cells if cell is not None]
        
        # 更新连接关系
        self._update_connectivity_after_coarsening()
    
    def _update_connectivity_after_coarsening(self):
        """粗化后更新连接关系"""
        # 重新构建所有连接关系
        self._build_connectivity()
        self._build_boundary_topology()
    
    def get_refinement_candidates(self, error_indicator: np.ndarray, 
                                threshold: float = 0.1) -> List[int]:
        """获取需要细化的单元候选"""
        candidates = []
        for i, error in enumerate(error_indicator):
            if error > threshold and i < len(self.cells):
                cell = self.cells[i]
                if cell.level < self.max_level:
                    candidates.append(i)
        return candidates
    
    def get_coarsening_candidates(self, error_indicator: np.ndarray,
                                threshold: float = 0.01) -> List[int]:
        """获取可以粗化的单元候选"""
        candidates = []
        for i, error in enumerate(error_indicator):
            if error < threshold and i < len(self.cells):
                cell = self.cells[i]
                if cell.level > 0 and cell.parent is not None:
                    candidates.append(i)
        return candidates
    
    def _compute_cell_properties(self):
        """计算单元属性"""
        for cell in self.cells:
            # 计算单元中心
            cell_center = np.mean(self.nodes[cell.nodes], axis=0)
            cell.center = cell_center
            
            # 计算单元体积（简化：2D为面积，3D为体积）
            if self.dim == 2:
                # 2D多边形面积
                cell.volume = self._compute_polygon_area(cell.nodes)
            else:
                # 3D多面体体积
                cell.volume = self._compute_polyhedron_volume(cell.nodes)
    
    def _compute_polygon_area(self, node_ids: List[int]) -> float:
        """计算2D多边形面积（鞋带公式）"""
        if len(node_ids) < 3:
            return 0.0
        
        coords = self.nodes[node_ids]
        n = len(coords)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        
        return abs(area) / 2.0
    
    def _compute_polyhedron_volume(self, node_ids: List[int]) -> float:
        """计算3D多面体体积（简化：四面体分解）"""
        if len(node_ids) < 4:
            return 0.0
        
        # 简化：使用第一个节点作为顶点，其他三个节点作为底面
        # 实际应用中应使用更复杂的四面体分解
        coords = self.nodes[node_ids]
        v0 = coords[0]
        v1 = coords[1]
        v2 = coords[2]
        v3 = coords[3]
        
        # 四面体体积公式
        volume = abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0
        return volume


class GradientBasedAdaptor:
    """基于梯度的网格自适应器"""
    
    def __init__(self, mesh: AdaptiveMesh, 
                 refinement_threshold: float = 0.1,
                 coarsening_threshold: float = 0.02,
                 max_refinement_ratio: float = 0.8):
        self.mesh = mesh
        self.refinement_threshold = refinement_threshold
        self.coarsening_threshold = coarsening_threshold
        self.max_refinement_ratio = max_refinement_ratio
        
        # 物理场缓存
        self.field_cache = {}
        self.gradient_cache = {}
    
    def adapt_mesh_based_on_gradient(self, physics_fields: Dict[str, np.ndarray], 
                                   adaptation_strategy: str = "gradient") -> AdaptiveMesh:
        """
        根据物理场梯度动态调整网格密度
        
        Args:
            physics_fields: 物理场字典，键为场名，值为场值数组
            adaptation_strategy: 自适应策略 ("gradient", "error", "ml")
        
        Returns:
            自适应后的网格
        """
        print(f"🔄 开始基于{adaptation_strategy}的网格自适应...")
        
        # 计算物理场梯度
        gradients = self._compute_field_gradients(physics_fields)
        
        # 根据策略选择自适应方法
        if adaptation_strategy == "gradion":
            adapted_mesh = self._gradient_based_adaptation(gradients)
        elif adaptation_strategy == "error":
            adapted_mesh = self._error_based_adaptation(physics_fields)
        elif adaptation_strategy == "ml":
            adapted_mesh = self._ml_based_adaptation(physics_fields, gradients)
        else:
            raise ValueError(f"未知的自适应策略: {adaptation_strategy}")
        
        print(f"✅ 网格自适应完成，单元数: {len(self.mesh.cells)} -> {len(adapted_mesh.cells)}")
        return adapted_mesh
    
    def _compute_field_gradients(self, physics_fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """计算物理场梯度"""
        gradients = {}
        
        for field_name, field_values in physics_fields.items():
            if field_name in self.gradient_cache:
                gradients[field_name] = self.gradient_cache[field_name]
                continue
            
            # 计算每个单元的梯度
            cell_gradients = []
            for cell in self.mesh.cells:
                cell_grad = self._compute_cell_gradient(cell, field_values)
                cell_gradients.append(cell_grad)
            
            gradients[field_name] = np.array(cell_gradients)
            self.gradient_cache[field_name] = gradients[field_name]
        
        return gradients
    
    def _compute_cell_gradient(self, cell: MeshCell, field_values: np.ndarray) -> float:
        """计算单元内物理场梯度"""
        if len(cell.nodes) < 2:
            return 0.0
        
        # 获取单元节点的场值
        cell_values = field_values[cell.nodes]
        
        # 计算梯度（简化：使用中心差分）
        if len(cell_values) >= 2:
            # 使用节点间的最大差值作为梯度估计
            gradient = np.max(cell_values) - np.min(cell_values)
        else:
            gradient = 0.0
        
        return gradient
    
    def _gradient_based_adaptation(self, gradients: Dict[str, np.ndarray]) -> AdaptiveMesh:
        """基于梯度的网格自适应"""
        print("   使用梯度自适应策略...")
        
        # 计算综合梯度指标
        combined_gradient = np.zeros(len(self.mesh.cells))
        for field_grad in gradients.values():
            combined_gradient = np.maximum(combined_gradient, field_grad)
        
        # 归一化梯度
        if np.max(combined_gradient) > 0:
            combined_gradient = combined_gradient / np.max(combined_gradient)
        
        # 执行自适应
        cells_to_refine = []
        cells_to_coarsen = []
        
        for i, cell in enumerate(self.mesh.cells):
            grad_value = combined_gradient[i]
            
            if grad_value > self.refinement_threshold and cell.level < self.mesh.max_level:
                # 细化网格
                cells_to_refine.append(cell.id)
            elif grad_value < self.coarsening_threshold and cell.level > 0:
                # 粗化网格
                cells_to_coarsen.append(cell.id)
        
        # 执行细化
        for cell_id in cells_to_refine:
            self._refine_cell(cell_id)
        
        # 执行粗化
        for cell_id in cells_to_coarsen:
            self._coarsen_cell(cell_id)
        
        return self.mesh
    
    def _error_based_adaptation(self, physics_fields: Dict[str, np.ndarray]) -> AdaptiveMesh:
        """基于误差的网格自适应"""
        print("   使用误差自适应策略...")
        
        # 计算误差指标（基于物理场残差）
        error_indicators = self._compute_error_indicators(physics_fields)
        
        # 归一化误差
        if np.max(error_indicators) > 0:
            error_indicators = error_indicators / np.max(error_indicators)
        
        # 执行自适应
        cells_to_refine = []
        for i, cell in enumerate(self.mesh.cells):
            if error_indicators[i] > self.refinement_threshold and cell.level < self.mesh.max_level:
                cells_to_refine.append(cell.id)
        
        # 执行细化
        for cell_id in cells_to_refine:
            self._refine_cell(cell_id)
        
        return self.mesh
    
    def _ml_based_adaptation(self, physics_fields: Dict[str, np.ndarray], 
                            gradients: Dict[str, np.ndarray]) -> AdaptiveMesh:
        """基于机器学习的网格自适应"""
        if not HAS_PYTORCH:
            print("   警告：PyTorch不可用，回退到梯度自适应")
            return self._gradient_based_adaptation(gradients)
        
        print("   使用ML自适应策略...")
        
        # 构建特征向量
        features = self._build_adaptation_features(physics_fields, gradients)
        
        # 使用预训练的ML模型预测是否需要细化
        adaptation_decisions = self._predict_adaptation(features)
        
        # 执行自适应
        for i, decision in enumerate(adaptation_decisions):
            cell = self.mesh.cells[i]
            if decision > 0.7 and cell.level < self.mesh.max_level:  # 细化阈值
                self._refine_cell(cell.id)
            elif decision < 0.3 and cell.level > 0:  # 粗化阈值
                self._coarsen_cell(cell.id)
        
        return self.mesh
    
    def _build_adaptation_features(self, physics_fields: Dict[str, np.ndarray], 
                                 gradients: Dict[str, np.ndarray]) -> np.ndarray:
        """构建ML自适应的特征向量"""
        n_cells = len(self.mesh.cells)
        n_fields = len(physics_fields)
        
        features = np.zeros((n_cells, n_fields * 2 + 3))  # 场值+梯度+几何特征
        
        for i, cell in enumerate(self.mesh.cells):
            feature_idx = 0
            
            # 物理场值
            for field_name, field_values in physics_fields.items():
                cell_values = field_values[cell.nodes]
                features[i, feature_idx] = np.mean(cell_values)
                feature_idx += 1
            
            # 物理场梯度
            for field_name, field_gradients in gradients.items():
                features[i, feature_idx] = field_gradients[i]
                feature_idx += 1
            
            # 几何特征
            features[i, feature_idx] = cell.level  # 细化级别
            features[i, feature_idx + 1] = cell.volume  # 单元体积
            features[i, feature_idx + 2] = len(cell.nodes)  # 节点数
        
        return features
    
    def _predict_adaptation(self, features: np.ndarray) -> np.ndarray:
        """使用ML模型预测自适应决策"""
        # 这里应该使用预训练的ML模型
        # 目前使用简单的启发式规则作为示例
        
        # 基于特征计算自适应概率
        decisions = np.zeros(len(features))
        
        for i, feature in enumerate(features):
            # 简化规则：梯度大、体积小的单元倾向于细化
            gradient_factor = np.mean(feature[len(features)//2:-3])  # 梯度特征
            volume_factor = 1.0 / (1.0 + feature[-2])  # 体积特征
            
            # 综合决策
            decisions[i] = 0.5 * gradient_factor + 0.3 * volume_factor + 0.2 * np.random.random()
            decisions[i] = np.clip(decisions[i], 0.0, 1.0)
        
        return decisions
    
    def _compute_error_indicators(self, physics_fields: Dict[str, np.ndarray]) -> np.ndarray:
        """计算误差指标"""
        n_cells = len(self.mesh.cells)
        error_indicators = np.zeros(n_cells)
        
        for i, cell in enumerate(self.mesh.cells):
            # 计算单元内物理场的残差（简化）
            cell_error = 0.0
            for field_values in physics_fields.values():
                cell_values = field_values[cell.nodes]
                # 使用方差作为误差指标
                cell_error += np.var(cell_values)
            
            error_indicators[i] = cell_error
        
        return error_indicators
    
    def _refine_cell(self, cell_id: int):
        """细化网格单元"""
        cell = next(c for c in self.mesh.cells if c.id == cell_id)
        
        if cell.level >= self.mesh.max_level:
            return
        
        # 创建子单元（2D：4个子单元，3D：8个子单元）
        if self.mesh.dim == 2:
            self._refine_2d_cell(cell)
        else:
            self._refine_3d_cell(cell)
    
    def _refine_2d_cell(self, cell: MeshCell):
        """细化2D网格单元"""
        # 获取单元节点坐标
        node_coords = self.mesh.nodes[cell.nodes]
        
        # 计算单元中心
        center = np.mean(node_coords, axis=0)
        
        # 添加中心节点
        center_node_id = len(self.mesh.nodes)
        self.mesh.nodes = np.vstack([self.mesh.nodes, center])
        
        # 创建4个子单元
        n_nodes = len(cell.nodes)
        for i in range(n_nodes):
            next_i = (i + 1) % n_nodes
            
            # 子单元节点：原节点i、原节点next_i、中心节点
            sub_cell_nodes = [cell.nodes[i], cell.nodes[next_i], center_node_id]
            
            # 创建子单元
            sub_cell = MeshCell(
                id=len(self.mesh.cells),
                nodes=sub_cell_nodes,
                level=cell.level + 1,
                parent=cell.id
            )
            
            self.mesh.cells.append(sub_cell)
            cell.children.append(sub_cell.id)
    
    def _refine_3d_cell(self, cell: MeshCell):
        """细化3D网格单元（简化实现）"""
        # 3D细化更复杂，这里简化处理
        # 实际应用中需要更复杂的四面体分解算法
        
        # 获取单元节点坐标
        node_coords = self.mesh.nodes[cell.nodes]
        
        # 计算单元中心
        center = np.mean(node_coords, axis=0)
        
        # 添加中心节点
        center_node_id = len(self.mesh.nodes)
        self.mesh.nodes = np.vstack([self.mesh.nodes, center])
        
        # 创建8个子单元（简化：基于四面体）
        if len(cell.nodes) >= 4:
            # 使用前4个节点创建四面体
            for i in range(4):
                sub_cell_nodes = [cell.nodes[i]] + [center_node_id]
                if i < 3:
                    sub_cell_nodes.append(cell.nodes[i + 1])
                else:
                    sub_cell_nodes.append(cell.nodes[0])
                
                # 创建子单元
                sub_cell = MeshCell(
                    id=len(self.mesh.cells),
                    nodes=sub_cell_nodes,
                    level=cell.level + 1,
                    parent=cell.id
                )
                
                self.mesh.cells.append(sub_cell)
                cell.children.append(sub_cell.id)
    
    def _coarsen_cell(self, cell_id: int):
        """粗化网格单元"""
        cell = next(c for c in self.mesh.cells if c.id == cell_id)
        
        if not cell.children or cell.level == 0:
            return
        
        # 移除子单元
        for child_id in cell.children:
            child = next(c for c in self.mesh.cells if c.id == child_id)
            self.mesh.cells.remove(child)
        
        # 清空子单元列表
        cell.children.clear()
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """获取自适应统计信息"""
        total_cells = len(self.mesh.cells)
        refined_cells = sum(1 for cell in self.mesh.cells if cell.level > 0)
        max_level = max(cell.level for cell in self.mesh.cells)
        
        # 计算网格质量指标
        cell_volumes = [cell.volume for cell in self.mesh.cells]
        volume_ratio = max(cell_volumes) / min(cell_volumes) if min(cell_volumes) > 0 else float('inf')
        
        return {
            "total_cells": total_cells,
            "refined_cells": refined_cells,
            "max_refinement_level": max_level,
            "volume_ratio": volume_ratio,
            "adaptation_efficiency": refined_cells / total_cells if total_cells > 0 else 0.0
        }


def create_adaptive_mesh_example():
    """创建自适应网格示例"""
    print("🔧 创建自适应网格示例...")
    
    # 创建简单2D网格
    nodes = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # 4个角点
        [0.5, 0.5]  # 中心点
    ])
    
    cells = [
        MeshCell(id=0, nodes=[0, 1, 4], level=0),  # 左下三角形
        MeshCell(id=1, nodes=[1, 2, 4], level=0),  # 右下三角形
        MeshCell(id=2, nodes=[2, 3, 4], level=0),  # 右上三角形
        MeshCell(id=3, nodes=[3, 0, 4], level=0)   # 左上三角形
    ]
    
    mesh = AdaptiveMesh(nodes=nodes, cells=cells, dim=2)
    
    # 创建物理场（示例：温度场）
    n_nodes = len(nodes)
    temperature_field = np.zeros(n_nodes)
    
    # 设置温度梯度（中心热源）
    for i in range(n_nodes):
        dist_to_center = np.linalg.norm(nodes[i] - np.array([0.5, 0.5]))
        temperature_field[i] = 100.0 * np.exp(-dist_to_center / 0.3)
    
    physics_fields = {"temperature": temperature_field}
    
    # 创建自适应器
    adaptor = GradientBasedAdaptor(mesh)
    
    # 执行自适应
    adapted_mesh = adaptor.adapt_mesh_based_on_gradient(
        physics_fields, adaptation_strategy="gradient"
    )
    
    # 输出统计信息
    stats = adaptor.get_adaptation_statistics()
    print(f"✅ 自适应网格创建完成:")
    print(f"   总单元数: {stats['total_cells']}")
    print(f"   细化单元数: {stats['refined_cells']}")
    print(f"   最大细化级别: {stats['max_refinement_level']}")
    print(f"   体积比: {stats['volume_ratio']:.2f}")
    print(f"   自适应效率: {stats['adaptation_efficiency']:.2%}")
    
    return adapted_mesh, adaptor


if __name__ == "__main__":
    # 运行示例
    mesh, adaptor = create_adaptive_mesh_example()
