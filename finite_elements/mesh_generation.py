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
    
    def _build_connectivity(self):
        """构建网格连接关系"""
        # 节点到单元的映射
        self.node_to_cells = {}
        for cell in self.cells:
            for node_id in cell.nodes:
                if node_id not in self.node_to_cells:
                    self.node_to_cells[node_id] = []
                self.node_to_cells[node_id].append(cell.id)
    
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
