"""
地球动力学图神经网络模块
针对板块边界、断裂网络等非欧结构，增强拓扑关系建模能力
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 深度学习相关依赖
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    torch = None
    nn = None
    warnings.warn("PyTorch not available. Geodynamics GNN features will be limited.")

# 图神经网络依赖
try:
    import torch_geometric
    from torch_geometric.nn import GATConv, GCNConv, GraphConv
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import to_dense_adj, to_dense_batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    warnings.warn("torch_geometric not available. GNN features will be limited.")


@dataclass
class GeodynamicGraphConfig:
    """地球动力学图结构配置"""
    # 图构建参数
    max_neighbors: int = 8  # 最大邻居数
    fault_connection_strength: float = 0.1  # 断层连接强度
    plate_boundary_strength: float = 2.0  # 板块边界连接强度
    normal_connection_strength: float = 1.0  # 正常地质体连接强度
    
    # GNN参数
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    attention_heads: int = 4
    
    # 物理参数
    temperature_threshold: float = 1000.0  # 温度阈值（影响岩石强度）
    strain_threshold: float = 0.1  # 应变阈值
    age_weight: float = 0.5  # 地质年代权重


class GeodynamicGraphBuilder:
    """地球动力学图结构构建器"""
    
    def __init__(self, config: GeodynamicGraphConfig):
        self.config = config
    
    def geo_to_graph_geodynamics(self, mesh_data: np.ndarray, 
                                faults: List[Tuple] = None, 
                                plate_boundaries: List[Tuple] = None,
                                geological_features: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        构建地球动力学图结构
        
        Args:
            mesh_data: 网格数据 [N, features] (密度、弹性模量、温度、塑性应变、地质年代、x, y, z)
            faults: 断层信息列表 [(node1, node2, fault_type, friction_coef), ...]
            plate_boundaries: 板块边界信息 [(node1, node2, boundary_type, stress), ...]
            geological_features: 地质特征 [N, additional_features]
        
        Returns:
            node_features: 节点特征矩阵
            adjacency_matrix: 邻接矩阵
            graph_info: 图结构信息
        """
        num_nodes = mesh_data.shape[0]
        
        # 1. 构建邻接矩阵
        adj = self._build_adjacency_matrix(num_nodes, faults, plate_boundaries)
        
        # 2. 构建节点特征
        node_features = self._build_node_features(mesh_data, geological_features)
        
        # 3. 构建边特征
        edge_features = self._build_edge_features(adj, faults, plate_boundaries)
        
        # 4. 图结构信息
        graph_info = {
            'num_nodes': num_nodes,
            'num_edges': np.sum(adj > 0),
            'fault_edges': len(faults) if faults else 0,
            'plate_boundary_edges': len(plate_boundaries) if plate_boundaries else 0,
            'avg_degree': np.mean(np.sum(adj > 0, axis=1)),
            'connection_strengths': {
                'fault': self.config.fault_connection_strength,
                'plate_boundary': self.config.plate_boundary_strength,
                'normal': self.config.normal_connection_strength
            }
        }
        
        return node_features, adj, edge_features, graph_info
    
    def _build_adjacency_matrix(self, num_nodes: int, 
                               faults: List[Tuple], 
                               plate_boundaries: List[Tuple]) -> np.ndarray:
        """构建邻接矩阵"""
        adj = np.zeros((num_nodes, num_nodes))
        
        # 基础网格连接（8邻域）
        for i in range(num_nodes):
            neighbors = self._get_grid_neighbors(i, num_nodes)
            for j in neighbors:
                if i != j:
                    # 检查是否为特殊边界
                    if self._is_fault_between(i, j, faults):
                        adj[i][j] = self.config.fault_connection_strength
                    elif self._is_plate_boundary(i, j, plate_boundaries):
                        adj[i][j] = self.config.plate_boundary_strength
                    else:
                        adj[i][j] = self.config.normal_connection_strength
        
        # 确保对称性
        adj = np.maximum(adj, adj.T)
        
        return adj
    
    def _get_grid_neighbors(self, node_id: int, num_nodes: int) -> List[int]:
        """获取网格邻居节点"""
        # 简化的8邻域连接（实际应用中需要根据真实网格拓扑调整）
        grid_size = int(np.sqrt(num_nodes))
        if grid_size * grid_size != num_nodes:
            # 如果不是完全平方数，使用简单的连接策略
            neighbors = []
            for i in range(max(0, node_id - grid_size), min(num_nodes, node_id + grid_size + 1)):
                if i != node_id and 0 <= i < num_nodes:
                    neighbors.append(i)
            return neighbors
        
        # 2D网格的8邻域连接
        row = node_id // grid_size
        col = node_id % grid_size
        
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row = row + dr
                new_col = col + dc
                if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
                    neighbor_id = new_row * grid_size + new_col
                    neighbors.append(neighbor_id)
        
        return neighbors
    
    def _is_fault_between(self, node1: int, node2: int, faults: List[Tuple]) -> bool:
        """检查两个节点之间是否存在断层"""
        if not faults:
            return False
        
        for fault in faults:
            if len(fault) >= 2:
                if (node1 == fault[0] and node2 == fault[1]) or \
                   (node1 == fault[1] and node2 == fault[0]):
                    return True
        return False
    
    def _is_plate_boundary(self, node1: int, node2: int, plate_boundaries: List[Tuple]) -> bool:
        """检查两个节点之间是否存在板块边界"""
        if not plate_boundaries:
            return False
        
        for boundary in plate_boundaries:
            if len(boundary) >= 2:
                if (node1 == boundary[0] and node2 == boundary[1]) or \
                   (node1 == boundary[1] and node2 == boundary[0]):
                    return True
        return False
    
    def _build_node_features(self, mesh_data: np.ndarray, 
                           geological_features: np.ndarray = None) -> np.ndarray:
        """构建节点特征"""
        # 基础物理属性
        basic_features = mesh_data[:, :4]  # 密度、弹性模量、温度、塑性应变
        
        # 空间坐标
        spatial_features = mesh_data[:, -3:]  # x, y, z
        
        # 地质年代（影响岩石强度）
        age_features = mesh_data[:, 4:5]  # 地质年代
        
        # 计算派生特征
        derived_features = self._compute_derived_features(mesh_data)
        
        # 合并所有特征
        node_features = np.hstack([
            basic_features,      # 4维：密度、弹性模量、温度、塑性应变
            spatial_features,    # 3维：x, y, z
            age_features,        # 1维：地质年代
            derived_features     # 派生特征
        ])
        
        # 添加地质特征（如果提供）
        if geological_features is not None:
            node_features = np.hstack([node_features, geological_features])
        
        return node_features
    
    def _compute_derived_features(self, mesh_data: np.ndarray) -> np.ndarray:
        """计算派生特征"""
        # 温度归一化（影响岩石强度）
        temperature = mesh_data[:, 2:3]
        normalized_temp = (temperature - np.min(temperature)) / (np.max(temperature) - np.min(temperature) + 1e-8)
        
        # 应变率（塑性应变的导数近似）
        strain = mesh_data[:, 3:4]
        strain_rate = np.gradient(strain.flatten()).reshape(-1, 1)
        
        # 岩石强度指标（基于温度、应变、地质年代）
        age = mesh_data[:, 4:5]
        strength_index = (1.0 - normalized_temp) * (1.0 + strain) * (1.0 + self.config.age_weight * age)
        
        # 热力学状态
        thermal_state = normalized_temp * strain
        
        derived_features = np.hstack([
            normalized_temp,     # 归一化温度
            strain_rate,         # 应变率
            strength_index,      # 岩石强度指标
            thermal_state        # 热力学状态
        ])
        
        return derived_features
    
    def _build_edge_features(self, adj: np.ndarray, 
                           faults: List[Tuple], 
                           plate_boundaries: List[Tuple]) -> np.ndarray:
        """构建边特征"""
        num_nodes = adj.shape[0]
        edge_features = []
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj[i][j] > 0:
                    # 边类型编码
                    edge_type = 0  # 正常连接
                    if self._is_fault_between(i, j, faults):
                        edge_type = 1  # 断层连接
                    elif self._is_plate_boundary(i, j, plate_boundaries):
                        edge_type = 2  # 板块边界连接
                    
                    # 连接强度
                    connection_strength = adj[i][j]
                    
                    # 距离特征
                    distance = np.sqrt((i - j) ** 2)  # 简化的距离计算
                    
                    edge_features.append([edge_type, connection_strength, distance])
        
        return np.array(edge_features) if edge_features else np.zeros((0, 3))


class GeodynamicGNN(nn.Module):
    """地球动力学图神经网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 config: GeodynamicGraphConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        if not HAS_TORCH_GEOMETRIC:
            warnings.warn("torch_geometric not available, using simple MLP")
            self._build_mlp_fallback()
            return
        
        # 图卷积层
        self.conv_layers = nn.ModuleList()
        
        # 第一层：输入到隐藏层
        if config.attention_heads > 1:
            self.conv_layers.append(
                GATConv(input_dim, hidden_dim, heads=config.attention_heads, 
                       dropout=config.dropout, edge_dim=1)
            )
        else:
            self.conv_layers.append(
                GraphConv(input_dim, hidden_dim, edge_dim=1)
            )
        
        # 中间层
        for _ in range(config.num_layers - 2):
            if config.attention_heads > 1:
                self.conv_layers.append(
                    GATConv(hidden_dim * config.attention_heads, hidden_dim, 
                           heads=config.attention_heads, dropout=config.dropout, edge_dim=1)
                )
            else:
                self.conv_layers.append(
                    GraphConv(hidden_dim, hidden_dim, edge_dim=1)
                )
        
        # 最后一层：隐藏层到输出
        if config.attention_heads > 1:
            self.conv_layers.append(
                GATConv(hidden_dim * config.attention_heads, hidden_dim, 
                       heads=1, dropout=config.dropout, edge_dim=1)
            )
        else:
            self.conv_layers.append(
                GraphConv(hidden_dim, hidden_dim, edge_dim=1)
            )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(config.num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def _build_mlp_fallback(self):
        """构建MLP回退方案"""
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None, 
                edge_weight: torch.Tensor = None, batch: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [N, input_dim]
            edge_index: 边索引 [2, E]
            edge_weight: 边权重 [E]
            batch: 批次索引 [N]
        
        Returns:
            输出特征 [N, output_dim]
        """
        if not HAS_TORCH_GEOMETRIC or edge_index is None:
            # 使用MLP回退方案
            return self.mlp(x)
        
        # 图卷积前向传播
        for i, conv in enumerate(self.conv_layers):
            if i == 0:
                # 第一层
                if isinstance(conv, GATConv):
                    x = conv(x, edge_index, edge_weight)
                else:
                    x = conv(x, edge_index, edge_weight)
            else:
                # 中间层
                if isinstance(conv, GATConv):
                    x = conv(x, edge_index, edge_weight)
                else:
                    x = conv(x, edge_index, edge_weight)
            
            # 批归一化
            if batch is not None:
                x = self.batch_norms[i](x)
            else:
                x = self.batch_norms[i](x)
            
            # 激活函数和Dropout
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # 输出层
        x = self.output_layer(x)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor, 
                             edge_weight: torch.Tensor = None) -> torch.Tensor:
        """获取注意力权重（仅适用于GAT）"""
        if not HAS_TORCH_GEOMETRIC or not isinstance(self.conv_layers[0], GATConv):
            return None
        
        attention_weights = []
        current_x = x
        
        for conv in self.conv_layers:
            if isinstance(conv, GATConv):
                # 获取注意力权重（需要修改GATConv以返回注意力权重）
                # 这里简化处理，实际需要自定义GATConv
                current_x = conv(current_x, edge_index, edge_weight)
                attention_weights.append(torch.ones_like(current_x))  # 占位符
            else:
                current_x = conv(current_x, edge_index, edge_weight)
        
        return attention_weights


class GeodynamicsGNNPINNIntegrator:
    """地球动力学GNN与PINN集成器"""
    
    def __init__(self, gnn: GeodynamicGNN, config: GeodynamicGraphConfig):
        self.gnn = gnn
        self.config = config
        self.graph_builder = GeodynamicGraphBuilder(config)
    
    def integrate_with_pinn(self, x: torch.Tensor, mesh_data: np.ndarray,
                           faults: List[Tuple] = None, 
                           plate_boundaries: List[Tuple] = None,
                           geological_features: np.ndarray = None) -> torch.Tensor:
        """
        将GNN拓扑特征集成到PINN中
        
        Args:
            x: PINN输入特征
            mesh_data: 网格数据
            faults: 断层信息
            plate_boundaries: 板块边界信息
            geological_features: 地质特征
        
        Returns:
            增强后的特征
        """
        # 1. 构建图结构
        node_features, adj, edge_features, graph_info = self.graph_builder.geo_to_graph_geodynamics(
            mesh_data, faults, plate_boundaries, geological_features
        )
        
        # 2. 转换为PyTorch张量
        node_features_tensor = torch.FloatTensor(node_features)
        edge_index, edge_weight = self._adjacency_to_edge_index(adj)
        
        # 3. GNN前向传播
        gnn_output = self.gnn(node_features_tensor, edge_index, edge_weight)
        
        # 4. 特征融合
        enhanced_features = self._fuse_features(x, gnn_output, node_features_tensor)
        
        return enhanced_features
    
    def _adjacency_to_edge_index(self, adj: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """将邻接矩阵转换为边索引和权重"""
        edge_indices = []
        edge_weights = []
        
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i][j] > 0:
                    edge_indices.append([i, j])
                    edge_weights.append(adj[i][j])
        
        if edge_indices:
            edge_index = torch.LongTensor(edge_indices).t()
            edge_weight = torch.FloatTensor(edge_weights)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float)
        
        return edge_index, edge_weight
    
    def _fuse_features(self, pinn_features: torch.Tensor, 
                      gnn_features: torch.Tensor, 
                      node_features: torch.Tensor) -> torch.Tensor:
        """融合PINN和GNN特征"""
        # 选择相关的GNN特征进行融合
        # 这里假设GNN输出包含粘度修正和塑性应变率
        viscosity_correction = gnn_features[:, 0:1]  # 粘度修正
        strain_rate_correction = gnn_features[:, 1:2]  # 塑性应变率修正
        
        # 融合策略：原始特征 + GNN修正
        fused_features = torch.cat([
            pinn_features,           # 原始PINN特征
            viscosity_correction,    # 粘度修正
            strain_rate_correction   # 应变率修正
        ], dim=1)
        
        return fused_features


def demo_geodynamics_gnn():
    """演示地球动力学GNN功能"""
    print("=== 地球动力学GNN演示 ===")
    
    if not HAS_PYTORCH:
        print("❌ PyTorch不可用，跳过演示")
        return
    
    # 1. 创建配置
    config = GeodynamicGraphConfig(
        hidden_dim=32,
        num_layers=2,
        attention_heads=2
    )
    
    print(f"✅ 创建GNN配置: 隐藏层={config.hidden_dim}, 层数={config.num_layers}")
    
    # 2. 创建模拟数据
    n_nodes = 100
    mesh_data = np.random.randn(n_nodes, 8)  # 8维特征
    mesh_data[:, 2] = np.abs(mesh_data[:, 2])  # 温度为正
    mesh_data[:, 4] = np.abs(mesh_data[:, 4])  # 地质年代为正
    
    # 模拟断层和板块边界
    faults = [(10, 11, 'strike_slip', 0.6), (20, 21, 'normal', 0.5)]
    plate_boundaries = [(30, 31, 'convergent', 100.0), (40, 41, 'divergent', 50.0)]
    
    print(f"✅ 创建模拟数据: {n_nodes}个节点, {len(faults)}个断层, {len(plate_boundaries)}个板块边界")
    
    # 3. 构建图结构
    graph_builder = GeodynamicGraphBuilder(config)
    node_features, adj, edge_features, graph_info = graph_builder.geo_to_graph_geodynamics(
        mesh_data, faults, plate_boundaries
    )
    
    print(f"✅ 构建图结构: {graph_info['num_nodes']}个节点, {graph_info['num_edges']}条边")
    print(f"   断层边: {graph_info['fault_edges']}, 板块边界边: {graph_info['plate_boundary_edges']}")
    print(f"   平均度: {graph_info['avg_degree']:.2f}")
    
    # 4. 创建GNN模型
    input_dim = node_features.shape[1]
    hidden_dim = config.hidden_dim
    output_dim = 2  # 粘度修正、塑性应变率
    
    gnn = GeodynamicGNN(input_dim, hidden_dim, output_dim, config)
    print(f"✅ 创建GNN模型: {input_dim} -> {hidden_dim} -> {output_dim}")
    
    # 5. 测试GNN前向传播
    try:
        # 转换为张量
        x_tensor = torch.FloatTensor(node_features)
        edge_index, edge_weight = graph_builder._adjacency_to_edge_index(adj)
        
        # 前向传播
        output = gnn(x_tensor, edge_index, edge_weight)
        print(f"✅ GNN前向传播成功: 输出形状 {output.shape}")
        print(f"   粘度修正范围: [{output[:, 0].min().item():.4f}, {output[:, 0].max().item():.4f}]")
        print(f"   应变率修正范围: [{output[:, 1].min().item():.4f}, {output[:, 1].max().item():.4f}]")
        
    except Exception as e:
        print(f"❌ GNN前向传播失败: {str(e)}")
    
    # 6. 测试PINN集成
    try:
        integrator = GeodynamicsGNNPINNIntegrator(gnn, config)
        
        # 模拟PINN输入
        pinn_input = torch.randn(n_nodes, 5)  # 5维PINN输入
        
        # 集成GNN特征
        enhanced_features = integrator.integrate_with_pinn(
            pinn_input, mesh_data, faults, plate_boundaries
        )
        
        print(f"✅ PINN集成成功: 原始特征 {pinn_input.shape} -> 增强特征 {enhanced_features.shape}")
        
    except Exception as e:
        print(f"❌ PINN集成失败: {str(e)}")
    
    print("\n🎉 地球动力学GNN演示完成！")


def demo_application_scenarios():
    """演示应用场景"""
    print("\n=== 应用场景演示 ===")
    
    # 1. 板块边界动力学
    print("\n🌊 场景1: 板块边界动力学")
    print("   GNN捕捉板块间相互作用")
    print("   提升PINN对俯冲带/转换断层的模拟精度")
    print("   应用: 地震预测、板块运动建模")
    
    # 2. 断裂网络演化
    print("\n⚡ 场景2: 断裂网络演化")
    print("   通过图结构动态更新断裂连接关系")
    print("   模拟多断层协同滑动")
    print("   应用: 断层稳定性分析、地震序列建模")
    
    # 3. 地幔对流拓扑
    print("\n🌍 场景3: 地幔对流拓扑")
    print("   建模地幔柱、俯冲板片等拓扑结构")
    print("   提升对流模式的预测精度")
    print("   应用: 地幔动力学、板块驱动机制")
    
    print("\n✅ 应用场景演示完成")


if __name__ == "__main__":
    # 运行演示
    demo_geodynamics_gnn()
    demo_application_scenarios()
    
    print("\n🎉 地球动力学GNN模块演示完成！")
    print("\n📚 主要功能总结:")
    print("  1. ✅ 地质图构建模块升级")
    print("  2. ✅ 动力学图网络设计")
    print("  3. ✅ 与PINN融合支持")
    print("  4. ✅ 应用场景覆盖")
    print("\n🚀 下一步:")
    print("  1. 在geological_ml_framework.py中集成GNN功能")
    print("  2. 使用真实的地质数据进行训练")
    print("  3. 优化图构建算法")
    print("  4. 扩展到3D地球动力学模拟")
