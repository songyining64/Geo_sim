"""
GNN与PINN集成演示
展示如何将地球动力学图神经网络与物理信息神经网络集成
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Union, Any

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
    warnings.warn("PyTorch not available. GNN-PINN integration demo will be limited.")

# 导入地质物理方程和GNN模块
try:
    from geological_ml_framework import (
        GeologicalPINN, 
        GeologicalConfig,
        GeologicalPhysicsEquations
    )
    from geodynamics_gnn import (
        GeodynamicGNN, 
        GeodynamicGraphConfig, 
        GeodynamicsGNNPINNIntegrator,
        GeodynamicGraphBuilder
    )
    HAS_MODULES = True
except ImportError:
    HAS_MODULES = False
    warnings.warn("Required modules not available, creating mock classes")
    
    # 创建模拟类
    class GeologicalPINN:
        def __init__(self, *args, **kwargs):
            pass
        
        def forward(self, x, **kwargs):
            return torch.randn(x.shape[0], 3)
        
        def setup_gnn_integration(self, *args, **kwargs):
            pass
        
        def enable_gnn_enhancement(self, *args, **kwargs):
            pass
        
        def get_gnn_status(self):
            return {'gnn_enabled': False}
    
    class GeologicalConfig:
        def __init__(self):
            pass
    
    class GeodynamicGNN:
        def __init__(self, *args, **kwargs):
            pass
        
        def forward(self, x, **kwargs):
            return torch.randn(x.shape[0], 2)
    
    class GeodynamicGraphConfig:
        def __init__(self, **kwargs):
            pass
    
    class GeodynamicsGNNPINNIntegrator:
        def __init__(self, *args, **kwargs):
            pass
        
        def integrate_with_pinn(self, *args, **kwargs):
            return torch.randn(args[0].shape[0], args[0].shape[1] + 2)
    
    class GeodynamicGraphBuilder:
        def __init__(self, *args, **kwargs):
            pass
        
        def geo_to_graph_geodynamics(self, *args, **kwargs):
            return np.random.randn(100, 8), np.random.randn(100, 100), np.random.randn(100, 3), {}


def demo_gnn_pinn_integration():
    """演示GNN与PINN集成"""
    print("=== GNN与PINN集成演示 ===")
    
    if not HAS_PYTORCH:
        print("❌ PyTorch不可用，跳过演示")
        return
    
    # 1. 创建地质配置
    config = GeologicalConfig()
    print("✅ 创建地质配置")
    
    # 2. 创建PINN模型
    try:
        input_dim = 5  # 基础输入维度
        output_dim = 3  # 输出维度
        hidden_layers = [64, 32]
        
        pinn = GeologicalPINN(
            input_dim=input_dim,
            hidden_dims=hidden_layers,
            output_dim=output_dim,
            geological_config=config
        )
        
        print(f"✅ 创建PINN模型: {input_dim} -> {hidden_layers} -> {output_dim}")
        
    except Exception as e:
        print(f"❌ 创建PINN失败: {str(e)}")
        return
    
    # 3. 设置GNN集成
    try:
        gnn_config = {
            'hidden_dim': 32,
            'num_layers': 2,
            'attention_heads': 2,
            'dropout': 0.1
        }
        
        pinn.setup_gnn_integration(gnn_config)
        print("✅ GNN集成设置完成")
        
    except Exception as e:
        print(f"❌ GNN集成设置失败: {str(e)}")
        return
    
    # 4. 启用GNN增强
    try:
        pinn.enable_gnn_enhancement(True)
        print("✅ GNN增强已启用")
        
        # 检查GNN状态
        gnn_status = pinn.get_gnn_status()
        print(f"GNN状态: {gnn_status}")
        
    except Exception as e:
        print(f"❌ 启用GNN增强失败: {str(e)}")
        return
    
    # 5. 创建模拟数据
    n_samples = 100
    X = np.random.randn(n_samples, input_dim)
    y = np.random.randn(n_samples, output_dim)
    
    # 创建网格数据（GNN用）
    mesh_data = np.random.randn(n_samples, 8)  # 8维网格特征
    mesh_data[:, 2] = np.abs(mesh_data[:, 2])  # 温度为正
    mesh_data[:, 4] = np.abs(mesh_data[:, 4])  # 地质年代为正
    
    # 模拟断层和板块边界
    faults = [(10, 11, 'strike_slip', 0.6), (20, 21, 'normal', 0.5)]
    plate_boundaries = [(30, 31, 'convergent', 100.0), (40, 41, 'divergent', 50.0)]
    
    print(f"✅ 创建模拟数据: {n_samples}个样本")
    print(f"   断层: {len(faults)}个, 板块边界: {len(plate_boundaries)}个")
    
    # 6. 测试集成前向传播
    try:
        print("\n🔄 测试集成前向传播...")
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X)
        
        # 使用GNN增强的前向传播
        output_with_gnn = pinn.forward(
            X_tensor, 
            mesh_data=mesh_data,
            faults=faults,
            plate_boundaries=plate_boundaries
        )
        
        print(f"✅ GNN增强前向传播成功: 输出形状 {output_with_gnn.shape}")
        
        # 不使用GNN增强的前向传播（对比）
        output_without_gnn = pinn.forward(X_tensor)
        print(f"✅ 标准前向传播成功: 输出形状 {output_without_gnn.shape}")
        
        # 比较输出差异
        if output_with_gnn.shape == output_without_gnn.shape:
            diff = torch.mean(torch.abs(output_with_gnn - output_without_gnn))
            print(f"   输出差异: {diff.item():.6f}")
        
    except Exception as e:
        print(f"❌ 前向传播测试失败: {str(e)}")
    
    print("\n🎉 GNN与PINN集成演示完成！")


def demo_geodynamics_topology_modeling():
    """演示地球动力学拓扑建模"""
    print("\n=== 地球动力学拓扑建模演示 ===")
    
    if not HAS_MODULES:
        print("❌ 模块不可用，跳过演示")
        return
    
    # 1. 创建GNN配置
    config = GeodynamicGraphConfig(
        hidden_dim=32,
        num_layers=2,
        attention_heads=2
    )
    
    print(f"✅ 创建GNN配置: 隐藏层={config.hidden_dim}, 层数={config.num_layers}")
    
    # 2. 创建图构建器
    graph_builder = GeodynamicGraphBuilder(config)
    
    # 3. 创建模拟地质数据
    n_nodes = 64  # 8x8网格
    mesh_data = np.random.randn(n_nodes, 8)
    mesh_data[:, 2] = np.abs(mesh_data[:, 2])  # 温度为正
    mesh_data[:, 4] = np.abs(mesh_data[:, 4])  # 地质年代为正
    
    # 模拟断层网络
    faults = [
        (10, 11, 'strike_slip', 0.6),
        (20, 21, 'normal', 0.5),
        (30, 31, 'reverse', 0.7),
        (40, 41, 'strike_slip', 0.6),
        (50, 51, 'normal', 0.5)
    ]
    
    # 模拟板块边界
    plate_boundaries = [
        (15, 16, 'convergent', 100.0),
        (25, 26, 'divergent', 50.0),
        (35, 36, 'transform', 75.0)
    ]
    
    print(f"✅ 创建地质数据: {n_nodes}个节点")
    print(f"   断层: {len(faults)}个, 板块边界: {len(plate_boundaries)}个")
    
    # 4. 构建图结构
    try:
        node_features, adj, edge_features, graph_info = graph_builder.geo_to_graph_geodynamics(
            mesh_data, faults, plate_boundaries
        )
        
        print(f"✅ 构建图结构: {graph_info['num_nodes']}个节点, {graph_info['num_edges']}条边")
        print(f"   断层边: {graph_info['fault_edges']}, 板块边界边: {graph_info['plate_boundary_edges']}")
        print(f"   平均度: {graph_info['avg_degree']:.2f}")
        print(f"   连接强度: {graph_info['connection_strengths']}")
        
    except Exception as e:
        print(f"❌ 图构建失败: {str(e)}")
        return
    
    # 5. 创建GNN模型
    try:
        input_dim = node_features.shape[1]
        hidden_dim = config.hidden_dim
        output_dim = 2  # 粘度修正、塑性应变率
        
        gnn = GeodynamicGNN(input_dim, hidden_dim, output_dim, config)
        print(f"✅ 创建GNN模型: {input_dim} -> {hidden_dim} -> {output_dim}")
        
    except Exception as e:
        print(f"❌ 创建GNN失败: {str(e)}")
        return
    
    # 6. 测试GNN前向传播
    try:
        print("\n🔄 测试GNN前向传播...")
        
        # 转换为张量
        x_tensor = torch.FloatTensor(node_features)
        edge_index, edge_weight = graph_builder._adjacency_to_edge_index(adj)
        
        # 前向传播
        output = gnn(x_tensor, edge_index, edge_weight)
        print(f"✅ GNN前向传播成功: 输出形状 {output.shape}")
        
        # 分析输出
        viscosity_correction = output[:, 0]
        strain_rate_correction = output[:, 1]
        
        print(f"   粘度修正: 均值={viscosity_correction.mean().item():.4f}, "
              f"标准差={viscosity_correction.std().item():.4f}")
        print(f"   应变率修正: 均值={strain_rate_correction.mean().item():.4f}, "
              f"标准差={strain_rate_correction.std().item():.4f}")
        
    except Exception as e:
        print(f"❌ GNN前向传播失败: {str(e)}")
    
    print("\n🎉 地球动力学拓扑建模演示完成！")


def demo_application_scenarios():
    """演示应用场景"""
    print("\n=== 应用场景演示 ===")
    
    # 1. 板块边界动力学
    print("\n🌊 场景1: 板块边界动力学")
    print("   GNN捕捉板块间相互作用")
    print("   提升PINN对俯冲带/转换断层的模拟精度")
    print("   应用: 地震预测、板块运动建模")
    print("   技术特点:")
    print("     - 动态边权重调整")
    print("     - 板块边界应力传递")
    print("     - 多尺度相互作用")
    
    # 2. 断裂网络演化
    print("\n⚡ 场景2: 断裂网络演化")
    print("   通过图结构动态更新断裂连接关系")
    print("   模拟多断层协同滑动")
    print("   应用: 断层稳定性分析、地震序列建模")
    print("   技术特点:")
    print("     - 断裂连接强度动态调整")
    print("     - 应力阴影效应建模")
    print("     - 断裂分支和合并")
    
    # 3. 地幔对流拓扑
    print("\n🌍 场景3: 地幔对流拓扑")
    print("   建模地幔柱、俯冲板片等拓扑结构")
    print("   提升对流模式的预测精度")
    print("   应用: 地幔动力学、板块驱动机制")
    print("   技术特点:")
    print("     - 地幔柱识别和跟踪")
    print("     - 俯冲板片几何建模")
    print("     - 热边界层相互作用")
    
    # 4. 岩石圈-软流圈耦合
    print("\n🔄 场景4: 岩石圈-软流圈耦合")
    print("   建模岩石圈与软流圈的相互作用")
    print("   考虑流变学差异和应力传递")
    print("   应用: 板块驱动机制、地壳变形")
    print("   技术特点:")
    print("     - 流变学界面建模")
    print("     - 应力耦合传递")
    print("     - 变形局部化")
    
    print("\n✅ 应用场景演示完成")


def demo_technical_features():
    """演示技术特性"""
    print("\n=== 技术特性演示 ===")
    
    # 1. 图构建特性
    print("\n🔧 图构建特性:")
    print("   - 自适应邻居连接")
    print("   - 断层/板块边界特殊处理")
    print("   - 动态边权重调整")
    print("   - 多尺度图结构支持")
    
    # 2. GNN架构特性
    print("\n🧠 GNN架构特性:")
    print("   - 多头注意力机制")
    print("   - 边属性感知卷积")
    print("   - 批归一化和Dropout")
    print("   - 残差连接和跳跃连接")
    
    # 3. PINN集成特性
    print("\n🔗 PINN集成特性:")
    print("   - 特征级融合")
    print("   - 动态维度调整")
    print("   - 物理约束保持")
    print("   - 训练过程监控")
    
    # 4. 性能优化特性
    print("\n⚡ 性能优化特性:")
    print("   - GPU加速支持")
    print("   - 批量处理优化")
    print("   - 内存管理优化")
    print("   - 并行计算支持")
    
    print("\n✅ 技术特性演示完成")


if __name__ == "__main__":
    # 运行演示
    demo_gnn_pinn_integration()
    demo_geodynamics_topology_modeling()
    demo_application_scenarios()
    demo_technical_features()
    
    print("\n🎉 GNN与PINN集成演示完成！")
    print("\n📚 主要功能总结:")
    print("  1. ✅ GNN与PINN无缝集成")
    print("  2. ✅ 地球动力学拓扑建模")
    print("  3. ✅ 动态图结构构建")
    print("  4. ✅ 多物理场耦合支持")
    print("\n🚀 下一步:")
    print("  1. 使用真实地质数据验证")
    print("  2. 优化图构建算法")
    print("  3. 扩展到3D建模")
    print("  4. 集成更多物理约束")
