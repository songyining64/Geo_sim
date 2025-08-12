#!/usr/bin/env python3
"""
强化学习与图神经网络在地质数值模拟中的融合演示
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

def demo_rl_solver_optimization():
    """演示强化学习求解器优化"""
    print("🤖 强化学习求解器优化演示")
    print("=" * 60)
    
    try:
        from advanced_ml import create_rl_solver_system
        
        # 创建RL求解器系统
        rl_system = create_rl_solver_system()
        
        # 配置求解器环境
        solver_config = {
            'max_steps': 50,
            'convergence_threshold': 1e-6
        }
        
        # 创建环境
        env = rl_system['environment'](solver_config)
        state_dim = len(env.reset())
        action_dim = len(env.action_bounds)
        
        print(f"📊 环境配置:")
        print(f"   状态维度: {state_dim}")
        print(f"   动作维度: {action_dim}")
        print(f"   最大步数: {env.max_steps}")
        
        # 创建RL优化器
        rl_optimizer = rl_system['optimizer'](state_dim, action_dim, solver_config)
        
        # 训练RL智能体
        print("\n🔧 训练RL智能体...")
        training_history = rl_optimizer.train(episodes=300)
        
        print(f"   训练完成，最终平均奖励: {training_history['final_avg_reward']:.4f}")
        
        # 测试优化后的策略
        print("\n🔧 测试优化后的求解策略...")
        test_state = np.array([0.0, 0.5, 0.1, 0.8, 0.3])
        optimal_strategy = rl_optimizer.optimize_solver_strategy(test_state)
        
        print("\n✅ RL求解器优化演示完成!")
        return True
        
    except Exception as e:
        print(f"❌ RL求解器优化演示失败: {e}")
        return False


def demo_geological_gnn():
    """演示地质图神经网络"""
    print("🕸️ 地质图神经网络演示")
    print("=" * 60)
    
    try:
        from geological_ml_framework import create_geological_ml_system
        
        # 创建地质ML系统
        ml_system = create_geological_ml_system()
        
        # 生成地质图数据
        print("📊 生成地质图数据...")
        n_points = 200
        spatial_coords = np.random.rand(n_points, 3) * 10.0
        geological_features = np.random.rand(n_points, 5)
        
        # 创建GNN模型
        print("\n🔧 创建地质GNN模型...")
        gnn = ml_system['gnn'](
            node_features=5,
            edge_features=2,
            hidden_dim=32,
            num_layers=2,
            output_dim=1,
            gnn_type='gcn'
        )
        
        # 创建地质图结构
        edge_index, edge_features = gnn.create_geological_graph(
            spatial_coords, geological_features, connectivity_radius=2.0
        )
        
        # 分析拓扑结构
        topology_analysis = gnn.analyze_topology(geological_features, edge_index)
        print(f"   拓扑分析结果:")
        print(f"     节点数: {topology_analysis['num_nodes']}")
        print(f"     边数: {topology_analysis['num_edges']}")
        
        # 训练GNN
        target = np.random.randn(n_points, 1)
        training_history = gnn.train(
            geological_features, edge_index, target, edge_features,
            epochs=50, learning_rate=0.001
        )
        
        print(f"   训练完成，最终损失: {training_history['loss'][-1]:.6f}")
        
        print("\n✅ 地质GNN演示完成!")
        return True
        
    except Exception as e:
        print(f"❌ 地质GNN演示失败: {e}")
        return False


def main():
    """主函数"""
    print("🌟 强化学习与图神经网络在地质数值模拟中的融合演示")
    print("=" * 80)
    
    # 1. RL求解器优化演示
    print("\n" + "="*60)
    demo_rl_solver_optimization()
    
    # 2. 地质GNN演示
    print("\n" + "="*60)
    demo_geological_gnn()
    
    print("\n🎉 所有演示完成！")


if __name__ == "__main__":
    main()
