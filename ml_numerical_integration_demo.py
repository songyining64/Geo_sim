"""
机器学习与数值模拟集成演示

展示如何将机器学习方法集成到传统数值模拟中，
包括自适应物理约束、多保真度建模等高级功能。
"""

import numpy as np
import time
import warnings

# 导入核心功能
try:
    from gpu_acceleration.advanced_ml import demo_rl_solver_optimization
    from gpu_acceleration.geological_ml_framework import demo_geological_ml
    from ensemble.multi_fidelity import demo_multi_fidelity
    print("✅ 成功导入所有模块")
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保所有依赖已正确安装")


def demo_adaptive_physical_constraints():
    """演示自适应物理约束功能"""
    print("\n=== 自适应物理约束演示 ===")
    
    try:
        from gpu_acceleration.geological_ml_framework import (
            PhysicsInformedNeuralNetwork, 
            AdaptivePhysicalConstraint,
            RLConstraintController
        )
        
        # 创建PINN模型
        pinn = PhysicsInformedNeuralNetwork(
            input_dim=3,
            output_dim=2,
            hidden_layers=[32, 16],
            adaptive_constraints=True
        )
        
        # 添加物理约束
        def darcy_constraint(x, y_pred):
            """Darcy流动方程约束"""
            return np.random.normal(0, 1e-6)  # 模拟残差
        
        def heat_constraint(x, y_pred):
            """热传导方程约束"""
            return np.random.normal(0, 1e-5)  # 模拟残差
        
        # 创建自适应约束
        darcy_constraint_adaptive = AdaptivePhysicalConstraint(
            name="Darcy方程",
            equation=darcy_constraint,
            initial_weight=1.0,
            min_weight=0.01,
            max_weight=5.0
        )
        
        heat_constraint_adaptive = AdaptivePhysicalConstraint(
            name="热传导方程",
            equation=heat_constraint,
            initial_weight=0.5,
            min_weight=0.01,
            max_weight=3.0
        )
        
        # 添加到PINN
        pinn.add_physical_constraint(darcy_constraint_adaptive)
        pinn.add_physical_constraint(heat_constraint_adaptive)
        
        print(f"✅ 添加物理约束: {len(pinn.physical_constraints)} 个")
        
        # 设置约束控制器
        pinn.setup_constraint_controller()
        print("✅ 设置约束控制器")
        
        # 模拟约束权重自适应过程
        print("🔄 模拟约束权重自适应过程...")
        for step in range(10):
            # 模拟计算残差
            residual1 = darcy_constraint_adaptive.compute_residual()
            residual2 = heat_constraint_adaptive.compute_residual()
            
            # 自适应调整权重
            darcy_constraint_adaptive.adapt_weight(residual1)
            heat_constraint_adaptive.adapt_weight(residual2)
            
            if step % 2 == 0:
                print(f"步骤 {step}: Darcy权重={darcy_constraint_adaptive.current_weight:.4f}, "
                      f"热传导权重={heat_constraint_adaptive.current_weight:.4f}")
        
        # 获取自适应摘要
        darcy_summary = darcy_constraint_adaptive.get_adaptation_summary()
        heat_summary = heat_constraint_adaptive.get_adaptation_summary()
        
        print(f"✅ Darcy约束自适应摘要: {darcy_summary['total_adaptations']} 次调整")
        print(f"✅ 热传导约束自适应摘要: {heat_summary['total_adaptations']} 次调整")
        
        # 获取控制器摘要
        if pinn.constraint_controller:
            control_summary = pinn.constraint_controller.get_control_summary()
            print(f"✅ 约束控制器摘要: {control_summary['total_iterations']} 次迭代")
        
    except Exception as e:
        print(f"❌ 自适应物理约束演示失败: {e}")


def demo_multi_fidelity_modeling():
    """演示多保真度建模功能"""
    print("\n=== 多保真度建模演示 ===")
    
    try:
        from ensemble.multi_fidelity import (
            FidelityLevel, 
            MultiFidelityConfig, 
            create_multi_fidelity_system
        )
        
        # 创建保真度级别
        low_fidelity = FidelityLevel(
            name="快速油藏仿真",
            level=0,
            description="使用简化PDE的快速油藏仿真",
            computational_cost=1.0,
            accuracy=0.75,
            data_requirements=800,
            training_time=90.0,
            model_type="neural_network",
            model_params={
                'hidden_layers': [32, 16],
                'activation': 'relu',
                'dropout': 0.1
            }
        )
        
        high_fidelity = FidelityLevel(
            name="精确油藏仿真",
            level=1,
            description="使用完整物理模型的精确油藏仿真",
            computational_cost=8.0,
            accuracy=0.92,
            data_requirements=4000,
            training_time=480.0,
            model_type="neural_network",
            model_params={
                'hidden_layers': [128, 64, 32],
                'activation': 'relu',
                'dropout': 0.2
            }
        )
        
        # 创建配置
        config = MultiFidelityConfig(
            name="油藏预测多保真度系统",
            description="结合快速和精确仿真的油藏生产预测系统",
            fidelity_levels=[low_fidelity, high_fidelity],
            co_training=MultiFidelityConfig.co_training(
                enabled=True,
                transfer_learning=True,
                knowledge_distillation=True,
                ensemble_method='weighted_average'
            ),
            training_strategy=MultiFidelityConfig.training_strategy(
                stage1_epochs=50,   # 低保真度预训练
                stage2_epochs=30,   # 高保真度微调
                transfer_epochs=15, # 知识迁移
                distillation_epochs=10  # 知识蒸馏
            )
        )
        
        print("✅ 创建多保真度配置")
        
        # 创建系统
        trainer = create_multi_fidelity_system(config)
        print("✅ 创建多保真度训练器")
        
        # 生成模拟数据
        np.random.seed(42)
        
        # 低保真度数据（快速仿真结果）
        X_low = np.random.randn(800, 6)   # 6个输入特征：压力、温度、饱和度等
        y_low = np.random.randn(800, 3)   # 3个输出：油、水、气产量
        
        # 高保真度数据（精确仿真结果）
        X_high = np.random.randn(4000, 6)
        y_high = np.random.randn(4000, 3)
        
        # 验证数据
        X_val_low = np.random.randn(150, 6)
        y_val_low = np.random.randn(150, 3)
        
        X_val_high = np.random.randn(800, 6)
        y_val_high = np.random.randn(800, 3)
        
        # 添加数据
        trainer.add_training_data(0, X_low, y_low)
        trainer.add_training_data(1, X_high, y_high)
        trainer.add_validation_data(0, X_val_low, y_val_low)
        trainer.add_validation_data(1, X_val_high, y_val_high)
        
        print("✅ 添加训练和验证数据")
        
        # 运行训练（使用较少的epochs进行演示）
        print("🚀 开始多保真度训练流程...")
        training_summary = trainer.run_full_training(input_dim=6, output_dim=3)
        
        print("✅ 训练完成")
        print(f"训练摘要: 阶段1={training_summary['stage1']}, 阶段2={training_summary['stage2']}")
        
        # 测试集成预测
        X_test = np.random.randn(100, 6)
        predictions = trainer.predict_with_ensemble(X_test)
        
        print(f"✅ 集成预测完成，输出形状: {predictions['ensemble'].shape}")
        
        # 显示性能对比
        if 'final_evaluation' in training_summary:
            print("\n📊 各保真度级别性能对比:")
            for level, metrics in training_summary['final_evaluation'].items():
                if level != 'ensemble':
                    print(f"  {level}: {metrics}")
        
    except Exception as e:
        print(f"❌ 多保真度建模演示失败: {e}")


def demo_integrated_features():
    """演示集成功能"""
    print("\n=== 集成功能演示 ===")
    
    try:
        # 1. 自适应物理约束
        demo_adaptive_physical_constraints()
        
        # 2. 多保真度建模
        demo_multi_fidelity_modeling()
        
        # 3. 强化学习求解器优化
        print("\n🔄 启动强化学习求解器优化演示...")
        demo_rl_solver_optimization()
        
        # 4. 地质图神经网络
        print("\n🔄 启动地质图神经网络演示...")
        demo_geological_gnn()
        
        print("\n🎉 所有集成功能演示完成！")
        
    except Exception as e:
        print(f"❌ 集成功能演示失败: {e}")


def main():
    """主函数"""
    print("🚀 机器学习与数值模拟集成演示")
    print("=" * 50)
    
    # 检查依赖
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch未安装")
        return
    
    try:
        import numpy as np
        print(f"✅ NumPy版本: {np.__version__}")
    except ImportError:
        print("❌ NumPy未安装")
        return
    
    # 运行演示
    print("\n选择演示模式:")
    print("1. 自适应物理约束")
    print("2. 多保真度建模")
    print("3. 强化学习求解器优化")
    print("4. 地质图神经网络")
    print("5. 所有功能集成演示")
    
    try:
        choice = input("\n请输入选择 (1-5, 默认5): ").strip()
        if not choice:
            choice = "5"
        
        if choice == "1":
            demo_adaptive_physical_constraints()
        elif choice == "2":
            demo_multi_fidelity_modeling()
        elif choice == "3":
            demo_rl_solver_optimization()
        elif choice == "4":
            demo_geological_gnn()
        elif choice == "5":
            demo_integrated_features()
        else:
            print("无效选择，运行所有功能演示")
            demo_integrated_features()
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
    
    print("\n👋 演示结束，感谢使用！")


if __name__ == "__main__":
    main()
