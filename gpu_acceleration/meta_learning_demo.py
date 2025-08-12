"""
地球动力学元学习演示
展示如何使用元学习快速适配不同地质场景
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# 导入地质ML框架
try:
    from geological_ml_framework import (
        GeologicalPINN, GeologicalConfig, GeologicalPhysicsEquations,
        GeodynamicMetaLearner, GeodynamicMetaTask
    )
    print("✅ 成功导入地质ML框架")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保在正确的目录下运行此脚本")
    exit(1)


def demo_meta_learning_basic():
    """演示基本元学习功能"""
    print("\n🚀 演示基本元学习功能...")
    
    # 1. 创建地质配置
    config = GeologicalConfig(
        reference_viscosity=1e21,
        thermal_expansion=3e-5,
        gravity=9.81,
        mu0=0.6,
        a=0.01,
        b=0.005
    )
    
    # 2. 创建PINN模型
    input_dim = 4  # 空间坐标 + 温度
    hidden_dims = [64, 128, 64]
    output_dim = 3  # 速度场 + 压力
    
    pinn = GeologicalPINN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        geological_config=config
    )
    
    # 3. 创建元学习器
    meta_learner = GeodynamicMetaLearner(
        pinn_model=pinn,
        meta_learning_rate=0.001,
        inner_learning_rate=0.005,
        adaptation_steps=3
    )
    
    # 4. 创建元任务
    meta_tasks = meta_learner.create_geodynamic_meta_tasks()
    print(f"   创建了 {len(meta_tasks)} 个元任务:")
    for task in meta_tasks:
        print(f"     - {task.name}: {task.geological_conditions}")
    
    return pinn, meta_learner, meta_tasks


def demo_meta_training():
    """演示元学习训练过程"""
    print("\n🔧 演示元学习训练过程...")
    
    pinn, meta_learner, meta_tasks = demo_meta_learning_basic()
    
    # 开始元学习训练
    print("\n   开始元学习训练...")
    meta_loss_history, adaptation_history = meta_learner.meta_train_geodynamics(
        meta_tasks=meta_tasks,
        meta_epochs=20,  # 减少轮数以快速演示
        task_samples=500  # 减少样本数以快速演示
    )
    
    # 显示训练结果
    print(f"\n   元学习训练完成!")
    print(f"   最终元损失: {meta_loss_history[-1]:.6f}")
    print(f"   损失减少: {meta_loss_history[0] - meta_loss_history[-1]:.6f}")
    
    return pinn, meta_learner, meta_loss_history, adaptation_history


def demo_rapid_adaptation():
    """演示快速适配到新区域"""
    print("\n🔄 演示快速适配到新区域...")
    
    pinn, meta_learner, meta_loss_history, adaptation_history = demo_meta_training()
    
    # 模拟新区域数据（喜马拉雅碰撞带）
    def generate_himalaya_data(num_samples):
        """生成喜马拉雅碰撞带数据"""
        X = torch.randn(num_samples, 4)
        # 喜马拉雅特征：高海拔、低温、复杂变形
        X[:, 3] = 200 + 100 * torch.randn(num_samples)  # 低温
        y = torch.randn(num_samples, 3)
        y[:, 0] = 0.01 + 0.005 * torch.randn(num_samples)  # 小变形
        return X, y
    
    new_region_data = generate_himalaya_data(300)
    
    # 快速适配
    print("\n   快速适配到喜马拉雅区域...")
    adaptation_result = meta_learner.adapt_to_new_region(
        new_region_data=new_region_data,
        adaptation_steps=5
    )
    
    print(f"\n   适配结果:")
    print(f"     - 适配步数: {adaptation_result['adaptation_steps']}")
    print(f"     - 最终数据损失: {adaptation_result['final_data_loss']:.6f}")
    print(f"     - 最终物理损失: {adaptation_result['final_physics_loss']:.6f}")
    print(f"     - 总损失减少: {adaptation_result['total_loss_reduction']:.6f}")
    
    return adaptation_result


def demo_cross_tectonic_domain():
    """演示跨构造域模拟"""
    print("\n🌍 演示跨构造域模拟...")
    
    pinn, meta_learner, meta_loss_history, adaptation_history = demo_meta_training()
    
    # 模拟从安第斯山脉到喜马拉雅的迁移
    print("\n   模拟从安第斯山脉到喜马拉雅的迁移...")
    
    # 安第斯山脉数据（俯冲带）
    def generate_andes_data(num_samples):
        """生成安第斯山脉数据"""
        X = torch.randn(num_samples, 4)
        X[:, 3] = 350 + 120 * torch.randn(num_samples)  # 中等温度
        y = torch.randn(num_samples, 3)
        y[:, 0] = -0.03 + 0.015 * torch.randn(num_samples)  # 压缩
        return X, y
    
    # 喜马拉雅数据（碰撞带）
    def generate_himalaya_data(num_samples):
        """生成喜马拉雅数据"""
        X = torch.randn(num_samples, 4)
        X[:, 3] = 180 + 80 * torch.randn(num_samples)  # 低温
        y = torch.randn(num_samples, 3)
        y[:, 0] = 0.008 + 0.004 * torch.randn(num_samples)  # 小变形
        return X, y
    
    # 测试迁移效果
    andes_data = generate_andes_data(200)
    himalaya_data = generate_himalaya_data(200)
    
    # 在安第斯数据上训练
    print("    在安第斯山脉数据上训练...")
    andes_result = meta_learner.adapt_to_new_region(andes_data, adaptation_steps=3)
    
    # 在喜马拉雅数据上快速适配
    print("    在喜马拉雅数据上快速适配...")
    himalaya_result = meta_learner.adapt_to_new_region(himalaya_data, adaptation_steps=2)
    
    print(f"\n   跨构造域迁移结果:")
    print(f"     - 安第斯训练损失减少: {andes_result['total_loss_reduction']:.6f}")
    print(f"     - 喜马拉雅适配损失减少: {himalaya_result['total_loss_reduction']:.6f}")
    print(f"     - 迁移效率: {himalaya_result['total_loss_reduction'] / andes_result['total_loss_reduction']:.2f}")
    
    return andes_result, himalaya_result


def demo_multi_scale_transfer():
    """演示多尺度迁移"""
    print("\n📏 演示多尺度迁移...")
    
    pinn, meta_learner, meta_loss_history, adaptation_history = demo_meta_training()
    
    # 模拟从区域尺度(100km)到局部断层尺度(10km)的迁移
    print("\n   模拟从区域尺度(100km)到局部断层尺度(10km)的迁移...")
    
    # 区域尺度数据（粗网格）
    def generate_regional_data(num_samples):
        """生成区域尺度数据"""
        X = torch.randn(num_samples, 4)
        X[:, :3] *= 100  # 100km尺度
        X[:, 3] = 400 + 150 * torch.randn(num_samples)
        y = torch.randn(num_samples, 3)
        y[:, 0] = 0.05 + 0.02 * torch.randn(num_samples)
        return X, y
    
    # 局部尺度数据（细网格）
    def generate_local_data(num_samples):
        """生成局部尺度数据"""
        X = torch.randn(num_samples, 4)
        X[:, :3] *= 10   # 10km尺度
        X[:, 3] = 350 + 100 * torch.randn(num_samples)
        y = torch.randn(num_samples, 3)
        y[:, 0] = 0.02 + 0.01 * torch.randn(num_samples)
        return X, y
    
    # 测试多尺度迁移
    regional_data = generate_regional_data(300)
    local_data = generate_local_data(400)
    
    # 在区域尺度上训练
    print("    在区域尺度上训练...")
    regional_result = meta_learner.adapt_to_new_region(regional_data, adaptation_steps=4)
    
    # 在局部尺度上快速适配
    print("    在局部尺度上快速适配...")
    local_result = meta_learner.adapt_to_new_region(local_data, adaptation_steps=2)
    
    print(f"\n   多尺度迁移结果:")
    print(f"     - 区域尺度训练损失减少: {regional_result['total_loss_reduction']:.6f}")
    print(f"     - 局部尺度适配损失减少: {local_result['total_loss_reduction']:.6f}")
    print(f"     - 尺度迁移效率: {local_result['total_loss_reduction'] / regional_result['total_loss_reduction']:.2f}")
    
    return regional_result, local_result


def demo_meta_learning_monitoring():
    """演示元学习监控和分析"""
    print("\n📊 演示元学习监控和分析...")
    
    pinn, meta_learner, meta_loss_history, adaptation_history = demo_meta_training()
    
    # 获取元学习状态
    status = meta_learner.get_meta_learning_status()
    
    print(f"\n   元学习状态:")
    print(f"     - 元学习率: {status['meta_learning_rate']}")
    print(f"     - 内循环学习率: {status['inner_learning_rate']}")
    print(f"     - 内循环步数: {status['adaptation_steps']}")
    print(f"     - 元学习轮数: {len(status['meta_loss_history'])}")
    
    # 分析任务性能
    print(f"\n   任务性能分析:")
    for task_name, performance in status['task_performance'].items():
        if 'epoch_19' in task_name:  # 最后一轮
            print(f"     - {task_name}:")
            print(f"       验证损失: {performance['validation_loss']:.6f}")
            print(f"       最终任务损失: {performance['final_task_loss']:.6f}")
    
    # 可视化元学习过程
    try:
        plt.figure(figsize=(12, 8))
        
        # 元损失历史
        plt.subplot(2, 2, 1)
        plt.plot(status['meta_loss_history'])
        plt.title('元损失历史')
        plt.xlabel('元学习轮次')
        plt.ylabel('元损失')
        plt.grid(True)
        
        # 任务损失对比
        plt.subplot(2, 2, 2)
        task_names = list(set([name.split('_epoch_')[0] for name in status['task_performance'].keys()]))
        final_losses = []
        for task_name in task_names:
            task_perfs = [v for k, v in status['task_performance'].items() if k.startswith(task_name)]
            if task_perfs:
                final_losses.append(task_perfs[-1]['final_task_loss'])
        
        plt.bar(task_names, final_losses)
        plt.title('各任务最终损失')
        plt.ylabel('损失值')
        plt.xticks(rotation=45)
        
        # 验证损失对比
        plt.subplot(2, 2, 3)
        val_losses = []
        for task_name in task_names:
            task_perfs = [v for k, v in status['task_performance'].items() if k.startswith(task_name)]
            if task_perfs:
                val_losses.append(task_perfs[-1]['validation_loss'])
        
        plt.bar(task_names, val_losses)
        plt.title('各任务验证损失')
        plt.ylabel('损失值')
        plt.xticks(rotation=45)
        
        # 损失减少趋势
        plt.subplot(2, 2, 4)
        loss_reductions = []
        for task_name in task_names:
            task_perfs = [v for k, v in status['task_performance'].items() if k.startswith(task_name)]
            if task_perfs:
                first_loss = task_perfs[0]['final_task_loss']
                last_loss = task_perfs[-1]['final_task_loss']
                loss_reductions.append(first_loss - last_loss)
        
        plt.bar(task_names, loss_reductions)
        plt.title('各任务损失减少')
        plt.ylabel('损失减少值')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('meta_learning_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n   可视化图表已保存为: meta_learning_analysis.png")
        
    except Exception as e:
        print(f"   可视化失败: {e}")
    
    return status


def demo_integration_with_existing_pinn():
    """演示与现有PINN的集成"""
    print("\n🔗 演示与现有PINN的集成...")
    
    # 创建PINN模型
    config = GeologicalConfig()
    pinn = GeologicalPINN(
        input_dim=4,
        hidden_dims=[32, 64, 32],
        output_dim=3,
        geological_config=config
    )
    
    # 直接使用PINN的元学习方法
    print("\n   使用PINN内置的元学习方法...")
    
    # 创建元任务
    meta_learner = GeodynamicMetaLearner(pinn)
    meta_tasks = meta_learner.create_geodynamic_meta_tasks()
    
    # 通过PINN进行元学习
    print("    开始元学习训练...")
    meta_loss_history, adaptation_history = pinn.meta_train_geodynamics(
        meta_tasks=meta_tasks,
        meta_epochs=15,
        task_samples=400
    )
    
    print(f"    元学习完成，最终损失: {meta_loss_history[-1]:.6f}")
    
    # 测试快速适配
    print("    测试快速适配到新区域...")
    new_data = (torch.randn(100, 4), torch.randn(100, 3))
    adaptation_result = pinn.adapt_to_new_region(new_data, adaptation_steps=3)
    
    print(f"    适配完成，损失减少: {adaptation_result['total_loss_reduction']:.6f}")
    
    # 获取元学习状态
    status = pinn.get_meta_learning_status()
    print(f"    元学习状态: {len(status['meta_loss_history'])} 轮完成")
    
    return pinn, meta_loss_history, adaptation_result


def main():
    """主演示函数"""
    print("🌍 地球动力学元学习功能演示")
    print("=" * 50)
    
    try:
        # 1. 基本元学习功能
        demo_meta_learning_basic()
        
        # 2. 元学习训练过程
        demo_meta_training()
        
        # 3. 快速适配到新区域
        demo_rapid_adaptation()
        
        # 4. 跨构造域模拟
        demo_cross_tectonic_domain()
        
        # 5. 多尺度迁移
        demo_multi_scale_transfer()
        
        # 6. 元学习监控和分析
        demo_meta_learning_monitoring()
        
        # 7. 与现有PINN的集成
        demo_integration_with_existing_pinn()
        
        print("\n✅ 所有元学习演示完成!")
        print("\n📋 功能总结:")
        print("   - 支持多种地球动力学构造场景")
        print("   - 快速适配到新区域（仅需少量数据）")
        print("   - 跨构造域模拟能力")
        print("   - 多尺度迁移支持")
        print("   - 完整的监控和分析工具")
        print("   - 与现有PINN无缝集成")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
