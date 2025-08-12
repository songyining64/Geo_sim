"""
集成学习模块

提供多种集成学习方法，包括模型集成、多保真度建模等。
"""

from .multi_fidelity import (
    # 配置类
    FidelityLevel,
    MultiFidelityConfig,
    
    # 模型类
    BaseFidelityModel,
    NeuralNetworkFidelityModel,
    TraditionalMLFidelityModel,
    
    # 训练器
    MultiFidelityTrainer,
    
    # 工厂函数
    create_multi_fidelity_system,
    
    # 演示函数
    demo_multi_fidelity
)

# 版本信息
__version__ = "1.0.0"
__author__ = "GeoSim Team"

# 导出所有公共接口
__all__ = [
    # 多保真度建模
    'FidelityLevel',
    'MultiFidelityConfig',
    'BaseFidelityModel',
    'NeuralNetworkFidelityModel',
    'TraditionalMLFidelityModel',
    'MultiFidelityTrainer',
    'create_multi_fidelity_system',
    'demo_multi_fidelity'
]

# 便捷函数
def quick_multi_fidelity_setup():
    """快速设置多保真度建模系统"""
    from .multi_fidelity import FidelityLevel, MultiFidelityConfig
    
    # 创建默认配置
    low_fidelity = FidelityLevel(
        name="快速仿真",
        level=0,
        description="使用简化物理模型的快速仿真",
        computational_cost=1.0,
        accuracy=0.8,
        data_requirements=1000,
        training_time=120.0,
        model_type="neural_network"
    )
    
    high_fidelity = FidelityLevel(
        name="精确仿真",
        level=1,
        description="使用完整物理模型的精确仿真",
        computational_cost=5.0,
        accuracy=0.95,
        data_requirements=3000,
        training_time=600.0,
        model_type="neural_network"
    )
    
    config = MultiFidelityConfig(
        name="默认多保真度系统",
        description="快速设置的多保真度建模系统",
        fidelity_levels=[low_fidelity, high_fidelity]
    )
    
    return create_multi_fidelity_system(config)


def demo_ensemble_features():
    """演示集成学习功能"""
    print("=== 集成学习功能演示 ===")
    
    # 演示多保真度建模
    try:
        demo_multi_fidelity()
        print("✅ 多保真度建模演示完成")
    except Exception as e:
        print(f"❌ 多保真度建模演示失败: {e}")
    
    print("\n🎉 集成学习功能演示完成！")


if __name__ == "__main__":
    demo_ensemble_features()
