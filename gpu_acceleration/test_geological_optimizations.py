#!/usr/bin/env python3
"""
测试地质场景优化实现

"""

import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 检查PyTorch可用性
try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    torch = None

from geological_ml_framework import (
    GeologicalConfig, BaseSolver, GeologicalPhysicsEquations,
    GeologicalPINN, GeologicalSurrogateModel, GeologicalUNet,
    GeologicalHybridAccelerator, GeologicalAdaptiveSolver,
    create_geological_ml_system
)

def test_base_solver_inheritance():
    """测试BaseSolver继承"""
    print("🔍 测试BaseSolver继承...")
    
    # 测试GeologicalPINN继承
    try:
        pinn = GeologicalPINN(4, [64, 32], 1)
        assert hasattr(pinn, 'device')
        assert hasattr(pinn, 'is_trained')
        assert hasattr(pinn, 'save_model')
        assert hasattr(pinn, 'load_model')
        print("   ✅ GeologicalPINN继承BaseSolver成功")
    except Exception as e:
        print(f"   ❌ GeologicalPINN继承失败: {e}")
        return False
    
    # 测试GeologicalSurrogateModel继承
    try:
        surrogate = GeologicalSurrogateModel('gaussian_process')
        assert hasattr(surrogate, 'device')
        assert hasattr(surrogate, 'is_trained')
        assert hasattr(surrogate, 'save_model')
        assert hasattr(surrogate, 'load_model')
        print("   ✅ GeologicalSurrogateModel继承BaseSolver成功")
    except Exception as e:
        print(f"   ❌ GeologicalSurrogateModel继承失败: {e}")
        return False
    
    return True

def test_geological_features():
    """测试地质特征支持"""
    print("🔍 测试地质特征支持...")
    
    # 生成测试数据
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    y = np.random.randn(n_samples, 1)
    geological_features = np.random.rand(n_samples, 3)  # 孔隙度、渗透率、粘度
    
    # 测试GeologicalPINN地质特征
    try:
        pinn = GeologicalPINN(4, [32, 16], 1)
        pinn.setup_training()  # 设置训练参数
        # 修复：确保参数正确传递
        result = pinn.train(X=X, y=y, geological_features=geological_features, epochs=10)
        assert result['train_time'] > 0
        print("   ✅ GeologicalPINN地质特征支持成功")
    except Exception as e:
        print(f"   ❌ GeologicalPINN地质特征失败: {e}")
        return False
    
    # 测试GeologicalSurrogateModel地质特征
    try:
        surrogate = GeologicalSurrogateModel('random_forest')
        result = surrogate.train(X, y.flatten(), geological_features=geological_features)
        assert result['training_time'] > 0
        print("   ✅ GeologicalSurrogateModel地质特征支持成功")
    except Exception as e:
        print(f"   ❌ GeologicalSurrogateModel地质特征失败: {e}")
        return False
    
    return True

def test_geological_physics_constraints():
    """测试地质物理约束增强"""
    print("🔍 测试地质物理约束增强...")
    
    if not HAS_PYTORCH:
        print("   ⚠️  PyTorch不可用，跳过物理约束测试")
        return True
    
    try:
        config = GeologicalConfig(
            porosity=0.25,
            permeability=1e-12,
            viscosity=1e-3,
            use_gpu=False  # 测试时禁用GPU
        )
        
        # 测试达西方程 - 修复张量维度
        x = torch.randn(10, 3, requires_grad=True)
        y = torch.randn(10, 1, requires_grad=True)
        
        # 确保y的维度正确
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        
        residual = GeologicalPhysicsEquations.darcy_equation(x, y, config)
        assert isinstance(residual, torch.Tensor)
        print("   ✅ 达西方程约束成功")
        
        # 测试热传导方程
        residual = GeologicalPhysicsEquations.heat_conduction_equation(x, y, config)
        assert isinstance(residual, torch.Tensor)
        print("   ✅ 热传导方程约束成功")
        
    except Exception as e:
        print(f"   ❌ 地质物理约束失败: {e}")
        return False
    
    return True

def test_hybrid_accelerator():
    """测试混合加速器增强功能"""
    print("🔍 测试混合加速器增强功能...")
    
    try:
        # 创建传统求解器模拟
        def traditional_solver(data):
            return np.random.randn(len(data['input']))
        
        hybrid = GeologicalHybridAccelerator(traditional_solver)
        
        # 添加ML模型
        surrogate = GeologicalSurrogateModel('random_forest')
        surrogate.train(np.random.randn(100, 4), np.random.randn(100))
        hybrid.add_ml_model('surrogate', surrogate)
        
        # 测试阶段加速策略
        hybrid.setup_stage_strategy('solver', 'surrogate')
        assert hybrid.stage_strategies['solver'] == 'surrogate'
        print("   ✅ 阶段加速策略设置成功")
        
        # 测试动态切换策略
        problem_data = {
            'input': np.random.randn(10, 4),
            'accuracy_requirement': 1e-6,  # 高精度要求
            'stage': 'solver'
        }
        result = hybrid.solve_hybrid(problem_data, use_ml=True, ml_model_name='surrogate')
        assert 'solution' in result
        print("   ✅ 动态切换策略成功")
        
    except Exception as e:
        print(f"   ❌ 混合加速器测试失败: {e}")
        return False
    
    return True

def test_adaptive_solver():
    """测试地质自适应求解器"""
    print("🔍 测试地质自适应求解器...")
    
    try:
        adaptive = GeologicalAdaptiveSolver()
        
        # 定义测试求解器
        def fast_solver(data):
            return np.random.randn(len(data['input']))
        
        def accurate_solver(data):
            return np.random.randn(len(data['input'])) * 0.1
        
        def fault_solver(data):
            return np.random.randn(len(data['input'])) * 0.05
        
        # 添加求解器
        adaptive.add_solver('fast', fast_solver, conditions={'size': ('<', 1000)})
        adaptive.add_solver('accurate', accurate_solver, conditions={'accuracy_requirement': ('>', 0.9)})
        adaptive.add_solver('fault', fault_solver, conditions={'has_faults': True})
        
        # 测试不同场景
        scenarios = [
            {'input': np.random.randn(100, 4), 'size': 100, 'name': '小规模问题'},
            {'input': np.random.randn(100, 4), 'accuracy_requirement': 0.95, 'name': '高精度要求'},
            {'input': np.random.randn(100, 4), 'has_faults': True, 'porosity': 0.3, 'name': '复杂地质条件'}
        ]
        
        for scenario in scenarios:
            result = adaptive.solve(scenario)
            assert 'solution' in result
            assert 'solver_name' in result
            print(f"   ✅ {scenario['name']} 求解成功: {result['solver_name']}")
        
    except Exception as e:
        print(f"   ❌ 地质自适应求解器测试失败: {e}")
        return False
    
    return True

def test_batch_processing():
    """测试批量处理和GPU加速"""
    print("🔍 测试批量处理和GPU加速...")
    
    try:
        # 创建大尺寸测试数据
        n_samples = 5000
        X = np.random.randn(n_samples, 4)
        y = np.random.randn(n_samples, 1)
        
        # 测试批量预测
        surrogate = GeologicalSurrogateModel('random_forest')
        surrogate.train(X, y.flatten())
        
        # 测试批量预测
        test_X = np.random.randn(2000, 4)
        predictions = surrogate.predict(test_X)
        assert predictions.shape[0] == 2000
        print("   ✅ 批量处理成功")
        
    except Exception as e:
        print(f"   ❌ 批量处理测试失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始地质场景优化测试")
    print("=" * 50)
    
    tests = [
        test_base_solver_inheritance,
        test_geological_features,
        test_geological_physics_constraints,
        test_hybrid_accelerator,
        test_adaptive_solver,
        test_batch_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"   ❌ 测试异常: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！地质场景优化实现成功")
        return True
    else:
        print("⚠️  部分测试失败，请检查实现")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
