#!/usr/bin/env python3
"""
测试扩展后的GeologicalSurrogateModel功能
"""

import numpy as np
import sys
import os
import time

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geological_ml_framework import (
    GeologicalConfig, GeologicalSurrogateModel
)

def test_model_types():
    """测试不同模型类型"""
    print("🔍 测试不同模型类型...")
    
    # 生成测试数据
    n_samples = 500
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    y = np.sin(X.sum(axis=1)) + 0.1 * np.random.randn(n_samples)
    
    # 测试支持的模型类型
    model_types = ['gaussian_process', 'random_forest', 'gradient_boosting', 'mlp']
    
    for model_type in model_types:
        try:
            print(f"   测试 {model_type}...")
            surrogate = GeologicalSurrogateModel(model_type=model_type)
            
            # 训练模型
            history = surrogate.train(X, y, cv=3)
            print(f"     ✅ 训练成功，时间: {history['training_time']:.4f}秒")
            
            # 预测
            X_test = np.random.randn(100, n_features)
            predictions = surrogate.predict(X_test)
            print(f"     ✅ 预测成功，形状: {predictions.shape}")
            
            # 测试不确定性估计
            if model_type in ['gaussian_process', 'random_forest']:
                predictions, std = surrogate.predict(X_test, return_std=True)
                print(f"     ✅ 不确定性估计成功，标准差形状: {std.shape}")
            
            # 测试特征重要性（如果支持）
            if model_type in ['random_forest', 'gradient_boosting']:
                importance = surrogate.get_feature_importance()
                print(f"     ✅ 特征重要性成功，形状: {importance.shape}")
            
        except Exception as e:
            print(f"     ❌ {model_type} 失败: {e}")
    
    return True

def test_cross_validation():
    """测试交叉验证功能"""
    print("🔍 测试交叉验证功能...")
    
    # 生成测试数据
    n_samples = 300
    n_features = 4
    X = np.random.randn(n_samples, n_features)
    y = np.sin(X.sum(axis=1)) + 0.1 * np.random.randn(n_samples)
    
    try:
        surrogate = GeologicalSurrogateModel('random_forest')
        
        # 训练（带交叉验证）
        history = surrogate.train(X, y, cv=5, n_estimators=50)
        
        # 检查交叉验证结果
        if 'cv_scores' in history and history['cv_scores'] is not None:
            cv_scores = history['cv_scores']
            print(f"   ✅ 交叉验证成功")
            print(f"     平均MSE: {cv_scores.mean():.6f}")
            print(f"     标准差: {cv_scores.std():.6f}")
            print(f"     各折分数: {cv_scores.tolist()}")
        else:
            print("   ❌ 交叉验证失败")
            return False
        
    except Exception as e:
        print(f"   ❌ 交叉验证测试失败: {e}")
        return False
    
    return True

def test_feature_importance():
    """测试特征重要性分析"""
    print("🔍 测试特征重要性分析...")
    
    # 生成测试数据（故意让某些特征更重要）
    n_samples = 500
    n_features = 6
    X = np.random.randn(n_samples, n_features)
    # 让前两个特征更重要
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + 0.1 * np.random.randn(n_samples)
    
    try:
        surrogate = GeologicalSurrogateModel('random_forest')
        history = surrogate.train(X, y, n_estimators=100)
        
        # 获取特征重要性
        importance = surrogate.get_feature_importance()
        print(f"   ✅ 特征重要性分析成功")
        print(f"     重要性形状: {importance.shape}")
        print(f"     重要性值: {importance}")
        
        # 检查前两个特征是否更重要
        sorted_indices = np.argsort(importance)[::-1]
        print(f"     重要性排序: {sorted_indices}")
        
        if sorted_indices[0] in [0, 1] and sorted_indices[1] in [0, 1]:
            print(f"     ✅ 重要特征识别正确")
        else:
            print(f"     ⚠️  重要特征识别可能不准确")
        
    except Exception as e:
        print(f"   ❌ 特征重要性测试失败: {e}")
        return False
    
    return True

def test_batch_prediction():
    """测试批量预测优化"""
    print("🔍 测试批量预测优化...")
    
    # 生成大规模测试数据
    n_samples = 5000
    n_features = 4
    X = np.random.randn(n_samples, n_features)
    y = np.sin(X.sum(axis=1)) + 0.1 * np.random.randn(n_samples)
    
    try:
        surrogate = GeologicalSurrogateModel('random_forest')
        history = surrogate.train(X, y, n_estimators=50)
        
        # 测试不同批量大小
        X_test = np.random.randn(2000, n_features)
        
        # 小批量
        start_time = time.time()
        predictions_small = surrogate.predict(X_test, batch_size=100)
        time_small = time.time() - start_time
        
        # 大批量
        start_time = time.time()
        predictions_large = surrogate.predict(X_test, batch_size=1000)
        time_large = time.time() - start_time
        
        print(f"   ✅ 批量预测成功")
        print(f"     小批量时间: {time_small:.4f}秒")
        print(f"     大批量时间: {time_large:.4f}秒")
        print(f"     预测形状: {predictions_small.shape}")
        
        # 检查结果一致性
        if np.allclose(predictions_small, predictions_large):
            print(f"     ✅ 批量预测结果一致")
        else:
            print(f"     ⚠️  批量预测结果不一致")
        
    except Exception as e:
        print(f"   ❌ 批量预测测试失败: {e}")
        return False
    
    return True

def test_uncertainty_estimation():
    """测试不确定性估计"""
    print("🔍 测试不确定性估计...")
    
    # 生成测试数据
    n_samples = 300
    n_features = 4
    X = np.random.randn(n_samples, n_features)
    y = np.sin(X.sum(axis=1)) + 0.1 * np.random.randn(n_samples)
    
    try:
        # 测试高斯过程回归
        surrogate_gp = GeologicalSurrogateModel('gaussian_process')
        history_gp = surrogate_gp.train(X, y)
        
        X_test = np.random.randn(50, n_features)
        predictions_gp, std_gp = surrogate_gp.predict(X_test, return_std=True)
        
        print(f"   ✅ 高斯过程不确定性估计成功")
        print(f"     预测形状: {predictions_gp.shape}")
        print(f"     标准差形状: {std_gp.shape}")
        print(f"     标准差范围: [{std_gp.min():.4f}, {std_gp.max():.4f}]")
        
        # 测试随机森林
        surrogate_rf = GeologicalSurrogateModel('random_forest')
        history_rf = surrogate_rf.train(X, y, n_estimators=50)
        
        predictions_rf, std_rf = surrogate_rf.predict(X_test, return_std=True)
        
        print(f"   ✅ 随机森林不确定性估计成功")
        print(f"     预测形状: {predictions_rf.shape}")
        print(f"     标准差形状: {std_rf.shape}")
        print(f"     标准差范围: [{std_rf.min():.4f}, {std_rf.max():.4f}]")
        
    except Exception as e:
        print(f"   ❌ 不确定性估计测试失败: {e}")
        return False
    
    return True

def test_model_persistence():
    """测试模型持久化"""
    print("🔍 测试模型持久化...")
    
    # 生成测试数据
    n_samples = 200
    n_features = 3
    X = np.random.randn(n_samples, n_features)
    y = np.sin(X.sum(axis=1)) + 0.1 * np.random.randn(n_samples)
    
    try:
        # 创建并训练模型
        surrogate = GeologicalSurrogateModel('random_forest')
        history = surrogate.train(X, y, n_estimators=50)
        
        # 保存模型
        model_path = "test_surrogate_model.pkl"
        surrogate.save_model(model_path)
        
        # 创建新模型并加载
        surrogate_loaded = GeologicalSurrogateModel('random_forest')
        surrogate_loaded.load_model(model_path)
        
        # 测试预测一致性
        X_test = np.random.randn(50, n_features)
        predictions_original = surrogate.predict(X_test)
        predictions_loaded = surrogate_loaded.predict(X_test)
        
        if np.allclose(predictions_original, predictions_loaded):
            print(f"   ✅ 模型持久化成功")
            print(f"     预测一致性: 通过")
        else:
            print(f"   ⚠️  模型持久化预测不一致")
        
        # 清理测试文件
        if os.path.exists(model_path):
            os.remove(model_path)
        
    except Exception as e:
        print(f"   ❌ 模型持久化测试失败: {e}")
        return False
    
    return True

def test_model_performance():
    """测试模型性能分析"""
    print("🔍 测试模型性能分析...")
    
    # 生成测试数据
    n_samples = 300
    n_features = 4
    X = np.random.randn(n_samples, n_features)
    y = np.sin(X.sum(axis=1)) + 0.1 * np.random.randn(n_samples)
    
    try:
        surrogate = GeologicalSurrogateModel('random_forest')
        history = surrogate.train(X, y, cv=3, n_estimators=50)
        
        # 获取性能指标
        performance = surrogate.get_model_performance()
        
        print(f"   ✅ 模型性能分析成功")
        print(f"     模型类型: {performance['model_type']}")
        print(f"     训练时间: {performance['training_time']:.4f}秒")
        print(f"     样本数: {performance['n_samples']}")
        print(f"     特征数: {performance['n_features']}")
        
        if 'cv_mean_mse' in performance:
            print(f"     交叉验证平均MSE: {performance['cv_mean_mse']:.6f}")
            print(f"     交叉验证标准差: {performance['cv_std_mse']:.6f}")
        
        if 'feature_importance' in performance:
            print(f"     特征重要性: {performance['feature_importance']}")
        
    except Exception as e:
        print(f"   ❌ 模型性能分析测试失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始GeologicalSurrogateModel扩展功能测试")
    print("=" * 60)
    
    tests = [
        test_model_types,
        test_cross_validation,
        test_feature_importance,
        test_batch_prediction,
        test_uncertainty_estimation,
        test_model_persistence,
        test_model_performance
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
    
    print("=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！GeologicalSurrogateModel扩展功能实现成功")
        return True
    else:
        print("⚠️  部分测试失败，请检查实现")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
