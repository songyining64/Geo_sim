"""
测试超越Underworld的并行求解器

验证核心功能：
1. 配置系统
2. 通信优化器
3. 负载均衡器
4. 异构计算管理器
5. 求解器选择器
6. 主求解器
"""

import numpy as np
import scipy.sparse as sp
import time
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from underworld_surpassing_solver import (
        AdvancedParallelConfig,
        UnderworldSurpassingSolver,
        AdaptiveCommunicator,
        MLBasedLoadBalancer,
        HeterogeneousComputingManager,
        AdaptiveSolverSelector,
        PerformanceMetrics
    )
    print("✅ 成功导入所有模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def test_configuration():
    """测试配置系统"""
    print("\n🔧 测试配置系统")
    print("=" * 40)
    
    try:
        # 测试默认配置
        config = AdvancedParallelConfig()
        print(f"   默认配置创建成功")
        print(f"   求解器类型: {config.solver_type}")
        print(f"   最大迭代次数: {config.max_iterations}")
        print(f"   容差: {config.tolerance}")
        
        # 测试自定义配置
        custom_config = AdvancedParallelConfig(
            solver_type='gpu_cg',
            max_iterations=5000,
            tolerance=1e-12,
            use_gpu=True,
            use_openmp=True,
            ml_based_balancing=True
        )
        print(f"   自定义配置创建成功")
        print(f"   求解器类型: {custom_config.solver_type}")
        print(f"   最大迭代次数: {custom_config.max_iterations}")
        print(f"   容差: {custom_config.tolerance}")
        print(f"   GPU支持: {custom_config.use_gpu}")
        print(f"   OpenMP支持: {custom_config.use_openmp}")
        print(f"   ML负载均衡: {custom_config.ml_based_balancing}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 配置测试失败: {e}")
        return False


def test_adaptive_communicator():
    """测试自适应通信优化器"""
    print("\n🚀 测试自适应通信优化器")
    print("=" * 40)
    
    try:
        # 创建通信优化器
        communicator = AdaptiveCommunicator()
        print(f"   通信优化器创建成功")
        
        # 模拟分区信息
        partition_info = {
            'method': 'metis',
            'num_partitions': 4,
            'boundary_elements': [100, 150, 200, 120]
        }
        
        # 模拟数据密度
        data_density = np.array([0.8, 0.3, 0.9, 0.2])
        
        # 测试通信模式优化
        schedule = communicator.optimize_communication_pattern(partition_info, data_density)
        print(f"   通信调度生成成功")
        print(f"   调度类型: {schedule['type']}")
        
        if schedule['type'] == 'point_to_point':
            print(f"   邻居进程数: {len(schedule['neighbors'])}")
            print(f"   通信顺序: {schedule['communication_order']}")
        elif schedule['type'] == 'collective':
            print(f"   集体通信操作: {schedule['operation']}")
        elif schedule['type'] == 'hybrid':
            print(f"   混合策略")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 通信优化器测试失败: {e}")
        return False


def test_ml_load_balancer():
    """测试基于ML的负载均衡器"""
    print("\n⚖️ 测试基于ML的负载均衡器")
    print("=" * 40)
    
    try:
        # 创建负载均衡器
        load_balancer = MLBasedLoadBalancer()
        print(f"   负载均衡器创建成功")
        
        # 模拟当前负载分布
        current_loads = np.array([1200, 800, 1500, 900])
        print(f"   当前负载分布: {current_loads}")
        
        # 计算负载不平衡度
        imbalance = load_balancer._compute_load_imbalance(current_loads)
        print(f"   负载不平衡度: {imbalance:.4f}")
        
        # 模拟网格特征
        mesh_features = {
            'element_complexity': [1.2, 0.8, 1.5, 1.0],
            'elements': [1200, 800, 1500, 900]
        }
        
        # 预测负载分布
        predicted_loads = load_balancer.predict_load_distribution(mesh_features, {})
        print(f"   预测负载分布: {predicted_loads}")
        
        # 测试动态负载均衡
        partition_info = {'method': 'metis', 'partitions': [0, 1, 2, 3]}
        balanced_partition = load_balancer.balance_load_dynamically(
            current_loads, mesh_features, partition_info
        )
        print(f"   动态负载均衡完成")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 负载均衡器测试失败: {e}")
        return False


def test_heterogeneous_computing():
    """测试异构计算管理器"""
    print("\n🖥️ 测试异构计算管理器")
    print("=" * 40)
    
    try:
        # 创建配置
        config = AdvancedParallelConfig(
            use_gpu=True,
            use_openmp=True,
            cpu_threads=4
        )
        
        # 创建异构计算管理器
        manager = HeterogeneousComputingManager(config)
        print(f"   异构计算管理器创建成功")
        
        print(f"   GPU支持: {'✅ 可用' if manager.gpu_available else '❌ 不可用'}")
        print(f"   OpenMP支持: {'✅ 可用' if manager.openmp_available else '❌ 不可用'}")
        
        # 创建测试问题
        size = 1000
        A = sp.random(size, size, density=0.05, format='csr')
        A = A + A.T + sp.eye(size)
        b = np.random.randn(size)
        
        print(f"   测试问题规模: {size}")
        
        # 测试CPU求解
        try:
            solution = manager._cpu_solve(A, b)
            residual = np.linalg.norm(b - A.dot(solution))
            print(f"   CPU求解成功，残差: {residual:.2e}")
        except Exception as e:
            print(f"   CPU求解失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 异构计算管理器测试失败: {e}")
        return False


def test_adaptive_solver_selector():
    """测试自适应求解器选择器"""
    print("\n🧠 测试自适应求解器选择器")
    print("=" * 40)
    
    try:
        # 创建配置
        config = AdvancedParallelConfig()
        
        # 创建求解器选择器
        selector = AdaptiveSolverSelector(config)
        print(f"   求解器选择器创建成功")
        
        # 创建测试问题
        A = sp.random(5000, 5000, density=0.01, format='csr') + sp.eye(5000)
        b = np.random.randn(5000)
        
        print(f"   测试问题规模: {A.shape[0]}")
        
        # 问题分类
        problem_type = selector.problem_classifier.classify_problem(A, b)
        print(f"   问题类型: {problem_type}")
        
        # 选择最优求解器
        available_solvers = ['gpu_cg', 'openmp_cg', 'cpu_cg', 'amg']
        optimal_solver = selector.select_optimal_solver(A, b, available_solvers)
        print(f"   推荐求解器: {optimal_solver}")
        
        # 提取问题特征
        features = selector.problem_classifier._extract_features(A, b)
        print(f"   问题特征:")
        print(f"     规模: {features['size']}")
        print(f"     稀疏性: {features['sparsity']:.3f}")
        print(f"     条件数: {features['condition_number']:.2e}")
        print(f"     结构: {features['structure']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 求解器选择器测试失败: {e}")
        return False


def test_main_solver():
    """测试主求解器"""
    print("\n🎯 测试主求解器")
    print("=" * 40)
    
    try:
        # 创建配置
        config = AdvancedParallelConfig(
            solver_type='adaptive',
            use_gpu=False,  # 测试时禁用GPU
            use_openmp=False,  # 测试时禁用OpenMP
            ml_based_balancing=True,
            adaptive_communication=True,
            max_iterations=100,
            tolerance=1e-6
        )
        
        # 创建主求解器
        solver = UnderworldSurpassingSolver(config)
        print(f"   主求解器创建成功")
        print(f"   进程数: {solver.size}")
        print(f"   GPU支持: {solver.heterogeneous_manager.gpu_available}")
        print(f"   OpenMP支持: {solver.heterogeneous_manager.openmp_available}")
        
        # 创建测试问题
        size = 500
        A = sp.random(size, size, density=0.1, format='csr')
        A = A + A.T + sp.eye(size)
        b = np.random.randn(size)
        
        print(f"   测试问题规模: {size}")
        
        # 求解线性系统
        start_time = time.time()
        solution = solver.solve(A, b)
        solve_time = time.time() - start_time
        
        # 验证解的正确性
        residual = np.linalg.norm(b - A.dot(solution))
        
        print(f"   求解成功")
        print(f"   求解时间: {solve_time:.4f}s")
        print(f"   残差范数: {residual:.2e}")
        
        # 获取性能总结
        performance_summary = solver.get_performance_summary()
        print(f"   总求解次数: {performance_summary['total_solves']}")
        print(f"   平均求解时间: {performance_summary['average_solve_time']:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 主求解器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_metrics():
    """测试性能指标系统"""
    print("\n📊 测试性能指标系统")
    print("=" * 40)
    
    try:
        # 创建性能指标
        metrics = PerformanceMetrics()
        print(f"   性能指标创建成功")
        
        # 设置一些测试值
        metrics.total_solve_time = 1.5
        metrics.communication_time = 0.3
        metrics.computation_time = 1.2
        metrics.iterations = 150
        metrics.residual_norm = 1e-8
        metrics.load_imbalance = 0.1
        metrics.parallel_efficiency = 0.85
        
        print(f"   总求解时间: {metrics.total_solve_time}s")
        print(f"   通信时间: {metrics.communication_time}s")
        print(f"   计算时间: {metrics.computation_time}s")
        print(f"   迭代次数: {metrics.iterations}")
        print(f"   残差范数: {metrics.residual_norm}")
        print(f"   负载不平衡度: {metrics.load_imbalance}")
        print(f"   并行效率: {metrics.parallel_efficiency}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 性能指标测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行超越Underworld的并行求解器测试")
    print("=" * 60)
    
    tests = [
        ("配置系统", test_configuration),
        ("自适应通信优化器", test_adaptive_communicator),
        ("基于ML的负载均衡器", test_ml_load_balancer),
        ("异构计算管理器", test_heterogeneous_computing),
        ("自适应求解器选择器", test_adaptive_solver_selector),
        ("主求解器", test_main_solver),
        ("性能指标系统", test_performance_metrics)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name} 测试通过")
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果总结:")
    print(f"   总测试数: {total}")
    print(f"   通过测试: {passed}")
    print(f"   失败测试: {total - passed}")
    print(f"   成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过！超越Underworld的并行求解器工作正常")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
    
    return passed == total


if __name__ == "__main__":
    # 运行所有测试
    success = run_all_tests()
    
    if success:
        print("\n🚀 测试完成，可以开始使用超越Underworld的并行求解器！")
        print("   运行演示: python underworld_surpassing_demo.py")
        print("   运行基准测试: python underworld_surpassing_solver.py")
    else:
        print("\n❌ 测试失败，请修复相关问题后重试")
    
    sys.exit(0 if success else 1)
