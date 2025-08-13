"""
超越Underworld的并行求解器演示脚本

展示核心突破功能：
1. 通信优化
2. 智能负载均衡
3. 异构计算
4. 自适应算法选择
5. 性能基准测试
"""

import numpy as np
import scipy.sparse as sp
import time
import json
from pathlib import Path

# 导入我们的高级并行求解器
from underworld_surpassing_solver import (
    AdvancedParallelConfig, 
    UnderworldSurpassingSolver,
    AdaptiveCommunicator,
    MLBasedLoadBalancer,
    HeterogeneousComputingManager,
    AdaptiveSolverSelector
)


def create_test_problems():
    """创建不同特征的测试问题"""
    print("🔧 创建测试问题...")
    
    test_problems = {}
    
    # 1. 大规模稀疏问题（适合GPU）
    print("   创建大规模稀疏问题...")
    size = 50000
    A_sparse = sp.random(size, size, density=0.005, format='csr')
    A_sparse = A_sparse + A_sparse.T + sp.eye(size) * 2  # 确保正定性
    b_sparse = np.random.randn(size)
    test_problems['large_sparse'] = (A_sparse, b_sparse, '适合GPU加速的大规模稀疏问题')
    
    # 2. 中等规模问题（适合OpenMP）
    print("   创建中等规模问题...")
    size = 10000
    A_medium = sp.random(size, size, density=0.02, format='csr')
    A_medium = A_medium + A_medium.T + sp.eye(size) * 1.5
    b_medium = np.random.randn(size)
    test_problems['medium'] = (A_medium, b_medium, '适合OpenMP并行的中等规模问题')
    
    # 3. 小规模密集问题（适合CPU）
    print("   创建小规模问题...")
    size = 2000
    A_small = sp.random(size, size, density=0.1, format='csr')
    A_small = A_small + A_small.T + sp.eye(size)
    b_small = np.random.randn(size)
    test_problems['small'] = (A_small, b_small, '适合CPU直接求解的小规模问题')
    
    # 4. 病态条件问题
    print("   创建病态条件问题...")
    size = 5000
    A_ill = sp.random(size, size, density=0.01, format='csr')
    A_ill = A_ill + A_ill.T + sp.eye(size) * 0.1  # 接近奇异的矩阵
    b_ill = np.random.randn(size)
    test_problems['ill_conditioned'] = (A_ill, b_ill, '病态条件问题，需要特殊处理')
    
    # 5. 块对角结构问题
    print("   创建块对角结构问题...")
    size = 8000
    block_size = 1000
    A_block = sp.lil_matrix((size, size))
    
    for i in range(0, size, block_size):
        end_i = min(i + block_size, size)
        A_block[i:end_i, i:end_i] = sp.random(end_i - i, end_i - i, density=0.05) + sp.eye(end_i - i)
    
    A_block = A_block.tocsr()
    b_block = np.random.randn(size)
    test_problems['block_structured'] = (A_block, b_block, '块对角结构问题，适合特殊优化')
    
    print(f"✅ 成功创建 {len(test_problems)} 个测试问题")
    return test_problems


def demonstrate_communication_optimization():
    """演示通信优化功能"""
    print("\n🚀 演示通信优化功能")
    print("=" * 50)
    
    # 创建通信优化器
    communicator = AdaptiveCommunicator()
    
    # 模拟分区信息
    partition_info = {
        'method': 'metis',
        'num_partitions': 4,
        'boundary_elements': [100, 150, 200, 120]
    }
    
    # 模拟数据密度
    data_density = np.array([0.8, 0.3, 0.9, 0.2])
    
    print("📊 分析通信模式...")
    schedule = communicator.optimize_communication_pattern(partition_info, data_density)
    
    print(f"   通信调度类型: {schedule['type']}")
    if schedule['type'] == 'point_to_point':
        print(f"   邻居进程数: {len(schedule['neighbors'])}")
        print(f"   通信顺序: {schedule['communication_order']}")
    elif schedule['type'] == 'collective':
        print(f"   集体通信操作: {schedule['operation']}")
        print(f"   优化策略: {schedule['optimization']}")
    elif schedule['type'] == 'hybrid':
        print(f"   混合策略: 点对点 + 集体通信")
    
    return schedule


def demonstrate_load_balancing():
    """演示智能负载均衡功能"""
    print("\n⚖️ 演示智能负载均衡功能")
    print("=" * 50)
    
    # 创建负载均衡器
    load_balancer = MLBasedLoadBalancer()
    
    # 模拟当前负载分布
    current_loads = np.array([1200, 800, 1500, 900])
    print(f"📊 当前负载分布: {current_loads}")
    
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
    print(f"🔮 预测负载分布: {predicted_loads}")
    
    # 动态负载均衡
    partition_info = {'method': 'metis', 'partitions': [0, 1, 2, 3]}
    balanced_partition = load_balancer.balance_load_dynamically(
        current_loads, mesh_features, partition_info
    )
    
    print(f"✅ 负载均衡完成")
    return balanced_partition


def demonstrate_heterogeneous_computing():
    """演示异构计算功能"""
    print("\n🖥️ 演示异构计算功能")
    print("=" * 50)
    
    # 创建异构计算管理器
    config = AdvancedParallelConfig(
        use_gpu=True,
        use_openmp=True,
        cpu_threads=4
    )
    
    manager = HeterogeneousComputingManager(config)
    
    print(f"   硬件支持状态:")
    print(f"     GPU: {'✅ 可用' if manager.gpu_available else '❌ 不可用'}")
    print(f"   OpenMP: {'✅ 可用' if manager.openmp_available else '❌ 不可用'}")
    
    # 创建测试问题
    size = 3000
    A = sp.random(size, size, density=0.02, format='csr')
    A = A + A.T + sp.eye(size)
    b = np.random.randn(size)
    
    print(f"\n🔍 测试问题规模: {size}")
    
    # 测试不同求解器
    solvers = ['auto', 'gpu_cg', 'openmp_cg', 'cpu_cg']
    
    for solver in solvers:
        print(f"\n   测试求解器: {solver}")
        start_time = time.time()
        
        try:
            solution = manager.solve_with_heterogeneous_computing(A, b, solver)
            solve_time = time.time() - start_time
            
            # 验证解的正确性
            residual = np.linalg.norm(b - A.dot(solution))
            print(f"     求解时间: {solve_time:.4f}s")
            print(f"     残差范数: {residual:.2e}")
            
        except Exception as e:
            print(f"     ❌ 求解失败: {e}")
    
    return manager


def demonstrate_adaptive_solver_selection():
    """演示自适应求解器选择功能"""
    print("\n🧠 演示自适应求解器选择功能")
    print("=" * 50)
    
    # 创建求解器选择器
    config = AdvancedParallelConfig()
    selector = AdaptiveSolverSelector(config)
    
    # 创建不同特征的问题
    problems = {
        'large_sparse': (sp.random(15000, 15000, density=0.005, format='csr'), np.random.randn(15000)),
        'ill_conditioned': (sp.random(5000, 5000, density=0.01, format='csr') + sp.eye(5000) * 0.01, np.random.randn(5000)),
        'block_structured': (sp.block_diag([sp.random(1000, 1000, density=0.05) + sp.eye(1000) for _ in range(5)]), np.random.randn(5000))
    }
    
    available_solvers = ['gpu_cg', 'openmp_cg', 'cpu_cg', 'amg']
    
    for problem_name, (A, b) in problems.items():
        print(f"\n🔍 分析问题: {problem_name}")
        print(f"   问题规模: {A.shape[0]}")
        
        # 问题分类
        problem_type = selector.problem_classifier.classify_problem(A, b)
        print(f"   问题类型: {problem_type}")
        
        # 选择最优求解器
        optimal_solver = selector.select_optimal_solver(A, b, available_solvers)
        print(f"   推荐求解器: {optimal_solver}")
        
        # 提取问题特征
        features = selector.problem_classifier._extract_features(A, b)
        print(f"   问题特征:")
        print(f"     规模: {features['size']}")
        print(f"     稀疏性: {features['sparsity']:.3f}")
        print(f"     条件数: {features['condition_number']:.2e}")
        print(f"     结构: {features['structure']}")
    
    return selector


def run_comprehensive_benchmark():
    """运行全面的性能基准测试"""
    print("\n🏆 运行全面性能基准测试")
    print("=" * 50)
    
    # 配置
    config = AdvancedParallelConfig(
        solver_type='adaptive',
        use_gpu=True,
        use_openmp=True,
        ml_based_balancing=True,
        adaptive_communication=True,
        max_iterations=1000,
        tolerance=1e-8
    )
    
    # 创建求解器
    solver = UnderworldSurpassingSolver(config)
    
    # 获取测试问题
    test_problems = create_test_problems()
    
    # 运行基准测试
    benchmark_results = {}
    
    for problem_name, (A, b, description) in test_problems.items():
        print(f"\n🔍 基准测试: {problem_name}")
        print(f"   描述: {description}")
        print(f"   规模: {A.shape[0]} x {A.shape[1]}")
        
        # 你的求解器
        start_time = time.time()
        your_solution = solver.solve(A, b)
        your_time = time.time() - start_time
        
        # 验证解的正确性
        residual = np.linalg.norm(b - A.dot(your_solution))
        
        # Underworld估计时间
        uw_time = solver._estimate_underworld_time(A, b)
        
        # 计算加速比
        speedup = uw_time / your_time if your_time > 0 else 0
        
        benchmark_results[problem_name] = {
            'problem_size': A.shape[0],
            'description': description,
            'your_solver_time': your_time,
            'underworld_estimated_time': uw_time,
            'speedup': speedup,
            'residual_norm': residual
        }
        
        print(f"   你的求解器: {your_time:.4f}s")
        print(f"   Underworld估计: {uw_time:.4f}s")
        print(f"   加速比: {speedup:.2f}x")
        print(f"   残差范数: {residual:.2e}")
    
    # 输出总结
    print("\n📊 基准测试结果总结")
    print("=" * 60)
    
    total_speedup = 0
    for problem_name, results in benchmark_results.items():
        print(f"{problem_name:20s}: 加速比 {results['speedup']:6.2f}x")
        total_speedup += results['speedup']
    
    avg_speedup = total_speedup / len(benchmark_results)
    print(f"\n🏆 平均加速比: {avg_speedup:.2f}x")
    
    return benchmark_results


def save_demo_results(results_dict):
    """保存演示结果"""
    output_file = "underworld_surpassing_demo_results.json"
    
    # 添加时间戳
    results_dict['timestamp'] = time.time()
    results_dict['demo_info'] = {
        'title': '超越Underworld的并行求解器演示',
        'description': '展示通信优化、智能负载均衡、异构计算、自适应算法选择等核心功能',
        'version': '1.0.0'
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 演示结果已保存到: {output_file}")
    return output_file


def main():
    """主演示函数"""
    print("🚀 超越Underworld的并行求解器演示")
    print("=" * 60)
    print("本演示将展示以下核心突破功能:")
    print("1. 🚀 通信优化: 非阻塞通信+计算重叠+自适应通信模式")
    print("2. ⚖️ 智能负载均衡: ML预测+动态负载迁移")
    print("3. 🖥️ 异构计算: MPI+GPU+OpenMP混合架构")
    print("4. 🧠 算法适应性: 问题感知的自适应求解策略")
    print("5. 🏆 性能基准测试: 与Underworld的性能对比")
    print("=" * 60)
    
    # 存储所有演示结果
    demo_results = {}
    
    try:
        # 1. 通信优化演示
        comm_schedule = demonstrate_communication_optimization()
        demo_results['communication_optimization'] = comm_schedule
        
        # 2. 负载均衡演示
        load_balance_result = demonstrate_load_balancing()
        demo_results['load_balancing'] = load_balance_result
        
        # 3. 异构计算演示
        heterogeneous_result = demonstrate_heterogeneous_computing()
        demo_results['heterogeneous_computing'] = {
            'gpu_available': heterogeneous_result.gpu_available,
            'openmp_available': heterogeneous_result.openmp_available,
            'gpu_performance_count': len(heterogeneous_result.gpu_performance),
            'cpu_performance_count': len(heterogeneous_result.cpu_performance)
        }
        
        # 4. 自适应求解器选择演示
        solver_selection_result = demonstrate_adaptive_solver_selection()
        demo_results['adaptive_solver_selection'] = {
            'problem_classifier_available': True,
            'performance_database_size': len(solver_selection_result.solver_performance_db)
        }
        
        # 5. 全面性能基准测试
        benchmark_results = run_comprehensive_benchmark()
        demo_results['benchmark_results'] = benchmark_results
        
        # 保存结果
        output_file = save_demo_results(demo_results)
        
        print(f"\n🎉 演示完成！")
        print(f"   所有功能演示成功")
        print(f"   结果已保存到: {output_file}")
        
        # 性能总结
        if 'benchmark_results' in demo_results:
            avg_speedup = np.mean([r['speedup'] for r in demo_results['benchmark_results'].values()])
            print(f"\n📈 性能总结:")
            print(f"   平均加速比: {avg_speedup:.2f}x")
            print(f"   测试问题数: {len(demo_results['benchmark_results'])}")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    return demo_results


if __name__ == "__main__":
    # 运行演示
    results = main()
    
    print(f"\n🔚 演示结束")
    print("感谢使用超越Underworld的并行求解器！")
