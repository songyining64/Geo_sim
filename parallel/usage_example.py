"""
高级并行求解器使用示例

展示如何使用合并后的高级并行求解器
"""

import numpy as np
import scipy.sparse as sp
from advanced_parallel_solver_v2 import (
    AdvancedParallelConfig, 
    AdvancedParallelSolver,
    ParallelCGSolver,
    ParallelConfig
)


def create_test_problem(size: int = 1000, density: float = 0.01):
    """创建测试问题"""
    # 创建稀疏矩阵
    A = sp.random(size, size, density=density, format='csr')
    A = A + A.T + sp.eye(size)  # 确保正定性
    b = np.random.randn(size)
    return A, b


def example_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 创建配置
    config = AdvancedParallelConfig(
        solver_type='adaptive',
        use_gpu=False,  # 如果没有GPU，设为False
        use_openmp=True,
        ml_based_balancing=True,
        adaptive_communication=True
    )
    
    # 创建求解器
    solver = AdvancedParallelSolver(config)
    
    # 创建测试问题
    A, b = create_test_problem(1000)
    
    # 求解
    print(f"问题规模: {A.shape[0]}")
    solution = solver.solve(A, b)
    
    # 检查结果
    residual = np.linalg.norm(b - A.dot(solution))
    print(f"残差范数: {residual:.2e}")
    
    # 获取性能总结
    performance = solver.get_performance_summary()
    print(f"求解时间: {performance['average_solve_time']:.4f}s")
    
    return solver, solution


def example_parallel_cg():
    """并行CG求解器示例"""
    print("\n=== 并行CG求解器示例 ===")
    
    # 创建配置
    config = ParallelConfig(
        solver_type='cg',
        max_iterations=1000,
        tolerance=1e-8,
        use_nonblocking=True,
        overlap_communication=True
    )
    
    # 创建求解器
    solver = ParallelCGSolver(config)
    
    # 创建测试问题
    A, b = create_test_problem(2000)
    
    # 求解
    print(f"问题规模: {A.shape[0]}")
    solution = solver.solve(A, b)
    
    # 检查结果
    residual = np.linalg.norm(b - A.dot(solution))
    print(f"残差范数: {residual:.2e}")
    print(f"迭代次数: {solver.stats.iterations}")
    print(f"求解时间: {solver.stats.solve_time:.4f}s")
    
    return solver, solution


def example_heterogeneous_computing():
    """异构计算示例"""
    print("\n=== 异构计算示例 ===")
    
    # 创建支持GPU和OpenMP的配置
    config = AdvancedParallelConfig(
        solver_type='adaptive',
        use_gpu=True,
        use_openmp=True,
        cpu_threads=4,
        gpu_memory_fraction=0.8
    )
    
    # 创建求解器
    solver = AdvancedParallelSolver(config)
    
    # 创建不同规模的测试问题
    problem_sizes = [1000, 5000, 10000]
    
    for size in problem_sizes:
        print(f"\n测试问题规模: {size}")
        A, b = create_test_problem(size)
        
        # 求解
        solution = solver.solve(A, b)
        
        # 检查结果
        residual = np.linalg.norm(b - A.dot(solution))
        print(f"  残差范数: {residual:.2e}")
        print(f"  求解时间: {solver.performance_metrics.total_solve_time:.4f}s")
    
    return solver


def example_communication_optimization():
    """通信优化示例"""
    print("\n=== 通信优化示例 ===")
    
    # 创建支持通信优化的配置
    config = AdvancedParallelConfig(
        solver_type='adaptive',
        adaptive_communication=True,
        use_nonblocking=True,
        overlap_communication=True,
        buffer_size=4096
    )
    
    # 创建求解器
    solver = AdvancedParallelSolver(config)
    
    # 创建测试问题
    A, b = create_test_problem(5000)
    
    # 求解
    solution = solver.solve(A, b)
    
    # 获取通信统计
    comm_stats = solver.communicator.get_communication_statistics()
    print(f"总通信次数: {comm_stats['total_communications']}")
    print(f"调度更新次数: {comm_stats['schedule_updates']}")
    
    return solver, solution


def main():
    """主函数"""
    print("🚀 高级并行求解器使用示例")
    print("=" * 50)
    
    try:
        # 基础使用
        solver1, solution1 = example_basic_usage()
        
        # 并行CG
        solver2, solution2 = example_parallel_cg()
        
        # 异构计算
        solver3 = example_heterogeneous_computing()
        
        # 通信优化
        solver4, solution4 = example_communication_optimization()
        
        print("\n✅ 所有示例运行成功！")
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
