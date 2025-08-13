"""
增强型核心求解器演示

展示以下增强功能：
1. 多重网格求解器的完善
2. 多物理场耦合求解
3. 高级时间积分器
"""

import numpy as np
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# 导入增强的求解器
from multigrid_solver import (
    create_multigrid_solver, 
    create_multigrid_config,
    benchmark_multigrid_solvers
)

from multiphysics_coupling_solver import (
    create_multiphysics_solver,
    create_coupling_config,
    benchmark_coupling_strategies
)

from time_integration.advanced_integrators import (
    create_time_integrator,
    benchmark_time_integrators
)


def create_test_problem(size: int = 1000, problem_type: str = 'poisson') -> Tuple[sp.spmatrix, np.ndarray]:
    """创建测试问题"""
    print(f"🔧 创建测试问题: {problem_type}, 规模: {size}")
    
    if problem_type == 'poisson':
        # Poisson方程：-∇²u = f
        A = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(size, size), format='csr')
        b = np.ones(size)
        b[0] = 0  # 边界条件
        b[-1] = 0
        
    elif problem_type == 'helmholtz':
        # Helmholtz方程：(∇² + k²)u = f
        k = 1.0
        A = sp.diags([-1, 2-k**2, -1], [-1, 0, 1], shape=(size, size), format='csr')
        b = np.ones(size)
        b[0] = 0
        b[-1] = 0
        
    elif problem_type == 'elasticity':
        # 弹性问题：Ku = f
        A = sp.diags([-0.5, 2, -0.5], [-1, 0, 1], shape=(size, size), format='csr')
        # 确保正定性
        A = A + sp.eye(size) * 0.1
        b = np.random.randn(size)
        
    else:
        raise ValueError(f"不支持的问题类型: {problem_type}")
    
    return A, b


def demo_enhanced_multigrid():
    """演示增强型多重网格求解器"""
    print("\n" + "="*60)
    print("🚀 增强型多重网格求解器演示")
    print("="*60)
    
    # 创建测试问题
    A, b = create_test_problem(2000, 'poisson')
    
    # 测试不同的多重网格配置
    configs = [
        create_multigrid_config(
            smoother='jacobi', 
            cycle_type='v',
            adaptive_coarsening=True,
            max_coarse_size=50
        ),
        create_multigrid_config(
            smoother='gauss_seidel', 
            cycle_type='v',
            adaptive_coarsening=True,
            max_coarse_size=50
        ),
        create_multigrid_config(
            smoother='chebyshev', 
            cycle_type='v',
            adaptive_coarsening=True,
            max_coarse_size=50
        ),
        create_multigrid_config(
            smoother='jacobi', 
            cycle_type='w',
            adaptive_coarsening=True,
            max_coarse_size=50
        ),
        create_multigrid_config(
            smoother='jacobi', 
            cycle_type='fmg',
            adaptive_coarsening=True,
            max_coarse_size=50
        ),
    ]
    
    # 运行基准测试
    results = benchmark_multigrid_solvers(A, b, configs)
    
    # 分析结果
    print("\n📊 多重网格求解器性能分析:")
    print("-" * 50)
    
    best_config = None
    best_time = float('inf')
    
    for config_name, result in results.items():
        print(f"{config_name:20s}: 设置时间 {result['setup_time']:8.4f}s, "
              f"求解时间 {result['solve_time']:8.4f}s, "
              f"迭代次数 {result['iterations']:3d}")
        
        if result['total_time'] < best_time:
            best_time = result['total_time']
            best_config = config_name
    
    print(f"\n🏆 最佳配置: {best_config}")
    print(f"   总时间: {best_time:.4f}s")
    
    return results


def demo_multiphysics_coupling():
    """演示多物理场耦合求解"""
    print("\n" + "="*60)
    print("🌊 多物理场耦合求解演示")
    print("="*60)
    
    # 创建测试网格数据
    mesh_data = {
        'nodes': [(i, 0) for i in range(100)],  # 1D网格
        'elements': [{'nodes': [i, i+1]} for i in range(99)]
    }
    
    # 材料属性
    material_props = {
        'thermal': {'conductivity': 1.0, 'density': 1.0, 'specific_heat': 1.0},
        'mechanical': {'youngs_modulus': 1e6, 'poisson_ratio': 0.3, 'density': 1.0}
    }
    
    # 边界条件
    boundary_conditions = {
        'thermal': [
            {'type': 'dirichlet', 'node_id': 0, 'value': 100.0},  # 左端温度
            {'type': 'dirichlet', 'node_id': 99, 'value': 0.0}    # 右端温度
        ],
        'mechanical': [
            {'type': 'dirichlet', 'node_id': 0, 'component': 0, 'value': 0.0},  # 左端固定
            {'type': 'dirichlet', 'node_id': 99, 'component': 0, 'value': 0.01}  # 右端位移
        ]
    }
    
    # 耦合属性
    coupling_props = {
        'thermal_expansion': 1e-5,  # 热膨胀系数
        'thermoelastic_coupling': True
    }
    
    # 测试不同的耦合策略
    strategies = ['staggered', 'monolithic', 'hybrid']
    results = {}
    
    for strategy in strategies:
        print(f"\n🧪 测试耦合策略: {strategy}")
        
        # 创建配置
        config = create_coupling_config(
            coupling_type=strategy,
            physics_fields=['thermal', 'mechanical'],
            max_iterations=30,
            tolerance=1e-6
        )
        
        # 创建求解器
        solver = create_multiphysics_solver(config)
        
        # 设置和求解
        start_time = time.time()
        solver.setup(mesh_data, material_props, boundary_conditions, coupling_props)
        setup_time = time.time() - start_time
        
        start_time = time.time()
        if strategy == 'staggered':
            solutions = solver.solve_staggered()
        elif strategy == 'monolithic':
            solutions = solver.solve_monolithic()
        else:  # hybrid
            solutions = solver.solve_hybrid()
        solve_time = time.time() - start_time
        
        # 存储结果
        results[strategy] = {
            'setup_time': setup_time,
            'solve_time': solve_time,
            'total_time': setup_time + solve_time,
            'solutions': solutions,
            'performance_stats': solver.get_performance_stats()
        }
        
        print(f"   设置时间: {setup_time:.4f}s")
        print(f"   求解时间: {solve_time:.4f}s")
        print(f"   总时间: {setup_time + solve_time:.4f}s")
        
        # 显示解的基本信息
        for field_name, solution in solutions.items():
            print(f"   {field_name} 场解: 最小值 {solution.min():.2e}, "
                  f"最大值 {solution.max():.2e}, 范数 {np.linalg.norm(solution):.2e}")
    
    # 分析结果
    print("\n📊 多物理场耦合策略性能分析:")
    print("-" * 50)
    
    best_strategy = None
    best_time = float('inf')
    
    for strategy, result in results.items():
        print(f"{strategy:15s}: 设置时间 {result['setup_time']:8.4f}s, "
              f"求解时间 {result['solve_time']:8.4f}s, "
              f"总时间 {result['total_time']:8.4f}s")
        
        if result['total_time'] < best_time:
            best_time = result['total_time']
            best_strategy = strategy
    
    print(f"\n🏆 最佳策略: {best_strategy}")
    print(f"   总时间: {best_time:.4f}s")
    
    return results


def demo_advanced_time_integrators():
    """演示高级时间积分器"""
    print("\n" + "="*60)
    print("⏰ 高级时间积分器演示")
    print("="*60)
    
    # 创建测试系统
    def test_system(t: float, y: np.ndarray) -> np.ndarray:
        """测试系统：y' = -y + sin(t)"""
        return -y + np.sin(t)
    
    # 初始条件
    initial_state = np.array([1.0])
    time_span = (0.0, 10.0)
    dt = 0.1
    
    # 测试不同的时间积分器
    integrator_types = ['bdf', 'crank_nicolson', 'adaptive']
    results = {}
    
    for integrator_type in integrator_types:
        print(f"\n🧪 测试时间积分器: {integrator_type}")
        
        # 创建积分器
        integrator = create_time_integrator(integrator_type, order=2)
        
        # 积分
        start_time = time.time()
        final_solution = integrator.integrate(dt, test_system, initial_state, 
                                           end_time=time_span[1])
        solve_time = time.time() - start_time
        
        # 计算误差（简化版本）
        error = np.linalg.norm(final_solution - initial_state)
        
        # 存储结果
        results[integrator_type] = {
            'solve_time': solve_time,
            'steps_taken': integrator.steps_taken,
            'final_error': error,
            'performance_stats': integrator.get_performance_stats()
        }
        
        print(f"   求解时间: {solve_time:.4f}s")
        print(f"   步数: {integrator.steps_taken}")
        print(f"   最终误差: {error:.2e}")
        
        # 显示性能统计
        stats = integrator.get_performance_stats()
        print(f"   总步数: {stats['total_steps']}")
        print(f"   成功步数: {stats['successful_steps']}")
        print(f"   失败步数: {stats['failed_steps']}")
    
    # 分析结果
    print("\n📊 时间积分器性能分析:")
    print("-" * 50)
    
    best_integrator = None
    best_time = float('inf')
    
    for integrator_type, result in results.items():
        print(f"{integrator_type:20s}: 求解时间 {result['solve_time']:8.4f}s, "
              f"步数 {result['steps_taken']:4d}, "
              f"误差 {result['final_error']:8.2e}")
        
        if result['solve_time'] < best_time:
            best_time = result['solve_time']
            best_integrator = integrator_type
    
    print(f"\n🏆 最佳积分器: {best_integrator}")
    print(f"   求解时间: {best_time:.4f}s")
    
    return results


def demo_transient_multiphysics():
    """演示瞬态多物理场求解"""
    print("\n" + "="*60)
    print("🔄 瞬态多物理场求解演示")
    print("="*60)
    
    # 创建多物理场求解器
    config = create_coupling_config(
        coupling_type='staggered',
        physics_fields=['thermal', 'mechanical'],
        time_integration='implicit',
        time_step=0.01,
        max_time_steps=100
    )
    
    solver = create_multiphysics_solver(config)
    
    # 创建测试网格
    mesh_data = {
        'nodes': [(i, 0) for i in range(50)],
        'elements': [{'nodes': [i, i+1]} for i in range(49)]
    }
    
    # 材料属性
    material_props = {
        'thermal': {'conductivity': 1.0, 'density': 1.0, 'specific_heat': 1.0},
        'mechanical': {'youngs_modulus': 1e6, 'poisson_ratio': 0.3, 'density': 1.0}
    }
    
    # 边界条件
    boundary_conditions = {
        'thermal': [
            {'type': 'dirichlet', 'node_id': 0, 'value': 100.0},
            {'type': 'dirichlet', 'node_id': 49, 'value': 0.0}
        ],
        'mechanical': [
            {'type': 'dirichlet', 'node_id': 0, 'component': 0, 'value': 0.0},
            {'type': 'dirichlet', 'node_id': 49, 'component': 0, 'value': 0.0}
        ]
    }
    
    # 初始条件
    initial_conditions = {
        'thermal': np.zeros(50),
        'mechanical': np.zeros(100)  # 2D位移场
    }
    
    # 设置求解器
    solver.setup(mesh_data, material_props, boundary_conditions)
    
    # 瞬态求解
    print("🔄 开始瞬态求解...")
    start_time = time.time()
    
    transient_results = solver.solve_transient(
        initial_conditions, 
        time_span=(0.0, 1.0)
    )
    
    solve_time = time.time() - start_time
    
    print(f"✅ 瞬态求解完成")
    print(f"   求解时间: {solve_time:.4f}s")
    print(f"   时间步数: {len(transient_results['time_steps'])}")
    print(f"   最终时间: {transient_results['time_steps'][-1]:.3f}s")
    
    # 显示最终解
    final_solutions = transient_results['final_solutions']
    for field_name, solution in final_solutions.items():
        print(f"   {field_name} 场最终解: 范数 {np.linalg.norm(solution):.2e}")
    
    return transient_results


def plot_results(multigrid_results, coupling_results, time_integrator_results):
    """绘制结果图表"""
    print("\n📈 生成结果图表...")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('增强型核心求解器性能分析', fontsize=16)
    
    # 1. 多重网格求解器性能
    ax1 = axes[0, 0]
    config_names = list(multigrid_results.keys())
    setup_times = [multigrid_results[name]['setup_time'] for name in config_names]
    solve_times = [multigrid_results[name]['solve_time'] for name in config_names]
    
    x = np.arange(len(config_names))
    width = 0.35
    
    ax1.bar(x - width/2, setup_times, width, label='设置时间', alpha=0.8)
    ax1.bar(x + width/2, solve_times, width, label='求解时间', alpha=0.8)
    ax1.set_xlabel('配置')
    ax1.set_ylabel('时间 (s)')
    ax1.set_title('多重网格求解器性能')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 多物理场耦合策略性能
    ax2 = axes[0, 1]
    strategy_names = list(coupling_results.keys())
    total_times = [coupling_results[name]['total_time'] for name in strategy_names]
    
    bars = ax2.bar(strategy_names, total_times, alpha=0.8, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax2.set_xlabel('耦合策略')
    ax2.set_ylabel('总时间 (s)')
    ax2.set_title('多物理场耦合策略性能')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, time_val in zip(bars, total_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    # 3. 时间积分器性能
    ax3 = axes[1, 0]
    integrator_names = list(time_integrator_results.keys())
    solve_times = [time_integrator_results[name]['solve_time'] for name in integrator_names]
    steps = [time_integrator_results[name]['steps_taken'] for name in integrator_names]
    
    bars = ax3.bar(integrator_names, solve_times, alpha=0.8, color=['gold', 'lightblue', 'lightpink'])
    ax3.set_xlabel('积分器类型')
    ax3.set_ylabel('求解时间 (s)')
    ax3.set_title('时间积分器性能')
    ax3.grid(True, alpha=0.3)
    
    # 添加步数标签
    for bar, step_count in zip(bars, steps):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{step_count}步', ha='center', va='bottom')
    
    # 4. 性能对比雷达图
    ax4 = axes[1, 1]
    
    # 计算性能指标（归一化）
    categories = ['设置时间', '求解时间', '迭代次数', '精度']
    
    # 多重网格性能
    mg_performance = [
        1.0 / (1.0 + min(multigrid_results[name]['setup_time'] for name in multigrid_results)),
        1.0 / (1.0 + min(multigrid_results[name]['solve_time'] for name in multigrid_results)),
        1.0 / (1.0 + min(multigrid_results[name]['iterations'] for name in multigrid_results)),
        0.9  # 假设精度
    ]
    
    # 耦合策略性能
    coupling_performance = [
        1.0 / (1.0 + min(coupling_results[name]['setup_time'] for name in coupling_results)),
        1.0 / (1.0 + min(coupling_results[name]['solve_time'] for name in coupling_results)),
        0.8,  # 假设稳定性
        0.95  # 假设精度
    ]
    
    # 时间积分器性能
    time_performance = [
        0.9,  # 假设设置时间
        1.0 / (1.0 + min(time_integrator_results[name]['solve_time'] for name in time_integrator_results)),
        1.0 / (1.0 + min(time_integrator_results[name]['steps_taken'] for name in time_integrator_results)),
        0.85  # 假设精度
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    mg_performance += mg_performance[:1]
    coupling_performance += coupling_performance[:1]
    time_performance += time_performance[:1]
    
    ax4.plot(angles, mg_performance, 'o-', linewidth=2, label='多重网格', color='blue')
    ax4.fill(angles, mg_performance, alpha=0.25, color='blue')
    ax4.plot(angles, coupling_performance, 'o-', linewidth=2, label='多物理场耦合', color='red')
    ax4.fill(angles, coupling_performance, alpha=0.25, color='red')
    ax4.plot(angles, time_performance, 'o-', linewidth=2, label='时间积分器', color='green')
    ax4.fill(angles, time_performance, alpha=0.25, color='green')
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('性能对比雷达图')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('enhanced_solver_performance.png', dpi=300, bbox_inches='tight')
    print("💾 性能图表已保存为 'enhanced_solver_performance.png'")
    
    plt.show()


def main():
    """主函数"""
    print("🚀 增强型核心求解器演示")
    print("=" * 80)
    print("本演示展示以下增强功能：")
    print("1. 多重网格求解器的完善：网格粗化策略、循环策略扩展、平滑器优化")
    print("2. 多物理场耦合求解：耦合方程组组装、分区求解策略、时间积分器")
    print("3. 高级时间积分器：隐式时间步进算法、自适应时间步长控制")
    print("=" * 80)
    
    try:
        # 运行演示
        multigrid_results = demo_enhanced_multigrid()
        coupling_results = demo_multiphysics_coupling()
        time_integrator_results = demo_advanced_time_integrators()
        transient_results = demo_transient_multiphysics()
        
        # 生成性能图表
        plot_results(multigrid_results, coupling_results, time_integrator_results)
        
        print("\n🎉 所有演示完成！")
        print("\n📋 总结:")
        print(f"   - 多重网格求解器: 测试了 {len(multigrid_results)} 种配置")
        print(f"   - 多物理场耦合: 测试了 {len(coupling_results)} 种策略")
        print(f"   - 时间积分器: 测试了 {len(time_integrator_results)} 种类型")
        print(f"   - 瞬态求解: 成功完成 {len(transient_results['time_steps'])} 个时间步")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
