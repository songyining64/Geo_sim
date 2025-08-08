"""
GPU加速综合演示脚本

展示完整的GPU加速功能，包括：
1. CUDA加速计算
2. 并行计算
3. 机器学习优化
4. 性能对比分析
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# 尝试导入GPU加速模块
try:
    from gpu_acceleration.cuda_acceleration import (
        CUDAAccelerator, GPUMatrixOperations, GPUSolver, create_cuda_accelerator
    )
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️ CUDA加速模块不可用")

try:
    from gpu_acceleration.parallel_computing import (
        MPIManager, ParallelSolver, create_parallel_solver
    )
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    print("⚠️ 并行计算模块不可用")

try:
    from gpu_acceleration.ml_optimization import (
        MLAccelerator, NeuralNetworkSolver, SurrogateModel, create_ml_accelerator
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ 机器学习优化模块不可用")


def demo_cuda_acceleration():
    """演示CUDA加速功能"""
    print("🚀 CUDA加速演示")
    print("=" * 50)
    
    if not CUDA_AVAILABLE:
        print("❌ CUDA加速模块不可用")
        return None
    
    try:
        # 创建CUDA加速器
        accelerator = create_cuda_accelerator()
        
        if not accelerator.is_available():
            print("❌ CUDA不可用，请检查GPU和CUDA安装")
            return None
        
        # 创建GPU矩阵运算
        matrix_ops = GPUMatrixOperations(accelerator)
        
        # 创建GPU求解器
        solver = GPUSolver(accelerator)
        
        # 测试矩阵乘法
        print("📊 测试矩阵乘法...")
        size = 1000
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)
        
        # CPU计算
        start_time = time.time()
        C_cpu = np.dot(A, B)
        cpu_time = time.time() - start_time
        
        # GPU计算
        start_time = time.time()
        C_gpu = matrix_ops.matrix_multiply(A, B)
        gpu_time = time.time() - start_time
        
        # 验证结果
        error = np.linalg.norm(C_cpu - C_gpu) / np.linalg.norm(C_cpu)
        
        print(f"   CPU时间: {cpu_time:.4f} 秒")
        print(f"   GPU时间: {gpu_time:.4f} 秒")
        print(f"   加速比: {cpu_time / gpu_time:.2f}x")
        print(f"   误差: {error:.2e}")
        
        # 测试线性系统求解
        print("\n🔧 测试线性系统求解...")
        A = np.random.rand(500, 500).astype(np.float32)
        A = A + A.T + 500 * np.eye(500)  # 确保正定
        b = np.random.rand(500).astype(np.float32)
        
        # CPU求解
        start_time = time.time()
        x_cpu = np.linalg.solve(A, b)
        cpu_time = time.time() - start_time
        
        # GPU求解
        start_time = time.time()
        x_gpu = solver.solve_linear_system(A, b)
        gpu_time = time.time() - start_time
        
        # 验证结果
        error = np.linalg.norm(x_cpu - x_gpu) / np.linalg.norm(x_cpu)
        
        print(f"   CPU时间: {cpu_time:.4f} 秒")
        print(f"   GPU时间: {gpu_time:.4f} 秒")
        print(f"   加速比: {cpu_time / gpu_time:.2f}x")
        print(f"   误差: {error:.2e}")
        
        # 显示性能统计
        print("\n📈 CUDA性能统计:")
        stats = solver.get_performance_stats()
        print(f"   矩阵运算次数: {stats['matrix_ops']}")
        print(f"   求解器运算次数: {stats['solver_ops']}")
        print(f"   总计算时间: {stats['total_time']:.4f} 秒")
        print(f"   GPU内存使用: {stats['memory_info']['used'] / 1024**3:.2f} GB")
        
        return stats
        
    except Exception as e:
        print(f"❌ CUDA加速演示失败: {e}")
        return None


def demo_parallel_computing():
    """演示并行计算功能"""
    print("\n🔄 并行计算演示")
    print("=" * 50)
    
    if not PARALLEL_AVAILABLE:
        print("❌ 并行计算模块不可用")
        return None
    
    try:
        # 创建并行求解器
        solver = create_parallel_solver()
        
        # 创建测试问题
        n_points = 1000
        A = np.random.rand(n_points, n_points)
        A = A + A.T + n_points * np.eye(n_points)  # 确保正定
        b = np.random.rand(n_points)
        
        # 并行求解
        print(f"🔧 使用 {solver.mpi.get_size()} 个进程求解 {n_points}x{n_points} 线性系统...")
        
        start_time = time.time()
        x_parallel = solver.solve_parallel_linear_system(A, b)
        parallel_time = time.time() - start_time
        
        # 串行求解（仅在根进程）
        if solver.mpi.is_root_process():
            start_time = time.time()
            x_serial = np.linalg.solve(A, b)
            serial_time = time.time() - start_time
            
            # 验证结果
            error = np.linalg.norm(x_parallel - x_serial) / np.linalg.norm(x_serial)
            
            print(f"   串行时间: {serial_time:.4f} 秒")
            print(f"   并行时间: {parallel_time:.4f} 秒")
            print(f"   加速比: {serial_time / parallel_time:.2f}x")
            print(f"   误差: {error:.2e}")
        
        # 显示性能统计
        stats = solver.get_performance_stats()
        print(f"\n📈 进程 {stats['rank']} 性能统计:")
        print(f"   求解时间: {stats['solve_time']:.4f} 秒")
        print(f"   迭代次数: {stats['iterations']}")
        
        return stats
        
    except Exception as e:
        print(f"❌ 并行计算演示失败: {e}")
        return None


def demo_ml_optimization():
    """演示机器学习优化功能"""
    print("\n🤖 机器学习优化演示")
    print("=" * 50)
    
    if not ML_AVAILABLE:
        print("❌ 机器学习优化模块不可用")
        return None
    
    try:
        # 创建机器学习加速器
        accelerator = create_ml_accelerator()
        
        # 创建测试数据
        n_samples = 1000
        X = np.random.rand(n_samples, 2) * 10
        y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * np.random.randn(n_samples)
        
        # 测试高斯过程代理模型
        print("📊 测试高斯过程代理模型...")
        gp_model = accelerator.create_surrogate_model('gp_test', 'gaussian_process')
        
        start_time = time.time()
        history = accelerator.train_surrogate_model('gp_test', X, y)
        training_time = time.time() - start_time
        
        # 测试预测
        X_test = np.random.rand(100, 2) * 10
        start_time = time.time()
        y_pred = accelerator.predict_with_surrogate('gp_test', X_test)
        prediction_time = time.time() - start_time
        
        print(f"   训练时间: {training_time:.4f} 秒")
        print(f"   预测时间: {prediction_time:.4f} 秒")
        print(f"   预测样本数: {len(X_test)}")
        
        # 测试神经网络求解器
        print("\n🧠 测试神经网络求解器...")
        nn_solver = accelerator.create_neural_solver('nn_test', 2, [64, 32], 1)
        
        start_time = time.time()
        history = accelerator.train_surrogate_model('nn_test', X, y, epochs=50, batch_size=32)
        training_time = time.time() - start_time
        
        start_time = time.time()
        y_pred_nn = accelerator.predict_with_surrogate('nn_test', X_test)
        prediction_time = time.time() - start_time
        
        print(f"   训练时间: {training_time:.4f} 秒")
        print(f"   预测时间: {prediction_time:.4f} 秒")
        print(f"   最终训练损失: {history['train_loss'][-1]:.6f}")
        
        # 显示性能统计
        print("\n📈 ML性能统计:")
        stats = accelerator.get_performance_stats()
        print(f"   设备: {stats['device']}")
        print(f"   使用GPU: {stats['use_gpu']}")
        print(f"   代理模型数量: {stats['num_surrogate_models']}")
        print(f"   神经网络求解器数量: {stats['num_neural_solvers']}")
        print(f"   总训练时间: {stats['training_time']:.4f} 秒")
        print(f"   总预测时间: {stats['prediction_time']:.4f} 秒")
        print(f"   训练模型数量: {stats['models_trained']}")
        
        return stats
        
    except Exception as e:
        print(f"❌ 机器学习优化演示失败: {e}")
        return None


def performance_comparison(cuda_stats, parallel_stats, ml_stats):
    """性能对比分析"""
    print("\n" + "=" * 80)
    print("📊 GPU加速性能对比分析")
    print("=" * 80)
    
    print("🎯 功能模块对比:")
    print(f"   CUDA加速: {'✅ 可用' if cuda_stats else '❌ 不可用'}")
    print(f"   并行计算: {'✅ 可用' if parallel_stats else '❌ 不可用'}")
    print(f"   机器学习: {'✅ 可用' if ml_stats else '❌ 不可用'}")
    
    if cuda_stats:
        print(f"\n🚀 CUDA加速性能:")
        print(f"   矩阵运算: {cuda_stats['matrix_ops']} 次")
        print(f"   求解器运算: {cuda_stats['solver_ops']} 次")
        print(f"   总计算时间: {cuda_stats['total_time']:.4f} 秒")
        print(f"   GPU内存使用: {cuda_stats['memory_info']['used'] / 1024**3:.2f} GB")
    
    if parallel_stats:
        print(f"\n🔄 并行计算性能:")
        print(f"   进程数: {parallel_stats['size']}")
        print(f"   求解时间: {parallel_stats['solve_time']:.4f} 秒")
        print(f"   迭代次数: {parallel_stats['iterations']}")
    
    if ml_stats:
        print(f"\n🤖 机器学习性能:")
        print(f"   设备: {ml_stats['device']}")
        print(f"   使用GPU: {ml_stats['use_gpu']}")
        print(f"   总训练时间: {ml_stats['training_time']:.4f} 秒")
        print(f"   总预测时间: {ml_stats['prediction_time']:.4f} 秒")
        print(f"   训练模型数: {ml_stats['models_trained']}")
    
    print("\n💡 优化建议:")
    if cuda_stats and cuda_stats['matrix_ops'] > 0:
        print("   • CUDA加速在矩阵运算方面表现优异")
    if parallel_stats and parallel_stats['size'] > 1:
        print("   • 并行计算可有效利用多核CPU资源")
    if ml_stats and ml_stats['use_gpu']:
        print("   • GPU加速机器学习训练可显著提升性能")
    
    print("   • 建议根据具体应用场景选择合适的加速方案")
    print("   • 对于大规模计算，可考虑GPU+MPI混合加速")


def main():
    """主函数"""
    print("🌍 GPU加速综合演示")
    print("=" * 80)
    print("本演示展示了完整的GPU加速功能")
    print("包括：CUDA加速、并行计算、机器学习优化等")
    print("=" * 80)
    
    # 检查模块可用性
    print("🔍 检查模块可用性:")
    print(f"   CUDA加速模块: {'✅ 可用' if CUDA_AVAILABLE else '❌ 不可用'}")
    print(f"   并行计算模块: {'✅ 可用' if PARALLEL_AVAILABLE else '❌ 不可用'}")
    print(f"   机器学习模块: {'✅ 可用' if ML_AVAILABLE else '❌ 不可用'}")
    
    # 运行演示
    cuda_stats = demo_cuda_acceleration()
    parallel_stats = demo_parallel_computing()
    ml_stats = demo_ml_optimization()
    
    # 性能对比分析
    performance_comparison(cuda_stats, parallel_stats, ml_stats)
    
    print("\n" + "=" * 80)
    print("✅ GPU加速综合演示完成!")
    print("=" * 80)
    print("🎯 实现的功能:")
    print("   • CUDA GPU加速计算")
    print("   • MPI并行计算")
    print("   • 机器学习优化")
    print("   • 性能对比分析")
    print("   • 自动回退机制")
    
    print("\n📈 技术特点:")
    print("   • 支持多种GPU加速库 (CuPy, Numba)")
    print("   • 灵活的并行计算策略")
    print("   • 深度学习集成")
    print("   • 代理模型优化")
    print("   • 完整的性能监控")
    
    print("\n🚀 应用场景:")
    print("   • 大规模科学计算")
    print("   • 有限元分析")
    print("   • 机器学习训练")
    print("   • 数值优化")
    print("   • 高性能计算")


if __name__ == "__main__":
    main() 