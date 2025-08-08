"""
GPU加速演示脚本
"""

import numpy as np
import time

def demo_gpu_acceleration():
    """演示GPU加速功能"""
    print("🚀 GPU加速演示")
    print("=" * 50)
    
    # 测试矩阵乘法
    print("📊 测试矩阵乘法...")
    size = 1000
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    
    # CPU计算
    start_time = time.time()
    C_cpu = np.dot(A, B)
    cpu_time = time.time() - start_time
    
    print(f"   CPU时间: {cpu_time:.4f} 秒")
    print(f"   矩阵大小: {size}x{size}")
    
    # 尝试GPU计算
    try:
        import cupy as cp
        print("   🎯 尝试CuPy GPU加速...")
        
        start_time = time.time()
        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        C_gpu = cp.dot(A_gpu, B_gpu)
        result = cp.asnumpy(C_gpu)
        gpu_time = time.time() - start_time
        
        error = np.linalg.norm(C_cpu - result) / np.linalg.norm(C_cpu)
        
        print(f"   GPU时间: {gpu_time:.4f} 秒")
        print(f"   加速比: {cpu_time / gpu_time:.2f}x")
        print(f"   误差: {error:.2e}")
        
    except ImportError:
        print("   ❌ CuPy不可用，跳过GPU计算")
    
    # 测试线性系统求解
    print("\n🔧 测试线性系统求解...")
    A = np.random.rand(500, 500).astype(np.float32)
    A = A + A.T + 500 * np.eye(500)
    b = np.random.rand(500).astype(np.float32)
    
    start_time = time.time()
    x_cpu = np.linalg.solve(A, b)
    cpu_time = time.time() - start_time
    
    print(f"   CPU时间: {cpu_time:.4f} 秒")
    
    try:
        import cupy as cp
        print("   🎯 尝试CuPy GPU求解...")
        
        start_time = time.time()
        A_gpu = cp.asarray(A)
        b_gpu = cp.asarray(b)
        x_gpu = cp.linalg.solve(A_gpu, b_gpu)
        result = cp.asnumpy(x_gpu)
        gpu_time = time.time() - start_time
        
        error = np.linalg.norm(x_cpu - result) / np.linalg.norm(x_cpu)
        
        print(f"   GPU时间: {gpu_time:.4f} 秒")
        print(f"   加速比: {cpu_time / gpu_time:.2f}x")
        print(f"   误差: {error:.2e}")
        
    except ImportError:
        print("   ❌ CuPy不可用，跳过GPU求解")
    
    print("\n✅ GPU加速演示完成!")


def demo_parallel_computing():
    """演示并行计算功能"""
    print("\n🔄 并行计算演示")
    print("=" * 50)
    
    try:
        from mpi4py import MPI
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        print(f"   MPI进程: {rank}/{size}")
        
        # 创建测试数据
        n_points = 1000
        A = np.random.rand(n_points, n_points)
        A = A + A.T + n_points * np.eye(n_points)
        b = np.random.rand(n_points)
        
        # 域分解
        points_per_process = n_points // size
        start_idx = rank * points_per_process
        end_idx = start_idx + points_per_process if rank < size - 1 else n_points
        
        local_b = b[start_idx:end_idx]
        local_A = A[start_idx:end_idx, :]
        
        # 本地求解
        from scipy.sparse.linalg import spsolve
        local_x = spsolve(local_A, local_b)
        
        # 收集结果
        all_x = comm.gather(local_x, root=0)
        
        if rank == 0:
            x_parallel = np.concatenate(all_x)
            
            # 串行求解对比
            start_time = time.time()
            x_serial = np.linalg.solve(A, b)
            serial_time = time.time() - start_time
            
            error = np.linalg.norm(x_parallel - x_serial) / np.linalg.norm(x_serial)
            
            print(f"   串行时间: {serial_time:.4f} 秒")
            print(f"   并行进程数: {size}")
            print(f"   误差: {error:.2e}")
        
        print("   ✅ 并行计算演示完成!")
        
    except ImportError:
        print("   ❌ MPI4py不可用，跳过并行计算")


def demo_ml_optimization():
    """演示机器学习优化功能"""
    print("\n🤖 机器学习优化演示")
    print("=" * 50)
    
    # 创建测试数据
    n_samples = 1000
    X = np.random.rand(n_samples, 2) * 10
    y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * np.random.randn(n_samples)
    
    # 测试高斯过程回归
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        
        print("   📊 测试高斯过程回归...")
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
        
        start_time = time.time()
        gp.fit(X, y)
        training_time = time.time() - start_time
        
        X_test = np.random.rand(100, 2) * 10
        start_time = time.time()
        y_pred = gp.predict(X_test)
        prediction_time = time.time() - start_time
        
        print(f"   训练时间: {training_time:.4f} 秒")
        print(f"   预测时间: {prediction_time:.4f} 秒")
        print(f"   预测样本数: {len(X_test)}")
        
    except ImportError:
        print("   ❌ scikit-learn不可用，跳过高斯过程回归")
    
    # 测试神经网络
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        print("   🧠 测试神经网络...")
        
        class SimpleNN(nn.Module):
            def __init__(self):
                super(SimpleNN, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(2, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
            
            def forward(self, x):
                return self.network(x)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        
        start_time = time.time()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        training_time = time.time() - start_time
        
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        start_time = time.time()
        with torch.no_grad():
            y_pred_nn = model(X_test_tensor).cpu().numpy()
        prediction_time = time.time() - start_time
        
        print(f"   训练时间: {training_time:.4f} 秒")
        print(f"   预测时间: {prediction_time:.4f} 秒")
        print(f"   设备: {device}")
        
    except ImportError:
        print("   ❌ PyTorch不可用，跳过神经网络")
    
    print("   ✅ 机器学习优化演示完成!")


def main():
    """主函数"""
    print("🌍 GPU加速综合演示")
    print("=" * 80)
    
    demo_gpu_acceleration()
    demo_parallel_computing()
    demo_ml_optimization()
    
    print("\n" + "=" * 80)
    print("✅ GPU加速综合演示完成!")
    print("=" * 80)
    print("🎯 实现的功能:")
    print("   • CUDA GPU加速计算")
    print("   • MPI并行计算")
    print("   • 机器学习优化")
    print("   • 自动回退机制")


if __name__ == "__main__":
    main() 