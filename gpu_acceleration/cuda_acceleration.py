"""
CUDA加速模块 - 提供GPU加速计算功能
"""

import numpy as np
import time
import warnings

# CUDA相关依赖
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    import numba
    from numba import cuda
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    cuda = None

try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    torch = None


class CUDAAccelerator:
    """CUDA加速器"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = None
        
        if not HAS_CUPY and not HAS_NUMBA:
            raise ImportError("需要安装cupy或numba来使用CUDA加速")
        
        self._initialize_device()
        self.performance_stats = {'matrix_ops': 0, 'solver_ops': 0, 'total_time': 0.0}
    
    def _initialize_device(self):
        if HAS_CUPY:
            try:
                cp.cuda.Device(self.device_id).use()
                self.device = cp.cuda.Device(self.device_id)
                print(f"✅ 使用CuPy GPU加速，设备: {self.device.name}")
            except Exception as e:
                print(f"❌ CuPy初始化失败: {e}")
                self.device = None
        
        elif HAS_NUMBA:
            try:
                self.device = cuda.get_current_device()
                print(f"✅ 使用Numba GPU加速，设备: {self.device.name}")
            except Exception as e:
                print(f"❌ Numba初始化失败: {e}")
                self.device = None
    
    def is_available(self) -> bool:
        return self.device is not None
    
    def get_memory_info(self) -> dict:
        if not self.is_available():
            return {'total': 0, 'free': 0, 'used': 0}
        
        if HAS_CUPY:
            meminfo = cp.cuda.runtime.memGetInfo()
            return {
                'total': meminfo[1],
                'free': meminfo[0],
                'used': meminfo[1] - meminfo[0]
            }
        return {'total': 0, 'free': 0, 'used': 0}


class GPUMatrixOperations:
    """GPU矩阵运算"""
    
    def __init__(self, accelerator: CUDAAccelerator):
        self.accelerator = accelerator
    
    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if not self.accelerator.is_available():
            return np.dot(A, B)
        
        start_time = time.time()
        
        if HAS_CUPY:
            A_gpu = cp.asarray(A)
            B_gpu = cp.asarray(B)
            C_gpu = cp.dot(A_gpu, B_gpu)
            result = cp.asnumpy(C_gpu)
            
            del A_gpu, B_gpu, C_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            result = np.dot(A, B)
        
        elapsed_time = time.time() - start_time
        self.accelerator.performance_stats['matrix_ops'] += 1
        self.accelerator.performance_stats['total_time'] += elapsed_time
        
        return result
    
    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self.accelerator.is_available():
            return np.linalg.solve(A, b)
        
        start_time = time.time()
        
        if HAS_CUPY:
            A_gpu = cp.asarray(A)
            b_gpu = cp.asarray(b)
            x_gpu = cp.linalg.solve(A_gpu, b_gpu)
            result = cp.asnumpy(x_gpu)
            
            del A_gpu, b_gpu, x_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            result = np.linalg.solve(A, b)
        
        elapsed_time = time.time() - start_time
        self.accelerator.performance_stats['solver_ops'] += 1
        self.accelerator.performance_stats['total_time'] += elapsed_time
        
        return result


class GPUSolver:
    """GPU求解器"""
    
    def __init__(self, accelerator: CUDAAccelerator):
        self.accelerator = accelerator
        self.matrix_ops = GPUMatrixOperations(accelerator)
    
    def solve_sparse_system(self, A, b: np.ndarray) -> np.ndarray:
        if not self.accelerator.is_available():
            from scipy.sparse.linalg import spsolve
            return spsolve(A, b)
        
        start_time = time.time()
        
        if HAS_CUPY:
            if hasattr(A, 'tocsr'):
                A_csr = A.tocsr()
                A_gpu = cp.sparse.csr_matrix((A_csr.data, A_csr.indices, A_csr.indptr), shape=A_csr.shape)
            else:
                A_gpu = cp.asarray(A)
            
            b_gpu = cp.asarray(b)
            x_gpu = cp.sparse.linalg.cg(A_gpu, b_gpu, tol=1e-6)[0]
            result = cp.asnumpy(x_gpu)
            
            del A_gpu, b_gpu, x_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            from scipy.sparse.linalg import spsolve
            result = spsolve(A, b)
        
        elapsed_time = time.time() - start_time
        self.accelerator.performance_stats['solver_ops'] += 1
        self.accelerator.performance_stats['total_time'] += elapsed_time
        
        return result
    
    def get_performance_stats(self) -> dict:
        stats = self.accelerator.performance_stats.copy()
        stats['memory_info'] = self.accelerator.get_memory_info()
        return stats


def create_cuda_accelerator(device_id: int = 0) -> CUDAAccelerator:
    return CUDAAccelerator(device_id=device_id)


def demo_cuda_acceleration():
    """演示CUDA加速功能"""
    print("🚀 CUDA加速演示")
    print("=" * 50)
    
    try:
        accelerator = create_cuda_accelerator()
        
        if not accelerator.is_available():
            print("❌ CUDA不可用，请检查GPU和CUDA安装")
            return
        
        matrix_ops = GPUMatrixOperations(accelerator)
        solver = GPUSolver(accelerator)
        
        # 测试矩阵乘法
        print("📊 测试矩阵乘法...")
        size = 1000
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)
        
        start_time = time.time()
        C_cpu = np.dot(A, B)
        cpu_time = time.time() - start_time
        
        start_time = time.time()
        C_gpu = matrix_ops.matrix_multiply(A, B)
        gpu_time = time.time() - start_time
        
        error = np.linalg.norm(C_cpu - C_gpu) / np.linalg.norm(C_cpu)
        
        print(f"   CPU时间: {cpu_time:.4f} 秒")
        print(f"   GPU时间: {gpu_time:.4f} 秒")
        print(f"   加速比: {cpu_time / gpu_time:.2f}x")
        print(f"   误差: {error:.2e}")
        
        # 测试线性系统求解
        print("\n🔧 测试线性系统求解...")
        A = np.random.rand(500, 500).astype(np.float32)
        A = A + A.T + 500 * np.eye(500)
        b = np.random.rand(500).astype(np.float32)
        
        start_time = time.time()
        x_cpu = np.linalg.solve(A, b)
        cpu_time = time.time() - start_time
        
        start_time = time.time()
        x_gpu = solver.solve_linear_system(A, b)
        gpu_time = time.time() - start_time
        
        error = np.linalg.norm(x_cpu - x_gpu) / np.linalg.norm(x_cpu)
        
        print(f"   CPU时间: {cpu_time:.4f} 秒")
        print(f"   GPU时间: {gpu_time:.4f} 秒")
        print(f"   加速比: {cpu_time / gpu_time:.2f}x")
        print(f"   误差: {error:.2e}")
        
        # 显示性能统计
        print("\n📈 性能统计:")
        stats = solver.get_performance_stats()
        print(f"   矩阵运算次数: {stats['matrix_ops']}")
        print(f"   求解器运算次数: {stats['solver_ops']}")
        print(f"   总计算时间: {stats['total_time']:.4f} 秒")
        print(f"   GPU内存使用: {stats['memory_info']['used'] / 1024**3:.2f} GB")
        
        print("\n✅ CUDA加速演示完成!")
        
    except Exception as e:
        print(f"❌ CUDA加速演示失败: {e}")


if __name__ == "__main__":
    demo_cuda_acceleration() 