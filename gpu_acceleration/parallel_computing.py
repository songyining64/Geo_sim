"""
并行计算模块
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import warnings

# MPI相关依赖
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None

try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class MPIManager:
    """MPI管理器"""
    
    def __init__(self):
        if not HAS_MPI:
            raise ImportError("需要安装mpi4py来使用并行计算")
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.is_root = self.rank == 0
        
        print(f"🔄 MPI进程 {self.rank}/{self.size} 初始化完成")
    
    def get_rank(self) -> int:
        """获取当前进程排名"""
        return self.rank
    
    def get_size(self) -> int:
        """获取总进程数"""
        return self.size
    
    def is_root_process(self) -> bool:
        """是否为根进程"""
        return self.is_root
    
    def barrier(self):
        """同步所有进程"""
        self.comm.Barrier()
    
    def broadcast(self, data, root: int = 0):
        """广播数据"""
        return self.comm.bcast(data, root=root)
    
    def gather(self, data, root: int = 0):
        """收集数据"""
        return self.comm.gather(data, root=root)
    
    def scatter(self, data, root: int = 0):
        """分发数据"""
        return self.comm.scatter(data, root=root)
    
    def reduce(self, data, op=MPI.SUM, root: int = 0):
        """归约操作"""
        return self.comm.reduce(data, op=op, root=root)
    
    def allreduce(self, data, op=MPI.SUM):
        """全局归约操作"""
        return self.comm.allreduce(data, op=op)


class DomainDecomposition:
    """域分解"""
    
    def __init__(self, mpi_manager: MPIManager):
        self.mpi = mpi_manager
        self.local_domain = None
        self.global_domain = None
    
    def decompose_1d(self, n_points: int) -> Tuple[int, int, int]:
        """
        1D域分解
        
        Args:
            n_points: 总点数
            
        Returns:
            start_idx: 起始索引
            end_idx: 结束索引
            local_size: 本地大小
        """
        rank = self.mpi.get_rank()
        size = self.mpi.get_size()
        
        # 计算每个进程的点数
        points_per_process = n_points // size
        remainder = n_points % size
        
        # 分配起始和结束索引
        start_idx = rank * points_per_process + min(rank, remainder)
        end_idx = start_idx + points_per_process + (1 if rank < remainder else 0)
        local_size = end_idx - start_idx
        
        return start_idx, end_idx, local_size
    
    def decompose_2d(self, nx: int, ny: int) -> Tuple[int, int, int, int]:
        """
        2D域分解
        
        Args:
            nx: x方向点数
            ny: y方向点数
            
        Returns:
            x_start, x_end, y_start, y_end: 本地域边界
        """
        rank = self.mpi.get_rank()
        size = self.mpi.get_size()
        
        # 简化的2D分解（按行分解）
        rows_per_process = ny // size
        remainder = ny % size
        
        y_start = rank * rows_per_process + min(rank, remainder)
        y_end = y_start + rows_per_process + (1 if rank < remainder else 0)
        
        return 0, nx, y_start, y_end
    
    def create_local_mesh(self, global_mesh, decomposition_info):
        """创建本地网格"""
        if len(decomposition_info) == 3:  # 1D
            start_idx, end_idx, local_size = decomposition_info
            local_coords = global_mesh.coordinates[start_idx:end_idx]
        else:  # 2D
            x_start, x_end, y_start, y_end = decomposition_info
            # 简化的2D网格提取
            local_coords = global_mesh.coordinates[y_start:y_end, x_start:x_end]
        
        return local_coords


class ParallelSolver:
    """并行求解器"""
    
    def __init__(self, mpi_manager: MPIManager):
        self.mpi = mpi_manager
        self.domain_decomp = DomainDecomposition(mpi_manager)
        self.performance_stats = {
            'solve_time': 0.0,
            'communication_time': 0.0,
            'iterations': 0
        }
    
    def solve_parallel_linear_system(self, A, b: np.ndarray, 
                                   max_iterations: int = 1000,
                                   tolerance: float = 1e-6) -> np.ndarray:
        """
        并行求解线性系统
        
        Args:
            A: 系数矩阵
            b: 右端向量
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            
        Returns:
            x: 解向量
        """
        if not HAS_SCIPY:
            raise ImportError("需要安装scipy来使用并行求解器")
        
        start_time = time.time()
        
        # 域分解
        n_points = len(b)
        decomposition_info = self.domain_decomp.decompose_1d(n_points)
        start_idx, end_idx, local_size = decomposition_info
        
        # 提取本地数据
        local_b = b[start_idx:end_idx]
        
        # 提取本地矩阵（简化处理）
        if hasattr(A, 'tocsr'):
            A_csr = A.tocsr()
            local_A = A_csr[start_idx:end_idx, :]
        else:
            local_A = A[start_idx:end_idx, :]
        
        # 本地求解
        local_x = spsolve(local_A, local_b)
        
        # 收集所有进程的解
        all_x = self.mpi.gather(local_x, root=0)
        
        if self.mpi.is_root_process():
            # 根进程合并解
            x = np.concatenate(all_x)
        else:
            x = None
        
        # 广播解到所有进程
        x = self.mpi.broadcast(x, root=0)
        
        # 更新性能统计
        solve_time = time.time() - start_time
        self.performance_stats['solve_time'] += solve_time
        self.performance_stats['iterations'] += 1
        
        return x
    
    def solve_jacobi_parallel(self, A, b: np.ndarray,
                            max_iterations: int = 1000,
                            tolerance: float = 1e-6) -> np.ndarray:
        """
        并行Jacobi迭代求解
        
        Args:
            A: 系数矩阵
            b: 右端向量
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            
        Returns:
            x: 解向量
        """
        start_time = time.time()
        
        # 域分解
        n_points = len(b)
        decomposition_info = self.domain_decomp.decompose_1d(n_points)
        start_idx, end_idx, local_size = decomposition_info
        
        # 初始化解向量
        x = np.zeros(n_points)
        local_x = x[start_idx:end_idx]
        local_b = b[start_idx:end_idx]
        
        # 提取本地矩阵
        if hasattr(A, 'tocsr'):
            A_csr = A.tocsr()
            local_A = A_csr[start_idx:end_idx, :]
        else:
            local_A = A[start_idx:end_idx, :]
        
        # Jacobi迭代
        for iteration in range(max_iterations):
            # 保存前一次迭代结果
            x_old = local_x.copy()
            
            # 本地Jacobi更新
            for i in range(local_size):
                global_i = start_idx + i
                sum_ax = 0.0
                diag = 0.0
                
                for j in range(n_points):
                    if j != global_i:
                        sum_ax += local_A[i, j] * x[j]
                    else:
                        diag = local_A[i, j]
                
                if abs(diag) > 1e-12:
                    local_x[i] = (local_b[i] - sum_ax) / diag
            
            # 更新全局解向量
            all_x = self.mpi.allgather(local_x)
            x = np.concatenate(all_x)
            
            # 检查收敛性
            local_residual = np.linalg.norm(local_x - x_old)
            global_residual = self.mpi.allreduce(local_residual, op=MPI.SUM)
            
            if global_residual < tolerance:
                break
        
        # 更新性能统计
        solve_time = time.time() - start_time
        self.performance_stats['solve_time'] += solve_time
        self.performance_stats['iterations'] += iteration + 1
        
        return x
    
    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        stats['rank'] = self.mpi.get_rank()
        stats['size'] = self.mpi.get_size()
        return stats


def create_parallel_solver() -> ParallelSolver:
    """创建并行求解器"""
    mpi_manager = MPIManager()
    return ParallelSolver(mpi_manager)


def demo_parallel_computing():
    """演示并行计算功能"""
    print("🔄 并行计算演示")
    print("=" * 50)
    
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
        
        print("\n✅ 并行计算演示完成!")
        
    except Exception as e:
        print(f"❌ 并行计算演示失败: {e}")


if __name__ == "__main__":
    demo_parallel_computing() 