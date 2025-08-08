"""
并行求解器模块 - 提供并行线性求解器
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time

# MPI相关依赖
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None


class ParallelSolver:
    """并行求解器基类"""
    
    def __init__(self, solver_type: str = 'cg'):
        self.solver_type = solver_type
        self.comm = MPI.COMM_WORLD if HAS_MPI else None
        self.rank = self.comm.Get_rank() if HAS_MPI else 0
        self.size = self.comm.Get_size() if HAS_MPI else 1
        self.solve_time = 0.0
        self.iterations = 0
        
    def solve(self, A: np.ndarray, b: np.ndarray, 
              partition_info: Dict = None) -> np.ndarray:
        """求解线性系统 Ax = b"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_solver_info(self) -> Dict:
        """获取求解器信息"""
        return {
            'solver_type': self.solver_type,
            'solve_time': self.solve_time,
            'iterations': self.iterations,
            'rank': self.rank,
            'size': self.size
        }


class ParallelCGSolver(ParallelSolver):
    """并行共轭梯度求解器"""
    
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6):
        super().__init__(solver_type='parallel_cg')
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def solve(self, A: np.ndarray, b: np.ndarray, 
              partition_info: Dict = None) -> np.ndarray:
        """并行共轭梯度求解"""
        start_time = time.time()
        
        # 获取局部数据
        local_size = len(b)
        x = np.zeros(local_size)
        r = b.copy()
        p = r.copy()
        
        # 计算初始残差范数
        r_norm_sq = np.dot(r, r)
        if HAS_MPI:
            global_r_norm_sq = self.comm.allreduce(r_norm_sq, op=MPI.SUM)
        else:
            global_r_norm_sq = r_norm_sq
        
        initial_residual = np.sqrt(global_r_norm_sq)
        residual = initial_residual
        
        # 迭代求解
        for iteration in range(self.max_iterations):
            # 计算 Ap
            Ap = self._matrix_vector_product(A, p, partition_info)
            
            # 计算 alpha
            pAp = np.dot(p, Ap)
            if HAS_MPI:
                global_pAp = self.comm.allreduce(pAp, op=MPI.SUM)
            else:
                global_pAp = pAp
            
            alpha = r_norm_sq / global_pAp
            
            # 更新解和残差
            x += alpha * p
            r -= alpha * Ap
            
            # 计算新的残差范数
            new_r_norm_sq = np.dot(r, r)
            if HAS_MPI:
                global_new_r_norm_sq = self.comm.allreduce(new_r_norm_sq, op=MPI.SUM)
            else:
                global_new_r_norm_sq = new_r_norm_sq
            
            residual = np.sqrt(global_new_r_norm_sq)
            
            # 检查收敛性
            if residual < self.tolerance * initial_residual:
                self.iterations = iteration + 1
                break
            
            # 计算 beta
            beta = global_new_r_norm_sq / r_norm_sq
            
            # 更新搜索方向
            p = r + beta * p
            r_norm_sq = global_new_r_norm_sq
        
        self.solve_time = time.time() - start_time
        self.iterations = iteration + 1
        
        return x
    
    def _matrix_vector_product(self, A: np.ndarray, x: np.ndarray, 
                              partition_info: Dict = None) -> np.ndarray:
        """矩阵向量乘积（考虑并行通信）"""
        # 局部矩阵向量乘积
        y = np.dot(A, x)
        
        # 如果有分区信息，处理通信
        if partition_info and HAS_MPI:
            y = self._handle_communication(y, partition_info)
        
        return y
    
    def _handle_communication(self, local_data: np.ndarray, 
                            partition_info: Dict) -> np.ndarray:
        """处理并行通信"""
        # 简化的通信处理
        # 实际实现需要根据具体的分区策略
        return local_data


class ParallelAMGSolver(ParallelSolver):
    """并行代数多重网格求解器"""
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        super().__init__(solver_type='parallel_amg')
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.coarsening_levels = 3
        
    def solve(self, A: np.ndarray, b: np.ndarray, 
              partition_info: Dict = None) -> np.ndarray:
        """并行AMG求解"""
        start_time = time.time()
        
        # 构建多重网格层次
        hierarchy = self._build_hierarchy(A, partition_info)
        
        # 初始解
        x = np.zeros(len(b))
        
        # V-cycle迭代
        for iteration in range(self.max_iterations):
            x = self._v_cycle(hierarchy, x, b, partition_info)
            
            # 检查收敛性
            residual = self._compute_residual(A, x, b, partition_info)
            if residual < self.tolerance:
                self.iterations = iteration + 1
                break
        
        self.solve_time = time.time() - start_time
        self.iterations = iteration + 1
        
        return x
    
    def _build_hierarchy(self, A: np.ndarray, partition_info: Dict) -> List[Dict]:
        """构建多重网格层次"""
        hierarchy = []
        current_A = A.copy()
        
        for level in range(self.coarsening_levels):
            # 粗化策略
            coarse_info = self._coarsen_matrix(current_A, partition_info)
            
            hierarchy.append({
                'matrix': current_A,
                'coarse_matrix': coarse_info['coarse_matrix'],
                'interpolation': coarse_info['interpolation'],
                'restriction': coarse_info['restriction']
            })
            
            current_A = coarse_info['coarse_matrix']
            
            # 如果矩阵太小，停止粗化
            if current_A.shape[0] < 10:
                break
        
        return hierarchy
    
    def _coarsen_matrix(self, A: np.ndarray, partition_info: Dict) -> Dict:
        """粗化矩阵"""
        n = A.shape[0]
        
        # 简化的粗化策略：选择强连接点
        strength_threshold = 0.1 * np.max(np.abs(A))
        strong_connections = np.abs(A) > strength_threshold
        
        # 选择粗网格点（C点）
        c_points = self._select_c_points(strong_connections)
        f_points = np.setdiff1d(np.arange(n), c_points)
        
        # 构建插值算子
        interpolation = self._build_interpolation(A, c_points, f_points)
        restriction = interpolation.T
        
        # 构建粗网格矩阵
        coarse_matrix = restriction @ A @ interpolation
        
        return {
            'coarse_matrix': coarse_matrix,
            'interpolation': interpolation,
            'restriction': restriction,
            'c_points': c_points,
            'f_points': f_points
        }
    
    def _select_c_points(self, strong_connections: np.ndarray) -> np.ndarray:
        """选择粗网格点"""
        n = strong_connections.shape[0]
        c_points = []
        marked = np.zeros(n, dtype=bool)
        
        # 使用最大独立集算法
        for i in range(n):
            if not marked[i]:
                c_points.append(i)
                marked[i] = True
                
                # 标记强连接的邻居
                neighbors = np.where(strong_connections[i])[0]
                marked[neighbors] = True
        
        return np.array(c_points)
    
    def _build_interpolation(self, A: np.ndarray, c_points: np.ndarray, 
                           f_points: np.ndarray) -> np.ndarray:
        """构建插值算子"""
        n = A.shape[0]
        nc = len(c_points)
        nf = len(f_points)
        
        interpolation = np.zeros((n, nc))
        
        # C点插值
        for i, c_point in enumerate(c_points):
            interpolation[c_point, i] = 1.0
        
        # F点插值
        for i, f_point in enumerate(f_points):
            # 找到强连接的C点
            strong_c_points = []
            for j, c_point in enumerate(c_points):
                if abs(A[f_point, c_point]) > 0.1 * np.max(np.abs(A[f_point])):
                    strong_c_points.append(j)
            
            if strong_c_points:
                # 计算插值权重
                weights = np.abs(A[f_point, c_points[strong_c_points]])
                weights = weights / np.sum(weights)
                
                for j, weight in zip(strong_c_points, weights):
                    interpolation[f_point, j] = weight
        
        return interpolation
    
    def _v_cycle(self, hierarchy: List[Dict], x: np.ndarray, b: np.ndarray, 
                 partition_info: Dict) -> np.ndarray:
        """V-cycle多重网格"""
        # 前向松弛
        x = self._relax(hierarchy[0]['matrix'], x, b, partition_info)
        
        # 计算残差
        residual = b - hierarchy[0]['matrix'] @ x
        
        # 限制到粗网格
        coarse_residual = hierarchy[0]['restriction'] @ residual
        
        # 递归求解粗网格问题
        if len(hierarchy) > 1:
            coarse_error = self._v_cycle(hierarchy[1:], 
                                       np.zeros_like(coarse_residual), 
                                       coarse_residual, partition_info)
        else:
            # 最粗网格直接求解
            coarse_error = np.linalg.solve(hierarchy[0]['coarse_matrix'], coarse_residual)
        
        # 插值回细网格
        error = hierarchy[0]['interpolation'] @ coarse_error
        x += error
        
        # 后向松弛
        x = self._relax(hierarchy[0]['matrix'], x, b, partition_info)
        
        return x
    
    def _relax(self, A: np.ndarray, x: np.ndarray, b: np.ndarray, 
               partition_info: Dict, num_sweeps: int = 2) -> np.ndarray:
        """松弛迭代"""
        n = len(x)
        
        for sweep in range(num_sweeps):
            for i in range(n):
                # Gauss-Seidel松弛
                residual = b[i] - np.dot(A[i], x)
                x[i] += residual / A[i, i]
        
        return x
    
    def _compute_residual(self, A: np.ndarray, x: np.ndarray, b: np.ndarray, 
                         partition_info: Dict) -> float:
        """计算残差范数"""
        residual = b - A @ x
        residual_norm = np.linalg.norm(residual)
        
        if HAS_MPI:
            global_residual_norm = self.comm.allreduce(residual_norm, op=MPI.SUM)
            return global_residual_norm
        else:
            return residual_norm


class ParallelSchwarzSolver(ParallelSolver):
    """并行Schwarz方法求解器"""
    
    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-6):
        super().__init__(solver_type='parallel_schwarz')
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def solve(self, A: np.ndarray, b: np.ndarray, 
              partition_info: Dict = None) -> np.ndarray:
        """并行Schwarz方法求解"""
        start_time = time.time()
        
        if partition_info is None:
            raise ValueError("Schwarz方法需要分区信息")
        
        # 初始解
        x = np.zeros(len(b))
        
        # 获取局部信息
        local_elements = partition_info['local_elements'].get(self.rank, [])
        boundary_nodes = partition_info['boundary_nodes'].get(self.rank, set())
        
        # 构建局部矩阵和右端项
        local_indices = list(local_elements) + list(boundary_nodes)
        local_A = A[np.ix_(local_indices, local_indices)]
        local_b = b[local_indices]
        
        # Schwarz迭代
        for iteration in range(self.max_iterations):
            # 局部求解
            local_x = np.linalg.solve(local_A, local_b)
            
            # 更新全局解
            x[local_indices] = local_x
            
            # 通信边界数据
            if HAS_MPI:
                x = self._communicate_boundary_data(x, partition_info)
            
            # 检查收敛性
            residual = self._compute_residual(A, x, b, partition_info)
            if residual < self.tolerance:
                self.iterations = iteration + 1
                break
        
        self.solve_time = time.time() - start_time
        self.iterations = iteration + 1
        
        return x
    
    def _communicate_boundary_data(self, x: np.ndarray, 
                                 partition_info: Dict) -> np.ndarray:
        """通信边界数据"""
        # 简化的边界通信
        # 实际实现需要根据具体的分区策略
        return x
    
    def _compute_residual(self, A: np.ndarray, x: np.ndarray, b: np.ndarray, 
                         partition_info: Dict) -> float:
        """计算残差范数"""
        residual = b - A @ x
        residual_norm = np.linalg.norm(residual)
        
        if HAS_MPI:
            global_residual_norm = self.comm.allreduce(residual_norm, op=MPI.SUM)
            return global_residual_norm
        else:
            return residual_norm


def create_parallel_solver(solver_type: str = 'cg', **kwargs) -> ParallelSolver:
    """创建并行求解器"""
    if solver_type == 'cg':
        return ParallelCGSolver(**kwargs)
    elif solver_type == 'amg':
        return ParallelAMGSolver(**kwargs)
    elif solver_type == 'schwarz':
        return ParallelSchwarzSolver(**kwargs)
    else:
        raise ValueError(f"不支持的求解器类型: {solver_type}")


def demo_parallel_solvers():
    """演示并行求解器功能"""
    print("🔧 并行求解器演示")
    print("=" * 50)
    
    # 创建测试矩阵
    n = 100
    A = np.random.rand(n, n)
    A = A + A.T + n * np.eye(n)  # 对称正定矩阵
    b = np.random.rand(n)
    
    # 创建简单的分区信息
    partition_info = {
        'local_elements': {0: list(range(n))},
        'boundary_nodes': {0: set()},
        'ghost_nodes': {0: set()}
    }
    
    # 测试不同的求解器
    solvers = {
        'parallel_cg': ParallelCGSolver(max_iterations=100, tolerance=1e-6),
        'parallel_amg': ParallelAMGSolver(max_iterations=10, tolerance=1e-6),
        'parallel_schwarz': ParallelSchwarzSolver(max_iterations=20, tolerance=1e-6)
    }
    
    for name, solver in solvers.items():
        print(f"\n🔧 测试 {name}...")
        
        try:
            x = solver.solve(A, b, partition_info)
            
            # 计算误差
            residual = np.linalg.norm(b - A @ x)
            relative_error = residual / np.linalg.norm(b)
            
            info = solver.get_solver_info()
            print(f"   求解时间: {info['solve_time']:.4f} 秒")
            print(f"   迭代次数: {info['iterations']}")
            print(f"   相对误差: {relative_error:.2e}")
            
        except Exception as e:
            print(f"   ❌ 求解失败: {e}")
    
    print("\n✅ 并行求解器演示完成!")


if __name__ == "__main__":
    demo_parallel_solvers() 