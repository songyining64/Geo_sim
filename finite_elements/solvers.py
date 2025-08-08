"""
高性能求解器模块
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
import time

# 可选依赖
try:
    from scipy.sparse import csr_matrix, lil_matrix
    from scipy.sparse.linalg import spsolve, gmres, cg, bicgstab, spilu
    from scipy.linalg import solve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    spsolve = gmres = cg = bicgstab = spilu = solve = None

try:
    from petsc4py import PETSc
    HAS_PETSC = True
except ImportError:
    HAS_PETSC = False
    PETSc = None

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


@dataclass
class SolverConfig:
    """求解器配置"""
    solver_type: str = "direct"  # 'direct', 'iterative', 'multigrid'
    method: str = "lu"  # 'lu', 'gmres', 'cg', 'bicgstab', 'amg'
    tolerance: float = 1e-8
    max_iterations: int = 1000
    preconditioner: str = "ilu"  # 'ilu', 'jacobi', 'amg', 'none'
    verbose: bool = False
    use_gpu: bool = False
    parallel: bool = False


class LinearSolver(ABC):
    """线性求解器抽象基类"""
    
    @abstractmethod
    def solve(self, A: Union[np.ndarray, csr_matrix], 
              b: np.ndarray) -> np.ndarray:
        """求解线性系统 Ax = b"""
        pass
    
    @abstractmethod
    def setup(self, A: Union[np.ndarray, csr_matrix]):
        """设置求解器（如预处理器）"""
        pass


class DirectSolver(LinearSolver):
    """直接求解器"""
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig(solver_type="direct")
        self.factorized = None
        
    def setup(self, A: Union[np.ndarray, csr_matrix]):
        """LU分解"""
        if self.config.method == "lu":
            if HAS_SCIPY and isinstance(A, csr_matrix):
                self.factorized = spilu(A.tocsc())
            else:
                self.factorized = np.linalg.lu_factor(A)
        else:
            raise ValueError(f"不支持的直接求解方法: {self.config.method}")
    
    def solve(self, A: Union[np.ndarray, csr_matrix], 
              b: np.ndarray) -> np.ndarray:
        """求解"""
        if self.factorized is None:
            self.setup(A)
        
        if self.config.method == "lu":
            if HAS_SCIPY and isinstance(A, csr_matrix):
                return self.factorized.solve(b)
            else:
                return solve(self.factorized, b)
        else:
            return spsolve(A, b)


class IterativeSolver(LinearSolver):
    """迭代求解器"""
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig(solver_type="iterative")
        self.preconditioner = None
        
    def setup(self, A: Union[np.ndarray, csr_matrix]):
        """设置预处理器"""
        if self.config.preconditioner == "ilu" and HAS_SCIPY:
            self.preconditioner = spilu(A.tocsc())
        elif self.config.preconditioner == "jacobi":
            # 雅可比预处理器
            diag = A.diagonal()
            self.preconditioner = lambda x: x / diag
        else:
            self.preconditioner = None
    
    def solve(self, A: Union[np.ndarray, csr_matrix], 
              b: np.ndarray) -> np.ndarray:
        """迭代求解"""
        if self.preconditioner is None:
            self.setup(A)
        
        if self.config.method == "gmres":
            return gmres(A, b, tol=self.config.tolerance, 
                        maxiter=self.config.max_iterations,
                        M=self.preconditioner)[0]
        elif self.config.method == "cg":
            return cg(A, b, tol=self.config.tolerance,
                     maxiter=self.config.max_iterations,
                     M=self.preconditioner)[0]
        elif self.config.method == "bicgstab":
            return bicgstab(A, b, tol=self.config.tolerance,
                           maxiter=self.config.max_iterations,
                           M=self.preconditioner)[0]
        else:
            raise ValueError(f"不支持的迭代方法: {self.config.method}")


class PETScSolver(LinearSolver):
    """PETSc求解器 - Underworld风格"""
    
    def __init__(self, config: SolverConfig = None):
        if not HAS_PETSC:
            raise ImportError("PETSc is required for PETScSolver")
        
        self.config = config or SolverConfig(solver_type="petsc")
        self.ksp = None
        self.pc = None
        
    def setup(self, A: Union[np.ndarray, csr_matrix]):
        """设置PETSc求解器"""
        # 创建KSP求解器
        self.ksp = PETSc.KSP().create()
        self.ksp.setType(self.config.method)
        self.ksp.setTolerances(rtol=self.config.tolerance, 
                              atol=self.config.tolerance,
                              max_it=self.config.max_iterations)
        
        # 设置预处理器
        self.pc = self.ksp.getPC()
        self.pc.setType(self.config.preconditioner)
        
        # 设置矩阵
        if isinstance(A, csr_matrix):
            petsc_A = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        else:
            petsc_A = PETSc.Mat().createDense(size=A.shape, array=A)
        
        self.ksp.setOperators(petsc_A)
        self.ksp.setUp()
        
    def solve(self, A: Union[np.ndarray, csr_matrix], 
              b: np.ndarray) -> np.ndarray:
        """PETSc求解"""
        if self.ksp is None:
            self.setup(A)
        
        # 创建PETSc向量
        petsc_b = PETSc.Vec().createWithArray(b)
        petsc_x = PETSc.Vec().createWithArray(np.zeros_like(b))
        
        # 求解
        self.ksp.solve(petsc_b, petsc_x)
        
        return petsc_x.getArray()


class MultigridSolver(LinearSolver):
    """多重网格求解器 - Underworld风格"""
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig(solver_type="multigrid")
        self.levels = []
        self.interpolators = []
        self.restrictors = []
        
    def setup(self, A: Union[np.ndarray, csr_matrix]):
        """设置多重网格层次"""
        # 这里需要实现网格粗化策略
        # 简化实现：使用几何多重网格
        self._setup_geometric_multigrid(A)
        
    def _setup_geometric_multigrid(self, A: Union[np.ndarray, csr_matrix]):
        """设置几何多重网格"""
        # 简化实现：假设已经有多层网格
        pass
        
    def solve(self, A: Union[np.ndarray, csr_matrix], 
              b: np.ndarray) -> np.ndarray:
        """多重网格求解"""
        if not self.levels:
            self.setup(A)
        
        x = np.zeros_like(b)
        
        # V循环多重网格
        for _ in range(self.config.max_iterations):
            x = self._v_cycle(A, b, x)
            
            # 检查收敛
            residual = np.linalg.norm(b - A @ x)
            if residual < self.config.tolerance:
                break
                
        return x
    
    def _v_cycle(self, A: Union[np.ndarray, csr_matrix], 
                 b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """V循环"""
        # 前平滑
        x = self._smooth(A, b, x, pre_smooth=True)
        
        # 计算残差
        r = b - A @ x
        
        # 限制到粗网格
        r_coarse = self._restrict(r)
        
        # 粗网格求解
        if len(self.levels) > 1:
            e_coarse = self._solve_coarse(r_coarse)
        else:
            e_coarse = np.linalg.solve(self.levels[-1], r_coarse)
        
        # 插值到细网格
        e = self._interpolate(e_coarse)
        
        # 校正
        x = x + e
        
        # 后平滑
        x = self._smooth(A, b, x, pre_smooth=False)
        
        return x
    
    def _smooth(self, A: Union[np.ndarray, csr_matrix], 
                b: np.ndarray, x: np.ndarray, 
                pre_smooth: bool = True) -> np.ndarray:
        """平滑"""
        # 雅可比平滑
        D = np.diag(A.diagonal())
        L_plus_U = A - D
        
        for _ in range(2 if pre_smooth else 1):
            x = np.linalg.solve(D, b - L_plus_U @ x)
        
        return x
    
    def _restrict(self, r: np.ndarray) -> np.ndarray:
        """限制算子"""
        # 简化实现
        return r[::2]
    
    def _interpolate(self, e_coarse: np.ndarray) -> np.ndarray:
        """插值算子"""
        # 简化实现
        n = len(e_coarse)
        e = np.zeros(2 * n - 1)
        e[::2] = e_coarse
        e[1::2] = 0.5 * (e_coarse[:-1] + e_coarse[1:])
        return e
    
    def _solve_coarse(self, r_coarse: np.ndarray) -> np.ndarray:
        """粗网格求解"""
        # 递归调用V循环
        return self._v_cycle(self.levels[-1], r_coarse, np.zeros_like(r_coarse))


class SolverFactory:
    """求解器工厂 - Underworld风格"""
    
    @staticmethod
    def create_solver(config: SolverConfig) -> LinearSolver:
        """创建求解器"""
        if config.solver_type == "direct":
            return DirectSolver(config)
        elif config.solver_type == "iterative":
            return IterativeSolver(config)
        elif config.solver_type == "petsc":
            return PETScSolver(config)
        elif config.solver_type == "multigrid":
            return MultigridSolver(config)
        else:
            raise ValueError(f"不支持的求解器类型: {config.solver_type}")


class NonlinearSolver:
    """非线性求解器"""
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()
        self.linear_solver = SolverFactory.create_solver(config)
        
    def solve(self, residual_func: Callable, jacobian_func: Callable,
              x0: np.ndarray, **kwargs) -> np.ndarray:
        """非线性求解"""
        x = x0.copy()
        
        for iteration in range(self.config.max_iterations):
            # 计算残差
            r = residual_func(x)
            
            # 检查收敛
            residual_norm = np.linalg.norm(r)
            if self.config.verbose:
                print(f"Iteration {iteration}: residual = {residual_norm:.2e}")
            
            if residual_norm < self.config.tolerance:
                break
            
            # 计算雅可比矩阵
            J = jacobian_func(x)
            
            # 求解线性系统
            dx = self.linear_solver.solve(J, -r)
            
            # 线搜索
            alpha = self._line_search(residual_func, x, dx)
            
            # 更新解
            x = x + alpha * dx
        
        return x
    
    def _line_search(self, residual_func: Callable, 
                    x: np.ndarray, dx: np.ndarray) -> float:
        """线搜索"""
        alpha = 1.0
        r0 = residual_func(x)
        r0_norm = np.linalg.norm(r0)
        
        for i in range(10):
            x_new = x + alpha * dx
            r_new = residual_func(x_new)
            r_new_norm = np.linalg.norm(r_new)
            
            if r_new_norm < r0_norm:
                return alpha
            
            alpha *= 0.5
        
        return alpha


class TimeStepper:
    """时间步进器"""
    
    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()
        self.linear_solver = SolverFactory.create_solver(config)
        
    def solve_implicit(self, mass_matrix: Union[np.ndarray, csr_matrix],
                      stiffness_matrix: Union[np.ndarray, csr_matrix],
                      force_vector: np.ndarray, initial_condition: np.ndarray,
                      time_steps: int, dt: float) -> np.ndarray:
        """隐式时间步进"""
        n_dofs = len(initial_condition)
        solution = np.zeros((time_steps + 1, n_dofs))
        solution[0] = initial_condition
        
        # 隐式欧拉：M * (u^{n+1} - u^n) / dt = -K * u^{n+1} + f
        # 整理得：(M + dt * K) * u^{n+1} = M * u^n + dt * f
        
        A = mass_matrix + dt * stiffness_matrix
        
        for n in range(time_steps):
            b = mass_matrix @ solution[n] + dt * force_vector
            solution[n + 1] = self.linear_solver.solve(A, b)
        
        return solution
    
    def solve_explicit(self, mass_matrix: Union[np.ndarray, csr_matrix],
                      stiffness_matrix: Union[np.ndarray, csr_matrix],
                      force_vector: np.ndarray, initial_condition: np.ndarray,
                      time_steps: int, dt: float) -> np.ndarray:
        """显式时间步进"""
        n_dofs = len(initial_condition)
        solution = np.zeros((time_steps + 1, n_dofs))
        solution[0] = initial_condition
        
        # 显式欧拉：M * (u^{n+1} - u^n) / dt = -K * u^n + f
        # 整理得：u^{n+1} = u^n + dt * M^{-1} * (-K * u^n + f)
        
        # 预计算质量矩阵的逆
        if isinstance(mass_matrix, csr_matrix):
            M_inv = spsolve(mass_matrix, np.eye(n_dofs))
        else:
            M_inv = np.linalg.inv(mass_matrix)
        
        for n in range(time_steps):
            residual = -stiffness_matrix @ solution[n] + force_vector
            solution[n + 1] = solution[n] + dt * (M_inv @ residual)
        
        return solution


# 便捷函数
def solve_linear_system(A: Union[np.ndarray, csr_matrix], 
                       b: np.ndarray, 
                       config: SolverConfig = None) -> np.ndarray:
    """求解线性系统"""
    solver = SolverFactory.create_solver(config or SolverConfig())
    return solver.solve(A, b)

def solve_nonlinear_system(residual_func: Callable, 
                          jacobian_func: Callable,
                          x0: np.ndarray, 
                          config: SolverConfig = None) -> np.ndarray:
    """求解非线性系统"""
    solver = NonlinearSolver(config)
    return solver.solve(residual_func, jacobian_func, x0) 