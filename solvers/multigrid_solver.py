"""
多重网格求解器实现
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


@dataclass
class MultigridConfig:
    """多重网格配置"""
    max_levels: int = 10  # 最大层数
    coarse_solver: str = 'direct'  # 粗网格求解器类型
    smoother: str = 'jacobi'  # 平滑器类型
    pre_smoothing_steps: int = 2  # 预平滑步数
    post_smoothing_steps: int = 2  # 后平滑步数
    tolerance: float = 1e-8  # 收敛容差
    max_iterations: int = 100  # 最大迭代次数
    cycle_type: str = 'v'  # 循环类型：'v', 'w', 'f'


class BaseMultigridSolver(ABC):
    """多重网格求解器基类"""
    
    def __init__(self, config: MultigridConfig = None):
        self.config = config or MultigridConfig()
        self.levels = []
        self.is_setup = False
    
    @abstractmethod
    def setup(self, A: sp.spmatrix, b: np.ndarray) -> None:
        """设置多重网格层次"""
        pass
    
    @abstractmethod
    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        """求解线性系统"""
        pass
    
    @abstractmethod
    def v_cycle(self, level: int, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """V-cycle算法"""
        pass


class AlgebraicMultigridSolver(BaseMultigridSolver):
    """代数多重网格求解器"""
    
    def __init__(self, config: MultigridConfig = None):
        super().__init__(config)
        self.interpolation_operators = []
        self.restriction_operators = []
        self.coarse_matrices = []
    
    def setup(self, A: sp.spmatrix, b: np.ndarray) -> None:
        """设置AMG层次"""
        print("🔄 设置代数多重网格...")
        
        self.levels = []
        self.interpolation_operators = []
        self.restriction_operators = []
        self.coarse_matrices = []
        
        current_A = A.copy()
        current_level = 0
        
        while current_level < self.config.max_levels and current_A.shape[0] > 10:
            # 创建当前层次
            level_data = {
                'matrix': current_A,
                'size': current_A.shape[0],
                'level': current_level
            }
            self.levels.append(level_data)
            
            # 检查是否可以继续粗化
            if current_A.shape[0] <= 10:
                break
            
            # 构建插值算子
            P = self._build_interpolation_operator(current_A)
            R = P.T  # 限制算子为插值算子的转置
            
            # 计算粗网格矩阵
            coarse_A = R @ current_A @ P
            
            # 存储算子
            self.interpolation_operators.append(P)
            self.restriction_operators.append(R)
            self.coarse_matrices.append(coarse_A)
            
            # 更新到下一层
            current_A = coarse_A
            current_level += 1
        
        # 添加最粗层
        if current_A.shape[0] > 0:
            level_data = {
                'matrix': current_A,
                'size': current_A.shape[0],
                'level': current_level
            }
            self.levels.append(level_data)
        
        self.is_setup = True
        print(f"✅ AMG设置完成，共 {len(self.levels)} 层")
    
    def _build_interpolation_operator(self, A: sp.spmatrix) -> sp.spmatrix:
        """构建插值算子"""
        n = A.shape[0]
        
        # 使用简单的代数多重网格策略
        # 这里实现一个简化的插值算子构建方法
        
        # 1. 选择粗网格点（使用强连接性）
        coarse_points = self._select_coarse_points(A)
        fine_points = list(set(range(n)) - set(coarse_points))
        
        # 2. 构建插值算子
        P_data = []
        P_rows = []
        P_cols = []
        
        # 粗网格点：单位插值
        for i, coarse_point in enumerate(coarse_points):
            P_data.append(1.0)
            P_rows.append(coarse_point)
            P_cols.append(i)
        
        # 细网格点：基于强连接的插值
        for fine_point in fine_points:
            # 找到与细网格点强连接的粗网格点
            strong_connections = self._find_strong_connections(A, fine_point, coarse_points)
            
            if strong_connections:
                # 计算插值权重
                weights = self._compute_interpolation_weights(A, fine_point, strong_connections)
                
                for coarse_idx, weight in zip(strong_connections, weights):
                    P_data.append(weight)
                    P_rows.append(fine_point)
                    P_cols.append(coarse_idx)
        
        # 构建稀疏矩阵
        P = sp.csr_matrix((P_data, (P_rows, P_cols)), shape=(n, len(coarse_points)))
        
        return P
    
    def _select_coarse_points(self, A: sp.spmatrix) -> List[int]:
        """选择粗网格点"""
        n = A.shape[0]
        
        # 使用简单的最大独立集策略
        # 这里可以实现更复杂的粗化策略
        
        # 计算每个点的连接数
        degrees = np.array(A.sum(axis=1)).flatten()
        
        # 选择度数最大的点作为粗网格点
        coarse_points = []
        marked = np.zeros(n, dtype=bool)
        
        while len(coarse_points) < n // 2:
            # 找到未标记的度数最大的点
            unmarked = np.where(~marked)[0]
            if len(unmarked) == 0:
                break
            
            max_degree_idx = unmarked[np.argmax(degrees[unmarked])]
            coarse_points.append(max_degree_idx)
            marked[max_degree_idx] = True
            
            # 标记相邻点
            neighbors = A[max_degree_idx].nonzero()[1]
            marked[neighbors] = True
        
        return coarse_points
    
    def _find_strong_connections(self, A: sp.spmatrix, point: int, coarse_points: List[int]) -> List[int]:
        """找到强连接"""
        # 简化的强连接定义：直接相邻的粗网格点
        neighbors = A[point].nonzero()[1]
        strong_connections = [i for i in neighbors if i in coarse_points]
        
        return strong_connections
    
    def _compute_interpolation_weights(self, A: sp.spmatrix, fine_point: int, 
                                     coarse_points: List[int]) -> List[float]:
        """计算插值权重"""
        # 简化的权重计算：等权重
        n_connections = len(coarse_points)
        if n_connections > 0:
            weights = [1.0 / n_connections] * n_connections
        else:
            weights = [1.0]
        
        return weights
    
    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        """求解线性系统"""
        if not self.is_setup:
            self.setup(A, b)
        
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        
        # 多重网格迭代
        for iteration in range(self.config.max_iterations):
            # 执行V-cycle
            x = self.v_cycle(0, b, x)
            
            # 检查收敛性
            residual = b - A @ x
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm < self.config.tolerance:
                print(f"✅ AMG收敛，迭代次数: {iteration + 1}")
                break
        
        return x
    
    def v_cycle(self, level: int, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """V-cycle算法"""
        if level >= len(self.levels) - 1:
            # 最粗层：直接求解
            A_coarse = self.levels[level]['matrix']
            x_coarse = self._solve_coarse_system(A_coarse, b)
            return x_coarse
        
        A = self.levels[level]['matrix']
        
        # 预平滑
        for _ in range(self.config.pre_smoothing_steps):
            x = self._smooth(A, b, x)
        
        # 计算残差
        residual = b - A @ x
        
        # 限制到粗网格
        if level < len(self.restriction_operators):
            R = self.restriction_operators[level]
            coarse_residual = R @ residual
            
            # 在粗网格上求解
            coarse_error = self.v_cycle(level + 1, coarse_residual, np.zeros_like(coarse_residual))
            
            # 插值回细网格
            P = self.interpolation_operators[level]
            error = P @ coarse_error
            
            # 修正解
            x = x + error
        
        # 后平滑
        for _ in range(self.config.post_smoothing_steps):
            x = self._smooth(A, b, x)
        
        return x
    
    def _smooth(self, A: sp.spmatrix, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """平滑器"""
        if self.config.smoother == 'jacobi':
            return self._jacobi_smooth(A, b, x)
        elif self.config.smoother == 'gauss_seidel':
            return self._gauss_seidel_smooth(A, b, x)
        else:
            return self._jacobi_smooth(A, b, x)
    
    def _jacobi_smooth(self, A: sp.spmatrix, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Jacobi平滑器"""
        D = sp.diags(A.diagonal())
        D_inv = sp.diags(1.0 / A.diagonal())
        L_plus_U = A - D
        
        x_new = D_inv @ (b - L_plus_U @ x)
        return x_new
    
    def _gauss_seidel_smooth(self, A: sp.spmatrix, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Gauss-Seidel平滑器"""
        A_dense = A.toarray()
        x_new = x.copy()
        
        for i in range(len(x)):
            x_new[i] = (b[i] - np.dot(A_dense[i, :i], x_new[:i]) - 
                       np.dot(A_dense[i, i+1:], x[i+1:])) / A_dense[i, i]
        
        return x_new
    
    def _solve_coarse_system(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        """求解粗网格系统"""
        if self.config.coarse_solver == 'direct':
            return sp.linalg.spsolve(A, b)
        else:
            return sp.linalg.cg(A, b)[0]


class GeometricMultigridSolver(BaseMultigridSolver):
    """几何多重网格求解器"""
    
    def __init__(self, config: MultigridConfig = None):
        super().__init__(config)
        self.meshes = []
        self.interpolation_operators = []
        self.restriction_operators = []
    
    def setup(self, A: sp.spmatrix, b: np.ndarray, mesh_data: Dict = None) -> None:
        """设置GMG层次"""
        print("🔄 设置几何多重网格...")
        
        if mesh_data is None:
            raise ValueError("几何多重网格需要网格数据")
        
        self.levels = []
        self.meshes = []
        self.interpolation_operators = []
        self.restriction_operators = []
        
        # 构建网格层次
        current_mesh = mesh_data
        current_level = 0
        
        while current_level < self.config.max_levels:
            # 存储当前层
            level_data = {
                'matrix': A,  # 这里应该根据网格重新组装矩阵
                'mesh': current_mesh,
                'size': len(current_mesh.get('nodes', [])),
                'level': current_level
            }
            self.levels.append(level_data)
            self.meshes.append(current_mesh)
            
            # 检查是否可以继续粗化
            if len(current_mesh.get('nodes', [])) <= 10:
                break
            
            # 粗化网格
            coarse_mesh = self._coarsen_mesh(current_mesh)
            
            # 构建插值和限制算子
            P, R = self._build_geometric_operators(current_mesh, coarse_mesh)
            
            self.interpolation_operators.append(P)
            self.restriction_operators.append(R)
            
            # 更新到下一层
            current_mesh = coarse_mesh
            current_level += 1
        
        self.is_setup = True
        print(f"✅ GMG设置完成，共 {len(self.levels)} 层")
    
    def _coarsen_mesh(self, mesh: Dict) -> Dict:
        """粗化网格"""
        # 简化的网格粗化策略
        # 这里可以实现更复杂的网格粗化算法
        
        nodes = mesh.get('nodes', [])
        elements = mesh.get('elements', [])
        
        if len(nodes) <= 10:
            return mesh
        
        # 选择粗网格点（每隔一个点选择一个）
        coarse_nodes = nodes[::2]
        coarse_elements = []
        
        # 重新编号元素
        node_mapping = {i: j for j, i in enumerate(range(0, len(nodes), 2))}
        
        for element in elements:
            element_nodes = element.get('nodes', [])
            coarse_element_nodes = []
            
            for node in element_nodes:
                if node in node_mapping:
                    coarse_element_nodes.append(node_mapping[node])
            
            if len(coarse_element_nodes) >= 3:  # 至少需要3个节点
                coarse_elements.append({'nodes': coarse_element_nodes})
        
        return {
            'nodes': coarse_nodes,
            'elements': coarse_elements
        }
    
    def _build_geometric_operators(self, fine_mesh: Dict, coarse_mesh: Dict) -> Tuple[sp.spmatrix, sp.spmatrix]:
        """构建几何插值和限制算子"""
        fine_nodes = fine_mesh.get('nodes', [])
        coarse_nodes = coarse_mesh.get('nodes', [])
        
        n_fine = len(fine_nodes)
        n_coarse = len(coarse_nodes)
        
        # 简化的插值算子：线性插值
        P_data = []
        P_rows = []
        P_cols = []
        
        for i, fine_node in enumerate(fine_nodes):
            # 找到最近的粗网格点
            distances = [np.linalg.norm(np.array(fine_node) - np.array(coarse_node)) 
                        for coarse_node in coarse_nodes]
            nearest_coarse = np.argmin(distances)
            
            P_data.append(1.0)
            P_rows.append(i)
            P_cols.append(nearest_coarse)
        
        P = sp.csr_matrix((P_data, (P_rows, P_cols)), shape=(n_fine, n_coarse))
        R = P.T  # 限制算子为插值算子的转置
        
        return P, R
    
    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray = None, 
              mesh_data: Dict = None) -> np.ndarray:
        """求解线性系统"""
        if not self.is_setup:
            self.setup(A, b, mesh_data)
        
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        
        # 多重网格迭代
        for iteration in range(self.config.max_iterations):
            # 执行V-cycle
            x = self.v_cycle(0, b, x)
            
            # 检查收敛性
            residual = b - A @ x
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm < self.config.tolerance:
                print(f"✅ GMG收敛，迭代次数: {iteration + 1}")
                break
        
        return x
    
    def v_cycle(self, level: int, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """V-cycle算法"""
        if level >= len(self.levels) - 1:
            # 最粗层：直接求解
            A_coarse = self.levels[level]['matrix']
            x_coarse = self._solve_coarse_system(A_coarse, b)
            return x_coarse
        
        A = self.levels[level]['matrix']
        
        # 预平滑
        for _ in range(self.config.pre_smoothing_steps):
            x = self._smooth(A, b, x)
        
        # 计算残差
        residual = b - A @ x
        
        # 限制到粗网格
        if level < len(self.restriction_operators):
            R = self.restriction_operators[level]
            coarse_residual = R @ residual
            
            # 在粗网格上求解
            coarse_error = self.v_cycle(level + 1, coarse_residual, np.zeros_like(coarse_residual))
            
            # 插值回细网格
            P = self.interpolation_operators[level]
            error = P @ coarse_error
            
            # 修正解
            x = x + error
        
        # 后平滑
        for _ in range(self.config.post_smoothing_steps):
            x = self._smooth(A, b, x)
        
        return x
    
    def _smooth(self, A: sp.spmatrix, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """平滑器"""
        if self.config.smoother == 'jacobi':
            return self._jacobi_smooth(A, b, x)
        elif self.config.smoother == 'gauss_seidel':
            return self._gauss_seidel_smooth(A, b, x)
        else:
            return self._jacobi_smooth(A, b, x)
    
    def _jacobi_smooth(self, A: sp.spmatrix, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Jacobi平滑器"""
        D = sp.diags(A.diagonal())
        D_inv = sp.diags(1.0 / A.diagonal())
        L_plus_U = A - D
        
        x_new = D_inv @ (b - L_plus_U @ x)
        return x_new
    
    def _gauss_seidel_smooth(self, A: sp.spmatrix, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Gauss-Seidel平滑器"""
        A_dense = A.toarray()
        x_new = x.copy()
        
        for i in range(len(x)):
            x_new[i] = (b[i] - np.dot(A_dense[i, :i], x_new[:i]) - 
                       np.dot(A_dense[i, i+1:], x[i+1:])) / A_dense[i, i]
        
        return x_new
    
    def _solve_coarse_system(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        """求解粗网格系统"""
        if self.config.coarse_solver == 'direct':
            return sp.linalg.spsolve(A, b)
        else:
            return sp.linalg.cg(A, b)[0]


# 工厂函数
def create_multigrid_solver(solver_type: str = 'amg', config: MultigridConfig = None) -> BaseMultigridSolver:
    """创建多重网格求解器"""
    if solver_type == 'amg':
        return AlgebraicMultigridSolver(config)
    elif solver_type == 'gmg':
        return GeometricMultigridSolver(config)
    else:
        raise ValueError(f"不支持的多重网格求解器类型: {solver_type}")


def create_multigrid_config(**kwargs) -> MultigridConfig:
    """创建多重网格配置"""
    return MultigridConfig(**kwargs)
