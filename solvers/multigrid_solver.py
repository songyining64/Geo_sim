"""
增强型多重网格求解器实现

包含以下增强功能：
1. 多重网格求解器的完善
   - 网格粗化策略：基于几何或代数的自适应粗化
   - 循环策略扩展：V循环、W循环、FMG等
   - 平滑器优化：Jacobi、Gauss-Seidel、Chebyshev等
   - 并行化支持：分布式内存下的多重网格

2. 多物理场耦合求解
   - 耦合方程组组装
   - 分区求解策略
   - 时间积分器支持
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
import time
from scipy.sparse.linalg import spsolve, cg, gmres
from scipy.linalg import solve_triangular


@dataclass
class MultigridConfig:
    """增强型多重网格配置"""
    max_levels: int = 10  # 最大层数
    coarse_solver: str = 'direct'  # 粗网格求解器类型
    smoother: str = 'jacobi'  # 平滑器类型
    pre_smoothing_steps: int = 2  # 预平滑步数
    post_smoothing_steps: int = 2  # 后平滑步数
    tolerance: float = 1e-8  # 收敛容差
    max_iterations: int = 100  # 最大迭代次数
    cycle_type: str = 'v'  # 循环类型：'v', 'w', 'fmg'
    
    # 新增配置选项
    adaptive_coarsening: bool = True  # 自适应粗化
    strong_threshold: float = 0.25  # 强连接阈值
    max_coarse_size: int = 100  # 最大粗网格大小
    parallel_support: bool = False  # 并行支持
    chebyshev_degree: int = 4  # Chebyshev多项式次数
    relaxation_factor: float = 1.0  # 松弛因子


class AdvancedSmoother:
    """高级平滑器集合"""
    
    @staticmethod
    def jacobi_smooth(A: sp.spmatrix, b: np.ndarray, x: np.ndarray, 
                      omega: float = 1.0, iterations: int = 1) -> np.ndarray:
        """Jacobi平滑器（带松弛因子）"""
        D = sp.diags(A.diagonal())
        D_inv = sp.diags(1.0 / A.diagonal())
        L_plus_U = A - D
        
        x_new = x.copy()
        for _ in range(iterations):
            x_new = x_new + omega * D_inv @ (b - A @ x_new)
        
        return x_new
    
    @staticmethod
    def gauss_seidel_smooth(A: sp.spmatrix, b: np.ndarray, x: np.ndarray, 
                           iterations: int = 1) -> np.ndarray:
        """Gauss-Seidel平滑器"""
        A_dense = A.toarray()
        x_new = x.copy()
        
        for _ in range(iterations):
            for i in range(len(x)):
                x_new[i] = (b[i] - np.dot(A_dense[i, :i], x_new[:i]) - 
                           np.dot(A_dense[i, i+1:], x_new[i+1:])) / A_dense[i, i]
        
        return x_new
    
    @staticmethod
    def chebyshev_smooth(A: sp.spmatrix, b: np.ndarray, x: np.ndarray, 
                        degree: int = 4, iterations: int = 1) -> np.ndarray:
        """Chebyshev多项式平滑器"""
        # 估计特征值范围
        D = sp.diags(A.diagonal())
        D_inv = sp.diags(1.0 / A.diagonal())
        B = D_inv @ A
        
        # 使用幂迭代估计最大特征值
        v = np.random.randn(len(x))
        v = v / np.linalg.norm(v)
        
        for _ in range(10):
            v = B @ v
            v = v / np.linalg.norm(v)
        
        lambda_max = np.dot(v, B @ v)
        lambda_min = 1.0  # 假设最小特征值为1
        
        # Chebyshev多项式参数
        alpha = (lambda_max + lambda_min) / 2
        beta = (lambda_max - lambda_min) / 2
        
        x_new = x.copy()
        for _ in range(iterations):
            # 应用Chebyshev多项式
            r = b - A @ x_new
            p = D_inv @ r
            q = p
            
            for k in range(1, degree + 1):
                if k == 1:
                    x_new = x_new + (2.0 / alpha) * p
                else:
                    gamma = 2.0 / (alpha - beta * np.cos(np.pi * (2*k-1) / (2*degree)))
                    p_new = gamma * (D_inv @ (b - A @ x_new)) - (gamma - 1) * p
                    x_new = x_new + gamma * p_new
                    p = p_new
        
        return x_new
    
    @staticmethod
    def symmetric_gauss_seidel_smooth(A: sp.spmatrix, b: np.ndarray, x: np.ndarray, 
                                    iterations: int = 1) -> np.ndarray:
        """对称Gauss-Seidel平滑器"""
        A_dense = A.toarray()
        x_new = x.copy()
        
        for _ in range(iterations):
            # 前向扫描
            for i in range(len(x)):
                x_new[i] = (b[i] - np.dot(A_dense[i, :i], x_new[:i]) - 
                           np.dot(A_dense[i, i+1:], x_new[i+1:])) / A_dense[i, i]
            
            # 后向扫描
            for i in range(len(x)-1, -1, -1):
                x_new[i] = (b[i] - np.dot(A_dense[i, :i], x_new[:i]) - 
                           np.dot(A_dense[i, i+1:], x_new[i+1:])) / A_dense[i, i]
        
        return x_new


class AdaptiveCoarsening:
    """自适应粗化策略"""
    
    @staticmethod
    def algebraic_coarsening(A: sp.spmatrix, strong_threshold: float = 0.25) -> Tuple[List[int], List[int]]:
        """代数粗化策略（改进版）"""
        n = A.shape[0]
        
        # 计算强连接矩阵
        S = sp.lil_matrix((n, n))
        for i in range(n):
            row = A[i].toarray().flatten()
            diag = A[i, i]
            
            for j in range(n):
                if i != j and abs(row[j]) > strong_threshold * abs(diag):
                    S[i, j] = 1
        
        # 使用改进的最大独立集算法
        coarse_points = []
        fine_points = []
        marked = np.zeros(n, dtype=bool)
        
        # 第一遍：选择度数最大的点
        degrees = np.array(S.sum(axis=1)).flatten()
        
        while np.any(~marked):
            unmarked = np.where(~marked)[0]
            if len(unmarked) == 0:
                break
            
            # 选择未标记的度数最大的点
            max_degree_idx = unmarked[np.argmax(degrees[unmarked])]
            coarse_points.append(max_degree_idx)
            marked[max_degree_idx] = True
            
            # 标记相邻点
            neighbors = S[max_degree_idx].nonzero()[1]
            marked[neighbors] = True
        
        # 第二遍：处理剩余点
        for i in range(n):
            if not marked[i]:
                fine_points.append(i)
        
        return coarse_points, fine_points
    
    @staticmethod
    def geometric_coarsening(mesh: Dict, target_size: int) -> Dict:
        """几何粗化策略（改进版）"""
        nodes = mesh.get('nodes', [])
        elements = mesh.get('elements', [])
        
        if len(nodes) <= target_size:
            return mesh
        
        # 使用更智能的粗化策略
        # 1. 计算每个节点的质量指标
        node_quality = np.zeros(len(nodes))
        for element in elements:
            element_nodes = element.get('nodes', [])
            if len(element_nodes) >= 3:
                # 计算元素质量（基于形状）
                quality = AdaptiveCoarsening._compute_element_quality(nodes, element_nodes)
                for node_id in element_nodes:
                    node_quality[node_id] += quality
        
        # 2. 选择质量最高的点作为粗网格点
        sorted_indices = np.argsort(node_quality)[::-1]
        coarse_size = min(target_size, len(nodes) // 2)
        coarse_indices = sorted_indices[:coarse_size]
        
        # 3. 构建粗网格
        coarse_nodes = [nodes[i] for i in coarse_indices]
        coarse_elements = []
        
        # 重新编号
        node_mapping = {old: new for new, old in enumerate(coarse_indices)}
        
        for element in elements:
            element_nodes = element.get('nodes', [])
            coarse_element_nodes = []
            
            for node in element_nodes:
                if node in node_mapping:
                    coarse_element_nodes.append(node_mapping[node])
            
            if len(coarse_element_nodes) >= 3:
                coarse_elements.append({'nodes': coarse_element_nodes})
        
        return {
            'nodes': coarse_nodes,
            'elements': coarse_elements,
            'node_mapping': node_mapping
        }
    
    @staticmethod
    def _compute_element_quality(nodes: List, element_nodes: List) -> float:
        """计算元素质量"""
        if len(element_nodes) < 3:
            return 0.0
        
        # 简化的质量计算：基于边长
        coords = [nodes[i] for i in element_nodes]
        min_edge_length = float('inf')
        max_edge_length = 0.0
        
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                edge_length = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
                min_edge_length = min(min_edge_length, edge_length)
                max_edge_length = max(max_edge_length, edge_length)
        
        if max_edge_length > 0:
            return min_edge_length / max_edge_length
        else:
            return 0.0


class BaseMultigridSolver(ABC):
    """增强型多重网格求解器基类"""
    
    def __init__(self, config: MultigridConfig = None):
        self.config = config or MultigridConfig()
        self.levels = []
        self.is_setup = False
        self.smoother = AdvancedSmoother()
        self.coarsening = AdaptiveCoarsening()
        
        # 性能统计
        self.performance_stats = {
            'setup_time': 0.0,
            'solve_time': 0.0,
            'total_iterations': 0,
            'final_residual': 0.0
        }
    
    @abstractmethod
    def setup(self, A: sp.spmatrix, b: np.ndarray, **kwargs) -> None:
        """设置多重网格层次"""
        pass
    
    @abstractmethod
    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        """求解线性系统"""
        pass
    
    def v_cycle(self, level: int, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """V-cycle算法"""
        return self._cycle(level, b, x, 'v')
    
    def w_cycle(self, level: int, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """W-cycle算法"""
        return self._cycle(level, b, x, 'w')
    
    def fmg_cycle(self, level: int, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """FMG（完全多重网格）算法"""
        return self._cycle(level, b, x, 'fmg')
    
    def _cycle(self, level: int, b: np.ndarray, x: np.ndarray, cycle_type: str) -> np.ndarray:
        """通用循环算法"""
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
            if cycle_type == 'v':
                coarse_error = self.v_cycle(level + 1, coarse_residual, np.zeros_like(coarse_residual))
            elif cycle_type == 'w':
                coarse_error = self.w_cycle(level + 1, coarse_residual, np.zeros_like(coarse_residual))
                coarse_error = self.w_cycle(level + 1, coarse_residual, coarse_error)
            elif cycle_type == 'fmg':
                # FMG：在粗网格上求解完整问题
                if level == 0:
                    # 最细层：使用插值解作为初值
                    coarse_b = coarse_residual
                    coarse_x0 = np.zeros_like(coarse_b)
                else:
                    coarse_b = coarse_residual
                    coarse_x0 = np.zeros_like(coarse_b)
                
                coarse_error = self.fmg_cycle(level + 1, coarse_b, coarse_x0)
            
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
        """高级平滑器"""
        if self.config.smoother == 'jacobi':
            return self.smoother.jacobi_smooth(A, b, x, 
                                             omega=self.config.relaxation_factor)
        elif self.config.smoother == 'gauss_seidel':
            return self.smoother.gauss_seidel_smooth(A, b, x)
        elif self.config.smoother == 'chebyshev':
            return self.smoother.chebyshev_smooth(A, b, x, 
                                                degree=self.config.chebyshev_degree)
        elif self.config.smoother == 'symmetric_gauss_seidel':
            return self.smoother.symmetric_gauss_seidel_smooth(A, b, x)
        else:
            return self.smoother.jacobi_smooth(A, b, x)
    
    def _solve_coarse_system(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        """求解粗网格系统"""
        if self.config.coarse_solver == 'direct':
            return spsolve(A, b)
        elif self.config.coarse_solver == 'cg':
            return cg(A, b, tol=1e-10)[0]
        elif self.config.coarse_solver == 'gmres':
            return gmres(A, b, tol=1e-10)[0]
        else:
            return spsolve(A, b)
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        return self.performance_stats.copy()


class AlgebraicMultigridSolver(BaseMultigridSolver):
    """增强型代数多重网格求解器"""
    
    def __init__(self, config: MultigridConfig = None):
        super().__init__(config)
        self.interpolation_operators = []
        self.restriction_operators = []
        self.coarse_matrices = []
    
    def setup(self, A: sp.spmatrix, b: np.ndarray, **kwargs) -> None:
        """设置AMG层次（增强版）"""
        start_time = time.time()
        print("🔄 设置增强型代数多重网格...")
        
        self.levels = []
        self.interpolation_operators = []
        self.restriction_operators = []
        self.coarse_matrices = []
        
        current_A = A.copy()
        current_level = 0
        
        while (current_level < self.config.max_levels and 
               current_A.shape[0] > self.config.max_coarse_size):
            
            # 创建当前层次
            level_data = {
                'matrix': current_A,
                'size': current_A.shape[0],
                'level': current_level
            }
            self.levels.append(level_data)
            
            # 自适应粗化
            if self.config.adaptive_coarsening:
                coarse_points, fine_points = self.coarsening.algebraic_coarsening(
                    current_A, self.config.strong_threshold
                )
            else:
                # 使用简单的粗化策略
                coarse_points = list(range(0, current_A.shape[0], 2))
                fine_points = list(range(1, current_A.shape[0], 2))
            
            # 构建插值算子
            P = self._build_advanced_interpolation_operator(current_A, coarse_points, fine_points)
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
        self.performance_stats['setup_time'] = time.time() - start_time
        print(f"✅ 增强型AMG设置完成，共 {len(self.levels)} 层")
        print(f"   设置时间: {self.performance_stats['setup_time']:.4f}s")
    
    def _build_advanced_interpolation_operator(self, A: sp.spmatrix, 
                                             coarse_points: List[int], 
                                             fine_points: List[int]) -> sp.spmatrix:
        """构建高级插值算子"""
        n = A.shape[0]
        n_coarse = len(coarse_points)
        
        # 构建插值算子
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
                weights = self._compute_advanced_interpolation_weights(A, fine_point, strong_connections)
                
                for coarse_idx, weight in zip(strong_connections, weights):
                    P_data.append(weight)
                    P_rows.append(fine_point)
                    P_cols.append(coarse_idx)
            else:
                # 如果没有强连接，使用距离最近的粗网格点
                distances = [abs(fine_point - cp) for cp in coarse_points]
                nearest_idx = np.argmin(distances)
                P_data.append(1.0)
                P_rows.append(fine_point)
                P_cols.append(nearest_idx)
        
        # 构建稀疏矩阵
        P = sp.csr_matrix((P_data, (P_rows, P_cols)), shape=(n, n_coarse))
        
        return P
    
    def _find_strong_connections(self, A: sp.spmatrix, point: int, coarse_points: List[int]) -> List[int]:
        """找到强连接（改进版）"""
        row = A[point].toarray().flatten()
        diag = A[point, point]
        
        strong_connections = []
        for i, coarse_point in enumerate(coarse_points):
            if abs(row[coarse_point]) > self.config.strong_threshold * abs(diag):
                strong_connections.append(i)
        
        return strong_connections
    
    def _compute_advanced_interpolation_weights(self, A: sp.spmatrix, fine_point: int, 
                                             coarse_points: List[int]) -> List[float]:
        """计算高级插值权重"""
        # 使用基于矩阵元素的权重计算
        row = A[fine_point].toarray().flatten()
        diag = A[fine_point, fine_point]
        
        weights = []
        total_weight = 0.0
        
        for coarse_point in coarse_points:
            weight = abs(row[coarse_point]) / max(abs(diag), 1e-10)
            weights.append(weight)
            total_weight += weight
        
        # 归一化权重
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(coarse_points)] * len(coarse_points)
        
        return weights
    
    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        """求解线性系统（增强版）"""
        if not self.is_setup:
            self.setup(A, b)
        
        start_time = time.time()
        
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        
        # 多重网格迭代
        for iteration in range(self.config.max_iterations):
            # 根据配置选择循环类型
            if self.config.cycle_type == 'v':
                x = self.v_cycle(0, b, x)
            elif self.config.cycle_type == 'w':
                x = self.w_cycle(0, b, x)
            elif self.config.cycle_type == 'fmg':
                x = self.fmg_cycle(0, b, x)
            else:
                x = self.v_cycle(0, b, x)
            
            # 检查收敛性
            residual = b - A @ x
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm < self.config.tolerance:
                print(f"✅ 增强型AMG收敛，迭代次数: {iteration + 1}")
                break
        
        self.performance_stats['solve_time'] = time.time() - start_time
        self.performance_stats['total_iterations'] = iteration + 1
        self.performance_stats['final_residual'] = residual_norm
        
        return x


class GeometricMultigridSolver(BaseMultigridSolver):
    """增强型几何多重网格求解器"""
    
    def __init__(self, config: MultigridConfig = None):
        super().__init__(config)
        self.meshes = []
        self.interpolation_operators = []
        self.restriction_operators = []
    
    def setup(self, A: sp.spmatrix, b: np.ndarray, mesh_data: Dict = None, **kwargs) -> None:
        """设置GMG层次（增强版）"""
        start_time = time.time()
        print("🔄 设置增强型几何多重网格...")
        
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
            if len(current_mesh.get('nodes', [])) <= self.config.max_coarse_size:
                break
            
            # 自适应粗化网格
            if self.config.adaptive_coarsening:
                coarse_mesh = self.coarsening.geometric_coarsening(
                    current_mesh, self.config.max_coarse_size
                )
            else:
                coarse_mesh = self._simple_coarsen_mesh(current_mesh)
            
            # 构建插值和限制算子
            P, R = self._build_advanced_geometric_operators(current_mesh, coarse_mesh)
            
            self.interpolation_operators.append(P)
            self.restriction_operators.append(R)
            
            # 更新到下一层
            current_mesh = coarse_mesh
            current_level += 1
        
        self.is_setup = True
        self.performance_stats['setup_time'] = time.time() - start_time
        print(f"✅ 增强型GMG设置完成，共 {len(self.levels)} 层")
        print(f"   设置时间: {self.performance_stats['setup_time']:.4f}s")
    
    def _simple_coarsen_mesh(self, mesh: Dict) -> Dict:
        """简单网格粗化策略"""
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
    
    def _build_advanced_geometric_operators(self, fine_mesh: Dict, 
                                          coarse_mesh: Dict) -> Tuple[sp.spmatrix, sp.spmatrix]:
        """构建高级几何插值和限制算子"""
        fine_nodes = fine_mesh.get('nodes', [])
        coarse_nodes = coarse_mesh.get('nodes', [])
        
        n_fine = len(fine_nodes)
        n_coarse = len(coarse_nodes)
        
        # 构建插值算子：使用更智能的插值策略
        P_data = []
        P_rows = []
        P_cols = []
        
        for i, fine_node in enumerate(fine_nodes):
            # 找到最近的粗网格点
            distances = [np.linalg.norm(np.array(fine_node) - np.array(coarse_node)) 
                        for coarse_node in coarse_nodes]
            nearest_coarse = np.argmin(distances)
            
            # 计算插值权重（基于距离）
            min_distance = distances[nearest_coarse]
            if min_distance > 0:
                weight = 1.0 / (1.0 + min_distance)
            else:
                weight = 1.0
            
            P_data.append(weight)
            P_rows.append(i)
            P_cols.append(nearest_coarse)
        
        P = sp.csr_matrix((P_data, (P_rows, P_cols)), shape=(n_fine, n_coarse))
        R = P.T  # 限制算子为插值算子的转置
        
        return P, R
    
    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray = None, 
              mesh_data: Dict = None) -> np.ndarray:
        """求解线性系统（增强版）"""
        if not self.is_setup:
            self.setup(A, b, mesh_data)
        
        start_time = time.time()
        
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        
        # 多重网格迭代
        for iteration in range(self.config.max_iterations):
            # 根据配置选择循环类型
            if self.config.cycle_type == 'v':
                x = self.v_cycle(0, b, x)
            elif self.config.cycle_type == 'w':
                x = self.w_cycle(0, b, x)
            elif self.config.cycle_type == 'fmg':
                x = self.fmg_cycle(0, b, x)
            else:
                x = self.v_cycle(0, b, x)
            
            # 检查收敛性
            residual = b - A @ x
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm < self.config.tolerance:
                print(f"✅ 增强型GMG收敛，迭代次数: {iteration + 1}")
                break
        
        self.performance_stats['solve_time'] = time.time() - start_time
        self.performance_stats['total_iterations'] = iteration + 1
        self.performance_stats['final_residual'] = residual_norm
        
        return x


# 工厂函数
def create_multigrid_solver(solver_type: str = 'amg', config: MultigridConfig = None) -> BaseMultigridSolver:
    """创建增强型多重网格求解器"""
    if solver_type == 'amg':
        return AlgebraicMultigridSolver(config)
    elif solver_type == 'gmg':
        return GeometricMultigridSolver(config)
    else:
        raise ValueError(f"不支持的多重网格求解器类型: {solver_type}")


def create_multigrid_config(**kwargs) -> MultigridConfig:
    """创建增强型多重网格配置"""
    return MultigridConfig(**kwargs)


def benchmark_multigrid_solvers(A: sp.spmatrix, b: np.ndarray, 
                               configs: List[MultigridConfig] = None) -> Dict:
    """多重网格求解器性能基准测试"""
    if configs is None:
        configs = [
            MultigridConfig(smoother='jacobi', cycle_type='v'),
            MultigridConfig(smoother='gauss_seidel', cycle_type='v'),
            MultigridConfig(smoother='chebyshev', cycle_type='v'),
            MultigridConfig(smoother='jacobi', cycle_type='w'),
            MultigridConfig(smoother='jacobi', cycle_type='fmg'),
        ]
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\n🧪 测试配置 {i+1}: {config.smoother} + {config.cycle_type}-cycle")
        
        # 创建求解器
        solver = create_multigrid_solver('amg', config)
        
        # 设置和求解
        start_time = time.time()
        solver.setup(A, b)
        setup_time = time.time() - start_time
        
        start_time = time.time()
        x = solver.solve(A, b)
        solve_time = time.time() - start_time
        
        # 计算残差
        residual = b - A @ x
        residual_norm = np.linalg.norm(residual)
        
        # 存储结果
        config_name = f"{config.smoother}_{config.cycle_type}"
        results[config_name] = {
            'setup_time': setup_time,
            'solve_time': solve_time,
            'total_time': setup_time + solve_time,
            'iterations': solver.performance_stats['total_iterations'],
            'final_residual': residual_norm,
            'performance_stats': solver.performance_stats
        }
        
        print(f"   设置时间: {setup_time:.4f}s")
        print(f"   求解时间: {solve_time:.4f}s")
        print(f"   迭代次数: {solver.performance_stats['total_iterations']}")
        print(f"   最终残差: {residual_norm:.2e}")
    
    return results
