"""
多物理场耦合求解器

实现多物理场耦合求解的核心功能：
1. 耦合方程组组装：温度-位移、流体-固体等耦合
2. 分区求解策略：分离式迭代和全耦合求解
3. 时间积分器：隐式时间步进算法
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
from scipy.integrate import solve_ivp


@dataclass
class CouplingConfig:
    """多物理场耦合配置"""
    # 物理场配置
    physics_fields: List[str] = None  # 物理场列表，如 ['thermal', 'mechanical']
    coupling_type: str = 'staggered'  # 耦合类型：'staggered', 'monolithic', 'hybrid'
    
    # 求解策略配置
    max_iterations: int = 50  # 最大迭代次数
    tolerance: float = 1e-6  # 收敛容差
    relaxation_factor: float = 0.8  # 松弛因子
    
    # 时间积分配置
    time_integration: str = 'implicit'  # 时间积分类型：'explicit', 'implicit', 'semi_implicit'
    time_step: float = 0.01  # 时间步长
    max_time_steps: int = 1000  # 最大时间步数
    
    # 并行配置
    parallel_support: bool = False  # 并行支持
    domain_decomposition: bool = True  # 域分解


class PhysicsField:
    """物理场基类"""
    
    def __init__(self, name: str, field_type: str):
        self.name = name
        self.field_type = field_type
        self.dofs = 0  # 自由度数量
        self.matrix = None  # 刚度矩阵
        self.rhs = None  # 右端项
        self.solution = None  # 解向量
        
    def assemble_matrix(self, mesh_data: Dict, material_props: Dict) -> sp.spmatrix:
        """组装物理场矩阵"""
        raise NotImplementedError("子类必须实现此方法")
    
    def assemble_rhs(self, mesh_data: Dict, boundary_conditions: Dict) -> np.ndarray:
        """组装物理场右端项"""
        raise NotImplementedError("子类必须实现此方法")
    
    def solve_field(self, solver_type: str = 'direct') -> np.ndarray:
        """求解单个物理场"""
        if self.matrix is None or self.rhs is None:
            raise ValueError("矩阵和右端项未组装")
        
        if solver_type == 'direct':
            self.solution = spsolve(self.matrix, self.rhs)
        elif solver_type == 'cg':
            self.solution = cg(self.matrix, self.rhs, tol=1e-10)[0]
        elif solver_type == 'gmres':
            self.solution = gmres(self.matrix, self.rhs, tol=1e-10)[0]
        else:
            raise ValueError(f"不支持的求解器类型: {solver_type}")
        
        return self.solution


class ThermalField(PhysicsField):
    """热场"""
    
    def __init__(self):
        super().__init__("thermal", "scalar")
    
    def assemble_matrix(self, mesh_data: Dict, material_props: Dict) -> sp.spmatrix:
        """组装热传导矩阵"""
        # 简化的热传导矩阵组装
        n_nodes = len(mesh_data.get('nodes', []))
        K = sp.lil_matrix((n_nodes, n_nodes))
        
        # 这里应该实现真正的有限元组装
        # 简化版本：对角占优矩阵
        for i in range(n_nodes):
            K[i, i] = 1.0
            if i > 0:
                K[i, i-1] = -0.1
            if i < n_nodes - 1:
                K[i, i+1] = -0.1
        
        self.matrix = K.tocsr()
        self.dofs = n_nodes
        return self.matrix
    
    def assemble_rhs(self, mesh_data: Dict, boundary_conditions: Dict) -> np.ndarray:
        """组装热场右端项"""
        n_nodes = len(mesh_data.get('nodes', []))
        f = np.zeros(n_nodes)
        
        # 应用边界条件
        for bc in boundary_conditions.get('thermal', []):
            if bc['type'] == 'dirichlet':
                node_id = bc['node_id']
                value = bc['value']
                f[node_id] = value
        
        self.rhs = f
        return self.rhs


class MechanicalField(PhysicsField):
    """力学场"""
    
    def __init__(self, dimension: int = 2):
        super().__init__("mechanical", "vector")
        self.dimension = dimension
    
    def assemble_matrix(self, mesh_data: Dict, material_props: Dict) -> sp.spmatrix:
        """组装力学刚度矩阵"""
        n_nodes = len(mesh_data.get('nodes', []))
        n_dofs = n_nodes * self.dimension
        K = sp.lil_matrix((n_dofs, n_dofs))
        
        # 简化的力学刚度矩阵组装
        # 这里应该实现真正的有限元组装
        for i in range(n_nodes):
            for d in range(self.dimension):
                dof_i = i * self.dimension + d
                K[dof_i, dof_i] = 1.0
                
                # 相邻节点的耦合
                if i > 0:
                    dof_prev = (i - 1) * self.dimension + d
                    K[dof_i, dof_prev] = -0.1
                if i < n_nodes - 1:
                    dof_next = (i + 1) * self.dimension + d
                    K[dof_i, dof_next] = -0.1
        
        self.matrix = K.tocsr()
        self.dofs = n_dofs
        return self.matrix
    
    def assemble_rhs(self, mesh_data: Dict, boundary_conditions: Dict) -> np.ndarray:
        """组装力学场右端项"""
        n_nodes = len(mesh_data.get('nodes', []))
        n_dofs = n_nodes * self.dimension
        f = np.zeros(n_dofs)
        
        # 应用边界条件
        for bc in boundary_conditions.get('mechanical', []):
            if bc['type'] == 'dirichlet':
                node_id = bc['node_id']
                component = bc.get('component', 0)
                value = bc['value']
                dof_id = node_id * self.dimension + component
                f[dof_id] = value
        
        self.rhs = f
        return self.rhs


class CouplingOperator:
    """耦合算子"""
    
    def __init__(self, source_field: PhysicsField, target_field: PhysicsField):
        self.source_field = source_field
        self.target_field = target_field
        self.coupling_matrix = None
    
    def assemble_coupling_matrix(self, mesh_data: Dict, coupling_props: Dict) -> sp.spmatrix:
        """组装耦合矩阵"""
        # 这里应该实现真正的耦合矩阵组装
        # 简化版本：零矩阵
        n_source = self.source_field.dofs
        n_target = self.target_field.dofs
        
        self.coupling_matrix = sp.csr_matrix((n_target, n_source))
        return self.coupling_matrix
    
    def apply_coupling(self, source_solution: np.ndarray) -> np.ndarray:
        """应用耦合效应"""
        if self.coupling_matrix is None:
            return np.zeros(self.target_field.dofs)
        
        return self.coupling_matrix @ source_solution


class MultiphysicsCouplingSolver:
    """多物理场耦合求解器"""
    
    def __init__(self, config: CouplingConfig):
        self.config = config
        self.physics_fields = {}
        self.coupling_operators = {}
        self.solutions = {}
        self.performance_stats = {
            'setup_time': 0.0,
            'solve_time': 0.0,
            'coupling_iterations': 0,
            'final_residual': 0.0
        }
        
        # 初始化物理场
        self._initialize_physics_fields()
    
    def _initialize_physics_fields(self):
        """初始化物理场"""
        if self.config.physics_fields is None:
            self.config.physics_fields = ['thermal', 'mechanical']
        
        for field_name in self.config.physics_fields:
            if field_name == 'thermal':
                self.physics_fields[field_name] = ThermalField()
            elif field_name == 'mechanical':
                self.physics_fields[field_name] = MechanicalField()
            else:
                raise ValueError(f"不支持的物理场类型: {field_name}")
    
    def setup(self, mesh_data: Dict, material_props: Dict, 
              boundary_conditions: Dict, coupling_props: Dict = None):
        """设置多物理场耦合系统"""
        start_time = time.time()
        print("🔄 设置多物理场耦合系统...")
        
        # 组装各物理场矩阵
        for field_name, field in self.physics_fields.items():
            print(f"   组装 {field_name} 场...")
            field.assemble_matrix(mesh_data, material_props)
            field.assemble_rhs(mesh_data, boundary_conditions)
        
        # 组装耦合矩阵
        if coupling_props:
            self._assemble_coupling_matrices(mesh_data, coupling_props)
        
        self.performance_stats['setup_time'] = time.time() - start_time
        print(f"✅ 多物理场耦合系统设置完成")
        print(f"   设置时间: {self.performance_stats['setup_time']:.4f}s")
    
    def _assemble_coupling_matrices(self, mesh_data: Dict, coupling_props: Dict):
        """组装耦合矩阵"""
        field_names = list(self.physics_fields.keys())
        
        for i, field1_name in enumerate(field_names):
            for j, field2_name in enumerate(field_names):
                if i != j:
                    coupling_key = f"{field1_name}_to_{field2_name}"
                    source_field = self.physics_fields[field1_name]
                    target_field = self.physics_fields[field2_name]
                    
                    coupling_op = CouplingOperator(source_field, target_field)
                    coupling_op.assemble_coupling_matrix(mesh_data, coupling_props)
                    self.coupling_operators[coupling_key] = coupling_op
    
    def solve_staggered(self, max_iterations: int = None, tolerance: float = None) -> Dict:
        """分离式迭代求解"""
        if max_iterations is None:
            max_iterations = self.config.max_iterations
        if tolerance is None:
            tolerance = self.config.tolerance
        
        print("🔄 开始分离式迭代求解...")
        start_time = time.time()
        
        # 初始化解
        for field_name, field in self.physics_fields.items():
            self.solutions[field_name] = np.zeros(field.dofs)
        
        # 分离式迭代
        for iteration in range(max_iterations):
            print(f"   迭代 {iteration + 1}/{max_iterations}")
            
            # 保存上一步解
            prev_solutions = {name: sol.copy() for name, sol in self.solutions.items()}
            
            # 依次求解各物理场
            for field_name, field in self.physics_fields.items():
                # 计算耦合项
                coupling_rhs = self._compute_coupling_rhs(field_name)
                
                # 求解物理场
                field.rhs += coupling_rhs
                self.solutions[field_name] = field.solve_field()
            
            # 检查收敛性
            converged = self._check_convergence(prev_solutions, tolerance)
            if converged:
                print(f"✅ 分离式迭代收敛，迭代次数: {iteration + 1}")
                break
        
        self.performance_stats['solve_time'] = time.time() - start_time
        self.performance_stats['coupling_iterations'] = iteration + 1
        
        return self.solutions
    
    def solve_monolithic(self) -> Dict:
        """全耦合求解"""
        print("🔄 开始全耦合求解...")
        start_time = time.time()
        
        # 构建全局耦合矩阵
        global_matrix, global_rhs = self._build_monolithic_system()
        
        # 求解全局系统
        global_solution = spsolve(global_matrix, global_rhs)
        
        # 分解解向量
        self.solutions = self._extract_field_solutions(global_solution)
        
        self.performance_stats['solve_time'] = time.time() - start_time
        print(f"✅ 全耦合求解完成")
        
        return self.solutions
    
    def solve_hybrid(self) -> Dict:
        """混合求解策略"""
        print("🔄 开始混合求解策略...")
        
        # 先进行几次分离式迭代
        self.solve_staggered(max_iterations=10, tolerance=1e-4)
        
        # 然后切换到全耦合求解
        self.solve_monolithic()
        
        return self.solutions
    
    def _compute_coupling_rhs(self, field_name: str) -> np.ndarray:
        """计算耦合右端项"""
        coupling_rhs = np.zeros(self.physics_fields[field_name].dofs)
        
        for coupling_key, coupling_op in self.coupling_operators.items():
            if coupling_key.endswith(f"_to_{field_name}"):
                source_field_name = coupling_key.split("_to_")[0]
                if source_field_name in self.solutions:
                    source_solution = self.solutions[source_field_name]
                    coupling_effect = coupling_op.apply_coupling(source_solution)
                    coupling_rhs += coupling_effect
        
        return coupling_rhs
    
    def _check_convergence(self, prev_solutions: Dict, tolerance: float) -> bool:
        """检查收敛性"""
        max_change = 0.0
        
        for field_name, field in self.physics_fields.items():
            if field_name in prev_solutions and field_name in self.solutions:
                prev_sol = prev_solutions[field_name]
                curr_sol = self.solutions[field_name]
                
                change = np.linalg.norm(curr_sol - prev_sol) / (np.linalg.norm(curr_sol) + 1e-10)
                max_change = max(max_change, change)
        
        return max_change < tolerance
    
    def _build_monolithic_system(self) -> Tuple[sp.spmatrix, np.ndarray]:
        """构建全局耦合系统"""
        # 计算总自由度
        total_dofs = sum(field.dofs for field in self.physics_fields.values())
        
        # 构建全局矩阵
        global_matrix = sp.lil_matrix((total_dofs, total_dofs))
        global_rhs = np.zeros(total_dofs)
        
        # 组装各物理场矩阵
        dof_offset = 0
        for field_name, field in self.physics_fields.items():
            # 主对角块
            global_matrix[dof_offset:dof_offset+field.dofs, 
                         dof_offset:dof_offset+field.dofs] = field.matrix
            
            # 右端项
            global_rhs[dof_offset:dof_offset+field.dofs] = field.rhs
            
            dof_offset += field.dofs
        
        # 组装耦合项
        for coupling_key, coupling_op in self.coupling_operators.items():
            source_field_name, target_field_name = coupling_key.split("_to_")
            
            if source_field_name in self.physics_fields and target_field_name in self.physics_fields:
                source_field = self.physics_fields[source_field_name]
                target_field = self.physics_fields[target_field_name]
                
                # 计算耦合矩阵在全局矩阵中的位置
                source_offset = self._get_field_dof_offset(source_field_name)
                target_offset = self._get_field_dof_offset(target_field_name)
                
                # 添加耦合项
                global_matrix[target_offset:target_offset+target_field.dofs,
                             source_offset:source_offset+source_field.dofs] += coupling_op.coupling_matrix
        
        return global_matrix.tocsr(), global_rhs
    
    def _get_field_dof_offset(self, field_name: str) -> int:
        """获取物理场在全局系统中的自由度偏移"""
        offset = 0
        for name, field in self.physics_fields.items():
            if name == field_name:
                return offset
            offset += field.dofs
        return 0
    
    def _extract_field_solutions(self, global_solution: np.ndarray) -> Dict:
        """从全局解向量中提取各物理场的解"""
        solutions = {}
        dof_offset = 0
        
        for field_name, field in self.physics_fields.items():
            solutions[field_name] = global_solution[dof_offset:dof_offset+field.dofs]
            dof_offset += field.dofs
        
        return solutions
    
    def solve_transient(self, initial_conditions: Dict, time_span: Tuple[float, float]) -> Dict:
        """瞬态求解"""
        print("🔄 开始瞬态求解...")
        
        # 设置初始条件
        for field_name, field in self.physics_fields.items():
            if field_name in initial_conditions:
                self.solutions[field_name] = initial_conditions[field_name].copy()
            else:
                self.solutions[field_name] = np.zeros(field.dofs)
        
        # 时间积分
        time_steps = []
        solutions_history = []
        
        current_time = time_span[0]
        while current_time < time_span[1]:
            time_steps.append(current_time)
            solutions_history.append({name: sol.copy() for name, sol in self.solutions.items()})
            
            # 时间步进
            if self.config.time_integration == 'implicit':
                self._implicit_time_step(current_time)
            elif self.config.time_integration == 'explicit':
                self._explicit_time_step(current_time)
            else:
                self._semi_implicit_time_step(current_time)
            
            current_time += self.config.time_step
            
            if len(time_steps) >= self.config.max_time_steps:
                break
        
        return {
            'time_steps': time_steps,
            'solutions_history': solutions_history,
            'final_solutions': self.solutions
        }
    
    def _implicit_time_step(self, current_time: float):
        """隐式时间步进"""
        # 使用分离式迭代求解当前时间步
        self.solve_staggered(max_iterations=20, tolerance=1e-6)
    
    def _explicit_time_step(self, current_time: float):
        """显式时间步进"""
        # 简化的显式时间步进
        for field_name, field in self.physics_fields.items():
            if field.matrix is not None:
                # 使用显式Euler方法
                dt = self.config.time_step
                current_sol = self.solutions[field_name]
                
                # 计算时间导数
                dudt = field.matrix @ current_sol + field.rhs
                
                # 更新解
                self.solutions[field_name] = current_sol + dt * dudt
    
    def _semi_implicit_time_step(self, current_time: float):
        """半隐式时间步进"""
        # 混合显式和隐式方法
        self._explicit_time_step(current_time)
        self._implicit_time_step(current_time)
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        return self.performance_stats.copy()


# 工厂函数
def create_multiphysics_solver(config: CouplingConfig = None) -> MultiphysicsCouplingSolver:
    """创建多物理场耦合求解器"""
    if config is None:
        config = CouplingConfig()
    return MultiphysicsCouplingSolver(config)


def create_coupling_config(**kwargs) -> CouplingConfig:
    """创建多物理场耦合配置"""
    return CouplingConfig(**kwargs)


def benchmark_coupling_strategies(mesh_data: Dict, material_props: Dict, 
                                boundary_conditions: Dict, coupling_props: Dict = None) -> Dict:
    """多物理场耦合策略性能基准测试"""
    strategies = ['staggered', 'monolithic', 'hybrid']
    results = {}
    
    for strategy in strategies:
        print(f"\n🧪 测试耦合策略: {strategy}")
        
        # 创建配置
        config = CouplingConfig(coupling_type=strategy)
        
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
            'performance_stats': solver.get_performance_stats()
        }
        
        print(f"   设置时间: {setup_time:.4f}s")
        print(f"   求解时间: {solve_time:.4f}s")
        print(f"   总时间: {setup_time + solve_time:.4f}s")
    
    return results
