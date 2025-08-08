"""
热-力学耦合模块
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
import time

# 可选依赖
try:
    from scipy.sparse import csr_matrix, lil_matrix
    from scipy.sparse.linalg import spsolve, gmres, cg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False


@dataclass
class CouplingState:
    """耦合状态"""
    temperature: np.ndarray  # 温度场
    displacement: np.ndarray  # 位移场
    stress: np.ndarray  # 应力场
    strain: np.ndarray  # 应变场
    thermal_strain: np.ndarray  # 热应变
    mechanical_heat: np.ndarray  # 机械功产热
    time: float = 0.0
    iteration: int = 0
    residual_norm: float = 0.0


@dataclass
class CouplingConfig:
    """耦合配置"""
    solver_type: str = 'iterative'  # 求解器类型: 'iterative', 'monolithic', 'staggered'
    max_iterations: int = 10  # 最大迭代次数
    tolerance: float = 1e-6  # 收敛容差
    relaxation_factor: float = 1.0  # 松弛因子
    adaptive_timestep: bool = True  # 自适应时间步长
    min_timestep: float = 1e-6  # 最小时间步长
    max_timestep: float = 1e-3  # 最大时间步长
    cfl_number: float = 0.5  # CFL数
    parallel_solver: bool = True  # 使用并行求解器
    preconditioner: str = 'jacobi'  # 预处理器类型


class ThermoMechanicalCoupling:
    """热-力学耦合求解器"""
    
    def __init__(self, mesh, 
                 thermal_conductivity: float = 3.0,
                 heat_capacity: float = 1000.0,
                 density: float = 2700.0,
                 thermal_expansion: float = 2.4e-5,
                 young_modulus: float = 70e9,
                 poisson_ratio: float = 0.3,
                 coupling_parameter: float = 1.0,
                 config: CouplingConfig = None):
        """
        初始化热-力学耦合求解器
        
        Args:
            mesh: 网格对象
            thermal_conductivity: 热导率 (W/m/K)
            heat_capacity: 比热容 (J/kg/K)
            density: 密度 (kg/m³)
            thermal_expansion: 热膨胀系数 (1/K)
            young_modulus: 杨氏模量 (Pa)
            poisson_ratio: 泊松比
            coupling_parameter: 耦合参数 (0-1)
            config: 耦合配置
        """
        self.mesh = mesh
        self.thermal_conductivity = thermal_conductivity
        self.heat_capacity = heat_capacity
        self.density = density
        self.thermal_expansion = thermal_expansion
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.coupling_parameter = coupling_parameter
        self.config = config or CouplingConfig()
        
        # 计算材料参数
        self.thermal_diffusivity = thermal_conductivity / (density * heat_capacity)
        self.lame_lambda = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        self.lame_mu = young_modulus / (2 * (1 + poisson_ratio))
        
        # MPI相关
        self.comm = MPI.COMM_WORLD if HAS_MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1
        
        # 求解器状态
        self.current_state = None
        self.previous_state = None
        
        # 性能监控
        self.performance_stats = {
            'solve_time': 0.0,
            'thermal_solve_time': 0.0,
            'mechanical_solve_time': 0.0,
            'coupling_time': 0.0,
            'iterations': 0,
            'residual_norm': 0.0
        }
        
        # 初始化矩阵
        self.K_thermal = None
        self.K_mechanical = None
        self.C_thermal = None
        self.C_coupling = None
        
        # 预处理器
        self.thermal_preconditioner = None
        self.mechanical_preconditioner = None
    
    def compute_thermal_stress(self, temperature: np.ndarray, 
                             reference_temperature: float = 293.15) -> np.ndarray:
        """
        计算热应力
        
        Args:
            temperature: 温度场
            reference_temperature: 参考温度
            
        Returns:
            thermal_stress: 热应力场
        """
        n_points = len(temperature)
        
        # 计算热应变
        thermal_strain = self.thermal_expansion * (temperature - reference_temperature)
        
        # 计算热应力 (平面应力假设)
        thermal_stress = np.zeros((n_points, 3))  # σxx, σyy, σxy
        
        # 热应力分量
        thermal_stress[:, 0] = -self.young_modulus * thermal_strain / (1 - self.poisson_ratio)
        thermal_stress[:, 1] = -self.young_modulus * thermal_strain / (1 - self.poisson_ratio)
        thermal_stress[:, 2] = 0.0  # 剪切应力
        
        return thermal_stress
    
    def compute_mechanical_heating(self, stress: np.ndarray, 
                                 strain_rate: np.ndarray,
                                 plastic_work: np.ndarray = None) -> np.ndarray:
        """
        计算机械功产热
        
        Args:
            stress: 应力场
            strain_rate: 应变率场
            plastic_work: 塑性功 (可选)
            
        Returns:
            mechanical_heat: 机械功产热场
        """
        n_points = len(stress)
        mechanical_heat = np.zeros(n_points)
        
        # 弹性功产热
        for i in range(n_points):
            # 应力-应变率点积
            stress_tensor = np.array([[stress[i, 0], stress[i, 2]],
                                    [stress[i, 2], stress[i, 1]]])
            strain_rate_tensor = np.array([[strain_rate[i, 0], strain_rate[i, 2]],
                                         [strain_rate[i, 2], strain_rate[i, 1]]])
            
            # 计算机械功
            mechanical_work = np.trace(stress_tensor @ strain_rate_tensor)
            mechanical_heat[i] = max(0, mechanical_work)  # 只考虑产热
        
        # 添加塑性功产热
        if plastic_work is not None:
            mechanical_heat += plastic_work
        
        return mechanical_heat
    
    def assemble_coupling_matrix(self, dt: float) -> Tuple[csr_matrix, csr_matrix, 
                                                          csr_matrix, csr_matrix]:
        """
        组装耦合矩阵
        
        Args:
            dt: 时间步长
            
        Returns:
            K_thermal: 热传导矩阵
            K_mechanical: 力学刚度矩阵
            C_thermal: 热容矩阵
            C_coupling: 耦合矩阵
        """
        # 组装热传导矩阵
        if self.K_thermal is None:
            self.K_thermal = self._assemble_thermal_matrix()
        
        # 组装力学刚度矩阵
        if self.K_mechanical is None:
            self.K_mechanical = self._assemble_mechanical_matrix()
        
        # 组装热容矩阵
        if self.C_thermal is None:
            self.C_thermal = self._assemble_thermal_capacity_matrix()
        
        # 组装耦合矩阵
        if self.C_coupling is None:
            self.C_coupling = self._assemble_coupling_matrix_internal(dt)
        
        return self.K_thermal, self.K_mechanical, self.C_thermal, self.C_coupling
    
    def solve_coupled_system(self, 
                           initial_temperature: np.ndarray,
                           initial_displacement: np.ndarray,
                           boundary_conditions: Dict,
                           time_steps: int,
                           dt: float,
                           heat_source: Optional[Callable] = None,
                           body_force: Optional[Callable] = None) -> List[CouplingState]:
        """
        求解耦合系统
        
        Args:
            initial_temperature: 初始温度场
            initial_displacement: 初始位移场
            boundary_conditions: 边界条件
            time_steps: 时间步数
            dt: 时间步长
            heat_source: 热源函数
            body_force: 体力函数
            
        Returns:
            solution_history: 解的历史
        """
        solution_history = []
        
        # 初始化状态
        current_state = CouplingState(
            temperature=initial_temperature.copy(),
            displacement=initial_displacement.copy(),
            stress=np.zeros_like(initial_displacement),
            strain=np.zeros_like(initial_displacement),
            thermal_strain=np.zeros_like(initial_temperature),
            mechanical_heat=np.zeros_like(initial_temperature),
            time=0.0
        )
        
        # 时间步进
        for step in range(time_steps):
            print(f"时间步 {step + 1}/{time_steps}")
            
            # 自适应时间步长
            if self.config.adaptive_timestep:
                dt = self._compute_adaptive_timestep(current_state, dt)
            
            # 求解耦合步骤
            new_state = self._solve_coupled_step(dt, boundary_conditions, heat_source, body_force)
            
            # 更新状态
            new_state.time = current_state.time + dt
            new_state.iteration = step
            
            # 保存解
            solution_history.append(new_state)
            current_state = new_state
            
            # 检查收敛性
            if new_state.residual_norm < self.config.tolerance:
                print(f"在时间步 {step + 1} 收敛")
                break
        
        return solution_history
    
    def _solve_coupled_step(self, dt: float, boundary_conditions: Dict,
                           heat_source: Optional[Callable] = None,
                           body_force: Optional[Callable] = None) -> CouplingState:
        """求解耦合步骤"""
        start_time = time.time()
        
        # 组装耦合矩阵
        K_thermal, K_mechanical, C_thermal, C_coupling = self.assemble_coupling_matrix(dt)
        
        # 计算源项
        heat_source_vector = self._compute_heat_source_vector(heat_source)
        body_force_vector = self._compute_body_force_vector(body_force)
        
        # 根据求解器类型选择求解策略
        if self.config.solver_type == 'iterative':
            new_state = self._solve_coupled_iterative(
                K_thermal, K_mechanical, C_thermal, C_coupling,
                heat_source_vector, body_force_vector, dt, boundary_conditions
            )
        elif self.config.solver_type == 'monolithic':
            new_state = self._solve_coupled_monolithic(
                K_thermal, K_mechanical, C_thermal, C_coupling,
                heat_source_vector, body_force_vector, dt, boundary_conditions
            )
        else:  # staggered
            new_state = self._solve_coupled_staggered(
                K_thermal, K_mechanical, C_thermal, C_coupling,
                heat_source_vector, body_force_vector, dt, boundary_conditions
            )
        
        self.performance_stats['solve_time'] = time.time() - start_time
        return new_state
    
    def _solve_coupled_iterative(self, K_thermal: csr_matrix, K_mechanical: csr_matrix,
                                C_thermal: csr_matrix, C_coupling: csr_matrix,
                                heat_source: np.ndarray, body_force: np.ndarray,
                                dt: float, boundary_conditions: Dict,
                                max_iterations: int = 10, tolerance: float = 1e-6) -> CouplingState:
        """迭代求解耦合系统"""
        # 初始化
        temperature = self.current_state.temperature.copy() if self.current_state else np.zeros_like(heat_source)
        displacement = self.current_state.displacement.copy() if self.current_state else np.zeros_like(body_force)
        
        # 迭代求解
        for iteration in range(max_iterations):
            temperature_old = temperature.copy()
            displacement_old = displacement.copy()
            
            # 求解热传导方程
            temperature = self._solve_thermal_step(
                K_thermal, C_thermal, C_coupling, displacement, heat_source, dt, boundary_conditions
            )
            
            # 求解力学方程
            displacement = self._solve_mechanical_step(
                K_mechanical, temperature, body_force, boundary_conditions
            )
            
            # 检查收敛性
            temp_residual = np.linalg.norm(temperature - temperature_old)
            disp_residual = np.linalg.norm(displacement - displacement_old)
            residual_norm = max(temp_residual, disp_residual)
            
            if residual_norm < tolerance:
                break
        
        # 计算应力和应变
        stress = self._compute_stress(displacement, temperature)
        strain = self._compute_strain(displacement)
        thermal_strain = self.thermal_expansion * (temperature - 293.15)
        mechanical_heat = self.compute_mechanical_heating(stress, strain / dt)
        
        return CouplingState(
            temperature=temperature,
            displacement=displacement,
            stress=stress,
            strain=strain,
            thermal_strain=thermal_strain,
            mechanical_heat=mechanical_heat,
            residual_norm=residual_norm
        )
    
    def _solve_coupled_monolithic(self, K_thermal: csr_matrix, K_mechanical: csr_matrix,
                                 C_thermal: csr_matrix, C_coupling: csr_matrix,
                                 heat_source: np.ndarray, body_force: np.ndarray,
                                 dt: float, boundary_conditions: Dict) -> CouplingState:
        """整体求解耦合系统"""
        # 构建整体系统矩阵
        n_thermal = K_thermal.shape[0]
        n_mechanical = K_mechanical.shape[0]
        
        # 整体矩阵
        A_monolithic = lil_matrix((n_thermal + n_mechanical, n_thermal + n_mechanical))
        b_monolithic = np.zeros(n_thermal + n_mechanical)
        
        # 填充热传导部分
        A_monolithic[:n_thermal, :n_thermal] = C_thermal / dt + K_thermal
        A_monolithic[:n_thermal, n_thermal:] = C_coupling
        b_monolithic[:n_thermal] = heat_source + C_thermal @ self.current_state.temperature / dt
        
        # 填充力学部分
        A_monolithic[n_thermal:, :n_thermal] = -C_coupling.T
        A_monolithic[n_thermal:, n_thermal:] = K_mechanical
        b_monolithic[n_thermal:] = body_force
        
        # 应用边界条件
        A_monolithic, b_monolithic = self._apply_monolithic_boundary_conditions(
            A_monolithic, b_monolithic, boundary_conditions
        )
        
        # 求解整体系统
        A_monolithic = A_monolithic.tocsr()
        solution = spsolve(A_monolithic, b_monolithic)
        
        # 提取解
        temperature = solution[:n_thermal]
        displacement = solution[n_thermal:]
        
        # 计算其他场
        stress = self._compute_stress(displacement, temperature)
        strain = self._compute_strain(displacement)
        thermal_strain = self.thermal_expansion * (temperature - 293.15)
        mechanical_heat = self.compute_mechanical_heating(stress, strain / dt)
        
        return CouplingState(
            temperature=temperature,
            displacement=displacement,
            stress=stress,
            strain=strain,
            thermal_strain=thermal_strain,
            mechanical_heat=mechanical_heat
        )
    
    def _solve_coupled_staggered(self, K_thermal: csr_matrix, K_mechanical: csr_matrix,
                                C_thermal: csr_matrix, C_coupling: csr_matrix,
                                heat_source: np.ndarray, body_force: np.ndarray,
                                dt: float, boundary_conditions: Dict) -> CouplingState:
        """交错求解耦合系统"""
        # 先求解热传导
        temperature = self._solve_thermal_step(
            K_thermal, C_thermal, C_coupling, self.current_state.displacement, 
            heat_source, dt, boundary_conditions
        )
        
        # 再求解力学
        displacement = self._solve_mechanical_step(
            K_mechanical, temperature, body_force, boundary_conditions
        )
        
        # 计算其他场
        stress = self._compute_stress(displacement, temperature)
        strain = self._compute_strain(displacement)
        thermal_strain = self.thermal_expansion * (temperature - 293.15)
        mechanical_heat = self.compute_mechanical_heating(stress, strain / dt)
        
        return CouplingState(
            temperature=temperature,
            displacement=displacement,
            stress=stress,
            strain=strain,
            thermal_strain=thermal_strain,
            mechanical_heat=mechanical_heat
        )
    
    def _solve_thermal_step(self, K_thermal: csr_matrix, C_thermal: csr_matrix,
                           C_coupling: csr_matrix, displacement: np.ndarray,
                           heat_source: np.ndarray, dt: float, 
                           boundary_conditions: Dict) -> np.ndarray:
        """求解热传导步骤"""
        start_time = time.time()
        
        # 构建热传导系统
        A_thermal = C_thermal / dt + K_thermal
        b_thermal = heat_source + C_thermal @ self.current_state.temperature / dt
        
        # 添加耦合项
        b_thermal += C_coupling @ displacement
        
        # 应用边界条件
        A_thermal, b_thermal = self._apply_thermal_boundary_conditions(
            A_thermal, b_thermal, boundary_conditions
        )
        
        # 求解
        temperature = spsolve(A_thermal, b_thermal)
        
        self.performance_stats['thermal_solve_time'] = time.time() - start_time
        return temperature
    
    def _solve_mechanical_step(self, K_mechanical: csr_matrix, temperature: np.ndarray,
                              body_force: np.ndarray, boundary_conditions: Dict) -> np.ndarray:
        """求解力学步骤"""
        start_time = time.time()
        
        # 计算热应力
        thermal_stress = self.compute_thermal_stress(temperature)
        thermal_force = self._compute_thermal_force(thermal_stress)
        
        # 构建力学系统
        A_mechanical = K_mechanical
        b_mechanical = body_force + thermal_force
        
        # 应用边界条件
        A_mechanical, b_mechanical = self._apply_mechanical_boundary_conditions(
            A_mechanical, b_mechanical, boundary_conditions
        )
        
        # 求解
        displacement = spsolve(A_mechanical, b_mechanical)
        
        self.performance_stats['mechanical_solve_time'] = time.time() - start_time
        return displacement
    
    def _compute_adaptive_timestep(self, current_state: CouplingState, current_dt: float) -> float:
        """计算自适应时间步长"""
        # 基于CFL条件的自适应时间步长
        if hasattr(self.mesh, 'get_min_element_size'):
            min_element_size = self.mesh.get_min_element_size()
        else:
            min_element_size = 1e-3  # 默认值
        
        # 计算特征速度
        thermal_velocity = self.thermal_diffusivity / min_element_size
        mechanical_velocity = np.sqrt(self.young_modulus / self.density)
        
        # 选择最小特征速度
        characteristic_velocity = min(thermal_velocity, mechanical_velocity)
        
        # 计算新的时间步长
        new_dt = self.config.cfl_number * min_element_size / characteristic_velocity
        
        # 限制在允许范围内
        new_dt = max(self.config.min_timestep, min(new_dt, self.config.max_timestep))
        
        return new_dt
    
    def _assemble_thermal_matrix(self) -> csr_matrix:
        """组装热传导矩阵"""
        # 简化的热传导矩阵组装
        n_points = len(self.mesh.nodes) if hasattr(self.mesh, 'nodes') else 100
        K_thermal = csr_matrix((n_points, n_points))
        
        # 这里应该根据具体的网格结构组装矩阵
        # 简化实现：使用拉普拉斯算子
        for i in range(n_points):
            if i > 0:
                K_thermal[i, i-1] = -self.thermal_conductivity
            K_thermal[i, i] = 2 * self.thermal_conductivity
            if i < n_points - 1:
                K_thermal[i, i+1] = -self.thermal_conductivity
        
        return K_thermal
    
    def _assemble_mechanical_matrix(self) -> csr_matrix:
        """组装力学刚度矩阵"""
        # 简化的力学刚度矩阵组装
        n_points = len(self.mesh.nodes) if hasattr(self.mesh, 'nodes') else 100
        K_mechanical = csr_matrix((n_points, n_points))
        
        # 这里应该根据具体的网格结构组装矩阵
        # 简化实现：使用弹性算子
        for i in range(n_points):
            if i > 0:
                K_mechanical[i, i-1] = -self.young_modulus
            K_mechanical[i, i] = 2 * self.young_modulus
            if i < n_points - 1:
                K_mechanical[i, i+1] = -self.young_modulus
        
        return K_mechanical
    
    def _assemble_thermal_capacity_matrix(self) -> csr_matrix:
        """组装热容矩阵"""
        n_points = len(self.mesh.nodes) if hasattr(self.mesh, 'nodes') else 100
        C_thermal = csr_matrix((n_points, n_points))
        
        # 对角矩阵
        for i in range(n_points):
            C_thermal[i, i] = self.density * self.heat_capacity
        
        return C_thermal
    
    def _assemble_coupling_matrix_internal(self, dt: float) -> csr_matrix:
        """组装内部耦合矩阵"""
        n_points = len(self.mesh.nodes) if hasattr(self.mesh, 'nodes') else 100
        C_coupling = csr_matrix((n_points, n_points))
        
        # 耦合矩阵：热膨胀对力学的影响
        for i in range(n_points):
            C_coupling[i, i] = self.thermal_expansion * self.young_modulus
        
        return C_coupling
    
    def _compute_heat_source_vector(self, heat_source: Optional[Callable]) -> np.ndarray:
        """计算热源向量"""
        n_points = len(self.mesh.nodes) if hasattr(self.mesh, 'nodes') else 100
        heat_source_vector = np.zeros(n_points)
        
        if heat_source is not None:
            for i in range(n_points):
                heat_source_vector[i] = heat_source(i, self.current_state.time if self.current_state else 0.0)
        
        return heat_source_vector
    
    def _compute_body_force_vector(self, body_force: Optional[Callable]) -> np.ndarray:
        """计算体力向量"""
        n_points = len(self.mesh.nodes) if hasattr(self.mesh, 'nodes') else 100
        body_force_vector = np.zeros(n_points)
        
        if body_force is not None:
            for i in range(n_points):
                body_force_vector[i] = body_force(i, self.current_state.time if self.current_state else 0.0)
        
        return body_force_vector
    
    def _compute_thermal_force(self, thermal_stress: np.ndarray) -> np.ndarray:
        """计算热力"""
        # 简化的热力计算
        thermal_force = np.zeros_like(thermal_stress[:, 0])
        
        # 计算热力的梯度
        for i in range(len(thermal_force)):
            if i > 0:
                thermal_force[i] += thermal_stress[i, 0] - thermal_stress[i-1, 0]
            if i < len(thermal_force) - 1:
                thermal_force[i] += thermal_stress[i+1, 0] - thermal_stress[i, 0]
        
        return thermal_force
    
    def _compute_stress(self, displacement: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """计算应力"""
        n_points = len(displacement)
        stress = np.zeros((n_points, 3))
        
        # 计算应变
        strain = self._compute_strain(displacement)
        
        # 计算热应变
        thermal_strain = self.thermal_expansion * (temperature - 293.15)
        
        # 计算应力 (平面应力)
        for i in range(n_points):
            stress[i, 0] = self.young_modulus / (1 - self.poisson_ratio**2) * (
                strain[i, 0] + self.poisson_ratio * strain[i, 1] - thermal_strain[i]
            )
            stress[i, 1] = self.young_modulus / (1 - self.poisson_ratio**2) * (
                strain[i, 1] + self.poisson_ratio * strain[i, 0] - thermal_strain[i]
            )
            stress[i, 2] = self.young_modulus / (2 * (1 + self.poisson_ratio)) * strain[i, 2]
        
        return stress
    
    def _compute_strain(self, displacement: np.ndarray) -> np.ndarray:
        """计算应变"""
        n_points = len(displacement)
        strain = np.zeros((n_points, 3))  # εxx, εyy, εxy
        
        # 简化的应变计算 (一维情况)
        for i in range(n_points):
            if i > 0:
                strain[i, 0] = (displacement[i] - displacement[i-1]) / 1.0  # 假设单元长度为1
            if i < n_points - 1:
                strain[i, 0] = (displacement[i+1] - displacement[i]) / 1.0
        
        return strain
    
    def _apply_thermal_boundary_conditions(self, A: csr_matrix, b: np.ndarray,
                                         boundary_conditions: Dict) -> Tuple[csr_matrix, np.ndarray]:
        """应用热边界条件"""
        # 简化的边界条件应用
        thermal_bc = boundary_conditions.get('thermal', {})
        
        for node_id, value in thermal_bc.items():
            if node_id < A.shape[0]:
                A[node_id, :] = 0
                A[node_id, node_id] = 1
                b[node_id] = value
        
        return A, b
    
    def _apply_mechanical_boundary_conditions(self, A: csr_matrix, b: np.ndarray,
                                            boundary_conditions: Dict) -> Tuple[csr_matrix, np.ndarray]:
        """应用力学边界条件"""
        # 简化的边界条件应用
        mechanical_bc = boundary_conditions.get('mechanical', {})
        
        for node_id, value in mechanical_bc.items():
            if node_id < A.shape[0]:
                A[node_id, :] = 0
                A[node_id, node_id] = 1
                b[node_id] = value
        
        return A, b
    
    def _apply_monolithic_boundary_conditions(self, A: csr_matrix, b: np.ndarray,
                                            boundary_conditions: Dict) -> Tuple[csr_matrix, np.ndarray]:
        """应用整体边界条件"""
        # 应用热边界条件
        A, b = self._apply_thermal_boundary_conditions(A, b, boundary_conditions)
        
        # 应用力学边界条件
        A, b = self._apply_mechanical_boundary_conditions(A, b, boundary_conditions)
        
        return A, b
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        return self.performance_stats.copy()
    
    def visualize_coupling_results(self, solution_history: List[CouplingState]):
        """可视化耦合结果"""
        try:
            import matplotlib.pyplot as plt
            
            times = [state.time for state in solution_history]
            temperatures = [np.mean(state.temperature) for state in solution_history]
            displacements = [np.mean(state.displacement) for state in solution_history]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # 温度演化
            ax1.plot(times, temperatures, 'b-', linewidth=2)
            ax1.set_xlabel('时间 (s)')
            ax1.set_ylabel('平均温度 (K)')
            ax1.set_title('温度演化')
            ax1.grid(True)
            
            # 位移演化
            ax2.plot(times, displacements, 'r-', linewidth=2)
            ax2.set_xlabel('时间 (s)')
            ax2.set_ylabel('平均位移 (m)')
            ax2.set_title('位移演化')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib未安装，无法可视化结果")


def create_thermo_mechanical_coupling(mesh, **kwargs) -> ThermoMechanicalCoupling:
    """创建热-力学耦合求解器"""
    return ThermoMechanicalCoupling(mesh, **kwargs) 