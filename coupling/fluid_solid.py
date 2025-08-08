"""
流体-固体耦合模块 (Fluid-Structure Interaction, FSI)

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
class FluidSolidState:
    """流体-固体耦合状态"""
    # 流体场
    fluid_velocity: np.ndarray  # 流体速度场
    fluid_pressure: np.ndarray  # 流体压力场
    fluid_temperature: np.ndarray  # 流体温度场
    
    # 固体场
    solid_displacement: np.ndarray  # 固体位移场
    solid_velocity: np.ndarray  # 固体速度场
    solid_stress: np.ndarray  # 固体应力场
    
    # 耦合场
    interface_force: np.ndarray  # 界面力
    interface_displacement: np.ndarray  # 界面位移
    mesh_deformation: np.ndarray  # 网格变形
    
    # 求解信息
    time: float = 0.0
    iteration: int = 0
    residual_norm: float = 0.0
    convergence_status: bool = False


@dataclass
class FSIConfig:
    """FSI配置"""
    solver_type: str = 'partitioned'  # 求解器类型: 'partitioned', 'monolithic', 'staggered'
    max_iterations: int = 10  # 最大迭代次数
    tolerance: float = 1e-6  # 收敛容差
    relaxation_factor: float = 0.5  # 松弛因子
    adaptive_timestep: bool = True  # 自适应时间步长
    min_timestep: float = 1e-6  # 最小时间步长
    max_timestep: float = 1e-3  # 最大时间步长
    cfl_number: float = 0.5  # CFL数
    interface_tracking: bool = True  # 界面追踪
    mesh_deformation: bool = True  # 网格变形
    parallel_solver: bool = True  # 使用并行求解器
    preconditioner: str = 'jacobi'  # 预处理器类型


class FluidSolver(ABC):
    """流体求解器抽象基类"""
    
    def __init__(self, mesh, 
                 fluid_density: float = 1000.0,
                 fluid_viscosity: float = 1e-3,
                 fluid_thermal_conductivity: float = 0.6,
                 fluid_heat_capacity: float = 4186.0):
        """
        初始化流体求解器
        
        Args:
            mesh: 流体网格
            fluid_density: 流体密度 (kg/m³)
            fluid_viscosity: 流体粘度 (Pa·s)
            fluid_thermal_conductivity: 流体热导率 (W/m/K)
            fluid_heat_capacity: 流体比热容 (J/kg/K)
        """
        self.mesh = mesh
        self.fluid_density = fluid_density
        self.fluid_viscosity = fluid_viscosity
        self.fluid_thermal_conductivity = fluid_thermal_conductivity
        self.fluid_heat_capacity = fluid_heat_capacity
        
        # 计算流体参数
        self.kinematic_viscosity = fluid_viscosity / fluid_density
        self.thermal_diffusivity = fluid_thermal_conductivity / (fluid_density * fluid_heat_capacity)
        
        # MPI相关
        self.comm = MPI.COMM_WORLD if HAS_MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1
        
        # 性能监控
        self.performance_stats = {
            'solve_time': 0.0,
            'pressure_solve_time': 0.0,
            'velocity_solve_time': 0.0,
            'energy_solve_time': 0.0
        }
    
    @abstractmethod
    def solve_fluid_step(self, 
                        velocity: np.ndarray,
                        pressure: np.ndarray,
                        temperature: np.ndarray,
                        boundary_conditions: Dict,
                        dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """求解流体步骤"""
        pass
    
    @abstractmethod
    def compute_interface_force(self, 
                              pressure: np.ndarray,
                              velocity: np.ndarray,
                              interface_nodes: np.ndarray) -> np.ndarray:
        """计算界面力"""
        pass
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        return self.performance_stats.copy()


class NavierStokesSolver(FluidSolver):
    """Navier-Stokes流体求解器"""
    
    def __init__(self, mesh, 
                 fluid_density: float = 1000.0,
                 fluid_viscosity: float = 1e-3,
                 fluid_thermal_conductivity: float = 0.6,
                 fluid_heat_capacity: float = 4186.0):
        super().__init__(mesh, fluid_density, fluid_viscosity, fluid_thermal_conductivity, fluid_heat_capacity)
        
        # 初始化矩阵
        self.A_velocity = None
        self.A_pressure = None
        self.A_temperature = None
        self.M_mass = None
        
        # 预处理器
        self.velocity_preconditioner = None
        self.pressure_preconditioner = None
    
    def solve_fluid_step(self, 
                        velocity: np.ndarray,
                        pressure: np.ndarray,
                        temperature: np.ndarray,
                        boundary_conditions: Dict,
                        dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """求解Navier-Stokes方程"""
        start_time = time.time()
        
        # 预测速度
        predicted_velocity = self._predict_velocity(velocity, pressure, dt)
        
        # 求解压力修正
        pressure_correction = self._solve_pressure_correction(predicted_velocity, dt)
        
        # 修正速度
        corrected_velocity = predicted_velocity - dt / self.fluid_density * self._compute_pressure_gradient(pressure_correction)
        
        # 求解能量方程
        new_temperature = self._solve_energy_equation(temperature, corrected_velocity, dt)
        
        # 更新压力
        new_pressure = pressure + pressure_correction
        
        self.performance_stats['solve_time'] = time.time() - start_time
        
        return corrected_velocity, new_pressure, new_temperature
    
    def compute_interface_force(self, 
                              pressure: np.ndarray,
                              velocity: np.ndarray,
                              interface_nodes: np.ndarray) -> np.ndarray:
        """计算界面力"""
        interface_force = np.zeros((len(interface_nodes), 2))  # 2D情况
        
        for i, node_id in enumerate(interface_nodes):
            # 压力力
            pressure_force = pressure[node_id] * self._compute_interface_normal(node_id)
            
            # 粘性力
            velocity_gradient = self._compute_velocity_gradient(velocity, node_id)
            viscous_force = self.fluid_viscosity * velocity_gradient
            
            interface_force[i] = pressure_force + viscous_force
        
        return interface_force
    
    def _predict_velocity(self, velocity: np.ndarray, pressure: np.ndarray, dt: float) -> np.ndarray:
        """预测速度"""
        start_time = time.time()
        
        # 简化的速度预测
        predicted_velocity = velocity.copy()
        
        # 添加对流项
        convection_term = self._compute_convection_term(velocity)
        predicted_velocity += dt * convection_term
        
        # 添加扩散项
        diffusion_term = self.kinematic_viscosity * self._compute_laplacian(velocity)
        predicted_velocity += dt * diffusion_term
        
        # 添加压力梯度项
        pressure_gradient = self._compute_pressure_gradient(pressure)
        predicted_velocity -= dt / self.fluid_density * pressure_gradient
        
        self.performance_stats['velocity_solve_time'] = time.time() - start_time
        return predicted_velocity
    
    def _solve_pressure_correction(self, predicted_velocity: np.ndarray, dt: float) -> np.ndarray:
        """求解压力修正"""
        start_time = time.time()
        
        # 计算散度
        divergence = self._compute_divergence(predicted_velocity)
        
        # 构建压力修正方程
        A_pressure = self._assemble_pressure_matrix()
        b_pressure = -self.fluid_density / dt * divergence
        
        # 求解压力修正
        pressure_correction = spsolve(A_pressure, b_pressure)
        
        self.performance_stats['pressure_solve_time'] = time.time() - start_time
        return pressure_correction
    
    def _solve_energy_equation(self, temperature: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
        """求解能量方程"""
        start_time = time.time()
        
        # 简化的能量方程求解
        new_temperature = temperature.copy()
        
        # 添加对流项
        convection_term = self._compute_temperature_convection(temperature, velocity)
        new_temperature += dt * convection_term
        
        # 添加扩散项
        diffusion_term = self.thermal_diffusivity * self._compute_laplacian(temperature)
        new_temperature += dt * diffusion_term
        
        self.performance_stats['energy_solve_time'] = time.time() - start_time
        return new_temperature
    
    def _compute_convection_term(self, velocity: np.ndarray) -> np.ndarray:
        """计算对流项"""
        # 简化的对流项计算
        convection = np.zeros_like(velocity)
        
        for i in range(len(velocity)):
            if i > 0 and i < len(velocity) - 1:
                convection[i] = velocity[i] * (velocity[i+1] - velocity[i-1]) / 2.0
        
        return convection
    
    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """计算拉普拉斯算子"""
        laplacian = np.zeros_like(field)
        
        for i in range(len(field)):
            if i > 0 and i < len(field) - 1:
                laplacian[i] = (field[i+1] - 2*field[i] + field[i-1]) / 1.0**2  # 假设网格间距为1
        
        return laplacian
    
    def _compute_pressure_gradient(self, pressure: np.ndarray) -> np.ndarray:
        """计算压力梯度"""
        gradient = np.zeros_like(pressure)
        
        for i in range(len(pressure)):
            if i > 0 and i < len(pressure) - 1:
                gradient[i] = (pressure[i+1] - pressure[i-1]) / 2.0
        
        return gradient
    
    def _compute_divergence(self, velocity: np.ndarray) -> np.ndarray:
        """计算散度"""
        divergence = np.zeros_like(velocity)
        
        for i in range(len(velocity)):
            if i > 0 and i < len(velocity) - 1:
                divergence[i] = (velocity[i+1] - velocity[i-1]) / 2.0
        
        return divergence
    
    def _compute_temperature_convection(self, temperature: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """计算温度对流"""
        convection = np.zeros_like(temperature)
        
        for i in range(len(temperature)):
            if i > 0 and i < len(temperature) - 1:
                convection[i] = velocity[i] * (temperature[i+1] - temperature[i-1]) / 2.0
        
        return convection
    
    def _compute_interface_normal(self, node_id: int) -> np.ndarray:
        """计算界面法向量"""
        # 简化的界面法向量计算
        return np.array([0.0, 1.0])  # 假设界面垂直向上
    
    def _compute_velocity_gradient(self, velocity: np.ndarray, node_id: int) -> np.ndarray:
        """计算速度梯度"""
        # 简化的速度梯度计算
        if node_id > 0 and node_id < len(velocity) - 1:
            gradient = (velocity[node_id+1] - velocity[node_id-1]) / 2.0
        else:
            gradient = 0.0
        
        return np.array([gradient, 0.0])
    
    def _assemble_pressure_matrix(self) -> csr_matrix:
        """组装压力矩阵"""
        n_points = len(self.mesh.nodes) if hasattr(self.mesh, 'nodes') else 100
        A_pressure = csr_matrix((n_points, n_points))
        
        # 简化的压力矩阵组装
        for i in range(n_points):
            if i > 0:
                A_pressure[i, i-1] = -1.0
            A_pressure[i, i] = 2.0
            if i < n_points - 1:
                A_pressure[i, i+1] = -1.0
        
        return A_pressure


class SolidSolver(ABC):
    """固体求解器抽象基类"""
    
    def __init__(self, mesh,
                 young_modulus: float = 70e9,
                 poisson_ratio: float = 0.3,
                 density: float = 2700.0,
                 damping_coefficient: float = 0.0):
        """
        初始化固体求解器
        
        Args:
            mesh: 固体网格
            young_modulus: 杨氏模量 (Pa)
            poisson_ratio: 泊松比
            density: 密度 (kg/m³)
            damping_coefficient: 阻尼系数
        """
        self.mesh = mesh
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.density = density
        self.damping_coefficient = damping_coefficient
        
        # 计算材料参数
        self.lame_lambda = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        self.lame_mu = young_modulus / (2 * (1 + poisson_ratio))
        
        # MPI相关
        self.comm = MPI.COMM_WORLD if HAS_MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1
        
        # 性能监控
        self.performance_stats = {
            'solve_time': 0.0,
            'mass_assembly_time': 0.0,
            'stiffness_assembly_time': 0.0
        }
        
        # 初始化矩阵
        self.M_mass = None
        self.K_stiffness = None
    
    @abstractmethod
    def solve_solid_step(self,
                        displacement: np.ndarray,
                        velocity: np.ndarray,
                        stress: np.ndarray,
                        interface_force: np.ndarray,
                        boundary_conditions: Dict,
                        dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """求解固体步骤"""
        pass
    
    @abstractmethod
    def compute_interface_displacement(self,
                                    displacement: np.ndarray,
                                    interface_nodes: np.ndarray) -> np.ndarray:
        """计算界面位移"""
        pass
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        return self.performance_stats.copy()


class ElasticSolidSolver(SolidSolver):
    """弹性固体求解器"""
    
    def __init__(self, mesh,
                 young_modulus: float = 70e9,
                 poisson_ratio: float = 0.3,
                 density: float = 2700.0,
                 damping_coefficient: float = 0.0):
        super().__init__(mesh, young_modulus, poisson_ratio, density, damping_coefficient)
    
    def solve_solid_step(self,
                        displacement: np.ndarray,
                        velocity: np.ndarray,
                        stress: np.ndarray,
                        interface_force: np.ndarray,
                        boundary_conditions: Dict,
                        dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """求解弹性固体步骤"""
        start_time = time.time()
        
        # 组装质量矩阵和刚度矩阵
        if self.M_mass is None:
            self.M_mass = self._assemble_mass_matrix()
        if self.K_stiffness is None:
            self.K_stiffness = self._assemble_stiffness_matrix()
        
        # 计算外部力
        external_force = self._compute_external_force(interface_force)
        
        # 构建动力学系统
        A_system = self.M_mass / dt**2 + self.damping_coefficient * self.M_mass / dt + self.K_stiffness
        b_system = external_force + self.M_mass @ (2*displacement - displacement) / dt**2
        
        # 应用边界条件
        A_system, b_system = self._apply_solid_boundary_conditions(A_system, b_system, boundary_conditions)
        
        # 求解位移
        new_displacement = spsolve(A_system, b_system)
        
        # 计算速度和应力
        new_velocity = (new_displacement - displacement) / dt
        new_stress = self._compute_stress(new_displacement)
        
        self.performance_stats['solve_time'] = time.time() - start_time
        
        return new_displacement, new_velocity, new_stress
    
    def compute_interface_displacement(self,
                                    displacement: np.ndarray,
                                    interface_nodes: np.ndarray) -> np.ndarray:
        """计算界面位移"""
        interface_displacement = np.zeros((len(interface_nodes), 2))  # 2D情况
        
        for i, node_id in enumerate(interface_nodes):
            interface_displacement[i] = displacement[node_id]
        
        return interface_displacement
    
    def _assemble_mass_matrix(self) -> csr_matrix:
        """组装质量矩阵"""
        n_points = len(self.mesh.nodes) if hasattr(self.mesh, 'nodes') else 100
        M_mass = csr_matrix((n_points, n_points))
        
        # 简化的质量矩阵组装
        for i in range(n_points):
            M_mass[i, i] = self.density
        
        return M_mass
    
    def _assemble_stiffness_matrix(self) -> csr_matrix:
        """组装刚度矩阵"""
        n_points = len(self.mesh.nodes) if hasattr(self.mesh, 'nodes') else 100
        K_stiffness = csr_matrix((n_points, n_points))
        
        # 简化的刚度矩阵组装
        for i in range(n_points):
            if i > 0:
                K_stiffness[i, i-1] = -self.young_modulus
            K_stiffness[i, i] = 2 * self.young_modulus
            if i < n_points - 1:
                K_stiffness[i, i+1] = -self.young_modulus
        
        return K_stiffness
    
    def _compute_external_force(self, interface_force: np.ndarray) -> np.ndarray:
        """计算外部力"""
        n_points = len(self.mesh.nodes) if hasattr(self.mesh, 'nodes') else 100
        external_force = np.zeros(n_points)
        
        # 将界面力分配到网格节点
        for i, force in enumerate(interface_force):
            if i < n_points:
                external_force[i] = force[0]  # 只考虑x方向力
        
        return external_force
    
    def _compute_stress(self, displacement: np.ndarray) -> np.ndarray:
        """计算应力"""
        n_points = len(displacement)
        stress = np.zeros((n_points, 3))  # σxx, σyy, σxy
        
        # 计算应变
        strain = self._compute_strain(displacement)
        
        # 计算应力 (平面应力)
        for i in range(n_points):
            stress[i, 0] = self.young_modulus / (1 - self.poisson_ratio**2) * (
                strain[i, 0] + self.poisson_ratio * strain[i, 1]
            )
            stress[i, 1] = self.young_modulus / (1 - self.poisson_ratio**2) * (
                strain[i, 1] + self.poisson_ratio * strain[i, 0]
            )
            stress[i, 2] = self.young_modulus / (2 * (1 + self.poisson_ratio)) * strain[i, 2]
        
        return stress
    
    def _compute_strain(self, displacement: np.ndarray) -> np.ndarray:
        """计算应变"""
        n_points = len(displacement)
        strain = np.zeros((n_points, 3))  # εxx, εyy, εxy
        
        # 简化的应变计算
        for i in range(n_points):
            if i > 0 and i < n_points - 1:
                strain[i, 0] = (displacement[i+1] - displacement[i-1]) / 2.0
        
        return strain
    
    def _apply_solid_boundary_conditions(self, A: csr_matrix, b: np.ndarray,
                                       boundary_conditions: Dict) -> Tuple[csr_matrix, np.ndarray]:
        """应用固体边界条件"""
        solid_bc = boundary_conditions.get('solid', {})
        
        for node_id, value in solid_bc.items():
            if node_id < A.shape[0]:
                A[node_id, :] = 0
                A[node_id, node_id] = 1
                b[node_id] = value
        
        return A, b


class FluidSolidCoupling:
    """流体-固体耦合求解器"""
    
    def __init__(self, 
                 fluid_mesh,
                 solid_mesh,
                 fluid_solver: FluidSolver,
                 solid_solver: SolidSolver,
                 interface_nodes: np.ndarray,
                 config: FSIConfig = None):
        """
        初始化流体-固体耦合求解器
        
        Args:
            fluid_mesh: 流体网格
            solid_mesh: 固体网格
            fluid_solver: 流体求解器
            solid_solver: 固体求解器
            interface_nodes: 界面节点
            config: FSI配置
        """
        self.fluid_mesh = fluid_mesh
        self.solid_mesh = solid_mesh
        self.fluid_solver = fluid_solver
        self.solid_solver = solid_solver
        self.interface_nodes = interface_nodes
        self.config = config or FSIConfig()
        
        # MPI相关
        self.comm = MPI.COMM_WORLD if HAS_MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1
        
        # 性能监控
        self.performance_stats = {
            'total_solve_time': 0.0,
            'fluid_solve_time': 0.0,
            'solid_solve_time': 0.0,
            'coupling_time': 0.0,
            'interface_tracking_time': 0.0,
            'mesh_deformation_time': 0.0,
            'iterations': 0,
            'residual_norm': 0.0
        }
        
        # 求解器状态
        self.current_state = None
        self.previous_state = None
    
    def solve_coupled_system(self,
                           initial_state: FluidSolidState,
                           boundary_conditions: Dict,
                           time_steps: int,
                           dt: float,
                           fluid_source: Optional[Callable] = None,
                           solid_source: Optional[Callable] = None) -> List[FluidSolidState]:
        """
        求解耦合系统
        
        Args:
            initial_state: 初始状态
            boundary_conditions: 边界条件
            time_steps: 时间步数
            dt: 时间步长
            fluid_source: 流体源项
            solid_source: 固体源项
            
        Returns:
            solution_history: 解的历史
        """
        solution_history = []
        current_state = initial_state
        
        # 时间步进
        for step in range(time_steps):
            print(f"FSI时间步 {step + 1}/{time_steps}")
            
            # 自适应时间步长
            if self.config.adaptive_timestep:
                dt = self._compute_adaptive_timestep(current_state, dt)
            
            # 求解耦合步骤
            new_state = self._solve_coupled_step(
                current_state, boundary_conditions, dt, fluid_source, solid_source
            )
            
            # 更新状态
            new_state.time = current_state.time + dt
            new_state.iteration = step
            
            # 保存解
            solution_history.append(new_state)
            current_state = new_state
            
            # 检查收敛性
            if new_state.convergence_status:
                print(f"FSI在时间步 {step + 1} 收敛")
                break
        
        return solution_history
    
    def _solve_coupled_step(self,
                          current_state: FluidSolidState,
                          boundary_conditions: Dict,
                          dt: float,
                          fluid_source: Optional[Callable],
                          solid_source: Optional[Callable]) -> FluidSolidState:
        """求解耦合步骤"""
        start_time = time.time()
        
        # 根据求解器类型选择求解策略
        if self.config.solver_type == 'partitioned':
            new_state = self._solve_partitioned_step(
                current_state, boundary_conditions, dt, fluid_source, solid_source
            )
        elif self.config.solver_type == 'monolithic':
            new_state = self._solve_monolithic_step(
                current_state, boundary_conditions, dt, fluid_source, solid_source
            )
        else:  # staggered
            new_state = self._solve_staggered_step(
                current_state, boundary_conditions, dt, fluid_source, solid_source
            )
        
        self.performance_stats['total_solve_time'] = time.time() - start_time
        return new_state
    
    def _solve_partitioned_step(self,
                              current_state: FluidSolidState,
                              boundary_conditions: Dict,
                              dt: float,
                              fluid_source: Optional[Callable],
                              solid_source: Optional[Callable]) -> FluidSolidState:
        """分区求解耦合步骤"""
        # 初始化
        fluid_velocity = current_state.fluid_velocity.copy()
        fluid_pressure = current_state.fluid_pressure.copy()
        fluid_temperature = current_state.fluid_temperature.copy()
        solid_displacement = current_state.solid_displacement.copy()
        solid_velocity = current_state.solid_velocity.copy()
        solid_stress = current_state.solid_stress.copy()
        
        # 迭代求解
        for iteration in range(self.config.max_iterations):
            fluid_velocity_old = fluid_velocity.copy()
            solid_displacement_old = solid_displacement.copy()
            
            # 步骤1: 求解流体
            fluid_velocity, fluid_pressure, fluid_temperature = self.fluid_solver.solve_fluid_step(
                fluid_velocity, fluid_pressure, fluid_temperature, boundary_conditions, dt
            )
            
            # 步骤2: 计算界面力
            interface_force = self.fluid_solver.compute_interface_force(
                fluid_pressure, fluid_velocity, self.interface_nodes
            )
            
            # 步骤3: 求解固体
            solid_displacement, solid_velocity, solid_stress = self.solid_solver.solve_solid_step(
                solid_displacement, solid_velocity, solid_stress, interface_force, boundary_conditions, dt
            )
            
            # 步骤4: 计算界面位移
            interface_displacement = self.solid_solver.compute_interface_displacement(
                solid_displacement, self.interface_nodes
            )
            
            # 步骤5: 更新网格变形
            if self.config.mesh_deformation:
                mesh_deformation = self._update_mesh_deformation(interface_displacement)
            else:
                mesh_deformation = current_state.mesh_deformation.copy()
            
            # 检查收敛性
            fluid_residual = np.linalg.norm(fluid_velocity - fluid_velocity_old)
            solid_residual = np.linalg.norm(solid_displacement - solid_displacement_old)
            residual_norm = max(fluid_residual, solid_residual)
            
            if residual_norm < self.config.tolerance:
                convergence_status = True
                break
        
        return FluidSolidState(
            fluid_velocity=fluid_velocity,
            fluid_pressure=fluid_pressure,
            fluid_temperature=fluid_temperature,
            solid_displacement=solid_displacement,
            solid_velocity=solid_velocity,
            solid_stress=solid_stress,
            interface_force=interface_force,
            interface_displacement=interface_displacement,
            mesh_deformation=mesh_deformation,
            residual_norm=residual_norm,
            convergence_status=convergence_status
        )
    
    def _solve_monolithic_step(self,
                             current_state: FluidSolidState,
                             boundary_conditions: Dict,
                             dt: float,
                             fluid_source: Optional[Callable],
                             solid_source: Optional[Callable]) -> FluidSolidState:
        """整体求解耦合步骤"""
        # 构建整体系统矩阵
        A_monolithic, b_monolithic = self._assemble_monolithic_system(
            current_state, boundary_conditions, dt, fluid_source, solid_source
        )
        
        # 求解整体系统
        solution = spsolve(A_monolithic, b_monolithic)
        
        # 提取解
        n_fluid = len(current_state.fluid_velocity)
        n_solid = len(current_state.solid_displacement)
        
        fluid_velocity = solution[:n_fluid]
        fluid_pressure = solution[n_fluid:2*n_fluid]
        fluid_temperature = solution[2*n_fluid:3*n_fluid]
        solid_displacement = solution[3*n_fluid:3*n_fluid+n_solid]
        solid_velocity = solution[3*n_fluid+n_solid:3*n_fluid+2*n_solid]
        
        # 计算其他场
        interface_force = self.fluid_solver.compute_interface_force(
            fluid_pressure, fluid_velocity, self.interface_nodes
        )
        interface_displacement = self.solid_solver.compute_interface_displacement(
            solid_displacement, self.interface_nodes
        )
        mesh_deformation = self._update_mesh_deformation(interface_displacement)
        
        return FluidSolidState(
            fluid_velocity=fluid_velocity,
            fluid_pressure=fluid_pressure,
            fluid_temperature=fluid_temperature,
            solid_displacement=solid_displacement,
            solid_velocity=solid_velocity,
            solid_stress=np.zeros_like(solid_displacement),
            interface_force=interface_force,
            interface_displacement=interface_displacement,
            mesh_deformation=mesh_deformation,
            convergence_status=True
        )
    
    def _solve_staggered_step(self,
                            current_state: FluidSolidState,
                            boundary_conditions: Dict,
                            dt: float,
                            fluid_source: Optional[Callable],
                            solid_source: Optional[Callable]) -> FluidSolidState:
        """交错求解耦合步骤"""
        # 步骤1: 求解流体
        fluid_velocity, fluid_pressure, fluid_temperature = self.fluid_solver.solve_fluid_step(
            current_state.fluid_velocity, current_state.fluid_pressure, 
            current_state.fluid_temperature, boundary_conditions, dt
        )
        
        # 步骤2: 计算界面力
        interface_force = self.fluid_solver.compute_interface_force(
            fluid_pressure, fluid_velocity, self.interface_nodes
        )
        
        # 步骤3: 求解固体
        solid_displacement, solid_velocity, solid_stress = self.solid_solver.solve_solid_step(
            current_state.solid_displacement, current_state.solid_velocity,
            current_state.solid_stress, interface_force, boundary_conditions, dt
        )
        
        # 步骤4: 计算界面位移和网格变形
        interface_displacement = self.solid_solver.compute_interface_displacement(
            solid_displacement, self.interface_nodes
        )
        mesh_deformation = self._update_mesh_deformation(interface_displacement)
        
        return FluidSolidState(
            fluid_velocity=fluid_velocity,
            fluid_pressure=fluid_pressure,
            fluid_temperature=fluid_temperature,
            solid_displacement=solid_displacement,
            solid_velocity=solid_velocity,
            solid_stress=solid_stress,
            interface_force=interface_force,
            interface_displacement=interface_displacement,
            mesh_deformation=mesh_deformation,
            convergence_status=True
        )
    
    def _compute_adaptive_timestep(self, current_state: FluidSolidState, current_dt: float) -> float:
        """计算自适应时间步长"""
        # 基于CFL条件的自适应时间步长
        if hasattr(self.fluid_mesh, 'get_min_element_size'):
            min_element_size = self.fluid_mesh.get_min_element_size()
        else:
            min_element_size = 1e-3  # 默认值
        
        # 计算特征速度
        fluid_velocity = np.max(np.abs(current_state.fluid_velocity))
        solid_velocity = np.max(np.abs(current_state.solid_velocity))
        characteristic_velocity = max(fluid_velocity, solid_velocity)
        
        # 计算新的时间步长
        new_dt = self.config.cfl_number * min_element_size / characteristic_velocity
        
        # 限制在允许范围内
        new_dt = max(self.config.min_timestep, min(new_dt, self.config.max_timestep))
        
        return new_dt
    
    def _update_mesh_deformation(self, interface_displacement: np.ndarray) -> np.ndarray:
        """更新网格变形"""
        # 简化的网格变形计算
        n_points = len(self.fluid_mesh.nodes) if hasattr(self.fluid_mesh, 'nodes') else 100
        mesh_deformation = np.zeros((n_points, 2))
        
        # 将界面位移插值到整个网格
        for i in range(n_points):
            if i < len(interface_displacement):
                mesh_deformation[i] = interface_displacement[i]
            else:
                # 简化的插值
                mesh_deformation[i] = interface_displacement[-1] if len(interface_displacement) > 0 else [0, 0]
        
        return mesh_deformation
    
    def _assemble_monolithic_system(self,
                                  current_state: FluidSolidState,
                                  boundary_conditions: Dict,
                                  dt: float,
                                  fluid_source: Optional[Callable],
                                  solid_source: Optional[Callable]) -> Tuple[csr_matrix, np.ndarray]:
        """组装整体系统矩阵"""
        # 简化的整体系统组装
        n_fluid = len(current_state.fluid_velocity)
        n_solid = len(current_state.solid_displacement)
        n_total = 3 * n_fluid + 2 * n_solid
        
        A_monolithic = lil_matrix((n_total, n_total))
        b_monolithic = np.zeros(n_total)
        
        # 这里应该实现完整的整体系统组装
        # 简化实现：使用对角矩阵
        for i in range(n_total):
            A_monolithic[i, i] = 1.0
        
        return A_monolithic.tocsr(), b_monolithic
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        return self.performance_stats.copy()
    
    def visualize_coupling_results(self, solution_history: List[FluidSolidState]):
        """可视化耦合结果"""
        try:
            import matplotlib.pyplot as plt
            
            times = [state.time for state in solution_history]
            fluid_velocities = [np.mean(state.fluid_velocity) for state in solution_history]
            solid_displacements = [np.mean(state.solid_displacement) for state in solution_history]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # 流体速度演化
            ax1.plot(times, fluid_velocities, 'b-', linewidth=2)
            ax1.set_xlabel('时间 (s)')
            ax1.set_ylabel('平均流体速度 (m/s)')
            ax1.set_title('流体速度演化')
            ax1.grid(True)
            
            # 固体位移演化
            ax2.plot(times, solid_displacements, 'r-', linewidth=2)
            ax2.set_xlabel('时间 (s)')
            ax2.set_ylabel('平均固体位移 (m)')
            ax2.set_title('固体位移演化')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib未安装，无法可视化结果")


def create_fluid_solid_coupling(fluid_mesh, solid_mesh, interface_nodes: np.ndarray, **kwargs) -> FluidSolidCoupling:
    """创建流体-固体耦合求解器"""
    # 创建默认求解器
    fluid_solver = NavierStokesSolver(fluid_mesh)
    solid_solver = ElasticSolidSolver(solid_mesh)
    
    # 创建配置
    config = FSIConfig(**kwargs)
    
    return FluidSolidCoupling(fluid_mesh, solid_mesh, fluid_solver, solid_solver, interface_nodes, config)


def demo_fluid_solid_coupling():
    """演示流体-固体耦合"""
    # 创建简单的网格
    class SimpleMesh:
        def __init__(self, n_points):
            self.nodes = np.linspace(0, 1, n_points)
            self.n_points = n_points
    
    # 创建网格
    fluid_mesh = SimpleMesh(50)
    solid_mesh = SimpleMesh(30)
    interface_nodes = np.arange(20, 30)  # 界面节点
    
    # 创建耦合求解器
    fsi_solver = create_fluid_solid_coupling(fluid_mesh, solid_mesh, interface_nodes)
    
    # 创建初始状态
    initial_state = FluidSolidState(
        fluid_velocity=np.zeros(50),
        fluid_pressure=np.zeros(50),
        fluid_temperature=np.ones(50) * 293.15,
        solid_displacement=np.zeros(30),
        solid_velocity=np.zeros(30),
        solid_stress=np.zeros((30, 3)),
        interface_force=np.zeros((10, 2)),
        interface_displacement=np.zeros((10, 2)),
        mesh_deformation=np.zeros((50, 2))
    )
    
    # 定义边界条件
    boundary_conditions = {
        'fluid': {0: 1.0, 49: 0.0},  # 入口和出口
        'solid': {0: 0.0, 29: 0.0}   # 固定边界
    }
    
    # 求解耦合系统
    solution_history = fsi_solver.solve_coupled_system(
        initial_state, boundary_conditions, time_steps=10, dt=1e-3
    )
    
    # 可视化结果
    fsi_solver.visualize_coupling_results(solution_history)
    
    return solution_history


if __name__ == "__main__":
    demo_fluid_solid_coupling()