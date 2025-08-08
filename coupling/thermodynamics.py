"""
热力学耦合模块 (Thermodynamics Coupling)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# 可选依赖
try:
    from scipy.sparse import csr_matrix, lil_matrix
    from scipy.sparse.linalg import spsolve, gmres
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False


@dataclass
class ThermodynamicState:
    """热力学状态"""
    temperature: np.ndarray  # 温度场
    pressure: np.ndarray  # 压力场
    density: np.ndarray  # 密度场
    velocity: np.ndarray  # 速度场
    phase_fraction: np.ndarray  # 相分数
    enthalpy: np.ndarray  # 焓场
    entropy: np.ndarray  # 熵场
    heat_flux: np.ndarray  # 热流密度
    time: float = 0.0


class HeatTransferSolver:
    """热传递求解器"""
    
    def __init__(self, mesh,
                 thermal_conductivity: float = 3.0,
                 heat_capacity: float = 1000.0,
                 density: float = 2700.0,
                 emissivity: float = 0.8,
                 convection_coefficient: float = 10.0):
        self.mesh = mesh
        self.thermal_conductivity = thermal_conductivity
        self.heat_capacity = heat_capacity
        self.density = density
        self.emissivity = emissivity
        self.convection_coefficient = convection_coefficient
        self.stefan_boltzmann = 5.670374419e-8
        self.thermal_diffusivity = thermal_conductivity / (density * heat_capacity)
        
        # MPI相关
        self.comm = MPI.COMM_WORLD if HAS_MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0
    
    def solve_heat_transfer(self, temperature: np.ndarray, velocity: np.ndarray,
                          boundary_conditions: Dict, dt: float,
                          heat_source: Optional[Callable] = None) -> np.ndarray:
        """求解热传递方程"""
        n_points = len(temperature)
        
        # 构建系统矩阵
        conduction_matrix = self._assemble_conduction_matrix()
        convection_matrix = self._assemble_convection_matrix(velocity)
        capacity_matrix = self._assemble_capacity_matrix()
        
        # 计算源项
        source_vector = self._compute_heat_source(temperature, heat_source)
        radiation_vector = self._compute_radiation_term(temperature)
        
        # 组装系统
        system_matrix = capacity_matrix / dt + conduction_matrix + convection_matrix
        system_vector = capacity_matrix @ temperature / dt + source_vector + radiation_vector
        
        # 应用边界条件
        system_matrix, system_vector = self._apply_thermal_boundary_conditions(
            system_matrix, system_vector, boundary_conditions
        )
        
        # 求解
        new_temperature = spsolve(system_matrix, system_vector)
        return new_temperature
    
    def _assemble_conduction_matrix(self) -> csr_matrix:
        """组装热传导矩阵"""
        n_points = self.mesh.n_points
        conduction_matrix = lil_matrix((n_points, n_points))
        
        for i in range(1, n_points - 1):
            conduction_matrix[i, i-1] = -self.thermal_conductivity
            conduction_matrix[i, i] = 2 * self.thermal_conductivity
            conduction_matrix[i, i+1] = -self.thermal_conductivity
        
        return conduction_matrix.tocsr()
    
    def _assemble_convection_matrix(self, velocity: np.ndarray) -> csr_matrix:
        """组装对流矩阵"""
        n_points = self.mesh.n_points
        convection_matrix = lil_matrix((n_points, n_points))
        
        for i in range(1, n_points - 1):
            convection_matrix[i, i-1] = -self.heat_capacity * self.density * velocity[i] / 2
            convection_matrix[i, i+1] = self.heat_capacity * self.density * velocity[i] / 2
        
        return convection_matrix.tocsr()
    
    def _assemble_capacity_matrix(self) -> csr_matrix:
        """组装热容矩阵"""
        n_points = self.mesh.n_points
        capacity_matrix = lil_matrix((n_points, n_points))
        
        for i in range(n_points):
            capacity_matrix[i, i] = self.heat_capacity * self.density
        
        return capacity_matrix.tocsr()
    
    def _compute_heat_source(self, temperature: np.ndarray, heat_source: Optional[Callable]) -> np.ndarray:
        """计算热源项"""
        n_points = len(temperature)
        source_vector = np.zeros(n_points)
        
        if heat_source is not None:
            for i in range(n_points):
                source_vector[i] = heat_source(temperature[i], i)
        
        return source_vector
    
    def _compute_radiation_term(self, temperature: np.ndarray) -> np.ndarray:
        """计算辐射项"""
        return self.emissivity * self.stefan_boltzmann * temperature**4
    
    def _apply_thermal_boundary_conditions(self, A: csr_matrix, b: np.ndarray, 
                                         boundary_conditions: Dict) -> Tuple[csr_matrix, np.ndarray]:
        """应用热边界条件"""
        if 'temperature' in boundary_conditions:
            for node_idx, value in boundary_conditions['temperature'].items():
                A[node_idx, :] = 0
                A[node_idx, node_idx] = 1
                b[node_idx] = value
        
        if 'heat_flux' in boundary_conditions:
            for node_idx, value in boundary_conditions['heat_flux'].items():
                b[node_idx] += value
        
        return A, b


class PhaseChangeSolver:
    """相变求解器"""
    
    def __init__(self, mesh, melting_temperature: float = 273.15,
                 latent_heat: float = 334e3, solid_heat_capacity: float = 2100.0,
                 liquid_heat_capacity: float = 4200.0):
        self.mesh = mesh
        self.melting_temperature = melting_temperature
        self.latent_heat = latent_heat
        self.solid_heat_capacity = solid_heat_capacity
        self.liquid_heat_capacity = liquid_heat_capacity
        self.mushy_zone_width = 5.0
        
        # MPI相关
        self.comm = MPI.COMM_WORLD if HAS_MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0
    
    def compute_phase_fraction(self, temperature: np.ndarray) -> np.ndarray:
        """计算相分数"""
        phase_fraction = np.zeros_like(temperature)
        
        for i, T in enumerate(temperature):
            if T < self.melting_temperature - self.mushy_zone_width / 2:
                phase_fraction[i] = 0.0
            elif T > self.melting_temperature + self.mushy_zone_width / 2:
                phase_fraction[i] = 1.0
            else:
                phase_fraction[i] = (T - (self.melting_temperature - self.mushy_zone_width / 2)) / self.mushy_zone_width
        
        return phase_fraction
    
    def compute_effective_heat_capacity(self, temperature: np.ndarray) -> np.ndarray:
        """计算有效比热容"""
        phase_fraction = self.compute_phase_fraction(temperature)
        base_heat_capacity = (1 - phase_fraction) * self.solid_heat_capacity + phase_fraction * self.liquid_heat_capacity
        latent_heat_term = self.latent_heat * self._compute_phase_fraction_derivative(temperature)
        return base_heat_capacity + latent_heat_term
    
    def _compute_phase_fraction_derivative(self, temperature: np.ndarray) -> np.ndarray:
        """计算相分数对温度的导数"""
        derivative = np.zeros_like(temperature)
        
        for i, T in enumerate(temperature):
            if (self.melting_temperature - self.mushy_zone_width / 2 <= T <= 
                self.melting_temperature + self.mushy_zone_width / 2):
                derivative[i] = 1.0 / self.mushy_zone_width
            else:
                derivative[i] = 0.0
        
        return derivative


class ThermodynamicCoupling:
    """热力学耦合求解器"""
    
    def __init__(self, mesh, heat_transfer_solver: HeatTransferSolver,
                 phase_change_solver: Optional[PhaseChangeSolver] = None,
                 coupling_parameters: Dict = None):
        self.mesh = mesh
        self.heat_transfer_solver = heat_transfer_solver
        self.phase_change_solver = phase_change_solver
        self.coupling_parameters = coupling_parameters or {}
        
        # 求解器状态
        self.current_state = None
        self.previous_state = None
        
        # MPI相关
        self.comm = MPI.COMM_WORLD if HAS_MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0
    
    def solve_thermodynamic_system(self, initial_state: ThermodynamicState,
                                 boundary_conditions: Dict, time_steps: int, dt: float,
                                 heat_source: Optional[Callable] = None,
                                 velocity_field: Optional[Callable] = None) -> List[ThermodynamicState]:
        """求解热力学系统"""
        solution_history = []
        current_state = initial_state
        
        for step in range(time_steps):
            print(f"Thermodynamics Step {step + 1}/{time_steps}")
            
            new_state = self._solve_thermodynamic_step(
                current_state, boundary_conditions, dt, heat_source, velocity_field
            )
            
            self.previous_state = current_state
            current_state = new_state
            self.current_state = current_state
            solution_history.append(current_state)
            
            if (step + 1) % 10 == 0:
                print(f"  Completed {step + 1}/{time_steps} steps")
        
        return solution_history
    
    def _solve_thermodynamic_step(self, current_state: ThermodynamicState,
                                boundary_conditions: Dict, dt: float,
                                heat_source: Optional[Callable],
                                velocity_field: Optional[Callable]) -> ThermodynamicState:
        """求解单个热力学时间步"""
        # 获取速度场
        if velocity_field is not None:
            velocity = velocity_field(current_state.temperature, current_state.time)
        else:
            velocity = current_state.velocity
        
        # 求解热传递
        new_temperature = self.heat_transfer_solver.solve_heat_transfer(
            current_state.temperature, velocity, boundary_conditions, dt, heat_source
        )
        
        # 计算相分数
        if self.phase_change_solver is not None:
            new_phase_fraction = self.phase_change_solver.compute_phase_fraction(new_temperature)
            new_effective_heat_capacity = self.phase_change_solver.compute_effective_heat_capacity(new_temperature)
        else:
            new_phase_fraction = np.zeros_like(new_temperature)
            new_effective_heat_capacity = np.full_like(new_temperature, self.heat_transfer_solver.heat_capacity)
        
        # 更新其他场
        new_density = self._update_density(current_state.density, new_phase_fraction)
        new_pressure = self._update_pressure(current_state.pressure, new_temperature, new_density)
        new_enthalpy = self._compute_enthalpy(new_temperature, new_phase_fraction)
        new_entropy = self._compute_entropy(new_temperature, new_phase_fraction)
        new_heat_flux = self._compute_heat_flux(new_temperature, velocity)
        
        # 创建新状态
        new_state = ThermodynamicState(
            temperature=new_temperature,
            pressure=new_pressure,
            density=new_density,
            velocity=velocity,
            phase_fraction=new_phase_fraction,
            enthalpy=new_enthalpy,
            entropy=new_entropy,
            heat_flux=new_heat_flux,
            time=current_state.time + dt
        )
        
        return new_state
    
    def _update_density(self, current_density: np.ndarray, phase_fraction: np.ndarray) -> np.ndarray:
        """更新密度"""
        solid_density = 2700.0
        liquid_density = 1000.0
        new_density = (1 - phase_fraction) * solid_density + phase_fraction * liquid_density
        return new_density
    
    def _update_pressure(self, current_pressure: np.ndarray, temperature: np.ndarray, density: np.ndarray) -> np.ndarray:
        """更新压力"""
        gas_constant = 8.314
        molar_mass = 0.018
        new_pressure = density * gas_constant * temperature / molar_mass
        return new_pressure
    
    def _compute_enthalpy(self, temperature: np.ndarray, phase_fraction: np.ndarray) -> np.ndarray:
        """计算焓"""
        reference_temperature = 273.15
        sensible_heat = self.heat_transfer_solver.heat_capacity * (temperature - reference_temperature)
        latent_heat = phase_fraction * self.phase_change_solver.latent_heat if self.phase_change_solver else 0.0
        enthalpy = sensible_heat + latent_heat
        return enthalpy
    
    def _compute_entropy(self, temperature: np.ndarray, phase_fraction: np.ndarray) -> np.ndarray:
        """计算熵"""
        reference_temperature = 273.15
        entropy = self.heat_transfer_solver.heat_capacity * np.log(temperature / reference_temperature)
        return entropy
    
    def _compute_heat_flux(self, temperature: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """计算热流密度"""
        conduction_flux = -self.heat_transfer_solver.thermal_conductivity * self._compute_temperature_gradient(temperature)
        convection_flux = self.heat_transfer_solver.heat_capacity * self.heat_transfer_solver.density * velocity * temperature
        total_flux = conduction_flux + convection_flux
        return total_flux
    
    def _compute_temperature_gradient(self, temperature: np.ndarray) -> np.ndarray:
        """计算温度梯度"""
        n_points = len(temperature)
        gradient = np.zeros_like(temperature)
        
        for i in range(1, n_points - 1):
            gradient[i] = (temperature[i+1] - temperature[i-1]) / 2.0
        
        return gradient
    
    def get_thermodynamic_energy(self) -> float:
        """计算热力学能量"""
        if self.current_state is None:
            return 0.0
        
        internal_energy = np.sum(self.current_state.enthalpy * self.current_state.density)
        kinetic_energy = 0.5 * np.sum(self.current_state.density * self.current_state.velocity**2)
        return internal_energy + kinetic_energy
    
    def visualize_thermodynamic_results(self, solution_history: List[ThermodynamicState]):
        """可视化热力学结果"""
        try:
            import matplotlib.pyplot as plt
            
            n_steps = len(solution_history)
            times = [state.time for state in solution_history]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('热力学耦合仿真结果', fontsize=16, fontweight='bold')
            
            # 温度演化
            temperature_mean = [np.mean(state.temperature) for state in solution_history]
            axes[0, 0].plot(times, temperature_mean, 'r-', linewidth=2)
            axes[0, 0].set_xlabel('时间 (s)')
            axes[0, 0].set_ylabel('平均温度 (K)')
            axes[0, 0].set_title('温度演化')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 压力演化
            pressure_mean = [np.mean(state.pressure) for state in solution_history]
            axes[0, 1].plot(times, pressure_mean, 'b-', linewidth=2)
            axes[0, 1].set_xlabel('时间 (s)')
            axes[0, 1].set_ylabel('平均压力 (Pa)')
            axes[0, 1].set_title('压力演化')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 相分数演化
            phase_fraction_mean = [np.mean(state.phase_fraction) for state in solution_history]
            axes[0, 2].plot(times, phase_fraction_mean, 'g-', linewidth=2)
            axes[0, 2].set_xlabel('时间 (s)')
            axes[0, 2].set_ylabel('平均相分数')
            axes[0, 2].set_title('相分数演化')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 焓演化
            enthalpy_mean = [np.mean(state.enthalpy) for state in solution_history]
            axes[1, 0].plot(times, enthalpy_mean, 'm-', linewidth=2)
            axes[1, 0].set_xlabel('时间 (s)')
            axes[1, 0].set_ylabel('平均焓 (J/kg)')
            axes[1, 0].set_title('焓演化')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 热流密度演化
            heat_flux_mean = [np.mean(np.abs(state.heat_flux)) for state in solution_history]
            axes[1, 1].plot(times, heat_flux_mean, 'c-', linewidth=2)
            axes[1, 1].set_xlabel('时间 (s)')
            axes[1, 1].set_ylabel('平均热流密度 (W/m²)')
            axes[1, 1].set_title('热流密度演化')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 密度演化
            density_mean = [np.mean(state.density) for state in solution_history]
            axes[1, 2].plot(times, density_mean, 'orange', linewidth=2)
            axes[1, 2].set_xlabel('时间 (s)')
            axes[1, 2].set_ylabel('平均密度 (kg/m³)')
            axes[1, 2].set_title('密度演化')
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('thermodynamic_coupling_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ 热力学耦合结果可视化完成")
            
        except ImportError:
            print("❌ 需要matplotlib进行可视化")
        except Exception as e:
            print(f"❌ 可视化失败: {e}")


# 便捷函数
def create_thermodynamic_coupling(mesh, **kwargs) -> ThermodynamicCoupling:
    """创建热力学耦合求解器"""
    heat_transfer_solver = HeatTransferSolver(mesh, **kwargs)
    
    phase_change_solver = None
    if kwargs.get('enable_phase_change', False):
        phase_change_solver = PhaseChangeSolver(mesh, **kwargs)
    
    coupling = ThermodynamicCoupling(
        mesh=mesh,
        heat_transfer_solver=heat_transfer_solver,
        phase_change_solver=phase_change_solver,
        coupling_parameters=kwargs
    )
    
    return coupling


def demo_thermodynamic_coupling():
    """演示热力学耦合功能"""
    print("🔥 热力学耦合演示")
    print("=" * 50)
    
    # 创建简化的网格
    class SimpleMesh:
        def __init__(self, n_points):
            self.n_points = n_points
            self.coordinates = np.linspace(0, 1, n_points).reshape(-1, 1)
    
    mesh = SimpleMesh(100)
    
    # 创建热力学耦合求解器
    coupling = create_thermodynamic_coupling(
        mesh,
        thermal_conductivity=3.0,
        heat_capacity=1000.0,
        density=2700.0,
        enable_phase_change=True,
        melting_temperature=273.15,
        latent_heat=334e3
    )
    
    # 创建初始状态
    initial_state = ThermodynamicState(
        temperature=np.ones(mesh.n_points) * 293.15,
        pressure=np.ones(mesh.n_points) * 1e5,
        density=np.ones(mesh.n_points) * 2700.0,
        velocity=np.zeros(mesh.n_points),
        phase_fraction=np.zeros(mesh.n_points),
        enthalpy=np.zeros(mesh.n_points),
        entropy=np.zeros(mesh.n_points),
        heat_flux=np.zeros(mesh.n_points)
    )
    
    # 设置边界条件
    boundary_conditions = {
        'temperature': {0: 373.15, -1: 273.15},
        'heat_flux': {}
    }
    
    # 定义热源函数
    def heat_source(temperature, node_idx):
        return 1000.0 if node_idx < mesh.n_points // 2 else 0.0
    
    # 求解热力学系统
    print("🔧 开始求解热力学耦合系统...")
    solution_history = coupling.solve_thermodynamic_system(
        initial_state=initial_state,
        boundary_conditions=boundary_conditions,
        time_steps=50,
        dt=0.1,
        heat_source=heat_source
    )
    
    print(f"✅ 求解完成，共 {len(solution_history)} 个时间步")
    
    # 计算最终状态
    final_state = solution_history[-1]
    print(f"   最终平均温度: {np.mean(final_state.temperature):.2f} K")
    print(f"   最终平均压力: {np.mean(final_state.pressure):.0f} Pa")
    print(f"   最终平均相分数: {np.mean(final_state.phase_fraction):.3f}")
    print(f"   最终平均焓: {np.mean(final_state.enthalpy):.0f} J/kg")
    
    # 可视化结果
    coupling.visualize_thermodynamic_results(solution_history)
    
    return solution_history


if __name__ == "__main__":
    demo_thermodynamic_coupling() 