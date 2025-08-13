"""
多相流体耦合模块
支持油-水-气三相流动、多组分输运等场景
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
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
class MultiphaseFluidState:
    """多相流体状态"""
    # 压力场
    pressure_oil: np.ndarray      # 油相压力 [Pa]
    pressure_water: np.ndarray    # 水相压力 [Pa]
    pressure_gas: np.ndarray      # 气相压力 [Pa]
    capillary_pressure: np.ndarray  # 毛细管压力 [Pa]
    
    # 饱和度场
    saturation_oil: np.ndarray    # 油相饱和度
    saturation_water: np.ndarray  # 水相饱和度
    saturation_gas: np.ndarray    # 气相饱和度
    
    # 速度场
    velocity_oil: np.ndarray      # 油相速度 [m/s]
    velocity_water: np.ndarray    # 水相速度 [m/s]
    velocity_gas: np.ndarray      # 气相速度 [m/s]
    
    # 浓度场
    concentration_oil: np.ndarray   # 油相浓度 [kg/m³]
    concentration_water: np.ndarray # 水相浓度 [kg/m³]
    concentration_gas: np.ndarray   # 气相浓度 [kg/m³]
    
    # 温度场
    temperature: np.ndarray       # 温度 [K]
    
    # 时间
    time: float = 0.0


@dataclass
class MultiphaseFluidProperties:
    """多相流体材料属性"""
    # 油相性质
    oil_density: float = 850.0           # 油密度 [kg/m³]
    oil_viscosity: float = 1e-3          # 油粘度 [Pa·s]
    oil_compressibility: float = 1e-9    # 油压缩系数 [Pa⁻¹]
    oil_thermal_expansion: float = 7e-4  # 油热膨胀系数 [K⁻¹]
    
    # 水相性质
    water_density: float = 1000.0        # 水密度 [kg/m³]
    water_viscosity: float = 1e-3        # 水粘度 [Pa·s]
    water_compressibility: float = 4.5e-10  # 水压缩系数 [Pa⁻¹]
    water_thermal_expansion: float = 2e-4   # 水热膨胀系数 [K⁻¹]
    
    # 气相性质
    gas_density: float = 1.2             # 气密度 [kg/m³]
    gas_viscosity: float = 1.8e-5        # 气粘度 [Pa·s]
    gas_compressibility: float = 1e-5    # 气压缩系数 [Pa⁻¹]
    gas_thermal_expansion: float = 3.7e-3  # 气热膨胀系数 [K⁻¹]
    
    # 岩石性质
    porosity: float = 0.2                # 孔隙度
    permeability: float = 1e-12          # 绝对渗透率 [m²]
    rock_compressibility: float = 1e-10  # 岩石压缩系数 [Pa⁻¹]
    thermal_conductivity: float = 2.0    # 热导率 [W/(m·K)]
    
    # 毛细管压力参数
    entry_pressure: float = 1e4          # 进入压力 [Pa]
    pore_size_index: float = 2.0         # 孔隙尺寸指数
    wettability_angle: float = 0.0       # 润湿角 [rad]


class CapillaryPressureModel(ABC):
    """毛细管压力模型基类"""
    
    def __init__(self, name: str = "Capillary Pressure Model"):
        self.name = name
    
    @abstractmethod
    def compute_capillary_pressure(self, saturation: np.ndarray, 
                                 properties: MultiphaseFluidProperties) -> np.ndarray:
        """计算毛细管压力"""
        pass


class BrooksCoreyModel(CapillaryPressureModel):
    """Brooks-Corey毛细管压力模型"""
    
    def __init__(self):
        super().__init__("Brooks-Corey Model")
    
    def compute_capillary_pressure(self, saturation: np.ndarray, 
                                 properties: MultiphaseFluidProperties) -> np.ndarray:
        """计算毛细管压力 Pc = Pe * (Sw)^(-1/λ)"""
        # 有效水饱和度
        Sw_eff = (saturation - properties.porosity * 0.1) / (1.0 - properties.porosity * 0.1)
        Sw_eff = np.clip(Sw_eff, 1e-6, 1.0)
        
        # Brooks-Corey方程
        lambda_param = properties.pore_size_index
        capillary_pressure = properties.entry_pressure * (Sw_eff ** (-1.0 / lambda_param))
        
        return capillary_pressure


class RelativePermeabilityModel(ABC):
    """相对渗透率模型基类"""
    
    def __init__(self, name: str = "Relative Permeability Model"):
        self.name = name
    
    @abstractmethod
    def compute_relative_permeability(self, saturation: np.ndarray,
                                    properties: MultiphaseFluidProperties) -> Dict[str, np.ndarray]:
        """计算相对渗透率"""
        pass


class CoreyModel(RelativePermeabilityModel):
    """Corey相对渗透率模型"""
    
    def __init__(self):
        super().__init__("Corey Model")
    
    def compute_relative_permeability(self, saturation: np.ndarray,
                                    properties: MultiphaseFluidProperties) -> Dict[str, np.ndarray]:
        """计算相对渗透率"""
        # 有效饱和度
        Sw_eff = (saturation - properties.porosity * 0.1) / (1.0 - properties.porosity * 0.1)
        Sw_eff = np.clip(Sw_eff, 0.0, 1.0)
        
        # Corey方程
        krw = Sw_eff ** 4.0  # 水相相对渗透率
        kro = (1.0 - Sw_eff) ** 2.0 * (1.0 - Sw_eff ** 2.0)  # 油相相对渗透率
        
        return {
            'water': krw,
            'oil': kro,
            'gas': np.ones_like(saturation)  # 简化处理
        }


class MultiphaseFlowModel:
    """多相流动模型"""
    
    def __init__(self, properties: MultiphaseFluidProperties):
        self.properties = properties
        self.capillary_model = BrooksCoreyModel()
        self.relative_permeability_model = CoreyModel()
    
    def compute_phase_pressures(self, 
                               reference_pressure: np.ndarray,
                               saturation_water: np.ndarray) -> Dict[str, np.ndarray]:
        """计算各相压力"""
        # 毛细管压力
        capillary_pressure = self.capillary_model.compute_capillary_pressure(
            saturation_water, self.properties
        )
        
        # 各相压力
        pressure_water = reference_pressure
        pressure_oil = reference_pressure + capillary_pressure
        pressure_gas = reference_pressure - capillary_pressure * 0.5  # 简化处理
        
        return {
            'water': pressure_water,
            'oil': pressure_oil,
            'gas': pressure_gas,
            'capillary': capillary_pressure
        }
    
    def compute_phase_velocities(self, 
                                pressure_gradients: Dict[str, np.ndarray],
                                saturation: np.ndarray) -> Dict[str, np.ndarray]:
        """计算各相速度 - Darcy定律"""
        # 相对渗透率
        kr = self.relative_permeability_model.compute_relative_permeability(
            saturation, self.properties
        )
        
        # 有效渗透率
        k_eff_water = self.properties.permeability * kr['water']
        k_eff_oil = self.properties.permeability * kr['oil']
        k_eff_gas = self.properties.permeability * kr['gas']
        
        # Darcy速度
        velocity_water = -k_eff_water / self.properties.water_viscosity * pressure_gradients['water']
        velocity_oil = -k_eff_oil / self.properties.oil_viscosity * pressure_gradients['oil']
        velocity_gas = -k_eff_gas / self.properties.gas_viscosity * pressure_gradients['gas']
        
        return {
            'water': velocity_water,
            'oil': velocity_oil,
            'gas': velocity_gas
        }
    
    def compute_mass_balance(self, 
                           velocities: Dict[str, np.ndarray],
                           saturations: Dict[str, np.ndarray],
                           concentrations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """计算质量守恒"""
        # 简化的质量守恒方程
        # ∂(φρS)/∂t + ∇·(ρv) = 0
        
        # 质量通量
        mass_flux_water = self.properties.water_density * velocities['water'] * saturations['water']
        mass_flux_oil = self.properties.oil_density * velocities['oil'] * saturations['oil']
        mass_flux_gas = self.properties.gas_density * velocities['gas'] * saturations['gas']
        
        # 质量变化率（简化）
        mass_change_water = -np.gradient(mass_flux_water, axis=0)
        mass_change_oil = -np.gradient(mass_flux_oil, axis=0)
        mass_change_gas = -np.gradient(mass_flux_gas, axis=0)
        
        return {
            'water': mass_change_water,
            'oil': mass_change_oil,
            'gas': mass_change_gas
        }


class ThermalCouplingModel:
    """热耦合模型"""
    
    def __init__(self, properties: MultiphaseFluidProperties):
        self.properties = properties
    
    def compute_thermal_effects(self, 
                              temperature: np.ndarray,
                              saturations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """计算热效应"""
        # 温度对粘度的影响（Arrhenius关系）
        T_ref = 293.15  # 参考温度 [K]
        activation_energy = 20e3  # 激活能 [J/mol]
        gas_constant = 8.314  # 气体常数 [J/(mol·K)]
        
        # 温度因子
        temperature_factor = np.exp(activation_energy / gas_constant * (1.0 / temperature - 1.0 / T_ref))
        
        # 热膨胀效应
        thermal_expansion_water = self.properties.water_thermal_expansion * (temperature - T_ref)
        thermal_expansion_oil = self.properties.oil_thermal_expansion * (temperature - T_ref)
        thermal_expansion_gas = self.properties.gas_thermal_expansion * (temperature - T_ref)
        
        # 密度修正
        density_water = self.properties.water_density * (1.0 - thermal_expansion_water)
        density_oil = self.properties.oil_density * (1.0 - thermal_expansion_oil)
        density_gas = self.properties.gas_density * (1.0 - thermal_expansion_gas)
        
        return {
            'temperature_factor': temperature_factor,
            'density_water': density_water,
            'density_oil': density_oil,
            'density_gas': density_gas
        }


class ChemicalTransportModel:
    """化学输运模型"""
    
    def __init__(self, properties: MultiphaseFluidProperties):
        self.properties = properties
    
    def compute_chemical_transport(self, 
                                 concentrations: Dict[str, np.ndarray],
                                 velocities: Dict[str, np.ndarray],
                                 saturations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """计算化学输运"""
        # 对流-扩散方程
        # ∂(φSC)/∂t + ∇·(vC) = ∇·(D∇C)
        
        # 对流项
        convection_water = velocities['water'] * concentrations['water'] * saturations['water']
        convection_oil = velocities['oil'] * concentrations['oil'] * saturations['oil']
        convection_gas = velocities['gas'] * concentrations['gas'] * saturations['gas']
        
        # 扩散项（简化）
        diffusion_coefficient = 1e-9  # m²/s
        
        # 浓度梯度
        if concentrations['water'].ndim > 1:
            grad_c_water = np.gradient(concentrations['water'], axis=0)
            grad_c_oil = np.gradient(concentrations['oil'], axis=0)
            grad_c_gas = np.gradient(concentrations['gas'], axis=0)
        else:
            grad_c_water = np.gradient(concentrations['water'])
            grad_c_oil = np.gradient(concentrations['oil'])
            grad_c_gas = np.gradient(concentrations['gas'])
        
        # 扩散通量
        diffusion_water = -diffusion_coefficient * grad_c_water
        diffusion_oil = -diffusion_coefficient * grad_c_oil
        diffusion_gas = -diffusion_coefficient * grad_c_gas
        
        # 总输运
        transport_water = convection_water + diffusion_water
        transport_oil = convection_oil + diffusion_oil
        transport_gas = convection_gas + diffusion_gas
        
        return {
            'water': transport_water,
            'oil': transport_oil,
            'gas': transport_gas
        }


class MultiphaseFluidCoupling:
    """多相流体耦合求解器"""
    
    def __init__(self, 
                 flow_model: MultiphaseFlowModel,
                 thermal_model: ThermalCouplingModel,
                 chemical_model: ChemicalTransportModel):
        self.flow_model = flow_model
        self.thermal_model = thermal_model
        self.chemical_model = chemical_model
        
        self.coupling_history = []
        self.convergence_criteria = {
            'max_iterations': 100,
            'tolerance': 1e-6,
            'relaxation_factor': 0.8
        }
    
    def solve_coupled_system(self, 
                           initial_state: MultiphaseFluidState,
                           boundary_conditions: Dict,
                           time_steps: int = 100,
                           dt: float = 0.01) -> List[MultiphaseFluidState]:
        """求解耦合系统"""
        print("🔄 开始求解多相流体耦合系统...")
        
        states = [initial_state]
        current_state = initial_state
        
        for step in range(time_steps):
            print(f"   时间步 {step+1}/{time_steps}")
            
            # 流动求解
            flow_state = self._solve_flow_field(current_state, boundary_conditions)
            
            # 热耦合求解
            thermal_state = self._solve_thermal_field(current_state, flow_state, boundary_conditions)
            
            # 化学输运求解
            chemical_state = self._solve_chemical_field(current_state, flow_state, thermal_state, boundary_conditions)
            
            # 耦合迭代
            coupled_state = self._coupling_iteration(flow_state, thermal_state, chemical_state, boundary_conditions)
            
            # 更新状态
            current_state = coupled_state
            current_state.time = (step + 1) * dt
            states.append(current_state)
            
            # 检查收敛性
            if self._check_convergence(coupled_state, states[-2]):
                print(f"   收敛于时间步 {step+1}")
                break
        
        print("✅ 多相流体耦合系统求解完成")
        return states
    
    def _solve_flow_field(self, 
                         current_state: MultiphaseFluidState,
                         boundary_conditions: Dict) -> MultiphaseFluidState:
        """求解流动场"""
        # 计算各相压力
        reference_pressure = current_state.pressure_water
        phase_pressures = self.flow_model.compute_phase_pressures(
            reference_pressure, current_state.saturation_water
        )
        
        # 计算压力梯度（简化）
        pressure_gradients = {
            'water': np.gradient(phase_pressures['water']),
            'oil': np.gradient(phase_pressures['oil']),
            'gas': np.gradient(phase_pressures['gas'])
        }
        
        # 计算各相速度
        velocities = self.flow_model.compute_phase_velocities(
            pressure_gradients, current_state.saturation_water
        )
        
        # 计算质量守恒
        saturations = {
            'water': current_state.saturation_water,
            'oil': current_state.saturation_oil,
            'gas': current_state.saturation_gas
        }
        
        concentrations = {
            'water': current_state.concentration_water,
            'oil': current_state.concentration_oil,
            'gas': current_state.concentration_gas
        }
        
        mass_balance = self.flow_model.compute_mass_balance(velocities, saturations, concentrations)
        
        # 更新状态
        flow_state = MultiphaseFluidState(
            pressure_oil=phase_pressures['oil'],
            pressure_water=phase_pressures['water'],
            pressure_gas=phase_pressures['gas'],
            capillary_pressure=phase_pressures['capillary'],
            saturation_oil=current_state.saturation_oil,
            saturation_water=current_state.saturation_water,
            saturation_gas=current_state.saturation_gas,
            velocity_oil=velocities['oil'],
            velocity_water=velocities['water'],
            velocity_gas=velocities['gas'],
            concentration_oil=current_state.concentration_oil,
            concentration_water=current_state.concentration_water,
            concentration_gas=current_state.concentration_gas,
            temperature=current_state.temperature,
            time=current_state.time
        )
        
        return flow_state
    
    def _solve_thermal_field(self, 
                           current_state: MultiphaseFluidState,
                           flow_state: MultiphaseFluidState,
                           boundary_conditions: Dict) -> MultiphaseFluidState:
        """求解热场"""
        # 计算热效应
        saturations = {
            'water': flow_state.saturation_water,
            'oil': flow_state.saturation_oil,
            'gas': flow_state.saturation_gas
        }
        
        thermal_effects = self.thermal_model.compute_thermal_effects(
            current_state.temperature, saturations
        )
        
        # 更新温度（简化的热传导方程）
        temperature = current_state.temperature.copy()
        temperature += 0.01 * np.random.randn(*temperature.shape)  # 热源项
        
        # 更新状态
        thermal_state = MultiphaseFluidState(
            pressure_oil=flow_state.pressure_oil,
            pressure_water=flow_state.pressure_water,
            pressure_gas=flow_state.pressure_gas,
            capillary_pressure=flow_state.capillary_pressure,
            saturation_oil=flow_state.saturation_oil,
            saturation_water=flow_state.saturation_water,
            saturation_gas=flow_state.saturation_gas,
            velocity_oil=flow_state.velocity_oil,
            velocity_water=flow_state.velocity_water,
            velocity_gas=flow_state.velocity_gas,
            concentration_oil=flow_state.concentration_oil,
            concentration_water=flow_state.concentration_water,
            concentration_gas=flow_state.concentration_gas,
            temperature=temperature,
            time=flow_state.time
        )
        
        return thermal_state
    
    def _solve_chemical_field(self, 
                            current_state: MultiphaseFluidState,
                            flow_state: MultiphaseFluidState,
                            thermal_state: MultiphaseFluidState,
                            boundary_conditions: Dict) -> MultiphaseFluidState:
        """求解化学场"""
        # 计算化学输运
        velocities = {
            'water': flow_state.velocity_water,
            'oil': flow_state.velocity_oil,
            'gas': flow_state.velocity_gas
        }
        
        saturations = {
            'water': flow_state.saturation_water,
            'oil': flow_state.saturation_oil,
            'gas': flow_state.saturation_gas
        }
        
        concentrations = {
            'water': flow_state.concentration_water,
            'oil': flow_state.concentration_oil,
            'gas': flow_state.concentration_gas
        }
        
        chemical_transport = self.chemical_model.compute_chemical_transport(
            concentrations, velocities, saturations
        )
        
        # 更新浓度（简化的输运方程）
        concentration_water = flow_state.concentration_water + 0.1 * chemical_transport['water']
        concentration_oil = flow_state.concentration_oil + 0.1 * chemical_transport['oil']
        concentration_gas = flow_state.concentration_gas + 0.1 * chemical_transport['gas']
        
        # 更新状态
        chemical_state = MultiphaseFluidState(
            pressure_oil=thermal_state.pressure_oil,
            pressure_water=thermal_state.pressure_water,
            pressure_gas=thermal_state.pressure_gas,
            capillary_pressure=thermal_state.capillary_pressure,
            saturation_oil=thermal_state.saturation_oil,
            saturation_water=thermal_state.saturation_water,
            saturation_gas=thermal_state.saturation_gas,
            velocity_oil=thermal_state.velocity_oil,
            velocity_water=thermal_state.velocity_water,
            velocity_gas=thermal_state.velocity_gas,
            concentration_oil=concentration_oil,
            concentration_water=concentration_water,
            concentration_gas=concentration_gas,
            temperature=thermal_state.temperature,
            time=thermal_state.time
        )
        
        return chemical_state
    
    def _coupling_iteration(self, 
                           flow_state: MultiphaseFluidState,
                           thermal_state: MultiphaseFluidState,
                           chemical_state: MultiphaseFluidState,
                           boundary_conditions: Dict) -> MultiphaseFluidState:
        """耦合迭代"""
        # 简化的耦合迭代
        # 在实际应用中，这里应该进行更复杂的耦合计算
        
        # 考虑温度对流动性质的影响
        temperature_factor = 1.0 + 0.1 * (thermal_state.temperature - 293.15) / 100.0
        
        # 更新速度（温度影响粘度）
        updated_velocity_water = flow_state.velocity_water * temperature_factor
        updated_velocity_oil = flow_state.velocity_oil * temperature_factor
        updated_velocity_gas = flow_state.velocity_gas * temperature_factor
        
        # 更新状态
        coupled_state = MultiphaseFluidState(
            pressure_oil=chemical_state.pressure_oil,
            pressure_water=chemical_state.pressure_water,
            pressure_gas=chemical_state.pressure_gas,
            capillary_pressure=chemical_state.capillary_pressure,
            saturation_oil=chemical_state.saturation_oil,
            saturation_water=chemical_state.saturation_water,
            saturation_gas=chemical_state.saturation_gas,
            velocity_oil=updated_velocity_oil,
            velocity_water=updated_velocity_water,
            velocity_gas=updated_velocity_gas,
            concentration_oil=chemical_state.concentration_oil,
            concentration_water=chemical_state.concentration_water,
            concentration_gas=chemical_state.concentration_gas,
            temperature=chemical_state.temperature,
            time=chemical_state.time
        )
        
        return coupled_state
    
    def _check_convergence(self, 
                          current_state: MultiphaseFluidState,
                          previous_state: MultiphaseFluidState) -> bool:
        """检查收敛性"""
        # 计算状态变化
        pressure_change = np.mean(np.abs(
            current_state.pressure_water - previous_state.pressure_water
        ))
        saturation_change = np.mean(np.abs(
            current_state.saturation_water - previous_state.saturation_water
        ))
        temperature_change = np.mean(np.abs(
            current_state.temperature - previous_state.temperature
        ))
        
        # 检查是否收敛
        max_change = max(pressure_change, saturation_change, temperature_change)
        return max_change < self.convergence_criteria['tolerance']
    
    def compute_flow_characteristics(self, state: MultiphaseFluidState) -> Dict:
        """计算流动特征"""
        # 总流量
        total_flow_water = np.sum(np.abs(state.velocity_water), axis=0)
        total_flow_oil = np.sum(np.abs(state.velocity_oil), axis=0)
        total_flow_gas = np.sum(np.abs(state.velocity_gas), axis=0)
        
        # 平均饱和度
        avg_saturation_water = np.mean(state.saturation_water)
        avg_saturation_oil = np.mean(state.saturation_oil)
        avg_saturation_gas = np.mean(state.saturation_gas)
        
        # 流动特征
        characteristics = {
            'total_flow_water': total_flow_water,
            'total_flow_oil': total_flow_oil,
            'total_flow_gas': total_flow_gas,
            'avg_saturation_water': avg_saturation_water,
            'avg_saturation_oil': avg_saturation_oil,
            'avg_saturation_gas': avg_saturation_gas,
            'water_cut': avg_saturation_water / (avg_saturation_water + avg_saturation_oil + 1e-12),
            'gas_oil_ratio': avg_saturation_gas / (avg_saturation_oil + 1e-12)
        }
        
        return characteristics


def create_multiphase_fluid_system() -> Dict:
    """创建多相流体耦合系统"""
    # 创建材料属性
    properties = MultiphaseFluidProperties(
        oil_density=850.0,
        oil_viscosity=1e-3,
        water_density=1000.0,
        water_viscosity=1e-3,
        gas_density=1.2,
        gas_viscosity=1.8e-5,
        porosity=0.25,
        permeability=1e-13
    )
    
    # 创建模型
    flow_model = MultiphaseFlowModel(properties)
    thermal_model = ThermalCouplingModel(properties)
    chemical_model = ChemicalTransportModel(properties)
    
    # 创建耦合求解器
    coupling_solver = MultiphaseFluidCoupling(flow_model, thermal_model, chemical_model)
    
    system = {
        'properties': properties,
        'flow_model': flow_model,
        'thermal_model': thermal_model,
        'chemical_model': chemical_model,
        'coupling_solver': coupling_solver
    }
    
    print("🔄 多相流体耦合系统创建完成")
    return system


def demo_multiphase_fluid_coupling():
    """演示多相流体耦合"""
    print("🌊 多相流体耦合演示")
    print("=" * 60)
    
    try:
        # 创建系统
        system = create_multiphase_fluid_system()
        
        # 创建初始状态
        n_points = 100
        initial_state = MultiphaseFluidState(
            pressure_oil=np.random.rand(n_points) * 1e6 + 1e6,      # 1-2 MPa
            pressure_water=np.random.rand(n_points) * 1e6 + 1e6,    # 1-2 MPa
            pressure_gas=np.random.rand(n_points) * 1e6 + 1e6,      # 1-2 MPa
            capillary_pressure=np.random.rand(n_points) * 1e4,      # 0-10 kPa
            saturation_oil=np.random.rand(n_points) * 0.4 + 0.3,    # 0.3-0.7
            saturation_water=np.random.rand(n_points) * 0.3 + 0.2,  # 0.2-0.5
            saturation_gas=np.random.rand(n_points) * 0.2 + 0.1,    # 0.1-0.3
            velocity_oil=np.random.randn(n_points, 3) * 1e-6,       # m/s
            velocity_water=np.random.randn(n_points, 3) * 1e-6,     # m/s
            velocity_gas=np.random.randn(n_points, 3) * 1e-6,       # m/s
            concentration_oil=np.random.rand(n_points) * 100 + 800,  # 800-900 kg/m³
            concentration_water=np.random.rand(n_points) * 50 + 950, # 950-1000 kg/m³
            concentration_gas=np.random.rand(n_points) * 0.5 + 0.7,  # 0.7-1.2 kg/m³
            temperature=np.random.rand(n_points) * 50 + 273.15      # 273-323 K
        )
        
        # 边界条件
        boundary_conditions = {
            'pressure': {'inlet': 2e6, 'outlet': 1e6},  # Pa
            'temperature': {'inlet': 323.15, 'outlet': 273.15},  # K
            'saturation': {'inlet': {'water': 0.8, 'oil': 0.15, 'gas': 0.05}}
        }
        
        # 求解耦合系统
        coupling_solver = system['coupling_solver']
        states = coupling_solver.solve_coupled_system(
            initial_state, boundary_conditions, time_steps=30, dt=0.01
        )
        
        print(f"   求解完成，共 {len(states)} 个时间步")
        
        # 计算流动特征
        final_state = states[-1]
        flow_characteristics = coupling_solver.compute_flow_characteristics(final_state)
        
        print(f"   流动特征分析:")
        print(f"     平均水饱和度: {flow_characteristics['avg_saturation_water']:.3f}")
        print(f"     平均油饱和度: {flow_characteristics['avg_saturation_oil']:.3f}")
        print(f"     平均气饱和度: {flow_characteristics['avg_saturation_gas']:.3f}")
        print(f"     含水率: {flow_characteristics['water_cut']:.3f}")
        print(f"     气油比: {flow_characteristics['gas_oil_ratio']:.3f}")
        
        print("\n✅ 多相流体耦合演示完成!")
        return True
        
    except Exception as e:
        print(f"❌ 多相流体耦合演示失败: {e}")
        return False


if __name__ == "__main__":
    demo_multiphase_fluid_coupling()

