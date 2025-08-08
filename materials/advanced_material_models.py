"""
高级材料模型系统
"""

import numpy as np
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
from scipy.sparse import csr_matrix
import warnings

# 常量
GAS_CONSTANT = 8.3144621  # J/(mol·K)
BURGERS_VECTOR = 0.5e-9   # m


@dataclass
class MaterialState:
    """材料状态"""
    temperature: np.ndarray  # 温度场
    pressure: np.ndarray     # 压力场
    strain_rate: np.ndarray  # 应变率场
    plastic_strain: np.ndarray  # 累积塑性应变
    time: float = 0.0


class Rheology(ABC):
    """流变学基类"""
    
    def __init__(self, name: str = "Undefined"):
        self.name = name
        self.material_state: Optional[MaterialState] = None
    
    @abstractmethod
    def compute_effective_viscosity(self) -> np.ndarray:
        """计算有效粘度"""
        pass
    
    def set_material_state(self, state: MaterialState):
        """设置材料状态"""
        self.material_state = state


class ConstantViscosity(Rheology):
    """常数粘度模型"""
    
    def __init__(self, viscosity: float, name: str = "Constant Viscosity"):
        super().__init__(name)
        self.viscosity = viscosity
    
    def compute_effective_viscosity(self) -> np.ndarray:
        if self.material_state is None:
            raise ValueError("Material state not set")
        return np.full_like(self.material_state.temperature, self.viscosity)


class PowerLawCreep(Rheology):
    """幂律蠕变粘度模型
    
    基于Underworld2的ViscousCreep实现：
    η_eff = 0.5 * A^(-1/n) * ε̇^((1-n)/n) * d^(m/n) * exp((E + PV)/(nRT))
    """
    
    def __init__(self, 
                 pre_exponential_factor: float = 1.0,
                 stress_exponent: float = 1.0,
                 activation_energy: float = 0.0,
                 activation_volume: float = 0.0,
                 grain_size: float = 1e-3,
                 grain_size_exponent: float = 0.0,
                 water_fugacity: Optional[float] = None,
                 water_fugacity_exponent: float = 0.0,
                 scaling_factor: float = 1.0,
                 name: str = "Power Law Creep"):
        super().__init__(name)
        self.pre_exponential_factor = pre_exponential_factor
        self.stress_exponent = stress_exponent
        self.activation_energy = activation_energy
        self.activation_volume = activation_volume
        self.grain_size = grain_size
        self.grain_size_exponent = grain_size_exponent
        self.water_fugacity = water_fugacity
        self.water_fugacity_exponent = water_fugacity_exponent
        self.scaling_factor = scaling_factor
    
    def compute_effective_viscosity(self) -> np.ndarray:
        if self.material_state is None:
            raise ValueError("Material state not set")
        
        state = self.material_state
        n = self.stress_exponent
        A = self.pre_exponential_factor
        Q = self.activation_energy
        Va = self.activation_volume
        p = self.grain_size_exponent
        d = self.grain_size
        r = self.water_fugacity_exponent
        fH2O = self.water_fugacity
        f = self.scaling_factor
        R = GAS_CONSTANT
        b = BURGERS_VECTOR
        
        # 计算应变率不变量
        strain_rate_invariant = np.sqrt(np.sum(state.strain_rate**2, axis=1))
        strain_rate_invariant = np.maximum(strain_rate_invariant, 1e-20)
        
        # 基础粘度
        mu_eff = f * 0.5 * A**(-1.0 / n)
        
        # 应变率依赖
        if abs(n - 1.0) > 1e-5:
            mu_eff *= strain_rate_invariant**((1.0 - n) / n)
        
        # 晶粒尺寸依赖
        if p > 0 and d > 0:
            mu_eff *= (d / b)**(p / n)
        
        # 水含量依赖
        if r > 0 and fH2O is not None:
            mu_eff *= fH2O**(-r / n)
        
        # 温度压力依赖
        if Q > 0 or Va > 0:
            exp_term = (Q + state.pressure * Va) / (R * state.temperature * n)
            mu_eff *= np.exp(exp_term)
        
        return mu_eff


class TemperatureDepthViscosity(Rheology):
    """温度深度依赖粘度模型"""
    
    def __init__(self, 
                 reference_viscosity: float,
                 temperature_factor: float,
                 depth_factor: float,
                 reference_depth: float,
                 name: str = "Temperature-Depth Viscosity"):
        super().__init__(name)
        self.reference_viscosity = reference_viscosity
        self.temperature_factor = temperature_factor
        self.depth_factor = depth_factor
        self.reference_depth = reference_depth
    
    def compute_effective_viscosity(self) -> np.ndarray:
        if self.material_state is None:
            raise ValueError("Material state not set")
        
        state = self.material_state
        # 简化的深度计算（假设z坐标）
        depth = -state.pressure / (9.81 * 2700)  # 简化的深度估计
        
        mu_eff = self.reference_viscosity * np.exp(
            self.temperature_factor * (state.temperature - 273.15) +
            self.depth_factor * (depth - self.reference_depth)
        )
        
        return mu_eff


class AdvancedPlasticity(Rheology):
    """高级塑性模型（基于Underworld2的DruckerPrager）"""
    
    def __init__(self,
                 cohesion: float,
                 friction_angle: float,
                 cohesion_after_softening: Optional[float] = None,
                 friction_after_softening: Optional[float] = None,
                 softening_start: float = 0.5,
                 softening_end: float = 1.5,
                 dimension: int = 2,
                 name: str = "Advanced Plasticity"):
        super().__init__(name)
        self.cohesion = cohesion
        self.friction_angle = np.radians(friction_angle)
        self.cohesion_after_softening = cohesion_after_softening or cohesion
        self.friction_after_softening = np.radians(friction_after_softening or friction_angle)
        self.softening_start = softening_start
        self.softening_end = softening_end
        self.dimension = dimension
    
    def _compute_weakening_factor(self, plastic_strain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算弱化因子"""
        # 线性弱化
        weakening_factor = np.ones_like(plastic_strain)
        mask = (plastic_strain >= self.softening_start) & (plastic_strain <= self.softening_end)
        
        if np.any(mask):
            factor = (plastic_strain[mask] - self.softening_start) / (self.softening_end - self.softening_start)
            weakening_factor[mask] = 1.0 - factor
        
        mask = plastic_strain > self.softening_end
        weakening_factor[mask] = 0.0
        
        return weakening_factor
    
    def _compute_yield_stress(self) -> np.ndarray:
        if self.material_state is None:
            raise ValueError("Material state not set")
        
        state = self.material_state
        
        # 计算弱化因子
        weakening_factor = self._compute_weakening_factor(state.plastic_strain)
        
        # 计算当前内聚力和摩擦角
        current_cohesion = (self.cohesion * weakening_factor + 
                           self.cohesion_after_softening * (1 - weakening_factor))
        current_friction = (self.friction_angle * weakening_factor + 
                           self.friction_after_softening * (1 - weakening_factor))
        
        if self.dimension == 2:
            # 2D Drucker-Prager
            yield_stress = (current_cohesion * np.cos(current_friction) + 
                           state.pressure * np.sin(current_friction))
        else:
            # 3D Drucker-Prager
            yield_stress = (6.0 * current_cohesion * np.cos(current_friction) + 
                           6.0 * np.sin(current_friction) * np.maximum(state.pressure, 0.0))
            yield_stress /= (np.sqrt(3.0) * (3.0 - np.sin(current_friction)))
        
        return yield_stress
    
    def compute_effective_viscosity(self) -> np.ndarray:
        if self.material_state is None:
            raise ValueError("Material state not set")
        
        state = self.material_state
        yield_stress = self._compute_yield_stress()
        
        # 计算应变率不变量
        strain_rate_invariant = np.sqrt(np.sum(state.strain_rate**2, axis=1))
        strain_rate_invariant = np.maximum(strain_rate_invariant, 1e-20)
        
        # 塑性粘度
        mu_eff = 0.5 * yield_stress / strain_rate_invariant
        
        return mu_eff


class VonMisesPlasticity(AdvancedPlasticity):
    """von Mises塑性模型"""
    
    def __init__(self, 
                 yield_stress: float,
                 yield_stress_after_softening: Optional[float] = None,
                 softening_start: float = 0.5,
                 softening_end: float = 1.5,
                 name: str = "Von Mises Plasticity"):
        super().__init__(
            cohesion=yield_stress,
            friction_angle=0.0,
            cohesion_after_softening=yield_stress_after_softening,
            friction_after_softening=0.0,
            softening_start=softening_start,
            softening_end=softening_end,
            name=name
        )


class CompositeRheology(Rheology):
    """复合流变学（组合多种变形机制）"""
    
    def __init__(self, rheologies: List[Rheology], combination_method: str = "harmonic"):
        super().__init__("Composite Rheology")
        self.rheologies = rheologies
        self.combination_method = combination_method
        
        # 设置材料状态
        for rheology in rheologies:
            rheology.material_state = self.material_state
    
    def set_material_state(self, state: MaterialState):
        super().set_material_state(state)
        for rheology in self.rheologies:
            rheology.set_material_state(state)
    
    def compute_effective_viscosity(self) -> np.ndarray:
        if not self.rheologies:
            raise ValueError("No rheologies provided")
        
        viscosities = [rheology.compute_effective_viscosity() for rheology in self.rheologies]
        
        if self.combination_method == "harmonic":
            # 调和平均
            total_inverse = np.zeros_like(viscosities[0])
            for mu in viscosities:
                total_inverse += 1.0 / mu
            return 1.0 / total_inverse
        
        elif self.combination_method == "minimum":
            # 最小值
            return np.minimum.reduce(viscosities)
        
        elif self.combination_method == "maximum":
            # 最大值
            return np.maximum.reduce(viscosities)
        
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")


class AdvancedDensity:
    """高级密度模型（温度压力依赖）"""
    
    def __init__(self,
                 reference_density: float,
                 thermal_expansivity: float = 3e-5,
                 compressibility: float = 0.0,
                 reference_temperature: float = 273.15,
                 reference_pressure: float = 0.0):
        self.reference_density = reference_density
        self.thermal_expansivity = thermal_expansivity
        self.compressibility = compressibility
        self.reference_temperature = reference_temperature
        self.reference_pressure = reference_pressure
    
    def compute_density(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算密度：ρ = ρ₀ * (1 + β*ΔP - α*ΔT)"""
        delta_t = temperature - self.reference_temperature
        delta_p = pressure - self.reference_pressure
        
        density = self.reference_density * (
            1.0 + self.compressibility * delta_p - self.thermal_expansivity * delta_t
        )
        
        return density


class Material:
    """高级材料类（基于Underworld2的Material）"""
    
    def __init__(self,
                 name: str = "Undefined",
                 density: Union[float, AdvancedDensity] = 2700.0,
                 viscosity: Optional[Rheology] = None,
                 plasticity: Optional[Rheology] = None,
                 thermal_conductivity: float = 3.0,
                 heat_capacity: float = 1000.0,
                 thermal_expansivity: float = 3e-5,
                 radiogenic_heat_production: float = 0.0,
                 min_viscosity: Optional[float] = None,
                 max_viscosity: Optional[float] = None,
                 stress_limiter: Optional[float] = None,
                 healing_rate: float = 0.0):
        
        self.name = name
        self.density = density if isinstance(density, AdvancedDensity) else AdvancedDensity(density)
        self.viscosity = viscosity
        self.plasticity = plasticity
        self.thermal_conductivity = thermal_conductivity
        self.heat_capacity = heat_capacity
        self.thermal_expansivity = thermal_expansivity
        self.radiogenic_heat_production = radiogenic_heat_production
        self.min_viscosity = min_viscosity
        self.max_viscosity = max_viscosity
        self.stress_limiter = stress_limiter
        self.healing_rate = healing_rate
        
        # 材料状态
        self.material_state: Optional[MaterialState] = None
    
    def set_material_state(self, state: MaterialState):
        """设置材料状态"""
        self.material_state = state
        if self.viscosity:
            self.viscosity.set_material_state(state)
        if self.plasticity:
            self.plasticity.set_material_state(state)
    
    def compute_effective_viscosity(self) -> np.ndarray:
        """计算有效粘度"""
        if self.viscosity is None and self.plasticity is None:
            raise ValueError("No rheology defined")
        
        viscosities = []
        
        # 粘性粘度
        if self.viscosity:
            visc_viscosity = self.viscosity.compute_effective_viscosity()
            viscosities.append(visc_viscosity)
        
        # 塑性粘度
        if self.plasticity:
            plastic_viscosity = self.plasticity.compute_effective_viscosity()
            viscosities.append(plastic_viscosity)
        
        # 组合粘度
        if len(viscosities) == 1:
            effective_viscosity = viscosities[0]
        else:
            # 使用调和平均
            total_inverse = np.zeros_like(viscosities[0])
            for mu in viscosities:
                total_inverse += 1.0 / mu
            effective_viscosity = 1.0 / total_inverse
        
        # 应用粘度限制
        if self.min_viscosity is not None:
            effective_viscosity = np.maximum(effective_viscosity, self.min_viscosity)
        if self.max_viscosity is not None:
            effective_viscosity = np.minimum(effective_viscosity, self.max_viscosity)
        
        return effective_viscosity
    
    def compute_density(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算密度"""
        return self.density.compute_density(temperature, pressure)


class MaterialRegistry:
    """材料注册系统（基于Underworld2的MaterialRegistry）"""
    
    def __init__(self, filename: Optional[str] = None):
        self.materials: Dict[str, Material] = {}
        
        if filename:
            self.load_from_file(filename)
        else:
            self._create_default_materials()
    
    def _create_default_materials(self):
        """创建默认材料"""
        
        # 地壳材料
        crust_viscosity = PowerLawCreep(
            pre_exponential_factor=1e-15,
            stress_exponent=3.0,
            activation_energy=154e3,
            activation_volume=0.0,
            grain_size=1e-3,
            name="Wet Quartz Creep"
        )
        
        crust_plasticity = AdvancedPlasticity(
            cohesion=20e6,  # 20 MPa
            friction_angle=7.0,  # 7 degrees
            cohesion_after_softening=5e6,  # 5 MPa
            friction_after_softening=3.0,  # 3 degrees
            name="Crust Plasticity"
        )
        
        crust = Material(
            name="Upper Crust",
            density=AdvancedDensity(2620.0, thermal_expansivity=3e-5),
            viscosity=crust_viscosity,
            plasticity=crust_plasticity,
            thermal_conductivity=2.5,
            heat_capacity=1000.0,
            radiogenic_heat_production=0.7e-3,  # 0.7 mW/m³
            min_viscosity=1e18,
            max_viscosity=1e25
        )
        
        # 地幔材料
        mantle_viscosity = PowerLawCreep(
            pre_exponential_factor=1e-15,
            stress_exponent=3.5,
            activation_energy=520e3,
            activation_volume=14e-6,
            grain_size=1e-2,
            name="Olivine Creep"
        )
        
        mantle = Material(
            name="Upper Mantle",
            density=AdvancedDensity(3300.0, thermal_expansivity=3e-5),
            viscosity=mantle_viscosity,
            thermal_conductivity=3.0,
            heat_capacity=1200.0,
            radiogenic_heat_production=0.02e-3,  # 0.02 mW/m³
            min_viscosity=1e19,
            max_viscosity=1e26
        )
        
        # 空气材料（用于自由表面）
        air = Material(
            name="Air",
            density=AdvancedDensity(1.0),
            viscosity=ConstantViscosity(1e19),
            thermal_conductivity=0.025,
            heat_capacity=100.0,
            min_viscosity=1e18,
            max_viscosity=1e21
        )
        
        self.materials = {
            "crust": crust,
            "mantle": mantle,
            "air": air
        }
    
    def add_material(self, name: str, material: Material):
        """添加材料"""
        self.materials[name] = material
    
    def get_material(self, name: str) -> Material:
        """获取材料"""
        if name not in self.materials:
            raise KeyError(f"Material '{name}' not found")
        return self.materials[name]
    
    def list_materials(self) -> List[str]:
        """列出所有材料"""
        return list(self.materials.keys())
    
    def load_from_file(self, filename: str):
        """从文件加载材料"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            for name, material_data in data.items():
                material = self._create_material_from_dict(name, material_data)
                self.materials[name] = material
                
        except Exception as e:
            warnings.warn(f"Failed to load materials from {filename}: {e}")
            self._create_default_materials()
    
    def _create_material_from_dict(self, name: str, data: dict) -> Material:
        """从字典创建材料"""
        # 简化的材料创建（实际应用中需要更复杂的解析）
        density = data.get('density', 2700.0)
        if isinstance(density, dict):
            density = AdvancedDensity(**density)
        else:
            density = AdvancedDensity(density)
        
        # 创建粘度模型
        viscosity = None
        if 'viscosity' in data:
            visc_data = data['viscosity']
            if visc_data.get('type') == 'power_law':
                viscosity = PowerLawCreep(**visc_data.get('parameters', {}))
            elif visc_data.get('type') == 'constant':
                viscosity = ConstantViscosity(visc_data.get('value', 1e21))
        
        # 创建塑性模型
        plasticity = None
        if 'plasticity' in data:
            plastic_data = data['plasticity']
            if plastic_data.get('type') == 'drucker_prager':
                plasticity = AdvancedPlasticity(**plastic_data.get('parameters', {}))
            elif plastic_data.get('type') == 'von_mises':
                plasticity = VonMisesPlasticity(**plastic_data.get('parameters', {}))
        
        return Material(
            name=name,
            density=density,
            viscosity=viscosity,
            plasticity=plasticity,
            thermal_conductivity=data.get('thermal_conductivity', 3.0),
            heat_capacity=data.get('heat_capacity', 1000.0),
            thermal_expansivity=data.get('thermal_expansivity', 3e-5),
            radiogenic_heat_production=data.get('radiogenic_heat_production', 0.0),
            min_viscosity=data.get('min_viscosity'),
            max_viscosity=data.get('max_viscosity')
        )


# 便捷函数
def create_crust_material() -> Material:
    """创建地壳材料"""
    registry = MaterialRegistry()
    return registry.get_material("crust")


def create_mantle_material() -> Material:
    """创建地幔材料"""
    registry = MaterialRegistry()
    return registry.get_material("mantle")


def create_air_material() -> Material:
    """创建空气材料"""
    registry = MaterialRegistry()
    return registry.get_material("air")


def create_composite_material(viscosity_rheologies: List[Rheology], 
                            plasticity_rheologies: List[Rheology] = None,
                            combination_method: str = "harmonic") -> Material:
    """创建复合材料"""
    rheologies = viscosity_rheologies.copy()
    if plasticity_rheologies:
        rheologies.extend(plasticity_rheologies)
    
    composite_rheology = CompositeRheology(rheologies, combination_method)
    
    return Material(
        name="Composite Material",
        viscosity=composite_rheology,
        density=AdvancedDensity(2800.0)
    )


def create_underworld2_style_material(name: str, 
                                    material_type: str = "crust",
                                    **kwargs) -> Material:
    """创建Underworld2风格的材料"""
    
    if material_type == "crust":
        # 湿石英地壳材料（基于Underworld2的Wet Quartz）
        viscosity = PowerLawCreep(
            pre_exponential_factor=kwargs.get('pre_exponential_factor', 1e-15),
            stress_exponent=kwargs.get('stress_exponent', 3.0),
            activation_energy=kwargs.get('activation_energy', 154e3),
            activation_volume=kwargs.get('activation_volume', 0.0),
            grain_size=kwargs.get('grain_size', 1e-3),
            name="Wet Quartz Creep"
        )
        
        plasticity = AdvancedPlasticity(
            cohesion=kwargs.get('cohesion', 20e6),
            friction_angle=kwargs.get('friction_angle', 7.0),
            cohesion_after_softening=kwargs.get('cohesion_after_softening', 5e6),
            friction_after_softening=kwargs.get('friction_after_softening', 3.0),
            name="Crust Plasticity"
        )
        
        return Material(
            name=name,
            density=AdvancedDensity(2620.0, thermal_expansivity=3e-5),
            viscosity=viscosity,
            plasticity=plasticity,
            thermal_conductivity=kwargs.get('thermal_conductivity', 2.5),
            heat_capacity=kwargs.get('heat_capacity', 1000.0),
            radiogenic_heat_production=kwargs.get('radiogenic_heat_production', 0.7e-3),
            min_viscosity=kwargs.get('min_viscosity', 1e18),
            max_viscosity=kwargs.get('max_viscosity', 1e25)
        )
    
    elif material_type == "mantle":
        # 橄榄石地幔材料（基于Underworld2的Olivine）
        viscosity = PowerLawCreep(
            pre_exponential_factor=kwargs.get('pre_exponential_factor', 1e-15),
            stress_exponent=kwargs.get('stress_exponent', 3.5),
            activation_energy=kwargs.get('activation_energy', 520e3),
            activation_volume=kwargs.get('activation_volume', 14e-6),
            grain_size=kwargs.get('grain_size', 1e-2),
            name="Olivine Creep"
        )
        
        return Material(
            name=name,
            density=AdvancedDensity(3300.0, thermal_expansivity=3e-5),
            viscosity=viscosity,
            thermal_conductivity=kwargs.get('thermal_conductivity', 3.0),
            heat_capacity=kwargs.get('heat_capacity', 1200.0),
            radiogenic_heat_production=kwargs.get('radiogenic_heat_production', 0.02e-3),
            min_viscosity=kwargs.get('min_viscosity', 1e19),
            max_viscosity=kwargs.get('max_viscosity', 1e26)
        )
    
    elif material_type == "air":
        # 空气材料（用于自由表面）
        return Material(
            name=name,
            density=AdvancedDensity(1.0),
            viscosity=ConstantViscosity(1e19),
            thermal_conductivity=kwargs.get('thermal_conductivity', 0.025),
            heat_capacity=kwargs.get('heat_capacity', 100.0),
            min_viscosity=kwargs.get('min_viscosity', 1e18),
            max_viscosity=kwargs.get('max_viscosity', 1e21)
        )
    
    else:
        raise ValueError(f"Unknown material type: {material_type}")


class MaterialSolver:
    """材料求解器（整合材料模型和求解过程）"""
    
    def __init__(self, materials: Dict[str, Material], mesh):
        self.materials = materials
        self.mesh = mesh
        self.material_field = None  # 材料场
        self.solution_history = []
    
    def set_material_field(self, material_field: np.ndarray):
        """设置材料场"""
        self.material_field = material_field
    
    def solve_material_properties(self, 
                                temperature: np.ndarray,
                                pressure: np.ndarray,
                                strain_rate: np.ndarray,
                                plastic_strain: np.ndarray,
                                time: float = 0.0) -> Dict[str, np.ndarray]:
        """求解材料属性"""
        
        # 创建材料状态
        material_state = MaterialState(
            temperature=temperature,
            pressure=pressure,
            strain_rate=strain_rate,
            plastic_strain=plastic_strain,
            time=time
        )
        
        # 为每个材料设置状态
        for material in self.materials.values():
            material.set_material_state(material_state)
        
        # 计算每个材料的属性
        results = {}
        for name, material in self.materials.items():
            viscosity = material.compute_effective_viscosity()
            density = material.compute_density(temperature, pressure)
            
            results[name] = {
                'viscosity': viscosity,
                'density': density,
                'thermal_conductivity': material.thermal_conductivity,
                'heat_capacity': material.heat_capacity,
                'radiogenic_heat_production': material.radiogenic_heat_production
            }
        
        return results
    
    def solve_with_material_field(self,
                                temperature: np.ndarray,
                                pressure: np.ndarray,
                                strain_rate: np.ndarray,
                                plastic_strain: np.ndarray,
                                time: float = 0.0) -> Dict[str, np.ndarray]:
        """基于材料场求解材料属性"""
        
        if self.material_field is None:
            raise ValueError("Material field not set")
        
        # 创建材料状态
        material_state = MaterialState(
            temperature=temperature,
            pressure=pressure,
            strain_rate=strain_rate,
            plastic_strain=plastic_strain,
            time=time
        )
        
        # 为每个材料设置状态
        for material in self.materials.values():
            material.set_material_state(material_state)
        
        # 基于材料场计算属性
        n_points = len(temperature)
        viscosity_field = np.zeros(n_points)
        density_field = np.zeros(n_points)
        thermal_conductivity_field = np.zeros(n_points)
        heat_capacity_field = np.zeros(n_points)
        radiogenic_heat_field = np.zeros(n_points)
        
        # 为每个材料计算属性
        material_properties = {}
        for name, material in self.materials.items():
            viscosity = material.compute_effective_viscosity()
            density = material.compute_density(temperature, pressure)
            
            material_properties[name] = {
                'viscosity': viscosity,
                'density': density,
                'thermal_conductivity': material.thermal_conductivity,
                'heat_capacity': material.heat_capacity,
                'radiogenic_heat_production': material.radiogenic_heat_production
            }
        
        # 基于材料场组合属性
        for i in range(n_points):
            material_id = int(self.material_field[i])
            material_name = list(self.materials.keys())[material_id]
            props = material_properties[material_name]
            
            viscosity_field[i] = props['viscosity'][i]
            density_field[i] = props['density'][i]
            thermal_conductivity_field[i] = props['thermal_conductivity']
            heat_capacity_field[i] = props['heat_capacity']
            radiogenic_heat_field[i] = props['radiogenic_heat_production']
        
        return {
            'viscosity': viscosity_field,
            'density': density_field,
            'thermal_conductivity': thermal_conductivity_field,
            'heat_capacity': heat_capacity_field,
            'radiogenic_heat_production': radiogenic_heat_field
        }


# 演示和测试函数
def demo_advanced_material_models():
    """演示高级材料模型功能"""
    print("🚀 高级材料模型演示")
    print("=" * 50)
    
    # 创建材料注册系统
    registry = MaterialRegistry()
    print(f"✅ 材料注册系统创建成功，包含材料: {registry.list_materials()}")
    
    # 测试地壳材料
    crust = registry.get_material("crust")
    print(f"\n🔧 测试地壳材料: {crust.name}")
    
    # 创建测试状态
    n_points = 100
    temperature = np.linspace(273, 1273, n_points)  # 0-1000°C
    pressure = np.linspace(0, 100e6, n_points)  # 0-100 MPa
    strain_rate = np.random.rand(n_points, 6) * 1e-15  # 随机应变率
    plastic_strain = np.random.rand(n_points) * 0.1  # 随机塑性应变
    
    material_state = MaterialState(
        temperature=temperature,
        pressure=pressure,
        strain_rate=strain_rate,
        plastic_strain=plastic_strain
    )
    
    crust.set_material_state(material_state)
    
    # 计算材料属性
    viscosity = crust.compute_effective_viscosity()
    density = crust.compute_density(temperature, pressure)
    
    print(f"   粘度范围: {np.min(viscosity):.2e} - {np.max(viscosity):.2e} Pa·s")
    print(f"   密度范围: {np.min(density):.1f} - {np.max(density):.1f} kg/m³")
    
    # 测试复合材料
    print(f"\n🔗 测试复合材料")
    mantle = registry.get_material("mantle")
    mantle.set_material_state(material_state)
    
    composite = create_composite_material(
        viscosity_rheologies=[crust.viscosity, mantle.viscosity],
        plasticity_rheologies=[crust.plasticity],
        combination_method="harmonic"
    )
    composite.set_material_state(material_state)
    
    composite_viscosity = composite.compute_effective_viscosity()
    print(f"   复合粘度范围: {np.min(composite_viscosity):.2e} - {np.max(composite_viscosity):.2e} Pa·s")
    
    # 测试Underworld2风格材料
    print(f"\n🌍 测试Underworld2风格材料")
    underworld_crust = create_underworld2_style_material(
        "Underworld Crust", 
        material_type="crust",
        cohesion=25e6,  # 25 MPa
        friction_angle=10.0  # 10 degrees
    )
    underworld_crust.set_material_state(material_state)
    
    uw_viscosity = underworld_crust.compute_effective_viscosity()
    print(f"   Underworld2地壳粘度范围: {np.min(uw_viscosity):.2e} - {np.max(uw_viscosity):.2e} Pa·s")
    
    print(f"\n✅ 高级材料模型演示完成!")


if __name__ == "__main__":
    demo_advanced_material_models() 