"""
相变动力学模型 
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class PhaseChangeKineticsState:
    """相变动力学状态"""
    temperature: np.ndarray  # 温度场
    pressure: np.ndarray     # 压力场
    melt_fraction: np.ndarray  # 熔体分数
    nucleation_density: np.ndarray  # 成核密度
    growth_rate: np.ndarray  # 生长速率
    phase_change_rate: np.ndarray  # 相变速率
    time: float = 0.0


class PhaseChangeKineticsModel(ABC):
    """相变动力学模型基类"""
    
    def __init__(self, name: str = "Phase Change Kinetics Model"):
        self.name = name
        self.kinetics_state: Optional[PhaseChangeKineticsState] = None
    
    @abstractmethod
    def compute_nucleation_rate(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算成核速率"""
        pass
    
    @abstractmethod
    def compute_growth_rate(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算生长速率"""
        pass
    
    @abstractmethod
    def compute_phase_change_rate(self, melt_fraction: np.ndarray, 
                                nucleation_rate: np.ndarray, 
                                growth_rate: np.ndarray) -> np.ndarray:
        """计算相变速率"""
        pass
    
    def set_kinetics_state(self, state: PhaseChangeKineticsState):
        """设置相变动力学状态"""
        self.kinetics_state = state


class RateLimitedPhaseChangeModel(PhaseChangeKineticsModel):
    """速率限制相变模型
    
    基于Underworld2的速率限制相变实现：
    考虑相变过程的动力学限制
    """
    
    def __init__(self,
                 max_phase_change_rate: float = 1e-3,  # 1/s
                 activation_energy: float = 200e3,  # J/mol
                 pre_exponential_factor: float = 1e6,  # 1/s
                 rate_limiting_factor: float = 1.0,
                 name: str = "Rate Limited Phase Change Model"):
        super().__init__(name)
        self.max_phase_change_rate = max_phase_change_rate
        self.activation_energy = activation_energy
        self.pre_exponential_factor = pre_exponential_factor
        self.rate_limiting_factor = rate_limiting_factor
        self.gas_constant = 8.314  # J/(mol·K)
    
    def compute_nucleation_rate(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算成核速率（基于Arrhenius方程）"""
        # 避免温度为零或负数
        temperature_safe = np.maximum(temperature, 1e-6)
        
        # Arrhenius方程：J = J₀ * exp(-E_a / (R * T))
        nucleation_rate = self.pre_exponential_factor * \
                         np.exp(-self.activation_energy / (self.gas_constant * temperature_safe))
        
        # 应用速率限制
        nucleation_rate = np.minimum(nucleation_rate, self.max_phase_change_rate)
        
        return nucleation_rate
    
    def compute_growth_rate(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算生长速率"""
        # 基于温度的简单生长模型
        temperature_safe = np.maximum(temperature, 1e-6)
        
        # 生长速率与温度成正比
        growth_rate = self.rate_limiting_factor * temperature_safe / 1000.0  # 归一化
        
        # 应用速率限制
        growth_rate = np.minimum(growth_rate, self.max_phase_change_rate)
        
        return growth_rate
    
    def compute_phase_change_rate(self, melt_fraction: np.ndarray, 
                                nucleation_rate: np.ndarray, 
                                growth_rate: np.ndarray) -> np.ndarray:
        """计算相变速率"""
        # 相变速率 = 成核速率 * 生长速率 * (1 - 熔体分数)
        # 考虑熔体分数对相变速率的影响
        phase_change_rate = nucleation_rate * growth_rate * (1.0 - melt_fraction)
        
        # 应用速率限制
        phase_change_rate = np.minimum(phase_change_rate, self.max_phase_change_rate)
        
        return phase_change_rate


class NucleationModel(PhaseChangeKineticsModel):
    """成核模型
    
    基于Underworld2的成核模型实现：
    考虑成核过程的统计特性
    """
    
    def __init__(self,
                 critical_nucleation_energy: float = 1e-18,  # J
                 nucleation_site_density: float = 1e12,  # 1/m³
                 nucleation_barrier: float = 1e-20,  # J
                 name: str = "Nucleation Model"):
        super().__init__(name)
        self.critical_nucleation_energy = critical_nucleation_energy
        self.nucleation_site_density = nucleation_site_density
        self.nucleation_barrier = nucleation_barrier
        self.gas_constant = 8.314  # J/(mol·K)
    
    def compute_nucleation_rate(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算成核速率"""
        # 避免温度为零或负数
        temperature_safe = np.maximum(temperature, 1e-6)
        
        # 成核速率：J = N * exp(-ΔG* / (k_B * T))
        # 其中 N 为成核位点密度，ΔG* 为成核能垒
        boltzmann_constant = 1.381e-23  # J/K
        
        nucleation_rate = self.nucleation_site_density * \
                         np.exp(-self.nucleation_barrier / (boltzmann_constant * temperature_safe))
        
        return nucleation_rate
    
    def compute_growth_rate(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算生长速率（成核模型中的简化版本）"""
        # 基于温度的简单生长模型
        temperature_safe = np.maximum(temperature, 1e-6)
        
        # 生长速率与温度成正比
        growth_rate = temperature_safe / 1000.0  # 归一化
        
        return growth_rate
    
    def compute_phase_change_rate(self, melt_fraction: np.ndarray, 
                                nucleation_rate: np.ndarray, 
                                growth_rate: np.ndarray) -> np.ndarray:
        """计算相变速率"""
        # 成核控制的相变速率
        phase_change_rate = nucleation_rate * (1.0 - melt_fraction)
        
        return phase_change_rate


class GrowthModel(PhaseChangeKineticsModel):
    """生长模型
    
    基于Underworld2的生长模型实现：
    考虑相变过程的生长机制
    """
    
    def __init__(self,
                 growth_activation_energy: float = 150e3,  # J/mol
                 growth_pre_factor: float = 1e5,  # m/s
                 growth_exponent: float = 1.0,
                 name: str = "Growth Model"):
        super().__init__(name)
        self.growth_activation_energy = growth_activation_energy
        self.growth_pre_factor = growth_pre_factor
        self.growth_exponent = growth_exponent
        self.gas_constant = 8.314  # J/(mol·K)
    
    def compute_nucleation_rate(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算成核速率（生长模型中的简化版本）"""
        # 简化的成核速率
        temperature_safe = np.maximum(temperature, 1e-6)
        
        nucleation_rate = 1e6 * np.exp(-100e3 / (self.gas_constant * temperature_safe))
        
        return nucleation_rate
    
    def compute_growth_rate(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算生长速率"""
        # 避免温度为零或负数
        temperature_safe = np.maximum(temperature, 1e-6)
        
        # 基于Arrhenius方程的生长速率
        growth_rate = self.growth_pre_factor * \
                     np.exp(-self.growth_activation_energy / (self.gas_constant * temperature_safe))
        
        # 应用生长指数
        growth_rate = growth_rate ** self.growth_exponent
        
        return growth_rate
    
    def compute_phase_change_rate(self, melt_fraction: np.ndarray, 
                                nucleation_rate: np.ndarray, 
                                growth_rate: np.ndarray) -> np.ndarray:
        """计算相变速率"""
        # 生长控制的相变速率
        phase_change_rate = growth_rate * melt_fraction * (1.0 - melt_fraction)
        
        return phase_change_rate


class CompositeKineticsModel(PhaseChangeKineticsModel):
    """复合动力学模型
    
    结合成核和生长过程的完整相变动力学模型
    """
    
    def __init__(self,
                 nucleation_model: NucleationModel,
                 growth_model: GrowthModel,
                 coupling_factor: float = 1.0,
                 name: str = "Composite Kinetics Model"):
        super().__init__(name)
        self.nucleation_model = nucleation_model
        self.growth_model = growth_model
        self.coupling_factor = coupling_factor
    
    def compute_nucleation_rate(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算成核速率"""
        return self.nucleation_model.compute_nucleation_rate(temperature, pressure)
    
    def compute_growth_rate(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算生长速率"""
        return self.growth_model.compute_growth_rate(temperature, pressure)
    
    def compute_phase_change_rate(self, melt_fraction: np.ndarray, 
                                nucleation_rate: np.ndarray, 
                                growth_rate: np.ndarray) -> np.ndarray:
        """计算相变速率"""
        # 结合成核和生长的相变速率
        nucleation_contribution = nucleation_rate * (1.0 - melt_fraction)
        growth_contribution = growth_rate * melt_fraction * (1.0 - melt_fraction)
        
        # 耦合成核和生长过程
        phase_change_rate = self.coupling_factor * (nucleation_contribution + growth_contribution)
        
        return phase_change_rate


class PhaseChangeKineticsSolver:
    """相变动力学求解器"""
    
    def __init__(self, kinetics_model: PhaseChangeKineticsModel):
        self.kinetics_model = kinetics_model
    
    def solve_phase_change_kinetics(self,
                                  temperature: np.ndarray,
                                  pressure: np.ndarray,
                                  melt_fraction: np.ndarray,
                                  dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """求解相变动力学"""
        
        # 设置动力学状态
        if self.kinetics_model.kinetics_state is None:
            state = PhaseChangeKineticsState(
                temperature=temperature,
                pressure=pressure,
                melt_fraction=melt_fraction,
                nucleation_density=np.zeros_like(melt_fraction),
                growth_rate=np.zeros_like(melt_fraction),
                phase_change_rate=np.zeros_like(melt_fraction)
            )
            self.kinetics_model.set_kinetics_state(state)
        
        # 计算成核速率
        nucleation_rate = self.kinetics_model.compute_nucleation_rate(temperature, pressure)
        
        # 计算生长速率
        growth_rate = self.kinetics_model.compute_growth_rate(temperature, pressure)
        
        # 计算相变速率
        phase_change_rate = self.kinetics_model.compute_phase_change_rate(
            melt_fraction, nucleation_rate, growth_rate)
        
        # 更新熔体分数
        new_melt_fraction = melt_fraction + phase_change_rate * dt
        
        # 确保熔体分数在[0, 1]范围内
        new_melt_fraction = np.clip(new_melt_fraction, 0.0, 1.0)
        
        return new_melt_fraction, nucleation_rate, growth_rate


def create_rate_limited_phase_change_model(max_rate: float = 1e-3,
                                         activation_energy: float = 200e3,
                                         pre_factor: float = 1e6) -> RateLimitedPhaseChangeModel:
    """创建速率限制相变模型"""
    return RateLimitedPhaseChangeModel(
        max_phase_change_rate=max_rate,
        activation_energy=activation_energy,
        pre_exponential_factor=pre_factor
    )


def create_nucleation_model(critical_energy: float = 1e-18,
                          site_density: float = 1e12,
                          barrier: float = 1e-20) -> NucleationModel:
    """创建成核模型"""
    return NucleationModel(
        critical_nucleation_energy=critical_energy,
        nucleation_site_density=site_density,
        nucleation_barrier=barrier
    )


def create_growth_model(activation_energy: float = 150e3,
                       pre_factor: float = 1e5,
                       exponent: float = 1.0) -> GrowthModel:
    """创建生长模型"""
    return GrowthModel(
        growth_activation_energy=activation_energy,
        growth_pre_factor=pre_factor,
        growth_exponent=exponent
    )


def create_composite_kinetics_model(nucleation_model: NucleationModel,
                                  growth_model: GrowthModel,
                                  coupling_factor: float = 1.0) -> CompositeKineticsModel:
    """创建复合动力学模型"""
    return CompositeKineticsModel(
        nucleation_model=nucleation_model,
        growth_model=growth_model,
        coupling_factor=coupling_factor
    )


def demo_phase_change_kinetics():
    """演示相变动力学模型功能"""
    print("🔧 相变动力学模型演示")
    print("=" * 50)
    
    # 创建测试数据
    n_points = 100
    temperature = np.linspace(1000, 2000, n_points)  # K
    pressure = np.ones(n_points) * 1e8  # Pa
    melt_fraction = np.zeros(n_points)
    dt = 1.0
    
    # 测试速率限制相变模型
    print("\n🔧 测试速率限制相变模型...")
    rate_limited_model = create_rate_limited_phase_change_model(
        max_rate=1e-3,
        activation_energy=200e3
    )
    
    solver = PhaseChangeKineticsSolver(rate_limited_model)
    new_melt_fraction, nucleation_rate, growth_rate = solver.solve_phase_change_kinetics(
        temperature, pressure, melt_fraction, dt)
    
    print(f"   最大成核速率: {nucleation_rate.max():.2e} 1/s")
    print(f"   最大生长速率: {growth_rate.max():.2e} m/s")
    print(f"   最大熔体分数: {new_melt_fraction.max():.3f}")
    
    # 测试成核模型
    print("\n🔧 测试成核模型...")
    nucleation_model = create_nucleation_model(
        critical_energy=1e-18,
        site_density=1e12
    )
    
    solver = PhaseChangeKineticsSolver(nucleation_model)
    new_melt_fraction, nucleation_rate, growth_rate = solver.solve_phase_change_kinetics(
        temperature, pressure, melt_fraction, dt)
    
    print(f"   最大成核速率: {nucleation_rate.max():.2e} 1/s")
    print(f"   成核位点密度: {nucleation_model.nucleation_site_density:.1e} 1/m³")
    
    # 测试生长模型
    print("\n🔧 测试生长模型...")
    growth_model = create_growth_model(
        activation_energy=150e3,
        pre_factor=1e5
    )
    
    solver = PhaseChangeKineticsSolver(growth_model)
    new_melt_fraction, nucleation_rate, growth_rate = solver.solve_phase_change_kinetics(
        temperature, pressure, melt_fraction, dt)
    
    print(f"   最大生长速率: {growth_rate.max():.2e} m/s")
    print(f"   生长激活能: {growth_model.growth_activation_energy:.1e} J/mol")
    
    # 测试复合动力学模型
    print("\n🔧 测试复合动力学模型...")
    composite_model = create_composite_kinetics_model(
        nucleation_model=nucleation_model,
        growth_model=growth_model,
        coupling_factor=1.0
    )
    
    solver = PhaseChangeKineticsSolver(composite_model)
    new_melt_fraction, nucleation_rate, growth_rate = solver.solve_phase_change_kinetics(
        temperature, pressure, melt_fraction, dt)
    
    print(f"   最大相变速率: {composite_model.compute_phase_change_rate(melt_fraction, nucleation_rate, growth_rate).max():.2e} 1/s")
    print(f"   耦合因子: {composite_model.coupling_factor}")
    
    print("\n✅ 相变动力学模型演示完成!")


if __name__ == "__main__":
    demo_phase_change_kinetics()
