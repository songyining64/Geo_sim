"""
相变模型
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class PhaseState:
    """相变状态"""
    temperature: np.ndarray  # 温度场
    pressure: np.ndarray     # 压力场
    melt_fraction: np.ndarray  # 熔体分数
    phase_composition: np.ndarray  # 相组成
    time: float = 0.0


class PhaseChangeModel(ABC):
    """相变模型基类"""
    
    def __init__(self, name: str = "Phase Change Model"):
        self.name = name
        self.phase_state: Optional[PhaseState] = None
    
    @abstractmethod
    def compute_melt_fraction(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算熔体分数"""
        pass
    
    @abstractmethod
    def compute_latent_heat(self, melt_fraction: np.ndarray) -> np.ndarray:
        """计算潜热"""
        pass
    
    @abstractmethod
    def compute_phase_change_stress(self, melt_fraction: np.ndarray) -> np.ndarray:
        """计算相变应力"""
        pass
    
    def set_phase_state(self, state: PhaseState):
        """设置相变状态"""
        self.phase_state = state


class SolidusLiquidusModel(PhaseChangeModel):
    """固相线-液相线模型
    
    基于Underworld2的相变模型实现：
    使用固相线和液相线定义相变区间
    """
    
    def __init__(self,
                 solidus_temperature: float = 1200.0,  # K
                 liquidus_temperature: float = 1400.0,  # K
                 latent_heat_fusion: float = 400e3,  # J/kg
                 melt_expansion: float = 0.1,  # 熔体膨胀系数
                 pressure_dependence: float = 0.0,  # 压力依赖性
                 name: str = "Solidus-Liquidus Model"):
        super().__init__(name)
        self.solidus_temperature = solidus_temperature
        self.liquidus_temperature = liquidus_temperature
        self.latent_heat_fusion = latent_heat_fusion
        self.melt_expansion = melt_expansion
        self.pressure_dependence = pressure_dependence
    
    def compute_solidus_temperature(self, pressure: np.ndarray) -> np.ndarray:
        """计算固相线温度"""
        return self.solidus_temperature + self.pressure_dependence * pressure
    
    def compute_liquidus_temperature(self, pressure: np.ndarray) -> np.ndarray:
        """计算液相线温度"""
        return self.liquidus_temperature + self.pressure_dependence * pressure
    
    def compute_melt_fraction(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算熔体分数"""
        solidus = self.compute_solidus_temperature(pressure)
        liquidus = self.compute_liquidus_temperature(pressure)
        
        # 线性插值
        melt_fraction = np.zeros_like(temperature)
        
        # 完全固态
        solid_mask = temperature <= solidus
        melt_fraction[solid_mask] = 0.0
        
        # 完全液态
        liquid_mask = temperature >= liquidus
        melt_fraction[liquid_mask] = 1.0
        
        # 相变区间
        transition_mask = (temperature > solidus) & (temperature < liquidus)
        if np.any(transition_mask):
            melt_fraction[transition_mask] = (temperature[transition_mask] - solidus[transition_mask]) / \
                                           (liquidus[transition_mask] - solidus[transition_mask])
        
        return np.clip(melt_fraction, 0.0, 1.0)
    
    def compute_latent_heat(self, melt_fraction: np.ndarray) -> np.ndarray:
        """计算潜热"""
        return self.latent_heat_fusion * melt_fraction
    
    def compute_phase_change_stress(self, melt_fraction: np.ndarray) -> np.ndarray:
        """计算相变应力"""
        # 简化的相变应力模型
        # 实际实现需要考虑更复杂的应力演化
        return np.zeros_like(melt_fraction)


class PeridotiteMeltingModel(PhaseChangeModel):
    """橄榄岩熔融模型
    
    基于实验数据的橄榄岩熔融模型
    """
    
    def __init__(self,
                 latent_heat_fusion: float = 400e3,  # J/kg
                 melt_expansion: float = 0.1,  # 熔体膨胀系数
                 name: str = "Peridotite Melting Model"):
        super().__init__(name)
        self.latent_heat_fusion = latent_heat_fusion
        self.melt_expansion = melt_expansion
        
        # 橄榄岩熔融参数（基于实验数据）
        self.dry_solidus_params = {
            'A': 1085.7,  # K
            'B': 132.9,   # K/GPa
            'C': -5.1     # K/GPa²
        }
        
        self.dry_liquidus_params = {
            'A': 1780.0,  # K
            'B': 45.0,    # K/GPa
            'C': -2.0     # K/GPa²
        }
    
    def compute_dry_solidus(self, pressure: np.ndarray) -> np.ndarray:
        """计算干固相线"""
        P_GPa = pressure / 1e9  # 转换为GPa
        return (self.dry_solidus_params['A'] + 
                self.dry_solidus_params['B'] * P_GPa + 
                self.dry_solidus_params['C'] * P_GPa**2)
    
    def compute_dry_liquidus(self, pressure: np.ndarray) -> np.ndarray:
        """计算干液相线"""
        P_GPa = pressure / 1e9  # 转换为GPa
        return (self.dry_liquidus_params['A'] + 
                self.dry_liquidus_params['B'] * P_GPa + 
                self.dry_liquidus_params['C'] * P_GPa**2)
    
    def compute_melt_fraction(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算熔体分数"""
        solidus = self.compute_dry_solidus(pressure)
        liquidus = self.compute_dry_liquidus(pressure)
        
        # 使用更复杂的熔融函数
        melt_fraction = np.zeros_like(temperature)
        
        # 完全固态
        solid_mask = temperature <= solidus
        melt_fraction[solid_mask] = 0.0
        
        # 完全液态
        liquid_mask = temperature >= liquidus
        melt_fraction[liquid_mask] = 1.0
        
        # 相变区间（使用非线性函数）
        transition_mask = (temperature > solidus) & (temperature < liquidus)
        if np.any(transition_mask):
            T_norm = (temperature[transition_mask] - solidus[transition_mask]) / \
                    (liquidus[transition_mask] - solidus[transition_mask])
            
            # 使用三次多项式拟合实验数据
            melt_fraction[transition_mask] = 3 * T_norm**2 - 2 * T_norm**3
        
        return np.clip(melt_fraction, 0.0, 1.0)
    
    def compute_latent_heat(self, melt_fraction: np.ndarray) -> np.ndarray:
        """计算潜热"""
        return self.latent_heat_fusion * melt_fraction
    
    def compute_phase_change_stress(self, melt_fraction: np.ndarray) -> np.ndarray:
        """计算相变应力"""
        # 考虑熔体膨胀的应力
        return -self.melt_expansion * melt_fraction


class PhaseChangeSolver:
    """相变求解器"""
    
    def __init__(self, phase_change_model: PhaseChangeModel):
        self.phase_change_model = phase_change_model
    
    def solve_phase_change(self,
                          temperature: np.ndarray,
                          pressure: np.ndarray,
                          melt_fraction: np.ndarray,
                          dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """求解相变"""
        
        # 设置相变状态
        state = PhaseState(
            temperature=temperature,
            pressure=pressure,
            melt_fraction=melt_fraction,
            phase_composition=np.zeros_like(melt_fraction)
        )
        self.phase_change_model.set_phase_state(state)
        
        # 计算新的熔体分数
        new_melt_fraction = self.phase_change_model.compute_melt_fraction(temperature, pressure)
        
        # 计算潜热
        latent_heat = self.phase_change_model.compute_latent_heat(new_melt_fraction)
        
        # 计算相变应力
        phase_change_stress = self.phase_change_model.compute_phase_change_stress(new_melt_fraction)
        
        return new_melt_fraction, latent_heat


class CompositePhaseChangeModel(PhaseChangeModel):
    """复合相变模型"""
    
    def __init__(self, models: List[PhaseChangeModel], weights: Optional[List[float]] = None):
        super().__init__("Composite Phase Change Model")
        self.models = models
        self.weights = weights if weights is not None else [1.0] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("权重数量必须与模型数量相同")
    
    def compute_melt_fraction(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算复合熔体分数"""
        melt_fractions = []
        for model in self.models:
            melt_fraction = model.compute_melt_fraction(temperature, pressure)
            melt_fractions.append(melt_fraction)
        
        # 加权平均
        composite_melt_fraction = np.zeros_like(temperature)
        for i, weight in enumerate(self.weights):
            composite_melt_fraction += weight * melt_fractions[i]
        
        return composite_melt_fraction
    
    def compute_latent_heat(self, melt_fraction: np.ndarray) -> np.ndarray:
        """计算复合潜热"""
        latent_heats = []
        for model in self.models:
            latent_heat = model.compute_latent_heat(melt_fraction)
            latent_heats.append(latent_heat)
        
        # 加权平均
        composite_latent_heat = np.zeros_like(melt_fraction)
        for i, weight in enumerate(self.weights):
            composite_latent_heat += weight * latent_heats[i]
        
        return composite_latent_heat
    
    def compute_phase_change_stress(self, melt_fraction: np.ndarray) -> np.ndarray:
        """计算复合相变应力"""
        phase_change_stresses = []
        for model in self.models:
            stress = model.compute_phase_change_stress(melt_fraction)
            phase_change_stresses.append(stress)
        
        # 加权平均
        composite_stress = np.zeros_like(melt_fraction)
        for i, weight in enumerate(self.weights):
            composite_stress += weight * phase_change_stresses[i]
        
        return composite_stress


def create_solidus_liquidus_model(solidus_temp: float = 1200.0,
                                 liquidus_temp: float = 1400.0,
                                 latent_heat: float = 400e3) -> SolidusLiquidusModel:
    """创建固相线-液相线模型"""
    return SolidusLiquidusModel(solidus_temp, liquidus_temp, latent_heat)


def create_peridotite_melting_model(latent_heat: float = 400e3) -> PeridotiteMeltingModel:
    """创建橄榄岩熔融模型"""
    return PeridotiteMeltingModel(latent_heat)


def create_composite_phase_change_model(models: List[PhaseChangeModel],
                                      weights: Optional[List[float]] = None) -> CompositePhaseChangeModel:
    """创建复合相变模型"""
    return CompositePhaseChangeModel(models, weights)


def demo_phase_change_models():
    """演示相变模型功能"""
    print("🔥 相变模型演示")
    print("=" * 50)
    
    # 创建测试数据
    n_points = 100
    temperature = np.linspace(1000, 1600, n_points)  # K
    pressure = np.full(n_points, 1e9)  # Pa
    melt_fraction = np.zeros(n_points)
    
    # 测试固相线-液相线模型
    print("\n🔥 测试固相线-液相线模型...")
    solidus_liquidus = SolidusLiquidusModel(
        solidus_temperature=1200.0,
        liquidus_temperature=1400.0,
        latent_heat_fusion=400e3
    )
    
    solver = PhaseChangeSolver(solidus_liquidus)
    new_melt_fraction, latent_heat = solver.solve_phase_change(
        temperature, pressure, melt_fraction, 1.0)
    
    print(f"   温度范围: {temperature.min():.0f} - {temperature.max():.0f} K")
    print(f"   熔体分数范围: {new_melt_fraction.min():.3f} - {new_melt_fraction.max():.3f}")
    print(f"   最大潜热: {latent_heat.max():.1e} J/kg")
    
    # 测试橄榄岩熔融模型
    print("\n🔥 测试橄榄岩熔融模型...")
    peridotite = PeridotiteMeltingModel(latent_heat_fusion=400e3)
    
    solver = PhaseChangeSolver(peridotite)
    new_melt_fraction, latent_heat = solver.solve_phase_change(
        temperature, pressure, melt_fraction, 1.0)
    
    print(f"   温度范围: {temperature.min():.0f} - {temperature.max():.0f} K")
    print(f"   熔体分数范围: {new_melt_fraction.min():.3f} - {new_melt_fraction.max():.3f}")
    print(f"   最大潜热: {latent_heat.max():.1e} J/kg")
    
    print("\n✅ 相变模型演示完成!")


if __name__ == "__main__":
    demo_phase_change_models()
