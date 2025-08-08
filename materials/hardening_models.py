"""
硬化模型
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class HardeningState:
    """硬化状态"""
    plastic_strain: np.ndarray  # 累积塑性应变
    hardening_variable: np.ndarray  # 硬化变量
    back_stress: np.ndarray  # 背应力
    cycle_count: np.ndarray  # 循环次数
    time: float = 0.0


class HardeningModel(ABC):
    """硬化模型基类"""
    
    def __init__(self, name: str = "Hardening Model"):
        self.name = name
        self.hardening_state: Optional[HardeningState] = None
    
    @abstractmethod
    def compute_hardening_stress(self, plastic_strain: np.ndarray) -> np.ndarray:
        """计算硬化应力"""
        pass
    
    @abstractmethod
    def compute_hardening_modulus(self, plastic_strain: np.ndarray) -> np.ndarray:
        """计算硬化模量"""
        pass
    
    @abstractmethod
    def update_hardening_state(self, plastic_strain_rate: np.ndarray, dt: float) -> np.ndarray:
        """更新硬化状态"""
        pass
    
    def set_hardening_state(self, state: HardeningState):
        """设置硬化状态"""
        self.hardening_state = state


class LinearHardeningModel(HardeningModel):
    """线性硬化模型
    
    基于Underworld2的线性硬化实现：
    σ_h = H * ε_p
    其中 H 为硬化模量，ε_p 为累积塑性应变
    """
    
    def __init__(self,
                 hardening_modulus: float = 1e9,  # Pa
                 initial_yield_stress: float = 50e6,  # Pa
                 max_hardening_stress: Optional[float] = None,  # Pa
                 name: str = "Linear Hardening Model"):
        super().__init__(name)
        self.hardening_modulus = hardening_modulus
        self.initial_yield_stress = initial_yield_stress
        self.max_hardening_stress = max_hardening_stress
    
    def compute_hardening_stress(self, plastic_strain: np.ndarray) -> np.ndarray:
        """计算硬化应力 σ_h = H * ε_p"""
        hardening_stress = self.hardening_modulus * plastic_strain
        
        # 应用最大硬化应力限制
        if self.max_hardening_stress is not None:
            hardening_stress = np.minimum(hardening_stress, self.max_hardening_stress)
        
        return hardening_stress
    
    def compute_hardening_modulus(self, plastic_strain: np.ndarray) -> np.ndarray:
        """计算硬化模量（线性硬化为常数）"""
        return np.full_like(plastic_strain, self.hardening_modulus)
    
    def update_hardening_state(self, plastic_strain_rate: np.ndarray, dt: float) -> np.ndarray:
        """更新硬化状态"""
        if self.hardening_state is None:
            return np.zeros_like(plastic_strain_rate)
        
        # 更新累积塑性应变
        new_plastic_strain = self.hardening_state.plastic_strain + np.abs(plastic_strain_rate) * dt
        
        # 更新硬化变量
        hardening_variable = self.compute_hardening_stress(new_plastic_strain)
        
        return hardening_variable


class NonlinearHardeningModel(HardeningModel):
    """非线性硬化模型
    
    基于Underworld2的非线性硬化实现：
    σ_h = σ_sat * (1 - exp(-H * ε_p / σ_sat))
    其中 σ_sat 为饱和应力，H 为初始硬化模量
    """
    
    def __init__(self,
                 saturation_stress: float = 200e6,  # Pa
                 initial_hardening_modulus: float = 2e9,  # Pa
                 hardening_exponent: float = 1.0,
                 initial_yield_stress: float = 50e6,  # Pa
                 name: str = "Nonlinear Hardening Model"):
        super().__init__(name)
        self.saturation_stress = saturation_stress
        self.initial_hardening_modulus = initial_hardening_modulus
        self.hardening_exponent = hardening_exponent
        self.initial_yield_stress = initial_yield_stress
    
    def compute_hardening_stress(self, plastic_strain: np.ndarray) -> np.ndarray:
        """计算硬化应力 σ_h = σ_sat * (1 - exp(-H * ε_p / σ_sat))"""
        # 避免数值问题
        plastic_strain_safe = np.maximum(plastic_strain, 1e-12)
        
        # 非线性硬化公式
        exponent = -self.initial_hardening_modulus * plastic_strain_safe / self.saturation_stress
        hardening_stress = self.saturation_stress * (1.0 - np.exp(exponent))
        
        return hardening_stress
    
    def compute_hardening_modulus(self, plastic_strain: np.ndarray) -> np.ndarray:
        """计算硬化模量 H = H_0 * exp(-H_0 * ε_p / σ_sat)"""
        # 避免数值问题
        plastic_strain_safe = np.maximum(plastic_strain, 1e-12)
        
        exponent = -self.initial_hardening_modulus * plastic_strain_safe / self.saturation_stress
        hardening_modulus = self.initial_hardening_modulus * np.exp(exponent)
        
        return hardening_modulus
    
    def update_hardening_state(self, plastic_strain_rate: np.ndarray, dt: float) -> np.ndarray:
        """更新硬化状态"""
        if self.hardening_state is None:
            return np.zeros_like(plastic_strain_rate)
        
        # 更新累积塑性应变
        new_plastic_strain = self.hardening_state.plastic_strain + np.abs(plastic_strain_rate) * dt
        
        # 更新硬化变量
        hardening_variable = self.compute_hardening_stress(new_plastic_strain)
        
        return hardening_variable


class CyclicHardeningModel(HardeningModel):
    """循环硬化模型
    
    基于Underworld2的循环硬化实现：
    考虑循环加载对材料硬化的影响
    """
    
    def __init__(self,
                 monotonic_hardening_modulus: float = 1e9,  # Pa
                 cyclic_hardening_modulus: float = 5e8,  # Pa
                 saturation_stress: float = 200e6,  # Pa
                 cycle_hardening_exponent: float = 0.5,
                 initial_yield_stress: float = 50e6,  # Pa
                 name: str = "Cyclic Hardening Model"):
        super().__init__(name)
        self.monotonic_hardening_modulus = monotonic_hardening_modulus
        self.cyclic_hardening_modulus = cyclic_hardening_modulus
        self.saturation_stress = saturation_stress
        self.cycle_hardening_exponent = cycle_hardening_exponent
        self.initial_yield_stress = initial_yield_stress
    
    def compute_hardening_stress(self, plastic_strain: np.ndarray) -> np.ndarray:
        """计算硬化应力（考虑循环效应）"""
        if self.hardening_state is None:
            return np.zeros_like(plastic_strain)
        
        # 单调硬化分量
        monotonic_hardening = self.monotonic_hardening_modulus * plastic_strain
        
        # 循环硬化分量
        cycle_count = self.hardening_state.cycle_count
        cycle_hardening = self.cyclic_hardening_modulus * (cycle_count ** self.cycle_hardening_exponent)
        
        # 总硬化应力
        total_hardening = monotonic_hardening + cycle_hardening
        
        # 应用饱和限制
        total_hardening = np.minimum(total_hardening, self.saturation_stress)
        
        return total_hardening
    
    def compute_hardening_modulus(self, plastic_strain: np.ndarray) -> np.ndarray:
        """计算硬化模量"""
        if self.hardening_state is None:
            return np.full_like(plastic_strain, self.monotonic_hardening_modulus)
        
        # 单调硬化模量
        monotonic_modulus = self.monotonic_hardening_modulus
        
        # 循环硬化模量
        cycle_count = self.hardening_state.cycle_count
        cycle_modulus = self.cyclic_hardening_modulus * self.cycle_hardening_exponent * \
                       (cycle_count ** (self.cycle_hardening_exponent - 1))
        
        # 总硬化模量
        total_modulus = monotonic_modulus + cycle_modulus
        
        return total_modulus
    
    def detect_cycle(self, plastic_strain_rate: np.ndarray, plastic_strain: np.ndarray) -> np.ndarray:
        """检测循环"""
        if self.hardening_state is None:
            return np.zeros_like(plastic_strain_rate)
        
        # 简单的循环检测：塑性应变率符号变化
        current_sign = np.sign(plastic_strain_rate)
        previous_sign = np.sign(self.hardening_state.plastic_strain - plastic_strain)
        
        # 检测符号变化
        cycle_detected = (current_sign != previous_sign) & (previous_sign != 0)
        
        return cycle_detected
    
    def update_hardening_state(self, plastic_strain_rate: np.ndarray, dt: float) -> np.ndarray:
        """更新硬化状态"""
        if self.hardening_state is None:
            return np.zeros_like(plastic_strain_rate)
        
        # 更新累积塑性应变
        new_plastic_strain = self.hardening_state.plastic_strain + np.abs(plastic_strain_rate) * dt
        
        # 检测循环
        cycle_detected = self.detect_cycle(plastic_strain_rate, new_plastic_strain)
        
        # 更新循环次数
        new_cycle_count = self.hardening_state.cycle_count.copy()
        new_cycle_count[cycle_detected] += 1
        
        # 更新硬化变量
        hardening_variable = self.compute_hardening_stress(new_plastic_strain)
        
        return hardening_variable


class HardeningSolver:
    """硬化求解器"""
    
    def __init__(self, hardening_model: HardeningModel):
        self.hardening_model = hardening_model
    
    def solve_hardening(self,
                       plastic_strain: np.ndarray,
                       plastic_strain_rate: np.ndarray,
                       dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """求解硬化"""
        
        # 设置硬化状态
        if self.hardening_model.hardening_state is None:
            state = HardeningState(
                plastic_strain=plastic_strain,
                hardening_variable=np.zeros_like(plastic_strain),
                back_stress=np.zeros_like(plastic_strain),
                cycle_count=np.zeros_like(plastic_strain)
            )
            self.hardening_model.set_hardening_state(state)
        
        # 更新硬化状态
        hardening_variable = self.hardening_model.update_hardening_state(plastic_strain_rate, dt)
        
        # 更新累积塑性应变
        new_plastic_strain = plastic_strain + np.abs(plastic_strain_rate) * dt
        
        return hardening_variable, new_plastic_strain


def create_linear_hardening_model(hardening_modulus: float = 1e9,
                                initial_yield_stress: float = 50e6,
                                max_hardening_stress: Optional[float] = None) -> LinearHardeningModel:
    """创建线性硬化模型"""
    return LinearHardeningModel(
        hardening_modulus=hardening_modulus,
        initial_yield_stress=initial_yield_stress,
        max_hardening_stress=max_hardening_stress
    )


def create_nonlinear_hardening_model(saturation_stress: float = 200e6,
                                   initial_hardening_modulus: float = 2e9,
                                   hardening_exponent: float = 1.0,
                                   initial_yield_stress: float = 50e6) -> NonlinearHardeningModel:
    """创建非线性硬化模型"""
    return NonlinearHardeningModel(
        saturation_stress=saturation_stress,
        initial_hardening_modulus=initial_hardening_modulus,
        hardening_exponent=hardening_exponent,
        initial_yield_stress=initial_yield_stress
    )


def create_cyclic_hardening_model(monotonic_hardening_modulus: float = 1e9,
                                cyclic_hardening_modulus: float = 5e8,
                                saturation_stress: float = 200e6,
                                cycle_hardening_exponent: float = 0.5,
                                initial_yield_stress: float = 50e6) -> CyclicHardeningModel:
    """创建循环硬化模型"""
    return CyclicHardeningModel(
        monotonic_hardening_modulus=monotonic_hardening_modulus,
        cyclic_hardening_modulus=cyclic_hardening_modulus,
        saturation_stress=saturation_stress,
        cycle_hardening_exponent=cycle_hardening_exponent,
        initial_yield_stress=initial_yield_stress
    )


def demo_hardening_models():
    """演示硬化模型功能"""
    print("🔧 硬化模型演示")
    print("=" * 50)
    
    # 创建测试数据
    n_points = 100
    plastic_strain = np.linspace(0, 0.1, n_points)
    plastic_strain_rate = np.ones(n_points) * 1e-6
    dt = 1.0
    
    # 测试线性硬化
    print("\n🔧 测试线性硬化...")
    linear_hardening = create_linear_hardening_model(
        hardening_modulus=1e9,
        initial_yield_stress=50e6
    )
    
    solver = HardeningSolver(linear_hardening)
    hardening_stress, new_plastic_strain = solver.solve_hardening(
        plastic_strain, plastic_strain_rate, dt)
    
    print(f"   最大硬化应力: {hardening_stress.max():.1e} Pa")
    print(f"   硬化模量: {linear_hardening.hardening_modulus:.1e} Pa")
    
    # 测试非线性硬化
    print("\n🔧 测试非线性硬化...")
    nonlinear_hardening = create_nonlinear_hardening_model(
        saturation_stress=200e6,
        initial_hardening_modulus=2e9,
        hardening_exponent=1.0
    )
    
    solver = HardeningSolver(nonlinear_hardening)
    hardening_stress, new_plastic_strain = solver.solve_hardening(
        plastic_strain, plastic_strain_rate, dt)
    
    print(f"   最大硬化应力: {hardening_stress.max():.1e} Pa")
    print(f"   饱和应力: {nonlinear_hardening.saturation_stress:.1e} Pa")
    
    # 测试循环硬化
    print("\n🔧 测试循环硬化...")
    cyclic_hardening = create_cyclic_hardening_model(
        monotonic_hardening_modulus=1e9,
        cyclic_hardening_modulus=5e8,
        saturation_stress=200e6
    )
    
    # 模拟循环加载
    cycle_plastic_strain = np.zeros(n_points)
    cycle_plastic_strain_rate = np.sin(np.linspace(0, 4*np.pi, n_points)) * 1e-6
    
    solver = HardeningSolver(cyclic_hardening)
    hardening_stress, new_plastic_strain = solver.solve_hardening(
        cycle_plastic_strain, cycle_plastic_strain_rate, dt)
    
    print(f"   最大硬化应力: {hardening_stress.max():.1e} Pa")
    print(f"   循环硬化模量: {cyclic_hardening.cyclic_hardening_modulus:.1e} Pa")
    
    print("\n✅ 硬化模型演示完成!")


if __name__ == "__main__":
    demo_hardening_models()
