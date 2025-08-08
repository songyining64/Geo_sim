"""
塑性模型
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class PlasticState:
    """塑性状态"""
    stress: np.ndarray  # 应力张量 (3x3 或 2x2)
    strain: np.ndarray  # 应变张量
    plastic_strain: np.ndarray  # 累积塑性应变
    hardening_variable: np.ndarray  # 硬化变量
    time: float = 0.0


class PlasticModel(ABC):
    """塑性模型基类"""
    
    def __init__(self, name: str = "Plastic Model"):
        self.name = name
        self.material_state: Optional[PlasticState] = None
    
    @abstractmethod
    def compute_yield_function(self, stress: np.ndarray) -> np.ndarray:
        """计算屈服函数 f(σ)"""
        pass
    
    @abstractmethod
    def compute_plastic_flow_direction(self, stress: np.ndarray) -> np.ndarray:
        """计算塑性流动方向 ∂f/∂σ"""
        pass
    
    @abstractmethod
    def compute_consistency_parameter(self, stress: np.ndarray, strain_rate: np.ndarray) -> np.ndarray:
        """计算一致性参数"""
        pass
    
    def set_material_state(self, state: PlasticState):
        """设置材料状态"""
        self.material_state = state


class VonMisesPlasticity(PlasticModel):
    """von Mises塑性模型
    
    基于Underworld2的VonMises实现：
    f(σ) = √(3J₂) - σ_y
    其中 J₂ = 1/2 s:s，s为偏应力张量
    """
    
    def __init__(self, 
                 yield_stress: float,
                 yield_stress_after_softening: Optional[float] = None,
                 softening_start: float = 0.5,
                 softening_end: float = 1.5,
                 dimension: int = 2,
                 name: str = "Von Mises Plasticity"):
        super().__init__(name)
        self.yield_stress = yield_stress
        self.yield_stress_after_softening = yield_stress_after_softening
        self.softening_start = softening_start
        self.softening_end = softening_end
        self.dimension = dimension
        
        # 弱化参数
        self.weakening_enabled = yield_stress_after_softening is not None
    
    def _compute_weakening_factor(self, plastic_strain: np.ndarray) -> np.ndarray:
        """计算弱化因子"""
        if not self.weakening_enabled:
            return np.ones_like(plastic_strain)
        
        # 线性弱化
        weakening_factor = np.ones_like(plastic_strain)
        
        # 开始弱化
        mask1 = plastic_strain >= self.softening_start
        mask2 = plastic_strain <= self.softening_end
        
        # 弱化区间
        weakening_region = mask1 & mask2
        
        if np.any(weakening_region):
            # 线性插值
            factor = (plastic_strain[weakening_region] - self.softening_start) / \
                    (self.softening_end - self.softening_start)
            weakening_factor[weakening_region] = 1.0 - factor * \
                (1.0 - self.yield_stress_after_softening / self.yield_stress)
        
        # 完全弱化
        mask3 = plastic_strain > self.softening_end
        if np.any(mask3):
            weakening_factor[mask3] = self.yield_stress_after_softening / self.yield_stress
        
        return weakening_factor
    
    def _compute_deviatoric_stress(self, stress: np.ndarray) -> np.ndarray:
        """计算偏应力张量"""
        if self.dimension == 2:
            # 2D情况
            mean_stress = 0.5 * (stress[:, 0, 0] + stress[:, 1, 1])
            deviatoric = stress.copy()
            deviatoric[:, 0, 0] -= mean_stress
            deviatoric[:, 1, 1] -= mean_stress
        else:
            # 3D情况
            mean_stress = (1.0/3.0) * (stress[:, 0, 0] + stress[:, 1, 1] + stress[:, 2, 2])
            deviatoric = stress.copy()
            deviatoric[:, 0, 0] -= mean_stress
            deviatoric[:, 1, 1] -= mean_stress
            deviatoric[:, 2, 2] -= mean_stress
        
        return deviatoric
    
    def _compute_j2_invariant(self, stress: np.ndarray) -> np.ndarray:
        """计算J₂不变量"""
        deviatoric = self._compute_deviatoric_stress(stress)
        
        if self.dimension == 2:
            # 2D: J₂ = 1/2 (s₁₁² + s₂₂² + 2s₁₂²)
            j2 = 0.5 * (deviatoric[:, 0, 0]**2 + deviatoric[:, 1, 1]**2 + 
                        2 * deviatoric[:, 0, 1]**2)
        else:
            # 3D: J₂ = 1/2 s:s
            j2 = 0.5 * (deviatoric[:, 0, 0]**2 + deviatoric[:, 1, 1]**2 + deviatoric[:, 2, 2]**2 +
                        2 * (deviatoric[:, 0, 1]**2 + deviatoric[:, 0, 2]**2 + deviatoric[:, 1, 2]**2))
        
        return j2
    
    def compute_yield_function(self, stress: np.ndarray) -> np.ndarray:
        """计算屈服函数 f(σ) = √(3J₂) - σ_y"""
        j2 = self._compute_j2_invariant(stress)
        von_mises_stress = np.sqrt(3 * j2)
        
        # 应用弱化
        if self.weakening_enabled and self.material_state is not None:
            weakening_factor = self._compute_weakening_factor(self.material_state.plastic_strain)
            yield_stress = self.yield_stress * weakening_factor
        else:
            yield_stress = self.yield_stress
        
        return von_mises_stress - yield_stress
    
    def compute_plastic_flow_direction(self, stress: np.ndarray) -> np.ndarray:
        """计算塑性流动方向 ∂f/∂σ"""
        deviatoric = self._compute_deviatoric_stress(stress)
        j2 = self._compute_j2_invariant(stress)
        
        # 避免除零
        j2_safe = np.maximum(j2, 1e-12)
        
        # ∂f/∂σ = √(3/2) * s / √(s:s)
        factor = np.sqrt(3.0 / (2.0 * j2_safe))
        
        flow_direction = np.zeros_like(stress)
        for i in range(stress.shape[1]):
            for j in range(stress.shape[2]):
                flow_direction[:, i, j] = factor * deviatoric[:, i, j]
        
        return flow_direction
    
    def compute_consistency_parameter(self, stress: np.ndarray, strain_rate: np.ndarray) -> np.ndarray:
        """计算一致性参数"""
        flow_direction = self.compute_plastic_flow_direction(stress)
        
        # 计算塑性流动方向与应变率的点积
        numerator = np.zeros(stress.shape[0])
        for i in range(stress.shape[1]):
            for j in range(stress.shape[2]):
                numerator += flow_direction[:, i, j] * strain_rate[:, i, j]
        
        # 计算塑性流动方向的自点积
        denominator = np.zeros(stress.shape[0])
        for i in range(stress.shape[1]):
            for j in range(stress.shape[2]):
                denominator += flow_direction[:, i, j] * flow_direction[:, i, j]
        
        # 避免除零
        denominator = np.maximum(denominator, 1e-12)
        
        return numerator / denominator


class DruckerPragerPlasticity(PlasticModel):
    """Drucker-Prager塑性模型
    
    基于Underworld2的DruckerPrager实现：
    f(σ) = αI₁ + √J₂ - k
    其中 I₁ = tr(σ)，J₂ = 1/2 s:s
    """
    
    def __init__(self,
                 cohesion: float,
                 friction_angle: float,
                 cohesion_after_softening: Optional[float] = None,
                 friction_after_softening: Optional[float] = None,
                 softening_start: float = 0.5,
                 softening_end: float = 1.5,
                 dimension: int = 2,
                 name: str = "Drucker-Prager Plasticity"):
        super().__init__(name)
        self.cohesion = cohesion
        self.friction_angle = friction_angle
        self.cohesion_after_softening = cohesion_after_softening
        self.friction_after_softening = friction_after_softening
        self.softening_start = softening_start
        self.softening_end = softening_end
        self.dimension = dimension
        
        # 计算Drucker-Prager参数
        self._compute_drucker_prager_parameters()
        
        # 弱化参数
        self.weakening_enabled = (cohesion_after_softening is not None or 
                                friction_after_softening is not None)
    
    def _compute_drucker_prager_parameters(self):
        """计算Drucker-Prager参数"""
        phi = np.radians(self.friction_angle)
        
        if self.dimension == 2:
            # 2D情况：平面应变
            self.alpha = np.sin(phi) / np.sqrt(3.0 * (3.0 + np.sin(phi)**2))
            self.k = 3.0 * self.cohesion * np.cos(phi) / np.sqrt(3.0 * (3.0 + np.sin(phi)**2))
        else:
            # 3D情况
            self.alpha = 2.0 * np.sin(phi) / (np.sqrt(3.0) * (3.0 - np.sin(phi)))
            self.k = 6.0 * self.cohesion * np.cos(phi) / (np.sqrt(3.0) * (3.0 - np.sin(phi)))
    
    def _compute_weakening_factor(self, plastic_strain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算弱化因子"""
        cohesion_factor = np.ones_like(plastic_strain)
        friction_factor = np.ones_like(plastic_strain)
        
        if not self.weakening_enabled:
            return cohesion_factor, friction_factor
        
        # 线性弱化
        mask1 = plastic_strain >= self.softening_start
        mask2 = plastic_strain <= self.softening_end
        weakening_region = mask1 & mask2
        
        if np.any(weakening_region):
            factor = (plastic_strain[weakening_region] - self.softening_start) / \
                    (self.softening_end - self.softening_start)
            
            # 内聚力弱化
            if self.cohesion_after_softening is not None:
                cohesion_factor[weakening_region] = 1.0 - factor * \
                    (1.0 - self.cohesion_after_softening / self.cohesion)
            
            # 摩擦角弱化
            if self.friction_after_softening is not None:
                friction_factor[weakening_region] = 1.0 - factor * \
                    (1.0 - self.friction_after_softening / self.friction_angle)
        
        # 完全弱化
        mask3 = plastic_strain > self.softening_end
        if np.any(mask3):
            if self.cohesion_after_softening is not None:
                cohesion_factor[mask3] = self.cohesion_after_softening / self.cohesion
            if self.friction_after_softening is not None:
                friction_factor[mask3] = self.friction_after_softening / self.friction_angle
        
        return cohesion_factor, friction_factor
    
    def _compute_deviatoric_stress(self, stress: np.ndarray) -> np.ndarray:
        """计算偏应力张量"""
        if self.dimension == 2:
            mean_stress = 0.5 * (stress[:, 0, 0] + stress[:, 1, 1])
            deviatoric = stress.copy()
            deviatoric[:, 0, 0] -= mean_stress
            deviatoric[:, 1, 1] -= mean_stress
        else:
            mean_stress = (1.0/3.0) * (stress[:, 0, 0] + stress[:, 1, 1] + stress[:, 2, 2])
            deviatoric = stress.copy()
            deviatoric[:, 0, 0] -= mean_stress
            deviatoric[:, 1, 1] -= mean_stress
            deviatoric[:, 2, 2] -= mean_stress
        
        return deviatoric
    
    def _compute_j2_invariant(self, stress: np.ndarray) -> np.ndarray:
        """计算J₂不变量"""
        deviatoric = self._compute_deviatoric_stress(stress)
        
        if self.dimension == 2:
            j2 = 0.5 * (deviatoric[:, 0, 0]**2 + deviatoric[:, 1, 1]**2 + 
                        2 * deviatoric[:, 0, 1]**2)
        else:
            j2 = 0.5 * (deviatoric[:, 0, 0]**2 + deviatoric[:, 1, 1]**2 + deviatoric[:, 2, 2]**2 +
                        2 * (deviatoric[:, 0, 1]**2 + deviatoric[:, 0, 2]**2 + deviatoric[:, 1, 2]**2))
        
        return j2
    
    def _compute_i1_invariant(self, stress: np.ndarray) -> np.ndarray:
        """计算I₁不变量（应力第一不变量）"""
        if self.dimension == 2:
            return stress[:, 0, 0] + stress[:, 1, 1]
        else:
            return stress[:, 0, 0] + stress[:, 1, 1] + stress[:, 2, 2]
    
    def compute_yield_function(self, stress: np.ndarray) -> np.ndarray:
        """计算屈服函数 f(σ) = αI₁ + √J₂ - k"""
        i1 = self._compute_i1_invariant(stress)
        j2 = self._compute_j2_invariant(stress)
        von_mises_stress = np.sqrt(j2)
        
        # 应用弱化
        if self.weakening_enabled and self.material_state is not None:
            cohesion_factor, friction_factor = self._compute_weakening_factor(
                self.material_state.plastic_strain)
            
            # 重新计算参数
            phi = np.radians(self.friction_angle * friction_factor)
            cohesion = self.cohesion * cohesion_factor
            
            if self.dimension == 2:
                alpha = np.sin(phi) / np.sqrt(3.0 * (3.0 + np.sin(phi)**2))
                k = 3.0 * cohesion * np.cos(phi) / np.sqrt(3.0 * (3.0 + np.sin(phi)**2))
            else:
                alpha = 2.0 * np.sin(phi) / (np.sqrt(3.0) * (3.0 - np.sin(phi)))
                k = 6.0 * cohesion * np.cos(phi) / (np.sqrt(3.0) * (3.0 - np.sin(phi)))
        else:
            alpha = self.alpha
            k = self.k
        
        return alpha * i1 + von_mises_stress - k
    
    def compute_plastic_flow_direction(self, stress: np.ndarray) -> np.ndarray:
        """计算塑性流动方向 ∂f/∂σ"""
        deviatoric = self._compute_deviatoric_stress(stress)
        j2 = self._compute_j2_invariant(stress)
        
        # 避免除零
        j2_safe = np.maximum(j2, 1e-12)
        
        # 应用弱化
        if self.weakening_enabled and self.material_state is not None:
            cohesion_factor, friction_factor = self._compute_weakening_factor(
                self.material_state.plastic_strain)
            phi = np.radians(self.friction_angle * friction_factor)
            
            if self.dimension == 2:
                alpha = np.sin(phi) / np.sqrt(3.0 * (3.0 + np.sin(phi)**2))
            else:
                alpha = 2.0 * np.sin(phi) / (np.sqrt(3.0) * (3.0 - np.sin(phi)))
        else:
            alpha = self.alpha
        
        # ∂f/∂σ = αI + s/(2√J₂)
        flow_direction = np.zeros_like(stress)
        
        # 偏应力部分
        factor = 1.0 / (2.0 * np.sqrt(j2_safe))
        for i in range(stress.shape[1]):
            for j in range(stress.shape[2]):
                flow_direction[:, i, j] = factor * deviatoric[:, i, j]
        
        # 静水压力部分
        for i in range(stress.shape[1]):
            flow_direction[:, i, i] += alpha
        
        return flow_direction
    
    def compute_consistency_parameter(self, stress: np.ndarray, strain_rate: np.ndarray) -> np.ndarray:
        """计算一致性参数"""
        flow_direction = self.compute_plastic_flow_direction(stress)
        
        # 计算塑性流动方向与应变率的点积
        numerator = np.zeros(stress.shape[0])
        for i in range(stress.shape[1]):
            for j in range(stress.shape[2]):
                numerator += flow_direction[:, i, j] * strain_rate[:, i, j]
        
        # 计算塑性流动方向的自点积
        denominator = np.zeros(stress.shape[0])
        for i in range(stress.shape[1]):
            for j in range(stress.shape[2]):
                denominator += flow_direction[:, i, j] * flow_direction[:, i, j]
        
        # 避免除零
        denominator = np.maximum(denominator, 1e-12)
        
        return numerator / denominator


class PlasticSolver:
    """塑性求解器"""
    
    def __init__(self, plastic_model: PlasticModel):
        self.plastic_model = plastic_model
    
    def solve_plastic_deformation(self, 
                                stress: np.ndarray,
                                strain_rate: np.ndarray,
                                plastic_strain: np.ndarray,
                                dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """求解塑性变形"""
        
        # 设置材料状态
        state = PlasticState(
            stress=stress,
            strain=np.zeros_like(stress),
            plastic_strain=plastic_strain,
            hardening_variable=np.zeros_like(plastic_strain)
        )
        self.plastic_model.set_material_state(state)
        
        # 计算屈服函数
        yield_function = self.plastic_model.compute_yield_function(stress)
        
        # 检查是否发生塑性变形
        plastic_mask = yield_function > 0
        
        if not np.any(plastic_mask):
            return stress, plastic_strain
        
        # 计算塑性流动方向
        flow_direction = self.plastic_model.compute_plastic_flow_direction(stress)
        
        # 计算一致性参数
        consistency_param = self.plastic_model.compute_consistency_parameter(stress, strain_rate)
        
        # 更新应力（返回映射算法）
        new_stress = stress.copy()
        for i in range(stress.shape[1]):
            for j in range(stress.shape[2]):
                new_stress[:, i, j] -= consistency_param * flow_direction[:, i, j] * dt
        
        # 更新累积塑性应变
        new_plastic_strain = plastic_strain.copy()
        plastic_strain_rate = np.abs(consistency_param)
        new_plastic_strain += plastic_strain_rate * dt
        
        return new_stress, new_plastic_strain


def demo_plastic_models():
    """演示塑性模型功能"""
    print("🔧 塑性模型演示")
    print("=" * 50)
    
    # 创建测试数据
    n_points = 100
    stress = np.zeros((n_points, 2, 2))
    stress[:, 0, 0] = np.linspace(0, 100e6, n_points)  # 轴向应力
    stress[:, 1, 1] = 0.3 * stress[:, 0, 0]  # 侧向应力
    stress[:, 0, 1] = stress[:, 1, 0] = 0.0  # 剪切应力
    
    strain_rate = np.zeros_like(stress)
    strain_rate[:, 0, 0] = 1e-6  # 轴向应变率
    
    plastic_strain = np.zeros(n_points)
    
    # 测试von Mises塑性
    print("\n🔧 测试 von Mises 塑性...")
    von_mises = VonMisesPlasticity(
        yield_stress=50e6,
        yield_stress_after_softening=25e6,
        softening_start=0.1,
        softening_end=0.3
    )
    
    solver = PlasticSolver(von_mises)
    new_stress, new_plastic_strain = solver.solve_plastic_deformation(
        stress, strain_rate, plastic_strain, 1.0)
    
    print(f"   初始应力范围: {stress[:, 0, 0].min():.1e} - {stress[:, 0, 0].max():.1e} Pa")
    print(f"   最终应力范围: {new_stress[:, 0, 0].min():.1e} - {new_stress[:, 0, 0].max():.1e} Pa")
    print(f"   最大塑性应变: {new_plastic_strain.max():.3f}")
    
    # 测试Drucker-Prager塑性
    print("\n🔧 测试 Drucker-Prager 塑性...")
    drucker_prager = DruckerPragerPlasticity(
        cohesion=20e6,
        friction_angle=30.0,
        cohesion_after_softening=10e6,
        friction_after_softening=15.0,
        softening_start=0.1,
        softening_end=0.3
    )
    
    solver = PlasticSolver(drucker_prager)
    new_stress, new_plastic_strain = solver.solve_plastic_deformation(
        stress, strain_rate, plastic_strain, 1.0)
    
    print(f"   初始应力范围: {stress[:, 0, 0].min():.1e} - {stress[:, 0, 0].max():.1e} Pa")
    print(f"   最终应力范围: {new_stress[:, 0, 0].min():.1e} - {new_stress[:, 0, 0].max():.1e} Pa")
    print(f"   最大塑性应变: {new_plastic_strain.max():.3f}")
    
    print("\n✅ 塑性模型演示完成!")


if __name__ == "__main__":
    demo_plastic_models() 