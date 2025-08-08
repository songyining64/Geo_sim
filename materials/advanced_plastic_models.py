"""
高级塑性模型实现
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PlasticState:
    """塑性状态类"""
    stress: np.ndarray  # 应力张量
    plastic_strain: np.ndarray  # 塑性应变
    hardening_variable: float  # 硬化变量
    damage_variable: float = 0.0  # 损伤变量
    
    def __post_init__(self):
        if self.stress.ndim == 1:
            self.stress = self.stress.reshape(3, 3)
        if self.plastic_strain.ndim == 1:
            self.plastic_strain = self.plastic_strain.reshape(3, 3)


class BasePlasticModel(ABC):
    """塑性模型基类"""
    
    def __init__(self, youngs_modulus: float, poissons_ratio: float):
        self.E = youngs_modulus
        self.nu = poissons_ratio
        self.G = youngs_modulus / (2 * (1 + poissons_ratio))
        self.K = youngs_modulus / (3 * (1 - 2 * poissons_ratio))
    
    @abstractmethod
    def compute_yield_function(self, stress: np.ndarray, state: PlasticState) -> float:
        """计算屈服函数"""
        pass
    
    @abstractmethod
    def compute_plastic_strain_increment(self, stress: np.ndarray, 
                                       strain_increment: np.ndarray,
                                       state: PlasticState) -> Tuple[np.ndarray, float]:
        """计算塑性应变增量"""
        pass
    
    @abstractmethod
    def update_stress(self, stress: np.ndarray, strain_increment: np.ndarray,
                     state: PlasticState) -> Tuple[np.ndarray, PlasticState]:
        """更新应力状态"""
        pass


class VonMisesPlasticity(BasePlasticModel):
    """von Mises塑性模型"""
    
    def __init__(self, youngs_modulus: float, poissons_ratio: float,
                 yield_stress: float, hardening_modulus: float = 0.0):
        super().__init__(youngs_modulus, poissons_ratio)
        self.sigma_y = yield_stress
        self.H = hardening_modulus
    
    def compute_deviatoric_stress(self, stress: np.ndarray) -> np.ndarray:
        """计算偏应力"""
        mean_stress = np.trace(stress) / 3.0
        return stress - mean_stress * np.eye(3)
    
    def compute_von_mises_stress(self, stress: np.ndarray) -> float:
        """计算von Mises应力"""
        s = self.compute_deviatoric_stress(stress)
        return np.sqrt(1.5 * np.sum(s * s))
    
    def compute_yield_function(self, stress: np.ndarray, state: PlasticState) -> float:
        """计算von Mises屈服函数"""
        sigma_vm = self.compute_von_mises_stress(stress)
        sigma_y_eff = self.sigma_y + self.H * state.hardening_variable
        return sigma_vm - sigma_y_eff
    
    def compute_plastic_strain_increment(self, stress: np.ndarray,
                                       strain_increment: np.ndarray,
                                       state: PlasticState) -> Tuple[np.ndarray, float]:
        """计算塑性应变增量（返回映射算法）"""
        # 弹性预测
        elastic_strain_increment = strain_increment
        stress_trial = stress + 2 * self.G * elastic_strain_increment
        
        # 检查屈服
        f_trial = self.compute_yield_function(stress_trial, state)
        
        if f_trial <= 0:
            # 弹性响应
            return np.zeros_like(strain_increment), 0.0
        
        # 塑性响应 - 返回映射
        s_trial = self.compute_deviatoric_stress(stress_trial)
        norm_s_trial = np.sqrt(np.sum(s_trial * s_trial))
        
        if norm_s_trial < 1e-12:
            return np.zeros_like(strain_increment), 0.0
        
        # 计算塑性乘子
        delta_gamma = f_trial / (3 * self.G + self.H)
        
        # 计算塑性应变增量
        plastic_strain_increment = delta_gamma * s_trial / norm_s_trial
        
        return plastic_strain_increment, delta_gamma
    
    def update_stress(self, stress: np.ndarray, strain_increment: np.ndarray,
                     state: PlasticState) -> Tuple[np.ndarray, PlasticState]:
        """更新应力状态"""
        # 计算塑性应变增量
        plastic_strain_increment, delta_gamma = self.compute_plastic_strain_increment(
            stress, strain_increment, state
        )
        
        # 更新应力
        elastic_strain_increment = strain_increment - plastic_strain_increment
        stress_new = stress + 2 * self.G * elastic_strain_increment
        
        # 更新状态
        state_new = PlasticState(
            stress=stress_new,
            plastic_strain=state.plastic_strain + plastic_strain_increment,
            hardening_variable=state.hardening_variable + delta_gamma,
            damage_variable=state.damage_variable
        )
        
        return stress_new, state_new


class DruckerPragerPlasticity(BasePlasticModel):
    """Drucker-Prager塑性模型"""
    
    def __init__(self, youngs_modulus: float, poissons_ratio: float,
                 cohesion: float, friction_angle: float, hardening_modulus: float = 0.0):
        super().__init__(youngs_modulus, poissons_ratio)
        self.c = cohesion
        self.phi = friction_angle
        self.H = hardening_modulus
        
        # 计算Drucker-Prager参数
        self.alpha = 2 * np.sin(self.phi) / (np.sqrt(3) * (3 - np.sin(self.phi)))
        self.k = 6 * self.c * np.cos(self.phi) / (np.sqrt(3) * (3 - np.sin(self.phi)))
    
    def compute_mean_stress(self, stress: np.ndarray) -> float:
        """计算平均应力"""
        return np.trace(stress) / 3.0
    
    def compute_yield_function(self, stress: np.ndarray, state: PlasticState) -> float:
        """计算Drucker-Prager屈服函数"""
        p = self.compute_mean_stress(stress)
        s = self.compute_deviatoric_stress(stress)
        q = np.sqrt(1.5 * np.sum(s * s))
        
        sigma_y_eff = self.k + self.H * state.hardening_variable
        return q + self.alpha * p - sigma_y_eff
    
    def compute_plastic_strain_increment(self, stress: np.ndarray,
                                       strain_increment: np.ndarray,
                                       state: PlasticState) -> Tuple[np.ndarray, float]:
        """计算塑性应变增量"""
        # 弹性预测
        elastic_strain_increment = strain_increment
        stress_trial = stress + 2 * self.G * elastic_strain_increment
        
        # 检查屈服
        f_trial = self.compute_yield_function(stress_trial, state)
        
        if f_trial <= 0:
            return np.zeros_like(strain_increment), 0.0
        
        # 塑性响应
        p_trial = self.compute_mean_stress(stress_trial)
        s_trial = self.compute_deviatoric_stress(stress_trial)
        q_trial = np.sqrt(1.5 * np.sum(s_trial * s_trial))
        
        if q_trial < 1e-12:
            return np.zeros_like(strain_increment), 0.0
        
        # 计算塑性乘子
        K_eff = self.K + self.alpha * self.alpha * self.H
        G_eff = self.G + 0.5 * self.H
        
        delta_gamma = f_trial / (3 * G_eff + self.alpha * self.alpha * K_eff)
        
        # 计算塑性应变增量
        deviatoric_part = delta_gamma * s_trial / q_trial
        volumetric_part = delta_gamma * self.alpha / 3.0
        
        plastic_strain_increment = deviatoric_part + volumetric_part * np.eye(3)
        
        return plastic_strain_increment, delta_gamma
    
    def update_stress(self, stress: np.ndarray, strain_increment: np.ndarray,
                     state: PlasticState) -> Tuple[np.ndarray, PlasticState]:
        """更新应力状态"""
        # 计算塑性应变增量
        plastic_strain_increment, delta_gamma = self.compute_plastic_strain_increment(
            stress, strain_increment, state
        )
        
        # 更新应力
        elastic_strain_increment = strain_increment - plastic_strain_increment
        stress_new = stress + 2 * self.G * elastic_strain_increment + self.K * np.trace(elastic_strain_increment) * np.eye(3)
        
        # 更新状态
        state_new = PlasticState(
            stress=stress_new,
            plastic_strain=state.plastic_strain + plastic_strain_increment,
            hardening_variable=state.hardening_variable + delta_gamma,
            damage_variable=state.damage_variable
        )
        
        return stress_new, state_new


class PlasticSolver:
    """塑性求解器"""
    
    def __init__(self, plastic_model: BasePlasticModel):
        self.plastic_model = plastic_model
        self.convergence_tolerance = 1e-8
        self.max_iterations = 50
    
    def solve(self, stress: np.ndarray, strain_increment: np.ndarray,
              state: PlasticState) -> Tuple[np.ndarray, PlasticState]:
        """求解塑性问题"""
        # 使用返回映射算法
        stress_new, state_new = self.plastic_model.update_stress(
            stress, strain_increment, state
        )
        
        return stress_new, state_new
    
    def solve_with_newton(self, stress: np.ndarray, strain_increment: np.ndarray,
                         state: PlasticState) -> Tuple[np.ndarray, PlasticState]:
        """使用Newton迭代求解塑性问题"""
        # 实现Newton迭代算法
        # 这里可以添加更复杂的迭代求解器
        return self.solve(stress, strain_increment, state)


# 工厂函数
def create_von_mises_plasticity(youngs_modulus: float, poissons_ratio: float,
                               yield_stress: float, hardening_modulus: float = 0.0) -> VonMisesPlasticity:
    """创建von Mises塑性模型"""
    return VonMisesPlasticity(youngs_modulus, poissons_ratio, yield_stress, hardening_modulus)


def create_drucker_prager_plasticity(youngs_modulus: float, poissons_ratio: float,
                                   cohesion: float, friction_angle: float,
                                   hardening_modulus: float = 0.0) -> DruckerPragerPlasticity:
    """创建Drucker-Prager塑性模型"""
    return DruckerPragerPlasticity(youngs_modulus, poissons_ratio, cohesion, friction_angle, hardening_modulus)
