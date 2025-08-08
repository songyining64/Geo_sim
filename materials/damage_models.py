"""
损伤模型
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class DamageState:
    """损伤状态"""
    damage_variable: np.ndarray  # 损伤变量 (0-1)
    equivalent_strain: np.ndarray  # 等效应变
    damage_rate: np.ndarray  # 损伤率
    fracture_energy: np.ndarray  # 断裂能
    time: float = 0.0


class DamageModel(ABC):
    """损伤模型基类"""
    
    def __init__(self, name: str = "Damage Model"):
        self.name = name
        self.damage_state: Optional[DamageState] = None
    
    @abstractmethod
    def compute_damage_evolution(self, strain: np.ndarray, stress: np.ndarray) -> np.ndarray:
        """计算损伤演化"""
        pass
    
    @abstractmethod
    def compute_effective_stiffness(self, damage: np.ndarray) -> np.ndarray:
        """计算有效刚度"""
        pass
    
    @abstractmethod
    def compute_damage_stress(self, stress: np.ndarray, damage: np.ndarray) -> np.ndarray:
        """计算损伤应力"""
        pass
    
    def set_damage_state(self, state: DamageState):
        """设置损伤状态"""
        self.damage_state = state


class IsotropicDamageModel(DamageModel):
    """各向同性损伤模型
    
    基于Underworld2的损伤模型实现：
    使用标量损伤变量描述材料退化
    """
    
    def __init__(self,
                 critical_strain: float = 0.01,
                 damage_exponent: float = 2.0,
                 fracture_energy: float = 100.0,  # J/m²
                 damage_threshold: float = 0.0,
                 name: str = "Isotropic Damage Model"):
        super().__init__(name)
        self.critical_strain = critical_strain
        self.damage_exponent = damage_exponent
        self.fracture_energy = fracture_energy
        self.damage_threshold = damage_threshold
    
    def compute_equivalent_strain(self, strain: np.ndarray) -> np.ndarray:
        """计算等效应变"""
        # von Mises等效应变
        if strain.ndim == 3:  # 3D情况
            # 提取应变分量
            eps_xx = strain[:, 0, 0]
            eps_yy = strain[:, 1, 1]
            eps_zz = strain[:, 2, 2]
            eps_xy = strain[:, 0, 1]
            eps_yz = strain[:, 1, 2]
            eps_xz = strain[:, 0, 2]
            
            # 计算偏应变
            eps_mean = (eps_xx + eps_yy + eps_zz) / 3.0
            eps_dev_xx = eps_xx - eps_mean
            eps_dev_yy = eps_yy - eps_mean
            eps_dev_zz = eps_zz - eps_mean
            
            # von Mises等效应变
            eps_eq = np.sqrt(0.5 * (eps_dev_xx**2 + eps_dev_yy**2 + eps_dev_zz**2 + 
                                   2 * (eps_xy**2 + eps_yz**2 + eps_xz**2)))
        else:  # 2D情况
            eps_xx = strain[:, 0, 0]
            eps_yy = strain[:, 1, 1]
            eps_xy = strain[:, 0, 1]
            
            eps_mean = (eps_xx + eps_yy) / 2.0
            eps_dev_xx = eps_xx - eps_mean
            eps_dev_yy = eps_yy - eps_mean
            
            eps_eq = np.sqrt(0.5 * (eps_dev_xx**2 + eps_dev_yy**2 + 2 * eps_xy**2))
        
        return eps_eq
    
    def compute_damage_evolution(self, strain: np.ndarray, stress: np.ndarray) -> np.ndarray:
        """计算损伤演化"""
        if self.damage_state is None:
            return np.zeros(strain.shape[0])
        
        eps_eq = self.compute_equivalent_strain(strain)
        current_damage = self.damage_state.damage_variable
        
        # 计算损伤演化
        damage_rate = np.zeros_like(eps_eq)
        
        # 只在等效应变超过阈值时演化损伤
        damage_mask = eps_eq > self.damage_threshold
        
        if np.any(damage_mask):
            # 使用指数损伤演化律
            normalized_strain = eps_eq[damage_mask] / self.critical_strain
            damage_rate[damage_mask] = (self.damage_exponent / self.critical_strain) * \
                                     normalized_strain**(self.damage_exponent - 1) * \
                                     (1 - current_damage[damage_mask])
        
        return damage_rate
    
    def compute_effective_stiffness(self, damage: np.ndarray) -> np.ndarray:
        """计算有效刚度"""
        # 使用连续损伤力学模型
        # E_eff = E * (1 - D)
        return 1.0 - damage
    
    def compute_damage_stress(self, stress: np.ndarray, damage: np.ndarray) -> np.ndarray:
        """计算损伤应力"""
        # 有效应力 = 名义应力 / (1 - D)
        effective_stress = np.zeros_like(stress)
        
        for i in range(stress.shape[1]):
            for j in range(stress.shape[2]):
                # 避免除零
                denominator = np.maximum(1.0 - damage, 1e-12)
                effective_stress[:, i, j] = stress[:, i, j] / denominator
        
        return effective_stress


class AnisotropicDamageModel(DamageModel):
    """各向异性损伤模型
    
    考虑损伤的方向性效应
    """
    
    def __init__(self,
                 critical_strain: float = 0.01,
                 damage_exponent: float = 2.0,
                 anisotropy_factor: float = 1.0,
                 name: str = "Anisotropic Damage Model"):
        super().__init__(name)
        self.critical_strain = critical_strain
        self.damage_exponent = damage_exponent
        self.anisotropy_factor = anisotropy_factor
    
    def compute_damage_evolution(self, strain: np.ndarray, stress: np.ndarray) -> np.ndarray:
        """计算各向异性损伤演化"""
        if self.damage_state is None:
            return np.zeros(strain.shape[0])
        
        # 计算主应变和主方向
        if strain.ndim == 3:  # 3D情况
            damage_rate = self._compute_3d_damage_evolution(strain)
        else:  # 2D情况
            damage_rate = self._compute_2d_damage_evolution(strain)
        
        return damage_rate
    
    def _compute_2d_damage_evolution(self, strain: np.ndarray) -> np.ndarray:
        """计算2D损伤演化"""
        n_points = strain.shape[0]
        damage_rate = np.zeros(n_points)
        
        for i in range(n_points):
            # 构建应变矩阵
            eps_matrix = np.array([
                [strain[i, 0, 0], strain[i, 0, 1]],
                [strain[i, 1, 0], strain[i, 1, 1]]
            ])
            
            # 计算特征值和特征向量
            eigenvals, eigenvecs = np.linalg.eigh(eps_matrix)
            
            # 计算主应变方向的损伤率
            max_strain = np.max(np.abs(eigenvals))
            if max_strain > self.critical_strain:
                normalized_strain = max_strain / self.critical_strain
                damage_rate[i] = (self.damage_exponent / self.critical_strain) * \
                               normalized_strain**(self.damage_exponent - 1)
        
        return damage_rate
    
    def _compute_3d_damage_evolution(self, strain: np.ndarray) -> np.ndarray:
        """计算3D损伤演化"""
        n_points = strain.shape[0]
        damage_rate = np.zeros(n_points)
        
        for i in range(n_points):
            # 构建应变矩阵
            eps_matrix = np.array([
                [strain[i, 0, 0], strain[i, 0, 1], strain[i, 0, 2]],
                [strain[i, 1, 0], strain[i, 1, 1], strain[i, 1, 2]],
                [strain[i, 2, 0], strain[i, 2, 1], strain[i, 2, 2]]
            ])
            
            # 计算特征值和特征向量
            eigenvals, eigenvecs = np.linalg.eigh(eps_matrix)
            
            # 计算主应变方向的损伤率
            max_strain = np.max(np.abs(eigenvals))
            if max_strain > self.critical_strain:
                normalized_strain = max_strain / self.critical_strain
                damage_rate[i] = (self.damage_exponent / self.critical_strain) * \
                               normalized_strain**(self.damage_exponent - 1)
        
        return damage_rate
    
    def compute_effective_stiffness(self, damage: np.ndarray) -> np.ndarray:
        """计算有效刚度（各向异性）"""
        # 简化的各向异性有效刚度
        return 1.0 - damage * self.anisotropy_factor
    
    def compute_damage_stress(self, stress: np.ndarray, damage: np.ndarray) -> np.ndarray:
        """计算损伤应力（各向异性）"""
        effective_stress = np.zeros_like(stress)
        
        for i in range(stress.shape[1]):
            for j in range(stress.shape[2]):
                denominator = np.maximum(1.0 - damage * self.anisotropy_factor, 1e-12)
                effective_stress[:, i, j] = stress[:, i, j] / denominator
        
        return effective_stress


class DamagePlasticityCoupling(DamageModel):
    """损伤-塑性耦合模型
    
    考虑损伤与塑性的相互作用
    """
    
    def __init__(self,
                 critical_strain: float = 0.01,
                 damage_exponent: float = 2.0,
                 plastic_coupling_factor: float = 1.0,
                 name: str = "Damage-Plasticity Coupling"):
        super().__init__(name)
        self.critical_strain = critical_strain
        self.damage_exponent = damage_exponent
        self.plastic_coupling_factor = plastic_coupling_factor
    
    def compute_damage_evolution(self, strain: np.ndarray, stress: np.ndarray) -> np.ndarray:
        """计算损伤-塑性耦合演化"""
        if self.damage_state is None:
            return np.zeros(strain.shape[0])
        
        # 计算等效应变
        eps_eq = self._compute_equivalent_strain(strain)
        
        # 计算塑性应变（简化）
        plastic_strain = np.maximum(eps_eq - self.critical_strain, 0.0)
        
        # 损伤演化（考虑塑性耦合）
        damage_rate = np.zeros_like(eps_eq)
        damage_mask = eps_eq > self.critical_strain
        
        if np.any(damage_mask):
            normalized_strain = eps_eq[damage_mask] / self.critical_strain
            plastic_factor = 1.0 + self.plastic_coupling_factor * plastic_strain[damage_mask]
            
            damage_rate[damage_mask] = (self.damage_exponent / self.critical_strain) * \
                                     normalized_strain**(self.damage_exponent - 1) * \
                                     plastic_factor
        
        return damage_rate
    
    def _compute_equivalent_strain(self, strain: np.ndarray) -> np.ndarray:
        """计算等效应变"""
        if strain.ndim == 3:  # 3D情况
            eps_xx = strain[:, 0, 0]
            eps_yy = strain[:, 1, 1]
            eps_zz = strain[:, 2, 2]
            eps_xy = strain[:, 0, 1]
            eps_yz = strain[:, 1, 2]
            eps_xz = strain[:, 0, 2]
            
            eps_mean = (eps_xx + eps_yy + eps_zz) / 3.0
            eps_dev_xx = eps_xx - eps_mean
            eps_dev_yy = eps_yy - eps_mean
            eps_dev_zz = eps_zz - eps_mean
            
            eps_eq = np.sqrt(0.5 * (eps_dev_xx**2 + eps_dev_yy**2 + eps_dev_zz**2 + 
                                   2 * (eps_xy**2 + eps_yz**2 + eps_xz**2)))
        else:  # 2D情况
            eps_xx = strain[:, 0, 0]
            eps_yy = strain[:, 1, 1]
            eps_xy = strain[:, 0, 1]
            
            eps_mean = (eps_xx + eps_yy) / 2.0
            eps_dev_xx = eps_xx - eps_mean
            eps_dev_yy = eps_yy - eps_mean
            
            eps_eq = np.sqrt(0.5 * (eps_dev_xx**2 + eps_dev_yy**2 + 2 * eps_xy**2))
        
        return eps_eq
    
    def compute_effective_stiffness(self, damage: np.ndarray) -> np.ndarray:
        """计算有效刚度"""
        return 1.0 - damage
    
    def compute_damage_stress(self, stress: np.ndarray, damage: np.ndarray) -> np.ndarray:
        """计算损伤应力"""
        effective_stress = np.zeros_like(stress)
        
        for i in range(stress.shape[1]):
            for j in range(stress.shape[2]):
                denominator = np.maximum(1.0 - damage, 1e-12)
                effective_stress[:, i, j] = stress[:, i, j] / denominator
        
        return effective_stress


class DamageSolver:
    """损伤求解器"""
    
    def __init__(self, damage_model: DamageModel):
        self.damage_model = damage_model
    
    def solve_damage_evolution(self,
                             strain: np.ndarray,
                             stress: np.ndarray,
                             damage: np.ndarray,
                             dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """求解损伤演化"""
        
        # 设置损伤状态
        state = DamageState(
            damage_variable=damage,
            equivalent_strain=np.zeros_like(damage),
            damage_rate=np.zeros_like(damage),
            fracture_energy=np.zeros_like(damage)
        )
        self.damage_model.set_damage_state(state)
        
        # 计算损伤演化率
        damage_rate = self.damage_model.compute_damage_evolution(strain, stress)
        
        # 更新损伤变量
        new_damage = damage + damage_rate * dt
        
        # 限制损伤变量在[0, 1]范围内
        new_damage = np.clip(new_damage, 0.0, 1.0)
        
        return new_damage, damage_rate


def create_isotropic_damage_model(critical_strain: float = 0.01,
                                damage_exponent: float = 2.0,
                                fracture_energy: float = 100.0) -> IsotropicDamageModel:
    """创建各向同性损伤模型"""
    return IsotropicDamageModel(critical_strain, damage_exponent, fracture_energy)


def create_anisotropic_damage_model(critical_strain: float = 0.01,
                                  damage_exponent: float = 2.0,
                                  anisotropy_factor: float = 1.0) -> AnisotropicDamageModel:
    """创建各向异性损伤模型"""
    return AnisotropicDamageModel(critical_strain, damage_exponent, anisotropy_factor)


def create_damage_plasticity_coupling(critical_strain: float = 0.01,
                                    damage_exponent: float = 2.0,
                                    plastic_coupling_factor: float = 1.0) -> DamagePlasticityCoupling:
    """创建损伤-塑性耦合模型"""
    return DamagePlasticityCoupling(critical_strain, damage_exponent, plastic_coupling_factor)


def demo_damage_models():
    """演示损伤模型功能"""
    print("🔧 损伤模型演示")
    print("=" * 50)
    
    # 创建测试数据
    n_points = 100
    strain = np.zeros((n_points, 2, 2))
    strain[:, 0, 0] = np.linspace(0, 0.02, n_points)  # 轴向应变
    strain[:, 1, 1] = -0.3 * strain[:, 0, 0]  # 侧向应变
    
    stress = np.zeros_like(strain)
    stress[:, 0, 0] = 70e9 * strain[:, 0, 0]  # 轴向应力
    stress[:, 1, 1] = 70e9 * strain[:, 1, 1]  # 侧向应力
    
    damage = np.zeros(n_points)
    
    # 测试各向同性损伤模型
    print("\n🔧 测试各向同性损伤模型...")
    isotropic_damage = IsotropicDamageModel(
        critical_strain=0.01,
        damage_exponent=2.0,
        fracture_energy=100.0
    )
    
    solver = DamageSolver(isotropic_damage)
    new_damage, damage_rate = solver.solve_damage_evolution(
        strain, stress, damage, 1.0)
    
    print(f"   应变范围: {strain[:, 0, 0].min():.3f} - {strain[:, 0, 0].max():.3f}")
    print(f"   损伤范围: {new_damage.min():.3f} - {new_damage.max():.3f}")
    print(f"   最大损伤率: {damage_rate.max():.3f}")
    
    # 测试各向异性损伤模型
    print("\n🔧 测试各向异性损伤模型...")
    anisotropic_damage = AnisotropicDamageModel(
        critical_strain=0.01,
        damage_exponent=2.0,
        anisotropy_factor=1.5
    )
    
    solver = DamageSolver(anisotropic_damage)
    new_damage, damage_rate = solver.solve_damage_evolution(
        strain, stress, damage, 1.0)
    
    print(f"   应变范围: {strain[:, 0, 0].min():.3f} - {strain[:, 0, 0].max():.3f}")
    print(f"   损伤范围: {new_damage.min():.3f} - {new_damage.max():.3f}")
    print(f"   最大损伤率: {damage_rate.max():.3f}")
    
    # 测试损伤-塑性耦合模型
    print("\n🔧 测试损伤-塑性耦合模型...")
    damage_plasticity = DamagePlasticityCoupling(
        critical_strain=0.01,
        damage_exponent=2.0,
        plastic_coupling_factor=1.0
    )
    
    solver = DamageSolver(damage_plasticity)
    new_damage, damage_rate = solver.solve_damage_evolution(
        strain, stress, damage, 1.0)
    
    print(f"   应变范围: {strain[:, 0, 0].min():.3f} - {strain[:, 0, 0].max():.3f}")
    print(f"   损伤范围: {new_damage.min():.3f} - {new_damage.max():.3f}")
    print(f"   最大损伤率: {damage_rate.max():.3f}")
    
    print("\n✅ 损伤模型演示完成!")


if __name__ == "__main__":
    demo_damage_models()
