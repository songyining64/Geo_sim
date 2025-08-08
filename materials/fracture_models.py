"""
断裂模拟模型
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class FractureState:
    """断裂状态"""
    crack_length: np.ndarray  # 裂纹长度
    crack_direction: np.ndarray  # 裂纹方向
    stress_intensity_factor: np.ndarray  # 应力强度因子
    fracture_energy: np.ndarray  # 断裂能
    damage_variable: np.ndarray  # 损伤变量
    time: float = 0.0


class FractureCriterion(ABC):
    """断裂准则基类"""
    
    def __init__(self, name: str = "Fracture Criterion"):
        self.name = name
    
    @abstractmethod
    def compute_fracture_criterion(self, stress: np.ndarray, strain: np.ndarray) -> np.ndarray:
        """计算断裂准则"""
        pass
    
    @abstractmethod
    def check_fracture_initiation(self, criterion_value: np.ndarray, threshold: float) -> np.ndarray:
        """检查断裂起始"""
        pass


class MaximumPrincipalStressCriterion(FractureCriterion):
    """最大主应力断裂准则
    
    基于Underworld2的最大主应力准则：
    σ_max ≥ σ_critical
    """
    
    def __init__(self, critical_stress: float = 100e6, name: str = "Maximum Principal Stress Criterion"):
        super().__init__(name)
        self.critical_stress = critical_stress
    
    def compute_principal_stresses(self, stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算主应力"""
        if stress.ndim == 3:  # 3D情况
            # 检查是否为3D应力张量
            if stress.shape[1] == 3 and stress.shape[2] == 3:
                # 提取应力分量
                sigma_xx = stress[:, 0, 0]
                sigma_yy = stress[:, 1, 1]
                sigma_zz = stress[:, 2, 2]
                sigma_xy = stress[:, 0, 1]
                sigma_yz = stress[:, 1, 2]
                sigma_xz = stress[:, 0, 2]
                
                # 计算主应力（简化版本）
                # 对于3D情况，使用数值方法计算特征值
                principal_stresses = np.zeros((stress.shape[0], 3))
                for i in range(stress.shape[0]):
                    stress_matrix = np.array([
                        [sigma_xx[i], sigma_xy[i], sigma_xz[i]],
                        [sigma_xy[i], sigma_yy[i], sigma_yz[i]],
                        [sigma_xz[i], sigma_yz[i], sigma_zz[i]]
                    ])
                    eigenvalues = np.linalg.eigvals(stress_matrix)
                    principal_stresses[i] = np.sort(eigenvalues)[::-1]  # 降序排列
                
                return principal_stresses[:, 0], principal_stresses[:, 1], principal_stresses[:, 2]
            else:
                # 2D情况但以3D数组形式存储
                sigma_xx = stress[:, 0, 0]
                sigma_yy = stress[:, 1, 1]
                sigma_xy = stress[:, 0, 1]
                
                # 计算主应力
                sigma_mean = (sigma_xx + sigma_yy) / 2.0
                sigma_diff = (sigma_xx - sigma_yy) / 2.0
                tau_max = np.sqrt(sigma_diff**2 + sigma_xy**2)
                
                sigma_1 = sigma_mean + tau_max
                sigma_2 = sigma_mean - tau_max
                sigma_3 = np.zeros_like(sigma_1)
                
                return sigma_1, sigma_2, sigma_3
        else:  # 2D情况
            # 2D主应力计算
            sigma_xx = stress[:, 0, 0]
            sigma_yy = stress[:, 1, 1]
            sigma_xy = stress[:, 0, 1]
            
            # 计算主应力
            sigma_mean = (sigma_xx + sigma_yy) / 2.0
            sigma_diff = (sigma_xx - sigma_yy) / 2.0
            tau_max = np.sqrt(sigma_diff**2 + sigma_xy**2)
            
            sigma_1 = sigma_mean + tau_max
            sigma_2 = sigma_mean - tau_max
            sigma_3 = np.zeros_like(sigma_1)
            
            return sigma_1, sigma_2, sigma_3
    
    def compute_fracture_criterion(self, stress: np.ndarray, strain: np.ndarray) -> np.ndarray:
        """计算断裂准则"""
        principal_stresses = self.compute_principal_stresses(stress)
        max_principal_stress = principal_stresses[0]
        
        return max_principal_stress
    
    def check_fracture_initiation(self, criterion_value: np.ndarray, threshold: float) -> np.ndarray:
        """检查断裂起始"""
        return criterion_value >= threshold


class MaximumPrincipalStrainCriterion(FractureCriterion):
    """最大主应变断裂准则
    
    基于Underworld2的最大主应变准则：
    ε_max ≥ ε_critical
    """
    
    def __init__(self, critical_strain: float = 0.01, name: str = "Maximum Principal Strain Criterion"):
        super().__init__(name)
        self.critical_strain = critical_strain
    
    def compute_principal_strains(self, strain: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算主应变"""
        if strain.ndim == 3:  # 3D情况
            # 检查是否为3D应变张量
            if strain.shape[1] == 3 and strain.shape[2] == 3:
                # 提取应变分量
                eps_xx = strain[:, 0, 0]
                eps_yy = strain[:, 1, 1]
                eps_zz = strain[:, 2, 2]
                eps_xy = strain[:, 0, 1]
                eps_yz = strain[:, 1, 2]
                eps_xz = strain[:, 0, 2]
                
                # 计算主应变（简化版本）
                principal_strains = np.zeros((strain.shape[0], 3))
                for i in range(strain.shape[0]):
                    strain_matrix = np.array([
                        [eps_xx[i], eps_xy[i], eps_xz[i]],
                        [eps_xy[i], eps_yy[i], eps_yz[i]],
                        [eps_xz[i], eps_yz[i], eps_zz[i]]
                    ])
                    eigenvalues = np.linalg.eigvals(strain_matrix)
                    principal_strains[i] = np.sort(eigenvalues)[::-1]  # 降序排列
                
                return principal_strains[:, 0], principal_strains[:, 1], principal_strains[:, 2]
            else:
                # 2D情况但以3D数组形式存储
                eps_xx = strain[:, 0, 0]
                eps_yy = strain[:, 1, 1]
                eps_xy = strain[:, 0, 1]
                
                # 计算主应变
                eps_mean = (eps_xx + eps_yy) / 2.0
                eps_diff = (eps_xx - eps_yy) / 2.0
                gamma_max = np.sqrt(eps_diff**2 + eps_xy**2)
                
                eps_1 = eps_mean + gamma_max
                eps_2 = eps_mean - gamma_max
                eps_3 = np.zeros_like(eps_1)
                
                return eps_1, eps_2, eps_3
        else:  # 2D情况
            # 2D主应变计算
            eps_xx = strain[:, 0, 0]
            eps_yy = strain[:, 1, 1]
            eps_xy = strain[:, 0, 1]
            
            # 计算主应变
            eps_mean = (eps_xx + eps_yy) / 2.0
            eps_diff = (eps_xx - eps_yy) / 2.0
            gamma_max = np.sqrt(eps_diff**2 + eps_xy**2)
            
            eps_1 = eps_mean + gamma_max
            eps_2 = eps_mean - gamma_max
            eps_3 = np.zeros_like(eps_1)
            
            return eps_1, eps_2, eps_3
    
    def compute_fracture_criterion(self, stress: np.ndarray, strain: np.ndarray) -> np.ndarray:
        """计算断裂准则"""
        principal_strains = self.compute_principal_strains(strain)
        max_principal_strain = principal_strains[0]
        
        return max_principal_strain
    
    def check_fracture_initiation(self, criterion_value: np.ndarray, threshold: float) -> np.ndarray:
        """检查断裂起始"""
        return criterion_value >= threshold


class EnergyReleaseRateCriterion(FractureCriterion):
    """能量释放率断裂准则
    
    基于Underworld2的能量释放率准则：
    G ≥ G_critical
    """
    
    def __init__(self, critical_energy_release_rate: float = 100.0, name: str = "Energy Release Rate Criterion"):
        super().__init__(name)
        self.critical_energy_release_rate = critical_energy_release_rate
    
    def compute_energy_release_rate(self, stress: np.ndarray, strain: np.ndarray, 
                                  crack_length: np.ndarray) -> np.ndarray:
        """计算能量释放率"""
        # 简化的能量释放率计算
        # G = π * σ² * a / E
        # 其中 σ 为远场应力，a 为裂纹长度，E 为弹性模量
        
        # 计算等效应力
        if stress.ndim == 3:  # 3D情况
            if stress.shape[1] == 3 and stress.shape[2] == 3:
                # 3D von Mises应力
                von_mises_stress = np.sqrt(0.5 * (
                    (stress[:, 0, 0] - stress[:, 1, 1])**2 + 
                    (stress[:, 1, 1] - stress[:, 2, 2])**2 + 
                    (stress[:, 2, 2] - stress[:, 0, 0])**2 + 
                    6 * (stress[:, 0, 1]**2 + stress[:, 1, 2]**2 + stress[:, 0, 2]**2)
                ))
            else:
                # 2D情况但以3D数组形式存储
                von_mises_stress = np.sqrt(
                    stress[:, 0, 0]**2 + stress[:, 1, 1]**2 - 
                    stress[:, 0, 0] * stress[:, 1, 1] + 3 * stress[:, 0, 1]**2
                )
        else:  # 2D情况
            von_mises_stress = np.sqrt(
                stress[:, 0, 0]**2 + stress[:, 1, 1]**2 - 
                stress[:, 0, 0] * stress[:, 1, 1] + 3 * stress[:, 0, 1]**2
            )
        
        # 假设弹性模量
        E = 200e9  # Pa
        
        # 计算能量释放率
        energy_release_rate = np.pi * von_mises_stress**2 * crack_length / E
        
        return energy_release_rate
    
    def compute_fracture_criterion(self, stress: np.ndarray, strain: np.ndarray) -> np.ndarray:
        """计算断裂准则"""
        # 假设裂纹长度（实际应用中应该从状态中获取）
        crack_length = np.ones(stress.shape[0]) * 1e-3  # 1mm
        
        energy_release_rate = self.compute_energy_release_rate(stress, strain, crack_length)
        
        return energy_release_rate
    
    def check_fracture_initiation(self, criterion_value: np.ndarray, threshold: float) -> np.ndarray:
        """检查断裂起始"""
        return criterion_value >= threshold


class CrackPropagationAlgorithm:
    """裂纹扩展算法"""
    
    def __init__(self, fracture_criterion: FractureCriterion):
        self.fracture_criterion = fracture_criterion
        self.fracture_state: Optional[FractureState] = None
    
    def compute_stress_intensity_factor(self, stress: np.ndarray, crack_length: np.ndarray) -> np.ndarray:
        """计算应力强度因子"""
        # 简化的应力强度因子计算
        # K_I = σ * √(π * a)
        # 其中 σ 为远场应力，a 为裂纹长度
        
        # 计算等效应力
        if stress.ndim == 3:  # 3D情况
            if stress.shape[1] == 3 and stress.shape[2] == 3:
                # 3D von Mises应力
                von_mises_stress = np.sqrt(0.5 * (
                    (stress[:, 0, 0] - stress[:, 1, 1])**2 + 
                    (stress[:, 1, 1] - stress[:, 2, 2])**2 + 
                    (stress[:, 2, 2] - stress[:, 0, 0])**2 + 
                    6 * (stress[:, 0, 1]**2 + stress[:, 1, 2]**2 + stress[:, 0, 2]**2)
                ))
            else:
                # 2D情况但以3D数组形式存储
                von_mises_stress = np.sqrt(
                    stress[:, 0, 0]**2 + stress[:, 1, 1]**2 - 
                    stress[:, 0, 0] * stress[:, 1, 1] + 3 * stress[:, 0, 1]**2
                )
        else:  # 2D情况
            von_mises_stress = np.sqrt(
                stress[:, 0, 0]**2 + stress[:, 1, 1]**2 - 
                stress[:, 0, 0] * stress[:, 1, 1] + 3 * stress[:, 0, 1]**2
            )
        
        # 计算应力强度因子
        stress_intensity_factor = von_mises_stress * np.sqrt(np.pi * crack_length)
        
        return stress_intensity_factor
    
    def compute_crack_growth_rate(self, stress_intensity_factor: np.ndarray, 
                                fracture_toughness: float) -> np.ndarray:
        """计算裂纹扩展速率"""
        # Paris定律：da/dN = C * (ΔK)^m
        # 其中 C 和 m 为材料参数，ΔK 为应力强度因子范围
        
        # 简化的裂纹扩展速率计算
        C = 1e-12  # 材料参数
        m = 3.0    # 材料参数
        
        # 计算应力强度因子范围（简化）
        delta_k = stress_intensity_factor - fracture_toughness
        delta_k = np.maximum(delta_k, 0.0)
        
        # 计算裂纹扩展速率
        crack_growth_rate = C * (delta_k ** m)
        
        return crack_growth_rate
    
    def update_crack_length(self, crack_length: np.ndarray, crack_growth_rate: np.ndarray, 
                          dt: float) -> np.ndarray:
        """更新裂纹长度"""
        new_crack_length = crack_length + crack_growth_rate * dt
        
        return new_crack_length
    
    def compute_crack_direction(self, stress: np.ndarray) -> np.ndarray:
        """计算裂纹扩展方向"""
        # 简化的裂纹扩展方向计算
        # 假设裂纹沿最大主应力方向扩展
        
        if stress.ndim == 3:  # 3D情况
            if stress.shape[1] == 3 and stress.shape[2] == 3:
                # 3D主应力方向
                crack_direction = np.zeros((stress.shape[0], 3))
                for i in range(stress.shape[0]):
                    stress_matrix = np.array([
                        [stress[i, 0, 0], stress[i, 0, 1], stress[i, 0, 2]],
                        [stress[i, 0, 1], stress[i, 1, 1], stress[i, 1, 2]],
                        [stress[i, 0, 2], stress[i, 1, 2], stress[i, 2, 2]]
                    ])
                    eigenvalues, eigenvectors = np.linalg.eig(stress_matrix)
                    max_eigenvalue_index = np.argmax(eigenvalues)
                    crack_direction[i] = eigenvectors[:, max_eigenvalue_index]
            else:
                # 2D情况但以3D数组形式存储
                crack_direction = np.zeros((stress.shape[0], 2))
                for i in range(stress.shape[0]):
                    stress_matrix = np.array([
                        [stress[i, 0, 0], stress[i, 0, 1]],
                        [stress[i, 0, 1], stress[i, 1, 1]]
                    ])
                    eigenvalues, eigenvectors = np.linalg.eig(stress_matrix)
                    max_eigenvalue_index = np.argmax(eigenvalues)
                    crack_direction[i] = eigenvectors[:, max_eigenvalue_index]
        else:  # 2D情况
            # 2D裂纹扩展方向
            crack_direction = np.zeros((stress.shape[0], 2))
            for i in range(stress.shape[0]):
                stress_matrix = np.array([
                    [stress[i, 0, 0], stress[i, 0, 1]],
                    [stress[i, 0, 1], stress[i, 1, 1]]
                ])
                eigenvalues, eigenvectors = np.linalg.eig(stress_matrix)
                max_eigenvalue_index = np.argmax(eigenvalues)
                crack_direction[i] = eigenvectors[:, max_eigenvalue_index]
        
        return crack_direction


class FractureSolver:
    """断裂求解器"""
    
    def __init__(self, fracture_criterion: FractureCriterion, 
                 crack_propagation_algorithm: CrackPropagationAlgorithm):
        self.fracture_criterion = fracture_criterion
        self.crack_propagation_algorithm = crack_propagation_algorithm
    
    def solve_fracture(self,
                      stress: np.ndarray,
                      strain: np.ndarray,
                      crack_length: np.ndarray,
                      fracture_toughness: float,
                      dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """求解断裂"""
        
        # 计算断裂准则
        criterion_value = self.fracture_criterion.compute_fracture_criterion(stress, strain)
        
        # 检查断裂起始
        fracture_initiated = self.fracture_criterion.check_fracture_initiation(
            criterion_value, fracture_toughness)
        
        # 计算应力强度因子
        stress_intensity_factor = self.crack_propagation_algorithm.compute_stress_intensity_factor(
            stress, crack_length)
        
        # 计算裂纹扩展速率
        crack_growth_rate = self.crack_propagation_algorithm.compute_crack_growth_rate(
            stress_intensity_factor, fracture_toughness)
        
        # 更新裂纹长度
        new_crack_length = self.crack_propagation_algorithm.update_crack_length(
            crack_length, crack_growth_rate, dt)
        
        # 计算裂纹扩展方向
        crack_direction = self.crack_propagation_algorithm.compute_crack_direction(stress)
        
        return new_crack_length, stress_intensity_factor, crack_direction


def create_maximum_principal_stress_criterion(critical_stress: float = 100e6) -> MaximumPrincipalStressCriterion:
    """创建最大主应力断裂准则"""
    return MaximumPrincipalStressCriterion(critical_stress=critical_stress)


def create_maximum_principal_strain_criterion(critical_strain: float = 0.01) -> MaximumPrincipalStrainCriterion:
    """创建最大主应变断裂准则"""
    return MaximumPrincipalStrainCriterion(critical_strain=critical_strain)


def create_energy_release_rate_criterion(critical_energy_release_rate: float = 100.0) -> EnergyReleaseRateCriterion:
    """创建能量释放率断裂准则"""
    return EnergyReleaseRateCriterion(critical_energy_release_rate=critical_energy_release_rate)


def create_crack_propagation_algorithm(fracture_criterion: FractureCriterion) -> CrackPropagationAlgorithm:
    """创建裂纹扩展算法"""
    return CrackPropagationAlgorithm(fracture_criterion)


def demo_fracture_models():
    """演示断裂模型功能"""
    print("🔧 断裂模型演示")
    print("=" * 50)
    
    # 创建测试数据
    n_points = 100
    stress = np.zeros((n_points, 2, 2))
    stress[:, 0, 0] = np.linspace(0, 200e6, n_points)  # 轴向应力
    stress[:, 1, 1] = 0.3 * stress[:, 0, 0]  # 侧向应力
    stress[:, 0, 1] = stress[:, 1, 0] = 0.0  # 剪切应力
    
    strain = np.zeros_like(stress)
    strain[:, 0, 0] = stress[:, 0, 0] / 200e9  # 弹性应变
    strain[:, 1, 1] = stress[:, 1, 1] / 200e9
    
    crack_length = np.ones(n_points) * 1e-3  # 1mm初始裂纹
    fracture_toughness = 50e6  # Pa·√m
    dt = 1.0
    
    # 测试最大主应力断裂准则
    print("\n🔧 测试最大主应力断裂准则...")
    stress_criterion = create_maximum_principal_stress_criterion(critical_stress=100e6)
    
    criterion_value = stress_criterion.compute_fracture_criterion(stress, strain)
    fracture_initiated = stress_criterion.check_fracture_initiation(criterion_value, 100e6)
    
    print(f"   最大主应力: {criterion_value.max():.1e} Pa")
    print(f"   断裂起始点: {np.sum(fracture_initiated)} 个")
    
    # 测试最大主应变断裂准则
    print("\n🔧 测试最大主应变断裂准则...")
    strain_criterion = create_maximum_principal_strain_criterion(critical_strain=0.01)
    
    criterion_value = strain_criterion.compute_fracture_criterion(stress, strain)
    fracture_initiated = strain_criterion.check_fracture_initiation(criterion_value, 0.01)
    
    print(f"   最大主应变: {criterion_value.max():.3f}")
    print(f"   断裂起始点: {np.sum(fracture_initiated)} 个")
    
    # 测试能量释放率断裂准则
    print("\n🔧 测试能量释放率断裂准则...")
    energy_criterion = create_energy_release_rate_criterion(critical_energy_release_rate=100.0)
    
    criterion_value = energy_criterion.compute_fracture_criterion(stress, strain)
    fracture_initiated = energy_criterion.check_fracture_initiation(criterion_value, 100.0)
    
    print(f"   最大能量释放率: {criterion_value.max():.1f} J/m²")
    print(f"   断裂起始点: {np.sum(fracture_initiated)} 个")
    
    # 测试裂纹扩展算法
    print("\n🔧 测试裂纹扩展算法...")
    crack_propagation = create_crack_propagation_algorithm(stress_criterion)
    
    stress_intensity_factor = crack_propagation.compute_stress_intensity_factor(stress, crack_length)
    crack_growth_rate = crack_propagation.compute_crack_growth_rate(stress_intensity_factor, fracture_toughness)
    new_crack_length = crack_propagation.update_crack_length(crack_length, crack_growth_rate, dt)
    
    print(f"   最大应力强度因子: {stress_intensity_factor.max():.1e} Pa·√m")
    print(f"   最大裂纹扩展速率: {crack_growth_rate.max():.2e} m/s")
    print(f"   最大裂纹长度: {new_crack_length.max():.3f} m")
    
    # 测试断裂求解器
    print("\n🔧 测试断裂求解器...")
    fracture_solver = FractureSolver(stress_criterion, crack_propagation)
    
    new_crack_length, stress_intensity_factor, crack_direction = fracture_solver.solve_fracture(
        stress, strain, crack_length, fracture_toughness, dt)
    
    print(f"   断裂求解完成")
    print(f"   更新后最大裂纹长度: {new_crack_length.max():.3f} m")
    print(f"   裂纹扩展方向维度: {crack_direction.shape}")
    
    print("\n✅ 断裂模型演示完成!")


if __name__ == "__main__":
    demo_fracture_models()
