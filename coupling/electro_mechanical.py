"""
电磁-力学耦合模块
适用于地质电磁勘探、地震电磁耦合等场景
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
class ElectroMechanicalState:
    """电磁-力学耦合状态"""
    electric_field: np.ndarray      # 电场强度 [V/m]
    magnetic_field: np.ndarray      # 磁场强度 [T]
    electric_potential: np.ndarray  # 电势 [V]
    magnetic_potential: np.ndarray  # 磁矢势 [Wb/m]
    displacement: np.ndarray        # 位移场 [m]
    stress: np.ndarray             # 应力场 [Pa]
    strain: np.ndarray             # 应变场
    piezoelectric_strain: np.ndarray  # 压电应变
    magnetostrictive_strain: np.ndarray  # 磁致伸缩应变
    time: float = 0.0


@dataclass
class ElectromagneticProperties:
    """电磁材料属性"""
    # 电学性质
    electric_conductivity: float = 1e-6      # 电导率 [S/m]
    electric_permittivity: float = 8.85e-12  # 介电常数 [F/m]
    electric_susceptibility: float = 0.0     # 电极化率
    
    # 磁学性质
    magnetic_permeability: float = 1.257e-6  # 磁导率 [H/m]
    magnetic_susceptibility: float = 0.0     # 磁化率
    
    # 压电性质
    piezoelectric_coefficient: float = 0.0   # 压电系数 [C/m²]
    piezoelectric_stiffness: float = 0.0     # 压电刚度 [N/m²]
    
    # 磁致伸缩性质
    magnetostrictive_coefficient: float = 0.0  # 磁致伸缩系数 [m/A]
    
    # 地质特定性质
    porosity: float = 0.2                    # 孔隙度
    water_saturation: float = 0.8            # 含水饱和度
    clay_content: float = 0.1                # 粘土含量


class ElectromagneticModel(ABC):
    """电磁模型基类"""
    
    def __init__(self, name: str = "Electromagnetic Model"):
        self.name = name
    
    @abstractmethod
    def compute_electric_field(self, electric_potential: np.ndarray, 
                             coordinates: np.ndarray) -> np.ndarray:
        """计算电场强度"""
        pass
    
    @abstractmethod
    def compute_magnetic_field(self, magnetic_potential: np.ndarray,
                             coordinates: np.ndarray) -> np.ndarray:
        """计算磁场强度"""
        pass


class MaxwellEquationsModel(ElectromagneticModel):
    """Maxwell方程组模型"""
    
    def __init__(self, properties: ElectromagneticProperties):
        super().__init__("Maxwell Equations Model")
        self.properties = properties
    
    def compute_electric_field(self, electric_potential: np.ndarray, 
                             coordinates: np.ndarray) -> np.ndarray:
        """计算电场强度 E = -∇φ"""
        if not HAS_SCIPY:
            warnings.warn("scipy不可用，使用有限差分近似")
            return self._finite_difference_gradient(electric_potential, coordinates)
        
        # 使用scipy计算梯度
        from scipy.ndimage import gradient
        
        if coordinates.shape[1] == 2:  # 2D
            grad_x, grad_y = gradient(electric_potential)
            electric_field = np.stack([-grad_x, -grad_y], axis=-1)
        elif coordinates.shape[1] == 3:  # 3D
            grad_x, grad_y, grad_z = gradient(electric_potential)
            electric_field = np.stack([-grad_x, -grad_y, -grad_z], axis=-1)
        else:
            raise ValueError("只支持2D和3D坐标")
        
        return electric_field
    
    def compute_magnetic_field(self, magnetic_potential: np.ndarray,
                             coordinates: np.ndarray) -> np.ndarray:
        """计算磁场强度 B = ∇ × A"""
        if not HAS_SCIPY:
            warnings.warn("scipy不可用，使用有限差分近似")
            return self._finite_difference_curl(magnetic_potential, coordinates)
        
        # 使用scipy计算旋度
        from scipy.ndimage import gradient
        
        if coordinates.shape[1] == 2:  # 2D (假设A_z = 0)
            grad_ax_x, grad_ax_y = gradient(magnetic_potential[:, 0])
            grad_ay_x, grad_ay_y = gradient(magnetic_potential[:, 1])
            
            # B_z = ∂A_y/∂x - ∂A_x/∂y
            magnetic_field = np.zeros_like(magnetic_potential)
            magnetic_field[:, 2] = grad_ay_x - grad_ax_y
            
        elif coordinates.shape[1] == 3:  # 3D
            grad_ax_x, grad_ax_y, grad_ax_z = gradient(magnetic_potential[:, 0])
            grad_ay_x, grad_ay_y, grad_ay_z = gradient(magnetic_potential[:, 1])
            grad_az_x, grad_az_y, grad_az_z = gradient(magnetic_potential[:, 2])
            
            # B = ∇ × A
            magnetic_field = np.stack([
                grad_ay_z - grad_az_y,  # B_x = ∂A_y/∂z - ∂A_z/∂y
                grad_az_x - grad_ax_z,  # B_y = ∂A_z/∂x - ∂A_x/∂z
                grad_ax_y - grad_ay_x   # B_z = ∂A_x/∂y - ∂A_y/∂x
            ], axis=-1)
        else:
            raise ValueError("只支持2D和3D坐标")
        
        return magnetic_field
    
    def _finite_difference_gradient(self, potential: np.ndarray, 
                                   coordinates: np.ndarray) -> np.ndarray:
        """有限差分计算梯度"""
        dx = np.diff(coordinates[:, 0])
        dy = np.diff(coordinates[:, 1]) if coordinates.shape[1] > 1 else np.array([1.0])
        
        grad_x = np.gradient(potential, dx)
        grad_y = np.gradient(potential, dy) if coordinates.shape[1] > 1 else np.zeros_like(potential)
        
        if coordinates.shape[1] == 2:
            return np.stack([-grad_x, -grad_y], axis=-1)
        else:
            grad_z = np.gradient(potential, np.array([1.0]))
            return np.stack([-grad_x, -grad_y, -grad_z], axis=-1)
    
    def _finite_difference_curl(self, potential: np.ndarray, 
                                coordinates: np.ndarray) -> np.ndarray:
        """有限差分计算旋度"""
        # 简化实现
        if coordinates.shape[1] == 2:
            return np.zeros((len(potential), 3))
        else:
            return np.zeros_like(potential)


class PiezoelectricModel:
    """压电模型"""
    
    def __init__(self, properties: ElectromagneticProperties):
        self.properties = properties
    
    def compute_piezoelectric_strain(self, electric_field: np.ndarray) -> np.ndarray:
        """计算压电应变 ε_p = d * E"""
        if self.properties.piezoelectric_coefficient == 0:
            return np.zeros_like(electric_field)
        
        # 压电应变张量
        d = self.properties.piezoelectric_coefficient
        
        # 简化的压电应变计算
        piezoelectric_strain = d * electric_field
        
        return piezoelectric_strain
    
    def compute_piezoelectric_stress(self, piezoelectric_strain: np.ndarray) -> np.ndarray:
        """计算压电应力 σ_p = C_p * ε_p"""
        if self.properties.piezoelectric_stiffness == 0:
            return np.zeros_like(piezoelectric_strain)
        
        # 压电应力
        piezoelectric_stress = self.properties.piezoelectric_stiffness * piezoelectric_strain
        
        return piezoelectric_stress


class MagnetostrictiveModel:
    """磁致伸缩模型"""
    
    def __init__(self, properties: ElectromagneticProperties):
        self.properties = properties
    
    def compute_magnetostrictive_strain(self, magnetic_field: np.ndarray) -> np.ndarray:
        """计算磁致伸缩应变 ε_m = λ * H²"""
        if self.properties.magnetostrictive_coefficient == 0:
            return np.zeros_like(magnetic_field)
        
        # 磁致伸缩系数
        lambda_coeff = self.properties.magnetostrictive_coefficient
        
        # 磁场强度平方
        H_squared = np.sum(magnetic_field**2, axis=-1, keepdims=True)
        
        # 磁致伸缩应变
        magnetostrictive_strain = lambda_coeff * H_squared
        
        return magnetostrictive_strain


class GeologicalElectromagneticModel:
    """地质电磁模型 - 考虑地质特征的影响"""
    
    def __init__(self, properties: ElectromagneticProperties):
        self.properties = properties
    
    def compute_effective_conductivity(self, porosity: np.ndarray, 
                                     water_saturation: np.ndarray,
                                     clay_content: np.ndarray) -> np.ndarray:
        """计算有效电导率 - Archie公式的扩展"""
        # 基础电导率
        sigma_0 = self.properties.electric_conductivity
        
        # 孔隙度影响
        porosity_factor = porosity ** 2.0
        
        # 含水饱和度影响
        saturation_factor = water_saturation ** 2.0
        
        # 粘土含量影响（粘土增加电导率）
        clay_factor = 1.0 + 5.0 * clay_content
        
        # 有效电导率
        effective_conductivity = sigma_0 * porosity_factor * saturation_factor * clay_factor
        
        return effective_conductivity
    
    def compute_effective_permittivity(self, porosity: np.ndarray,
                                     water_saturation: np.ndarray,
                                     clay_content: np.ndarray) -> np.ndarray:
        """计算有效介电常数 - 混合介质模型"""
        # 各组分介电常数
        epsilon_air = 1.0
        epsilon_water = 80.0
        epsilon_clay = 10.0
        epsilon_rock = 5.0
        
        # 体积分数
        air_fraction = porosity * (1.0 - water_saturation)
        water_fraction = porosity * water_saturation
        clay_fraction = clay_content
        rock_fraction = 1.0 - porosity - clay_content
        
        # 有效介电常数（简化混合模型）
        effective_permittivity = (air_fraction * epsilon_air + 
                                water_fraction * epsilon_water +
                                clay_fraction * epsilon_clay +
                                rock_fraction * epsilon_rock)
        
        return effective_permittivity * self.properties.electric_permittivity


class ElectroMechanicalCoupling:
    """电磁-力学耦合求解器"""
    
    def __init__(self, 
                 electromagnetic_model: ElectromagneticModel,
                 piezoelectric_model: PiezoelectricModel,
                 magnetostrictive_model: MagnetostrictiveModel,
                 geological_model: GeologicalElectromagneticModel):
        self.em_model = electromagnetic_model
        self.piezo_model = piezoelectric_model
        self.magneto_model = magnetostrictive_model
        self.geo_model = geological_model
        
        self.coupling_history = []
        self.convergence_criteria = {
            'max_iterations': 50,
            'tolerance': 1e-6,
            'relaxation_factor': 0.7
        }
    
    def solve_coupled_system(self, 
                           initial_state: ElectroMechanicalState,
                           boundary_conditions: Dict,
                           time_steps: int = 100,
                           dt: float = 0.01) -> List[ElectroMechanicalState]:
        """求解耦合系统"""
        print("🔄 开始求解电磁-力学耦合系统...")
        
        states = [initial_state]
        current_state = initial_state
        
        for step in range(time_steps):
            print(f"   时间步 {step+1}/{time_steps}")
            
            # 电磁场求解
            em_state = self._solve_electromagnetic_field(current_state, boundary_conditions)
            
            # 力学场求解
            mech_state = self._solve_mechanical_field(current_state, em_state, boundary_conditions)
            
            # 耦合迭代
            coupled_state = self._coupling_iteration(em_state, mech_state, boundary_conditions)
            
            # 更新状态
            current_state = coupled_state
            current_state.time = (step + 1) * dt
            states.append(current_state)
            
            # 检查收敛性
            if self._check_convergence(coupled_state, states[-2]):
                print(f"   收敛于时间步 {step+1}")
                break
        
        print("✅ 电磁-力学耦合系统求解完成")
        return states
    
    def _solve_electromagnetic_field(self, 
                                   current_state: ElectroMechanicalState,
                                   boundary_conditions: Dict) -> ElectroMechanicalState:
        """求解电磁场"""
        # 简化的电磁场求解
        # 在实际应用中，这里应该求解Maxwell方程组
        
        # 更新电势（简化的扩散方程）
        electric_potential = current_state.electric_potential.copy()
        electric_potential += 0.1 * np.random.randn(*electric_potential.shape)
        
        # 计算电场
        coordinates = np.random.rand(len(electric_potential), 3)  # 简化的坐标
        electric_field = self.em_model.compute_electric_field(electric_potential, coordinates)
        
        # 更新状态
        em_state = ElectroMechanicalState(
            electric_field=electric_field,
            electric_potential=electric_potential,
            magnetic_field=current_state.magnetic_field,
            magnetic_potential=current_state.magnetic_potential,
            displacement=current_state.displacement,
            stress=current_state.stress,
            strain=current_state.strain,
            piezoelectric_strain=current_state.piezoelectric_strain,
            magnetostrictive_strain=current_state.magnetostrictive_strain,
            time=current_state.time
        )
        
        return em_state
    
    def _solve_mechanical_field(self, 
                               current_state: ElectroMechanicalState,
                               em_state: ElectroMechanicalState,
                               boundary_conditions: Dict) -> ElectroMechanicalState:
        """求解力学场"""
        # 计算压电应变
        piezoelectric_strain = self.piezo_model.compute_piezoelectric_strain(
            em_state.electric_field
        )
        
        # 计算磁致伸缩应变
        magnetostrictive_strain = self.magneto_model.compute_magnetostrictive_strain(
            em_state.magnetic_field
        )
        
        # 总应变
        total_strain = current_state.strain + piezoelectric_strain + magnetostrictive_strain
        
        # 简化的应力计算（胡克定律）
        youngs_modulus = 30e9  # Pa
        poissons_ratio = 0.25
        
        # 平面应力状态下的应力
        if total_strain.shape[-1] == 2:  # 2D
            stress = youngs_modulus / (1 - poissons_ratio**2) * np.array([
                total_strain[:, 0] + poissons_ratio * total_strain[:, 1],
                total_strain[:, 1] + poissons_ratio * total_strain[:, 0]
            ]).T
        else:  # 3D
            stress = youngs_modulus / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio)) * np.array([
                (1 - poissons_ratio) * total_strain[:, 0] + poissons_ratio * (total_strain[:, 1] + total_strain[:, 2]),
                (1 - poissons_ratio) * total_strain[:, 1] + poissons_ratio * (total_strain[:, 0] + total_strain[:, 2]),
                (1 - poissons_ratio) * total_strain[:, 2] + poissons_ratio * (total_strain[:, 0] + total_strain[:, 1])
            ]).T
        
        # 更新状态
        mech_state = ElectroMechanicalState(
            electric_field=em_state.electric_field,
            electric_potential=em_state.electric_potential,
            magnetic_field=em_state.magnetic_field,
            magnetic_potential=em_state.magnetic_potential,
            displacement=current_state.displacement,
            stress=stress,
            strain=total_strain,
            piezoelectric_strain=piezoelectric_strain,
            magnetostrictive_strain=magnetostrictive_strain,
            time=current_state.time
        )
        
        return mech_state
    
    def _coupling_iteration(self, 
                           em_state: ElectroMechanicalState,
                           mech_state: ElectroMechanicalState,
                           boundary_conditions: Dict) -> ElectroMechanicalState:
        """耦合迭代"""
        # 简化的耦合迭代
        # 在实际应用中，这里应该进行更复杂的耦合计算
        
        # 考虑应力对电磁性质的影响
        stress_factor = 1.0 + 0.1 * np.mean(np.abs(mech_state.stress)) / 1e9
        
        # 更新电场（应力影响电导率）
        updated_electric_field = em_state.electric_field * stress_factor
        
        # 更新状态
        coupled_state = ElectroMechanicalState(
            electric_field=updated_electric_field,
            electric_potential=em_state.electric_potential,
            magnetic_field=em_state.magnetic_field,
            magnetic_potential=em_state.magnetic_potential,
            displacement=mech_state.displacement,
            stress=mech_state.stress,
            strain=mech_state.strain,
            piezoelectric_strain=mech_state.piezoelectric_strain,
            magnetostrictive_strain=mech_state.magnetostrictive_strain,
            time=em_state.time
        )
        
        return coupled_state
    
    def _check_convergence(self, 
                          current_state: ElectroMechanicalState,
                          previous_state: ElectroMechanicalState) -> bool:
        """检查收敛性"""
        # 计算状态变化
        electric_change = np.mean(np.abs(
            current_state.electric_field - previous_state.electric_field
        ))
        stress_change = np.mean(np.abs(
            current_state.stress - previous_state.stress
        ))
        
        # 检查是否收敛
        max_change = max(electric_change, stress_change)
        return max_change < self.convergence_criteria['tolerance']
    
    def compute_geological_response(self, 
                                  porosity: np.ndarray,
                                  water_saturation: np.ndarray,
                                  clay_content: np.ndarray) -> Dict:
        """计算地质响应"""
        # 有效电导率
        effective_conductivity = self.geo_model.compute_effective_conductivity(
            porosity, water_saturation, clay_content
        )
        
        # 有效介电常数
        effective_permittivity = self.geo_model.compute_effective_permittivity(
            porosity, water_saturation, clay_content
        )
        
        # 地质响应
        response = {
            'effective_conductivity': effective_conductivity,
            'effective_permittivity': effective_permittivity,
            'resistivity': 1.0 / (effective_conductivity + 1e-12),  # 电阻率
            'impedance': np.sqrt(effective_permittivity / (effective_conductivity + 1e-12))
        }
        
        return response


def create_electro_mechanical_system() -> Dict:
    """创建电磁-力学耦合系统"""
    # 创建材料属性
    properties = ElectromagneticProperties(
        electric_conductivity=1e-4,      # 地质材料典型值
        electric_permittivity=8.85e-12,
        magnetic_permeability=1.257e-6,
        piezoelectric_coefficient=1e-12,  # 地质材料压电系数
        magnetostrictive_coefficient=1e-15  # 地质材料磁致伸缩系数
    )
    
    # 创建模型
    em_model = MaxwellEquationsModel(properties)
    piezo_model = PiezoelectricModel(properties)
    magneto_model = MagnetostrictiveModel(properties)
    geo_model = GeologicalElectromagneticModel(properties)
    
    # 创建耦合求解器
    coupling_solver = ElectroMechanicalCoupling(
        em_model, piezo_model, magneto_model, geo_model
    )
    
    system = {
        'properties': properties,
        'em_model': em_model,
        'piezo_model': piezo_model,
        'magneto_model': magneto_model,
        'geo_model': geo_model,
        'coupling_solver': coupling_solver
    }
    
    print("🔄 电磁-力学耦合系统创建完成")
    return system


def demo_electro_mechanical_coupling():
    """演示电磁-力学耦合"""
    print("⚡ 电磁-力学耦合演示")
    print("=" * 60)
    
    try:
        # 创建系统
        system = create_electro_mechanical_system()
        
        # 创建初始状态
        n_points = 100
        initial_state = ElectroMechanicalState(
            electric_field=np.random.randn(n_points, 3) * 1e3,      # V/m
            magnetic_field=np.random.randn(n_points, 3) * 1e-6,     # T
            electric_potential=np.random.randn(n_points) * 1e3,     # V
            magnetic_potential=np.random.randn(n_points, 3) * 1e-6, # Wb/m
            displacement=np.random.randn(n_points, 3) * 1e-6,       # m
            stress=np.random.randn(n_points, 3) * 1e6,             # Pa
            strain=np.random.randn(n_points, 3) * 1e-6,
            piezoelectric_strain=np.zeros((n_points, 3)),
            magnetostrictive_strain=np.zeros((n_points, 3))
        )
        
        # 边界条件
        boundary_conditions = {
            'electric_potential': {'top': 1000, 'bottom': 0},  # V
            'displacement': {'left': 'fixed', 'right': 'free'},
            'stress': {'top': 'free', 'bottom': 'fixed'}
        }
        
        # 求解耦合系统
        coupling_solver = system['coupling_solver']
        states = coupling_solver.solve_coupled_system(
            initial_state, boundary_conditions, time_steps=20, dt=0.01
        )
        
        print(f"   求解完成，共 {len(states)} 个时间步")
        
        # 计算地质响应
        porosity = np.random.rand(n_points) * 0.3 + 0.1      # 0.1-0.4
        water_saturation = np.random.rand(n_points) * 0.4 + 0.6  # 0.6-1.0
        clay_content = np.random.rand(n_points) * 0.2 + 0.05     # 0.05-0.25
        
        geological_response = coupling_solver.compute_geological_response(
            porosity, water_saturation, clay_content
        )
        
        print(f"   地质响应计算完成:")
        print(f"     平均电阻率: {np.mean(geological_response['resistivity']):.2e} Ω·m")
        print(f"     平均阻抗: {np.mean(geological_response['impedance']):.2e} Ω")
        
        print("\n✅ 电磁-力学耦合演示完成!")
        return True
        
    except Exception as e:
        print(f"❌ 电磁-力学耦合演示失败: {e}")
        return False


if __name__ == "__main__":
    demo_electro_mechanical_coupling()
