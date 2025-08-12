"""
化学-力学耦合模块
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
class ChemicalMechanicalState:
    """化学-力学耦合状态"""
    concentration: np.ndarray  # 化学浓度场
    displacement: np.ndarray  # 位移场
    stress: np.ndarray  # 应力场
    strain: np.ndarray  # 应变场
    chemical_strain: np.ndarray  # 化学应变
    reaction_rate: np.ndarray  # 反应速率
    diffusion_flux: np.ndarray  # 扩散通量
    time: float = 0.0


class ChemicalReactionModel(ABC):
    """化学反应模型基类"""

    def __init__(self, name: str = "Chemical Reaction Model"):
        self.name = name

    @abstractmethod
    def compute_reaction_rate(self, concentration: np.ndarray, 
                             temperature: np.ndarray,
                             pressure: np.ndarray) -> np.ndarray:
        """计算反应速率"""
        pass

    @abstractmethod
    def compute_equilibrium_concentration(self, temperature: np.ndarray,
                                         pressure: np.ndarray) -> np.ndarray:
        """计算平衡浓度"""
        pass


class ArrheniusReactionModel(ChemicalReactionModel):
    """Arrhenius反应模型

    基于Underworld2的Arrhenius反应模型：
    rate = A * exp(-E_a / (R * T)) * [C]^n
    """

    def __init__(self,
                 pre_exponential_factor: float = 1e6,  # 1/s
                 activation_energy: float = 100e3,  # J/mol
                 reaction_order: float = 1.0,
                 gas_constant: float = 8.314,  # J/(mol·K)
                 name: str = "Arrhenius Reaction Model"):
        super().__init__()
        self.pre_exponential_factor = pre_exponential_factor
        self.activation_energy = activation_energy
        self.reaction_order = reaction_order
        self.gas_constant = gas_constant

    def compute_reaction_rate(self, concentration: np.ndarray,
                             temperature: np.ndarray,
                             pressure: np.ndarray) -> np.ndarray:
        """计算反应速率"""
        # 避免温度为零或负数
        temperature_safe = np.maximum(temperature, 1e-6)

        # Arrhenius方程
        arrhenius_factor = self.pre_exponential_factor * \
                          np.exp(-self.activation_energy / (self.gas_constant * temperature_safe))

        # 浓度依赖项
        concentration_factor = concentration ** self.reaction_order

        # 总反应速率
        reaction_rate = arrhenius_factor * concentration_factor

        return reaction_rate

    def compute_equilibrium_concentration(self, temperature: np.ndarray,
                                         pressure: np.ndarray) -> np.ndarray:
        """计算平衡浓度"""
        # 简化的平衡浓度计算
        # 基于温度和压力的平衡浓度
        T_ref = 298.15  # 参考温度 [K]
        P_ref = 1e5     # 参考压力 [Pa]
        
        # 温度影响
        temperature_factor = np.exp(-1000 * (1.0 / temperature - 1.0 / T_ref))
        
        # 压力影响
        pressure_factor = np.exp(-1e-9 * (pressure - P_ref))
        
        # 平衡浓度
        equilibrium_concentration = 1.0 * temperature_factor * pressure_factor
        
        return equilibrium_concentration


class MineralDissolutionModel(ChemicalReactionModel):
    """矿物溶解模型 - 考虑对岩体强度的影响"""
    
    def __init__(self,
                 mineral_type: str = "calcite",
                 dissolution_rate: float = 1e-8,  # mol/(m²·s)
                 surface_area: float = 1e3,       # m²/m³
                 activation_energy: float = 50e3,  # J/mol
                 name: str = "Mineral Dissolution Model"):
        super().__init__()
        self.mineral_type = mineral_type
        self.dissolution_rate = dissolution_rate
        self.surface_area = surface_area
        self.activation_energy = activation_energy
        
        # 矿物性质
        self.mineral_properties = {
            'calcite': {
                'molar_mass': 100.09,      # g/mol
                'density': 2710.0,         # kg/m³
                'strength_contribution': 0.3,  # 强度贡献因子
                'solubility_product': 1e-8.48  # 溶度积
            },
            'quartz': {
                'molar_mass': 60.08,
                'density': 2650.0,
                'strength_contribution': 0.4,
                'solubility_product': 1e-9.96
            },
            'clay': {
                'molar_mass': 258.0,
                'density': 2600.0,
                'strength_contribution': 0.2,
                'solubility_product': 1e-7.0
            }
        }
    
    def compute_dissolution_rate(self, 
                                concentration: np.ndarray,
                                temperature: np.ndarray,
                                pressure: np.ndarray,
                                ph: np.ndarray) -> np.ndarray:
        """计算矿物溶解速率"""
        # 获取矿物性质
        props = self.mineral_properties.get(self.mineral_type, self.mineral_properties['calcite'])
        
        # 温度影响（Arrhenius关系）
        T_ref = 298.15
        gas_constant = 8.314
        temperature_factor = np.exp(-self.activation_energy / gas_constant * (1.0 / temperature - 1.0 / T_ref))
        
        # pH影响
        ph_factor = np.where(ph < 7.0, 1.0 + 0.5 * (7.0 - ph), 1.0)
        
        # 浓度影响（远离平衡）
        equilibrium_conc = props['solubility_product'] * np.ones_like(concentration)
        concentration_factor = np.maximum(0.0, 1.0 - concentration / (equilibrium_conc + 1e-12))
        
        # 总溶解速率
        dissolution_rate = (self.dissolution_rate * self.surface_area * 
                          temperature_factor * ph_factor * concentration_factor)
        
        return dissolution_rate
    
    def compute_strength_degradation(self, 
                                   initial_mineral_content: np.ndarray,
                                   dissolved_mineral: np.ndarray,
                                   time: float) -> Dict[str, np.ndarray]:
        """计算强度退化"""
        props = self.mineral_properties.get(self.mineral_type, self.mineral_properties['calcite'])
        
        # 剩余矿物含量
        remaining_mineral = np.maximum(0.0, initial_mineral_content - dissolved_mineral)
        mineral_fraction = remaining_mineral / (initial_mineral_content + 1e-12)
        
        # 强度退化因子
        strength_degradation = 1.0 - props['strength_contribution'] * (1.0 - mineral_fraction)
        
        # 弹性模量退化
        youngs_modulus_factor = strength_degradation ** 1.5  # 非线性关系
        
        # 内聚力退化
        cohesion_factor = strength_degradation ** 2.0
        
        # 摩擦角变化（矿物溶解增加摩擦角）
        friction_angle_change = 5.0 * (1.0 - mineral_fraction)  # 度
        
        return {
            'strength_degradation': strength_degradation,
            'youngs_modulus_factor': youngs_modulus_factor,
            'cohesion_factor': cohesion_factor,
            'friction_angle_change': friction_angle_change,
            'remaining_mineral_fraction': mineral_fraction
        }


class MineralPrecipitationModel(ChemicalReactionModel):
    """矿物沉淀模型 - 考虑对岩体强度的影响"""
    
    def __init__(self,
                 mineral_type: str = "calcite",
                 precipitation_rate: float = 1e-9,  # mol/(m³·s)
                 nucleation_rate: float = 1e6,      # 1/(m³·s)
                 growth_rate: float = 1e-10,        # m/s
                 name: str = "Mineral Precipitation Model"):
        super().__init__()
        self.mineral_type = mineral_type
        self.precipitation_rate = precipitation_rate
        self.nucleation_rate = nucleation_rate
        self.growth_rate = growth_rate
        
        # 沉淀矿物性质
        self.precipitation_properties = {
            'calcite': {
                'crystal_structure': 'rhombohedral',
                'strength_enhancement': 0.2,  # 强度增强因子
                'porosity_reduction': 0.1,    # 孔隙度减少因子
                'cementation_factor': 0.3     # 胶结因子
            },
            'quartz': {
                'crystal_structure': 'hexagonal',
                'strength_enhancement': 0.25,
                'porosity_reduction': 0.08,
                'cementation_factor': 0.25
            },
            'clay': {
                'crystal_structure': 'layered',
                'strength_enhancement': 0.15,
                'porosity_reduction': 0.12,
                'cementation_factor': 0.2
            }
        }
    
    def compute_precipitation_rate(self, 
                                 concentration: np.ndarray,
                                 temperature: np.ndarray,
                                 pressure: np.ndarray,
                                 supersaturation: np.ndarray) -> np.ndarray:
        """计算矿物沉淀速率"""
        props = self.precipitation_properties.get(self.mineral_type, self.precipitation_properties['calcite'])
        
        # 温度影响
        T_ref = 298.15
        temperature_factor = np.exp(-2000 * (1.0 / temperature - 1.0 / T_ref))
        
        # 过饱和度影响
        supersaturation_factor = np.maximum(0.0, supersaturation - 1.0)
        
        # 总沉淀速率
        precipitation_rate = (self.precipitation_rate * temperature_factor * 
                            supersaturation_factor ** 2.0)
        
        return precipitation_rate
    
    def compute_strength_enhancement(self, 
                                   precipitated_mineral: np.ndarray,
                                   initial_porosity: np.ndarray,
                                   time: float) -> Dict[str, np.ndarray]:
        """计算强度增强"""
        props = self.precipitation_properties.get(self.mineral_type, self.precipitation_properties['calcite'])
        
        # 矿物含量增加
        mineral_increase = precipitated_mineral / (1.0 + precipitated_mineral)
        
        # 强度增强因子
        strength_enhancement = 1.0 + props['strength_enhancement'] * mineral_increase
        
        # 孔隙度减少
        porosity_reduction = props['porosity_reduction'] * mineral_increase
        current_porosity = initial_porosity * (1.0 - porosity_reduction)
        
        # 胶结增强
        cementation_factor = 1.0 + props['cementation_factor'] * mineral_increase
        
        # 弹性模量增强
        youngs_modulus_enhancement = strength_enhancement ** 1.2
        
        # 内聚力增强
        cohesion_enhancement = strength_enhancement ** 1.5
        
        return {
            'strength_enhancement': strength_enhancement,
            'porosity_reduction': porosity_reduction,
            'current_porosity': current_porosity,
            'cementation_factor': cementation_factor,
            'youngs_modulus_enhancement': youngs_modulus_enhancement,
            'cohesion_enhancement': cohesion_enhancement
        }


class ChemicalMechanicalCoupling:
    """化学-力学耦合求解器 - 增强版"""
    
    def __init__(self, 
                 chemical_model: ChemicalReactionModel,
                 mechanical_model: 'MechanicalModel',
                 dissolution_model: MineralDissolutionModel = None,
                 precipitation_model: MineralPrecipitationModel = None):
        self.chemical_model = chemical_model
        self.mechanical_model = mechanical_model
        self.dissolution_model = dissolution_model
        self.precipitation_model = precipitation_model
        
        self.coupling_history = []
        self.convergence_criteria = {
            'max_iterations': 100,
            'tolerance': 1e-6,
            'relaxation_factor': 0.8
        }
        
        # 岩体强度参数
        self.initial_strength_params = {
            'youngs_modulus': 30e9,      # Pa
            'poissons_ratio': 0.25,
            'cohesion': 20e6,            # Pa
            'friction_angle': 30.0,      # 度
            'tensile_strength': 5e6      # Pa
        }
    
    def solve_coupled_system(self, 
                           initial_state: ChemicalMechanicalState,
                           boundary_conditions: Dict,
                           time_steps: int = 100,
                           dt: float = 0.01) -> List[ChemicalMechanicalState]:
        """求解耦合系统"""
        print("🔄 开始求解化学-力学耦合系统...")
        
        states = [initial_state]
        current_state = initial_state
        
        for step in range(time_steps):
            print(f"   时间步 {step+1}/{time_steps}")
            
            # 化学反应求解
            chemical_state = self._solve_chemical_field(current_state, boundary_conditions)
            
            # 力学场求解
            mechanical_state = self._solve_mechanical_field(current_state, chemical_state, boundary_conditions)
            
            # 强度演化计算
            strength_evolution = self._compute_strength_evolution(current_state, chemical_state, step * dt)
            
            # 耦合迭代
            coupled_state = self._coupling_iteration(chemical_state, mechanical_state, strength_evolution, boundary_conditions)
            
            # 更新状态
            current_state = coupled_state
            current_state.time = (step + 1) * dt
            states.append(current_state)
            
            # 检查收敛性
            if self._check_convergence(coupled_state, states[-2]):
                print(f"   收敛于时间步 {step+1}")
                break
        
        print("✅ 化学-力学耦合系统求解完成")
        return states
    
    def _solve_chemical_field(self, 
                            current_state: ChemicalMechanicalState,
                            boundary_conditions: Dict) -> ChemicalMechanicalState:
        """求解化学场"""
        # 计算反应速率
        reaction_rate = self.chemical_model.compute_reaction_rate(
            current_state.concentration,
            np.ones_like(current_state.concentration) * 298.15,  # 简化温度
            np.ones_like(current_state.concentration) * 1e5     # 简化压力
        )
        
        # 更新浓度
        concentration = current_state.concentration + reaction_rate * 0.01  # 时间步长
        
        # 更新状态
        chemical_state = ChemicalMechanicalState(
            concentration=concentration,
            displacement=current_state.displacement,
            stress=current_state.stress,
            strain=current_state.strain,
            chemical_strain=current_state.chemical_strain,
            reaction_rate=reaction_rate,
            diffusion_flux=current_state.diffusion_flux,
            time=current_state.time
        )
        
        return chemical_state
    
    def _solve_mechanical_field(self, 
                              current_state: ChemicalMechanicalState,
                              chemical_state: ChemicalMechanicalState,
                              boundary_conditions: Dict) -> ChemicalMechanicalState:
        """求解力学场"""
        # 简化的力学求解
        # 在实际应用中，这里应该求解完整的力学方程
        
        # 更新位移（简化）
        displacement = current_state.displacement + 0.1 * np.random.randn(*current_state.displacement.shape)
        
        # 更新状态
        mechanical_state = ChemicalMechanicalState(
            concentration=chemical_state.concentration,
            displacement=displacement,
            stress=current_state.stress,
            strain=current_state.strain,
            chemical_strain=current_state.chemical_strain,
            reaction_rate=chemical_state.reaction_rate,
            diffusion_flux=current_state.diffusion_flux,
            time=chemical_state.time
        )
        
        return mechanical_state
    
    def _compute_strength_evolution(self, 
                                  current_state: ChemicalMechanicalState,
                                  chemical_state: ChemicalMechanicalState,
                                  time: float) -> Dict[str, np.ndarray]:
        """计算强度演化"""
        strength_evolution = {}
        
        # 矿物溶解影响
        if self.dissolution_model is not None:
            # 模拟溶解过程
            initial_mineral_content = np.ones_like(chemical_state.concentration) * 0.3
            dissolved_mineral = chemical_state.reaction_rate * time
            
            dissolution_effects = self.dissolution_model.compute_strength_degradation(
                initial_mineral_content, dissolved_mineral, time
            )
            
            strength_evolution.update({
                'dissolution_effects': dissolution_effects,
                'strength_degradation': dissolution_effects['strength_degradation']
            })
        
        # 矿物沉淀影响
        if self.precipitation_model is not None:
            # 模拟沉淀过程
            precipitated_mineral = chemical_state.reaction_rate * time * 0.1
            initial_porosity = np.ones_like(chemical_state.concentration) * 0.2
            
            precipitation_effects = self.precipitation_model.compute_strength_enhancement(
                precipitated_mineral, initial_porosity, time
            )
            
            strength_evolution.update({
                'precipitation_effects': precipitation_effects,
                'strength_enhancement': precipitation_effects['strength_enhancement']
            })
        
        # 综合强度演化
        if 'strength_degradation' in strength_evolution and 'strength_enhancement' in strength_evolution:
            # 溶解和沉淀的综合效应
            net_strength_change = (strength_evolution['strength_enhancement'] * 
                                 strength_evolution['strength_degradation'])
            strength_evolution['net_strength_change'] = net_strength_change
        elif 'strength_degradation' in strength_evolution:
            strength_evolution['net_strength_change'] = strength_evolution['strength_degradation']
        elif 'strength_enhancement' in strength_evolution:
            strength_evolution['net_strength_change'] = strength_evolution['strength_enhancement']
        else:
            strength_evolution['net_strength_change'] = np.ones_like(chemical_state.concentration)
        
        return strength_evolution
    
    def _coupling_iteration(self, 
                           chemical_state: ChemicalMechanicalState,
                           mechanical_state: ChemicalMechanicalState,
                           strength_evolution: Dict,
                           boundary_conditions: Dict) -> ChemicalMechanicalState:
        """耦合迭代"""
        # 考虑强度演化对力学性质的影响
        net_strength_change = strength_evolution.get('net_strength_change', 
                                                   np.ones_like(chemical_state.concentration))
        
        # 更新应力（强度变化影响应力分布）
        updated_stress = mechanical_state.stress * net_strength_change.reshape(-1, 1)
        
        # 更新状态
        coupled_state = ChemicalMechanicalState(
            concentration=chemical_state.concentration,
            displacement=mechanical_state.displacement,
            stress=updated_stress,
            strain=mechanical_state.strain,
            chemical_strain=mechanical_state.chemical_strain,
            reaction_rate=chemical_state.reaction_rate,
            diffusion_flux=chemical_state.diffusion_flux,
            time=chemical_state.time
        )
        
        return coupled_state
    
    def _check_convergence(self, 
                          current_state: ChemicalMechanicalState,
                          previous_state: ChemicalMechanicalState) -> bool:
        """检查收敛性"""
        # 计算状态变化
        concentration_change = np.mean(np.abs(
            current_state.concentration - previous_state.concentration
        ))
        displacement_change = np.mean(np.abs(
            current_state.displacement - previous_state.displacement
        ))
        
        # 检查是否收敛
        max_change = max(concentration_change, displacement_change)
        return max_change < self.convergence_criteria['tolerance']
    
    def get_strength_evolution_summary(self, states: List[ChemicalMechanicalState]) -> Dict:
        """获取强度演化总结"""
        if not states:
            return {}
        
        # 分析强度变化趋势
        final_state = states[-1]
        initial_state = states[0]
        
        # 强度变化
        strength_change = np.mean(final_state.stress) / (np.mean(initial_state.stress) + 1e-12)
        
        # 化学影响
        chemical_influence = np.mean(final_state.reaction_rate) / (np.mean(initial_state.reaction_rate) + 1e-12)
        
        summary = {
            'total_time_steps': len(states),
            'final_time': final_state.time,
            'strength_change_factor': strength_change,
            'chemical_influence_factor': chemical_influence,
            'final_concentration': np.mean(final_state.concentration),
            'final_displacement': np.mean(np.abs(final_state.displacement))
        }
        
        return summary
        temperature_safe = np.maximum(temperature, 1e-6)
        
        # 基于温度的平衡浓度
        equilibrium_concentration = 1.0 / (1.0 + np.exp(-(temperature_safe - 1000) / 100))

        return equilibrium_concentration


class DiffusionModel(ABC):
    """扩散模型基类"""

    def __init__(self, name: str = "Diffusion Model"):
        self.name = name

    @abstractmethod
    def compute_diffusion_coefficient(self, concentration: np.ndarray,
                                     temperature: np.ndarray,
                                     pressure: np.ndarray) -> np.ndarray:
        """计算扩散系数"""
        pass

    @abstractmethod
    def compute_diffusion_flux(self, concentration: np.ndarray,
                              concentration_gradient: np.ndarray,
                              temperature: np.ndarray,
                              pressure: np.ndarray) -> np.ndarray:
        """计算扩散通量"""
        pass


class TemperatureDependentDiffusionModel(DiffusionModel):
    """温度依赖扩散模型

    基于Underworld2的温度依赖扩散模型：
    D = D_0 * exp(-E_D / (R * T))
    """

    def __init__(self,
                 pre_exponential_diffusivity: float = 1e-6,  # m²/s
                 diffusion_activation_energy: float = 150e3,  # J/mol
                 gas_constant: float = 8.314,  # J/(mol·K)
                 name: str = "Temperature Dependent Diffusion Model"):
        super().__init__(name)
        self.pre_exponential_diffusivity = pre_exponential_diffusivity
        self.diffusion_activation_energy = diffusion_activation_energy
        self.gas_constant = gas_constant

    def compute_diffusion_coefficient(self, concentration: np.ndarray,
                                     temperature: np.ndarray,
                                     pressure: np.ndarray) -> np.ndarray:
        """计算扩散系数"""
        # 避免温度为零或负数
        temperature_safe = np.maximum(temperature, 1e-6)

        # Arrhenius扩散系数
        diffusion_coefficient = self.pre_exponential_diffusivity * \
                               np.exp(-self.diffusion_activation_energy / (self.gas_constant * temperature_safe))

        return diffusion_coefficient

    def compute_diffusion_flux(self, concentration: np.ndarray,
                              concentration_gradient: np.ndarray,
                              temperature: np.ndarray,
                              pressure: np.ndarray) -> np.ndarray:
        """计算扩散通量"""
        # 计算扩散系数
        diffusion_coefficient = self.compute_diffusion_coefficient(concentration, temperature, pressure)

        # Fick扩散定律：J = -D * ∇C
        diffusion_flux = -diffusion_coefficient[:, np.newaxis] * concentration_gradient

        return diffusion_flux


class StressChemicalCoupling:
    """应力-化学耦合模型"""

    def __init__(self,
                 chemical_expansion_coefficient: float = 1e-3,  # 1/mol
                 stress_coupling_factor: float = 1.0,
                 name: str = "Stress-Chemical Coupling"):
        self.chemical_expansion_coefficient = chemical_expansion_coefficient
        self.stress_coupling_factor = stress_coupling_factor
        self.name = name

    def compute_chemical_strain(self, concentration: np.ndarray,
                               reference_concentration: float = 0.0) -> np.ndarray:
        """计算化学应变"""
        # 化学应变：ε_chem = β * (C - C_ref)
        chemical_strain = self.chemical_expansion_coefficient * (concentration - reference_concentration)

        return chemical_strain

    def compute_stress_effect_on_reaction(self, stress: np.ndarray,
                                         reaction_rate: np.ndarray) -> np.ndarray:
        """计算应力对反应的影响"""
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

        # 应力对反应速率的影响（简化模型）
        stress_factor = 1.0 + self.stress_coupling_factor * von_mises_stress / 1e9  # 归一化

        modified_reaction_rate = reaction_rate * stress_factor

        return modified_reaction_rate


class ChemicalMechanicalCoupling:
    """化学-力学耦合求解器"""

    def __init__(self, mesh,
                 reaction_model: ChemicalReactionModel,
                 diffusion_model: DiffusionModel,
                 stress_chemical_coupling: StressChemicalCoupling,
                 young_modulus: float = 70e9,
                 poisson_ratio: float = 0.3,
                 coupling_parameter: float = 1.0):
        """
        初始化化学-力学耦合求解器

        Args:
            mesh: 网格对象
            reaction_model: 化学反应模型
            diffusion_model: 扩散模型
            stress_chemical_coupling: 应力-化学耦合模型
            young_modulus: 杨氏模量 (Pa)
            poisson_ratio: 泊松比
            coupling_parameter: 耦合参数 (0-1)
        """
        self.mesh = mesh
        self.reaction_model = reaction_model
        self.diffusion_model = diffusion_model
        self.stress_chemical_coupling = stress_chemical_coupling
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.coupling_parameter = coupling_parameter

        # 计算材料参数
        self.lame_lambda = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        self.lame_mu = young_modulus / (2 * (1 + poisson_ratio))

        # MPI相关
        self.comm = MPI.COMM_WORLD if HAS_MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0

        # 求解器状态
        self.current_state = None
        self.previous_state = None

    def compute_chemical_stress(self, concentration: np.ndarray,
                               reference_concentration: float = 0.0) -> np.ndarray:
        """计算化学应力"""
        n_points = len(concentration)

        # 计算化学应变
        chemical_strain = self.stress_chemical_coupling.compute_chemical_strain(
            concentration, reference_concentration)

        # 计算化学应力（简化版本）
        chemical_stress = np.zeros((n_points, 2, 2))
        chemical_stress[:, 0, 0] = self.young_modulus * chemical_strain
        chemical_stress[:, 1, 1] = self.young_modulus * chemical_strain * self.poisson_ratio

        return chemical_stress

    def compute_chemical_force(self, chemical_stress: np.ndarray) -> np.ndarray:
        """计算化学力"""
        # 计算化学应力的散度
        n_points = len(chemical_stress)
        chemical_force = np.zeros((n_points, 2))

        for i in range(1, n_points - 1):
            # 简化的散度计算
            chemical_force[i, 0] = (chemical_stress[i+1, 0, 0] - chemical_stress[i-1, 0, 0]) / 2.0
            chemical_force[i, 1] = (chemical_stress[i+1, 1, 1] - chemical_stress[i-1, 1, 1]) / 2.0

        return chemical_force

    def assemble_coupling_matrix(self, dt: float) -> Tuple[csr_matrix, csr_matrix,
                                                          csr_matrix, csr_matrix]:
        """组装耦合矩阵"""
        if not HAS_SCIPY:
            raise ImportError("需要scipy来组装耦合矩阵")

        n_points = self.mesh.n_points

        # 化学扩散矩阵
        K_chemical = self._assemble_chemical_matrix()

        # 力学矩阵
        K_mechanical = self._assemble_mechanical_matrix()

        # 化学容量矩阵
        C_chemical = self._assemble_chemical_capacity_matrix()

        # 耦合矩阵
        C_coupling = self._assemble_coupling_matrix_internal(dt)

        return K_chemical, K_mechanical, C_chemical, C_coupling

    def solve_coupled_system(self,
                            initial_concentration: np.ndarray,
                            initial_displacement: np.ndarray,
                            boundary_conditions: Dict,
                            time_steps: int,
                            dt: float,
                            temperature: Optional[np.ndarray] = None,
                            pressure: Optional[np.ndarray] = None,
                            chemical_source: Optional[Callable] = None,
                            body_force: Optional[Callable] = None) -> List[ChemicalMechanicalState]:
        """求解耦合系统"""
        if temperature is None:
            temperature = np.full_like(initial_concentration, 293.15)
        if pressure is None:
            pressure = np.full_like(initial_concentration, 1e5)

        # 初始化状态
        current_state = ChemicalMechanicalState(
            concentration=initial_concentration.copy(),
            displacement=initial_displacement.copy(),
            stress=np.zeros((len(initial_concentration), 2, 2)),
            strain=np.zeros((len(initial_concentration), 2, 2)),
            chemical_strain=np.zeros_like(initial_concentration),
            reaction_rate=np.zeros_like(initial_concentration),
            diffusion_flux=np.zeros((len(initial_concentration), 2)),
            time=0.0
        )

        solution_history = [current_state]

        # 时间步进
        for step in range(time_steps):
            if self.rank == 0:
                print(f"化学-力学耦合求解步骤 {step + 1}/{time_steps}")

            # 求解单个时间步
            new_state = self._solve_coupled_step(
                current_state, dt, boundary_conditions, temperature, pressure,
                chemical_source, body_force
            )

            solution_history.append(new_state)
            current_state = new_state

        return solution_history

    def _solve_coupled_step(self, current_state: ChemicalMechanicalState,
                           dt: float, boundary_conditions: Dict,
                           temperature: np.ndarray, pressure: np.ndarray,
                           chemical_source: Optional[Callable],
                           body_force: Optional[Callable]) -> ChemicalMechanicalState:
        """求解单个耦合时间步"""
        # 设置当前状态
        self.current_state = current_state
        
        # 组装耦合矩阵
        K_chemical, K_mechanical, C_chemical, C_coupling = self.assemble_coupling_matrix(dt)

        # 计算源项
        chemical_source_vector = self._compute_chemical_source_vector(chemical_source)
        body_force_vector = self._compute_body_force_vector(body_force)

        # 迭代求解
        new_concentration, new_displacement = self._solve_coupled_iterative(
            K_chemical, K_mechanical, C_chemical, C_coupling,
            chemical_source_vector, body_force_vector,
            current_state, dt, boundary_conditions, temperature, pressure
        )

        # 更新状态
        new_chemical_strain = self.stress_chemical_coupling.compute_chemical_strain(new_concentration)
        new_stress = self._update_stress(new_displacement, new_chemical_strain)
        new_strain = self._update_strain(new_displacement)
        new_reaction_rate = self.reaction_model.compute_reaction_rate(
            new_concentration, temperature, pressure)
        new_diffusion_flux = self._compute_diffusion_flux(new_concentration, temperature, pressure)

        new_state = ChemicalMechanicalState(
            concentration=new_concentration,
            displacement=new_displacement,
            stress=new_stress,
            strain=new_strain,
            chemical_strain=new_chemical_strain,
            reaction_rate=new_reaction_rate,
            diffusion_flux=new_diffusion_flux,
            time=current_state.time + dt
        )

        return new_state

    def _solve_coupled_iterative(self, K_chemical: csr_matrix, K_mechanical: csr_matrix,
                                C_chemical: csr_matrix, C_coupling: csr_matrix,
                                chemical_source: np.ndarray, body_force: np.ndarray,
                                current_state: ChemicalMechanicalState,
                                dt: float, boundary_conditions: Dict,
                                temperature: np.ndarray, pressure: np.ndarray,
                                max_iterations: int = 10, tolerance: float = 1e-6):
        """迭代求解耦合系统"""
        concentration = current_state.concentration.copy()
        displacement = current_state.displacement.copy()
        
        # 确保位移是2D数组
        if displacement.ndim == 1:
            displacement = displacement.reshape(-1, 1)

        for iteration in range(max_iterations):
            # 保存前一次迭代的结果
            concentration_prev = concentration.copy()
            displacement_prev = displacement.copy()

            # 求解化学场
            concentration = self._solve_chemical_step(
                K_chemical, C_chemical, C_coupling, displacement,
                chemical_source, dt, boundary_conditions, temperature, pressure
            )

            # 求解力学场
            displacement = self._solve_mechanical_step(
                K_mechanical, concentration, body_force, boundary_conditions
            )
            
            # 确保位移是2D数组
            if displacement.ndim == 1:
                displacement = displacement.reshape(-1, 1)

            # 检查收敛性
            concentration_error = np.linalg.norm(concentration - concentration_prev)
            displacement_error = np.linalg.norm(displacement - displacement_prev)

            if concentration_error < tolerance and displacement_error < tolerance:
                if self.rank == 0:
                    print(f"耦合迭代收敛于第 {iteration + 1} 次迭代")
                break

        return concentration, displacement

    def _solve_chemical_step(self, K_chemical: csr_matrix, C_chemical: csr_matrix,
                            C_coupling: csr_matrix, displacement: np.ndarray,
                            chemical_source: np.ndarray, dt: float,
                            boundary_conditions: Dict, temperature: np.ndarray,
                            pressure: np.ndarray) -> np.ndarray:
        """求解化学场"""
        n_points = len(displacement)

        # 获取当前浓度
        current_concentration = self.current_state.concentration if self.current_state is not None else np.zeros(n_points)

        # 组装化学系统
        system_matrix = C_chemical / dt + K_chemical
        system_vector = C_chemical @ current_concentration / dt + chemical_source

        # 应用边界条件
        system_matrix, system_vector = self._apply_chemical_boundary_conditions(
            system_matrix, system_vector, boundary_conditions
        )

        # 求解
        new_concentration = spsolve(system_matrix, system_vector)

        return new_concentration

    def _solve_mechanical_step(self, K_mechanical: csr_matrix, concentration: np.ndarray,
                              body_force: np.ndarray, boundary_conditions: Dict) -> np.ndarray:
        """求解力学场"""
        n_points = len(concentration)

        # 计算化学应力
        chemical_stress = self.compute_chemical_stress(concentration)
        chemical_force = self.compute_chemical_force(chemical_stress)

        # 组装力学系统
        system_matrix = K_mechanical
        system_vector = body_force[:, 0] if body_force.ndim == 2 else body_force  # 只取x方向分量

        # 应用边界条件
        system_matrix, system_vector = self._apply_mechanical_boundary_conditions(
            system_matrix, system_vector, boundary_conditions
        )

        # 求解
        new_displacement = spsolve(system_matrix, system_vector)
        
        # 确保返回2D数组
        if new_displacement.ndim == 1:
            new_displacement = new_displacement.reshape(-1, 1)

        return new_displacement

    def _assemble_chemical_matrix(self) -> csr_matrix:
        """组装化学扩散矩阵"""
        n_points = self.mesh.n_points
        chemical_matrix = lil_matrix((n_points, n_points))

        for i in range(1, n_points - 1):
            chemical_matrix[i, i-1] = -1.0
            chemical_matrix[i, i] = 2.0
            chemical_matrix[i, i+1] = -1.0

        return chemical_matrix.tocsr()

    def _assemble_mechanical_matrix(self) -> csr_matrix:
        """组装力学矩阵"""
        n_points = self.mesh.n_points
        mechanical_matrix = lil_matrix((n_points, n_points))

        for i in range(1, n_points - 1):
            mechanical_matrix[i, i-1] = -self.lame_mu
            mechanical_matrix[i, i] = 2 * (self.lame_lambda + self.lame_mu)
            mechanical_matrix[i, i+1] = -self.lame_mu

        return mechanical_matrix.tocsr()

    def _assemble_chemical_capacity_matrix(self) -> csr_matrix:
        """组装化学容量矩阵"""
        n_points = self.mesh.n_points
        capacity_matrix = lil_matrix((n_points, n_points))

        for i in range(n_points):
            capacity_matrix[i, i] = 1.0

        return capacity_matrix.tocsr()

    def _assemble_coupling_matrix_internal(self, dt: float) -> csr_matrix:
        """组装内部耦合矩阵"""
        n_points = self.mesh.n_points
        coupling_matrix = lil_matrix((n_points, n_points))

        # 简化的耦合矩阵
        for i in range(n_points):
            coupling_matrix[i, i] = self.coupling_parameter * dt

        return coupling_matrix.tocsr()

    def _compute_chemical_source_vector(self, chemical_source: Optional[Callable]) -> np.ndarray:
        """计算化学源项向量"""
        n_points = self.mesh.n_points
        source_vector = np.zeros(n_points)

        if chemical_source is not None:
            current_time = self.current_state.time if self.current_state is not None else 0.0
            for i in range(n_points):
                source_vector[i] = chemical_source(i, current_time)

        return source_vector

    def _compute_body_force_vector(self, body_force: Optional[Callable]) -> np.ndarray:
        """计算体力向量"""
        n_points = self.mesh.n_points
        force_vector = np.zeros((n_points, 2))

        if body_force is not None:
            current_time = self.current_state.time if self.current_state is not None else 0.0
            for i in range(n_points):
                force_vector[i] = body_force(i, current_time)

        return force_vector

    def _compute_diffusion_flux(self, concentration: np.ndarray,
                               temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """计算扩散通量"""
        n_points = len(concentration)
        diffusion_flux = np.zeros((n_points, 2))

        # 计算浓度梯度（简化版本）
        for i in range(1, n_points - 1):
            concentration_gradient = np.array([
                (concentration[i+1] - concentration[i-1]) / 2.0,
                0.0  # 假设2D问题
            ])

            diffusion_flux[i] = self.diffusion_model.compute_diffusion_flux(
                concentration[i:i+1], concentration_gradient[np.newaxis, :],
                temperature[i:i+1], pressure[i:i+1]
            )[0]

        return diffusion_flux

    def _update_stress(self, displacement: np.ndarray, chemical_strain: np.ndarray) -> np.ndarray:
        """更新应力"""
        n_points = len(displacement)
        stress = np.zeros((n_points, 2, 2))

        # 确保位移是2D数组
        if displacement.ndim == 1:
            displacement = displacement.reshape(-1, 1)

        for i in range(1, n_points - 1):
            # 计算应变
            strain_xx = (displacement[i+1, 0] - displacement[i-1, 0]) / 2.0
            strain_yy = chemical_strain[i]  # 化学应变

            # 计算应力
            stress[i, 0, 0] = self.lame_lambda * (strain_xx + strain_yy) + 2 * self.lame_mu * strain_xx
            stress[i, 1, 1] = self.lame_lambda * (strain_xx + strain_yy) + 2 * self.lame_mu * strain_yy

        return stress

    def _update_strain(self, displacement: np.ndarray) -> np.ndarray:
        """更新应变"""
        n_points = len(displacement)
        strain = np.zeros((n_points, 2, 2))

        # 确保位移是2D数组
        if displacement.ndim == 1:
            displacement = displacement.reshape(-1, 1)

        for i in range(1, n_points - 1):
            strain[i, 0, 0] = (displacement[i+1, 0] - displacement[i-1, 0]) / 2.0

        return strain

    def _apply_chemical_boundary_conditions(self, A: csr_matrix, b: np.ndarray,
                                           boundary_conditions: Dict) -> Tuple[csr_matrix, np.ndarray]:
        """应用化学边界条件"""
        # 简化的边界条件处理
        if 'concentration' in boundary_conditions:
            for node_id, value in boundary_conditions['concentration'].items():
                A[node_id, :] = 0
                A[node_id, node_id] = 1
                b[node_id] = value

        return A, b

    def _apply_mechanical_boundary_conditions(self, A: csr_matrix, b: np.ndarray,
                                             boundary_conditions: Dict) -> Tuple[csr_matrix, np.ndarray]:
        """应用力学边界条件"""
        # 简化的边界条件处理
        if 'displacement' in boundary_conditions:
            for node_id, value in boundary_conditions['displacement'].items():
                A[node_id, :] = 0
                A[node_id, node_id] = 1
                b[node_id] = value

        return A, b

    def get_coupling_energy(self) -> float:
        """获取耦合能"""
        if self.current_state is None:
            return 0.0

        # 计算耦合能（简化版本）
        coupling_energy = np.sum(self.current_state.chemical_strain * 
                                self.current_state.stress[:, 0, 0])

        return coupling_energy

    def visualize_coupling_results(self, solution_history: List[ChemicalMechanicalState]):
        """可视化耦合结果"""
        try:
            import matplotlib.pyplot as plt

            n_steps = len(solution_history)
            times = [state.time for state in solution_history]

            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. 浓度演化
            concentrations = [state.concentration for state in solution_history]
            axes[0, 0].plot(times, [np.mean(conc) for conc in concentrations], 'b-', linewidth=2)
            axes[0, 0].set_title('平均浓度演化')
            axes[0, 0].set_xlabel('时间 (s)')
            axes[0, 0].set_ylabel('浓度')
            axes[0, 0].grid(True)

            # 2. 位移演化
            displacements = [state.displacement for state in solution_history]
            axes[0, 1].plot(times, [np.mean(np.abs(disp)) for disp in displacements], 'r-', linewidth=2)
            axes[0, 1].set_title('平均位移演化')
            axes[0, 1].set_xlabel('时间 (s)')
            axes[0, 1].set_ylabel('位移 (m)')
            axes[0, 1].grid(True)

            # 3. 反应速率
            reaction_rates = [state.reaction_rate for state in solution_history]
            axes[1, 0].plot(times, [np.mean(rate) for rate in reaction_rates], 'g-', linewidth=2)
            axes[1, 0].set_title('平均反应速率演化')
            axes[1, 0].set_xlabel('时间 (s)')
            axes[1, 0].set_ylabel('反应速率 (1/s)')
            axes[1, 0].grid(True)

            # 4. 耦合能
            coupling_energies = [self.get_coupling_energy() for _ in solution_history]
            axes[1, 1].plot(times, coupling_energies, 'm-', linewidth=2)
            axes[1, 1].set_title('耦合能演化')
            axes[1, 1].set_xlabel('时间 (s)')
            axes[1, 1].set_ylabel('耦合能 (J)')
            axes[1, 1].grid(True)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("需要matplotlib来可视化结果")


def create_chemical_mechanical_coupling(mesh, **kwargs) -> ChemicalMechanicalCoupling:
    """创建化学-力学耦合求解器"""
    # 默认参数
    default_reaction_model = ArrheniusReactionModel()
    default_diffusion_model = TemperatureDependentDiffusionModel()
    default_stress_chemical_coupling = StressChemicalCoupling()

    # 从kwargs中获取参数
    reaction_model = kwargs.get('reaction_model', default_reaction_model)
    diffusion_model = kwargs.get('diffusion_model', default_diffusion_model)
    stress_chemical_coupling = kwargs.get('stress_chemical_coupling', default_stress_chemical_coupling)
    young_modulus = kwargs.get('young_modulus', 70e9)
    poisson_ratio = kwargs.get('poisson_ratio', 0.3)
    coupling_parameter = kwargs.get('coupling_parameter', 1.0)

    return ChemicalMechanicalCoupling(
        mesh=mesh,
        reaction_model=reaction_model,
        diffusion_model=diffusion_model,
        stress_chemical_coupling=stress_chemical_coupling,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        coupling_parameter=coupling_parameter
    )


def demo_chemical_mechanical_coupling():
    """演示化学-力学耦合功能"""
    print("🔧 化学-力学耦合模块演示")
    print("=" * 50)

    # 创建简化的网格
    class MockMesh:
        def __init__(self, n_points=100):
            self.n_points = n_points

    mesh = MockMesh(100)

    # 创建化学反应模型
    print("\n🔧 创建化学反应模型...")
    reaction_model = ArrheniusReactionModel(
        pre_exponential_factor=1e6,
        activation_energy=100e3,
        reaction_order=1.0
    )

    # 创建扩散模型
    print("\n🔧 创建扩散模型...")
    diffusion_model = TemperatureDependentDiffusionModel(
        pre_exponential_diffusivity=1e-6,
        diffusion_activation_energy=150e3
    )

    # 创建应力-化学耦合模型
    print("\n🔧 创建应力-化学耦合模型...")
    stress_chemical_coupling = StressChemicalCoupling(
        chemical_expansion_coefficient=1e-3,
        stress_coupling_factor=1.0
    )

    # 创建化学-力学耦合求解器
    print("\n🔧 创建化学-力学耦合求解器...")
    coupling_solver = create_chemical_mechanical_coupling(
        mesh=mesh,
        reaction_model=reaction_model,
        diffusion_model=diffusion_model,
        stress_chemical_coupling=stress_chemical_coupling,
        young_modulus=70e9,
        poisson_ratio=0.3,
        coupling_parameter=1.0
    )

    # 创建初始条件
    print("\n🔧 创建初始条件...")
    initial_concentration = np.ones(100) * 0.5
    initial_displacement = np.zeros((100, 2))  # 2D位移数组
    temperature = np.linspace(1000, 2000, 100)  # K
    pressure = np.ones(100) * 1e8  # Pa

    # 定义边界条件
    boundary_conditions = {
        'concentration': {0: 1.0, 99: 0.0},  # 左边界浓度1.0，右边界浓度0.0
        'displacement': {0: 0.0, 99: 0.0}    # 两端固定
    }

    # 定义源项
    def chemical_source(node_id, time):
        return 0.0  # 无化学源项

    def body_force(node_id, time):
        return np.array([0.0, -9.81])  # 重力

    # 求解耦合系统
    print("\n🔧 求解化学-力学耦合系统...")
    solution_history = coupling_solver.solve_coupled_system(
        initial_concentration=initial_concentration,
        initial_displacement=initial_displacement,
        boundary_conditions=boundary_conditions,
        time_steps=10,
        dt=1.0,
        temperature=temperature,
        pressure=pressure,
        chemical_source=chemical_source,
        body_force=body_force
    )

    # 分析结果
    print("\n🔧 分析结果...")
    final_state = solution_history[-1]
    print(f"   最终平均浓度: {np.mean(final_state.concentration):.4f}")
    print(f"   最终平均位移: {np.mean(np.abs(final_state.displacement)):.2e} m")
    print(f"   最终平均反应速率: {np.mean(final_state.reaction_rate):.2e} 1/s")
    print(f"   最终平均化学应变: {np.mean(final_state.chemical_strain):.2e}")
    print(f"   耦合能: {coupling_solver.get_coupling_energy():.2e} J")

    print("\n✅ 化学-力学耦合模块演示完成!")


if __name__ == "__main__":
    demo_chemical_mechanical_coupling()
