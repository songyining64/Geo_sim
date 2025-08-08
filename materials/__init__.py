"""
Materials package for geological simulation.

This package contains material models and constitutive laws for geological simulations,
including plasticity models, phase change models, damage models, and advanced material models.
"""

from .plastic_models import (
    PlasticState,
    PlasticModel,
    VonMisesPlasticity,
    DruckerPragerPlasticity,
    PlasticSolver
)

from .phase_change_models import (
    PhaseState,
    PhaseChangeModel,
    SolidusLiquidusModel,
    PeridotiteMeltingModel,
    PhaseChangeSolver,
    CompositePhaseChangeModel
)

from .damage_models import (
    DamageState,
    DamageModel,
    IsotropicDamageModel,
    AnisotropicDamageModel,
    DamagePlasticityCoupling,
    DamageSolver
)

from .hardening_models import (
    HardeningState,
    HardeningModel,
    LinearHardeningModel,
    NonlinearHardeningModel,
    CyclicHardeningModel,
    HardeningSolver,
    create_linear_hardening_model,
    create_nonlinear_hardening_model,
    create_cyclic_hardening_model
)

from .phase_change_kinetics import (
    PhaseChangeKineticsState,
    PhaseChangeKineticsModel,
    RateLimitedPhaseChangeModel,
    NucleationModel,
    GrowthModel,
    CompositeKineticsModel,
    PhaseChangeKineticsSolver,
    create_rate_limited_phase_change_model,
    create_nucleation_model,
    create_growth_model,
    create_composite_kinetics_model
)

from .fracture_models import (
    FractureState,
    FractureCriterion,
    MaximumPrincipalStressCriterion,
    MaximumPrincipalStrainCriterion,
    EnergyReleaseRateCriterion,
    CrackPropagationAlgorithm,
    FractureSolver,
    create_maximum_principal_stress_criterion,
    create_maximum_principal_strain_criterion,
    create_energy_release_rate_criterion,
    create_crack_propagation_algorithm
)

from .experimental_validation import (
    ExperimentalData,
    ValidationResult,
    ExperimentalDataLoader,
    ModelValidator,
    PhaseChangeModelValidator,
    PlasticityModelValidator,
    ValidationReport
)

from .advanced_material_models import (
    MaterialState,
    Rheology,
    ConstantViscosity,
    PowerLawCreep,
    TemperatureDepthViscosity,
    AdvancedPlasticity,
    VonMisesPlasticity as AdvancedVonMisesPlasticity,
    CompositeRheology,
    AdvancedDensity,
    Material,
    MaterialRegistry,
    MaterialSolver,
    create_crust_material,
    create_mantle_material,
    create_air_material,
    create_composite_material,
    create_underworld2_style_material,
    demo_advanced_material_models
)

__all__ = [
    # 基础塑性模型
    'PlasticState',
    'PlasticModel', 
    'VonMisesPlasticity',
    'DruckerPragerPlasticity',
    'PlasticSolver',
    
    # 相变模型
    'PhaseState',
    'PhaseChangeModel',
    'SolidusLiquidusModel',
    'PeridotiteMeltingModel',
    'PhaseChangeSolver',
    'CompositePhaseChangeModel',
    
    # 损伤模型
    'DamageState',
    'DamageModel',
    'IsotropicDamageModel',
    'AnisotropicDamageModel',
    'DamagePlasticityCoupling',
    'DamageSolver',
    
    # 硬化模型
    'HardeningState',
    'HardeningModel',
    'LinearHardeningModel',
    'NonlinearHardeningModel',
    'CyclicHardeningModel',
    'HardeningSolver',
    'create_linear_hardening_model',
    'create_nonlinear_hardening_model',
    'create_cyclic_hardening_model',
    
    # 相变动力学
    'PhaseChangeKineticsState',
    'PhaseChangeKineticsModel',
    'RateLimitedPhaseChangeModel',
    'NucleationModel',
    'GrowthModel',
    'CompositeKineticsModel',
    'PhaseChangeKineticsSolver',
    'create_rate_limited_phase_change_model',
    'create_nucleation_model',
    'create_growth_model',
    'create_composite_kinetics_model',
    
    # 断裂模拟
    'FractureState',
    'FractureCriterion',
    'MaximumPrincipalStressCriterion',
    'MaximumPrincipalStrainCriterion',
    'EnergyReleaseRateCriterion',
    'CrackPropagationAlgorithm',
    'FractureSolver',
    'create_maximum_principal_stress_criterion',
    'create_maximum_principal_strain_criterion',
    'create_energy_release_rate_criterion',
    'create_crack_propagation_algorithm',
    
    # 实验验证
    'ExperimentalData',
    'ValidationResult',
    'ExperimentalDataLoader',
    'ModelValidator',
    'PhaseChangeModelValidator',
    'PlasticityModelValidator',
    'ValidationReport',
    
    # 高级材料模型
    'MaterialState',
    'Rheology',
    'ConstantViscosity',
    'PowerLawCreep',
    'TemperatureDepthViscosity',
    'AdvancedPlasticity',
    'AdvancedVonMisesPlasticity',
    'CompositeRheology',
    'AdvancedDensity',
    'Material',
    'MaterialRegistry',
    'MaterialSolver',
    
    # 便捷函数
    'create_crust_material',
    'create_mantle_material',
    'create_air_material',
    'create_composite_material',
    'create_underworld2_style_material',
    'demo_advanced_material_models'
] 