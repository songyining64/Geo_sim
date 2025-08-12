"""
Coupling package for multi-physics simulations.

This package contains coupling modules for multi-physics simulations,
including thermal-mechanical coupling, fluid-solid coupling, thermodynamic coupling, and chemical-mechanical coupling.
"""

from .thermal_mechanical import (
    CouplingState,
    ThermoMechanicalCoupling,
    create_thermo_mechanical_coupling
)

# 流体-固体耦合模块
try:
    from .fluid_solid import (
        FluidSolidState,
        FluidSolver,
        SolidSolver,
        NavierStokesSolver,
        ElasticSolidSolver,
        FluidSolidCoupling,
        create_fluid_solid_coupling,
        demo_fluid_solid_coupling
    )
    FLUID_SOLID_AVAILABLE = True
except ImportError:
    FLUID_SOLID_AVAILABLE = False

# 热力学耦合模块
try:
    from .thermodynamics import (
        ThermodynamicState,
        HeatTransferSolver,
        PhaseChangeSolver,
        ThermodynamicCoupling,
        create_thermodynamic_coupling,
        demo_thermodynamic_coupling
    )
    THERMODYNAMICS_AVAILABLE = True
except ImportError:
    THERMODYNAMICS_AVAILABLE = False

# 化学-力学耦合模块
try:
    from .chemical_mechanical import (
        ChemicalMechanicalState,
        ChemicalReactionModel,
        ArrheniusReactionModel,
        MineralDissolutionModel,
        MineralPrecipitationModel,
        DiffusionModel,
        TemperatureDependentDiffusionModel,
        StressChemicalCoupling,
        ChemicalMechanicalCoupling,
        create_chemical_mechanical_coupling,
        demo_chemical_mechanical_coupling
    )
    CHEMICAL_MECHANICAL_AVAILABLE = True
except ImportError:
    CHEMICAL_MECHANICAL_AVAILABLE = False

# 电磁-力学耦合模块
try:
    from .electro_mechanical import (
        ElectroMechanicalState,
        ElectromagneticProperties,
        ElectromagneticModel,
        MaxwellEquationsModel,
        PiezoelectricModel,
        MagnetostrictiveModel,
        GeologicalElectromagneticModel,
        ElectroMechanicalCoupling,
        create_electro_mechanical_system,
        demo_electro_mechanical_coupling
    )
    ELECTRO_MECHANICAL_AVAILABLE = True
except ImportError:
    ELECTRO_MECHANICAL_AVAILABLE = False

# 多相流体耦合模块
try:
    from .multiphase_fluid import (
        MultiphaseFluidState,
        MultiphaseFluidProperties,
        CapillaryPressureModel,
        BrooksCoreyModel,
        RelativePermeabilityModel,
        CoreyModel,
        MultiphaseFlowModel,
        ThermalCouplingModel,
        ChemicalTransportModel,
        MultiphaseFluidCoupling,
        create_multiphase_fluid_system,
        demo_multiphase_fluid_coupling
    )
    MULTIPHASE_FLUID_AVAILABLE = True
except ImportError:
    MULTIPHASE_FLUID_AVAILABLE = False

__all__ = [
    # 热-力学耦合
    'CouplingState',
    'ThermoMechanicalCoupling',
    'create_thermo_mechanical_coupling',
]

# 添加流体-固体耦合
if FLUID_SOLID_AVAILABLE:
    __all__.extend([
        'FluidSolidState',
        'FluidSolver',
        'SolidSolver',
        'NavierStokesSolver',
        'ElasticSolidSolver',
        'FluidSolidCoupling',
        'create_fluid_solid_coupling',
        'demo_fluid_solid_coupling'
    ])

# 添加热力学耦合
if THERMODYNAMICS_AVAILABLE:
    __all__.extend([
        'ThermodynamicState',
        'HeatTransferSolver',
        'PhaseChangeSolver',
        'ThermodynamicCoupling',
        'create_thermodynamic_coupling',
        'demo_thermodynamic_coupling'
    ])

# 添加化学-力学耦合
if CHEMICAL_MECHANICAL_AVAILABLE:
    __all__.extend([
        'ChemicalMechanicalState',
        'ChemicalReactionModel',
        'ArrheniusReactionModel',
        'MineralDissolutionModel',
        'MineralPrecipitationModel',
        'DiffusionModel',
        'TemperatureDependentDiffusionModel',
        'StressChemicalCoupling',
        'ChemicalMechanicalCoupling',
        'create_chemical_mechanical_coupling',
        'demo_chemical_mechanical_coupling'
    ])

# 添加电磁-力学耦合
if ELECTRO_MECHANICAL_AVAILABLE:
    __all__.extend([
        'ElectroMechanicalState',
        'ElectromagneticProperties',
        'ElectromagneticModel',
        'MaxwellEquationsModel',
        'PiezoelectricModel',
        'MagnetostrictiveModel',
        'GeologicalElectromagneticModel',
        'ElectroMechanicalCoupling',
        'create_electro_mechanical_system',
        'demo_electro_mechanical_coupling'
    ])

# 添加多相流体耦合
if MULTIPHASE_FLUID_AVAILABLE:
    __all__.extend([
        'MultiphaseFluidState',
        'MultiphaseFluidProperties',
        'CapillaryPressureModel',
        'BrooksCoreyModel',
        'RelativePermeabilityModel',
        'CoreyModel',
        'MultiphaseFlowModel',
        'ThermalCouplingModel',
        'ChemicalTransportModel',
        'MultiphaseFluidCoupling',
        'create_multiphase_fluid_system',
        'demo_multiphase_fluid_coupling'
    ]) 