"""
核心模块 - 统一API接口和调试工具

提供所有核心模块（有限元、多物理场耦合、ML模型）的统一接口，
包括setup()、run()、visualize()等方法，减少用户学习成本。

同时提供增强的调试与可视化工具，支持实时监控、错误诊断和性能分析。
"""

import warnings

from .unified_api import (
    # 配置类
    SimulationConfig,
    
    # 结果类
    SimulationResult,
    
    # 仿真器基类
    BaseSimulator,
    FiniteElementSimulator,
    MultiPhysicsSimulator,
    MLSimulator,
    
    # 工厂函数
    create_simulator,
    load_config_from_template
)

try:
    from .geodynamic_simulation import (
        GeodynamicConfig,
        GeodynamicSimulation,
        create_mantle_convection_simulation,
        create_lithospheric_deformation_simulation
    )
    HAS_GEODYNAMIC = True
except ImportError as e:
    HAS_GEODYNAMIC = False
    warnings.warn(f"geodynamic_simulation module not available: {e}")

# 完整接口演示模块（如果可用）
try:
    from .complete_interface_demo import (
        demo_basic_simulation,
        demo_adaptive_mesh_refinement,
        demo_gpu_acceleration,
        demo_visualization,
        demo_multi_physics_coupling,
        demo_complete_workflow,
        main as run_complete_demo
    )
    HAS_COMPLETE_DEMO = True
except ImportError:
    HAS_COMPLETE_DEMO = False
    warnings.warn("complete_interface_demo module not available.")

from .debug_tools import (
    # 配置类
    DebugConfig,
    
    # 约束和指标类
    PhysicalConstraint,
    ConvergenceMetrics,
    
    # 监控和诊断类
    RealTimeMonitor,
    ErrorDiagnostic,
    AdvancedVisualizer,
    DebugManager,
    
    # 便捷函数
    create_debug_manager,
    quick_debug_setup
)

# 版本信息
__version__ = "1.0.0"
__author__ = "GeoSim Team"

# 导出所有公共接口
__all__ = [
    # 统一API
    'SimulationConfig',
    'SimulationResult',
    'BaseSimulator',
    'FiniteElementSimulator',
    'MultiPhysicsSimulator',
    'MLSimulator',
    'create_simulator',
    'load_config_from_template',
    
    # 地质动力学仿真
    'GeodynamicConfig',
    'GeodynamicSimulation',
    'create_mantle_convection_simulation',
    'create_lithospheric_deformation_simulation',
    
    # 完整接口演示
    'demo_basic_simulation',
    'demo_adaptive_mesh_refinement',
    'demo_gpu_acceleration',
    'demo_visualization',
    'demo_multi_physics_coupling',
    'demo_complete_workflow',
    'run_complete_demo',
    
    # 调试工具
    'DebugConfig',
    'PhysicalConstraint',
    'ConvergenceMetrics',
    'RealTimeMonitor',
    'ErrorDiagnostic',
    'AdvancedVisualizer',
    'DebugManager',
    'create_debug_manager',
    'quick_debug_setup'
]

# 快速导入别名
def quick_start():
    """快速开始函数 - 创建默认配置的仿真器和调试管理器"""
    from .unified_api import SimulationConfig
    from .debug_tools import quick_debug_setup
    
    # 创建默认配置
    config = SimulationConfig(
        name="quick_start_simulation",
        description="快速开始仿真配置",
        physics_params={
            'gravity': 9.81,
            'thermal_expansion': 3e-5,
            'thermal_diffusivity': 1e-6
        },
        numerical_params={
            'time_steps': 100,
            'dt': 0.01,
            'tolerance': 1e-6
        }
    )
    
    # 创建调试管理器
    debug_manager = quick_debug_setup()
    
    return config, debug_manager


def load_scenario_template(scenario_name: str):
    """加载场景配置模板
    
    Args:
        scenario_name: 场景名称，支持 'reservoir_simulation', 'seismic_inversion'
    
    Returns:
        SimulationConfig: 加载的配置对象
    """
    try:
        return load_config_from_template(scenario_name)
    except FileNotFoundError:
        available_templates = ['reservoir_simulation', 'seismic_inversion']
        raise ValueError(f"场景模板 '{scenario_name}' 不存在。可用模板: {available_templates}")


def create_simulation_with_debug(simulator_type: str, config: 'SimulationConfig' = None,
                                debug_config: 'DebugConfig' = None):
    """创建带调试功能的仿真器
    
    Args:
        simulator_type: 仿真器类型 ('finite_element', 'multi_physics', 'ml')
        config: 仿真配置
        debug_config: 调试配置
    
    Returns:
        tuple: (仿真器, 调试管理器)
    """
    # 创建仿真器
    simulator = create_simulator(simulator_type, config)
    
    # 创建调试管理器
    debug_manager = create_debug_manager(debug_config)
    
    return simulator, debug_manager


# 示例用法
def demo_unified_api():
    """演示统一API的使用"""
    print("=== 统一API演示 ===")
    
    # 1. 加载场景配置模板
    try:
        reservoir_config = load_scenario_template('reservoir_simulation')
        print(f"✅ 加载油气藏模拟配置: {reservoir_config.name}")
        print(f"   描述: {reservoir_config.description}")
        print(f"   时间步数: {reservoir_config.numerical_params['time_steps']}")
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
    
    # 2. 创建仿真器
    try:
        simulator = create_simulator('finite_element')
        print(f"✅ 创建有限元仿真器: {type(simulator).__name__}")
    except Exception as e:
        print(f"❌ 创建仿真器失败: {e}")
    
    # 3. 创建调试管理器
    try:
        debug_manager = quick_debug_setup()
        print(f"✅ 创建调试管理器: {type(debug_manager).__name__}")
    except Exception as e:
        print(f"❌ 创建调试管理器失败: {e}")


def demo_debug_tools():
    """演示调试工具的使用"""
    print("\n=== 调试工具演示 ===")
    
    try:
        # 创建调试管理器
        debug_manager = quick_debug_setup()
        
        # 添加物理约束示例
        def darcy_equation(x, y):
            """Darcy方程残差"""
            return np.random.normal(0, 1e-6)  # 模拟残差
        
        def heat_equation(x, y):
            """热传导方程残差"""
            return np.random.normal(0, 1e-5)  # 模拟残差
        
        debug_manager.add_physical_constraint(
            name="Darcy方程",
            equation=darcy_equation,
            weight=1.0,
            tolerance=1e-6,
            description="Darcy流动方程约束"
        )
        
        debug_manager.add_physical_constraint(
            name="热传导方程",
            equation=heat_equation,
            weight=0.5,
            tolerance=1e-5,
            description="热传导方程约束"
        )
        
        print(f"✅ 添加物理约束: {len(debug_manager.monitor.constraints)} 个")
        
        # 创建监控仪表板
        debug_manager.create_dashboards()
        print("✅ 创建监控仪表板")
        
        # 获取调试摘要
        summary = debug_manager.get_debug_summary()
        print(f"✅ 调试摘要: {summary['total_constraints']} 个约束")
        
    except Exception as e:
        print(f"❌ 调试工具演示失败: {e}")


if __name__ == "__main__":
    # 运行演示
    demo_unified_api()
    demo_debug_tools()
    
    print("\n🎉 核心模块演示完成！")
    print("使用 help(core) 查看详细文档")

