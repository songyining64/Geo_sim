"""
统一API和调试工具演示

展示如何使用GeoSim的统一接口和增强的调试工具，
包括场景配置模板、实时监控、错误诊断等功能。
"""

import numpy as np
import time
import warnings
from pathlib import Path

# 导入核心模块
from core import (
    SimulationConfig, create_simulator, load_scenario_template,
    create_simulation_with_debug, quick_debug_setup,
    PhysicalConstraint
)


def demo_scenario_templates():
    """演示场景配置模板的使用"""
    print("=== 场景配置模板演示 ===\n")
    
    try:
        # 1. 加载油气藏模拟配置
        print("1. 加载油气藏模拟配置...")
        reservoir_config = load_scenario_template('reservoir_simulation')
        print(f"   ✅ 配置名称: {reservoir_config.name}")
        print(f"   ✅ 描述: {reservoir_config.description}")
        print(f"   ✅ 时间步数: {reservoir_config.numerical_params['time_steps']}")
        print(f"   ✅ 时间步长: {reservoir_config.numerical_params['dt']} 秒")
        print(f"   ✅ 油粘度: {reservoir_config.physics_params['fluid']['oil_viscosity']} Pa·s")
        print(f"   ✅ 孔隙度: {reservoir_config.physics_params['rock']['porosity']}")
        
        # 2. 加载地震反演配置
        print("\n2. 加载地震反演配置...")
        seismic_config = load_scenario_template('seismic_inversion')
        print(f"   ✅ 配置名称: {seismic_config.name}")
        print(f"   ✅ 描述: {seismic_config.description}")
        print(f"   ✅ 最大迭代次数: {seismic_config.numerical_params['inversion']['max_iterations']}")
        print(f"   ✅ 正则化权重: {seismic_config.numerical_params['inversion']['regularization_weight']}")
        
        # 3. 修改配置参数
        print("\n3. 修改配置参数...")
        reservoir_config.numerical_params['time_steps'] = 500
        reservoir_config.physics_params['fluid']['oil_viscosity'] = 2.0e-3
        reservoir_config.output_params['save_frequency'] = 20
        
        print(f"   ✅ 时间步数已更新: {reservoir_config.numerical_params['time_steps']}")
        print(f"   ✅ 油粘度已更新: {reservoir_config.physics_params['fluid']['oil_viscosity']} Pa·s")
        print(f"   ✅ 保存频率已更新: {reservoir_config.output_params['save_frequency']}")
        
        # 4. 保存修改后的配置
        print("\n4. 保存修改后的配置...")
        output_dir = Path("./demo_output")
        output_dir.mkdir(exist_ok=True)
        
        reservoir_config.to_yaml(output_dir / "modified_reservoir_config.yaml")
        print(f"   ✅ 配置已保存到: {output_dir / 'modified_reservoir_config.yaml'}")
        
        return reservoir_config, seismic_config
        
    except Exception as e:
        print(f"   ❌ 场景配置模板演示失败: {e}")
        return None, None


def demo_unified_api():
    """演示统一API的使用"""
    print("\n=== 统一API演示 ===\n")
    
    try:
        # 1. 创建自定义配置
        print("1. 创建自定义配置...")
        custom_config = SimulationConfig(
            name="custom_heat_conduction",
            description="自定义热传导问题",
            physics_params={
                'thermal_diffusivity': 1e-6,
                'thermal_conductivity': 2.0,
                'specific_heat': 920.0
            },
            numerical_params={
                'time_steps': 100,
                'dt': 0.01,
                'tolerance': 1e-6,
                'max_iterations': 1000
            },
            output_params={
                'save_frequency': 10,
                'output_dir': './demo_output/heat_conduction',
                'save_format': 'h5'
            }
        )
        print(f"   ✅ 自定义配置已创建: {custom_config.name}")
        
        # 2. 创建不同类型的仿真器
        print("\n2. 创建不同类型的仿真器...")
        
        # 有限元仿真器
        fe_simulator = create_simulator('finite_element', custom_config)
        print(f"   ✅ 有限元仿真器: {type(fe_simulator).__name__}")
        
        # 多物理场仿真器
        mp_simulator = create_simulator('multi_physics', custom_config)
        print(f"   ✅ 多物理场仿真器: {type(mp_simulator).__name__}")
        
        # ML仿真器
        ml_simulator = create_simulator('ml', custom_config)
        print(f"   ✅ ML仿真器: {type(ml_simulator).__name__}")
        
        # 3. 检查仿真器状态
        print("\n3. 检查仿真器状态...")
        for name, simulator in [("有限元", fe_simulator), ("多物理场", mp_simulator), ("ML", ml_simulator)]:
            status = simulator.get_status()
            print(f"   {name}仿真器状态:")
            print(f"     - 已初始化: {status['is_initialized']}")
            print(f"     - 运行中: {status['is_running']}")
            print(f"     - 当前步骤: {status['current_step']}")
            print(f"     - 有结果: {status['has_result']}")
        
        return custom_config, fe_simulator, mp_simulator, ml_simulator
        
    except Exception as e:
        print(f"   ❌ 统一API演示失败: {e}")
        return None, None, None, None


def demo_debug_tools():
    """演示调试工具的使用"""
    print("\n=== 调试工具演示 ===\n")
    
    try:
        # 1. 创建调试管理器
        print("1. 创建调试管理器...")
        debug_manager = quick_debug_setup()
        print(f"   ✅ 调试管理器已创建: {type(debug_manager).__name__}")
        
        # 2. 添加物理约束
        print("\n2. 添加物理约束...")
        
        def darcy_equation(x, y):
            """Darcy方程残差 - 模拟"""
            return np.random.normal(0, 1e-6)
        
        def heat_equation(x, y):
            """热传导方程残差 - 模拟"""
            return np.random.normal(0, 1e-5)
        
        def mass_conservation(x, y):
            """质量守恒方程残差 - 模拟"""
            return np.random.normal(0, 1e-7)
        
        # 添加约束
        constraints = [
            ("Darcy方程", darcy_equation, 1.0, 1e-6, "Darcy流动方程约束"),
            ("热传导方程", heat_equation, 0.5, 1e-5, "热传导方程约束"),
            ("质量守恒", mass_conservation, 1.0, 1e-7, "质量守恒方程约束")
        ]
        
        for name, equation, weight, tolerance, description in constraints:
            debug_manager.add_physical_constraint(
                name=name,
                equation=equation,
                weight=weight,
                tolerance=tolerance,
                description=description
            )
            print(f"   ✅ 已添加约束: {name}")
        
        # 3. 创建监控仪表板
        print("\n3. 创建监控仪表板...")
        debug_manager.create_dashboards()
        print("   ✅ 监控仪表板已创建")
        
        # 4. 启动调试监控
        print("\n4. 启动调试监控...")
        debug_manager.start_debugging()
        print("   ✅ 调试监控已启动")
        
        # 5. 模拟一些监控数据
        print("\n5. 模拟监控数据...")
        for i in range(5):
            time.sleep(0.5)  # 等待监控数据更新
            summary = debug_manager.get_debug_summary()
            print(f"   步骤 {i+1}: {summary['monitoring_summary']['total_communications']} 次通信")
        
        # 6. 停止调试监控
        print("\n6. 停止调试监控...")
        debug_manager.stop_debugging()
        print("   ✅ 调试监控已停止")
        
        # 7. 获取最终摘要
        print("\n7. 获取调试摘要...")
        final_summary = debug_manager.get_debug_summary()
        print(f"   ✅ 调试摘要:")
        print(f"     - 调试活跃: {final_summary['debug_active']}")
        print(f"     - 总约束数: {final_summary['total_constraints']}")
        print(f"     - 总错误数: {final_summary['error_summary']['total_errors']}")
        
        return debug_manager
        
    except Exception as e:
        print(f"   ❌ 调试工具演示失败: {e}")
        return None


def demo_error_diagnosis():
    """演示错误诊断功能"""
    print("\n=== 错误诊断演示 ===\n")
    
    try:
        # 1. 创建调试管理器
        debug_manager = quick_debug_setup()
        
        # 2. 模拟不同类型的错误
        print("1. 模拟不同类型的错误...")
        
        # 模拟网格质量错误
        try:
            raise ValueError("网格质量差：存在长宽比大于10的单元")
        except Exception as e:
            print("   模拟网格质量错误...")
            error_info = debug_manager.diagnose_error(e, context={'mesh': 'test_mesh'})
            print(f"   ✅ 错误诊断完成")
            print(f"      - 错误类型: {error_info['error_type']}")
            print(f"      - 诊断结果: {len(error_info['diagnosis'])} 个")
            print(f"      - 修复建议: {len(error_info['suggestions'])} 个")
        
        # 模拟数值稳定性错误
        try:
            raise RuntimeError("数值不稳定：时间步长过大导致发散")
        except Exception as e:
            print("\n   模拟数值稳定性错误...")
            error_info = debug_manager.diagnose_error(e, context={'solver': 'explicit'})
            print(f"   ✅ 错误诊断完成")
            print(f"      - 错误类型: {error_info['error_type']}")
            print(f"      - 诊断结果: {len(error_info['diagnosis'])} 个")
            print(f"      - 修复建议: {len(error_info['suggestions'])} 个")
        
        # 模拟内存错误
        try:
            raise MemoryError("内存不足：网格规模过大")
        except Exception as e:
            print("\n   模拟内存错误...")
            error_info = debug_manager.diagnose_error(e, context={'mesh_size': 'large'})
            print(f"   ✅ 错误诊断完成")
            print(f"      - 错误类型: {error_info['error_type']}")
            print(f"      - 诊断结果: {len(error_info['diagnosis'])} 个")
            print(f"      - 修复建议: {len(error_info['suggestions'])} 个")
        
        # 3. 获取错误摘要
        print("\n2. 获取错误摘要...")
        error_summary = debug_manager.diagnostic.get_error_summary()
        print(f"   ✅ 错误摘要:")
        print(f"     - 总错误数: {error_summary['total_errors']}")
        print(f"     - 错误类型分布: {error_summary['error_types']}")
        print(f"     - 常见问题: {len(error_summary['common_issues'])} 个")
        
        return debug_manager
        
    except Exception as e:
        print(f"   ❌ 错误诊断演示失败: {e}")
        return None


def demo_integrated_workflow():
    """演示集成工作流程"""
    print("\n=== 集成工作流程演示 ===\n")
    
    try:
        # 1. 加载场景配置
        print("1. 加载场景配置...")
        config = load_scenario_template('reservoir_simulation')
        print(f"   ✅ 配置已加载: {config.name}")
        
        # 2. 创建带调试功能的仿真器
        print("\n2. 创建带调试功能的仿真器...")
        simulator, debug_manager = create_simulation_with_debug('multi_physics', config)
        print(f"   ✅ 仿真器已创建: {type(simulator).__name__}")
        print(f"   ✅ 调试管理器已创建: {type(debug_manager).__name__}")
        
        # 3. 添加物理约束
        print("\n3. 添加物理约束...")
        
        def pressure_constraint(x, y):
            """压力约束：不能为负值"""
            return np.maximum(0, x) - x
        
        def saturation_constraint(x, y):
            """饱和度约束：必须在[0,1]范围内"""
            return np.clip(x, 0, 1) - x
        
        debug_manager.add_physical_constraint(
            name="压力非负约束",
            equation=pressure_constraint,
            weight=1.0,
            tolerance=1e-6,
            description="压力值不能为负"
        )
        
        debug_manager.add_physical_constraint(
            name="饱和度范围约束",
            equation=saturation_constraint,
            weight=0.8,
            tolerance=1e-6,
            description="饱和度必须在[0,1]范围内"
        )
        
        print(f"   ✅ 已添加 {len(debug_manager.monitor.constraints)} 个物理约束")
        
        # 4. 启动调试监控
        print("\n4. 启动调试监控...")
        debug_manager.start_debugging()
        debug_manager.create_dashboards()
        print("   ✅ 调试监控已启动")
        
        # 5. 模拟仿真过程
        print("\n5. 模拟仿真过程...")
        for step in range(10):
            time.sleep(0.3)  # 模拟计算时间
            print(f"   步骤 {step+1}/10 完成")
            
            # 模拟一些约束违反
            if step % 3 == 0:
                print(f"     ⚠️  检测到约束违反")
        
        # 6. 停止调试监控
        print("\n6. 停止调试监控...")
        debug_manager.stop_debugging()
        print("   ✅ 调试监控已停止")
        
        # 7. 获取最终结果
        print("\n7. 获取最终结果...")
        final_summary = debug_manager.get_debug_summary()
        print(f"   ✅ 工作流程完成:")
        print(f"     - 总约束数: {final_summary['total_constraints']}")
        print(f"     - 监控状态: {'活跃' if final_summary['monitoring_summary']['monitoring_active'] else '停止'}")
        print(f"     - 总错误数: {final_summary['error_summary']['total_errors']}")
        
        return simulator, debug_manager
        
    except Exception as e:
        print(f"   ❌ 集成工作流程演示失败: {e}")
        return None, None


def demo_config_management():
    """演示配置管理功能"""
    print("\n=== 配置管理演示 ===\n")
    
    try:
        # 1. 创建配置
        print("1. 创建配置...")
        config = SimulationConfig(
            name="demo_config",
            description="演示配置",
            version="1.0.0",
            physics_params={
                'gravity': 9.81,
                'thermal_expansion': 3e-5,
                'young_modulus': 20e9
            },
            numerical_params={
                'time_steps': 200,
                'dt': 0.005,
                'tolerance': 1e-8
            },
            boundary_conditions={
                'temperature': {'top': 400, 'bottom': 300},
                'pressure': {'left': 'fixed', 'right': 'free'}
            },
            output_params={
                'save_frequency': 20,
                'output_dir': './demo_output/config_demo',
                'save_format': 'h5'
            }
        )
        print(f"   ✅ 配置已创建: {config.name}")
        
        # 2. 配置验证
        print("\n2. 配置验证...")
        print(f"   ✅ 物理参数: {len(config.physics_params)} 个")
        print(f"   ✅ 数值参数: {len(config.numerical_params)} 个")
        print(f"   ✅ 边界条件: {len(config.boundary_conditions)} 个")
        print(f"   ✅ 输出参数: {len(config.output_params)} 个")
        
        # 3. 配置导出
        print("\n3. 配置导出...")
        output_dir = Path("./demo_output")
        output_dir.mkdir(exist_ok=True)
        
        # 导出为YAML
        yaml_path = output_dir / "demo_config.yaml"
        config.to_yaml(str(yaml_path))
        print(f"   ✅ YAML配置已保存: {yaml_path}")
        
        # 导出为JSON
        json_path = output_dir / "demo_config.json"
        config.to_json(str(json_path))
        print(f"   ✅ JSON配置已保存: {json_path}")
        
        # 4. 配置修改
        print("\n4. 配置修改...")
        config.physics_params['gravity'] = 9.82
        config.numerical_params['time_steps'] = 300
        config.output_params['save_frequency'] = 15
        
        print(f"   ✅ 重力已更新: {config.physics_params['gravity']} m/s²")
        print(f"   ✅ 时间步数已更新: {config.numerical_params['time_steps']}")
        print(f"   ✅ 保存频率已更新: {config.output_params['save_frequency']}")
        
        # 5. 配置比较
        print("\n5. 配置比较...")
        config_dict = config.to_dict()
        print(f"   ✅ 配置字典键数: {len(config_dict)}")
        print(f"   ✅ 配置版本: {config_dict['version']}")
        
        return config
        
    except Exception as e:
        print(f"   ❌ 配置管理演示失败: {e}")
        return None


def main():
    """主演示函数"""
    print("🚀 GeoSim 统一API和调试工具演示")
    print("=" * 60)
    
    try:
        # 运行各个演示
        reservoir_config, seismic_config = demo_scenario_templates()
        
        custom_config, fe_sim, mp_sim, ml_sim = demo_unified_api()
        
        debug_manager = demo_debug_tools()
        
        error_diagnostic = demo_error_diagnosis()
        
        simulator, debug_mgr = demo_integrated_workflow()
        
        config_manager = demo_config_management()
        
        print("\n" + "=" * 60)
        print("🎉 所有演示完成！")
        print("\n📚 学习要点:")
        print("1. 使用场景配置模板快速启动仿真")
        print("2. 统一API简化了不同模块的使用")
        print("3. 调试工具提供实时监控和错误诊断")
        print("4. 配置管理支持灵活的参数调整")
        print("5. 集成工作流程提高开发效率")
        
        print("\n🔧 下一步:")
        print("1. 尝试修改配置参数")
        print("2. 添加自定义物理约束")
        print("3. 运行实际的仿真案例")
        print("4. 探索更多高级功能")
        
        print("\n📁 输出文件:")
        demo_output = Path("./demo_output")
        if demo_output.exists():
            for file in demo_output.glob("*"):
                print(f"   - {file}")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
