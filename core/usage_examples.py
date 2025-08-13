"""
统一接口使用示例

提供GeodynamicSimulation类在实际项目中的使用示例，
包括地幔对流、岩石圈变形、多物理场耦合等典型应用。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入统一接口
from .geodynamic_simulation import (
    GeodynamicSimulation, 
    GeodynamicConfig,
    create_mantle_convection_simulation,
    create_lithospheric_deformation_simulation
)

# 导入边界条件
from ..finite_elements.boundary_conditions import DirichletBC, NeumannBC, RobinBC

# 导入材料
from ..materials import create_mantle_material, create_crust_material, create_air_material


def example_mantle_convection_2d():
    """
    示例1: 2D地幔对流模拟
    
    模拟地幔中的热对流过程，包括温度场和速度场的演化。
    """
    print("=" * 60)
    print("示例1: 2D地幔对流模拟")
    print("=" * 60)
    
    # 使用便捷函数创建地幔对流模拟
    sim = create_mantle_convection_simulation(
        nx=40, ny=40, aspect_ratio=2.0
    )
    
    # 自定义配置
    sim.config.numerical_params.update({
        'time_steps': 100,
        'dt': 0.01,
        'tolerance': 1e-6
    })
    
    sim.config.output_params.update({
        'save_frequency': 10,
        'output_dir': './output/mantle_convection_2d'
    })
    
    # 设置求解器参数
    sim.config.solver_params.update({
        'linear_solver': 'multigrid',
        'preconditioner': 'amg',
        'tolerance': 1e-8,
        'max_iterations': 1000,
        'multigrid_cycles': 'v',
        'smoother': 'gauss_seidel'
    })
    
    # 设置时间积分参数
    sim.config.time_integration.update({
        'method': 'bdf',
        'order': 2,
        'adaptive_timestep': True,
        'min_dt': 1e-6,
        'max_dt': 1e2,
        'error_tolerance': 1e-3
    })
    
    # 初始化仿真环境
    sim.initialize()
    
    print("地幔对流模拟设置完成:")
    print(f"  - 网格尺寸: 40x40")
    print(f"  - 长宽比: 2.0")
    print(f"  - 时间步数: 100")
    print(f"  - 求解器: 多重网格 (V循环)")
    print(f"  - 时间积分: 2阶BDF + 自适应步长")
    
    # 运行仿真（这里只是演示，实际需要实现求解逻辑）
    print("\n开始运行仿真...")
    try:
        # 模拟运行过程
        for step in range(10):  # 只运行前10步作为演示
            print(f"  时间步 {step + 1}/10")
            # 这里应该调用实际的求解逻辑
        
        print("仿真运行完成（演示模式）")
        
        # 导出结果
        output_dir = Path("./output/mantle_convection_2d")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模拟一些结果数据
        n_nodes = len(sim.mesh.nodes)
        sim.temperature_field = np.random.uniform(273.15, 1573.15, n_nodes)
        sim.velocity_field = np.random.uniform(-1e-6, 1e-6, (n_nodes, 2))
        sim.pressure_field = np.random.uniform(1e5, 1e8, n_nodes)
        
        # 导出VTK格式
        try:
            sim.export_vtk(str(output_dir / "mantle_convection.vtk"))
            print(f"  - VTK文件已导出")
        except ImportError:
            print("  - VTK导出失败: VTK库不可用")
        
        # 导出HDF5格式
        try:
            sim.export_hdf5(str(output_dir / "mantle_convection.h5"))
            print(f"  - HDF5文件已导出")
        except ImportError:
            print("  - HDF5导出失败: h5py库不可用")
        
        return sim
        
    except Exception as e:
        print(f"仿真运行失败: {e}")
        return None


def example_lithospheric_deformation():
    """
    示例2: 岩石圈变形模拟
    
    模拟岩石圈在构造应力作用下的变形过程，包括应变场和应力场的演化。
    """
    print("\n" + "=" * 60)
    print("示例2: 岩石圈变形模拟")
    print("=" * 60)
    
    # 创建岩石圈变形模拟
    sim = create_lithospheric_deformation_simulation(nx=30, ny=30)
    
    # 自定义配置
    sim.config.numerical_params.update({
        'time_steps': 200,
        'dt': 1e6,  # 1 Ma
        'tolerance': 1e-8
    })
    
    sim.config.output_params.update({
        'save_frequency': 20,
        'output_dir': './output/lithospheric_deformation'
    })
    
    # 设置求解器参数
    sim.config.solver_params.update({
        'linear_solver': 'multigrid',
        'preconditioner': 'amg',
        'tolerance': 1e-10,
        'max_iterations': 2000,
        'multigrid_cycles': 'w',
        'smoother': 'chebyshev'
    })
    
    # 设置时间积分参数
    sim.config.time_integration.update({
        'method': 'bdf',
        'order': 3,
        'adaptive_timestep': True,
        'min_dt': 1e5,   # 0.1 Ma
        'max_dt': 1e7,   # 10 Ma
        'error_tolerance': 1e-4
    })
    
    # 初始化仿真环境
    sim.initialize()
    
    print("岩石圈变形模拟设置完成:")
    print(f"  - 网格尺寸: 30x30")
    print(f"  - 时间步长: 1 Ma")
    print(f"  - 时间步数: 200")
    print(f"  - 求解器: 多重网格 (W循环)")
    print(f"  - 时间积分: 3阶BDF + 自适应步长")
    
    # 运行仿真（演示模式）
    print("\n开始运行仿真...")
    try:
        for step in range(10):  # 只运行前10步作为演示
            print(f"  时间步 {step + 1}/10")
        
        print("仿真运行完成（演示模式）")
        
        # 导出结果
        output_dir = Path("./output/lithospheric_deformation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模拟结果数据
        n_nodes = len(sim.mesh.nodes)
        sim.temperature_field = np.random.uniform(273.15, 1273.15, n_nodes)
        sim.velocity_field = np.random.uniform(-1e-9, 1e-9, (n_nodes, 2))
        sim.pressure_field = np.random.uniform(1e5, 1e9, n_nodes)
        
        # 导出结果
        try:
            sim.export_vtk(str(output_dir / "lithospheric_deformation.vtk"))
            print(f"  - VTK文件已导出")
        except ImportError:
            print("  - VTK导出失败: VTK库不可用")
        
        return sim
        
    except Exception as e:
        print(f"仿真运行失败: {e}")
        return None


def example_multi_physics_coupling():
    """
    示例3: 多物理场耦合模拟
    
    模拟热-力学耦合过程，包括温度场、位移场和应力场的相互影响。
    """
    print("\n" + "=" * 60)
    print("示例3: 多物理场耦合模拟")
    print("=" * 60)
    
    # 创建自定义配置
    config = GeodynamicConfig(
        name="multi_physics_coupling",
        description="热-力学耦合模拟",
        material_properties={
            'density': 3000.0,
            'viscosity': 5e21,
            'thermal_expansion': 2.5e-5,
            'thermal_conductivity': 2.8,
            'specific_heat': 1100.0,
            'young_modulus': 50e9,
            'poisson_ratio': 0.25
        },
        solver_params={
            'linear_solver': 'multiphysics',
            'preconditioner': 'amg',
            'tolerance': 1e-8,
            'max_iterations': 1500,
            'coupling_strategy': 'monolithic'
        },
        time_integration={
            'method': 'bdf',
            'order': 2,
            'adaptive_timestep': True,
            'error_tolerance': 1e-4
        }
    )
    
    # 创建仿真对象
    sim = GeodynamicSimulation(config)
    
    # 创建三角形网格
    sim.create_mesh("triangular", nx=20, ny=20, 
                    x_range=(0.0, 1000.0), y_range=(0.0, 500.0))
    
    # 添加多种材料
    mantle_material = create_mantle_material()
    crust_material = create_crust_material()
    
    sim.add_material(mantle_material)
    sim.add_material(crust_material)
    
    # 添加复杂边界条件
    # 时间相关的温度边界条件
    def time_dependent_temp(t):
        return 273.15 + 50 * np.sin(2 * np.pi * t / 1e7)
    
    sim.add_boundary_condition(
        DirichletBC(0, "temperature", time_dependent_temp, is_time_dependent=True), "top"
    )
    
    # Robin边界条件（混合边界条件）
    sim.add_boundary_condition(
        RobinBC(1, "temperature", alpha=1.0, beta=0.1, gamma=1273.15), "bottom"
    )
    
    # 位移边界条件
    sim.add_boundary_condition(
        DirichletBC(2, "displacement", 0.0), "left"
    )
    sim.add_boundary_condition(
        DirichletBC(3, "displacement", 0.0), "right"
    )
    
    # 设置多物理场求解器
    sim.setup_solver("multiphysics", 
                     coupling_config={
                         'thermal_mechanical': True,
                         'coupling_iterations': 3,
                         'coupling_tolerance': 1e-6
                     })
    
    # 设置自适应时间积分器
    sim.setup_time_integrator("bdf", 
                              order=2,
                              adaptive_timestep=True,
                              error_tolerance=1e-4)
    
    # 初始化
    sim.initialize()
    
    print("多物理场耦合模拟设置完成:")
    print(f"  - 网格类型: 三角形网格")
    print(f"  - 材料数量: {len(sim.materials)} 种")
    print(f"  - 边界条件: {len(sim.boundary_conditions)} 个")
    print(f"  - 求解器: 多物理场耦合求解器")
    print(f"  - 耦合策略: 整体求解")
    print(f"  - 时间积分: 2阶BDF + 自适应步长")
    
    # 运行仿真（演示模式）
    print("\n开始运行仿真...")
    try:
        for step in range(10):  # 只运行前10步作为演示
            print(f"  时间步 {step + 1}/10")
        
        print("仿真运行完成（演示模式）")
        
        # 导出结果
        output_dir = Path("./output/multi_physics_coupling")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模拟结果数据
        n_nodes = len(sim.mesh.nodes)
        sim.temperature_field = np.random.uniform(273.15, 1273.15, n_nodes)
        sim.velocity_field = np.random.uniform(-1e-8, 1e-8, (n_nodes, 2))
        sim.pressure_field = np.random.uniform(1e5, 1e8, n_nodes)
        
        # 导出结果
        try:
            sim.export_vtk(str(output_dir / "multi_physics_coupling.vtk"))
            print(f"  - VTK文件已导出")
        except ImportError:
            print("  - VTK导出失败: VTK库不可用")
        
        return sim
        
    except Exception as e:
        print(f"仿真运行失败: {e}")
        return None


def example_parallel_computing():
    """
    示例4: 并行计算模拟
    
    展示如何使用并行求解器处理大规模问题。
    """
    print("\n" + "=" * 60)
    print("示例4: 并行计算模拟")
    print("=" * 60)
    
    # 创建大规模网格的仿真
    sim = GeodynamicSimulation()
    
    # 创建大规模网格
    sim.create_mesh("rectangular", nx=100, ny=100, 
                    x_range=(0.0, 2000.0), y_range=(0.0, 1000.0))
    
    # 添加材料
    mantle_material = create_mantle_material()
    sim.add_material(mantle_material)
    
    # 设置边界条件
    sim.add_boundary_condition(
        DirichletBC(0, "temperature", 273.15), "top"
    )
    sim.add_boundary_condition(
        DirichletBC(1, "temperature", 1573.15), "bottom"
    )
    sim.add_boundary_condition(
        NeumannBC(2, "velocity", 0.0), "left"
    )
    sim.add_boundary_condition(
        NeumannBC(3, "velocity", 0.0), "right"
    )
    
    # 设置并行求解器
    sim.setup_solver("parallel", 
                     parallel_config={
                         'n_processes': 4,
                         'use_gpu': True,
                         'mixed_precision': True,
                         'load_balancing': True,
                         'communication_optimization': True
                     })
    
    # 设置时间积分器
    sim.setup_time_integrator("bdf", order=2)
    
    # 初始化
    sim.initialize()
    
    print("并行计算模拟设置完成:")
    print(f"  - 网格尺寸: 100x100 ({len(sim.mesh.nodes)} 节点)")
    print(f"  - 求解器: 并行求解器")
    print(f"  - 进程数: 4")
    print(f"  - GPU加速: 是")
    print(f"  - 混合精度: 是")
    print(f"  - 负载均衡: 是")
    
    # 运行仿真（演示模式）
    print("\n开始运行仿真...")
    try:
        for step in range(5):  # 只运行前5步作为演示
            print(f"  时间步 {step + 1}/5")
        
        print("仿真运行完成（演示模式）")
        
        # 获取性能摘要
        performance = sim.get_performance_summary()
        if performance:
            print("\n性能摘要:")
            for key, value in performance.items():
                print(f"  - {key}: {value}")
        
        return sim
        
    except Exception as e:
        print(f"仿真运行失败: {e}")
        return None


def run_all_examples():
    """运行所有示例"""
    print("地质动力学仿真统一接口使用示例")
    print("=" * 80)
    
    simulations = []
    
    try:
        # 运行所有示例
        sim1 = example_mantle_convection_2d()
        if sim1:
            simulations.append(("地幔对流2D", sim1))
        
        sim2 = example_lithospheric_deformation()
        if sim2:
            simulations.append(("岩石圈变形", sim2))
        
        sim3 = example_multi_physics_coupling()
        if sim3:
            simulations.append(("多物理场耦合", sim3))
        
        sim4 = example_parallel_computing()
        if sim4:
            simulations.append(("并行计算", sim4))
        
        print("\n" + "=" * 80)
        print("所有示例运行完成！")
        print("=" * 80)
        
        if simulations:
            print(f"\n成功创建了 {len(simulations)} 个仿真对象:")
            for name, sim in simulations:
                print(f"  - {name}: {len(sim.mesh.nodes)} 节点")
            
            print("\n使用建议:")
            print("1. 这些示例展示了不同复杂度的仿真设置")
            print("2. 可以根据实际需求修改配置参数")
            print("3. 实际运行时需要实现具体的求解逻辑")
            print("4. 导出的VTK文件可以在ParaView中查看")
            print("5. 使用内置性能监控优化仿真参数")
        
        return simulations
        
    except Exception as e:
        print(f"\n示例运行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    # 运行所有示例
    simulations = run_all_examples()
    
    # 保存示例结果
    if simulations:
        print(f"\n示例结果已保存到相应的输出目录")
        print("可以在后续代码中使用这些仿真对象进行实际的仿真计算")
