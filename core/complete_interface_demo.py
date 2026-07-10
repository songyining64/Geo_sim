"""
完整接口演示模块
"""

import numpy as np
from typing import Optional


def demo_basic_simulation(config=None):
    """基础仿真演示"""
    print("基础仿真演示开始...")
    from .geodynamic_simulation import GeodynamicSimulation, GeodynamicConfig
    sim = GeodynamicSimulation(config)
    sim.create_mesh("rectangular", nx=10, ny=10)
    from ..materials import create_mantle_material
    sim.add_material(create_mantle_material())
    sim.setup_solver()
    sim.setup()
    print("基础仿真演示完成")
    return sim


def demo_adaptive_mesh_refinement():
    """自适应网格细化演示"""
    print("自适应网格细化演示开始...")
    from .geodynamic_simulation import GeodynamicSimulation
    sim = GeodynamicSimulation()
    sim.create_mesh("rectangular", nx=5, ny=5)
    from ..materials import create_mantle_material
    sim.add_material(create_mantle_material())
    sim.enable_adaptive_mesh_refinement()
    print("自适应网格细化演示完成")


def demo_gpu_acceleration():
    """GPU加速演示"""
    print("GPU加速演示开始...")
    try:
        from .geodynamic_simulation import GeodynamicSimulation
        sim = GeodynamicSimulation()
        sim.create_mesh("rectangular", nx=5, ny=5)
        from ..materials import create_mantle_material
        sim.add_material(create_mantle_material())
        sim.enable_gpu_acceleration()
        print("GPU加速演示完成")
    except Exception as e:
        print(f"GPU加速演示跳过: {e}")


def demo_visualization():
    """可视化演示"""
    print("可视化演示开始...")
    from .geodynamic_simulation import GeodynamicSimulation
    sim = GeodynamicSimulation()
    sim.create_mesh("rectangular", nx=5, ny=5)
    from ..materials import create_mantle_material
    sim.add_material(create_mantle_material())
    sim.setup_visualization()
    print("可视化演示完成")


def demo_multi_physics_coupling():
    """多物理场耦合演示"""
    print("多物理场耦合演示开始...")
    from .geodynamic_simulation import GeodynamicSimulation
    sim = GeodynamicSimulation()
    sim.create_mesh("rectangular", nx=5, ny=5)
    from ..materials import create_mantle_material
    sim.add_material(create_mantle_material())
    sim.setup_multi_physics_coupling()
    print("多物理场耦合演示完成")


def demo_complete_workflow():
    """完整工作流演示"""
    print("完整工作流演示开始...")
    from .geodynamic_simulation import GeodynamicSimulation, GeodynamicConfig
    config = GeodynamicConfig()
    sim = GeodynamicSimulation(config)
    sim.create_mesh("rectangular", nx=20, ny=20)
    from ..materials import create_mantle_material
    sim.add_material(create_mantle_material())
    sim.setup_solver()
    sim.setup()
    print("完整工作流演示完成")
    return sim


def main():
    """主函数 - 运行所有演示"""
    print("=" * 50)
    print("Geo_sim 完整接口演示")
    print("=" * 50)
    demo_basic_simulation()
    demo_adaptive_mesh_refinement()
    demo_gpu_acceleration()
    demo_visualization()
    demo_multi_physics_coupling()
    demo_complete_workflow()
    print("所有演示完成!")


if __name__ == "__main__":
    main()
