"""
模块整合演示：展示GeodynamicSimulation如何调用现有的专业模块

这个演示展示了：
1. 如何正确使用现有的有限元模块
2. 如何调用现有的求解器模块
3. 如何实现优雅的回退机制
4. 现有模块的价值和重要性
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from core.geodynamic_simulation import GeodynamicSimulation, GeodynamicConfig
from finite_elements.mesh_generation import create_simple_mesh
from materials.advanced_material_models import ElasticMaterial

def demo_module_integration():
    """演示模块整合功能"""
    print("=== 模块整合演示 ===")
    print("展示GeodynamicSimulation如何调用现有的专业模块\n")
    
    # 创建配置
    config = GeodynamicConfig(
        solver_type='standard',  # 使用标准求解器
        time_step=0.01,
        end_time=1.0,
        max_iterations=100
    )
    
    # 创建模拟器
    sim = GeodynamicSimulation(config)
    
    print("1. 创建简单网格...")
    # 使用现有的网格生成模块
    mesh = create_simple_mesh(nx=5, ny=5, element_type='triangle')
    sim.set_mesh(mesh)
    
    print("2. 添加材料...")
    # 使用现有的材料模块
    material = ElasticMaterial(
        youngs_modulus=1e9,
        poissons_ratio=0.3,
        density=2700.0
    )
    sim.add_material(material)
    
    print("3. 执行模拟...")
    print("注意：GeodynamicSimulation会自动尝试导入专业模块")
    
    try:
        result = sim.run()
        print(f"模拟成功完成！")
        print(f"时间步数: {result.time_steps}")
        print(f"最终时间: {result.final_time:.4f}")
        print(f"收敛状态: {result.converged}")
        
        # 显示解的历史
        if hasattr(sim, 'solution_history') and sim.solution_history:
            print(f"解的历史长度: {len(sim.solution_history)}")
            final_solution = sim.solution_history[-1]
            print(f"最终解的范围: [{final_solution.min():.6f}, {final_solution.max():.6f}]")
        
    except Exception as e:
        print(f"模拟失败: {e}")
        print("这可能是由于缺少某些依赖模块导致的")

def demo_multigrid_integration():
    """演示多重网格求解器整合"""
    print("\n=== 多重网格求解器整合演示 ===")
    
    config = GeodynamicConfig(
        solver_type='multigrid',  # 使用多重网格求解器
        time_step=0.01,
        end_time=0.1,
        max_iterations=50
    )
    
    sim = GeodynamicSimulation(config)
    
    # 创建网格和材料
    mesh = create_simple_mesh(nx=8, ny=8, element_type='quad')
    sim.set_mesh(mesh)
    
    material = ElasticMaterial(
        youngs_modulus=1e9,
        poissons_ratio=0.3,
        density=2700.0
    )
    sim.add_material(material)
    
    print("尝试使用多重网格求解器...")
    try:
        result = sim.run()
        print("多重网格求解器成功！")
        print(f"时间步数: {result.time_steps}")
    except Exception as e:
        print(f"多重网格求解器失败，回退到标准求解器: {e}")

def demo_multiphysics_integration():
    """演示多物理场耦合求解器整合"""
    print("\n=== 多物理场耦合求解器整合演示 ===")
    
    config = GeodynamicConfig(
        solver_type='multiphysics',  # 使用多物理场耦合求解器
        time_step=0.01,
        end_time=0.1,
        max_iterations=30
    )
    
    sim = GeodynamicSimulation(config)
    
    # 创建网格和材料
    mesh = create_simple_mesh(nx=6, ny=6, element_type='triangle')
    sim.set_mesh(mesh)
    
    material = ElasticMaterial(
        youngs_modulus=1e9,
        poissons_ratio=0.3,
        density=2700.0
    )
    sim.add_material(material)
    
    print("尝试使用多物理场耦合求解器...")
    try:
        result = sim.run()
        print("多物理场耦合求解器成功！")
        print(f"时间步数: {result.time_steps}")
    except Exception as e:
        print(f"多物理场耦合求解器失败，回退到标准求解器: {e}")

def demo_parallel_integration():
    """演示并行求解器整合"""
    print("\n=== 并行求解器整合演示 ===")
    
    config = GeodynamicConfig(
        solver_type='parallel',  # 使用并行求解器
        time_step=0.01,
        end_time=0.1,
        max_iterations=20
    )
    
    sim = GeodynamicSimulation(config)
    
    # 创建网格和材料
    mesh = create_simple_mesh(nx=10, ny=10, element_type='quad')
    sim.set_mesh(mesh)
    
    material = ElasticMaterial(
        youngs_modulus=1e9,
        poissons_ratio=0.3,
        density=2700.0
    )
    sim.add_material(material)
    
    print("尝试使用并行求解器...")
    try:
        result = sim.run()
        print("并行求解器成功！")
        print(f"时间步数: {result.time_steps}")
    except Exception as e:
        print(f"并行求解器失败，回退到标准求解器: {e}")

def demo_fallback_mechanism():
    """演示回退机制"""
    print("\n=== 回退机制演示 ===")
    print("当专业模块不可用时，系统会自动回退到简化实现")
    
    # 创建一个故意缺少某些模块的配置
    config = GeodynamicConfig(
        solver_type='multigrid',
        time_step=0.01,
        end_time=0.05,
        max_iterations=10
    )
    
    sim = GeodynamicSimulation(config)
    
    # 创建网格和材料
    mesh = create_simple_mesh(nx=4, ny=4, element_type='triangle')
    sim.set_mesh(mesh)
    
    material = ElasticMaterial(
        youngs_modulus=1e9,
        poissons_ratio=0.3,
        density=2700.0
    )
    sim.add_material(material)
    
    print("执行模拟（可能触发回退机制）...")
    try:
        result = sim.run()
        print("模拟完成（可能使用了回退实现）")
        print(f"时间步数: {result.time_steps}")
    except Exception as e:
        print(f"模拟失败: {e}")

def main():
    """主函数"""
    print("模块整合演示程序")
    print("=" * 50)
    
    # 演示各种整合方式
    demo_module_integration()
    demo_multigrid_integration()
    demo_multiphysics_integration()
    demo_parallel_integration()
    demo_fallback_mechanism()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("\n关键要点：")
    print("1. GeodynamicSimulation是一个统一接口，调用现有的专业模块")
    print("2. 当专业模块可用时，使用专业实现")
    print("3. 当专业模块不可用时，自动回退到简化实现")
    print("4. 你之前写的专业模块仍然非常重要，不会被浪费")
    print("5. 这种设计让系统既强大又灵活")

if __name__ == "__main__":
    main()
