# GeoSim 快速入门指南

## 快速开始

### 1. 安装

```bash
# 安装基础依赖
pip install numpy scipy matplotlib

# 安装机器学习依赖（推荐）
pip install torch scikit-learn

# 安装并行计算依赖（可选）
pip install mpi4py

# 安装GPU加速依赖（可选）
pip install cupy pyopencl
```

```bash
# 克隆项目
git clone <repository-url>
cd geo_sim

# 安装依赖
pip install -r requirements.txt
```

### 2. 基本使用

#### 机器学习加速（推荐）

```python
import numpy as np
from gpu_acceleration.geological_ml_framework import GeologicalPINN, GeologicalConfig

# 创建地质PINN
geological_config = GeologicalConfig(porosity=0.2, has_faults=True)
pinn = GeologicalPINN(input_dim=4, hidden_dims=[64, 32], output_dim=1, 
                     geological_config=geological_config)

# 训练模型
X = np.random.randn(1000, 4)
y = np.random.randn(1000, 1)
result = pinn.train(X, y, epochs=200)
print(f"训练完成，最终损失: {result['total_loss'][-1]:.6f}")

# 预测
predictions = pinn.predict(X[:10])
print(f"预测结果形状: {predictions.shape}")
```

#### 代理模型

```python
from gpu_acceleration.geological_ml_framework import GeologicalSurrogateModel

# 创建代理模型
surrogate = GeologicalSurrogateModel('gaussian_process')
surrogate.train(X, y.flatten())

# 快速预测
predictions, std = surrogate.predict(X[:10], return_std=True)
print(f"预测结果: {predictions.shape}, 不确定性: {std.shape}")
```

#### 并行求解器

```python
import numpy as np
import scipy.sparse as sp
from parallel.advanced_parallel_solver import create_parallel_solver, ParallelConfig

# 创建求解器
config = ParallelConfig(solver_type='cg', max_iterations=1000, tolerance=1e-8)
solver = create_parallel_solver('cg', config)

# 创建测试问题
n = 1000
A = sp.random(n, n, density=0.01, format='csr')
A = A + A.T + sp.eye(n)
b = np.random.rand(n)

# 求解
x = solver.solve(A, b)
print(f"求解完成，残差: {solver.get_performance_stats().residual_norm:.2e}")
```

#### 热-力学耦合

```python
from coupling.thermal_mechanical import ThermoMechanicalCoupling, CouplingConfig

# 创建简单网格
class SimpleMesh:
    def __init__(self, n_points):
        self.nodes = np.linspace(0, 1, n_points)
        self.n_points = n_points

mesh = SimpleMesh(50)

# 创建耦合求解器
config = CouplingConfig(solver_type='iterative', max_iterations=10, tolerance=1e-6)
coupling = ThermoMechanicalCoupling(mesh, config=config)

# 定义问题
initial_temperature = np.ones(50) * 293.15
initial_displacement = np.zeros(50)
boundary_conditions = {
    'thermal': {0: 373.15, 49: 293.15},
    'mechanical': {0: 0.0, 49: 0.0}
}

# 求解
solution_history = coupling.solve_coupled_system(
    initial_temperature, initial_displacement,
    boundary_conditions, time_steps=20, dt=1e-3
)

print(f"求解完成，共 {len(solution_history)} 个时间步")
```

#### 流体-固体耦合

```python
from coupling.fluid_solid import create_fluid_solid_coupling, FSIConfig, FluidSolidState

# 创建网格
class SimpleMesh:
    def __init__(self, n_points):
        self.nodes = np.linspace(0, 1, n_points)
        self.n_points = n_points

fluid_mesh = SimpleMesh(30)
solid_mesh = SimpleMesh(20)
interface_nodes = np.arange(15, 20)

# 创建求解器
config = FSIConfig(solver_type='partitioned', max_iterations=10, tolerance=1e-6)
fsi_solver = create_fluid_solid_coupling(fluid_mesh, solid_mesh, interface_nodes, config=config)

# 创建初始状态
initial_state = FluidSolidState(
    fluid_velocity=np.zeros(30),
    fluid_pressure=np.zeros(30),
    fluid_temperature=np.ones(30) * 293.15,
    solid_displacement=np.zeros(20),
    solid_velocity=np.zeros(20),
    solid_stress=np.zeros((20, 3)),
    interface_force=np.zeros((5, 2)),
    interface_displacement=np.zeros((5, 2)),
    mesh_deformation=np.zeros((30, 2))
)

# 定义边界条件
boundary_conditions = {
    'fluid': {0: 1.0, 29: 0.0},
    'solid': {0: 0.0, 19: 0.0}
}

# 求解
solution_history = fsi_solver.solve_coupled_system(
    initial_state, boundary_conditions, time_steps=20, dt=1e-3
)

print(f"FSI求解完成，共 {len(solution_history)} 个时间步")
```

#### 化学-力学耦合

```python
from coupling.chemical_mechanical import create_chemical_mechanical_coupling, ChemicalMechanicalConfig

# 创建网格
class SimpleMesh:
    def __init__(self, n_points):
        self.nodes = np.linspace(0, 1, n_points)
        self.n_points = n_points

mesh = SimpleMesh(50)

# 创建求解器
config = ChemicalMechanicalConfig(solver_type='iterative', max_iterations=10, tolerance=1e-6)
coupling = create_chemical_mechanical_coupling(mesh, config=config)

# 定义问题
initial_concentration = np.ones(50) * 0.1
initial_displacement = np.zeros(50)
boundary_conditions = {
    'chemical': {0: 1.0, 49: 0.0},
    'mechanical': {0: 0.0, 49: 0.0}
}

# 定义源项
def chemical_source(node_id, time):
    return 0.1 * np.exp(-time) if node_id < 25 else 0.0

# 求解
solution_history = coupling.solve_coupled_system(
    initial_concentration, initial_displacement,
    boundary_conditions, time_steps=20, dt=1e-3,
    chemical_source=chemical_source
)

print(f"化学-力学耦合求解完成，共 {len(solution_history)} 个时间步")
```

### 3. 可视化

```python
from visualization.plotting import plot_2d_field, plot_contour
import numpy as np

# 创建测试数据
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)

# 绘制2D场
plot_2d_field(X, Y, Z, title="测试场", save_path="test_field.png")

# 绘制等高线
plot_contour(X, Y, Z, levels=20, title="等高线", save_path="test_contour.png")
```

### 4. 性能测试

```python
import time
import numpy as np
import scipy.sparse as sp
from parallel.advanced_parallel_solver import create_parallel_solver, ParallelConfig

def benchmark_solvers():
    """测试不同求解器的性能"""
    
    # 创建测试矩阵
    n = 2000
    A = sp.random(n, n, density=0.005, format='csr')
    A = A + A.T + sp.eye(n)
    b = np.random.rand(n)
    
    # 测试不同求解器
    solvers = ['cg', 'gmres']
    
    for solver_type in solvers:
        print(f"\n测试 {solver_type.upper()} 求解器:")
        
        config = ParallelConfig(solver_type=solver_type, max_iterations=1000, tolerance=1e-8)
        solver = create_parallel_solver(solver_type, config)
        
        start_time = time.time()
        x = solver.solve(A, b)
        end_time = time.time()
        
        stats = solver.get_performance_stats()
        print(f"  求解时间: {end_time - start_time:.4f}秒")
        print(f"  迭代次数: {stats.iterations}")
        print(f"  残差范数: {stats.residual_norm:.2e}")

if __name__ == "__main__":
    benchmark_solvers()
```

### 5. 常见配置

#### 并行计算配置

```python
# 高性能配置
high_performance_config = ParallelConfig(
    solver_type='cg',
    max_iterations=2000,
    tolerance=1e-10,
    communication_optimization=True,
    load_balancing=True,
    use_nonblocking=True,
    preconditioner='ilu'
)

# 快速测试配置
quick_test_config = ParallelConfig(
    solver_type='cg',
    max_iterations=100,
    tolerance=1e-6,
    communication_optimization=False,
    load_balancing=False,
    preconditioner='jacobi'
)
```

#### 耦合求解配置

```python
# 高精度配置
high_accuracy_config = CouplingConfig(
    solver_type='iterative',
    max_iterations=20,
    tolerance=1e-8,
    adaptive_timestep=True,
    min_timestep=1e-6,
    max_timestep=1e-3
)

# 快速配置
fast_config = CouplingConfig(
    solver_type='staggered',
    max_iterations=5,
    tolerance=1e-4,
    adaptive_timestep=False
)
```

### 6. 故障排除

#### 检查依赖

```python
def check_dependencies():
    """检查依赖包是否可用"""
    
    dependencies = {
        'numpy': '核心数值计算',
        'scipy': '科学计算',
        'matplotlib': '可视化',
        'mpi4py': '并行计算',
        'cupy': 'CUDA加速',
        'plotly': '交互式可视化'
    }
    
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"✓ {package}: {description}")
        except ImportError:
            print(f"✗ {package}: {description} (未安装)")
    
    # 检查MPI
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        print(f"✓ MPI: 可用 (进程数: {comm.Get_size()})")
    except ImportError:
        print("✗ MPI: 不可用 (将使用串行模式)")

if __name__ == "__main__":
    check_dependencies()
```

#### 内存使用监控

```python
import psutil
import os

def monitor_memory():
    """监控内存使用"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"内存使用: {memory_info.rss / 1024 / 1024:.1f} MB")

# 在关键计算前后调用
monitor_memory()
# ... 执行计算 ...
monitor_memory()
```

### 7. 示例脚本

#### 完整的热-力学耦合示例

```python
#!/usr/bin/env python3
"""
完整的热-力学耦合示例
"""

import numpy as np
from coupling.thermal_mechanical import ThermoMechanicalCoupling, CouplingConfig

def main():
    print("开始热-力学耦合计算...")
    
    # 创建网格
    class SimpleMesh:
        def __init__(self, n_points):
            self.nodes = np.linspace(0, 1, n_points)
            self.n_points = n_points
    
    mesh = SimpleMesh(100)
    
    # 创建配置
    config = CouplingConfig(
        solver_type='iterative',
        max_iterations=10,
        tolerance=1e-6,
        adaptive_timestep=True
    )
    
    # 创建耦合求解器
    coupling = ThermoMechanicalCoupling(mesh, config=config)
    
    # 初始条件
    initial_temperature = np.ones(100) * 293.15
    initial_displacement = np.zeros(100)
    
    # 边界条件
    boundary_conditions = {
        'thermal': {0: 373.15, 99: 293.15},
        'mechanical': {0: 0.0, 99: 0.0}
    }
    
    # 源项
    def heat_source(node_id, time):
        return 1000.0 if node_id < 50 else 0.0
    
    def body_force(node_id, time):
        return 9.81 * 2700.0
    
    # 求解
    print("开始求解...")
    solution_history = coupling.solve_coupled_system(
        initial_temperature, initial_displacement,
        boundary_conditions, time_steps=50, dt=1e-3,
        heat_source=heat_source, body_force=body_force
    )
    
    # 结果分析
    print(f"求解完成，共 {len(solution_history)} 个时间步")
    
    final_state = solution_history[-1]
    print(f"最终温度范围: {final_state.temperature.min():.2f} - {final_state.temperature.max():.2f} K")
    print(f"最终位移范围: {final_state.displacement.min():.2e} - {final_state.displacement.max():.2e} m")
    
    # 性能统计
    stats = coupling.get_performance_stats()
    print(f"总求解时间: {stats['solve_time']:.4f}秒")
    print(f"热求解时间: {stats['thermal_solve_time']:.4f}秒")
    print(f"力学求解时间: {stats['mechanical_solve_time']:.4f}秒")
    
    # 可视化
    print("生成可视化结果...")
    coupling.visualize_coupling_results(solution_history)
    
    print("计算完成！")

if __name__ == "__main__":
    main()
```

### 8. 下一步

1. **阅读完整文档**：查看 `USAGE_DOCUMENTATION.md` 了解详细用法
2. **运行示例**：执行 `python examples/` 目录下的示例脚本
3. **性能优化**：根据具体问题调整配置参数
4. **扩展功能**：基于现有模块开发新的物理模型

### 9. 获取帮助

- 查看 `USAGE_DOCUMENTATION.md` 获取详细使用说明
- 运行示例脚本学习基本用法
- 检查 `requirements.txt` 确保所有依赖已安装
- 使用 `check_dependencies()` 函数诊断环境问题

---

**注意**：本快速入门指南提供了基本的使用方法。对于复杂问题或性能优化，请参考完整的使用文档。
