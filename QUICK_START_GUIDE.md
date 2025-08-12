# GeoSim 快速入门指南

## 概述

GeoSim 是一个集成了数值模拟、机器学习和多物理场耦合的地质数值模拟库。本指南将帮助您快速上手，从基础概念到完整的端到端仿真案例。

## 目录

1. [安装与配置](#安装与配置)
2. [基础概念](#基础概念)
3. [快速开始](#快速开始)
4. [端到端案例](#端到端案例)
5. [核心函数详解](#核心函数详解)
6. [高级功能](#高级功能)
7. [故障排除](#故障排除)
8. [最佳实践](#最佳实践)

## 安装与配置

### 系统要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 8GB+ RAM (推荐16GB+)
- 10GB+ 磁盘空间

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/geo_sim.git
cd geo_sim

# 2. 创建虚拟环境
python -m venv geo_sim_env
source geo_sim_env/bin/activate  # Linux/Mac
# 或
geo_sim_env\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import core; print('✅ 安装成功!')"
```

### 配置检查

```python
from core import check_dependencies

# 检查依赖库
status = check_dependencies()
print(f"依赖检查结果: {status}")
```

## 基础概念

### 1. 统一API架构

GeoSim 采用统一的API设计，所有核心模块都遵循相同的接口模式：

```python
# 统一接口模式
simulator = create_simulator('finite_element', config)
simulator.setup(mesh=mesh, boundary_conditions=bc)
result = simulator.run()
simulator.visualize()
```

### 2. 配置驱动

使用YAML配置文件管理仿真参数，支持场景模板：

```python
from core import load_scenario_template

# 加载预定义场景
config = load_scenario_template('reservoir_simulation')
print(f"场景: {config.name}")
print(f"时间步数: {config.numerical_params['time_steps']}")
```

### 3. 调试与监控

内置实时监控和错误诊断工具：

```python
from core import create_simulation_with_debug

# 创建带调试功能的仿真器
simulator, debug_manager = create_simulation_with_debug('finite_element')
debug_manager.start_debugging()
```

## 快速开始

### 第一个仿真：热传导问题

```python
import numpy as np
from core import SimulationConfig, create_simulator

# 1. 创建配置
config = SimulationConfig(
    name="heat_conduction_demo",
    description="简单热传导问题",
    physics_params={
        'thermal_diffusivity': 1e-6,  # 热扩散系数 (m²/s)
        'thermal_conductivity': 2.0,  # 热导率 (W/m·K)
        'specific_heat': 920.0        # 比热容 (J/kg·K)
    },
    numerical_params={
        'time_steps': 100,
        'dt': 0.01,
        'tolerance': 1e-6
    }
)

# 2. 创建仿真器
simulator = create_simulator('finite_element', config)

# 3. 设置问题
def create_simple_mesh():
    """创建简单网格"""
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    X, Y = np.meshgrid(x, y)
    return X, Y

# 4. 运行仿真
mesh = create_simple_mesh()
simulator.setup(mesh=mesh)
result = simulator.run()

print(f"仿真完成，耗时: {result.duration:.2f} 秒")
```

### 使用场景模板

```python
from core import load_scenario_template, create_simulator

# 加载油气藏模拟模板
config = load_scenario_template('reservoir_simulation')

# 修改关键参数
config.numerical_params['time_steps'] = 500
config.physics_params['fluid']['oil_viscosity'] = 2.0e-3

# 创建仿真器
simulator = create_simulator('multi_physics', config)
```

## 端到端案例

### 案例1：用真实油藏数据训练代理模型

#### 步骤1：数据准备

```python
import pandas as pd
import numpy as np
from core import SimulationConfig, MLSimulator

# 加载真实油藏数据
def load_reservoir_data(filepath: str):
    """加载油藏数据"""
    data = pd.read_csv(filepath)
    
    # 数据预处理
    features = ['depth', 'porosity', 'permeability', 'pressure', 'temperature']
    target = 'oil_production_rate'
    
    X = data[features].values
    y = data[target].values
    
    return X, y

# 加载数据
X_train, y_train = load_reservoir_data('./data/reservoir_training.csv')
X_test, y_test = load_reservoir_data('./data/reservoir_testing.csv')

print(f"训练数据: {X_train.shape}")
print(f"测试数据: {X_test.shape}")
```

#### 步骤2：创建ML仿真器

```python
from core import create_simulation_with_debug

# 创建ML仿真器配置
ml_config = SimulationConfig(
    name="reservoir_proxy_model",
    description="油藏生产预测代理模型",
    physics_params={
        'input_features': ['depth', 'porosity', 'permeability', 'pressure', 'temperature'],
        'output_target': 'oil_production_rate',
        'model_type': 'neural_network'
    },
    numerical_params={
        'training_epochs': 1000,
        'batch_size': 32,
        'learning_rate': 0.001,
        'validation_split': 0.2
    }
)

# 创建带调试功能的ML仿真器
simulator, debug_manager = create_simulation_with_debug('ml', ml_config)

# 添加物理约束
def production_rate_constraint(predicted, actual):
    """生产速率物理约束：不能为负值"""
    return np.maximum(0, predicted) - predicted

def pressure_dependency_constraint(features, predicted):
    """压力依赖约束：压力越高，生产速率应该越大"""
    pressure = features[:, 3]  # 压力特征
    return np.gradient(predicted) - np.gradient(pressure)

debug_manager.add_physical_constraint(
    name="生产速率非负约束",
    equation=production_rate_constraint,
    weight=1.0,
    tolerance=1e-6
)

debug_manager.add_physical_constraint(
    name="压力依赖约束",
    equation=pressure_dependency_constraint,
    weight=0.5,
    tolerance=1e-5
)
```

#### 步骤3：训练模型

```python
# 设置训练数据
simulator.setup(
    training_data=(X_train, y_train),
    validation_data=(X_test, y_test),
    model_architecture='mlp',  # 多层感知机
    hidden_layers=[64, 32, 16]
)

# 开始调试监控
debug_manager.start_debugging()
debug_manager.create_dashboards()

# 训练模型
print("开始训练代理模型...")
result = simulator.run(mode='training')

print(f"训练完成!")
print(f"最终损失: {result.performance_metrics.get('final_loss', 'N/A')}")
print(f"验证准确率: {result.performance_metrics.get('validation_accuracy', 'N/A')}")
```

#### 步骤4：模型验证与预测

```python
# 模型验证
validation_result = simulator.run(mode='validation')

# 预测新数据
def predict_production_rate(simulator, new_data):
    """预测新油藏的生产速率"""
    predictions = simulator.predict(new_data)
    return predictions

# 示例：预测新油藏
new_reservoir = np.array([[
    2000,    # 深度 (m)
    0.25,    # 孔隙度
    1e-12,   # 渗透率 (m²)
    2.5e7,   # 压力 (Pa)
    350      # 温度 (K)
]])

predicted_rate = predict_production_rate(simulator, new_reservoir)
print(f"预测生产速率: {predicted_rate[0]:.2f} m³/day")
```

#### 步骤5：与传统模拟对比

```python
from core import create_simulator, load_scenario_template

def run_traditional_simulation(reservoir_params):
    """运行传统数值模拟"""
    # 加载配置
    config = load_scenario_template('reservoir_simulation')
    
    # 设置参数
    config.physics_params['rock']['porosity'] = reservoir_params[1]
    config.physics_params['rock']['permeability'] = reservoir_params[2]
    config.physics_params['fluid']['oil_density'] = 850.0
    
    # 创建仿真器
    simulator = create_simulator('multi_physics', config)
    
    # 运行仿真
    simulator.setup()
    result = simulator.run()
    
    return result

# 运行传统模拟
traditional_result = run_traditional_simulation(new_reservoir[0])

# 对比结果
print("=== 结果对比 ===")
print(f"代理模型预测: {predicted_rate[0]:.2f} m³/day")
print(f"传统模拟结果: {traditional_result.data.get('oil_production_rate', 'N/A')}")

# 计算误差
if 'oil_production_rate' in traditional_result.data:
    error = abs(predicted_rate[0] - traditional_result.data['oil_production_rate'])
    relative_error = error / traditional_result.data['oil_production_rate'] * 100
    print(f"绝对误差: {error:.2f} m³/day")
    print(f"相对误差: {relative_error:.2f}%")
```

### 案例2：地震反演与地质建模

#### 步骤1：地震数据处理

```python
import numpy as np
from scipy import signal
from core import SimulationConfig, create_simulator

def process_seismic_data(seismic_file: str):
    """处理地震数据"""
    # 加载地震数据
    data = np.load(seismic_file)
    
    # 数据预处理
    # 1. 去噪
    denoised = signal.wiener(data)
    
    # 2. 滤波
    b, a = signal.butter(4, 0.1, 'low')
    filtered = signal.filtfilt(b, a, denoised)
    
    # 3. 振幅校正
    corrected = filtered * np.exp(-0.1 * np.arange(len(filtered)))
    
    return corrected

# 处理数据
observed_data = process_seismic_data('./data/seismic_observed.npy')
print(f"地震数据形状: {observed_data.shape}")
```

#### 步骤2：创建反演仿真器

```python
# 加载地震反演配置
inversion_config = load_scenario_template('seismic_inversion')

# 修改反演参数
inversion_config.numerical_params['inversion']['max_iterations'] = 200
inversion_config.numerical_params['inversion']['regularization_weight'] = 0.005

# 创建反演仿真器
inversion_simulator = create_simulator('ml', inversion_config)

# 设置反演问题
inversion_simulator.setup(
    observed_data=observed_data,
    initial_model='smooth_velocity',
    regularization='tikhonov',
    optimization_algorithm='lbfgs'
)
```

#### 步骤3：执行反演

```python
# 运行反演
print("开始地震反演...")
inversion_result = inversion_simulator.run(mode='inversion')

# 分析结果
print(f"反演完成!")
print(f"最终失配: {inversion_result.performance_metrics.get('final_misfit', 'N/A')}")
print(f"收敛迭代数: {inversion_result.performance_metrics.get('convergence_iterations', 'N/A')}")

# 获取反演模型
velocity_model = inversion_result.data.get('velocity_model')
density_model = inversion_result.data.get('density_model')
```

#### 步骤4：地质解释与建模

```python
def interpret_geology(velocity_model, density_model):
    """地质解释"""
    # 基于速度-密度关系识别岩性
    vp_vs_ratio = velocity_model['vp'] / velocity_model['vs']
    
    # 岩性分类
    lithology = np.zeros_like(vp_vs_ratio)
    
    # 砂岩
    sandstone_mask = (vp_vs_ratio < 1.7) & (density_model > 2000)
    lithology[sandstone_mask] = 1
    
    # 泥岩
    shale_mask = (vp_vs_ratio > 1.8) & (density_model < 2500)
    lithology[shale_mask] = 2
    
    # 碳酸盐岩
    carbonate_mask = (vp_vs_ratio > 1.6) & (density_model > 2500)
    lithology[carbonate_mask] = 3
    
    return lithology

# 执行地质解释
lithology_model = interpret_geology(velocity_model, density_model)

print("=== 地质解释结果 ===")
print(f"砂岩体积: {np.sum(lithology_model == 1)} 网格点")
print(f"泥岩体积: {np.sum(lithology_model == 2)} 网格点")
print(f"碳酸盐岩体积: {np.sum(lithology_model == 3)} 网格点")
```

## 核心函数详解

### 1. ThermoMechanicalCoupling.simulate()

```python
class ThermoMechanicalCoupling:
    def simulate(self, 
                 mesh: np.ndarray,
                 initial_temperature: np.ndarray,
                 initial_displacement: np.ndarray,
                 boundary_conditions: Dict[str, Any],
                 time_steps: int = 100,
                 dt: float = 0.01,
                 thermal_diffusivity: float = 1e-6,
                 young_modulus: float = 20e9,
                 poisson_ratio: float = 0.25,
                 thermal_expansion: float = 3e-5,
                 **kwargs) -> Dict[str, Any]:
        """
        热-力学耦合仿真
        
        Args:
            mesh: 网格坐标 (n_nodes, 3)
            initial_temperature: 初始温度场 (n_nodes,)
            initial_displacement: 初始位移场 (n_nodes, 3)
            boundary_conditions: 边界条件字典
            time_steps: 时间步数
            dt: 时间步长 (秒)
            thermal_diffusivity: 热扩散系数 (m²/s)
                - 典型值: 1e-7 到 1e-5 m²/s
                - 岩石: 1e-6 m²/s
                - 土壤: 1e-7 m²/s
                - 金属: 1e-5 m²/s
            young_modulus: 杨氏模量 (Pa)
                - 软土: 1e7 到 1e8 Pa
                - 硬岩: 1e10 到 1e11 Pa
                - 混凝土: 2e10 到 4e10 Pa
            poisson_ratio: 泊松比
                - 范围: 0.0 到 0.5
                - 岩石: 0.2 到 0.3
                - 土壤: 0.3 到 0.4
                - 金属: 0.25 到 0.35
            thermal_expansion: 热膨胀系数 (1/K)
                - 岩石: 2e-5 到 3e-5 1/K
                - 混凝土: 1e-5 到 1.2e-5 1/K
                - 金属: 1e-5 到 2e-5 1/K
        
        Returns:
            Dict[str, Any]: 仿真结果
                - 'temperature': 温度场历史 (time_steps, n_nodes)
                - 'displacement': 位移场历史 (time_steps, n_nodes, 3)
                - 'stress': 应力场历史 (time_steps, n_nodes, 6)
                - 'strain': 应变场历史 (time_steps, n_nodes, 6)
                - 'convergence_history': 收敛历史
                - 'performance_metrics': 性能指标
        
        Raises:
            ValueError: 参数超出合理范围
            RuntimeError: 仿真过程中出现数值问题
        
        Example:
            >>> # 创建简单网格
            >>> x = np.linspace(0, 1, 11)
            >>> y = np.linspace(0, 1, 11)
            >>> X, Y = np.meshgrid(x, y)
            >>> mesh = np.column_stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])
            
            >>> # 设置初始条件
            >>> initial_temp = 300 * np.ones(len(mesh))  # 300K
            >>> initial_disp = np.zeros((len(mesh), 3))
            
            >>> # 设置边界条件
            >>> bc = {
            ...     'temperature': {'top': 400, 'bottom': 300},  # K
            ...     'displacement': {'left': 'fixed', 'right': 'free'}
            ... }
            
            >>> # 运行仿真
            >>> coupling = ThermoMechanicalCoupling()
            >>> result = coupling.simulate(
            ...     mesh=mesh,
            ...     initial_temperature=initial_temp,
            ...     initial_displacement=initial_disp,
            ...     boundary_conditions=bc,
            ...     time_steps=50,
            ...     dt=0.02,
            ...     thermal_diffusivity=1e-6,
            ...     young_modulus=20e9,
            ...     poisson_ratio=0.25,
            ...     thermal_expansion=3e-5
            ... )
            
            >>> print(f"仿真完成，最终温度范围: {result['temperature'][-1].min():.1f} - {result['temperature'][-1].max():.1f} K")
        """
        # 参数验证
        self._validate_parameters(
            thermal_diffusivity, young_modulus, poisson_ratio, thermal_expansion
        )
        
        # 初始化
        self._setup_simulation(mesh, initial_temperature, initial_displacement)
        
        # 时间步进
        for step in range(time_steps):
            # 热传导求解
            temperature = self._solve_heat_conduction(step, dt, thermal_diffusivity)
            
            # 热应力计算
            thermal_stress = self._compute_thermal_stress(temperature, thermal_expansion)
            
            # 力学求解
            displacement = self._solve_mechanics(step, dt, young_modulus, poisson_ratio, thermal_stress)
            
            # 更新状态
            self._update_state(temperature, displacement, step)
            
            # 收敛检查
            if self._check_convergence(step):
                break
        
        return self._collect_results()
    
    def _validate_parameters(self, thermal_diffusivity, young_modulus, poisson_ratio, thermal_expansion):
        """验证参数合理性"""
        if not (1e-8 <= thermal_diffusivity <= 1e-4):
            raise ValueError(f"热扩散系数 {thermal_diffusivity} 超出合理范围 [1e-8, 1e-4] m²/s")
        
        if not (1e6 <= young_modulus <= 1e12):
            raise ValueError(f"杨氏模量 {young_modulus} 超出合理范围 [1e6, 1e12] Pa")
        
        if not (0.0 <= poisson_ratio <= 0.5):
            raise ValueError(f"泊松比 {poisson_ratio} 超出合理范围 [0.0, 0.5]")
        
        if not (1e-6 <= thermal_expansion <= 1e-4):
            raise ValueError(f"热膨胀系数 {thermal_expansion} 超出合理范围 [1e-6, 1e-4] 1/K")
```

### 2. MultiphaseFluidCoupling.simulate()

```python
class MultiphaseFluidCoupling:
    def simulate(self,
                 mesh: np.ndarray,
                 initial_saturations: Dict[str, np.ndarray],
                 initial_pressure: np.ndarray,
                 boundary_conditions: Dict[str, Any],
                 time_steps: int = 100,
                 dt: float = 86400.0,  # 1天
                 fluid_properties: Dict[str, Any] = None,
                 rock_properties: Dict[str, Any] = None,
                 capillary_model: str = 'brooks_corey',
                 relative_permeability_model: str = 'corey',
                 **kwargs) -> Dict[str, Any]:
        """
        多相流体耦合仿真
        
        Args:
            mesh: 网格坐标 (n_elements, 3)
            initial_saturations: 初始饱和度
                - 'oil': 油饱和度 (n_elements,)
                - 'water': 水饱和度 (n_elements,)
                - 'gas': 气饱和度 (n_elements,)
            initial_pressure: 初始压力场 (n_elements,)
            boundary_conditions: 边界条件
            time_steps: 时间步数
            dt: 时间步长 (秒)
            fluid_properties: 流体属性
                - 'oil_viscosity': 油粘度 (Pa·s)
                - 'water_viscosity': 水粘度 (Pa·s)
                - 'gas_viscosity': 气粘度 (Pa·s)
                - 'oil_density': 油密度 (kg/m³)
                - 'water_density': 水密度 (kg/m³)
                - 'gas_density': 气密度 (kg/m³)
            rock_properties: 岩石属性
                - 'porosity': 孔隙度
                - 'permeability': 渗透率 (m²)
                - 'capillary_pressure_params': 毛细管压力参数
            capillary_model: 毛细管压力模型
                - 'brooks_corey': Brooks-Corey模型
                - 'van_genuchten': van Genuchten模型
            relative_permeability_model: 相对渗透率模型
                - 'corey': Corey模型
                - 'brooks_corey': Brooks-Corey模型
        
        Returns:
            Dict[str, Any]: 仿真结果
                - 'oil_saturation': 油饱和度历史
                - 'water_saturation': 水饱和度历史
                - 'gas_saturation': 气饱和度历史
                - 'pressure': 压力场历史
                - 'oil_production': 油产量历史
                - 'water_production': 水产量历史
                - 'gas_production': 气产量历史
        
        Example:
            >>> # 设置初始条件
            >>> n_elements = 1000
            >>> initial_sat = {
            ...     'oil': 0.7 * np.ones(n_elements),
            ...     'water': 0.3 * np.ones(n_elements),
            ...     'gas': np.zeros(n_elements)
            ... }
            >>> initial_pressure = 2e7 * np.ones(n_elements)  # 20 MPa
            
            >>> # 流体属性
            >>> fluid_props = {
            ...     'oil_viscosity': 1e-3,      # 1 mPa·s
            ...     'water_viscosity': 1e-3,    # 1 mPa·s
            ...     'gas_viscosity': 1e-5,      # 0.01 mPa·s
            ...     'oil_density': 850,          # kg/m³
            ...     'water_density': 1000,       # kg/m³
            ...     'gas_density': 1.2           # kg/m³
            ... }
            
            >>> # 岩石属性
            >>> rock_props = {
            ...     'porosity': 0.2,
            ...     'permeability': 1e-12,      # 1 mD
            ...     'capillary_pressure_params': {
            ...         'entry_pressure': 5000,  # Pa
            ...         'lambda': 2.0
            ...     }
            ... }
            
            >>> # 运行仿真
            >>> coupling = MultiphaseFluidCoupling()
            >>> result = coupling.simulate(
            ...     mesh=mesh,
            ...     initial_saturations=initial_sat,
            ...     initial_pressure=initial_pressure,
            ...     boundary_conditions=bc,
            ...     time_steps=365,              # 1年
            ...     dt=86400,                    # 1天
            ...     fluid_properties=fluid_props,
            ...     rock_properties=rock_props
            ... )
        """
        # 参数验证和初始化
        self._validate_inputs(initial_saturations, initial_pressure, fluid_properties, rock_properties)
        self._setup_simulation(mesh, initial_saturations, initial_pressure)
        
        # 时间步进
        for step in range(time_steps):
            # 压力求解
            pressure = self._solve_pressure_equation(step, dt)
            
            # 饱和度求解
            saturations = self._solve_saturation_equations(step, dt, pressure)
            
            # 产量计算
            production = self._compute_production_rates(saturations, pressure)
            
            # 更新状态
            self._update_state(pressure, saturations, production, step)
            
            # 收敛检查
            if self._check_convergence(step):
                break
        
        return self._collect_results()
```

## 高级功能

### 1. 自适应网格加密

```python
from core import AdaptiveMeshRefinement

def adaptive_simulation():
    """自适应网格加密仿真"""
    # 创建自适应网格管理器
    amr = AdaptiveMeshRefinement(
        initial_mesh=initial_mesh,
        refinement_criteria='gradient',
        max_refinement_levels=3,
        refinement_threshold=0.1
    )
    
    # 运行自适应仿真
    for step in range(time_steps):
        # 求解当前网格
        solution = solve_on_current_mesh()
        
        # 评估误差
        error_indicators = amr.estimate_error(solution)
        
        # 决定是否加密
        if amr.should_refine(error_indicators):
            amr.refine_mesh(error_indicators)
            solution = interpolate_solution(solution, amr.get_new_mesh())
        
        # 继续仿真
        continue_simulation(solution)
```

### 2. 并行计算

```python
from core import ParallelSimulationManager

def parallel_simulation():
    """并行仿真"""
    # 创建并行管理器
    parallel_manager = ParallelSimulationManager(
        num_processes=4,
        domain_decomposition='metis',
        load_balancing=True
    )
    
    # 设置并行仿真
    parallel_manager.setup_simulation(
        mesh=large_mesh,
        solver_type='gmres',
        preconditioner='ilu'
    )
    
    # 运行并行仿真
    result = parallel_manager.run()
    
    print(f"并行效率: {result.parallel_efficiency:.2%}")
```

### 3. GPU加速

```python
from core import GPUSimulationManager

def gpu_acceleration():
    """GPU加速仿真"""
    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("GPU不可用，使用CPU")
        return
    
    # 创建GPU管理器
    gpu_manager = GPUSimulationManager(
        device='cuda:0',
        mixed_precision=True,
        memory_optimization=True
    )
    
    # 设置GPU仿真
    gpu_manager.setup_simulation(
        model=neural_network,
        training_data=training_data,
        batch_size=128
    )
    
    # 运行GPU仿真
    result = gpu_manager.run()
    
    print(f"GPU加速比: {result.speedup_factor:.2f}x")
```

## 故障排除

### 常见问题及解决方案

#### 1. 网格质量问题

**问题**: 仿真过程中出现"网格质量差"错误

**解决方案**:
```python
from core import MeshQualityChecker

# 检查网格质量
checker = MeshQualityChecker()
quality_report = checker.check_mesh(mesh)

if quality_report.has_issues:
    print("网格质量问题:")
    for issue in quality_report.issues:
        print(f"  - {issue.description}")
        print(f"    建议: {issue.suggestion}")
    
    # 自动修复
    fixed_mesh = checker.auto_fix_mesh(mesh)
    print("网格已自动修复")
```

#### 2. 数值稳定性问题

**问题**: 仿真发散或出现NaN值

**解决方案**:
```python
# 调整数值参数
config.numerical_params.update({
    'dt': config.numerical_params['dt'] * 0.5,  # 减小时间步长
    'tolerance': config.numerical_params['tolerance'] * 10,  # 放宽收敛容差
    'max_iterations': config.numerical_params['max_iterations'] * 2  # 增加最大迭代次数
})

# 使用更稳定的求解器
config.numerical_params['linear_solver']['type'] = 'direct'  # 直接求解器
config.numerical_params['linear_solver']['preconditioner'] = 'ilu'  # ILU预处理器
```

#### 3. 内存不足问题

**问题**: 出现"Out of Memory"错误

**解决方案**:
```python
# 启用内存优化
config.performance['memory_optimization'].update({
    'enabled': True,
    'chunk_size': 500,  # 减小分块大小
    'compression': True,
    'swap_to_disk': True  # 启用磁盘交换
})

# 使用分块计算
from core import ChunkedSimulation

chunked_sim = ChunkedSimulation(
    simulator=simulator,
    chunk_size=500,
    overlap_size=10
)

result = chunked_sim.run()
```

### 错误诊断工具

```python
from core import ErrorDiagnostic

# 创建错误诊断器
diagnostic = ErrorDiagnostic()

try:
    # 运行仿真
    result = simulator.run()
except Exception as e:
    # 诊断错误
    error_info = diagnostic.diagnose_error(e, context={
        'mesh': mesh,
        'boundary_conditions': boundary_conditions,
        'parameters': config.physics_params
    })
    
    print("=== 错误诊断报告 ===")
    print(f"错误类型: {error_info['error_type']}")
    print(f"错误信息: {error_info['error_message']}")
    
    print("\n诊断结果:")
    for diagnosis in error_info['diagnosis']:
        print(f"  - {diagnosis}")
    
    print("\n修复建议:")
    for suggestion in error_info['suggestions']:
        print(f"  - {suggestion}")
```

## 最佳实践

### 1. 配置管理

```python
# 使用配置文件管理参数
import yaml

def load_simulation_config(config_file: str):
    """加载仿真配置"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 验证配置
    validate_config(config_data)
    
    return SimulationConfig(**config_data)

def validate_config(config_data: dict):
    """验证配置参数"""
    required_fields = ['name', 'physics_params', 'numerical_params']
    for field in required_fields:
        if field not in config_data:
            raise ValueError(f"缺少必需字段: {field}")
    
    # 验证数值参数
    num_params = config_data['numerical_params']
    if num_params['dt'] <= 0:
        raise ValueError("时间步长必须大于0")
    if num_params['time_steps'] <= 0:
        raise ValueError("时间步数必须大于0")
```

### 2. 结果验证

```python
def validate_simulation_results(result: SimulationResult):
    """验证仿真结果"""
    # 物理合理性检查
    if 'temperature' in result.data:
        temp = result.data['temperature']
        if np.any(temp < 0) or np.any(temp > 1000):
            warnings.warn("温度值超出合理范围")
    
    if 'pressure' in result.data:
        pressure = result.data['pressure']
        if np.any(pressure < 0) or np.any(pressure > 1e9):
            warnings.warn("压力值超出合理范围")
    
    # 数值稳定性检查
    if 'displacement' in result.data:
        disp = result.data['displacement']
        if np.any(np.isnan(disp)) or np.any(np.isinf(disp)):
            raise ValueError("位移场包含NaN或Inf值")
    
    # 收敛性检查
    if result.convergence_info:
        final_residual = result.convergence_info.get('final_residual', float('inf'))
        if final_residual > result.config.numerical_params['tolerance']:
            warnings.warn("仿真未完全收敛")
```

### 3. 性能优化

```python
def optimize_simulation_performance(config: SimulationConfig):
    """优化仿真性能"""
    # 并行计算优化
    if config.performance['parallel']['enabled']:
        # 根据问题规模选择进程数
        problem_size = estimate_problem_size(config)
        optimal_processes = min(
            config.performance['parallel']['num_processes'],
            problem_size // 10000  # 每个进程处理10000个网格点
        )
        config.performance['parallel']['num_processes'] = optimal_processes
    
    # GPU加速优化
    if config.performance['gpu_acceleration']['enabled']:
        # 启用混合精度
        config.performance['gpu_acceleration']['mixed_precision'] = True
        
        # 内存优化
        config.performance['memory_optimization'].update({
            'enabled': True,
            'chunk_size': 1000,
            'compression': True
        })
    
    # 求解器优化
    if config.numerical_params['linear_solver']['type'] == 'iterative':
        # 选择最佳预处理器
        config.numerical_params['linear_solver']['preconditioner'] = 'amg'  # 代数多重网格
```

### 4. 数据管理

```python
def manage_simulation_data(result: SimulationResult, output_dir: str):
    """管理仿真数据"""
    import h5py
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 保存主要结果
    with h5py.File(output_path / 'simulation_results.h5', 'w') as f:
        # 保存配置
        config_group = f.create_group('config')
        for key, value in result.config.to_dict().items():
            config_group.attrs[key] = str(value)
        
        # 保存数据
        for key, data in result.data.items():
            if isinstance(data, np.ndarray):
                f.create_dataset(f'data/{key}', data=data, compression='gzip')
        
        # 保存性能指标
        perf_group = f.create_group('performance')
        for key, value in result.performance_metrics.items():
            perf_group.attrs[key] = value
    
    # 保存配置文件
    result.config.to_yaml(output_path / 'simulation_config.yaml')
    
    # 保存错误日志
    if result.errors:
        with open(output_path / 'error_log.txt', 'w', encoding='utf-8') as f:
            for error in result.errors:
                f.write(f"{error}\n")
    
    print(f"仿真数据已保存到: {output_path}")
```

## 总结

本快速入门指南涵盖了GeoSim的主要功能和使用方法：

1. **统一API**: 所有核心模块都遵循相同的接口模式
2. **配置驱动**: 使用YAML配置文件管理仿真参数
3. **场景模板**: 预定义的油气藏模拟、地震反演等场景
4. **调试工具**: 实时监控、错误诊断和性能分析
5. **端到端案例**: 从数据处理到结果验证的完整流程
6. **最佳实践**: 配置管理、结果验证、性能优化等

通过本指南，您应该能够：
- 快速上手GeoSim的基本功能
- 理解统一API的设计理念
- 使用场景模板快速启动仿真
- 利用调试工具诊断和解决问题
- 构建完整的端到端仿真流程

如需更多帮助，请参考：
- 详细API文档
- 示例代码库
- 用户论坛
- 技术支持

祝您使用愉快！🚀
