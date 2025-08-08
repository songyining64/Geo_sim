# 化学-力学耦合模块总结

## 概述

化学-力学耦合模块实现了完整的化学-力学多物理场耦合功能，基于Underworld2的设计理念，提供了化学反应动力学、化学扩散、应力-化学耦合等核心功能。

## 核心功能

### 1. 化学反应模型

#### ArrheniusReactionModel
- **功能**: 基于Arrhenius方程的化学反应模型
- **公式**: `rate = A * exp(-E_a / (R * T)) * [C]^n`
- **参数**:
  - `pre_exponential_factor`: 指前因子 (1/s)
  - `activation_energy`: 激活能 (J/mol)
  - `reaction_order`: 反应级数
  - `gas_constant`: 气体常数 (J/(mol·K))

#### 主要方法
- `compute_reaction_rate()`: 计算反应速率
- `compute_equilibrium_concentration()`: 计算平衡浓度

### 2. 扩散模型

#### TemperatureDependentDiffusionModel
- **功能**: 温度依赖的扩散模型
- **公式**: `D = D_0 * exp(-E_D / (R * T))`
- **参数**:
  - `pre_exponential_diffusivity`: 指前扩散系数 (m²/s)
  - `diffusion_activation_energy`: 扩散激活能 (J/mol)
  - `gas_constant`: 气体常数 (J/(mol·K))

#### 主要方法
- `compute_diffusion_coefficient()`: 计算扩散系数
- `compute_diffusion_flux()`: 计算扩散通量

### 3. 应力-化学耦合模型

#### StressChemicalCoupling
- **功能**: 应力与化学过程的耦合模型
- **特性**:
  - 化学应变计算: `ε_chem = β * (C - C_ref)`
  - 应力对反应的影响
  - 化学膨胀系数

#### 主要方法
- `compute_chemical_strain()`: 计算化学应变
- `compute_stress_effect_on_reaction()`: 计算应力对反应的影响

### 4. 化学-力学耦合求解器

#### ChemicalMechanicalCoupling
- **功能**: 完整的化学-力学耦合求解器
- **特性**:
  - 迭代求解耦合系统
  - 自动组装耦合矩阵
  - 边界条件处理
  - 收敛性检查

#### 主要方法
- `solve_coupled_system()`: 求解耦合系统
- `assemble_coupling_matrix()`: 组装耦合矩阵
- `compute_chemical_stress()`: 计算化学应力
- `compute_chemical_force()`: 计算化学力

## 使用示例

### 基本使用

```python
from coupling.chemical_mechanical import (
    ChemicalMechanicalCoupling,
    ArrheniusReactionModel,
    TemperatureDependentDiffusionModel,
    StressChemicalCoupling,
    create_chemical_mechanical_coupling
)

# 创建模型
reaction_model = ArrheniusReactionModel(
    pre_exponential_factor=1e6,
    activation_energy=100e3,
    reaction_order=1.0
)

diffusion_model = TemperatureDependentDiffusionModel(
    pre_exponential_diffusivity=1e-6,
    diffusion_activation_energy=150e3
)

stress_chemical_coupling = StressChemicalCoupling(
    chemical_expansion_coefficient=1e-3,
    stress_coupling_factor=1.0
)

# 创建耦合求解器
coupling_solver = create_chemical_mechanical_coupling(
    mesh=mesh,
    reaction_model=reaction_model,
    diffusion_model=diffusion_model,
    stress_chemical_coupling=stress_chemical_coupling,
    young_modulus=70e9,
    poisson_ratio=0.3,
    coupling_parameter=1.0
)

# 求解耦合系统
solution_history = coupling_solver.solve_coupled_system(
    initial_concentration=initial_concentration,
    initial_displacement=initial_displacement,
    boundary_conditions=boundary_conditions,
    time_steps=10,
    dt=1.0,
    temperature=temperature,
    pressure=pressure
)
```

### 边界条件设置

```python
# 化学边界条件
boundary_conditions = {
    'concentration': {0: 1.0, 99: 0.0},  # 左边界浓度1.0，右边界浓度0.0
    'displacement': {0: 0.0, 99: 0.0}    # 两端固定
}
```

### 源项定义

```python
# 化学源项
def chemical_source(node_id, time):
    return 0.0  # 无化学源项

# 体力
def body_force(node_id, time):
    return np.array([0.0, -9.81])  # 重力
```

## 技术特性

### 1. 数值方法
- **有限元方法**: 基于网格的数值求解
- **迭代求解**: 化学-力学场迭代耦合
- **收敛性检查**: 自动收敛性判断
- **稀疏矩阵**: 使用scipy稀疏矩阵优化

### 2. 并行计算
- **MPI支持**: 支持并行计算
- **域分解**: 自动域分解
- **负载均衡**: 动态负载均衡

### 3. 可视化
- **结果可视化**: 内置可视化功能
- **多场显示**: 浓度、位移、应力、应变等场
- **时间演化**: 时间序列显示

## 应用场景

### 1. 地质过程模拟
- **岩石化学反应**: 矿物相变、溶解-沉淀
- **应力-化学耦合**: 应力对化学反应的影响
- **扩散过程**: 化学组分扩散

### 2. 材料科学
- **相变过程**: 材料相变动力学
- **应力腐蚀**: 应力对腐蚀的影响
- **扩散焊接**: 扩散连接过程

### 3. 环境科学
- **污染物扩散**: 污染物在环境中的扩散
- **化学反应**: 环境中的化学反应
- **应力影响**: 应力对化学反应的影响

## 性能优化

### 1. 计算效率
- **稀疏矩阵**: 使用稀疏矩阵存储
- **迭代求解**: 高效的迭代算法
- **并行计算**: MPI并行支持

### 2. 内存管理
- **内存优化**: 高效的内存使用
- **垃圾回收**: 自动内存管理
- **缓存优化**: 数据缓存策略

## 扩展性

### 1. 模型扩展
- **新反应模型**: 易于添加新的反应模型
- **新扩散模型**: 支持自定义扩散模型
- **新耦合模型**: 灵活的耦合模型接口

### 2. 功能扩展
- **多组分**: 支持多组分系统
- **复杂几何**: 支持复杂几何形状
- **非线性**: 支持非线性材料

## 总结

化学-力学耦合模块提供了完整的化学-力学多物理场耦合功能，具有以下特点：

1. **完整性**: 涵盖化学反应、扩散、应力-化学耦合等核心功能
2. **可扩展性**: 模块化设计，易于扩展新功能
3. **高性能**: 基于稀疏矩阵和并行计算的高效实现
4. **易用性**: 简洁的API接口和丰富的示例
5. **稳定性**: 完善的错误处理和收敛性检查

该模块为地质过程模拟、材料科学、环境科学等领域提供了强大的数值模拟工具。
