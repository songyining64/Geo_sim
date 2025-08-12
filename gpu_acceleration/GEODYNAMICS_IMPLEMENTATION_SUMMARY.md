# 地球动力学物理约束实现总结

## 概述

本文档总结了在现有 `GeologicalPINN` 中添加的地球动力学核心过程物理约束功能，包括新增的关键方程实现和多场耦合适配。

## 已实现功能

### 1. 新增关键方程实现

#### 1.1 断层滑动方程 (fault_slip_equation)
- **物理意义**: 断层摩擦本构关系，模拟地震周期的滑动-闭锁过程
- **数学形式**: v = v₀ exp((μ₀ + a ln(v/v₀) - b ln(θ))/L)
- **输出维度**: [slip_rate, state_variable, stress] (3维)
- **关键参数**: 摩擦系数、直接效应参数、演化效应参数、特征滑移距离

#### 1.2 地幔对流方程 (mantle_convection_equation)
- **物理意义**: 结合Stokes方程和热传导方程，模拟地幔热对流
- **数学形式**: 综合动量守恒、质量守恒、能量守恒
- **输出维度**: [v_x, v_y, v_z, p, T] (5维)
- **关键特性**: 温度依赖粘度、浮力驱动、对流-传导耦合

#### 1.3 板块构造方程 (plate_tectonics_equation)
- **物理意义**: 结合弹性力学和热传导，模拟板块运动
- **数学形式**: 弹性平衡 + 热传导 + 热弹性耦合
- **输出维度**: [u_x, u_y, u_z, T, stress] (5维)
- **关键特性**: 热应力、板块边界条件、位移梯度约束

#### 1.4 化学输运方程 (chemical_transport_equation)
- **物理意义**: 考虑对流-扩散-反应的化学输运过程
- **数学形式**: ∂C/∂t + v·∇C = ∇·(D∇C) + R
- **输出维度**: [C, v_x, v_y, v_z, T] (5维)
- **关键特性**: 温度依赖扩散、对流输运、化学反应

### 2. 多场耦合适配

#### 2.1 自适应权重分配
```python
# 权重分配策略
if "stokes_equation" in equation_name:
    weight = 100.0  # 地幔流动权重最高（核心过程）
elif "mantle_convection_equation" in equation_name:
    weight = 80.0   # 地幔对流（综合方程）
elif "fault_slip_equation" in equation_name:
    weight = 50.0   # 断层过程权重
elif "plate_tectonics_equation" in equation_name:
    weight = 60.0   # 板块构造（综合方程）
elif "heat_conduction_equation" in equation_name:
    weight = 10.0   # 热传导次之
# ... 其他方程
```

#### 2.2 权重设计原则
1. **地幔流动权重最高 (100.0)**: 地球动力学核心过程
2. **综合方程权重较高 (60-80)**: 多物理场耦合
3. **断层过程权重中等 (50.0)**: 地震动力学重要
4. **基础方程权重较低 (8-20)**: 辅助物理过程

### 3. 配置参数扩展

#### 3.1 断层摩擦参数
```python
# 新增：断层摩擦参数
mu0: float = 0.6                     # 静摩擦系数
a: float = 0.01                      # 直接效应参数
b: float = 0.005                     # 演化效应参数
L: float = 0.1                       # 特征滑移距离 (m)
v0: float = 1e-6                     # 参考滑动速率 (m/s)
```

#### 3.2 化学输运参数
```python
# 新增：化学输运参数
activation_energy: float = 50e3      # J/mol，激活能
diffusion_coefficient: float = 1e-9  # m²/s，扩散系数
reaction_rate: float = 0.01          # s⁻¹，反应速率常数
```

## 使用方法

### 1. 基本使用

```python
from gpu_acceleration.geological_ml_framework import (
    GeologicalPINN, 
    GeologicalConfig,
    GeologicalPhysicsEquations
)

# 创建配置
config = GeologicalConfig()

# 创建PINN模型
pinn = GeologicalPINN(
    input_dim=3,      # x, y, t
    output_dim=5,     # 根据物理方程要求调整
    hidden_dims=[64, 64, 32],
    geological_config=config
)

# 添加地球动力学物理约束
physics_equations = [
    GeologicalPhysicsEquations.mantle_convection_equation,  # 地幔对流
    GeologicalPhysicsEquations.fault_slip_equation,         # 断层滑动
    GeologicalPhysicsEquations.chemical_transport_equation, # 化学输运
]

pinn.physics_equations = physics_equations

# 训练模型
training_history = pinn.train(
    X, y, 
    physics_points=physics_points,
    epochs=1000
)
```

### 2. 应用场景配置

#### 2.1 地幔对流模拟
```python
# 使用Stokes方程约束替代达西定律
# 结合热传导方程模拟温度-流动耦合
physics_equations = [
    GeologicalPhysicsEquations.mantle_convection_equation,
    GeologicalPhysicsEquations.mass_conservation_equation
]
```

#### 2.2 断层动力学
```python
# 通过fault_slip_equation约束模拟地震周期
physics_equations = [
    GeologicalPhysicsEquations.fault_slip_equation,
    GeologicalPhysicsEquations.elastic_equilibrium_equation
]
```

#### 2.3 板块构造
```python
# 结合弹性力学和热传导
physics_equations = [
    GeologicalPhysicsEquations.plate_tectonics_equation,
    GeologicalPhysicsEquations.heat_conduction_equation
]
```

## 技术特性

### 1. 自动微分支持
- 所有物理方程都支持PyTorch自动微分
- 自动计算梯度，支持反向传播
- 兼容现有的PINN训练流程

### 2. 数值稳定性
- 使用安全的数值计算（避免除零、对数负数等）
- 合理的参数范围和默认值
- 残差归一化和权重平衡

### 3. 扩展性
- 模块化设计，易于添加新的物理方程
- 支持不同维度的输入输出
- 可配置的物理参数

## 性能优化

### 1. 计算效率
- 向量化计算，支持批量处理
- 智能权重分配，避免过度约束
- 可选的物理约束点采样

### 2. 内存管理
- 自动清理中间计算结果
- 支持GPU加速（如果可用）
- 可配置的批量大小

## 监控和分析

### 1. 损失监控
```python
# 获取各方程的损失详情
if hasattr(pinn, 'equation_losses'):
    for eq_name, eq_loss in pinn.equation_losses.items():
        print(f"{eq_name}: {eq_loss:.6f}")
```

### 2. 训练历史
```python
# 训练历史包含物理损失分解
training_history = {
    'total_loss': [...],
    'data_loss': [...],
    'physics_loss': [...],  # 总物理损失
    'equation_losses': {...}  # 各方程损失
}
```

## 故障排除

### 1. 常见问题

#### 1.1 维度不匹配
- **症状**: 输出维度与物理方程要求不匹配
- **解决**: 检查PINN输出维度，确保与物理方程匹配

#### 1.2 数值不稳定
- **症状**: 残差过大或NaN值
- **解决**: 调整权重分配，检查物理参数范围

#### 1.3 收敛缓慢
- **症状**: 物理损失下降缓慢
- **解决**: 调整学习率，优化权重分配策略

### 2. 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查物理约束状态
for eq in pinn.physics_equations:
    print(f"方程: {eq.__name__}")
    print(f"权重: {get_equation_weight(eq)}")

# 监控训练过程
for epoch in range(epochs):
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: 物理损失 = {physics_loss:.6f}")
        print(f"方程损失详情: {pinn.equation_losses}")
```

## 下一步发展

### 1. 短期目标
- [ ] 集成自适应物理约束功能
- [ ] 添加更多地球动力学方程
- [ ] 优化权重分配策略

### 2. 中期目标
- [ ] 支持3D地球动力学模拟
- [ ] 添加边界条件处理
- [ ] 实现并行计算支持

### 3. 长期目标
- [ ] 扩展到其他行星体模拟
- [ ] 集成观测数据约束
- [ ] 支持实时地球动力学监测

## 总结

通过添加地球动力学专用物理约束方程和多场耦合适配功能，现有的 `GeologicalPINN` 系统现在可以：

1. **模拟地幔对流**: 使用Stokes方程替代达西定律，结合热传导方程
2. **分析断层动力学**: 通过摩擦本构关系模拟地震周期
3. **研究板块构造**: 结合弹性力学和热传导，考虑热弹性耦合
4. **处理化学输运**: 考虑对流-扩散-反应的综合过程

这些功能与现有的自适应物理约束和多保真度建模功能完全兼容，为地球动力学研究提供了强大的数值模拟工具。
