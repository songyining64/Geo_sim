# 自适应物理约束与强化学习控制器集成指南

## 概述

本指南说明如何在现有的 `gpu_acceleration/geological_ml_framework.py` 文件中集成自适应物理约束和强化学习控制器功能，实现PINN物理损失权重的动态调整。

## 核心功能

### 1. 自适应物理约束 (AdaptivePhysicalConstraint)

- **动态权重调整**: 基于PDE残差大小自动调整约束权重
- **历史记录**: 跟踪残差、权重和调整历史
- **趋势分析**: 自动分析残差和权重变化趋势
- **参数控制**: 可配置的调整速率和权重范围

### 2. 强化学习控制器 (RLConstraintController)

- **智能决策**: 使用PPO算法优化权重调整策略
- **状态感知**: 监控约束残差、权重历史、调整趋势
- **自适应控制**: 根据实时性能自动调整控制频率
- **启发式备选**: 当RL不可用时自动降级到启发式控制

## 集成步骤

### 步骤1: 添加依赖检查

在文件开头添加可选依赖检查：

```python
# 可选依赖检查
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_STABLE_BASELINES3 = True
except ImportError:
    HAS_STABLE_BASELINES3 = False
    warnings.warn("stable-baselines3 not available. RL features will be limited.")
```

### 步骤2: 添加核心类

在 `GeologicalPINN` 类之后添加以下类：

1. **AdaptivePhysicalConstraint** - 自适应物理约束类
2. **RLConstraintController** - 强化学习控制器类
3. **AdaptiveGeologicalPINN** - 支持自适应约束的PINN类

### 步骤3: 更新现有函数

修改 `create_geological_ml_system` 函数，添加 `adaptive_constraints` 参数：

```python
def create_geological_ml_system(input_dim: int, output_dim: int, 
                                hidden_layers: List[int] = [64, 64],
                                adaptive_constraints: bool = False) -> Union[GeologicalPINN, AdaptiveGeologicalPINN]:
    """创建地质机器学习系统 - 支持自适应约束"""
    
    if adaptive_constraints:
        return create_adaptive_geological_ml_system(input_dim, output_dim, hidden_layers, True)
    else:
        # 使用原有的GeologicalPINN
        return GeologicalPINN(input_dim, hidden_dims, output_dim)
```

### 步骤4: 添加新的工厂函数

```python
def create_adaptive_geological_ml_system(input_dim: int, output_dim: int, 
                                       hidden_layers: List[int] = [64, 64],
                                       adaptive_constraints: bool = True) -> AdaptiveGeologicalPINN:
    """创建自适应地质机器学习系统"""
    
    pinn = AdaptiveGeologicalPINN(
        input_dim=input_dim,
        hidden_dims=hidden_layers,
        output_dim=output_dim,
        adaptive_constraints=adaptive_constraints
    )
    
    return pinn
```

## 使用方法

### 基本用法

```python
from gpu_acceleration.geological_ml_framework import (
    create_adaptive_geological_ml_system,
    AdaptivePhysicalConstraint
)

# 1. 创建自适应PINN
pinn = create_adaptive_geological_ml_system(
    input_dim=5,
    output_dim=3,
    hidden_layers=[64, 32],
    adaptive_constraints=True
)

# 2. 定义物理约束方程
def darcy_equation(x, y_pred):
    """Darcy流动方程约束"""
    # 计算真实的PDE残差
    return compute_darcy_residual(x, y_pred)

def heat_equation(x, y_pred):
    """热传导方程约束"""
    return compute_heat_residual(x, y_pred)

# 3. 创建自适应约束
darcy_constraint = AdaptivePhysicalConstraint(
    name="Darcy方程",
    equation=darcy_equation,
    initial_weight=1.0,
    min_weight=0.01,
    max_weight=5.0,
    adaptation_rate=0.1
)

heat_constraint = AdaptivePhysicalConstraint(
    name="热传导方程",
    equation=heat_equation,
    initial_weight=0.5,
    min_weight=0.01,
    max_weight=3.0,
    adaptation_rate=0.08
)

# 4. 添加到PINN
pinn.add_physical_constraint(darcy_constraint)
pinn.add_physical_constraint(heat_constraint)

# 5. 设置约束控制器
pinn.setup_constraint_controller()

# 6. 使用自适应约束训练
training_history = pinn.train_with_adaptive_constraints(
    X, y, 
    physics_points=physics_points,
    epochs=1000
)
```

### 高级用法：自定义约束控制器

```python
# 创建自定义约束控制器
controller = RLConstraintController(
    constraints=[darcy_constraint, heat_constraint],
    state_dim=6,  # 2个约束 × 3个状态维度
    action_dim=2, # 2个约束的权重调整
    learning_rate=1e-3,
    gamma=0.99
)

# 手动控制约束调整
for step in range(100):
    # 计算当前残差
    darcy_residual = darcy_constraint.compute_residual()
    heat_residual = heat_constraint.compute_residual()
    
    # 控制器自动调整权重
    controller.control_constraints()
    
    # 获取调整摘要
    if step % 10 == 0:
        summary = controller.get_control_summary()
        print(f"步骤 {step}: {summary}")
```

## 监控和分析

### 约束状态监控

```python
# 获取约束自适应摘要
darcy_summary = darcy_constraint.get_adaptation_summary()
print(f"Darcy约束调整次数: {darcy_summary['total_adaptations']}")
print(f"残差趋势: {darcy_summary['residual_trend']}")
print(f"权重趋势: {darcy_summary['weight_trend']}")

# 获取控制器摘要
if pinn.constraint_controller:
    control_summary = pinn.constraint_controller.get_control_summary()
    print(f"总控制迭代: {control_summary['total_iterations']}")
    print(f"平均奖励: {control_summary['avg_reward']:.4f}")
```

### 训练历史分析

```python
# 获取训练摘要
training_summary = pinn.get_training_summary()
print(f"总步数: {training_summary['total_steps']}")
print(f"约束违反次数: {training_summary['constraint_violations']}")
print(f"控制器状态: {'激活' if training_summary['constraint_controller_active'] else '未激活'}")

# 分析约束权重变化
constraint_weights = training_history['constraint_weights']
for epoch, weights in enumerate(constraint_weights):
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: 权重={weights}")
```

## 配置参数

### AdaptivePhysicalConstraint 参数

- `name`: 约束名称，用于标识和监控
- `equation`: 物理约束方程函数，返回残差值
- `initial_weight`: 初始权重值
- `min_weight`: 最小权重限制
- `max_weight`: 最大权重限制
- `adaptation_rate`: 权重调整速率

### RLConstraintController 参数

- `constraints`: 约束列表
- `state_dim`: 状态向量维度
- `action_dim`: 动作向量维度
- `learning_rate`: 学习率
- `gamma`: 折扣因子

## 性能优化

### 1. 调整频率控制

```python
# 设置控制频率
controller.control_frequency = 10  # 每10次迭代控制一次

# 设置时间间隔
controller.last_control_time = time.time() - 5  # 允许立即控制
```

### 2. 权重调整范围

```python
# 设置合理的权重范围
constraint.min_weight = 0.001  # 避免权重过小
constraint.max_weight = 2.0    # 避免权重过大
constraint.adaptation_rate = 0.05  # 降低调整速率
```

### 3. 残差窗口大小

```python
# 调整残差窗口大小
constraint.residual_window = 20  # 增加窗口大小，提高稳定性
constraint.target_residual = 1e-5  # 调整目标残差
```

## 故障排除

### 常见问题

1. **权重发散**
   - 降低 `adaptation_rate`
   - 缩小权重范围
   - 检查约束方程稳定性

2. **RL代理失败**
   - 检查 `stable-baselines3` 安装
   - 控制器会自动降级到启发式控制
   - 检查状态和动作维度设置

3. **收敛缓慢**
   - 增加 `adaptation_rate`
   - 调整权重范围
   - 检查约束方程实现

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查约束状态
for constraint in pinn.physical_constraints:
    print(f"约束 {constraint.name}: 权重={constraint.current_weight:.4f}")
    print(f"残差历史长度: {len(constraint.residual_history)}")
    print(f"调整历史长度: {len(constraint.adaptation_history)}")

# 检查控制器状态
if pinn.constraint_controller:
    summary = pinn.constraint_controller.get_control_summary()
    print(f"控制器状态: {summary}")
```

## 总结

通过集成自适应物理约束和强化学习控制器，现有的PINN系统可以：

1. **自动优化约束权重**: 基于残差大小动态调整
2. **加速收敛**: 智能权重调整策略
3. **提高稳定性**: 避免固定权重导致的问题
4. **智能控制**: RL代理自动学习最优策略

这些功能与现有的 `GeologicalPINN` 完全兼容，可以通过 `adaptive_constraints` 参数选择是否启用。
