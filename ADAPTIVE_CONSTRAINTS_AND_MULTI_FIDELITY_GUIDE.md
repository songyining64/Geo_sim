# 自适应物理约束与多保真度建模指南

## 概述

本指南介绍GeoSim库中新增的两个核心功能：

1. **自适应物理约束 (Adaptive Physical Constraints)**: 动态调整PINN物理损失权重，基于PDE残差大小自动优化约束强度
2. **多保真度建模 (Multi-Fidelity Modeling)**: 两阶段训练过程，结合低保真度快速仿真和高保真度精确仿真，通过协同训练减少计算成本

## 目录

1. [自适应物理约束](#自适应物理约束)
2. [多保真度建模](#多保真度建模)
3. [集成使用](#集成使用)
4. [性能优化](#性能优化)
5. [最佳实践](#最佳实践)
6. [故障排除](#故障排除)

## 自适应物理约束

### 核心概念

自适应物理约束系统通过动态调整物理损失权重，解决传统PINN中固定权重导致的收敛问题：

- **动态权重调整**: 根据PDE残差大小自动调整约束权重
- **强化学习控制**: 使用RL控制器优化权重调整策略
- **实时监控**: 持续监控约束违反情况并自适应响应

### 主要组件

#### 1. AdaptivePhysicalConstraint 类

```python
from gpu_acceleration.geological_ml_framework import AdaptivePhysicalConstraint

# 创建自适应约束
constraint = AdaptivePhysicalConstraint(
    name="Darcy方程",
    equation=darcy_equation_function,
    initial_weight=1.0,
    min_weight=0.01,
    max_weight=5.0,
    adaptation_rate=0.1
)
```

**参数说明**:
- `name`: 约束名称，用于标识和监控
- `equation`: 物理约束方程函数，返回残差值
- `initial_weight`: 初始权重值
- `min_weight`: 最小权重限制
- `max_weight`: 最大权重限制
- `adaptation_rate`: 权重调整速率

#### 2. RLConstraintController 类

```python
from gpu_acceleration.geological_ml_framework import RLConstraintController

# 创建约束控制器
controller = RLConstraintController(
    constraints=[constraint1, constraint2],
    state_dim=10,
    action_dim=2,
    learning_rate=3e-4
)
```

**功能特性**:
- **状态感知**: 监控约束残差、权重历史、调整趋势
- **智能决策**: 使用PPO算法优化权重调整策略
- **自适应控制**: 根据实时性能自动调整控制频率

#### 3. PhysicsInformedNeuralNetwork 增强版

```python
from gpu_acceleration.geological_ml_framework import PhysicsInformedNeuralNetwork

# 创建支持自适应约束的PINN
pinn = PhysicsInformedNeuralNetwork(
    input_dim=5,
    output_dim=3,
    hidden_layers=[64, 32],
    adaptive_constraints=True  # 启用自适应约束
)

# 添加物理约束
pinn.add_physical_constraint(darcy_constraint)
pinn.add_physical_constraint(heat_constraint)

# 设置约束控制器
pinn.setup_constraint_controller()
```

### 使用示例

#### 基本用法

```python
import numpy as np
from gpu_acceleration.geological_ml_framework import (
    PhysicsInformedNeuralNetwork, 
    AdaptivePhysicalConstraint
)

# 定义物理约束方程
def darcy_equation(x, y_pred):
    """Darcy流动方程残差"""
    # 这里应该实现真实的Darcy方程
    # 示例中返回模拟残差
    return np.random.normal(0, 1e-6)

def heat_equation(x, y_pred):
    """热传导方程残差"""
    return np.random.normal(0, 1e-5)

# 创建自适应约束
darcy_constraint = AdaptivePhysicalConstraint(
    name="Darcy方程",
    equation=darcy_equation,
    initial_weight=1.0,
    min_weight=0.01,
    max_weight=5.0
)

heat_constraint = AdaptivePhysicalConstraint(
    name="热传导方程",
    equation=heat_equation,
    initial_weight=0.5,
    min_weight=0.01,
    max_weight=3.0
)

# 创建PINN并添加约束
pinn = PhysicsInformedNeuralNetwork(
    input_dim=3,
    output_dim=2,
    adaptive_constraints=True
)

pinn.add_physical_constraint(darcy_constraint)
pinn.add_physical_constraint(heat_constraint)
pinn.setup_constraint_controller()

# 训练过程中约束会自动调整
```

#### 高级用法：自定义约束控制器

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

### 监控和分析

#### 约束状态监控

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

#### 可视化监控

```python
import matplotlib.pyplot as plt

# 绘制权重变化历史
weights = darcy_constraint.weight_history
iterations = range(len(weights))

plt.figure(figsize=(10, 6))
plt.plot(iterations, weights, 'b-', linewidth=2, label='Darcy约束权重')
plt.xlabel('迭代次数')
plt.ylabel('权重值')
plt.title('约束权重自适应变化')
plt.legend()
plt.grid(True)
plt.show()
```

## 多保真度建模

### 核心概念

多保真度建模通过结合不同精度和计算成本的仿真模型，实现高效准确的预测：

- **两阶段训练**: 低保真度预训练 + 高保真度微调
- **协同训练**: 低成本模型辅助高成本模型学习
- **知识迁移**: 在保真度级别间传递学习到的特征
- **集成预测**: 结合多个保真度级别的预测结果

### 主要组件

#### 1. FidelityLevel 类

```python
from ensemble.multi_fidelity import FidelityLevel

# 定义低保真度级别
low_fidelity = FidelityLevel(
    name="快速仿真",
    level=0,
    description="使用简化PDE的快速仿真",
    computational_cost=1.0,      # 相对计算成本
    accuracy=0.75,               # 预期精度 (0-1)
    data_requirements=1000,      # 所需数据量
    training_time=120.0,         # 预期训练时间 (秒)
    model_type="neural_network", # 模型类型
    model_params={
        'hidden_layers': [32, 16],
        'activation': 'relu',
        'dropout': 0.1
    }
)
```

#### 2. MultiFidelityConfig 类

```python
from ensemble.multi_fidelity import MultiFidelityConfig

config = MultiFidelityConfig(
    name="油藏预测系统",
    description="结合快速和精确仿真的油藏预测系统",
    fidelity_levels=[low_fidelity, high_fidelity],
    
    # 协同训练配置
    co_training=MultiFidelityConfig.co_training(
        enabled=True,
        transfer_learning=True,      # 启用知识迁移
        knowledge_distillation=True, # 启用知识蒸馏
        ensemble_method='weighted_average'  # 集成方法
    ),
    
    # 训练策略配置
    training_strategy=MultiFidelityConfig.training_strategy(
        stage1_epochs=1000,     # 低保真度预训练
        stage2_epochs=500,      # 高保真度微调
        transfer_epochs=200,    # 知识迁移
        distillation_epochs=100  # 知识蒸馏
    )
)
```

#### 3. MultiFidelityTrainer 类

```python
from ensemble.multi_fidelity import create_multi_fidelity_system

# 创建训练器
trainer = create_multi_fidelity_system(config)

# 添加训练数据
trainer.add_training_data(0, X_low, y_low)      # 低保真度数据
trainer.add_training_data(1, X_high, y_high)    # 高保真度数据

# 添加验证数据
trainer.add_validation_data(0, X_val_low, y_val_low)
trainer.add_validation_data(1, X_val_high, y_val_high)

# 运行完整训练流程
training_summary = trainer.run_full_training(input_dim=6, output_dim=3)
```

### 使用示例

#### 基本工作流程

```python
import numpy as np
from ensemble.multi_fidelity import (
    FidelityLevel, 
    MultiFidelityConfig, 
    create_multi_fidelity_system
)

# 1. 定义保真度级别
low_fidelity = FidelityLevel(
    name="快速油藏仿真",
    level=0,
    description="使用简化PDE的快速仿真",
    computational_cost=1.0,
    accuracy=0.75,
    data_requirements=800,
    training_time=90.0,
    model_type="neural_network"
)

high_fidelity = FidelityLevel(
    name="精确油藏仿真",
    level=1,
    description="使用完整物理模型的精确仿真",
    computational_cost=8.0,
    accuracy=0.92,
    data_requirements=4000,
    training_time=480.0,
    model_type="neural_network"
)

# 2. 创建配置
config = MultiFidelityConfig(
    name="油藏预测多保真度系统",
    description="结合快速和精确仿真的油藏生产预测系统",
    fidelity_levels=[low_fidelity, high_fidelity],
    co_training=MultiFidelityConfig.co_training(
        enabled=True,
        transfer_learning=True,
        knowledge_distillation=True,
        ensemble_method='weighted_average'
    )
)

# 3. 创建训练器
trainer = create_multi_fidelity_system(config)

# 4. 准备数据
# 低保真度数据（快速仿真结果）
X_low = np.random.randn(800, 6)   # 6个输入特征
y_low = np.random.randn(800, 3)   # 3个输出目标

# 高保真度数据（精确仿真结果）
X_high = np.random.randn(4000, 6)
y_high = np.random.randn(4000, 3)

# 5. 添加数据
trainer.add_training_data(0, X_low, y_low)
trainer.add_training_data(1, X_high, y_high)

# 6. 运行训练
training_summary = trainer.run_full_training(input_dim=6, output_dim=3)

# 7. 使用集成模型预测
X_test = np.random.randn(100, 6)
predictions = trainer.predict_with_ensemble(X_test)
```

#### 高级用法：自定义训练策略

```python
# 自定义训练策略
class CustomMultiFidelityTrainer(MultiFidelityTrainer):
    def custom_co_training(self):
        """自定义协同训练策略"""
        # 实现特定的协同训练逻辑
        pass
    
    def custom_ensemble_method(self, predictions):
        """自定义集成方法"""
        # 实现特定的集成策略
        pass

# 使用自定义训练器
custom_trainer = CustomMultiFidelityTrainer(config)
```

### 训练流程详解

#### 阶段1：低保真度预训练

```python
# 自动执行低保真度预训练
stage1_result = trainer.stage1_low_fidelity_pretraining()

print(f"阶段1完成:")
print(f"  训练轮数: {stage1_result['total_epochs']}")
print(f"  最终训练损失: {stage1_result['final_train_loss']:.6f}")
print(f"  最终验证损失: {stage1_result['final_val_loss']:.6f}")
```

#### 阶段2：高保真度微调

```python
# 自动执行高保真度微调
stage2_result = trainer.stage2_high_fidelity_finetuning()

print(f"阶段2完成:")
print(f"  训练轮数: {stage2_result['total_epochs']}")
print(f"  最终训练损失: {stage2_result['final_train_loss']:.6f}")
print(f"  最终验证损失: {stage2_result['final_val_loss']:.6f}")
```

#### 协同训练

```python
# 执行协同训练
co_training_result = trainer.co_training()

print(f"协同训练完成:")
print(f"  知识迁移: {'启用' if config.co_training.transfer_learning else '禁用'}")
print(f"  知识蒸馏: {'启用' if config.co_training.knowledge_distillation else '禁用'}")
print(f"  集成方法: {config.co_training.ensemble_method}")
```

### 性能监控

#### 训练进度监控

```python
# 获取训练进度
progress = trainer.training_progress

print("训练进度:")
for stage, result in progress.items():
    print(f"  {stage}: {result}")

# 获取模型性能摘要
performance = trainer._get_model_performance_summary()
for level, info in performance.items():
    print(f"  {level}: {info['model_type']}, 训练完成: {info['training_completed']}")
```

#### 集成性能评估

```python
# 获取最终评估结果
final_evaluation = trainer._final_evaluation()

print("最终评估结果:")
for level, metrics in final_evaluation.items():
    if level != 'ensemble':
        print(f"  {level}: MSE={metrics.get('mse', 'N/A'):.6f}, "
              f"R²={metrics.get('r2', 'N/A'):.4f}")
    else:
        print(f"  {level}: MSE={metrics.get('ensemble_mse', 'N/A'):.6f}, "
              f"R²={metrics.get('ensemble_r2', 'N/A'):.4f}")
```

## 集成使用

### 结合自适应约束和多保真度建模

```python
from gpu_acceleration.geological_ml_framework import (
    PhysicsInformedNeuralNetwork, 
    AdaptivePhysicalConstraint
)
from ensemble.multi_fidelity import (
    FidelityLevel, 
    MultiFidelityConfig, 
    create_multi_fidelity_system
)

# 1. 创建多保真度配置
low_fidelity = FidelityLevel(
    name="快速仿真",
    level=0,
    model_type="neural_network",
    model_params={'hidden_layers': [32, 16]}
)

high_fidelity = FidelityLevel(
    name="精确仿真",
    level=1,
    model_type="neural_network",
    model_params={'hidden_layers': [128, 64, 32]}
)

config = MultiFidelityConfig(
    name="自适应约束多保真度系统",
    fidelity_levels=[low_fidelity, high_fidelity],
    co_training=MultiFidelityConfig.co_training(enabled=True)
)

# 2. 创建多保真度训练器
trainer = create_multi_fidelity_system(config)

# 3. 为每个保真度级别创建带自适应约束的PINN
for fidelity in config.fidelity_levels:
    pinn = PhysicsInformedNeuralNetwork(
        input_dim=6,
        output_dim=3,
        hidden_layers=fidelity.model_params['hidden_layers'],
        adaptive_constraints=True
    )
    
    # 添加物理约束
    darcy_constraint = AdaptivePhysicalConstraint(
        name=f"Darcy方程_{fidelity.name}",
        equation=lambda x, y: np.random.normal(0, 1e-6),
        initial_weight=1.0
    )
    
    pinn.add_physical_constraint(darcy_constraint)
    pinn.setup_constraint_controller()
    
    # 将PINN关联到训练器
    trainer.models[fidelity.level] = pinn

# 4. 运行训练
training_summary = trainer.run_full_training(input_dim=6, output_dim=3)
```

### 实时监控和调试

```python
from core import create_debug_manager

# 创建调试管理器
debug_manager = create_debug_manager()

# 添加物理约束监控
def monitor_constraints():
    for level, model in trainer.models.items():
        if hasattr(model, 'physical_constraints'):
            for constraint in model.physical_constraints:
                debug_manager.add_physical_constraint(
                    name=f"Level_{level}_{constraint.name}",
                    equation=constraint.compute_residual,
                    weight=constraint.current_weight
                )

# 启动监控
debug_manager.start_debugging()
debug_manager.create_dashboards()

# 运行训练
training_summary = trainer.run_full_training(input_dim=6, output_dim=3)

# 停止监控
debug_manager.stop_debugging()
```

## 性能优化

### 计算资源优化

#### 1. 并行训练

```python
# 启用并行训练
config.performance.parallel_training = True

# 设置进程数
import multiprocessing
num_processes = multiprocessing.cpu_count()
print(f"使用 {num_processes} 个进程进行并行训练")
```

#### 2. GPU加速

```python
# 检查GPU可用性
import torch
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("使用CPU训练")

# 在模型中使用GPU
for model in trainer.models.values():
    if hasattr(model, 'model') and hasattr(model.model, 'to'):
        model.model.to(device)
```

#### 3. 内存优化

```python
# 启用内存优化
config.performance.memory_optimization = True

# 设置分块大小
config.performance.chunk_size = 500

# 启用压缩
config.performance.compression = True
```

### 训练策略优化

#### 1. 早停策略

```python
# 启用早停
config.performance.early_stopping = True

# 设置早停参数
for fidelity in config.fidelity_levels:
    if fidelity.model_type == 'neural_network':
        fidelity.training_params['early_stopping_patience'] = 10
        fidelity.training_params['min_delta'] = 1e-6
```

#### 2. 学习率调度

```python
# 启用自适应学习率
config.performance.adaptive_learning_rate = True

# 设置学习率调度器
for fidelity in config.fidelity_levels:
    if fidelity.model_type == 'neural_network':
        fidelity.training_params['lr_scheduler'] = 'reduce_on_plateau'
        fidelity.training_params['lr_patience'] = 5
        fidelity.training_params['lr_factor'] = 0.5
```

#### 3. 批量大小优化

```python
# 动态批量大小
for fidelity in config.fidelity_levels:
    if fidelity.model_type == 'neural_network':
        # 根据数据大小调整批量大小
        data_size = fidelity.data_requirements
        if data_size < 1000:
            fidelity.training_params['batch_size'] = 16
        elif data_size < 5000:
            fidelity.training_params['batch_size'] = 32
        else:
            fidelity.training_params['batch_size'] = 64
```

## 最佳实践

### 1. 约束设计

- **物理合理性**: 确保约束方程符合物理规律
- **数值稳定性**: 避免约束导致数值不稳定
- **权重平衡**: 合理设置初始权重和调整范围

### 2. 保真度级别设计

- **成本效益**: 平衡计算成本和精度要求
- **数据一致性**: 确保不同保真度级别的数据格式一致
- **模型兼容性**: 选择兼容的模型架构便于知识迁移

### 3. 训练策略

- **渐进训练**: 从低保真度开始，逐步提高精度
- **验证监控**: 持续监控验证性能，避免过拟合
- **早停机制**: 合理设置早停条件，节省计算资源

### 4. 集成方法

- **权重分配**: 基于精度和可靠性分配集成权重
- **多样性**: 选择不同架构的模型提高集成效果
- **稳定性**: 使用鲁棒的集成方法减少异常值影响

## 故障排除

### 常见问题

#### 1. 约束权重发散

**症状**: 约束权重不断增大或减小
**原因**: 调整速率过高或约束方程不稳定
**解决方案**:
```python
# 降低调整速率
constraint.adaptation_rate = 0.05  # 从0.1降低到0.05

# 限制权重范围
constraint.min_weight = 0.001
constraint.max_weight = 2.0

# 检查约束方程稳定性
def stable_constraint(x, y_pred):
    try:
        residual = compute_residual(x, y_pred)
        return np.clip(residual, -1e3, 1e3)  # 限制残差范围
    except:
        return 0.0
```

#### 2. 多保真度训练不收敛

**症状**: 高保真度模型训练损失不下降
**原因**: 知识迁移失败或数据不匹配
**解决方案**:
```python
# 检查数据格式一致性
print(f"低保真度数据形状: {X_low.shape}")
print(f"高保真度数据形状: {X_high.shape}")

# 启用渐进训练
config.training_strategy.stage1_epochs = 2000  # 增加预训练轮数
config.training_strategy.stage2_epochs = 1000  # 增加微调轮数

# 调整知识迁移参数
config.co_training.transfer_learning = True
config.training_strategy.transfer_epochs = 500
```

#### 3. 内存不足

**症状**: 训练过程中出现内存错误
**原因**: 模型过大或批量大小过大
**解决方案**:
```python
# 减少模型大小
for fidelity in config.fidelity_levels:
    if fidelity.model_type == 'neural_network':
        fidelity.model_params['hidden_layers'] = [32, 16]  # 减少隐藏层

# 减少批量大小
fidelity.training_params['batch_size'] = 16

# 启用梯度累积
fidelity.training_params['gradient_accumulation_steps'] = 4
```

#### 4. 训练速度慢

**症状**: 训练时间过长
**原因**: 模型复杂度过高或优化器设置不当
**解决方案**:
```python
# 使用更高效的优化器
fidelity.training_params['optimizer'] = 'adamw'
fidelity.training_params['learning_rate'] = 0.001

# 启用混合精度训练
if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler
    # 在训练循环中使用混合精度

# 减少验证频率
fidelity.training_params['validation_frequency'] = 5
```

### 调试技巧

#### 1. 启用详细日志

```python
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 在训练过程中添加详细日志
for epoch in range(epochs):
    if epoch % 10 == 0:
        logging.debug(f"Epoch {epoch}: Loss = {loss.item():.6f}")
        logging.debug(f"Constraint weights: {[c.current_weight for c in constraints]}")
```

#### 2. 可视化监控

```python
# 实时绘制训练曲线
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_training_progress():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    def update(frame):
        # 更新训练曲线
        pass
    
    ani = FuncAnimation(fig, update, interval=1000)
    plt.show()
```

#### 3. 性能分析

```python
import cProfile
import pstats

# 性能分析
profiler = cProfile.Profile()
profiler.enable()

# 运行训练
training_summary = trainer.run_full_training(input_dim=6, output_dim=3)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 显示前20个最耗时的函数
```

## 总结

自适应物理约束和多保真度建模为GeoSim库提供了强大的功能扩展：

### 主要优势

1. **智能约束管理**: 自动调整物理约束强度，提高PINN收敛性
2. **计算效率**: 通过多保真度建模减少高精度仿真的计算成本
3. **知识迁移**: 在模型间有效传递学习到的特征
4. **鲁棒性**: 集成多个保真度级别提高预测稳定性

### 应用场景

- **油藏模拟**: 结合快速和精确的流体流动模型
- **地质建模**: 多尺度地质结构建模
- **地震反演**: 不同分辨率的波形反演
- **环境模拟**: 多精度的大气和水文模型

### 发展方向

1. **自动化配置**: 基于问题特征自动选择最优配置
2. **在线学习**: 在仿真过程中持续改进模型
3. **分布式训练**: 支持大规模分布式多保真度训练
4. **不确定性量化**: 集成预测的不确定性评估

通过合理使用这些功能，可以显著提高地质数值模拟的效率和准确性，为复杂地质问题的求解提供强有力的工具支持。
