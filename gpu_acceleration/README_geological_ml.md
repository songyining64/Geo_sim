# 地质数值模拟ML/DL融合框架

## 概述

本框架基于Underworld2的设计理念，专门针对地质数值模拟的核心挑战，实现了机器学习（ML）和深度学习（DL）技术与传统数值模拟的有效融合。

### 核心挑战

1. **多尺度性**：从微观孔隙（微米级）到宏观油藏（千米级）
2. **强非线性**：复杂地质条件下的物理过程
3. **数据稀疏性**：昂贵的地质数据获取
4. **物理可解释性**：必须遵循地质物理规律

### 核心思路

**用ML解决传统方法的效率瓶颈，同时用地质物理约束保证模型可靠性**

## 主要功能

### 1. 物理信息神经网络（PINN）

**文件**：`geological_ml_framework.py` - `GeologicalPINN`

**核心思想**：将地质物理方程（如达西定律、热传导方程）作为软约束嵌入神经网络，强制模型输出满足地质物理规律，实现"小数据+强物理"的地质建模。

**应用场景**：
- 油藏压力场预测
- 热传导模拟
- 岩石力学分析

**示例**：
```python
from geological_ml_framework import GeologicalPINN, GeologicalPhysicsEquations

# 创建PINN模型
pinn = GeologicalPINN(
    input_dim=4,  # x, y, z, t
    hidden_dims=[128, 64, 32],
    output_dim=1,  # 压力场
    physics_equations=[GeologicalPhysicsEquations.darcy_equation]
)

# 训练模型
result = pinn.train(X, y, epochs=500, physics_weight=1.0)
```

### 2. 地质代理模型（Surrogate Model）

**文件**：`geological_ml_framework.py` - `GeologicalSurrogateModel`

**核心思想**：用传统地质数值模拟生成"地质参数→模拟输出"的数据集，训练ML模型学习这种映射，后续用模型直接预测，替代完整模拟流程。

**应用场景**：
- 参数反演加速
- 敏感性分析
- 不确定性量化

**示例**：
```python
from geological_ml_framework import GeologicalSurrogateModel

# 创建代理模型
surrogate = GeologicalSurrogateModel('gaussian_process')

# 训练模型
result = surrogate.train(X_params, y_output)

# 预测（支持不确定性）
predictions, std = surrogate.predict(new_params, return_std=True)
```

### 3. 地质UNet（空间场处理）

**文件**：`geological_ml_framework.py` - `GeologicalUNet`

**核心思想**：用UNet实现"低精度地质数据→高精度地质场"的端到端映射，如地震数据反演地质结构、地质场超分辨率重建等。

**应用场景**：
- 地震数据反演
- 地质场超分辨率
- 储层表征

**示例**：
```python
from geological_ml_framework import GeologicalUNet

# 创建UNet模型
unet = GeologicalUNet(
    input_channels=1,  # 地震数据
    output_channels=1,  # 孔隙度场
    initial_features=64,
    depth=4
)

# 训练模型
result = unet.train_model(seismic_data, porosity_data, epochs=100)
```

### 4. 多尺度桥接器（Multi-Scale Bridge）

**文件**：`geological_ml_framework.py` - `GeologicalMultiScaleBridge`

**核心思想**：在跨尺度地质模拟中（如从微观孔隙到宏观油藏），用ML模型替代小尺度精细模拟，将小尺度结果"打包"为大尺度模型的参数。

**应用场景**：
- 微观-宏观参数映射
- 跨尺度模拟加速
- 等效参数计算

**示例**：
```python
from geological_ml_framework import GeologicalMultiScaleBridge

# 创建桥接模型
bridge = GeologicalMultiScaleBridge()
bridge.setup_bridge_model(
    input_dim=5,  # 微观参数
    output_dim=1,  # 宏观渗透率
    model_type='neural_network'
)

# 训练桥接
result = bridge.train_bridge(micro_data, macro_data)
```

### 5. 混合加速器（Hybrid Accelerator）

**文件**：`geological_ml_framework.py` - `GeologicalHybridAccelerator`

**核心思想**：无法完全替代传统地质模拟（需高精度），但可加速其中耗时步骤（如迭代求解、网格自适应）。

**应用场景**：
- 初始猜测加速
- 预条件子学习
- 自适应网格优化

**示例**：
```python
from geological_ml_framework import GeologicalHybridAccelerator

# 创建混合加速器
hybrid = GeologicalHybridAccelerator(traditional_solver=reservoir_simulator)

# 添加ML模型
hybrid.add_ml_model('pinn_initial', pinn_model)
hybrid.setup_acceleration_strategy('initial_guess', 'pinn_initial')

# 混合求解
result = hybrid.solve_hybrid(problem_data)
```

## 应用示例

### 1. 油藏模拟示例

**文件**：`geological_examples.py` - `ReservoirSimulationExample`

**功能**：用PINN求解达西方程，预测油藏压力场分布。

**特点**：
- 物理约束：达西定律
- 边界条件：井位约束
- 时间演化：压力场随时间变化

**使用方法**：
```python
from geological_examples import ReservoirSimulationExample

# 创建示例
reservoir_example = ReservoirSimulationExample()

# 运行PINN模拟
result = reservoir_example.run_pinn_simulation()

# 可视化结果
reservoir_example.visualize_results(result['model'])
```

### 2. 地震反演示例

**文件**：`geological_examples.py` - `SeismicInversionExample`

**功能**：用UNet从地震数据反演地质结构（如孔隙度场）。

**特点**：
- 端到端映射：地震数据→地质属性
- 空间相关性：利用UNet的卷积结构
- 噪声鲁棒性：处理地震数据噪声

**使用方法**：
```python
from geological_examples import SeismicInversionExample

# 创建示例
seismic_example = SeismicInversionExample()

# 运行UNet反演
result = seismic_example.run_unet_inversion()

# 可视化结果
seismic_example.visualize_inversion_results(
    result['model'], result['seismic_data'], result['geological_data']
)
```

### 3. 多尺度建模示例

**文件**：`geological_examples.py` - `MultiScaleModelingExample`

**功能**：用桥接模型连接微观和宏观尺度，从微观孔隙参数预测宏观渗透率。

**特点**：
- 跨尺度映射：微观→宏观
- 物理基础：基于Kozeny-Carman方程
- 参数化建模：多参数输入

**使用方法**：
```python
from geological_examples import MultiScaleModelingExample

# 创建示例
multiscale_example = MultiScaleModelingExample()

# 运行多尺度桥接
result = multiscale_example.run_multiscale_bridge()

# 可视化结果
multiscale_example.visualize_bridge_results(
    result['bridge'], result['micro_data'], result['macro_data']
)
```

### 4. 参数反演示例

**文件**：`geological_examples.py` - `ParameterInversionExample`

**功能**：用代理模型加速参数优化，从产量数据反演地质参数。

**特点**：
- 快速优化：代理模型替代昂贵模拟
- 不确定性量化：预测标准差
- 多参数反演：同时优化多个参数

**使用方法**：
```python
from geological_examples import ParameterInversionExample

# 创建示例
inversion_example = ParameterInversionExample()

# 运行参数反演
result = inversion_example.run_parameter_inversion()

# 可视化结果
inversion_example.visualize_inversion_results(
    result['surrogate'], result['optimal_params'], result['target_production']
)
```

## 安装和依赖

### 必需依赖

```bash
pip install torch torchvision torchaudio
pip install scikit-learn
pip install numpy matplotlib seaborn
pip install scipy
```

### 可选依赖

```bash
# 用于高级可视化
pip install plotly
pip install ipywidgets

# 用于并行计算
pip install joblib
pip install mpi4py
```

## 使用方法

### 1. 基本使用

```python
# 导入框架
from geological_ml_framework import create_geological_ml_system

# 创建ML系统
ml_system = create_geological_ml_system()

# 使用PINN
pinn = ml_system['pinn'](input_dim=4, hidden_dims=[64, 32], output_dim=1)
result = pinn.train(X, y, epochs=200)
```

### 2. 高级使用

```python
# 自定义物理方程
def custom_physics_equation(x, y, config):
    # 实现自定义物理约束
    return residual

# 创建带自定义物理约束的PINN
pinn = GeologicalPINN(
    input_dim=4,
    hidden_dims=[128, 64, 32],
    output_dim=1,
    physics_equations=[custom_physics_equation]
)
```

### 3. 完整示例

```python
# 运行所有示例
from geological_examples import run_all_examples

# 执行所有示例
run_all_examples()
```

## 性能优化

### 1. GPU加速

框架自动检测GPU并优先使用：

```python
# 检查设备
print(f"使用设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

### 2. 并行计算

支持多进程训练：

```python
# 使用多进程
import multiprocessing as mp

# 设置进程数
mp.set_start_method('spawn', force=True)
```

### 3. 内存优化

- 批处理训练
- 梯度累积
- 模型检查点

## 扩展开发

### 1. 添加新的物理方程

```python
class CustomPhysicsEquations:
    @staticmethod
    def custom_equation(x: torch.Tensor, y: torch.Tensor, config: GeologicalConfig) -> torch.Tensor:
        # 实现自定义物理方程
        return residual
```

### 2. 添加新的模型类型

```python
class CustomModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # 实现自定义模型
    
    def forward(self, x):
        # 实现前向传播
        return output
```

### 3. 添加新的示例

```python
class CustomExample:
    def __init__(self):
        # 初始化示例
    
    def run_example(self):
        # 实现示例逻辑
        pass
```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 发送邮件
- 参与讨论

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 实现基础ML/DL框架
- 添加地质物理约束
- 提供完整示例

### v1.1.0 (计划中)
- 添加更多物理方程
- 优化性能
- 增加可视化功能
- 完善文档
