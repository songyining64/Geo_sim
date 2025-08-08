"""
机器学习优化模块 - 提供神经网络求解器和代理模型
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Callable, Union
import warnings

# 深度学习相关依赖
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    torch = None
    nn = None
    optim = None
    warnings.warn("PyTorch not available. Deep learning features will be limited.")

try:
    import sklearn
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    GaussianProcessRegressor = None
    warnings.warn("scikit-learn not available. Surrogate models will be limited.")


class NeuralNetworkSolver(nn.Module):
    """神经网络求解器 - 基础版本"""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, 
                 activation: str = 'relu', dropout: float = 0.1):
        super(NeuralNetworkSolver, self).__init__()
        
        if not HAS_PYTORCH:
            raise ImportError("需要安装PyTorch来使用神经网络求解器")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        self.optimizer = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        print(f"🔄 神经网络求解器初始化完成 - 设备: {self.device}")
    
    def forward(self, x):
        return self.network(x)
    
    def setup_training(self, learning_rate: float = 0.001, weight_decay: float = 1e-5):
        """设置训练参数"""
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
    
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
                   batch_size: int = 32, validation_split: float = 0.2) -> dict:
        """训练模型"""
        if self.optimizer is None:
            self.setup_training()
        
        # 数据预处理
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # 划分训练集和验证集
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_tensor, y_tensor, test_size=validation_split, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = X_tensor, None, y_tensor, None
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        history = {'train_loss': [], 'val_loss': [], 'train_time': 0.0}
        start_time = time.time()
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            # 计算平均损失
            avg_train_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # 验证损失
            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self(X_val)
                    val_loss = self.criterion(val_outputs, y_val)
                    history['val_loss'].append(val_loss.item())
            
            if (epoch + 1) % 10 == 0:
                val_info = f", val_loss={history['val_loss'][-1]:.6f}" if X_val is not None else ""
                print(f"   Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.6f}{val_info}")
        
        history['train_time'] = time.time() - start_time
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self(X_tensor)
            return outputs.cpu().numpy()
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save(self.state_dict(), filepath)
        print(f"📁 模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        self.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"📁 模型已加载: {filepath}")


class UNetSolver(nn.Module):
    """U-Net求解器 - 用于空间场预测"""
    
    def __init__(self, input_channels: int, output_channels: int, 
                 initial_features: int = 64, depth: int = 4):
        super(UNetSolver, self).__init__()
        
        if not HAS_PYTORCH:
            raise ImportError("需要安装PyTorch来使用U-Net求解器")
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.initial_features = initial_features
        self.depth = depth
        
        # 编码器
        self.encoder = nn.ModuleList()
        in_channels = input_channels
        
        for i in range(depth):
            out_channels = initial_features * (2 ** i)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels
        
        # 解码器
        self.decoder = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            out_channels = initial_features * (2 ** i)
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels * 2, out_channels, 2, stride=2),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels
        
        # 最终输出层
        self.final_conv = nn.Conv2d(initial_features, output_channels, 1)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        print(f"🔄 U-Net求解器初始化完成 - 设备: {self.device}")
    
    def forward(self, x):
        # 编码
        encoder_outputs = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_outputs.append(x)
            x = F.max_pool2d(x, 2)
        
        # 解码
        for i, decoder_layer in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat([x, encoder_outputs[-(i+1)]], dim=1)
            x = decoder_layer(x)
        
        return self.final_conv(x)
    
    def setup_training(self, learning_rate: float = 0.001):
        """设置训练参数"""
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
                   batch_size: int = 8) -> dict:
        """训练模型"""
        if self.optimizer is None:
            self.setup_training()
        
        # 数据预处理 - 确保是4D张量 (batch, channels, height, width)
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        if y.ndim == 3:
            y = y[:, np.newaxis, :, :]
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        history = {'train_loss': [], 'train_time': 0.0}
        start_time = time.time()
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")
        
        history['train_time'] = time.time() - start_time
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.eval()
        with torch.no_grad():
            if X.ndim == 3:
                X = X[:, np.newaxis, :, :]
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self(X_tensor)
            return outputs.cpu().numpy()


class PINNSolver(nn.Module):
    """物理信息神经网络（PINN）求解器"""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 physics_equations: List[Callable] = None):
        super(PINNSolver, self).__init__()
        
        if not HAS_PYTORCH:
            raise ImportError("需要安装PyTorch来使用PINN求解器")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.physics_equations = physics_equations or []
        
        # 构建网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())  # PINN通常使用Tanh激活函数
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        print(f"🔄 PINN求解器初始化完成 - 设备: {self.device}")
    
    def forward(self, x):
        return self.network(x)
    
    def setup_training(self, learning_rate: float = 0.001):
        """设置训练参数"""
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def compute_physics_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算物理约束损失"""
        if not self.physics_equations:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        for equation in self.physics_equations:
            # 计算物理方程的残差
            residual = equation(x, y)
            total_loss += torch.mean(residual ** 2)
        
        return total_loss
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   physics_points: np.ndarray = None,
                   epochs: int = 1000, 
                   physics_weight: float = 1.0) -> dict:
        """训练PINN模型"""
        if self.optimizer is None:
            self.setup_training()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        if physics_points is not None:
            physics_tensor = torch.FloatTensor(physics_points).to(self.device)
        
        history = {'total_loss': [], 'data_loss': [], 'physics_loss': [], 'train_time': 0.0}
        start_time = time.time()
        
        for epoch in range(epochs):
            self.train()
            self.optimizer.zero_grad()
            
            # 数据损失
            outputs = self(X_tensor)
            data_loss = F.mse_loss(outputs, y_tensor)
            
            # 物理损失
            if physics_points is not None and self.physics_equations:
                physics_outputs = self(physics_tensor)
                physics_loss = self.compute_physics_loss(physics_tensor, physics_outputs)
            else:
                physics_loss = torch.tensor(0.0, device=self.device)
            
            # 总损失
            total_loss = data_loss + physics_weight * physics_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            history['total_loss'].append(total_loss.item())
            history['data_loss'].append(data_loss.item())
            history['physics_loss'].append(physics_loss.item())
            
            if (epoch + 1) % 100 == 0:
                print(f"   Epoch {epoch+1}/{epochs}: total_loss={total_loss.item():.6f}, "
                      f"data_loss={data_loss.item():.6f}, physics_loss={physics_loss.item():.6f}")
        
        history['train_time'] = time.time() - start_time
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self(X_tensor)
            return outputs.cpu().numpy()


class SurrogateModel:
    """代理模型 - 用于快速预测"""
    
    def __init__(self, model_type: str = 'gaussian_process'):
        self.model_type = model_type
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained = False
        
        if model_type == 'gaussian_process' and not HAS_SKLEARN:
            raise ImportError("需要安装scikit-learn来使用高斯过程回归")
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
        """训练代理模型"""
        start_time = time.time()
        
        # 数据标准化
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        if self.model_type == 'gaussian_process':
            # 高斯过程回归
            kernel = RBF(length_scale=1.0) + ConstantKernel()
            self.model = GaussianProcessRegressor(kernel=kernel, random_state=42)
            self.model.fit(X_scaled, y_scaled)
        
        self.is_trained = True
        training_time = time.time() - start_time
        
        return {'training_time': training_time, 'model_type': self.model_type}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X_scaled = self.scaler_X.transform(X)
        
        if self.model_type == 'gaussian_process':
            y_pred_scaled, std = self.model.predict(X_scaled, return_std=True)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            return y_pred, std
        
        return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()


class MLAccelerator:
    """机器学习加速器 - 主控制器"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and HAS_PYTORCH
        self.models = {}
        self.surrogate_models = {}
        self.performance_stats = {
            'total_training_time': 0.0,
            'total_prediction_time': 0.0,
            'models_created': 0,
            'predictions_made': 0
        }
        
        print(f"🔄 ML加速器初始化完成 - GPU: {self.use_gpu}")
    
    def create_neural_solver(self, name: str, input_dim: int, hidden_dims: list, 
                           output_dim: int, solver_type: str = 'basic') -> NeuralNetworkSolver:
        """创建神经网络求解器"""
        if solver_type == 'basic':
            solver = NeuralNetworkSolver(input_dim, hidden_dims, output_dim)
        elif solver_type == 'unet':
            solver = UNetSolver(input_dim, output_dim)
        else:
            raise ValueError(f"不支持的求解器类型: {solver_type}")
        
        self.models[name] = solver
        self.performance_stats['models_created'] += 1
        
        print(f"✅ 神经网络求解器 '{name}' 创建完成")
        return solver
    
    def create_pinn_solver(self, name: str, input_dim: int, hidden_dims: list, 
                          output_dim: int, physics_equations: List[Callable] = None) -> PINNSolver:
        """创建PINN求解器"""
        solver = PINNSolver(input_dim, hidden_dims, output_dim, physics_equations)
        self.models[name] = solver
        self.performance_stats['models_created'] += 1
        
        print(f"✅ PINN求解器 '{name}' 创建完成")
        return solver
    
    def create_surrogate_model(self, name: str, model_type: str = 'gaussian_process') -> SurrogateModel:
        """创建代理模型"""
        surrogate = SurrogateModel(model_type)
        self.surrogate_models[name] = surrogate
        self.performance_stats['models_created'] += 1
        
        print(f"✅ 代理模型 '{name}' 创建完成")
        return surrogate
    
    def train_model(self, name: str, X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
        """训练模型"""
        if name in self.models:
            model = self.models[name]
            if isinstance(model, PINNSolver):
                physics_points = kwargs.get('physics_points', None)
                physics_weight = kwargs.get('physics_weight', 1.0)
                result = model.train_model(X, y, physics_points, **kwargs)
            else:
                result = model.train_model(X, y, **kwargs)
        elif name in self.surrogate_models:
            model = self.surrogate_models[name]
            result = model.train(X, y, **kwargs)
        else:
            raise ValueError(f"模型 '{name}' 不存在")
        
        self.performance_stats['total_training_time'] += result.get('training_time', 0.0)
        return result
    
    def predict(self, name: str, X: np.ndarray) -> np.ndarray:
        """预测"""
        start_time = time.time()
        
        if name in self.models:
            result = self.models[name].predict(X)
        elif name in self.surrogate_models:
            result = self.surrogate_models[name].predict(X)
        else:
            raise ValueError(f"模型 '{name}' 不存在")
        
        prediction_time = time.time() - start_time
        self.performance_stats['total_prediction_time'] += prediction_time
        self.performance_stats['predictions_made'] += 1
        
        return result
    
    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        return self.performance_stats.copy()
    
    def save_model(self, name: str, filepath: str):
        """保存模型"""
        if name in self.models:
            self.models[name].save_model(filepath)
        else:
            raise ValueError(f"模型 '{name}' 不存在")
    
    def load_model(self, name: str, filepath: str):
        """加载模型"""
        if name in self.models:
            self.models[name].load_model(filepath)
        else:
            raise ValueError(f"模型 '{name}' 不存在")


def create_ml_accelerator(use_gpu: bool = True) -> MLAccelerator:
    """创建ML加速器"""
    return MLAccelerator(use_gpu=use_gpu)


def demo_ml_optimization():
    """演示机器学习优化功能"""
    print("🤖 机器学习优化演示")
    print("=" * 50)
    
    # 创建ML加速器
    accelerator = create_ml_accelerator()
    
    # 生成测试数据
    n_samples = 1000
    input_dim = 10
    output_dim = 5
    
    X = np.random.randn(n_samples, input_dim)
    y = np.random.randn(n_samples, output_dim)
    
    print(f"📊 测试数据: {n_samples} 样本, 输入维度: {input_dim}, 输出维度: {output_dim}")
    
    # 1. 测试基础神经网络求解器
    print("\n🔧 测试基础神经网络求解器...")
    try:
        neural_solver = accelerator.create_neural_solver(
            'basic_solver', input_dim, [64, 32], output_dim
        )
        neural_solver.setup_training()
        result = accelerator.train_model('basic_solver', X, y, epochs=50)
        print(f"   训练时间: {result['train_time']:.4f} 秒")
        
        # 预测
        predictions = accelerator.predict('basic_solver', X[:10])
        print(f"   预测形状: {predictions.shape}")
        
    except Exception as e:
        print(f"   ❌ 基础神经网络求解器失败: {e}")
    
    # 2. 测试U-Net求解器
    print("\n🔧 测试U-Net求解器...")
    try:
        # 生成2D数据
        n_samples_2d = 100
        height, width = 32, 32
        X_2d = np.random.randn(n_samples_2d, height, width)
        y_2d = np.random.randn(n_samples_2d, height, width)
        
        unet_solver = accelerator.create_neural_solver(
            'unet_solver', 1, [64, 32], 1, solver_type='unet'
        )
        unet_solver.setup_training()
        result = accelerator.train_model('unet_solver', X_2d, y_2d, epochs=20)
        print(f"   训练时间: {result['train_time']:.4f} 秒")
        
    except Exception as e:
        print(f"   ❌ U-Net求解器失败: {e}")
    
    # 3. 测试PINN求解器
    print("\n🔧 测试PINN求解器...")
    try:
        # 定义简单的物理方程（示例：热传导方程）
        def heat_equation(x, y):
            # 简化的热传导方程残差
            return torch.mean(torch.abs(y - 0.1 * torch.sum(x, dim=1, keepdim=True)))
        
        pinn_solver = accelerator.create_pinn_solver(
            'pinn_solver', input_dim, [64, 32], output_dim, [heat_equation]
        )
        pinn_solver.setup_training()
        result = accelerator.train_model('pinn_solver', X, y, epochs=100)
        print(f"   训练时间: {result['train_time']:.4f} 秒")
        
    except Exception as e:
        print(f"   ❌ PINN求解器失败: {e}")
    
    # 4. 测试代理模型
    print("\n🔧 测试代理模型...")
    try:
        surrogate = accelerator.create_surrogate_model('gpr_model', 'gaussian_process')
        result = accelerator.train_model('gpr_model', X, y[:, 0])  # 只预测第一个输出
        print(f"   训练时间: {result['training_time']:.4f} 秒")
        
        # 预测
        predictions, std = accelerator.predict('gpr_model', X[:10])
        print(f"   预测形状: {predictions.shape}, 标准差形状: {std.shape}")
        
    except Exception as e:
        print(f"   ❌ 代理模型失败: {e}")
    
    # 显示性能统计
    stats = accelerator.get_performance_stats()
    print(f"\n📈 性能统计:")
    print(f"   总训练时间: {stats['total_training_time']:.4f} 秒")
    print(f"   总预测时间: {stats['total_prediction_time']:.4f} 秒")
    print(f"   创建的模型数: {stats['models_created']}")
    print(f"   预测次数: {stats['predictions_made']}")
    
    print("\n✅ 机器学习优化演示完成!")


if __name__ == "__main__":
    demo_ml_optimization() 