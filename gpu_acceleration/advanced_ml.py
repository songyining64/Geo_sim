"""
高级机器学习加速数值模拟框架
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 深度学习相关依赖
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.nn.parameter import Parameter
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    torch = None
    nn = None
    optim = None
    warnings.warn("PyTorch not available. Advanced ML features will be limited.")

try:
    import sklearn
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available. Advanced ML features will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Visualization features will be limited.")


@dataclass
class PhysicsConfig:
    """物理配置类"""
    # 地质模拟常用参数
    rayleigh_number: float = 1e6
    prandtl_number: float = 1.0
    gravity: float = 9.81
    thermal_expansion: float = 3e-5
    thermal_diffusivity: float = 1e-6
    reference_density: float = 3300.0
    reference_viscosity: float = 1e21
    
    # 边界条件
    boundary_conditions: Dict = None
    
    def __post_init__(self):
        if self.boundary_conditions is None:
            self.boundary_conditions = {
                'temperature': {'top': 0, 'bottom': 1},
                'velocity': {'top': 'free_slip', 'bottom': 'free_slip'},
                'pressure': {'top': 'free', 'bottom': 'free'}
            }


class BaseSolver(ABC):
    """基础求解器抽象类 - 实现代码复用"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.training_history = {}
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass
    
    def save_model(self, filepath: str):
        """保存模型"""
        try:
            if hasattr(self, 'state_dict'):
                torch.save(self.state_dict(), filepath)
                print(f"📁 模型已保存: {filepath}")
            else:
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(self, f)
                print(f"📁 模型已保存: {filepath}")
        except Exception as e:
            raise RuntimeError(f"保存模型失败：{str(e)}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        try:
            if hasattr(self, 'load_state_dict'):
                self.load_state_dict(torch.load(filepath, map_location=self.device))
                print(f"📁 模型已加载: {filepath}")
            else:
                import pickle
                with open(filepath, 'rb') as f:
                    loaded_model = pickle.load(f)
                    self.__dict__.update(loaded_model.__dict__)
                print(f"📁 模型已加载: {filepath}")
        except FileNotFoundError:
            raise FileNotFoundError(f"模型文件不存在：{filepath}")
        except Exception as e:
            raise RuntimeError(f"加载模型失败：{str(e)}")


class PhysicsInformedNeuralNetwork(BaseSolver, nn.Module):
    """
    物理信息神经网络（PINN） - 高级版本
    
    核心思想：将物理方程作为软约束嵌入神经网络的损失函数，
    强制模型输出满足物理规律，实现"小数据+物理知识"的联合学习
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 physics_equations: List[Callable] = None,
                 boundary_conditions: List[Callable] = None,
                 physics_config: PhysicsConfig = None):
        BaseSolver.__init__(self)
        nn.Module.__init__(self)
        
        if not HAS_PYTORCH:
            raise ImportError("需要安装PyTorch来使用PINN")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.physics_equations = physics_equations or []
        self.boundary_conditions = boundary_conditions or []
        self.physics_config = physics_config or PhysicsConfig()
        
        # 构建网络 - 使用残差连接和批归一化
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Tanh())  # PINN通常使用Tanh激活函数
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # 初始化权重
        self._initialize_weights()
        
        self.optimizer = None
        self.to(self.device)
        
        print(f"🔄 高级PINN初始化完成 - 设备: {self.device}")
        print(f"   网络结构: {input_dim} -> {hidden_dims} -> {output_dim}")
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
    
    def setup_training(self, learning_rate: float = 0.001, weight_decay: float = 1e-5):
        """设置训练参数"""
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
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
    
    def compute_boundary_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算边界条件损失"""
        if not self.boundary_conditions:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        for bc in self.boundary_conditions:
            # 计算边界条件的残差
            residual = bc(x, y)
            total_loss += torch.mean(residual ** 2)
        
        return total_loss
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              physics_points: np.ndarray = None,
              boundary_points: np.ndarray = None,
              epochs: int = 1000, 
              physics_weight: float = 1.0,
              boundary_weight: float = 1.0,
              batch_size: int = 32,
              validation_split: float = 0.2) -> dict:
        """训练PINN模型"""
        if self.optimizer is None:
            self.setup_training()
        
        # 验证输入数据
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X和y样本数不匹配：{X.shape[0]} vs {y.shape[0]}")
        
        # 数据分割
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        if physics_points is not None:
            physics_tensor = torch.FloatTensor(physics_points).to(self.device)
        
        if boundary_points is not None:
            boundary_tensor = torch.FloatTensor(boundary_points).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        history = {
            'total_loss': [], 'data_loss': [], 'physics_loss': [], 
            'boundary_loss': [], 'val_loss': [], 'train_time': 0.0
        }
        start_time = time.time()
        
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        for epoch in range(epochs):
            self.train()
            epoch_losses = {'total': 0, 'data': 0, 'physics': 0, 'boundary': 0}
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # 数据损失
                outputs = self(batch_X)
                data_loss = F.mse_loss(outputs, batch_y)
                
                # 物理损失
                if physics_points is not None and self.physics_equations:
                    physics_outputs = self(physics_tensor)
                    physics_loss = self.compute_physics_loss(physics_tensor, physics_outputs)
                else:
                    physics_loss = torch.tensor(0.0, device=self.device)
                
                # 边界损失
                if boundary_points is not None and self.boundary_conditions:
                    boundary_outputs = self(boundary_tensor)
                    boundary_loss = self.compute_boundary_loss(boundary_tensor, boundary_outputs)
                else:
                    boundary_loss = torch.tensor(0.0, device=self.device)
                
                # 总损失
                total_loss = data_loss + physics_weight * physics_loss + boundary_weight * boundary_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                # 累积损失
                epoch_losses['total'] += total_loss.item()
                epoch_losses['data'] += data_loss.item()
                epoch_losses['physics'] += physics_loss.item()
                epoch_losses['boundary'] += boundary_loss.item()
            
            # 计算平均损失
            num_batches = len(train_loader)
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
            
            # 验证损失
            val_loss = None
            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self(X_val_tensor)
                    val_loss = F.mse_loss(val_outputs, y_val_tensor).item()
                    history['val_loss'].append(val_loss)
                    
                    # 早停
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"   早停在第 {epoch+1} 轮")
                        break
            
            # 记录历史
            history['total_loss'].append(epoch_losses['total'])
            history['data_loss'].append(epoch_losses['data'])
            history['physics_loss'].append(epoch_losses['physics'])
            history['boundary_loss'].append(epoch_losses['boundary'])
            
            if (epoch + 1) % 100 == 0:
                val_info = f", val_loss={val_loss:.6f}" if val_loss is not None else ""
                print(f"   Epoch {epoch+1}/{epochs}: total_loss={epoch_losses['total']:.6f}, "
                      f"data_loss={epoch_losses['data']:.6f}, physics_loss={epoch_losses['physics']:.6f}, "
                      f"boundary_loss={epoch_losses['boundary']:.6f}{val_info}")
        
        history['train_time'] = time.time() - start_time
        self.is_trained = True
        self.training_history = history
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self(X_tensor)
            return outputs.cpu().numpy()


class SurrogateModelAdvanced(BaseSolver):
    """
    高级代理模型 - 支持多种算法
    
    核心思想：用传统数值模拟生成"输入参数→输出物理场"的数据集，
    训练DL模型学习这种映射，后续用模型直接预测，替代完整模拟流程
    """
    
    def __init__(self, model_type: str = 'gaussian_process', **kwargs):
        super().__init__()
        self.model_type = model_type
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.kwargs = kwargs
        
        if model_type == 'gaussian_process' and not HAS_SKLEARN:
            raise ImportError("需要安装scikit-learn来使用高斯过程回归")
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
        """训练代理模型"""
        # 验证输入数据合法性
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X和y样本数不匹配：{X.shape[0]} vs {y.shape[0]}")
        
        start_time = time.time()
        
        # 数据标准化
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        if self.model_type == 'gaussian_process':
            # 高斯过程回归
            kernel = self.kwargs.get('kernel', RBF(length_scale=1.0) + ConstantKernel())
            self.model = GaussianProcessRegressor(kernel=kernel, random_state=42)
            self.model.fit(X_scaled, y_scaled)
        
        elif self.model_type == 'random_forest':
            # 随机森林回归
            n_estimators = self.kwargs.get('n_estimators', 100)
            if not isinstance(n_estimators, int) or n_estimators <= 0:
                raise ValueError(f"n_estimators必须为正整数，实际为：{n_estimators}")
            
            max_depth = self.kwargs.get('max_depth', None)
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            self.model.fit(X_scaled, y_scaled)
        
        elif self.model_type == 'gradient_boosting':
            # 梯度提升回归
            n_estimators = self.kwargs.get('n_estimators', 100)
            if not isinstance(n_estimators, int) or n_estimators <= 0:
                raise ValueError(f"n_estimators必须为正整数，实际为：{n_estimators}")
            
            learning_rate = self.kwargs.get('learning_rate', 0.1)
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
                raise ValueError(f"learning_rate必须为正数，实际为：{learning_rate}")
            
            self.model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            self.model.fit(X_scaled, y_scaled)
        
        elif self.model_type == 'neural_network':
            # 神经网络回归
            hidden_layer_sizes = self.kwargs.get('hidden_layer_sizes', (100, 50))
            max_iter = self.kwargs.get('max_iter', 1000)
            if not isinstance(max_iter, int) or max_iter <= 0:
                raise ValueError(f"max_iter必须为正整数，实际为：{max_iter}")
            
            self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)
            self.model.fit(X_scaled, y_scaled)
        
        self.is_trained = True
        training_time = time.time() - start_time
        
        self.training_history = {
            'training_time': training_time,
            'model_type': self.model_type,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        return self.training_history
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        X_scaled = self.scaler_X.transform(X)
        
        if self.model_type == 'gaussian_process':
            if return_std:
                y_pred_scaled, std = self.model.predict(X_scaled, return_std=True)
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                return y_pred, std
            else:
                y_pred_scaled = self.model.predict(X_scaled)
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                return y_pred
        
        elif self.model_type in ['random_forest', 'gradient_boosting', 'neural_network']:
            y_pred_scaled = self.model.predict(X_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            if return_std:
                # 随机森林：通过树的预测差异估计标准差
                if self.model_type == 'random_forest':
                    tree_preds = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
                    std = np.std(tree_preds, axis=0)  # 树间预测标准差
                    std = self.scaler_y.inverse_transform(std.reshape(-1, 1)).flatten()  # 反归一化
                else:
                    std = np.zeros_like(y_pred)  # 其他模型暂不支持
                return y_pred, std
            return y_pred
        
        return y_pred


class MultiScaleMLBridge:
    """
    多尺度ML桥接器 - 用于跨尺度模拟
    
    核心思想：在跨尺度模拟中（如从板块运动到岩石变形），
    用ML模型替代小尺度精细模拟，将小尺度结果"打包"为大尺度模型的参数
    """
    
    def __init__(self, fine_scale_model: Callable = None, coarse_scale_model: Callable = None):
        self.fine_scale_model = fine_scale_model
        self.coarse_scale_model = coarse_scale_model
        self.bridge_model = None
        self.is_trained = False
        self.scale_ratio = 1.0
        self.bridge_type = 'neural_network'
    
    def setup_bridge_model(self, input_dim: int, output_dim: int, model_type: str = 'neural_network'):
        """设置桥接模型"""
        self.bridge_type = model_type
        
        if model_type == 'neural_network' and HAS_PYTORCH:
            self.bridge_model = PhysicsInformedNeuralNetwork(
                input_dim, [128, 64, 32], output_dim
            )
        elif model_type == 'surrogate':
            self.bridge_model = SurrogateModelAdvanced('gaussian_process')
        else:
            raise ValueError(f"不支持的桥接模型类型: {model_type}")
    
    def train_bridge(self, fine_data: np.ndarray, coarse_data: np.ndarray, **kwargs) -> dict:
        """训练桥接模型"""
        if self.bridge_model is None:
            raise ValueError("桥接模型尚未设置")
        
        if isinstance(self.bridge_model, PhysicsInformedNeuralNetwork):
            self.bridge_model.setup_training()
            result = self.bridge_model.train(fine_data, coarse_data, **kwargs)
        else:
            result = self.bridge_model.train(fine_data, coarse_data, **kwargs)
        
        self.is_trained = True
        return result
    
    def predict_coarse_from_fine(self, fine_data: np.ndarray) -> np.ndarray:
        """从细尺度数据预测粗尺度数据"""
        if not self.is_trained:
            raise ValueError("桥接模型尚未训练")
        
        return self.bridge_model.predict(fine_data)
    
    def predict_fine_from_coarse(self, coarse_data: np.ndarray) -> np.ndarray:
        """从粗尺度数据预测细尺度数据"""
        if not self.is_trained:
            raise ValueError("桥接模型尚未训练")
        
        # 这里需要实现反向映射
        # 简化实现：返回粗尺度数据的插值
        return coarse_data
    
    def set_scale_ratio(self, ratio: float):
        """设置尺度比例"""
        self.scale_ratio = ratio


class HybridMLAccelerator:
    """
    混合ML加速器 - 结合传统求解器和ML
    
    核心思想：无法完全替代传统求解器（需高精度），
    但可加速其中耗时步骤（如迭代求解、网格自适应）
    """
    
    def __init__(self, traditional_solver: Callable = None):
        self.traditional_solver = traditional_solver
        self.ml_models = {}
        self.performance_stats = {
            'traditional_time': 0.0,
            'ml_time': 0.0,
            'speedup': 0.0,
            'accuracy_loss': 0.0,
            'total_calls': 0
        }
        self.acceleration_strategies = {
            'initial_guess': None,
            'preconditioner': None,
            'adaptive_mesh': None,
            'multiscale': None
        }
    
    def add_ml_model(self, name: str, model: Union[PhysicsInformedNeuralNetwork, SurrogateModelAdvanced]):
        """添加ML模型"""
        self.ml_models[name] = model
        print(f"✅ 添加ML模型: {name}")
    
    def setup_acceleration_strategy(self, strategy: str, model_name: str):
        """设置加速策略"""
        if model_name in self.ml_models:
            self.acceleration_strategies[strategy] = model_name
            print(f"✅ 设置加速策略: {strategy} -> {model_name}")
        else:
            raise ValueError(f"ML模型 {model_name} 不存在")
    
    def solve_hybrid(self, problem_data: Dict, use_ml: bool = True, 
                    ml_model_name: str = None) -> Dict:
        """混合求解"""
        start_time = time.time()
        self.performance_stats['total_calls'] += 1
        
        if use_ml and ml_model_name and ml_model_name in self.ml_models:
            # 使用ML模型
            ml_model = self.ml_models[ml_model_name]
            result = ml_model.predict(problem_data['input'])
            ml_time = time.time() - start_time
            
            self.performance_stats['ml_time'] += ml_time
            
            return {
                'solution': result,
                'method': 'ml',
                'time': ml_time,
                'model_name': ml_model_name
            }
        
        else:
            # 使用传统求解器
            if self.traditional_solver is None:
                raise ValueError("传统求解器未设置")
            
            # 检查是否有ML加速策略
            if self.acceleration_strategies['initial_guess']:
                # 使用ML预测初始猜测
                initial_guess = self.ml_models[self.acceleration_strategies['initial_guess']].predict(
                    problem_data['input']
                )
                problem_data['initial_guess'] = initial_guess
            
            result = self.traditional_solver(problem_data)
            traditional_time = time.time() - start_time
            
            self.performance_stats['traditional_time'] += traditional_time
            
            return {
                'solution': result,
                'method': 'traditional',
                'time': traditional_time
            }
    
    def compare_methods(self, problem_data: Dict, ml_model_name: str = None) -> Dict:
        """比较传统方法和ML方法"""
        # 传统方法
        traditional_result = self.solve_hybrid(problem_data, use_ml=False)
        
        # ML方法
        if ml_model_name and ml_model_name in self.ml_models:
            ml_result = self.solve_hybrid(problem_data, use_ml=True, ml_model_name=ml_model_name)
            
            # 计算加速比
            speedup = traditional_result['time'] / ml_result['time']
            
            # 计算精度损失（简化）
            accuracy_loss = np.mean(np.abs(
                traditional_result['solution'] - ml_result['solution']
            ))
            
            self.performance_stats['speedup'] = speedup
            self.performance_stats['accuracy_loss'] = accuracy_loss
            
            return {
                'traditional': traditional_result,
                'ml': ml_result,
                'speedup': speedup,
                'accuracy_loss': accuracy_loss
            }
        
        return {'traditional': traditional_result}
    
    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        if stats['total_calls'] > 0:
            stats['avg_traditional_time'] = stats['traditional_time'] / stats['total_calls']
            stats['avg_ml_time'] = stats['ml_time'] / stats['total_calls']
        return stats


class AdaptiveMLSolver:
    """
    自适应ML求解器 - 根据问题特征自动选择最佳方法
    
    核心思想：根据问题规模、精度要求、计算资源等特征，
    自动选择最适合的求解方法（传统数值方法 vs ML加速方法）
    """
    
    def __init__(self):
        self.solvers = {}
        self.performance_history = []
        self.adaptive_rules = {}
        self.selection_strategy = 'performance_based'
        
        # 可配置评分权重
        self.score_weights = {
            'problem_feature': 1.0,
            'accuracy': 0.5,
            'speed': 0.5,
            'priority': 0.1
        }
    
    def add_solver(self, name: str, solver: Callable, 
                  conditions: Dict = None, priority: int = 1):
        """添加求解器"""
        self.solvers[name] = {
            'solver': solver,
            'conditions': conditions or {},
            'priority': priority,
            'performance': {'accuracy': 0.0, 'speed': 0.0, 'usage_count': 0}
        }
        print(f"✅ 添加求解器: {name}")
    
    def set_selection_strategy(self, strategy: str):
        """设置选择策略"""
        valid_strategies = ['performance_based', 'rule_based', 'hybrid']
        if strategy in valid_strategies:
            self.selection_strategy = strategy
            print(f"✅ 设置选择策略: {strategy}")
        else:
            raise ValueError(f"不支持的选择策略: {strategy}")
    
    def set_score_weights(self, weights: Dict[str, float]):
        """设置评分权重"""
        for key, value in weights.items():
            if key in self.score_weights:
                self.score_weights[key] = value
        print(f"✅ 评分权重已更新: {self.score_weights}")
    
    def select_best_solver(self, problem_data: Dict) -> str:
        """选择最佳求解器"""
        if self.selection_strategy == 'performance_based':
            return self._select_by_performance(problem_data)
        elif self.selection_strategy == 'rule_based':
            return self._select_by_rules(problem_data)
        elif self.selection_strategy == 'hybrid':
            return self._select_hybrid(problem_data)
        
        return None
    
    def _select_by_performance(self, problem_data: Dict) -> str:
        """基于性能选择求解器"""
        best_solver = None
        best_score = -1
        
        for name, solver_info in self.solvers.items():
            score = self._evaluate_solver_performance(name, solver_info, problem_data)
            if score > best_score:
                best_score = score
                best_solver = name
        
        return best_solver
    
    def _select_by_rules(self, problem_data: Dict) -> str:
        """基于规则选择求解器"""
        for name, solver_info in self.solvers.items():
            if self._check_conditions(solver_info['conditions'], problem_data):
                return name
        return None
    
    def _select_hybrid(self, problem_data: Dict) -> str:
        """混合选择策略"""
        # 先检查规则
        rule_based = self._select_by_rules(problem_data)
        if rule_based:
            return rule_based
        
        # 再基于性能选择
        return self._select_by_performance(problem_data)
    
    def _evaluate_solver_performance(self, name: str, solver_info: Dict, problem_data: Dict) -> float:
        """评估求解器性能"""
        score = 0.0
        weights = self.score_weights
        
        # 扩展问题特征评估（支持精度要求）
        if 'size' in problem_data:
            if problem_data['size'] < 1000 and solver_info['conditions'].get('small_problems', False):
                score += weights['problem_feature']
            elif problem_data['size'] >= 1000 and solver_info['conditions'].get('large_problems', False):
                score += weights['problem_feature']
        
        # 支持精度要求匹配（新增）
        if 'accuracy_requirement' in problem_data:
            req = problem_data['accuracy_requirement']
            if solver_info['performance']['accuracy'] >= req:
                score += weights['problem_feature'] * 0.5  # 额外加分
        
        # 支持计算资源限制（新增）
        if 'compute_resource' in problem_data:
            resource = problem_data['compute_resource']
            if solver_info['conditions'].get('resource_requirement', 'low') == resource:
                score += weights['problem_feature'] * 0.3
        
        # 动态权重计算
        performance = solver_info['performance']
        score += performance['accuracy'] * weights['accuracy'] + performance['speed'] * weights['speed']
        score += solver_info['priority'] * weights['priority']
        
        return score
    
    def _check_conditions(self, conditions: Dict, problem_data: Dict) -> bool:
        """检查条件是否满足 - 支持范围条件"""
        for key, cond in conditions.items():
            if key not in problem_data:
                return False
            val = problem_data[key]
            
            # 支持范围条件（如 ('>', 1000)、('<=', 0.95)）
            if isinstance(cond, tuple) and len(cond) == 2:
                op, threshold = cond
                if op == '>':
                    if not (val > threshold):
                        return False
                elif op == '<':
                    if not (val < threshold):
                        return False
                elif op == '>=':
                    if not (val >= threshold):
                        return False
                elif op == '<=':
                    if not (val <= threshold):
                        return False
                else:
                    return False
            # 原逻辑：支持等值或列表包含
            elif isinstance(cond, (list, tuple)):
                if val not in cond:
                    return False
            else:
                if val != cond:
                    return False
        return True
    
    def solve(self, problem_data: Dict) -> Dict:
        """自适应求解"""
        # 选择最佳求解器
        best_solver_name = self.select_best_solver(problem_data)
        
        if best_solver_name is None:
            raise ValueError("没有可用的求解器")
        
        # 执行求解
        solver_info = self.solvers[best_solver_name]
        solver = solver_info['solver']
        
        start_time = time.time()
        result = solver(problem_data)
        solve_time = time.time() - start_time
        
        # 更新性能统计
        solver_info['performance']['usage_count'] += 1
        solver_info['performance']['speed'] = 1.0 / solve_time
        
        # 新增：若提供参考解，计算精度
        if 'reference_solution' in problem_data:
            mse = np.mean((result - problem_data['reference_solution'])**2)
            accuracy = 1.0 / (1.0 + mse)  # 归一化到 [0,1]
            solver_info['performance']['accuracy'] = accuracy
        
        # 记录历史
        self.performance_history.append({
            'solver': best_solver_name,
            'time': solve_time,
            'problem_size': problem_data.get('size', 0),
            'timestamp': time.time()
        })
        
        return {
            'solution': result,
            'solver_used': best_solver_name,
            'time': solve_time
        }
    
    def get_performance_summary(self) -> Dict:
        """获取性能总结"""
        summary = {}
        
        for name, solver_info in self.solvers.items():
            performance = solver_info['performance']
            summary[name] = {
                'usage_count': performance['usage_count'],
                'avg_speed': performance['speed'],
                'avg_accuracy': performance['accuracy']
            }
        
        return summary


class RLAgent(nn.Module):
    """
    强化学习智能体 - 用于优化数值求解策略
    
    核心思想：通过强化学习自动选择最优的求解参数（时间步长、网格加密方案等），
    减少人工调参成本，提升求解效率
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        if not HAS_PYTORCH:
            raise ImportError("需要安装PyTorch来使用RL智能体")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Actor网络（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )
        
        # Critic网络（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
        self.to(self.device)
        
        print(f"🔄 RL智能体初始化完成 - 设备: {self.device}")
        print(f"   状态维度: {state_dim}, 动作维度: {action_dim}")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播 - 返回动作"""
        return self.actor(state)
    
    def get_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """获取动作（带探索噪声）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.forward(state_tensor).squeeze(0)
            # 添加探索噪声
            noise = torch.randn_like(action) * noise_scale
            action = action + noise
            # 裁剪到有效范围
            action = torch.clamp(action, -1.0, 1.0)
        
        return action.cpu().numpy()
    
    def get_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """获取状态-动作价值"""
        return self.critic(torch.cat([state, action], dim=1))


class SolverEnvironment:
    """
    求解器环境 - 模拟数值求解过程，为RL提供训练环境
    
    核心思想：将数值求解过程建模为强化学习环境，
    智能体通过与环境交互学习最优的求解策略
    """
    
    def __init__(self, solver_config: Dict, physics_config: PhysicsConfig = None):
        self.solver_config = solver_config
        self.physics_config = physics_config or PhysicsConfig()
        self.max_steps = solver_config.get('max_steps', 100)
        self.current_step = 0
        self.convergence_history = []
        self.performance_metrics = {}
        
        # 求解策略参数范围
        self.action_bounds = {
            'time_step': (0.001, 0.1),      # 时间步长
            'mesh_refinement': (0.1, 2.0),  # 网格加密因子
            'tolerance': (1e-6, 1e-3),      # 收敛容差
            'max_iterations': (50, 500)      # 最大迭代次数
        }
        
        print(f"🔄 求解器环境初始化完成")
        print(f"   最大步数: {self.max_steps}")
        print(f"   动作参数: {list(self.action_bounds.keys())}")
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.convergence_history = []
        self.performance_metrics = {}
        
        # 返回初始状态
        initial_state = self._get_state()
        return initial_state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作"""
        if self.current_step >= self.max_steps:
            return self._get_state(), 0.0, True, {}
        
        # 解析动作（从[-1,1]映射到实际参数范围）
        solver_params = self._action_to_params(action)
        
        # 模拟求解过程
        reward, metrics = self._simulate_solving(solver_params)
        
        # 更新状态
        self.current_step += 1
        self.convergence_history.append(metrics.get('convergence', 0.0))
        self.performance_metrics.update(metrics)
        
        # 检查是否完成
        done = self.current_step >= self.max_steps or metrics.get('converged', False)
        
        return self._get_state(), reward, done, metrics
    
    def _action_to_params(self, action: np.ndarray) -> Dict:
        """将动作转换为求解器参数"""
        params = {}
        action_names = list(self.action_bounds.keys())
        
        for i, name in enumerate(action_names):
            if i < len(action):
                # 将[-1,1]映射到实际范围
                action_val = action[i]
                min_val, max_val = self.action_bounds[name]
                param_val = min_val + (action_val + 1) * (max_val - min_val) / 2
                params[name] = param_val
        
        return params
    
    def _simulate_solving(self, solver_params: Dict) -> Tuple[float, Dict]:
        """模拟求解过程，计算奖励和指标"""
        # 模拟求解时间（基于参数）
        time_step = solver_params.get('time_step', 0.01)
        mesh_refinement = solver_params.get('mesh_refinement', 1.0)
        tolerance = solver_params.get('tolerance', 1e-4)
        max_iterations = solver_params.get('max_iterations', 100)
        
        # 模拟收敛过程
        convergence_rate = 1.0 / (1.0 + tolerance * 1000)  # 容差越小，收敛越快
        mesh_efficiency = 1.0 / (1.0 + abs(mesh_refinement - 1.0))  # 网格因子接近1时效率最高
        
        # 模拟迭代次数
        actual_iterations = min(max_iterations, int(50 / convergence_rate))
        
        # 计算奖励（综合考虑效率、精度、稳定性）
        efficiency_reward = 1.0 / (1.0 + time_step * 100)  # 时间步长越小越好
        accuracy_reward = 1.0 / (1.0 + tolerance * 1e6)    # 容差越小越好
        stability_reward = 1.0 / (1.0 + abs(mesh_refinement - 1.0))  # 网格稳定性
        
        # 收敛奖励
        converged = actual_iterations < max_iterations
        convergence_reward = 10.0 if converged else 0.0
        
        # 总奖励
        total_reward = (efficiency_reward + accuracy_reward + stability_reward + convergence_reward) / 4
        
        # 性能指标
        metrics = {
            'convergence': convergence_rate,
            'mesh_efficiency': mesh_efficiency,
            'iterations': actual_iterations,
            'converged': converged,
            'time_step': time_step,
            'mesh_refinement': mesh_refinement,
            'tolerance': tolerance
        }
        
        return total_reward, metrics
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        state = []
        
        # 当前步数（归一化）
        state.append(self.current_step / self.max_steps)
        
        # 收敛历史统计
        if self.convergence_history:
            state.extend([
                np.mean(self.convergence_history),
                np.std(self.convergence_history),
                self.convergence_history[-1] if self.convergence_history else 0.0
            ])
        else:
            state.extend([0.0, 0.0, 0.0])
        
        # 性能指标
        for key in ['mesh_efficiency', 'iterations']:
            if key in self.performance_metrics:
                # 归一化到[0,1]
                if key == 'iterations':
                    val = self.performance_metrics[key] / 500.0  # 假设最大500次迭代
                else:
                    val = self.performance_metrics[key]
                state.append(val)
            else:
                state.append(0.0)
        
        return np.array(state, dtype=np.float32)


class RLSolverOptimizer(BaseSolver):
    """
    强化学习求解器优化器 - 使用RL自动优化数值求解策略
    
    核心思想：通过强化学习训练智能体，自动选择最优的求解参数，
    实现"自学习"的数值求解优化
    """
    
    def __init__(self, state_dim: int, action_dim: int, solver_config: Dict = None):
        super().__init__()
        
        if not HAS_PYTORCH:
            raise ImportError("需要安装PyTorch来使用RL求解器优化器")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.solver_config = solver_config or {}
        
        # 创建RL智能体和环境
        self.agent = RLAgent(state_dim, action_dim)
        self.environment = SolverEnvironment(solver_config)
        
        # 训练参数
        self.learning_rate = 0.001
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005   # 软更新参数
        
        # 经验回放缓冲区
        self.replay_buffer = []
        self.buffer_size = 10000
        self.batch_size = 64
        
        # 目标网络（用于稳定训练）
        self.target_agent = RLAgent(state_dim, action_dim)
        self._update_target_network()
        
        self.optimizer_actor = optim.Adam(self.agent.actor.parameters(), lr=self.learning_rate)
        self.optimizer_critic = optim.Adam(self.agent.critic.parameters(), lr=self.learning_rate)
        
        print(f"🔄 RL求解器优化器初始化完成")
        print(f"   状态维度: {state_dim}, 动作维度: {action_dim}")
    
    def _update_target_network(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _store_experience(self, state: np.ndarray, action: np.ndarray, 
                         reward: float, next_state: np.ndarray, done: bool):
        """存储经验到回放缓冲区"""
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.append(experience)
        
        # 限制缓冲区大小
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def _sample_batch(self) -> List[Tuple]:
        """从回放缓冲区采样批次"""
        if len(self.replay_buffer) < self.batch_size:
            return []
        
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        return [self.replay_buffer[i] for i in indices]
    
    def _update_networks(self, batch: List[Tuple]):
        """更新网络参数"""
        if not batch:
            return
        
        states = torch.FloatTensor(np.array([exp[0] for exp in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([exp[1] for exp in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([exp[2] for exp in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp[3] for exp in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([exp[4] for exp in batch])).to(self.device)
        
        # 更新Critic网络
        current_q_values = self.agent.get_value(states, actions)
        next_actions = self.target_agent(next_states)
        next_q_values = self.target_agent.get_value(next_states, next_actions)
        target_q_values = rewards + (self.gamma * next_q_values * (~dones).float())
        
        critic_loss = F.mse_loss(current_q_values, target_q_values.detach())
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        # 更新Actor网络
        actor_loss = -self.agent.get_value(states, self.agent(states)).mean()
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        # 软更新目标网络
        self._update_target_network()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def train(self, episodes: int = 1000, **kwargs) -> dict:
        """训练RL智能体"""
        print(f"🔄 开始训练RL求解器优化器，总轮数: {episodes}")
        
        episode_rewards = []
        episode_lengths = []
        training_losses = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            episode_reward = 0.0
            episode_length = 0
            
            while True:
                # 选择动作
                action = self.agent.get_action(state, noise_scale=max(0.01, 0.1 * (1 - episode / episodes)))
                
                # 执行动作
                next_state, reward, done, info = self.environment.step(action)
                
                # 存储经验
                self._store_experience(state, action, reward, next_state, done)
                
                # 更新网络
                batch = self._sample_batch()
                if batch:
                    loss_info = self._update_networks(batch)
                    training_losses.append(loss_info)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print(f"   轮数 {episode+1}/{episodes}: 平均奖励={avg_reward:.4f}, 平均长度={avg_length:.1f}")
        
        self.is_trained = True
        
        training_history = {
            'episodes': episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'training_losses': training_losses,
            'final_avg_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
        }
        
        print(f"✅ RL训练完成，最终平均奖励: {training_history['final_avg_reward']:.4f}")
        return training_history
    
    def optimize_solver_strategy(self, problem_state: np.ndarray) -> Dict:
        """优化求解策略"""
        if not self.is_trained:
            raise ValueError("RL智能体尚未训练")
        
        # 使用训练好的智能体选择最优动作
        optimal_action = self.agent.get_action(problem_state, noise_scale=0.0)
        
        # 转换为求解器参数
        solver_params = self.environment._action_to_params(optimal_action)
        
        print(f"🔧 RL优化求解策略:")
        for param, value in solver_params.items():
            print(f"   {param}: {value:.6f}")
        
        return solver_params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测最优求解策略"""
        if not self.is_trained:
            raise ValueError("RL智能体尚未训练")
        
        strategies = []
        for state in X:
            action = self.agent.get_action(state, noise_scale=0.0)
            strategy = self.environment._action_to_params(action)
            strategies.append(list(strategy.values()))
        
        return np.array(strategies)
    
    def evaluate_strategy(self, strategy: Dict, problem_state: np.ndarray) -> Dict:
        """评估求解策略的性能"""
        # 将策略转换为动作
        action = np.array([strategy.get(param, 0.0) for param in self.environment.action_bounds.keys()])
        
        # 在环境中测试策略
        state = problem_state
        total_reward = 0.0
        step_count = 0
        
        for _ in range(self.environment.max_steps):
            next_state, reward, done, info = self.environment.step(action)
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        return {
            'total_reward': total_reward,
            'step_count': step_count,
            'efficiency': total_reward / max(step_count, 1),
            'convergence': info.get('converged', False)
        }


def create_advanced_ml_system() -> Dict:
    """创建高级ML系统"""
    system = {
        'pinn': PhysicsInformedNeuralNetwork,
        'surrogate': SurrogateModelAdvanced,
        'bridge': MultiScaleMLBridge,
        'hybrid': HybridMLAccelerator,
        'adaptive': AdaptiveMLSolver,
        'rl_agent': RLAgent,
        'rl_environment': SolverEnvironment,
        'rl_optimizer': RLSolverOptimizer
    }
    
    print("🔄 高级ML系统创建完成")
    return system


class RLAgent(nn.Module):
    """
    强化学习智能体 - 用于优化数值求解策略
    
    核心思想：通过强化学习自动选择最优的求解参数（时间步长、网格加密方案等），
    减少人工调参成本，提升求解效率
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        if not HAS_PYTORCH:
            raise ImportError("需要安装PyTorch来使用RL智能体")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Actor网络（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )
        
        # Critic网络（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
        self.to(self.device)
        
        print(f"🔄 RL智能体初始化完成 - 设备: {self.device}")
        print(f"   状态维度: {state_dim}, 动作维度: {action_dim}")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播 - 返回动作"""
        return self.actor(state)
    
    def get_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """获取动作（带探索噪声）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.forward(state_tensor).squeeze(0)
            # 添加探索噪声
            noise = torch.randn_like(action) * noise_scale
            action = action + noise
            # 裁剪到有效范围
            action = torch.clamp(action, -1.0, 1.0)
        
        return action.cpu().numpy()
    
    def get_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """获取状态-动作价值"""
        return self.critic(torch.cat([state, action], dim=1))


class SolverEnvironment:
    """
    求解器环境 - 模拟数值求解过程，为RL提供训练环境
    
    核心思想：将数值求解过程建模为强化学习环境，
    智能体通过与环境交互学习最优的求解策略
    """
    
    def __init__(self, solver_config: Dict, physics_config: PhysicsConfig = None):
        self.solver_config = solver_config
        self.physics_config = physics_config or PhysicsConfig()
        self.max_steps = solver_config.get('max_steps', 100)
        self.current_step = 0
        self.convergence_history = []
        self.performance_metrics = {}
        
        # 求解策略参数范围
        self.action_bounds = {
            'time_step': (0.001, 0.1),      # 时间步长
            'mesh_refinement': (0.1, 2.0),  # 网格加密因子
            'tolerance': (1e-6, 1e-3),      # 收敛容差
            'max_iterations': (50, 500)      # 最大迭代次数
        }
        
        print(f"🔄 求解器环境初始化完成")
        print(f"   最大步数: {self.max_steps}")
        print(f"   动作参数: {list(self.action_bounds.keys())}")
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.convergence_history = []
        self.performance_metrics = {}
        
        # 返回初始状态
        initial_state = self._get_state()
        return initial_state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作"""
        if self.current_step >= self.max_steps:
            return self._get_state(), 0.0, True, {}
        
        # 解析动作（从[-1,1]映射到实际参数范围）
        solver_params = self._action_to_params(action)
        
        # 模拟求解过程
        reward, metrics = self._simulate_solving(solver_params)
        
        # 更新状态
        self.current_step += 1
        self.convergence_history.append(metrics.get('convergence', 0.0))
        self.performance_metrics.update(metrics)
        
        # 检查是否完成
        done = self.current_step >= self.max_steps or metrics.get('converged', False)
        
        return self._get_state(), reward, done, metrics
    
    def _action_to_params(self, action: np.ndarray) -> Dict:
        """将动作转换为求解器参数"""
        params = {}
        action_names = list(self.action_bounds.keys())
        
        for i, name in enumerate(action_names):
            if i < len(action):
                # 将[-1,1]映射到实际范围
                action_val = action[i]
                min_val, max_val = self.action_bounds[name]
                param_val = min_val + (action_val + 1) * (max_val - min_val) / 2
                params[name] = param_val
        
        return params
    
    def _simulate_solving(self, solver_params: Dict) -> Tuple[float, Dict]:
        """模拟求解过程，计算奖励和指标"""
        # 模拟求解时间（基于参数）
        time_step = solver_params.get('time_step', 0.01)
        mesh_refinement = solver_params.get('mesh_refinement', 1.0)
        tolerance = solver_params.get('tolerance', 1e-4)
        max_iterations = solver_params.get('max_iterations', 100)
        
        # 模拟收敛过程
        convergence_rate = 1.0 / (1.0 + tolerance * 1000)  # 容差越小，收敛越快
        mesh_efficiency = 1.0 / (1.0 + abs(mesh_refinement - 1.0))  # 网格因子接近1时效率最高
        
        # 模拟迭代次数
        actual_iterations = min(max_iterations, int(50 / convergence_rate))
        
        # 计算奖励（综合考虑效率、精度、稳定性）
        efficiency_reward = 1.0 / (1.0 + time_step * 100)  # 时间步长越小越好
        accuracy_reward = 1.0 / (1.0 + tolerance * 1e6)    # 容差越小越好
        stability_reward = 1.0 / (1.0 + abs(mesh_refinement - 1.0))  # 网格稳定性
        
        # 收敛奖励
        converged = actual_iterations < max_iterations
        convergence_reward = 10.0 if converged else 0.0
        
        # 总奖励
        total_reward = (efficiency_reward + accuracy_reward + stability_reward + convergence_reward) / 4
        
        # 性能指标
        metrics = {
            'convergence': convergence_rate,
            'mesh_efficiency': mesh_efficiency,
            'iterations': actual_iterations,
            'converged': converged,
            'time_step': time_step,
            'mesh_refinement': mesh_refinement,
            'tolerance': tolerance
        }
        
        return total_reward, metrics
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        state = []
        
        # 当前步数（归一化）
        state.append(self.current_step / self.max_steps)
        
        # 收敛历史统计
        if self.convergence_history:
            state.extend([
                np.mean(self.convergence_history),
                np.std(self.convergence_history),
                self.convergence_history[-1] if self.convergence_history else 0.0
            ])
        else:
            state.extend([0.0, 0.0, 0.0])
        
        # 性能指标
        for key in ['mesh_efficiency', 'iterations']:
            if key in self.performance_metrics:
                # 归一化到[0,1]
                if key == 'iterations':
                    val = self.performance_metrics[key] / 500.0  # 假设最大500次迭代
                else:
                    val = self.performance_metrics[key]
                state.append(val)
            else:
                state.append(0.0)
        
        return np.array(state, dtype=np.float32)


class RLSolverOptimizer(BaseSolver):
    """
    强化学习求解器优化器 - 使用RL自动优化数值求解策略
    
    核心思想：通过强化学习训练智能体，自动选择最优的求解参数，
    实现"自学习"的数值求解优化
    """
    
    def __init__(self, state_dim: int, action_dim: int, solver_config: Dict = None):
        super().__init__()
        
        if not HAS_PYTORCH:
            raise ImportError("需要安装PyTorch来使用RL求解器优化器")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.solver_config = solver_config or {}
        
        # 创建RL智能体和环境
        self.agent = RLAgent(state_dim, action_dim)
        self.environment = SolverEnvironment(solver_config)
        
        # 训练参数
        self.learning_rate = 0.001
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005   # 软更新参数
        
        # 经验回放缓冲区
        self.replay_buffer = []
        self.buffer_size = 10000
        self.batch_size = 64
        
        # 目标网络（用于稳定训练）
        self.target_agent = RLAgent(state_dim, action_dim)
        self._update_target_network()
        
        self.optimizer_actor = optim.Adam(self.agent.actor.parameters(), lr=self.learning_rate)
        self.optimizer_critic = optim.Adam(self.agent.critic.parameters(), lr=self.learning_rate)
        
        print(f"🔄 RL求解器优化器初始化完成")
        print(f"   状态维度: {state_dim}, 动作维度: {action_dim}")
    
    def _update_target_network(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _store_experience(self, state: np.ndarray, action: np.ndarray, 
                         reward: float, next_state: np.ndarray, done: bool):
        """存储经验到回放缓冲区"""
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.append(experience)
        
        # 限制缓冲区大小
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def _sample_batch(self) -> List[Tuple]:
        """从回放缓冲区采样批次"""
        if len(self.replay_buffer) < self.batch_size:
            return []
        
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        return [self.replay_buffer[i] for i in indices]
    
    def _update_networks(self, batch: List[Tuple]):
        """更新网络参数"""
        if not batch:
            return
        
        states = torch.FloatTensor(np.array([exp[0] for exp in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([exp[1] for exp in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([exp[2] for exp in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp[3] for exp in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([exp[4] for exp in batch])).to(self.device)
        
        # 更新Critic网络
        current_q_values = self.agent.get_value(states, actions)
        next_actions = self.target_agent(next_states)
        next_q_values = self.target_agent.get_value(next_states, next_actions)
        target_q_values = rewards + (self.gamma * next_q_values * (~dones).float())
        
        critic_loss = F.mse_loss(current_q_values, target_q_values.detach())
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        # 更新Actor网络
        actor_loss = -self.agent.get_value(states, self.agent(states)).mean()
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        # 软更新目标网络
        self._update_target_network()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def train(self, episodes: int = 1000, **kwargs) -> dict:
        """训练RL智能体"""
        print(f"🔄 开始训练RL求解器优化器，总轮数: {episodes}")
        
        episode_rewards = []
        episode_lengths = []
        training_losses = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            episode_reward = 0.0
            episode_length = 0
            
            while True:
                # 选择动作
                action = self.agent.get_action(state, noise_scale=max(0.01, 0.1 * (1 - episode / episodes)))
                
                # 执行动作
                next_state, reward, done, info = self.environment.step(action)
                
                # 存储经验
                self._store_experience(state, action, reward, next_state, done)
                
                # 更新网络
                batch = self._sample_batch()
                if batch:
                    loss_info = self._update_networks(batch)
                    training_losses.append(loss_info)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print(f"   轮数 {episode+1}/{episodes}: 平均奖励={avg_reward:.4f}, 平均长度={avg_length:.1f}")
        
        self.is_trained = True
        
        training_history = {
            'episodes': episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'training_losses': training_losses,
            'final_avg_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
        }
        
        print(f"✅ RL训练完成，最终平均奖励: {training_history['final_avg_reward']:.4f}")
        return training_history
    
    def optimize_solver_strategy(self, problem_state: np.ndarray) -> Dict:
        """优化求解策略"""
        if not self.is_trained:
            raise ValueError("RL智能体尚未训练")
        
        # 使用训练好的智能体选择最优动作
        optimal_action = self.agent.get_action(problem_state, noise_scale=0.0)
        
        # 转换为求解器参数
        solver_params = self.environment._action_to_params(optimal_action)
        
        print(f"🔧 RL优化求解策略:")
        for param, value in solver_params.items():
            print(f"   {param}: {value:.6f}")
        
        return solver_params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测最优求解策略"""
        if not self.is_trained:
            raise ValueError("RL智能体尚未训练")
        
        strategies = []
        for state in X:
            action = self.agent.get_action(state, noise_scale=0.0)
            strategy = self.environment._action_to_params(action)
            strategies.append(list(strategy.values()))
        
        return np.array(strategies)
    
    def evaluate_strategy(self, strategy: Dict, problem_state: np.ndarray) -> Dict:
        """评估求解策略的性能"""
        # 将策略转换为动作
        action = np.array([strategy.get(param, 0.0) for param in self.environment.action_bounds.keys()])
        
        # 在环境中测试策略
        state = problem_state
        total_reward = 0.0
        step_count = 0
        
        for _ in range(self.environment.max_steps):
            next_state, reward, done, info = self.environment.step(action)
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        return {
            'total_reward': total_reward,
            'step_count': step_count,
            'efficiency': total_reward / max(step_count, 1),
            'convergence': info.get('converged', False)
        }


def create_rl_solver_system() -> Dict:
    """创建RL求解器系统"""
    system = {
        'agent': RLAgent,
        'environment': SolverEnvironment,
        'optimizer': RLSolverOptimizer
    }
    
    print("🔄 RL求解器系统创建完成")
    return system


def demo_rl_solver_optimization():
    """演示RL求解器优化"""
    print("🤖 强化学习求解器优化演示")
    print("=" * 60)
    
    try:
        # 创建RL求解器系统
        rl_system = create_rl_solver_system()
        
        # 配置求解器环境
        solver_config = {
            'max_steps': 50,
            'convergence_threshold': 1e-6
        }
        
        # 创建环境
        env = rl_system['environment'](solver_config)
        state_dim = len(env.reset())
        action_dim = len(env.action_bounds)
        
        print(f"📊 环境配置:")
        print(f"   状态维度: {state_dim}")
        print(f"   动作维度: {action_dim}")
        print(f"   最大步数: {env.max_steps}")
        
        # 创建RL优化器
        rl_optimizer = rl_system['optimizer'](state_dim, action_dim, solver_config)
        
        # 训练RL智能体
        print("\n🔧 训练RL智能体...")
        training_history = rl_optimizer.train(episodes=500)
        
        print(f"   训练完成，最终平均奖励: {training_history['final_avg_reward']:.4f}")
        
        # 测试优化后的策略
        print("\n🔧 测试优化后的求解策略...")
        
        # 模拟问题状态
        test_state = np.array([0.0, 0.5, 0.1, 0.8, 0.3])
        
        # 获取最优策略
        optimal_strategy = rl_optimizer.optimize_solver_strategy(test_state)
        
        # 评估策略性能
        performance = rl_optimizer.evaluate_strategy(optimal_strategy, test_state)
        
        print(f"   策略性能评估:")
        print(f"     总奖励: {performance['total_reward']:.4f}")
        print(f"     步数: {performance['step_count']}")
        print(f"     效率: {performance['efficiency']:.4f}")
        print(f"     收敛: {performance['convergence']}")
        
        print("\n✅ RL求解器优化演示完成!")
        return True
        
    except Exception as e:
        print(f"❌ RL求解器优化演示失败: {e}")
        return False


def demo_advanced_ml():
    """演示高级ML功能"""
    print("🤖 高级机器学习加速数值模拟演示")
    print("=" * 60)
    
    # 固定随机种子，确保结果可复现
    np.random.seed(42)
    if HAS_PYTORCH:
        torch.manual_seed(42)
    
    # 创建高级ML系统
    ml_system = create_advanced_ml_system()
    
    # 生成测试数据
    n_samples = 1000
    input_dim = 5
    output_dim = 3
    
    X = np.random.randn(n_samples, input_dim)
    y = np.random.randn(n_samples, output_dim)
    
    print(f"📊 测试数据: {n_samples} 样本, 输入维度: {input_dim}, 输出维度: {output_dim}")
    
    # 1. 测试高级PINN
    print("\n🔧 测试高级PINN...")
    try:
        # 定义物理方程（示例：热传导方程）
        def heat_equation(x, y):
            return torch.mean(torch.abs(y - 0.1 * torch.sum(x, dim=1, keepdim=True)))
        
        # 定义边界条件
        def boundary_condition(x, y):
            return torch.mean(torch.abs(y[:, 0] - 0.0))  # 第一个输出在边界上为0
        
        pinn = ml_system['pinn'](input_dim, [64, 32], output_dim, 
                                physics_equations=[heat_equation],
                                boundary_conditions=[boundary_condition])
        
        pinn.setup_training()
        result = pinn.train(X, y, epochs=200)
        print(f"   训练时间: {result['train_time']:.4f} 秒")
        print(f"   最终总损失: {result['total_loss'][-1]:.6f}")
        
    except Exception as e:
        print(f"   ❌ 高级PINN失败: {e}")
    
    # 2. 测试高级代理模型
    print("\n🔧 测试高级代理模型...")
    try:
        surrogate = ml_system['surrogate']('gaussian_process', kernel=RBF(length_scale=1.0) + Matern(length_scale=1.0))
        result = surrogate.train(X, y[:, 0])  # 只预测第一个输出
        print(f"   训练时间: {result['training_time']:.4f} 秒")
        
        # 预测
        predictions, std = surrogate.predict(X[:10], return_std=True)
        print(f"   预测形状: {predictions.shape}, 标准差形状: {std.shape}")
        
    except Exception as e:
        print(f"   ❌ 高级代理模型失败: {e}")
    
    # 3. 测试多尺度桥接
    print("\n🔧 测试多尺度桥接...")
    try:
        bridge = ml_system['bridge']()
        bridge.setup_bridge_model(input_dim, output_dim, 'neural_network')
        
        # 模拟细尺度和粗尺度数据
        fine_data = X
        coarse_data = y
        
        result = bridge.train_bridge(fine_data, coarse_data, epochs=100)
        print(f"   桥接模型训练完成")
        
        # 测试桥接
        coarse_pred = bridge.predict_coarse_from_fine(X[:10])
        print(f"   粗尺度预测形状: {coarse_pred.shape}")
        
    except Exception as e:
        print(f"   ❌ 多尺度桥接失败: {e}")
    
    # 4. 测试混合加速器
    print("\n🔧 测试混合加速器...")
    try:
        hybrid_accelerator = ml_system['hybrid']()
        
        # 添加ML模型
        surrogate_model = ml_system['surrogate']('random_forest')
        surrogate_model.train(X, y[:, 0])
        hybrid_accelerator.add_ml_model('surrogate', surrogate_model)
        
        # 设置加速策略
        hybrid_accelerator.setup_acceleration_strategy('initial_guess', 'surrogate')
        
        # 测试混合求解
        problem_data = {'input': X[:10]}
        result = hybrid_accelerator.solve_hybrid(problem_data, use_ml=True, ml_model_name='surrogate')
        print(f"   混合求解完成，使用模型: {result['model_name']}")
        print(f"   求解时间: {result['time']:.4f} 秒")
        
    except Exception as e:
        print(f"   ❌ 混合加速器失败: {e}")
    
    # 5. 测试自适应求解器
    print("\n🔧 测试自适应求解器...")
    try:
        adaptive_solver = ml_system['adaptive']()
        
        # 添加不同的求解器
        def fast_solver(data):
            return np.random.randn(data.get('size', 100))
        
        def accurate_solver(data):
            time.sleep(0.01)  # 模拟计算时间
            return np.random.randn(data.get('size', 100))
        
        # 使用新的条件格式
        adaptive_solver.add_solver('fast', fast_solver, 
                                 conditions={'size': ('<', 1000)}, priority=1)
        adaptive_solver.add_solver('accurate', accurate_solver, 
                                 conditions={'size': ('>=', 1000)}, priority=2)
        
        # 设置选择策略和评分权重
        adaptive_solver.set_selection_strategy('hybrid')
        adaptive_solver.set_score_weights({
            'problem_feature': 1.0,
            'accuracy': 0.6,
            'speed': 0.4,
            'priority': 0.1
        })
        
        # 测试求解
        problem_data = {'size': 500, 'accuracy_requirement': 0.8}
        result = adaptive_solver.solve(problem_data)
        print(f"   使用的求解器: {result['solver_used']}")
        print(f"   求解时间: {result['time']:.4f} 秒")
        
    except Exception as e:
        print(f"   ❌ 自适应求解器失败: {e}")
    
    print("\n✅ 高级机器学习加速数值模拟演示完成!")


def create_rl_solver_system() -> Dict:
    """创建RL求解器系统"""
    system = {
        'agent': RLAgent,
        'environment': SolverEnvironment,
        'optimizer': RLSolverOptimizer
    }
    
    print("🔄 RL求解器系统创建完成")
    return system


def demo_rl_solver_optimization():
    """演示RL求解器优化"""
    print("🤖 强化学习求解器优化演示")
    print("=" * 60)
    
    try:
        # 创建RL求解器系统
        rl_system = create_rl_solver_system()
        
        # 配置求解器环境
        solver_config = {
            'max_steps': 50,
            'convergence_threshold': 1e-6
        }
        
        # 创建环境
        env = rl_system['environment'](solver_config)
        state_dim = len(env.reset())
        action_dim = len(env.action_bounds)
        
        print(f"📊 环境配置:")
        print(f"   状态维度: {state_dim}")
        print(f"   动作维度: {action_dim}")
        print(f"   最大步数: {env.max_steps}")
        
        # 创建RL优化器
        rl_optimizer = rl_system['optimizer'](state_dim, action_dim, solver_config)
        
        # 训练RL智能体
        print("\n🔧 训练RL智能体...")
        training_history = rl_optimizer.train(episodes=500)
        
        print(f"   训练完成，最终平均奖励: {training_history['final_avg_reward']:.4f}")
        
        # 测试优化后的策略
        print("\n🔧 测试优化后的求解策略...")
        
        # 模拟问题状态
        test_state = np.array([0.0, 0.5, 0.1, 0.8, 0.3])
        
        # 获取最优策略
        optimal_strategy = rl_optimizer.optimize_solver_strategy(test_state)
        
        # 评估策略性能
        performance = rl_optimizer.evaluate_strategy(optimal_strategy, test_state)
        
        print(f"   策略性能评估:")
        print(f"     总奖励: {performance['total_reward']:.4f}")
        print(f"     步数: {performance['step_count']}")
        print(f"     效率: {performance['efficiency']:.4f}")
        print(f"     收敛: {performance['convergence']}")
        
        print("\n✅ RL求解器优化演示完成!")
        return True
        
    except Exception as e:
        print(f"❌ RL求解器优化演示失败: {e}")
        return False


if __name__ == "__main__":
    demo_advanced_ml()
    demo_rl_solver_optimization()
