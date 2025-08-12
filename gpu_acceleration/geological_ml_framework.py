"""
地质数值模拟ML/DL融合框架
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
    warnings.warn("PyTorch not available. Geological ML features will be limited.")

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
    warnings.warn("scikit-learn not available. Geological ML features will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Visualization features will be limited.")


@dataclass
class GeologicalConfig:
    """地质配置类 - 增强版"""
    # 基础地质参数
    porosity: float = 0.2
    permeability: float = 1e-12  # m²
    viscosity: float = 1e-3      # Pa·s
    density: float = 1000.0      # kg/m³
    compressibility: float = 1e-9 # Pa⁻¹
    thermal_conductivity: float = 2.0  # W/(m·K)
    specific_heat: float = 1000.0      # J/(kg·K)
    
    # 新增：地质力学参数
    youngs_modulus: float = 1e9      # Pa，杨氏模量
    poissons_ratio: float = 0.25     # 泊松比
    cohesion: float = 1e6            # Pa，内聚力
    friction_angle: float = 30.0     # 度，内摩擦角
    
    # 新增：地幔对流参数
    reference_viscosity: float = 1e21    # Pa·s，参考黏度
    thermal_expansion: float = 3e-5      # K⁻¹，热膨胀系数
    gravity: float = 9.81                # m/s²，重力加速度
    rayleigh_number: float = 1e7         # 瑞利数
    
    # 新增：GPU加速支持
    use_gpu: bool = True
    
    # 边界条件
    boundary_conditions: Dict = None
    
    def __post_init__(self):
        if self.boundary_conditions is None:
            self.boundary_conditions = {
                'pressure': {'top': 'free', 'bottom': 'fixed'},
                'temperature': {'top': 25, 'bottom': 100},
                'flow': {'inlet': 'fixed_rate', 'outlet': 'fixed_pressure'}
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


class GeologicalPhysicsEquations:
    """地质物理方程集合 - 增强版"""
    
    @staticmethod
    def darcy_equation(x: torch.Tensor, y: torch.Tensor, config: GeologicalConfig) -> torch.Tensor:
        """
        达西定律：∇·(k/μ ∇p) = q
        
        Args:
            x: 输入坐标 (x, y, z, t)
            y: 模型输出 (压力场 p)
            config: 地质配置参数
        
        Returns:
            达西方程残差
        """
        if not HAS_PYTORCH:
            return torch.tensor(0.0)
        
        # 改进版本：更精确的达西定律实现
        p = y.unsqueeze(-1) if y.dim() == 1 else y
        
        # 计算压力梯度的二阶导数（拉普拉斯算子）
        p_grad_x = torch.autograd.grad(
            p.sum(), x[:, 0], 
            grad_outputs=torch.ones_like(p), 
            create_graph=True, retain_graph=True
        )[0]
        
        p_grad_y = torch.autograd.grad(
            p.sum(), x[:, 1], 
            grad_outputs=torch.ones_like(p), 
            create_graph=True, retain_graph=True
        )[0]
        
        if x.shape[1] > 2:  # 3D情况
            p_grad_z = torch.autograd.grad(
                p.sum(), x[:, 2], 
                grad_outputs=torch.ones_like(p), 
                create_graph=True, retain_graph=True
            )[0]
            laplacian_p = torch.autograd.grad(p_grad_x.sum(), x[:, 0], create_graph=True)[0] + \
                          torch.autograd.grad(p_grad_y.sum(), x[:, 1], create_graph=True)[0] + \
                          torch.autograd.grad(p_grad_z.sum(), x[:, 2], create_graph=True)[0]
        else:  # 2D情况
            laplacian_p = torch.autograd.grad(p_grad_x.sum(), x[:, 0], create_graph=True)[0] + \
                          torch.autograd.grad(p_grad_y.sum(), x[:, 1], create_graph=True)[0]
        
        # 达西定律残差：∇·(k/μ ∇p) = q
        k_over_mu = config.permeability / config.viscosity
        source_term = 0.01  # 源项
        residual = k_over_mu * laplacian_p - source_term
        
        return torch.mean(torch.abs(residual))
    
    @staticmethod
    def heat_conduction_equation(x: torch.Tensor, y: torch.Tensor, config: GeologicalConfig) -> torch.Tensor:
        """
        热传导方程：ρc∂T/∂t = ∇·(k∇T) + Q
        
        Args:
            x: 输入坐标 (x, y, z, t)
            y: 模型输出 (温度场 T)
            config: 地质配置参数
        
        Returns:
            热传导方程残差
        """
        if not HAS_PYTORCH:
            return torch.tensor(0.0)
        
        # 改进版本：更精确的热传导方程实现
        T = y.unsqueeze(-1) if y.dim() == 1 else y
        
        # 计算温度梯度的二阶导数
        T_grad_x = torch.autograd.grad(
            T.sum(), x[:, 0], 
            grad_outputs=torch.ones_like(T), 
            create_graph=True, retain_graph=True
        )[0]
        
        T_grad_y = torch.autograd.grad(
            T.sum(), x[:, 1], 
            grad_outputs=torch.ones_like(T), 
            create_graph=True, retain_graph=True
        )[0]
        
        if x.shape[1] > 2:  # 3D情况
            T_grad_z = torch.autograd.grad(
                T.sum(), x[:, 2], 
                grad_outputs=torch.ones_like(T), 
                create_graph=True, retain_graph=True
            )[0]
            laplacian_T = torch.autograd.grad(T_grad_x.sum(), x[:, 0], create_graph=True)[0] + \
                          torch.autograd.grad(T_grad_y.sum(), x[:, 1], create_graph=True)[0] + \
                          torch.autograd.grad(T_grad_z.sum(), x[:, 2], create_graph=True)[0]
        else:  # 2D情况
            laplacian_T = torch.autograd.grad(T_grad_x.sum(), x[:, 0], create_graph=True)[0] + \
                          torch.autograd.grad(T_grad_y.sum(), x[:, 1], create_graph=True)[0]
        
        # 热传导方程残差：ρc∂T/∂t = ∇·(k∇T) + Q
        # 简化：忽略时间导数，稳态热传导
        heat_source = 0.01  # 热源
        residual = config.thermal_conductivity * laplacian_T + heat_source
        
        return torch.mean(torch.abs(residual))
    
    @staticmethod
    def elastic_equilibrium_equation(x: torch.Tensor, y: torch.Tensor, config: GeologicalConfig) -> torch.Tensor:
        """
        弹性力学平衡方程：∇·σ + f = 0
        
        Args:
            x: 输入坐标 (x, y, z)
            y: 模型输出 (位移场 u)
            config: 地质配置参数
        
        Returns:
            弹性平衡方程残差
        """
        if not HAS_PYTORCH:
            return torch.tensor(0.0)
        
        # 改进版本：更精确的弹性力学方程实现
        u = y.unsqueeze(-1) if y.dim() == 1 else y
        
        # 计算位移梯度的二阶导数
        u_grad_x = torch.autograd.grad(
            u.sum(), x[:, 0], 
            grad_outputs=torch.ones_like(u), 
            create_graph=True, retain_graph=True
        )[0]
        
        u_grad_y = torch.autograd.grad(
            u.sum(), x[:, 1], 
            grad_outputs=torch.ones_like(u), 
            create_graph=True, retain_graph=True
        )[0]
        
        if x.shape[1] > 2:  # 3D情况
            u_grad_z = torch.autograd.grad(
                u.sum(), x[:, 2], 
                grad_outputs=torch.ones_like(u), 
                create_graph=True, retain_graph=True
            )[0]
            laplacian_u = torch.autograd.grad(u_grad_x.sum(), x[:, 0], create_graph=True)[0] + \
                          torch.autograd.grad(u_grad_y.sum(), x[:, 1], create_graph=True)[0] + \
                          torch.autograd.grad(u_grad_z.sum(), x[:, 2], create_graph=True)[0]
        else:  # 2D情况
            laplacian_u = torch.autograd.grad(u_grad_x.sum(), x[:, 0], create_graph=True)[0] + \
                          torch.autograd.grad(u_grad_y.sum(), x[:, 1], create_graph=True)[0]
        
        # 弹性平衡方程残差：∇·σ + f = 0
        # 使用配置中的杨氏模量和泊松比
        body_force = config.gravity * config.density  # 重力
        residual = config.youngs_modulus / (2 * (1 + config.poissons_ratio)) * laplacian_u + body_force
        
        return torch.mean(torch.abs(residual))
    
    @staticmethod
    def stokes_equation(x: torch.Tensor, y: torch.Tensor, config: GeologicalConfig) -> torch.Tensor:
        """
        地幔对流的Stokes动量方程残差：
        ∇·(η(∇v + ∇v^T)) - ∇p + ρgα(T-T0) = 0
        
        Args:
            x: 输入坐标 (x, y, z)
            y: 模型输出 [v_x, v_y, v_z, p, T] 5个物理量
            config: 地质配置参数
        
        Returns:
            Stokes方程残差
        """
        if not HAS_PYTORCH:
            return torch.tensor(0.0)
        
        # 解析输出：速度、压力、温度
        vx, vy, vz, p, T = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]
        
        # 计算速度梯度（空间导数）
        vx_x = torch.autograd.grad(vx, x[:, 0], grad_outputs=torch.ones_like(vx), create_graph=True)[0]
        vx_y = torch.autograd.grad(vx, x[:, 1], grad_outputs=torch.ones_like(vx), create_graph=True)[0]
        vx_z = torch.autograd.grad(vx, x[:, 2], grad_outputs=torch.ones_like(vx), create_graph=True)[0]
        
        vy_x = torch.autograd.grad(vy, x[:, 0], grad_outputs=torch.ones_like(vy), create_graph=True)[0]
        vy_y = torch.autograd.grad(vy, x[:, 1], grad_outputs=torch.ones_like(vy), create_graph=True)[0]
        vy_z = torch.autograd.grad(vy, x[:, 2], grad_outputs=torch.ones_like(vy), create_graph=True)[0]
        
        vz_x = torch.autograd.grad(vz, x[:, 0], grad_outputs=torch.ones_like(vz), create_graph=True)[0]
        vz_y = torch.autograd.grad(vz, x[:, 1], grad_outputs=torch.ones_like(vz), create_graph=True)[0]
        vz_z = torch.autograd.grad(vz, x[:, 2], grad_outputs=torch.ones_like(vz), create_graph=True)[0]
        
        # 计算压力梯度
        p_x = torch.autograd.grad(p, x[:, 0], grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, x[:, 1], grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_z = torch.autograd.grad(p, x[:, 2], grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        # 黏度与温度的非线性关系（地质模拟中关键特性）
        # 使用Arrhenius关系：η = η₀ * exp(E/R * (1/T - 1/T₀))
        T_ref = 273.15  # 参考温度 (K)
        activation_energy = 200e3  # 激活能 (J/mol)
        gas_constant = 8.314  # 气体常数 (J/(mol·K))
        
        # 避免除零，使用安全的温度计算
        T_safe = torch.clamp(T + T_ref, min=1e-6)
        eta = config.reference_viscosity * torch.exp(
            activation_energy / gas_constant * (1.0 / T_safe - 1.0 / T_ref)
        )
        
        # 计算应变率张量 D = (∇v + ∇v^T) / 2
        Dxx = vx_x
        Dyy = vy_y
        Dzz = vz_z
        Dxy = (vx_y + vy_x) / 2
        Dxz = (vx_z + vz_x) / 2
        Dyz = (vy_z + vz_y) / 2
        
        # 计算应力张量 σ = 2ηD
        sigma_xx = 2 * eta * Dxx
        sigma_yy = 2 * eta * Dyy
        sigma_zz = 2 * eta * Dzz
        sigma_xy = 2 * eta * Dxy
        sigma_xz = 2 * eta * Dxz
        sigma_yz = 2 * eta * Dyz
        
        # 计算应力散度 ∇·σ
        sigma_xx_x = torch.autograd.grad(sigma_xx.sum(), x[:, 0], create_graph=True)[0]
        sigma_xy_y = torch.autograd.grad(sigma_xy.sum(), x[:, 1], create_graph=True)[0]
        sigma_xz_z = torch.autograd.grad(sigma_xz.sum(), x[:, 2], create_graph=True)[0]
        
        sigma_yx_x = torch.autograd.grad(sigma_xy.sum(), x[:, 0], create_graph=True)[0]
        sigma_yy_y = torch.autograd.grad(sigma_yy.sum(), x[:, 1], create_graph=True)[0]
        sigma_yz_z = torch.autograd.grad(sigma_yz.sum(), x[:, 2], create_graph=True)[0]
        
        sigma_zx_x = torch.autograd.grad(sigma_xz.sum(), x[:, 0], create_graph=True)[0]
        sigma_zy_y = torch.autograd.grad(sigma_yz.sum(), x[:, 1], create_graph=True)[0]
        sigma_zz_z = torch.autograd.grad(sigma_zz.sum(), x[:, 2], create_graph=True)[0]
        
        # 动量方程残差：∇·σ - ∇p + ρgα(T-T0) = 0
        residual_x = (sigma_xx_x + sigma_xy_y + sigma_xz_z) - p_x + config.density * config.gravity * config.thermal_expansion * T
        residual_y = (sigma_yx_x + sigma_yy_y + sigma_yz_z) - p_y + config.density * config.gravity * config.thermal_expansion * T
        residual_z = (sigma_zx_x + sigma_zy_y + sigma_zz_z) - p_z + config.density * config.gravity * config.thermal_expansion * T
        
        # 返回残差的L2范数
        return torch.sqrt(torch.mean(residual_x**2 + residual_y**2 + residual_z**2))
    
    @staticmethod
    def mass_conservation_equation(x: torch.Tensor, y: torch.Tensor, config: GeologicalConfig) -> torch.Tensor:
        """
        质量守恒方程：∇·v = 0
        
        Args:
            x: 输入坐标 (x, y, z)
            y: 模型输出 [v_x, v_y, v_z] 速度场
        
        Returns:
            质量守恒方程残差
        """
        if not HAS_PYTORCH:
            return torch.tensor(0.0)
        
        vx, vy, vz = y[:, 0], y[:, 1], y[:, 2]
        
        # 计算速度散度
        vx_grad_x = torch.autograd.grad(
            vx.sum(), x[:, 0], 
            grad_outputs=torch.ones_like(vx), 
            create_graph=True, retain_graph=True
        )[0]
        
        vy_grad_y = torch.autograd.grad(
            vy.sum(), x[:, 1], 
            grad_outputs=torch.ones_like(vy), 
            create_graph=True, retain_graph=True
        )[0]
        
        vz_grad_z = torch.autograd.grad(
            vz.sum(), x[:, 2], 
            grad_outputs=torch.ones_like(vz), 
            create_graph=True, retain_graph=True
        )[0]
        
        # 质量守恒残差：∇·v = 0
        residual = vx_grad_x + vy_grad_y + vz_grad_z
        
        return torch.mean(torch.abs(residual))


class GeologicalPINN(BaseSolver, nn.Module):
    """
    地质物理信息神经网络（Geological PINN）
    
    核心思想：将地质物理方程（如达西定律、热传导方程）作为软约束嵌入神经网络，
    强制模型输出满足地质物理规律，实现"小数据+强物理"的地质建模
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 physics_equations: List[Callable] = None,
                 boundary_conditions: List[Callable] = None,
                 geological_config: GeologicalConfig = None):
        BaseSolver.__init__(self)
        nn.Module.__init__(self)
        
        if not HAS_PYTORCH:
            raise ImportError("需要安装PyTorch来使用Geological PINN")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.physics_equations = physics_equations or []
        self.boundary_conditions = boundary_conditions or []
        self.geological_config = geological_config or GeologicalConfig()
        
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
        
        print(f"🔄 地质PINN初始化完成 - 设备: {self.device}")
        print(f"   网络结构: {input_dim} -> {hidden_dims} -> {output_dim}")
        print(f"   物理方程数量: {len(self.physics_equations)}")
    
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
        """计算地质物理约束损失"""
        if not self.physics_equations:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        for equation in self.physics_equations:
            # 计算物理方程的残差
            residual = equation(x, y, self.geological_config)
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
              geological_features: np.ndarray = None,
              epochs: int = 1000, 
              physics_weight: float = 1.0,
              boundary_weight: float = 1.0,
              batch_size: int = 32,
              validation_split: float = 0.2) -> dict:
        """训练地质PINN模型"""
        if self.optimizer is None:
            self.setup_training()
        
        # 验证输入数据
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X和y样本数不匹配：{X.shape[0]} vs {y.shape[0]}")
        
        # 合并地质特征（如果提供）
        if geological_features is not None:
            if geological_features.shape[0] != X.shape[0]:
                raise ValueError(f"地质特征样本数不匹配：{geological_features.shape[0]} vs {X.shape[0]}")
            X = np.hstack([X, geological_features])
            print(f"   合并地质特征，输入维度: {X.shape[1]}")
            
            # 动态调整网络输入维度
            if X.shape[1] != self.input_dim:
                print(f"   动态调整网络输入维度: {self.input_dim} -> {X.shape[1]}")
                self._adjust_input_dim(X.shape[1])
        
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
            nn.Module.train(self, True)  # 设置为训练模式
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
                nn.Module.eval(self)
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
        return history
    
    def _adjust_input_dim(self, new_input_dim: int):
        """动态调整网络输入维度"""
        if new_input_dim != self.input_dim:
            # 重新构建第一层
            old_first_layer = self.layers[0]
            new_first_layer = nn.Linear(new_input_dim, old_first_layer.out_features)
            
            # 初始化新层权重
            nn.init.xavier_normal_(new_first_layer.weight)
            nn.init.constant_(new_first_layer.bias, 0)
            
            # 替换第一层
            self.layers[0] = new_first_layer
            self.input_dim = new_input_dim
            
            # 移动到设备
            self.to(self.device)
            
            print(f"   网络输入维度已调整: {new_input_dim}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self(X_tensor)
            return outputs.cpu().numpy()


class GeologicalSurrogateModel(BaseSolver):
    """
    地质代理模型 - 专门用于地质数值模拟加速
    
    核心思想：用传统地质数值模拟生成"地质参数→模拟输出"的数据集，
    训练ML模型学习这种映射，后续用模型直接预测，替代完整模拟流程
    
    扩展功能：
    1. 支持多种模型类型：高斯过程、随机森林、梯度提升、XGBoost、LightGBM等
    2. 交叉验证和模型评估
    3. 特征重要性分析
    4. 批量预测优化
    5. 不确定性估计
    6. 模型持久化
    """
    
    def __init__(self, model_type: str = 'gaussian_process', geological_config: GeologicalConfig = None):
        super().__init__()
        self.model_type = model_type
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.geological_config = geological_config or GeologicalConfig()
        self.training_history = {}
        self.feature_importance = None
        self.cv_scores = None
        
        # 检查依赖
        if model_type in ['gaussian_process', 'random_forest', 'gradient_boosting', 'mlp'] and not HAS_SKLEARN:
            raise ImportError("需要安装scikit-learn来使用该代理模型")
        
        # 检查XGBoost和LightGBM依赖
        if model_type in ['xgboost', 'lightgbm']:
            try:
                if model_type == 'xgboost':
                    import xgboost as xgb
                    self.xgb = xgb
                elif model_type == 'lightgbm':
                    import lightgbm as lgb
                    self.lgb = lgb
            except ImportError:
                raise ImportError(f"需要安装{model_type}来使用该代理模型")
    
    def train(self, X: np.ndarray, y: np.ndarray, geological_features: np.ndarray = None, 
              cv: int = 0, **kwargs) -> dict:
        """
        训练地质代理模型（支持交叉验证和高维输出）
        
        Args:
            X: 输入特征 (n_samples, n_params)，如 [黏度系数, 热导率, 边界温度]
            y: 输出物理场 (n_samples, H, W, D) 或 (n_samples, n_outputs)，如三维温度场
            geological_features: 地质特征（可选）
            cv: 交叉验证折数（0表示不进行交叉验证）
            **kwargs: 模型特定参数
        """
        # 验证输入数据合法性
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X和y样本数不匹配：{X.shape[0]} vs {y.shape[0]}")
        
        start_time = time.time()
        
        # 处理高维输出（如3D场数据）
        if y.ndim > 2:
            self.y_shape = y.shape[1:]  # 记录原始形状 (H, W, D)
            y_reshaped = y.reshape(y.shape[0], -1)  # 展平为 (n_samples, H*W*D)
            print(f"   检测到高维输出，原始形状: {y.shape}，展平后: {y_reshaped.shape}")
        else:
            self.y_shape = None
            y_reshaped = y
        
        # 合并地质特征（如果提供）
        if geological_features is not None:
            if geological_features.shape[0] != X.shape[0]:
                raise ValueError(f"地质特征样本数不匹配：{geological_features.shape[0]} vs {X.shape[0]}")
            X = np.hstack([X, geological_features])
            print(f"   合并地质特征，输入维度: {X.shape[1]}")
        
        # 数据标准化
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y_reshaped.reshape(-1, 1)).flatten()
        
        # 根据模型类型初始化并训练
        if self.model_type == 'gaussian_process':
            kernel = kwargs.get('kernel', RBF(length_scale=1.0) + ConstantKernel())
            self.model = GaussianProcessRegressor(kernel=kernel, random_state=42)
            self.model.fit(X_scaled, y_scaled)
        
        elif self.model_type == 'random_forest':
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', None)
            min_samples_split = kwargs.get('min_samples_split', 2)
            min_samples_leaf = kwargs.get('min_samples_leaf', 1)
            
            if not isinstance(n_estimators, int) or n_estimators <= 0:
                raise ValueError(f"n_estimators必须为正整数，实际为：{n_estimators}")
            
            self.model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            self.model.fit(X_scaled, y_scaled)
            
            # 计算特征重要性
            self.feature_importance = self.model.feature_importances_
        
        elif self.model_type == 'gradient_boosting':
            n_estimators = kwargs.get('n_estimators', 100)
            learning_rate = kwargs.get('learning_rate', 0.1)
            max_depth = kwargs.get('max_depth', 3)
            
            if not isinstance(n_estimators, int) or n_estimators <= 0:
                raise ValueError(f"n_estimators必须为正整数，实际为：{n_estimators}")
            
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
                raise ValueError(f"learning_rate必须为正数，实际为：{learning_rate}")
            
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators, 
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
            self.model.fit(X_scaled, y_scaled)
            
            # 计算特征重要性
            self.feature_importance = self.model.feature_importances_
        
        elif self.model_type == 'mlp':
            hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (100, 50))
            max_iter = kwargs.get('max_iter', 1000)
            learning_rate_init = kwargs.get('learning_rate_init', 0.001)
            
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                learning_rate_init=learning_rate_init,
                random_state=42
            )
            self.model.fit(X_scaled, y_scaled)
        
        elif self.model_type == 'xgboost':
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', 3)
            learning_rate = kwargs.get('learning_rate', 0.1)
            
            self.model = self.xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )
            self.model.fit(X_scaled, y_scaled)
            
            # 计算特征重要性
            self.feature_importance = self.model.feature_importances_
        
        elif self.model_type == 'lightgbm':
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', 3)
            learning_rate = kwargs.get('learning_rate', 0.1)
            
            self.model = self.lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )
            self.model.fit(X_scaled, y_scaled)
            
            # 计算特征重要性
            self.feature_importance = self.model.feature_importances_
        
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 交叉验证（可选）
        if cv > 1:
            from sklearn.model_selection import cross_val_score
            self.cv_scores = cross_val_score(
                self.model, X_scaled, y_scaled, cv=cv, scoring='neg_mean_squared_error'
            )
            self.cv_scores = -self.cv_scores  # 转换为MSE
            print(f"   交叉验证MSE: {self.cv_scores.mean():.6f} ± {self.cv_scores.std():.6f}")
        
        self.is_trained = True
        training_time = time.time() - start_time
        
        self.training_history = {
            'training_time': training_time,
            'model_type': self.model_type,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'geological_config': self.geological_config,
            'cv_scores': self.cv_scores,
            'feature_importance': self.feature_importance
        }
        
        return self.training_history
    
    def predict(self, X: np.ndarray, return_std: bool = False, batch_size: int = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        预测 - 支持批量处理、不确定性估计和高维输出恢复
        
        Args:
            X: 输入特征
            return_std: 是否返回标准差
            batch_size: 批量大小（None表示自动选择）
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 自动选择批量大小
        if batch_size is None:
            batch_size = 1024 if self.geological_config.use_gpu else len(X)
        
        # 批量处理大尺寸地质数据
        n_batches = (len(X) + batch_size - 1) // batch_size
        predictions = []
        stds = [] if return_std else None
        
        for i in range(n_batches):
            X_batch = X[i*batch_size : (i+1)*batch_size]
            X_scaled = self.scaler_X.transform(X_batch)
            
            # GPU加速（如果支持且启用）
            if (HAS_PYTORCH and isinstance(self.model, torch.nn.Module) 
                and self.geological_config.use_gpu and torch.cuda.is_available()):
                X_scaled = torch.tensor(X_scaled, device=self.device)
                with torch.no_grad():
                    y_pred_scaled = self.model(X_scaled).cpu().detach().numpy()
            else:
                y_pred_scaled = self.model.predict(X_scaled)
            
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            predictions.append(y_pred)
            
            # 处理不确定性估计
            if return_std:
                if self.model_type == 'gaussian_process':
                    _, std_scaled = self.model.predict(X_scaled, return_std=True)
                    std = self.scaler_y.inverse_transform(std_scaled.reshape(-1, 1)).flatten()
                elif self.model_type == 'random_forest':
                    # 随机森林：通过树的预测差异估计标准差
                    tree_preds = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
                    std_scaled = np.std(tree_preds, axis=0)
                    std = self.scaler_y.inverse_transform(std_scaled.reshape(-1, 1)).flatten()
                elif self.model_type in ['xgboost', 'lightgbm']:
                    # XGBoost/LightGBM：通过多次预测估计不确定性
                    n_estimators = len(self.model.estimators_) if hasattr(self.model, 'estimators_') else 10
                    tree_preds = []
                    for _ in range(min(n_estimators, 10)):  # 限制预测次数
                        pred = self.model.predict(X_scaled)
                        tree_preds.append(pred)
                    std_scaled = np.std(tree_preds, axis=0)
                    std = self.scaler_y.inverse_transform(std_scaled.reshape(-1, 1)).flatten()
                else:
                    std = np.zeros_like(y_pred)
                stds.append(std)
        
        predictions = np.concatenate(predictions)
        
        # 恢复高维输出形状（如果训练时是高维数据）
        if self.y_shape is not None and not return_std:
            predictions = predictions.reshape(-1, *self.y_shape)
            print(f"   恢复高维输出形状: {predictions.shape}")
        
        if return_std:
            stds = np.concatenate(stds)
            # 对于高维输出，标准差也需要恢复形状
            if self.y_shape is not None:
                stds = stds.reshape(-1, *self.y_shape)
            return predictions, stds
        
        return predictions
    
    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性（仅支持树模型）"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        if self.model_type in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
            if self.feature_importance is not None:
                return self.feature_importance
            else:
                return self.model.feature_importances_
        else:
            raise NotImplementedError(f"{self.model_type}不支持特征重要性分析")
    
    def get_model_performance(self) -> dict:
        """获取模型性能指标"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        performance = {
            'model_type': self.model_type,
            'training_time': self.training_history.get('training_time', 0),
            'n_samples': self.training_history.get('n_samples', 0),
            'n_features': self.training_history.get('n_features', 0)
        }
        
        if self.cv_scores is not None:
            performance.update({
                'cv_mean_mse': self.cv_scores.mean(),
                'cv_std_mse': self.cv_scores.std(),
                'cv_scores': self.cv_scores.tolist()
            })
        
        if self.feature_importance is not None:
            performance['feature_importance'] = self.feature_importance.tolist()
        
        return performance
    
    def save_model(self, filepath: str):
        """保存模型（包含scaler和模型参数）"""
        try:
            import pickle
            model_data = {
                'model_type': self.model_type,
                'model': self.model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'feature_importance': self.feature_importance,
                'cv_scores': self.cv_scores,
                'geological_config': self.geological_config
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"📁 地质代理模型已保存: {filepath}")
        except Exception as e:
            raise RuntimeError(f"保存模型失败：{str(e)}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model_type = model_data['model_type']
            self.model = model_data['model']
            self.scaler_X = model_data['scaler_X']
            self.scaler_y = model_data['scaler_y']
            self.is_trained = model_data['is_trained']
            self.training_history = model_data.get('training_history', {})
            self.feature_importance = model_data.get('feature_importance', None)
            self.cv_scores = model_data.get('cv_scores', None)
            self.geological_config = model_data.get('geological_config', GeologicalConfig())
            
            print(f"📁 地质代理模型已加载: {filepath}")
        except FileNotFoundError:
            raise FileNotFoundError(f"模型文件不存在：{filepath}")
        except Exception as e:
            raise RuntimeError(f"加载模型失败：{str(e)}")
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'geological_config': self.geological_config
        }
        
        if self.is_trained:
            info.update({
                'n_samples': self.training_history.get('n_samples', 0),
                'n_features': self.training_history.get('n_features', 0),
                'training_time': self.training_history.get('training_time', 0)
            })
        
        return info


class GeologicalUNet(BaseSolver, nn.Module):
    """
    地质UNet - 专门用于处理地质空间场数据
    
    核心思想：用UNet实现"低精度地质数据→高精度地质场"的端到端映射，
    如地震数据反演地质结构、地质场超分辨率重建等
    """
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1, 
                 initial_features: int = 64, depth: int = 4):
        BaseSolver.__init__(self)
        nn.Module.__init__(self)
        
        if not HAS_PYTORCH:
            raise ImportError("需要安装PyTorch来使用Geological UNet")
        
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
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
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
                    nn.ConvTranspose2d(in_channels * 2, out_channels, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels
        
        # 输出层
        self.output_layer = nn.Conv2d(initial_features, output_channels, kernel_size=1)
        
        self.to(self.device)
        
        print(f"🔄 地质UNet初始化完成 - 设备: {self.device}")
        print(f"   输入通道: {input_channels}, 输出通道: {output_channels}")
        print(f"   初始特征: {initial_features}, 深度: {depth}")
    
    def forward(self, x):
        """前向传播"""
        # 编码器
        encoder_outputs = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            encoder_outputs.append(x)
            x = F.max_pool2d(x, 2)
        
        # 解码器
        for i, decoder_block in enumerate(self.decoder):
            # 上采样
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # 跳跃连接
            skip_connection = encoder_outputs[-(i + 1)]
            x = torch.cat([x, skip_connection], dim=1)
            x = decoder_block(x)
        
        # 输出层
        x = self.output_layer(x)
        return x
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
              batch_size: int = 8, learning_rate: float = 0.001) -> dict:
        """训练UNet模型"""
        # 验证输入数据
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X和y样本数不匹配：{X.shape[0]} vs {y.shape[0]}")
        
        # 数据预处理
        if X.ndim == 3:
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        if y.ndim == 3:
            y = y.reshape(y.shape[0], 1, y.shape[1], y.shape[2])
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 优化器
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        history = {'loss': [], 'train_time': 0.0}
        start_time = time.time()
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            history['loss'].append(epoch_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}: loss={epoch_loss:.6f}")
        
        history['train_time'] = time.time() - start_time
        self.is_trained = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.eval()
        with torch.no_grad():
            if X.ndim == 3:
                X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
            
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self(X_tensor)
            return outputs.cpu().numpy()


class GeologicalMultiScaleBridge:
    """
    地质多尺度桥接器 - 用于跨尺度地质模拟
    
    核心思想：在跨尺度地质模拟中（如从微观孔隙到宏观油藏），
    用ML模型替代小尺度精细模拟，将小尺度结果"打包"为大尺度模型的参数
    
    增强功能：
    1. 地质尺度约束（质量守恒、动量守恒等）
    2. 微观特征提取（孔隙度统计、梯度分析等）
    3. 支持多种桥接模型类型
    4. 地质特定的尺度转换逻辑
    """
    
    def __init__(self, fine_scale_model: Callable = None, coarse_scale_model: Callable = None):
        self.fine_scale_model = fine_scale_model
        self.coarse_scale_model = coarse_scale_model
        self.bridge_model = None
        self.is_trained = False
        self.scale_ratio = 1000.0  # 地质尺度比（1千米=1e6毫米）
        self.bridge_type = 'neural_network'
        self.geology_constraints = []  # 地质尺度约束
        self.fine_features_extractor = None  # 微观特征提取器
        
    def add_geology_constraint(self, constraint: Callable):
        """添加地质尺度转换约束（如质量守恒、动量守恒）"""
        self.geology_constraints.append(constraint)
        print(f"✅ 添加地质约束: {constraint.__name__ if hasattr(constraint, '__name__') else 'custom'}")
    
    def set_fine_features_extractor(self, extractor: Callable):
        """设置微观特征提取器"""
        self.fine_features_extractor = extractor
        print(f"✅ 设置微观特征提取器: {extractor.__name__ if hasattr(extractor, '__name__') else 'custom'}")
    
    def setup_bridge_model(self, input_dim: int, output_dim: int, model_type: str = 'neural_network'):
        """设置桥接模型"""
        self.bridge_type = model_type
        
        if model_type == 'neural_network' and HAS_PYTORCH:
            self.bridge_model = GeologicalPINN(
                input_dim, [128, 64, 32], output_dim
            )
        elif model_type == 'surrogate':
            self.bridge_model = GeologicalSurrogateModel('gaussian_process')
        elif model_type == 'random_forest':
            self.bridge_model = GeologicalSurrogateModel('random_forest', n_estimators=200)
        elif model_type == 'gradient_boosting':
            self.bridge_model = GeologicalSurrogateModel('gradient_boosting')
        else:
            raise ValueError(f"不支持的桥接模型类型: {model_type}")
        
        print(f"✅ 设置桥接模型: {model_type}, 输入维度: {input_dim}, 输出维度: {output_dim}")
    
    def train_bridge(self, fine_scale_data: np.ndarray, coarse_scale_params: np.ndarray, **kwargs) -> dict:
        """
        训练尺度桥接模型：从微观数据映射到宏观参数
        
        Args:
            fine_scale_data: 微观模拟结果（如岩石孔隙度分布）
            coarse_scale_params: 宏观等效参数（如等效黏度、渗透率）
        """
        if self.bridge_model is None:
            raise ValueError("桥接模型尚未设置")
        
        # 提取微观特征（如孔隙度均值、梯度、各向异性）
        if self.fine_features_extractor is not None:
            fine_features = self.fine_features_extractor(fine_scale_data)
        else:
            fine_features = self._extract_fine_features(fine_scale_data)
        
        print(f"   微观特征维度: {fine_features.shape}")
        print(f"   宏观参数维度: {coarse_scale_params.shape}")
        
        # 训练桥接模型，同时施加地质约束
        if self.bridge_type == 'neural_network' and isinstance(self.bridge_model, GeologicalPINN):
            # 设置物理约束
            if self.geology_constraints:
                self.bridge_model.physics_equations = self.geology_constraints
                print(f"   应用地质约束: {len(self.geology_constraints)} 个")
            
            # 训练PINN
            self.bridge_model.setup_training()
            result = self.bridge_model.train(
                fine_features, coarse_scale_params, 
                physics_weight=1.0,  # 地质约束权重
                **kwargs
            )
        else:
            # 训练其他类型的代理模型
            result = self.bridge_model.train(fine_features, coarse_scale_params, **kwargs)
        
        self.is_trained = True
        return result
    
    def _extract_fine_features(self, fine_data: np.ndarray) -> np.ndarray:
        """从微观数据中提取对宏观有效的特征（地质领域知识）"""
        features = []
        
        for sample in fine_data:
            sample_features = []
            
            # 基础统计特征
            sample_features.extend([
                np.mean(sample),           # 均值
                np.std(sample),            # 标准差
                np.percentile(sample, 25), # 25分位数
                np.percentile(sample, 75), # 75分位数
                np.max(sample),            # 最大值
                np.min(sample)             # 最小值
            ])
            
            # 空间梯度特征（反映非均匀性）
            if sample.ndim > 1:
                # 计算空间梯度
                if sample.ndim == 2:  # 2D数据
                    grad_x = np.gradient(sample, axis=1)
                    grad_y = np.gradient(sample, axis=0)
                    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                elif sample.ndim == 3:  # 3D数据
                    grad_x = np.gradient(sample, axis=2)
                    grad_y = np.gradient(sample, axis=1)
                    grad_z = np.gradient(sample, axis=0)
                    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
                else:
                    gradient_magnitude = np.zeros_like(sample)
                
                sample_features.extend([
                    np.mean(gradient_magnitude),  # 平均梯度强度
                    np.std(gradient_magnitude),   # 梯度强度标准差
                    np.max(gradient_magnitude)    # 最大梯度强度
                ])
            else:
                # 1D数据
                gradient = np.gradient(sample)
                sample_features.extend([
                    np.mean(np.abs(gradient)),    # 平均梯度强度
                    np.std(gradient),             # 梯度标准差
                    np.max(np.abs(gradient))      # 最大梯度强度
                ])
            
            # 各向异性特征（地质结构特征）
            if sample.ndim > 1:
                # 计算不同方向的方差
                if sample.ndim == 2:
                    var_x = np.var(sample, axis=1)  # 沿x方向方差
                    var_y = np.var(sample, axis=0)  # 沿y方向方差
                    anisotropy = np.std(var_x) / (np.std(var_y) + 1e-10)  # 各向异性比
                elif sample.ndim == 3:
                    var_x = np.var(sample, axis=(1, 2))  # 沿x方向方差
                    var_y = np.var(sample, axis=(0, 2))  # 沿y方向方差
                    var_z = np.var(sample, axis=(0, 1))  # 沿z方向方差
                    anisotropy = np.std([np.std(var_x), np.std(var_y), np.std(var_z)])
                
                sample_features.append(anisotropy)
            else:
                sample_features.append(0.0)  # 1D数据无各向异性
            
            features.append(sample_features)
        
        return np.array(features)
    
    def predict_coarse_from_fine(self, fine_data: np.ndarray) -> np.ndarray:
        """从细尺度数据预测粗尺度数据"""
        if not self.is_trained:
            raise ValueError("桥接模型尚未训练")
        
        # 提取微观特征
        if self.fine_features_extractor is not None:
            fine_features = self.fine_features_extractor(fine_data)
        else:
            fine_features = self._extract_fine_features(fine_data)
        
        # 预测宏观参数
        return self.bridge_model.predict(fine_features)
    
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
        print(f"✅ 设置地质尺度比: 1:{ratio}")
    
    def get_bridge_info(self) -> dict:
        """获取桥接器信息"""
        return {
            'bridge_type': self.bridge_type,
            'scale_ratio': self.scale_ratio,
            'is_trained': self.is_trained,
            'geology_constraints': len(self.geology_constraints),
            'fine_features_extractor': self.fine_features_extractor is not None
        }


class GeologicalHybridAccelerator:
    """
    地质混合加速器 - 结合传统地质数值模拟和ML
    
    核心思想：无法完全替代传统地质模拟（需高精度），
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
        # 新增：阶段加速策略
        self.stage_strategies = {
            'mesh_generation': None,
            'solver': None,
            'postprocessing': None
        }
    
    def add_ml_model(self, name: str, model: Union[GeologicalPINN, GeologicalSurrogateModel]):
        """添加ML模型"""
        self.ml_models[name] = model
        print(f"✅ 添加地质ML模型: {name}")
    
    def setup_acceleration_strategy(self, strategy: str, model_name: str):
        """设置加速策略"""
        if model_name in self.ml_models:
            self.acceleration_strategies[strategy] = model_name
            print(f"✅ 设置地质加速策略: {strategy} -> {model_name}")
        else:
            raise ValueError(f"ML模型 {model_name} 不存在")
    
    def setup_stage_strategy(self, stage: str, model_name: str):
        """为模拟的不同阶段设置加速模型"""
        valid_stages = ['mesh_generation', 'solver', 'postprocessing']
        if stage in valid_stages:
            if model_name in self.ml_models:
                self.stage_strategies[stage] = model_name
                print(f"✅ 设置地质阶段加速策略: {stage} -> {model_name}")
            else:
                raise ValueError(f"ML模型 {model_name} 不存在")
        else:
            raise ValueError(f"无效的阶段类型: {stage}，有效选项: {valid_stages}")
    
    def solve_hybrid(self, problem_data: Dict, use_ml: bool = True, 
                    ml_model_name: str = None) -> Dict:
        """混合求解 - 支持动态切换策略"""
        start_time = time.time()
        self.performance_stats['total_calls'] += 1
        
        # 新增：若问题要求高精度，强制使用传统方法+ML加速
        if problem_data.get('accuracy_requirement', 0) < 1e-5 and use_ml:
            use_ml = False  # 禁用纯ML预测，仅用ML做初始猜测
            if ml_model_name:
                self.acceleration_strategies['initial_guess'] = ml_model_name
            print(f"   高精度要求，启用ML初始猜测加速")
        
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
                print(f"   使用ML初始猜测加速")
            
            # 检查阶段加速策略
            if 'solver' in problem_data.get('stage', '') and self.stage_strategies['solver']:
                # 使用ML加速求解阶段
                solver_model = self.ml_models[self.stage_strategies['solver']]
                problem_data['solver_acceleration'] = solver_model
            
            result = self.traditional_solver(problem_data)
            traditional_time = time.time() - start_time
            
            self.performance_stats['traditional_time'] += traditional_time
            
            return {
                'solution': result,
                'method': 'traditional',
                'time': traditional_time
            }
    
    def compare_methods(self, problem_data: Dict, ml_model_name: str = None) -> Dict:
        """比较传统方法和ML方法 - 增强地质场景评估"""
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
            
            result = {
                'traditional': traditional_result,
                'ml': ml_result,
                'speedup': speedup,
                'accuracy_loss': accuracy_loss
            }
            
            # 新增：与地质实测数据的拟合度评估
            if 'field_measurements' in problem_data:
                field_data = problem_data['field_measurements']
                ml_fit = np.mean(np.abs(ml_result['solution'] - field_data))
                traditional_fit = np.mean(np.abs(traditional_result['solution'] - field_data))
                result.update({
                    'ml_field_fit': ml_fit,
                    'traditional_field_fit': traditional_fit
                })
            
            return result
        
        return {'traditional': traditional_result}
    
    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        if stats['total_calls'] > 0:
            stats['avg_traditional_time'] = stats['traditional_time'] / stats['total_calls']
            stats['avg_ml_time'] = stats['ml_time'] / stats['total_calls']
        return stats


class GeologicalAdaptiveSolver:
    """
    地质自适应求解器 - 专门针对地质场景的求解器选择
    
    核心思想：根据地质问题特征（如是否含断层、孔隙度分布、精度要求等）
    自动选择最优的求解策略（ML加速或传统方法）
    """
    
    def __init__(self):
        self.solvers = {}
        self.selection_strategy = 'performance'  # 'performance', 'rules', 'hybrid'
        self.score_weights = {
            'problem_feature': 1.0,
            'accuracy': 0.5,
            'speed': 0.5,
            'priority': 0.1,
            'geological_complexity': 0.3  # 新增：地质复杂度权重
        }
        self.performance_history = {}
    
    def add_solver(self, name: str, solver: Callable, 
                  conditions: Dict = None, priority: int = 1):
        """添加求解器"""
        self.solvers[name] = {
            'solver': solver,
            'conditions': conditions or {},
            'priority': priority,
            'performance': {
                'accuracy': 0.0,
                'speed': 0.0,
                'geological_fit': 0.0  # 新增：地质拟合度
            }
        }
        print(f"✅ 添加地质求解器: {name}")
    
    def set_selection_strategy(self, strategy: str):
        """设置选择策略"""
        valid_strategies = ['performance', 'rules', 'hybrid']
        if strategy in valid_strategies:
            self.selection_strategy = strategy
            print(f"✅ 设置地质选择策略: {strategy}")
        else:
            raise ValueError(f"无效的选择策略: {strategy}")
    
    def set_score_weights(self, weights: Dict[str, float]):
        """设置评分权重"""
        self.score_weights.update(weights)
        print(f"✅ 更新地质评分权重: {weights}")
    
    def select_best_solver(self, problem_data: Dict) -> str:
        """选择最佳求解器"""
        if not self.solvers:
            raise ValueError("没有可用的求解器")
        
        if self.selection_strategy == 'performance':
            return self._select_by_performance(problem_data)
        elif self.selection_strategy == 'rules':
            return self._select_by_rules(problem_data)
        elif self.selection_strategy == 'hybrid':
            return self._select_hybrid(problem_data)
        else:
            raise ValueError(f"未知的选择策略: {self.selection_strategy}")
    
    def _select_by_performance(self, problem_data: Dict) -> str:
        """基于性能选择求解器"""
        best_solver = None
        best_score = -float('inf')
        
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
        
        # 如果没有匹配的规则，返回优先级最高的求解器
        return max(self.solvers.keys(), key=lambda k: self.solvers[k]['priority'])
    
    def _select_hybrid(self, problem_data: Dict) -> str:
        """混合选择策略"""
        # 首先尝试规则匹配
        for name, solver_info in self.solvers.items():
            if self._check_conditions(solver_info['conditions'], problem_data):
                return name
        
        # 如果没有匹配的规则，使用性能选择
        return self._select_by_performance(problem_data)
    
    def _evaluate_solver_performance(self, name: str, solver_info: Dict, problem_data: Dict) -> float:
        """评估求解器性能 - 增强地质场景评估"""
        score = 0.0
        weights = self.score_weights
        
        # 问题特征评估
        if 'size' in problem_data:
            if problem_data['size'] < 1000 and solver_info['conditions'].get('small_problems', False):
                score += weights['problem_feature']
            elif problem_data['size'] >= 1000 and solver_info['conditions'].get('large_problems', False):
                score += weights['problem_feature']
        
        # 精度要求匹配
        if 'accuracy_requirement' in problem_data:
            req = problem_data['accuracy_requirement']
            if solver_info['performance']['accuracy'] >= req:
                score += weights['problem_feature'] * 0.5
        
        # 新增：地质特征评估
        if problem_data.get('has_faults', False) and solver_info['conditions'].get('handles_faults', False):
            score += weights['geological_complexity'] * 0.8  # 断层处理能力加分
        
        if problem_data.get('porosity', 0) > 0.2:
            score += solver_info['performance']['accuracy'] * weights['accuracy'] * 1.2  # 高孔隙度区域提升精度权重
        
        # 性能评估
        performance = solver_info['performance']
        score += performance['accuracy'] * weights['accuracy'] + performance['speed'] * weights['speed']
        score += solver_info['priority'] * weights['priority']
        
        return score
    
    def _check_conditions(self, conditions: Dict, problem_data: Dict) -> bool:
        """检查条件 - 支持范围条件"""
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
        """求解"""
        start_time = time.time()
        
        # 选择最佳求解器
        best_solver_name = self.select_best_solver(problem_data)
        solver_info = self.solvers[best_solver_name]
        solver = solver_info['solver']
        
        print(f"🔍 选择地质求解器: {best_solver_name}")
        
        # 执行求解
        try:
            result = solver(problem_data)
            solve_time = time.time() - start_time
            
            # 更新性能统计
            solver_info['performance']['speed'] = 1.0 / (1.0 + solve_time)  # 归一化速度
            
            # 计算精度（如果有参考解）
            if 'reference_solution' in problem_data:
                mse = np.mean((result - problem_data['reference_solution'])**2)
                accuracy = 1.0 / (1.0 + mse)  # 归一化到 [0,1]
                solver_info['performance']['accuracy'] = accuracy
            
            # 计算地质拟合度（如果有实测数据）
            if 'field_measurements' in problem_data:
                field_fit = np.mean(np.abs(result - problem_data['field_measurements']))
                geological_fit = 1.0 / (1.0 + field_fit)  # 归一化到 [0,1]
                solver_info['performance']['geological_fit'] = geological_fit
            
            return {
                'solution': result,
                'solver_name': best_solver_name,
                'time': solve_time,
                'performance': solver_info['performance']
            }
        
        except Exception as e:
            print(f"❌ 求解器 {best_solver_name} 失败: {e}")
            raise
    
    def get_performance_summary(self) -> Dict:
        """获取性能总结"""
        summary = {
            'total_solvers': len(self.solvers),
            'selection_strategy': self.selection_strategy,
            'solver_performance': {}
        }
        
        for name, solver_info in self.solvers.items():
            summary['solver_performance'][name] = {
                'priority': solver_info['priority'],
                'performance': solver_info['performance'],
                'conditions': solver_info['conditions']
            }
        
        return summary


class GeologicalGNN(BaseSolver, nn.Module):
    """
    地质图神经网络 - 专门用于处理地质体的拓扑结构
    
    核心思想：利用图神经网络处理地质体的复杂拓扑关系（如断层网络、裂隙分布），
    提升复杂几何场景的模拟精度，实现"结构感知"的地质建模
    """
    
    def __init__(self, node_features: int, edge_features: int, hidden_dim: int = 64, 
                 num_layers: int = 3, output_dim: int = 1, gnn_type: str = 'gcn'):
        BaseSolver.__init__(self)
        nn.Module.__init__(self)
        
        if not HAS_PYTORCH:
            raise ImportError("需要安装PyTorch来使用地质GNN")
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.gnn_type = gnn_type
        
        # 图卷积层
        self.gnn_layers = nn.ModuleList()
        
        if gnn_type == 'gcn':
            # 图卷积网络
            for i in range(num_layers):
                if i == 0:
                    in_dim = node_features
                else:
                    in_dim = hidden_dim
                
                self.gnn_layers.append(
                    nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    )
                )
        
        elif gnn_type == 'gat':
            # 图注意力网络
            for i in range(num_layers):
                if i == 0:
                    in_dim = node_features
                else:
                    in_dim = hidden_dim
                
                self.gnn_layers.append(
                    nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    )
                )
        
        # 边特征处理
        if edge_features > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 地质特定的注意力机制
        self.geological_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4, 
            dropout=0.1
        )
        
        self.to(self.device)
        
        print(f"🔄 地质GNN初始化完成 - 设备: {self.device}")
        print(f"   节点特征: {node_features}, 边特征: {edge_features}")
        print(f"   隐藏维度: {hidden_dim}, 层数: {num_layers}")
        print(f"   GNN类型: {gnn_type}")
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                edge_features: torch.Tensor = None, batch: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: 节点特征 [num_nodes, node_features]
            edge_index: 边索引 [2, num_edges]
            edge_features: 边特征 [num_edges, edge_features] (可选)
            batch: 批次索引 [num_nodes] (可选)
        """
        x = node_features
        
        # 边特征编码
        if edge_features is not None and hasattr(self, 'edge_encoder'):
            edge_embeddings = self.edge_encoder(edge_features)
        else:
            edge_embeddings = None
        
        # 图卷积层
        for i, layer in enumerate(self.gnn_layers):
            if self.gnn_type == 'gcn':
                x = self._gcn_layer(x, edge_index, edge_embeddings, layer)
            elif self.gnn_type == 'gat':
                x = self._gat_layer(x, edge_index, edge_embeddings, layer)
            
            # 地质特定的注意力机制
            if i == self.num_layers - 1:  # 最后一层应用注意力
                x = self._apply_geological_attention(x, batch)
        
        # 输出层
        x = self.output_layer(x)
        
        return x
    
    def _gcn_layer(self, x: torch.Tensor, edge_index: torch.Tensor, 
                   edge_embeddings: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        """图卷积层"""
        # 简化的图卷积实现
        # 在实际应用中，建议使用torch_geometric库
        
        # 计算邻接矩阵
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        
        # 添加自环
        adj = adj + torch.eye(num_nodes, device=x.device)
        
        # 归一化
        degree = adj.sum(dim=1, keepdim=True)
        degree = torch.clamp(degree, min=1e-12)
        adj_norm = adj / degree.sqrt()
        
        # 图卷积
        x = torch.mm(adj_norm, x)
        x = layer(x)
        
        return x
    
    def _gat_layer(self, x: torch.Tensor, edge_index: torch.Tensor, 
                   edge_embeddings: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        """图注意力层"""
        # 简化的图注意力实现
        # 在实际应用中，建议使用torch_geometric库
        
        # 基础变换
        x = layer(x)
        
        # 简单的注意力机制
        if edge_embeddings is not None:
            # 使用边特征增强节点表示
            edge_weights = torch.sigmoid(edge_embeddings.mean(dim=1))
            x = x + torch.scatter_add(
                torch.zeros_like(x), 0, 
                edge_index[1].unsqueeze(1).expand(-1, x.size(1)), 
                x[edge_index[0]] * edge_weights.unsqueeze(1)
            )
        
        return x
    
    def _apply_geological_attention(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """应用地质特定的注意力机制"""
        if batch is not None:
            # 批次处理
            batch_size = batch.max().item() + 1
            x_batched = []
            
            for i in range(batch_size):
                mask = (batch == i)
                if mask.sum() > 0:
                    x_i = x[mask]
                    # 应用自注意力
                    x_i_attended, _ = self.geological_attention(
                        x_i.unsqueeze(0), x_i.unsqueeze(0), x_i.unsqueeze(0)
                    )
                    x_batched.append(x_i_attended.squeeze(0))
            
            if x_batched:
                x = torch.cat(x_batched, dim=0)
        else:
            # 全局注意力
            x_attended, _ = self.geological_attention(
                x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0)
            )
            x = x_attended.squeeze(0)
        
        return x
    
    def train(self, node_features: np.ndarray, edge_index: np.ndarray, 
              target: np.ndarray, edge_features: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001) -> dict:
        """训练地质GNN"""
        # 验证输入数据
        if node_features.shape[0] != target.shape[0]:
            raise ValueError(f"节点特征和目标的节点数不匹配：{node_features.shape[0]} vs {target.shape[0]}")
        
        # 转换为张量
        node_features_tensor = torch.FloatTensor(node_features).to(self.device)
        edge_index_tensor = torch.LongTensor(edge_index).to(self.device)
        target_tensor = torch.FloatTensor(target).to(self.device)
        
        if edge_features is not None:
            edge_features_tensor = torch.FloatTensor(edge_features).to(self.device)
        else:
            edge_features_tensor = None
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        history = {'loss': [], 'train_time': 0.0}
        start_time = time.time()
        
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.forward(node_features_tensor, edge_index_tensor, edge_features_tensor)
            
            # 计算损失
            loss = criterion(outputs, target_tensor)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
            
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1}/{epochs}: loss={loss.item():.6f}")
        
        history['train_time'] = time.time() - start_time
        self.is_trained = True
        
        print(f"✅ 地质GNN训练完成，耗时: {history['train_time']:.4f}秒")
        return history
    
    def predict(self, node_features: np.ndarray, edge_index: np.ndarray, 
                edge_features: np.ndarray = None) -> np.ndarray:
        """预测"""
        self.eval()
        with torch.no_grad():
            node_features_tensor = torch.FloatTensor(node_features).to(self.device)
            edge_index_tensor = torch.LongTensor(edge_index).to(self.device)
            
            if edge_features is not None:
                edge_features_tensor = torch.FloatTensor(edge_features).to(self.device)
            else:
                edge_features_tensor = None
            
            outputs = self.forward(node_features_tensor, edge_index_tensor, edge_features_tensor)
            return outputs.cpu().numpy()
    
    def create_geological_graph(self, spatial_coords: np.ndarray, 
                               geological_features: np.ndarray,
                               connectivity_radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建地质图结构
        
        Args:
            spatial_coords: 空间坐标 [num_points, 3]
            geological_features: 地质特征 [num_points, num_features]
            connectivity_radius: 连接半径
            
        Returns:
            edge_index: 边索引 [2, num_edges]
            edge_features: 边特征 [num_edges, edge_dim]
        """
        num_points = spatial_coords.shape[0]
        
        # 计算点之间的距离
        distances = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(i+1, num_points):
                dist = np.linalg.norm(spatial_coords[i] - spatial_coords[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # 创建边（基于距离阈值）
        edges = []
        edge_features = []
        
        for i in range(num_points):
            for j in range(num_points):
                if i != j and distances[i, j] <= connectivity_radius:
                    edges.append([i, j])
                    
                    # 边特征：距离、地质特征差异
                    edge_feat = [
                        distances[i, j],
                        np.mean(np.abs(geological_features[i] - geological_features[j]))
                    ]
                    edge_features.append(edge_feat)
        
        if edges:
            edge_index = np.array(edges).T
            edge_features = np.array(edge_features)
        else:
            edge_index = np.array([[0], [0]])  # 空图
            edge_features = np.array([[0, 0]])
        
        print(f"   创建地质图: {num_points} 个节点, {edge_index.shape[1]} 条边")
        return edge_index, edge_features
    
    def analyze_topology(self, node_features: np.ndarray, edge_index: np.ndarray) -> Dict:
        """分析地质拓扑结构"""
        analysis = {}
        
        # 节点统计
        analysis['num_nodes'] = node_features.shape[0]
        analysis['num_edges'] = edge_index.shape[1]
        
        # 度分布
        degrees = np.zeros(node_features.shape[0])
        for i in range(edge_index.shape[1]):
            degrees[edge_index[0, i]] += 1
            degrees[edge_index[1, i]] += 1
        
        analysis['avg_degree'] = np.mean(degrees)
        analysis['max_degree'] = np.max(degrees)
        analysis['min_degree'] = np.min(degrees)
        
        # 连通性分析
        if edge_index.shape[1] > 0:
            # 简化的连通性检查
            connected_nodes = set(edge_index.flatten())
            analysis['connectivity'] = len(connected_nodes) / node_features.shape[0]
        else:
            analysis['connectivity'] = 0.0
        
        return analysis


def create_geological_ml_system() -> Dict:
    """创建地质ML系统"""
    system = {
        'pinn': GeologicalPINN,
        'surrogate': GeologicalSurrogateModel,
        'unet': GeologicalUNet,
        'bridge': GeologicalMultiScaleBridge,
        'hybrid': GeologicalHybridAccelerator,
        'adaptive': GeologicalAdaptiveSolver,  # 新增：地质自适应求解器
        'gnn': GeologicalGNN,  # 新增：地质图神经网络
        'physics_equations': GeologicalPhysicsEquations
    }
    
    print("🔄 地质ML系统创建完成")
    return system


def demo_geological_ml():
    """演示地质ML功能"""
    print("🤖 地质数值模拟ML/DL融合演示")
    print("=" * 60)
    
    # 固定随机种子，确保结果可复现
    np.random.seed(42)
    if HAS_PYTORCH:
        torch.manual_seed(42)
    
    # 创建地质ML系统
    ml_system = create_geological_ml_system()
    
    # 生成测试数据
    n_samples = 1000
    input_dim = 4  # x, y, z, t
    output_dim = 1  # 压力场
    
    X = np.random.randn(n_samples, input_dim)
    y = np.random.randn(n_samples, output_dim)
    
    # 生成地质特征数据
    geological_features = np.random.rand(n_samples, 3)  # 孔隙度、渗透率、粘度
    
    print(f"📊 测试数据: {n_samples} 样本, 输入维度: {input_dim}, 输出维度: {output_dim}")
    print(f"   地质特征维度: {geological_features.shape[1]}")
    
    # 1. 测试地质PINN（增强版）
    print("\n🔧 测试地质PINN（增强版）...")
    try:
        # 定义地质物理方程
        darcy_eq = lambda x, y, config: ml_system['physics_equations'].darcy_equation(x, y, config)
        
        pinn = ml_system['pinn'](input_dim, [64, 32], output_dim, 
                                physics_equations=[darcy_eq])
        
        pinn.setup_training()
        result = pinn.train(X, y, geological_features=geological_features, epochs=200)
        print(f"   训练时间: {result['train_time']:.4f} 秒")
        print(f"   最终总损失: {result['total_loss'][-1]:.6f}")
        
    except Exception as e:
        print(f"   ❌ 地质PINN失败: {e}")
    
    # 2. 测试地质代理模型（增强版）
    print("\n🔧 测试地质代理模型（增强版）...")
    try:
        surrogate = ml_system['surrogate']('gaussian_process')
        result = surrogate.train(X, y.flatten(), geological_features=geological_features)
        print(f"   训练时间: {result['training_time']:.4f} 秒")
        
        # 预测
        predictions, std = surrogate.predict(X[:10], return_std=True)
        print(f"   预测形状: {predictions.shape}, 标准差形状: {std.shape}")
        
    except Exception as e:
        print(f"   ❌ 地质代理模型失败: {e}")
    
    # 3. 测试地质UNet
    print("\n🔧 测试地质UNet...")
    try:
        # 生成2D空间数据
        n_samples_2d = 100
        height, width = 64, 64
        X_2d = np.random.randn(n_samples_2d, height, width)
        y_2d = np.random.randn(n_samples_2d, height, width)
        
        unet = ml_system['unet'](input_channels=1, output_channels=1, depth=3)
        result = unet.train(X_2d, y_2d, epochs=50)
        print(f"   训练时间: {result['train_time']:.4f} 秒")
        print(f"   最终损失: {result['loss'][-1]:.6f}")
        
    except Exception as e:
        print(f"   ❌ 地质UNet失败: {e}")
    
    # 4. 测试多尺度桥接
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
    
    # 5. 测试混合加速器（增强版）
    print("\n🔧 测试混合加速器（增强版）...")
    try:
        hybrid_accelerator = ml_system['hybrid']()
        
        # 添加ML模型
        surrogate_model = ml_system['surrogate']('random_forest')
        surrogate_model.train(X, y.flatten(), geological_features=geological_features)
        hybrid_accelerator.add_ml_model('surrogate', surrogate_model)
        
        # 设置加速策略
        hybrid_accelerator.setup_acceleration_strategy('initial_guess', 'surrogate')
        hybrid_accelerator.setup_stage_strategy('solver', 'surrogate')
        
        # 测试混合求解
        problem_data = {
            'input': X[:10],
            'accuracy_requirement': 1e-4,  # 高精度要求
            'stage': 'solver'
        }
        result = hybrid_accelerator.solve_hybrid(problem_data, use_ml=True, ml_model_name='surrogate')
        print(f"   混合求解完成，使用模型: {result['model_name']}")
        print(f"   求解时间: {result['time']:.4f} 秒")
        
    except Exception as e:
        print(f"   ❌ 混合加速器失败: {e}")
    
    # 6. 测试地质自适应求解器（新增）
    print("\n🔧 测试地质自适应求解器...")
    try:
        adaptive_solver = ml_system['adaptive']()
        
        # 定义求解器
        def fast_solver(data):
            return np.random.randn(len(data['input']))
        
        def accurate_solver(data):
            return np.random.randn(len(data['input'])) * 0.1  # 更精确
        
        def fault_solver(data):
            return np.random.randn(len(data['input'])) * 0.05  # 断层处理
        
        # 添加求解器
        adaptive_solver.add_solver('fast', fast_solver, 
                                  conditions={'size': ('<', 1000)}, priority=1)
        adaptive_solver.add_solver('accurate', accurate_solver, 
                                  conditions={'accuracy_requirement': ('>', 0.9)}, priority=2)
        adaptive_solver.add_solver('fault', fault_solver, 
                                  conditions={'has_faults': True}, priority=3)
        
        # 设置选择策略
        adaptive_solver.set_selection_strategy('hybrid')
        adaptive_solver.set_score_weights({'geological_complexity': 0.5})
        
        # 测试不同场景
        scenarios = [
            {'input': X[:100], 'size': 100, 'name': '小规模问题'},
            {'input': X[:100], 'accuracy_requirement': 0.95, 'name': '高精度要求'},
            {'input': X[:100], 'has_faults': True, 'porosity': 0.3, 'name': '复杂地质条件'}
        ]
        
        for scenario in scenarios:
            print(f"   测试场景: {scenario['name']}")
            result = adaptive_solver.solve(scenario)
            print(f"     选择求解器: {result['solver_name']}")
            print(f"     求解时间: {result['time']:.4f} 秒")
        
    except Exception as e:
        print(f"   ❌ 地质自适应求解器失败: {e}")
    
    # 7. 测试地质GNN（新增）
    print("\n🔧 测试地质GNN...")
    try:
        # 生成地质图数据
        n_points = 200
        spatial_coords = np.random.rand(n_points, 3) * 10.0  # 3D空间坐标
        geological_features = np.random.rand(n_points, 5)    # 地质特征（孔隙度、渗透率等）
        
        # 创建GNN模型
        gnn = ml_system['gnn'](
            node_features=5,      # 地质特征维度
            edge_features=2,      # 边特征维度（距离、特征差异）
            hidden_dim=32,
            num_layers=2,
            output_dim=1,
            gnn_type='gcn'
        )
        
        # 创建地质图结构
        edge_index, edge_features = gnn.create_geological_graph(
            spatial_coords, geological_features, connectivity_radius=2.0
        )
        
        # 分析拓扑结构
        topology_analysis = gnn.analyze_topology(geological_features, edge_index)
        print(f"   拓扑分析结果:")
        print(f"     节点数: {topology_analysis['num_nodes']}")
        print(f"     边数: {topology_analysis['num_edges']}")
        print(f"     平均度: {topology_analysis['avg_degree']:.2f}")
        print(f"     连通性: {topology_analysis['connectivity']:.2f}")
        
        # 生成目标数据（模拟地质场值）
        target = np.random.randn(n_points, 1)
        
        # 训练GNN
        print("   训练地质GNN...")
        training_history = gnn.train(
            geological_features, edge_index, target, edge_features,
            epochs=50, learning_rate=0.001
        )
        
        print(f"   训练完成，最终损失: {training_history['loss'][-1]:.6f}")
        
        # 测试预测
        predictions = gnn.predict(geological_features, edge_index, edge_features)
        print(f"   预测形状: {predictions.shape}")
        
        # 计算预测精度
        mse = np.mean((predictions - target)**2)
        print(f"   预测MSE: {mse:.6f}")
        
    except Exception as e:
        print(f"   ❌ 地质GNN失败: {e}")
    
    print("\n✅ 地质数值模拟ML/DL融合演示完成!")


if __name__ == "__main__":
    demo_geological_ml()
