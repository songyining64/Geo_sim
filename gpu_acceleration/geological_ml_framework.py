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

# 可选依赖检查
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_STABLE_BASELINES3 = True
except ImportError:
    HAS_STABLE_BASELINES3 = False
    warnings.warn("stable-baselines3 not available. RL features will be limited.")


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
    
    # 新增：断层摩擦参数
    mu0: float = 0.6                     # 静摩擦系数
    a: float = 0.01                      # 直接效应参数
    b: float = 0.005                     # 演化效应参数
    L: float = 0.1                       # 特征滑移距离 (m)
    v0: float = 1e-6                     # 参考滑动速率 (m/s)
    
    # 新增：化学输运参数
    activation_energy: float = 50e3      # J/mol，激活能
    diffusion_coefficient: float = 1e-9  # m²/s，扩散系数
    reaction_rate: float = 0.01          # s⁻¹，反应速率常数
    
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
    
    @staticmethod
    def fault_slip_equation(x: torch.Tensor, y: torch.Tensor, config: GeologicalConfig) -> torch.Tensor:
        """
        断层滑动方程（摩擦本构）：v = v₀ exp((μ₀ + a ln(v/v₀) - b ln(θ))/L)
        
        Args:
            x: 输入坐标 (x, y, z, t)
            y: 模型输出 [slip_rate, state_variable, stress] 滑动速率、状态变量、应力
            config: 地质配置参数
        
        Returns:
            断层滑动方程残差
        """
        if not HAS_PYTORCH:
            return torch.tensor(0.0)
        
        # 解析输出：滑动速率、状态变量、应力
        slip_rate = y[:, 0]  # 滑动速率
        state = y[:, 1]      # 状态变量 (摩擦状态)
        stress = y[:, 2]     # 应力
        
        # 摩擦参数（从配置中获取，如果没有则使用默认值）
        mu0 = getattr(config, 'mu0', 0.6)      # 静摩擦系数
        a = getattr(config, 'a', 0.01)         # 直接效应参数
        b = getattr(config, 'b', 0.005)        # 演化效应参数
        L = getattr(config, 'L', 0.1)          # 特征滑移距离
        v0 = getattr(config, 'v0', 1e-6)      # 参考滑动速率
        
        # 计算摩擦系数（速率-状态摩擦本构）
        # μ = μ₀ + a ln(v/v₀) + b ln(θ/θ₀)
        theta0 = 1.0  # 参考状态变量
        mu = mu0 + a * torch.log(slip_rate / (v0 + 1e-12)) + b * torch.log(state / (theta0 + 1e-12))
        
        # 状态演化方程（ageing law）
        # dθ/dt = 1 - vθ/L
        if x.shape[1] > 3:  # 有时间维度
            time_derivative = torch.autograd.grad(
                state.sum(), x[:, 3], 
                grad_outputs=torch.ones_like(state), 
                create_graph=True, retain_graph=True
            )[0]
        else:
            # 如果没有时间维度，使用稳态近似
            time_derivative = torch.zeros_like(state)
        
        # 状态演化残差
        state_evolution_residual = time_derivative - (1.0 - slip_rate * state / L)
        
        # 滑动速率与摩擦系数的关系残差
        # τ = μσ (应力 = 摩擦系数 × 正应力)
        normal_stress = config.density * config.gravity * 1000.0  # 简化的正应力计算
        stress_residual = stress - mu * normal_stress
        
        # 总残差
        total_residual = torch.mean(torch.square(state_evolution_residual)) + \
                        torch.mean(torch.square(stress_residual))
        
        return total_residual
    
    @staticmethod
    def mantle_convection_equation(x: torch.Tensor, y: torch.Tensor, config: GeologicalConfig) -> torch.Tensor:
        """
        地幔对流方程：结合Stokes方程和热传导方程
        
        Args:
            x: 输入坐标 (x, y, z, t)
            y: 模型输出 [v_x, v_y, v_z, p, T] 速度场、压力、温度
            config: 地质配置参数
        
        Returns:
            地幔对流方程残差
        """
        if not HAS_PYTORCH:
            return torch.tensor(0.0)
        
        # 解析输出
        vx, vy, vz, p, T = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]
        
        # 1. Stokes方程残差（动量守恒）
        stokes_residual = GeologicalPhysicsEquations.stokes_equation(x, y, config)
        
        # 2. 质量守恒残差
        mass_residual = GeologicalPhysicsEquations.mass_conservation_equation(x, y[:, :3], config)
        
        # 3. 热传导方程残差（考虑对流项）
        # ρc(∂T/∂t + v·∇T) = ∇·(k∇T) + Q
        if x.shape[1] > 3:  # 有时间维度
            # 时间导数
            T_t = torch.autograd.grad(
                T.sum(), x[:, 3], 
                grad_outputs=torch.ones_like(T), 
                create_graph=True, retain_graph=True
            )[0]
        else:
            T_t = torch.zeros_like(T)
        
        # 对流项 v·∇T
        T_x = torch.autograd.grad(T.sum(), x[:, 0], create_graph=True)[0]
        T_y = torch.autograd.grad(T.sum(), x[:, 1], create_graph=True)[0]
        T_z = torch.autograd.grad(T.sum(), x[:, 2], create_graph=True)[0]
        
        convection_term = vx * T_x + vy * T_y + vz * T_z
        
        # 热传导项 ∇·(k∇T)
        T_grad_x = torch.autograd.grad(T.sum(), x[:, 0], create_graph=True)[0]
        T_grad_y = torch.autograd.grad(T.sum(), x[:, 1], create_graph=True)[0]
        T_grad_z = torch.autograd.grad(T.sum(), x[:, 2], create_graph=True)[0]
        
        T_xx = torch.autograd.grad(T_grad_x.sum(), x[:, 0], create_graph=True)[0]
        T_yy = torch.autograd.grad(T_grad_y.sum(), x[:, 1], create_graph=True)[0]
        T_zz = torch.autograd.grad(T_grad_z.sum(), x[:, 2], create_graph=True)[0]
        
        conduction_term = config.thermal_conductivity * (T_xx + T_yy + T_zz)
        
        # 热源项（放射性衰变、粘性耗散等）
        viscous_heating = 0.0  # 简化，实际应计算 η(∇v:∇v)
        heat_source = 0.01 + viscous_heating
        
        # 热传导方程残差
        heat_residual = config.density * config.specific_heat * (T_t + convection_term) - \
                       conduction_term - heat_source
        
        # 4. 浮力项（Boussinesq近似）
        # 密度变化：ρ = ρ₀(1 - α(T - T₀))
        T_ref = 273.15
        density_variation = config.density * config.thermal_expansion * (T - T_ref)
        
        # 总残差（加权组合）
        total_residual = (10.0 * stokes_residual + 
                          5.0 * mass_residual + 
                          2.0 * torch.mean(torch.square(heat_residual)) +
                          1.0 * torch.mean(torch.square(density_variation)))
        
        return total_residual
    
    @staticmethod
    def plate_tectonics_equation(x: torch.Tensor, y: torch.Tensor, config: GeologicalConfig) -> torch.Tensor:
        """
        板块构造方程：结合弹性力学和热传导
        
        Args:
            x: 输入坐标 (x, y, z, t)
            y: 模型输出 [u_x, u_y, u_z, T, stress] 位移场、温度、应力
            config: 地质配置参数
        
        Returns:
            板块构造方程残差
        """
        if not HAS_PYTORCH:
            return torch.tensor(0.0)
        
        # 解析输出
        ux, uy, uz, T, stress = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]
        
        # 1. 弹性平衡方程残差
        elastic_residual = GeologicalPhysicsEquations.elastic_equilibrium_equation(x, y[:, :3], config)
        
        # 2. 热传导方程残差
        heat_residual = GeologicalPhysicsEquations.heat_conduction_equation(x, y[:, 3:4], config)
        
        # 3. 热弹性耦合项
        # 热应力：σ_th = -Eα(T - T₀)/(1 - 2ν)
        T_ref = 273.15
        thermal_stress = -config.youngs_modulus * config.thermal_expansion * (T - T_ref) / \
                        (1 - 2 * config.poissons_ratio)
        
        # 热应力残差
        thermal_stress_residual = stress - thermal_stress
        
        # 4. 板块边界条件（简化）
        # 在板块边界处，位移梯度应该较大
        u_grad_x = torch.autograd.grad(ux.sum(), x[:, 0], create_graph=True)[0]
        u_grad_y = torch.autograd.grad(uy.sum(), x[:, 1], create_graph=True)[0]
        u_grad_z = torch.autograd.grad(uz.sum(), x[:, 2], create_graph=True)[0]
        
        # 位移梯度应该满足一定的约束（板块边界特征）
        boundary_constraint = torch.mean(torch.square(u_grad_x + u_grad_y + u_grad_z))
        
        # 总残差
        total_residual = (5.0 * elastic_residual + 
                          3.0 * heat_residual + 
                          2.0 * torch.mean(torch.square(thermal_stress_residual)) +
                          1.0 * boundary_constraint)
        
        return total_residual
    
    @staticmethod
    def chemical_transport_equation(x: torch.Tensor, y: torch.Tensor, config: GeologicalConfig) -> torch.Tensor:
        """
        化学输运方程：考虑对流-扩散-反应
        
        Args:
            x: 输入坐标 (x, y, z, t)
            y: 模型输出 [C, v_x, v_y, v_z, T] 浓度、速度场、温度
            config: 地质配置参数
        
        Returns:
            化学输运方程残差
        """
        if not HAS_PYTORCH:
            return torch.tensor(0.0)
        
        # 解析输出
        C, vx, vy, vz, T = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4]
        
        # 扩散系数（温度依赖）
        D0 = 1e-9  # 参考扩散系数
        D = D0 * torch.exp(-config.activation_energy / (8.314 * (T + 273.15)))
        
        # 对流项 v·∇C
        C_x = torch.autograd.grad(C.sum(), x[:, 0], create_graph=True)[0]
        C_y = torch.autograd.grad(C.sum(), x[:, 1], create_graph=True)[0]
        C_z = torch.autograd.grad(C.sum(), x[:, 2], create_graph=True)[0]
        
        convection_term = vx * C_x + vy * C_y + vz * C_z
        
        # 扩散项 ∇·(D∇C)
        C_grad_x = torch.autograd.grad(C.sum(), x[:, 0], create_graph=True)[0]
        C_grad_y = torch.autograd.grad(C.sum(), x[:, 1], create_graph=True)[0]
        C_grad_z = torch.autograd.grad(C.sum(), x[:, 2], create_graph=True)[0]
        
        C_xx = torch.autograd.grad(C_grad_x.sum(), x[:, 0], create_graph=True)[0]
        C_yy = torch.autograd.grad(C_grad_y.sum(), x[:, 1], create_graph=True)[0]
        C_zz = torch.autograd.grad(C_grad_z.sum(), x[:, 2], create_graph=True)[0]
        
        diffusion_term = D * (C_xx + C_yy + C_zz)
        
        # 反应项（简化的一级反应）
        reaction_rate = 0.01  # 反应速率常数
        reaction_term = reaction_rate * C
        
        # 时间导数
        if x.shape[1] > 3:  # 有时间维度
            C_t = torch.autograd.grad(C.sum(), x[:, 3], create_graph=True)[0]
        else:
            C_t = torch.zeros_like(C)
        
        # 化学输运方程残差：∂C/∂t + v·∇C = ∇·(D∇C) + R
        transport_residual = C_t + convection_term - diffusion_term - reaction_term
        
        return torch.mean(torch.square(transport_residual))


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
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None, 
                edge_weight: torch.Tensor = None, mesh_data: np.ndarray = None,
                faults: List[Tuple] = None, plate_boundaries: List[Tuple] = None,
                geological_features: np.ndarray = None) -> torch.Tensor:
        """
        前向传播 - 支持GNN增强
        
        Args:
            x: 输入特征
            edge_index: 图边索引（GNN用）
            edge_weight: 图边权重（GNN用）
            mesh_data: 网格数据（GNN图构建用）
            faults: 断层信息（GNN图构建用）
            plate_boundaries: 板块边界信息（GNN图构建用）
            geological_features: 地质特征（GNN图构建用）
        
        Returns:
            模型输出
        """
        # 检查是否启用GNN增强
        if hasattr(self, 'gnn_integrator') and self.gnn_integrator is not None:
            if mesh_data is not None:
                # 使用GNN增强特征
                enhanced_x = self.gnn_integrator.integrate_with_pinn(
                    x, mesh_data, faults, plate_boundaries, geological_features
                )
                # 更新输入特征
                x = enhanced_x
                # 动态调整网络输入维度（如果需要）
                if x.shape[1] != self.input_dim:
                    self._adjust_input_dim(x.shape[1])
        
        # 原有的前向传播逻辑
        if self.use_gpu and torch.cuda.is_available():
            x = x.cuda()
        
        # 前向传播
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # 不是最后一层
                x = F.relu(x)
                if self.dropout_rate > 0:
                    x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        return x
    
    def setup_training(self, learning_rate: float = 0.001, weight_decay: float = 1e-5):
        """设置训练参数"""
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
    
    def setup_gnn_integration(self, gnn_config: Dict = None):
        """
        设置GNN集成
        
        Args:
            gnn_config: GNN配置参数
        """
        try:
            from .geodynamics_gnn import GeodynamicGNN, GeodynamicGraphConfig, GeodynamicsGNNPINNIntegrator
            
            # 创建GNN配置
            if gnn_config is None:
                gnn_config = {
                    'hidden_dim': 64,
                    'num_layers': 3,
                    'attention_heads': 4,
                    'dropout': 0.1
                }
            
            config = GeodynamicGraphConfig(**gnn_config)
            
            # 创建GNN模型
            gnn_input_dim = 8  # 基础特征维度
            gnn_output_dim = 2  # 粘度修正、塑性应变率
            gnn = GeodynamicGNN(gnn_input_dim, config.hidden_dim, gnn_output_dim, config)
            
            # 创建集成器
            self.gnn_integrator = GeodynamicsGNNPINNIntegrator(gnn, config)
            
            print(f"✅ GNN集成设置完成: 隐藏层={config.hidden_dim}, 层数={config.num_layers}")
            
        except ImportError as e:
            warnings.warn(f"无法导入GNN模块: {str(e)}")
            self.gnn_integrator = None
        except Exception as e:
            warnings.warn(f"GNN集成设置失败: {str(e)}")
            self.gnn_integrator = None
    
    def enable_gnn_enhancement(self, enable: bool = True):
        """启用/禁用GNN增强"""
        if enable and not hasattr(self, 'gnn_integrator'):
            self.setup_gnn_integration()
        
        if hasattr(self, 'gnn_integrator'):
            if enable:
                print("✅ GNN增强已启用")
            else:
                print("❌ GNN增强已禁用")
                self.gnn_integrator = None
        else:
            print("❌ GNN集成器未设置")
    
    def get_gnn_status(self) -> Dict[str, Any]:
        """获取GNN集成状态"""
        status = {
            'gnn_enabled': False,
            'gnn_integrator': None,
            'gnn_config': None
        }
        
        if hasattr(self, 'gnn_integrator') and self.gnn_integrator is not None:
            status['gnn_enabled'] = True
            status['gnn_integrator'] = type(self.gnn_integrator).__name__
            if hasattr(self.gnn_integrator, 'config'):
                status['gnn_config'] = {
                    'hidden_dim': self.gnn_integrator.config.hidden_dim,
                    'num_layers': self.gnn_integrator.config.num_layers,
                    'attention_heads': self.gnn_integrator.config.attention_heads
                }
        
        return status
    
    def compute_physics_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算地质物理约束损失 - 支持地球动力学多场耦合适配"""
        if not self.physics_equations:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # 针对不同物理场分配权重
        for equation in self.physics_equations:
            # 计算物理方程的残差
            residual = equation(x, y, self.geological_config)
            
            # 根据方程类型分配权重
            equation_name = equation.__name__ if hasattr(equation, '__name__') else str(equation)
            
            if "stokes_equation" in equation_name:
                # 地幔流动权重更高（核心过程）
                weight = 100.0
            elif "mantle_convection_equation" in equation_name:
                # 地幔对流（综合方程）
                weight = 80.0
            elif "fault_slip_equation" in equation_name:
                # 断层过程权重
                weight = 50.0
            elif "plate_tectonics_equation" in equation_name:
                # 板块构造（综合方程）
                weight = 60.0
            elif "heat_conduction_equation" in equation_name:
                # 热传导次之
                weight = 10.0
            elif "elastic_equilibrium_equation" in equation_name:
                # 弹性力学
                weight = 20.0
            elif "chemical_transport_equation" in equation_name:
                # 化学输运
                weight = 15.0
            elif "darcy_equation" in equation_name:
                # 达西流动
                weight = 8.0
            else:
                # 其他方程默认权重
                weight = 1.0
            
            # 应用权重并累加到总损失
            weighted_loss = weight * torch.mean(residual ** 2)
            total_loss += weighted_loss
            
            # 记录各方程的损失（用于监控）
            if not hasattr(self, 'equation_losses'):
                self.equation_losses = {}
            self.equation_losses[equation_name] = weighted_loss.item()
        
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


def create_geological_ml_system() -> Dict:
    """创建地质ML系统"""
    system = {
        'pinn': GeologicalPINN,
        'surrogate': GeologicalSurrogateModel,
        'unet': GeologicalUNet,
        'bridge': GeologicalMultiScaleBridge,
        'hybrid': GeologicalHybridAccelerator,
        'adaptive': GeologicalAdaptiveSolver,  # 新增：地质自适应求解器
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
    
    print("\n✅ 地质数值模拟ML/DL融合演示完成!")


# ==================== 元学习功能 ====================

class GeodynamicMetaTask:
    """地球动力学元任务类"""
    
    def __init__(self, name: str, data_generator: Callable, 
                 geological_conditions: Dict[str, Any]):
        self.name = name
        self.data_generator = data_generator
        self.geological_conditions = geological_conditions
        self.task_data = None
        self.validation_data = None
    
    def generate_data(self, num_samples: int = 1000):
        """生成任务数据"""
        self.task_data = self.data_generator(num_samples)
        # 分割训练和验证数据
        split_idx = int(0.8 * num_samples)
        self.validation_data = (
            self.task_data[0][split_idx:], 
            self.task_data[1][split_idx:]
        )
        self.task_data = (
            self.task_data[0][:split_idx], 
            self.task_data[1][:split_idx]
        )
        return self.task_data
    
    def get_validation_data(self):
        """获取验证数据"""
        return self.validation_data


class GeodynamicMetaLearner:
    """地球动力学元学习器"""
    
    def __init__(self, pinn_model: 'GeologicalPINN', 
                 meta_learning_rate: float = 0.001,
                 inner_learning_rate: float = 0.005,
                 adaptation_steps: int = 3):
        self.pinn_model = pinn_model
        self.meta_learning_rate = meta_learning_rate
        self.inner_learning_rate = inner_learning_rate
        self.adaptation_steps = adaptation_steps
        
        # 元学习优化器
        self.meta_optimizer = optim.Adam(
            self.pinn_model.parameters(), 
            lr=meta_learning_rate
        )
        
        # 记录元学习过程
        self.meta_loss_history = []
        self.adaptation_history = []
        self.task_performance = {}
    
    def create_geodynamic_meta_tasks(self) -> List[GeodynamicMetaTask]:
        """创建地球动力学元任务集（不同构造场景）"""
        meta_tasks = []
        
        # 任务1：大洋中脊扩张（高温、低粘度）
        def generate_ridge_data(num_samples):
            """生成中脊区域的温度-速度样本"""
            X = torch.randn(num_samples, 4)  # 空间坐标 + 温度
            # 中脊特征：高温、低粘度、扩张速度
            X[:, 3] = 800 + 200 * torch.randn(num_samples)  # 高温
            y = torch.randn(num_samples, 3)  # 速度场 + 压力
            y[:, 0] = 0.1 + 0.05 * torch.randn(num_samples)  # 扩张速度
            return X, y
        
        ridge_task = GeodynamicMetaTask(
            "大洋中脊扩张",
            generate_ridge_data,
            {"temperature_range": (600, 1000), "viscosity": "low", "tectonic_type": "divergent"}
        )
        meta_tasks.append(ridge_task)
        
        # 任务2：俯冲带（高压、高粘度差异）
        def generate_subduction_data(num_samples):
            """生成俯冲带数据"""
            X = torch.randn(num_samples, 4)
            # 俯冲带特征：高压、高粘度差异、压缩应力
            X[:, 3] = 400 + 100 * torch.randn(num_samples)  # 中等温度
            y = torch.randn(num_samples, 3)
            y[:, 0] = -0.05 + 0.02 * torch.randn(num_samples)  # 压缩速度
            return X, y
        
        subduction_task = GeodynamicMetaTask(
            "俯冲带",
            generate_subduction_data,
            {"pressure_range": (1e8, 1e9), "viscosity": "high", "tectonic_type": "convergent"}
        )
        meta_tasks.append(subduction_task)
        
        # 任务3：大陆碰撞带（复杂弹性变形）
        def generate_collision_data(num_samples):
            """生成大陆碰撞带数据"""
            X = torch.randn(num_samples, 4)
            # 碰撞带特征：复杂变形、高弹性模量
            X[:, 3] = 300 + 150 * torch.randn(num_samples)  # 低温
            y = torch.randn(num_samples, 3)
            y[:, 0] = 0.02 + 0.01 * torch.randn(num_samples)  # 小变形
            return X, y
        
        collision_task = GeodynamicMetaTask(
            "大陆碰撞带",
            generate_collision_data,
            {"deformation_type": "complex", "elastic_modulus": "high", "tectonic_type": "collision"}
        )
        meta_tasks.append(collision_task)
        
        return meta_tasks
    
    def meta_train_geodynamics(self, meta_tasks: List[GeodynamicMetaTask], 
                               meta_epochs: int = 50, 
                               task_samples: int = 1000):
        """元学习训练适配 - 针对地球动力学任务调整"""
        print(f"🚀 开始地球动力学元学习训练...")
        print(f"   元任务数量: {len(meta_tasks)}")
        print(f"   元学习轮数: {meta_epochs}")
        print(f"   内循环步数: {self.adaptation_steps}")
        
        for meta_epoch in range(meta_epochs):
            meta_loss = 0.0
            epoch_adaptations = []
            
            for task_idx, task in enumerate(meta_tasks):
                # 生成任务数据
                X_task, y_task = task.generate_data(task_samples)
                X_val, y_val = task.get_validation_data()
                
                # 保存初始参数
                initial_params = {n: p.clone() for n, p in self.pinn_model.named_parameters()}
                
                # 内循环：适配特定构造场景（如俯冲带）
                task_losses = []
                for step in range(self.adaptation_steps):
                    outputs = self.pinn_model(X_task)
                    
                    # 重点惩罚物理残差（保证跨场景的物理一致性）
                    data_loss = F.mse_loss(outputs, y_task)
                    physics_loss = self.pinn_model.compute_physics_loss(X_task, outputs)
                    task_loss = 0.5 * data_loss + 0.5 * physics_loss
                    
                    task_losses.append(task_loss.item())
                    
                    # 内循环更新（仅微调上层参数，保留底层物理特征）
                    self.pinn_model.zero_grad()
                    task_loss.backward(retain_graph=True)
                    
                    with torch.no_grad():
                        for name, p in self.pinn_model.named_parameters():
                            if "output_layer" in name or "conv2" in name:
                                # 上层参数可微调
                                p -= self.inner_learning_rate * p.grad
                            # 底层参数保持不变，保留通用物理特征
                
                # 元损失：泛化到任务验证集
                val_outputs = self.pinn_model(X_val)
                val_loss = self.pinn_model.compute_physics_loss(X_val, val_outputs)
                meta_loss += val_loss
                
                # 记录任务性能
                self.task_performance[f"{task.name}_epoch_{meta_epoch}"] = {
                    "task_losses": task_losses,
                    "validation_loss": val_loss.item(),
                    "final_task_loss": task_losses[-1]
                }
                
                epoch_adaptations.append({
                    "task": task.name,
                    "task_losses": task_losses,
                    "validation_loss": val_loss.item()
                })
                
                # 恢复初始参数
                self.pinn_model.load_state_dict(initial_params)
            
            # 外循环更新：保留对所有构造场景通用的特征（如粘度-温度关系）
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            # 记录元学习过程
            self.meta_loss_history.append(meta_loss.item())
            self.adaptation_history.append(epoch_adaptations)
            
            if meta_epoch % 10 == 0:
                print(f"   元学习轮次 {meta_epoch}: 元损失 = {meta_loss.item():.6f}")
        
        print(f"✅ 地球动力学元学习训练完成!")
        print(f"   最终元损失: {self.meta_loss_history[-1]:.6f}")
        return self.meta_loss_history, self.adaptation_history
    
    def adapt_to_new_region(self, new_region_data: Tuple[torch.Tensor, torch.Tensor],
                           adaptation_steps: int = 5) -> Dict[str, Any]:
        """快速适配到新区域"""
        print(f"🔄 快速适配到新区域...")
        
        X_new, y_new = new_region_data
        initial_params = {n: p.clone() for n, p in self.pinn_model.named_parameters()}
        
        adaptation_losses = []
        
        for step in range(adaptation_steps):
            outputs = self.pinn_model(X_new)
            
            # 新区域损失：数据损失 + 物理损失
            data_loss = F.mse_loss(outputs, y_new)
            physics_loss = self.pinn_model.compute_physics_loss(X_new, outputs)
            total_loss = 0.5 * data_loss + 0.5 * physics_loss
            
            adaptation_losses.append(total_loss.item())
            
            # 快速适配：仅更新上层参数
            self.pinn_model.zero_grad()
            total_loss.backward(retain_graph=True)
            
            with torch.no_grad():
                for name, p in self.pinn_model.named_parameters():
                    if "output_layer" in name or "conv2" in name:
                        p -= self.inner_learning_rate * p.grad
        
        # 评估适配效果
        final_outputs = self.pinn_model(X_new)
        final_data_loss = F.mse_loss(final_outputs, y_new).item()
        final_physics_loss = self.pinn_model.compute_physics_loss(X_new, final_outputs).item()
        
        # 恢复元学习参数
        self.pinn_model.load_state_dict(initial_params)
        
        adaptation_result = {
            "adaptation_steps": adaptation_steps,
            "loss_history": adaptation_losses,
            "final_data_loss": final_data_loss,
            "final_physics_loss": final_physics_loss,
            "total_loss_reduction": adaptation_losses[0] - adaptation_losses[-1]
        }
        
        print(f"   适配完成，总损失减少: {adaptation_result['total_loss_reduction']:.6f}")
        return adaptation_result
    
    def get_meta_learning_status(self) -> Dict[str, Any]:
        """获取元学习状态"""
        return {
            "meta_loss_history": self.meta_loss_history,
            "adaptation_history": self.adaptation_history,
            "task_performance": self.task_performance,
            "meta_learning_rate": self.meta_learning_rate,
            "inner_learning_rate": self.inner_learning_rate,
            "adaptation_steps": self.adaptation_steps
        }


# 在GeologicalPINN类中添加元学习支持
def add_meta_learning_support_to_pinn():
    """为GeologicalPINN类添加元学习支持"""
    
    def meta_train_geodynamics(self, meta_tasks, meta_epochs=50, task_samples=1000):
        """元学习训练方法"""
        if not hasattr(self, 'meta_learner'):
            self.meta_learner = GeodynamicMetaLearner(self)
        return self.meta_learner.meta_train_geodynamics(meta_tasks, meta_epochs, task_samples)
    
    def adapt_to_new_region(self, new_region_data, adaptation_steps=5):
        """快速适配到新区域"""
        if not hasattr(self, 'meta_learner'):
            self.meta_learner = GeodynamicMetaLearner(self)
        return self.meta_learner.adapt_to_new_region(new_region_data, adaptation_steps)
    
    def get_meta_learning_status(self):
        """获取元学习状态"""
        if hasattr(self, 'meta_learner'):
            return self.meta_learner.get_meta_learning_status()
        return {"status": "Meta-learning not initialized"}
    
    # 动态添加方法到GeologicalPINN类
    GeologicalPINN.meta_train_geodynamics = meta_train_geodynamics
    GeologicalPINN.adapt_to_new_region = adapt_to_new_region
    GeologicalPINN.get_meta_learning_status = get_meta_learning_status


# 初始化时添加元学习支持
add_meta_learning_support_to_pinn()


# ==================== RL强化学习功能 ====================

class DQNAgent:
    """DQN智能体 - 用于时间步长优化"""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # 简单的神经网络Q函数
        if HAS_PYTORCH:
            self.q_network = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            )
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.epsilon = 0.1  # 探索率
        else:
            self.q_network = None
            self.optimizer = None
            self.epsilon = 0.1
    
    def choose_action(self, state: np.ndarray) -> int:
        """选择动作（ε-贪婪策略）"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        if self.q_network is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        else:
            return np.random.randint(0, self.action_dim)
    
    def learn(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """学习更新Q值"""
        if self.q_network is None:
            return
        
        # 计算目标Q值
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            next_q_values = self.q_network(next_state_tensor)
            target_q = reward + 0.99 * next_q_values.max()
        
        # 计算当前Q值
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        current_q_values = self.q_network(state_tensor)
        current_q = current_q_values[0, action]
        
        # 计算损失并更新
        loss = F.mse_loss(current_q, torch.tensor(target_q))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PPORLAgent:
    """PPO强化学习智能体 - 用于反演优化"""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.0003):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        if HAS_PYTORCH:
            # Actor网络（策略）
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Tanh()  # 输出范围[-1, 1]
            )
            
            # Critic网络（价值）
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
            
            # PPO参数
            self.clip_ratio = 0.2
            self.value_coef = 0.5
            self.entropy_coef = 0.01
        else:
            self.actor = None
            self.critic = None
            self.actor_optimizer = None
            self.critic_optimizer = None
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """选择动作"""
        if self.actor is None:
            return np.random.randn(self.action_dim) * 0.1
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_mean = self.actor(state_tensor)
            action_std = torch.ones_like(action_mean) * 0.1
            action_dist = torch.distributions.Normal(action_mean, action_std)
            action = action_dist.sample()
            return action.squeeze().numpy()
    
    def update(self, states: List[np.ndarray], actions: List[np.ndarray], 
               rewards: List[float], next_states: List[np.ndarray]):
        """更新策略和价值网络"""
        if self.actor is None or self.critic is None:
            return
        
        # 转换为张量
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.FloatTensor(np.array(actions))
        rewards_tensor = torch.FloatTensor(rewards)
        
        # 计算优势函数（简化版）
        advantages = rewards_tensor
        
        # 更新Critic
        values = self.critic(states_tensor).squeeze()
        value_loss = F.mse_loss(values, rewards_tensor)
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor（简化版PPO）
        action_mean = self.actor(states_tensor)
        action_std = torch.ones_like(action_mean) * 0.1
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = action_dist.log_prob(actions_tensor).sum(dim=1)
        policy_loss = -(log_probs * advantages).mean()
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()


class RLTimeStepOptimizer:
    """RL时间步长优化器 - 用于地幔对流等长时程模拟"""
    
    def __init__(self, solver, base_dt: float = 1e6):
        self.solver = solver
        self.base_dt = base_dt
        
        # 状态：当前速度场梯度、温度变化率、上一步误差
        self.agent = DQNAgent(state_dim=15, action_dim=4)  # 4种时间步缩放因子，状态维度为15（5步×3特征）
        
        # 状态历史
        self.state_history = []
        self.error_history = []
        self.dt_history = []
        
        # 优化参数
        self.min_dt_scale = 0.1
        self.max_dt_scale = 2.0
        self.target_error = 1e-6
    
    def _get_state_features(self, state_history: List[Dict]) -> np.ndarray:
        """提取当前状态特征"""
        if len(state_history) < 5:
            # 如果历史不足，用零填充
            padding = [{'velocity_grad': 0.0, 'temp_change': 0.0, 'error': 0.0}] * (5 - len(state_history))
            state_history = padding + state_history
        
        # 提取最近5步的特征
        features = []
        for state in state_history[-5:]:
            features.extend([
                state.get('velocity_grad', 0.0),
                state.get('temp_change', 0.0),
                state.get('error', 0.0)
            ])
        
        # 归一化特征
        features = np.array(features)
        if np.max(features) > 0:
            features = features / np.max(features)
        
        return features
    
    def _smoothness_reward(self, new_state: Dict) -> float:
        """计算物理场平滑性奖励"""
        # 基于速度场梯度和温度变化率的平滑性
        velocity_grad = new_state.get('velocity_grad', 0.0)
        temp_change = new_state.get('temp_change', 0.0)
        
        # 平滑性奖励：梯度变化越小越好
        smoothness = 1.0 / (1.0 + abs(velocity_grad) + abs(temp_change))
        return smoothness
    
    def optimize(self, state_history: List[Dict], max_steps: int = 1000) -> Dict[str, Any]:
        """优化时间步长"""
        print(f"🚀 开始RL时间步长优化...")
        print(f"   最大步数: {max_steps}")
        print(f"   基础时间步: {self.base_dt}")
        
        optimization_results = {
            'dt_history': [],
            'error_history': [],
            'reward_history': [],
            'efficiency_improvement': 0.0
        }
        
        for step in range(max_steps):
            # 提取当前状态特征
            current_state = self._get_state_features(state_history[-5:])
            
            # 选择时间步长（如1x, 1.5x, 0.5x, 2x）
            action = self.agent.choose_action(current_state)
            dt_scale = [0.5, 1.0, 1.5, 2.0][action]
            dt = self.base_dt * dt_scale
            
            # 限制时间步范围
            dt = np.clip(dt, self.base_dt * self.min_dt_scale, self.base_dt * self.max_dt_scale)
            
            # 执行模拟步（这里用模拟数据）
            new_state, error = self._simulate_step(dt, state_history[-1] if state_history else {})
            
            # 奖励设计：误差小（+）、步长大（+）、物理场平滑（+）
            error_penalty = 1.0 - min(error / self.target_error, 1.0)
            dt_reward = np.log(dt / self.base_dt)
            smoothness_reward = 0.1 * self._smoothness_reward(new_state)
            
            reward = error_penalty + dt_reward + smoothness_reward
            
            # 学习更新
            self.agent.learn(current_state, action, reward, self._get_state_features([new_state]))
            
            # 记录历史
            self.state_history.append(new_state)
            self.error_history.append(error)
            self.dt_history.append(dt)
            
            optimization_results['dt_history'].append(dt)
            optimization_results['error_history'].append(error)
            optimization_results['reward_history'].append(reward)
            
            # 更新状态历史
            state_history.append(new_state)
            
            if step % 100 == 0:
                print(f"   步数 {step}: dt={dt:.2e}, error={error:.2e}, reward={reward:.4f}")
        
        # 计算效率提升
        if len(optimization_results['dt_history']) > 1:
            avg_dt = np.mean(optimization_results['dt_history'])
            efficiency_improvement = (avg_dt - self.base_dt) / self.base_dt * 100
            optimization_results['efficiency_improvement'] = efficiency_improvement
        
        print(f"✅ RL时间步长优化完成!")
        print(f"   平均时间步: {np.mean(optimization_results['dt_history']):.2e}")
        print(f"   效率提升: {optimization_results['efficiency_improvement']:.1f}%")
        
        return optimization_results
    
    def _simulate_step(self, dt: float, prev_state: Dict) -> Tuple[Dict, float]:
        """模拟一步计算（实际应用中替换为真实求解器）"""
        # 模拟地幔对流状态
        velocity_grad = np.random.normal(0.1, 0.05) * (1 + np.random.random() * 0.1)
        temp_change = np.random.normal(0.05, 0.02) * (1 + np.random.random() * 0.1)
        
        # 误差与时间步相关
        error = np.random.normal(1e-6, 1e-7) * (dt / self.base_dt) ** 2
        
        new_state = {
            'velocity_grad': velocity_grad,
            'temp_change': temp_change,
            'error': error,
            'dt': dt
        }
        
        return new_state, error


class InversionRLAgent:
    """RL地球物理反演智能体 - 用于地下参数反演"""
    
    def __init__(self, forward_model, param_dim: int = 10):
        self.forward_model = forward_model  # PINN正演模型
        self.param_dim = param_dim
        
        # 动作：调整粘度参数的方向和幅度
        self.agent = PPORLAgent(state_dim=10, action_dim=5)  # 10个观测点残差，5种调整策略
        
        # 反演参数
        self.param_bounds = {
            'viscosity': (1e18, 1e24),  # Pa·s
            'density': (2000, 4000),     # kg/m³
            'thermal_conductivity': (1.0, 5.0)  # W/(m·K)
        }
        
        # 反演历史
        self.inversion_history = {
            'params': [],
            'residuals': [],
            'rewards': []
        }
    
    def _get_residual_features(self, obs_data: np.ndarray, pred_data: np.ndarray) -> np.ndarray:
        """获取观测残差特征"""
        residuals = obs_data - pred_data
        
        # 计算残差统计特征
        features = [
            np.mean(residuals),
            np.std(residuals),
            np.max(np.abs(residuals)),
            np.min(residuals),
            np.max(residuals),
            np.percentile(residuals, 25),
            np.percentile(residuals, 50),
            np.percentile(residuals, 75),
            np.sum(residuals > 0),  # 正残差数量
            np.sum(residuals < 0)   # 负残差数量
        ]
        
        # 归一化特征
        features = np.array(features)
        if np.max(np.abs(features)) > 0:
            features = features / np.max(np.abs(features))
        
        return features
    
    def _adjust_params(self, current_params: Dict[str, np.ndarray], 
                       action: np.ndarray) -> Dict[str, np.ndarray]:
        """根据RL动作调整参数"""
        new_params = current_params.copy()
        
        # 动作映射到参数调整
        # action[0]: 粘度调整幅度
        # action[1]: 密度调整幅度  
        # action[2]: 热导率调整幅度
        # action[3]: 空间平滑度
        # action[4]: 时间平滑度
        
        for param_name in current_params.keys():
            if param_name in self.param_bounds:
                param_array = current_params[param_name]
                bounds = self.param_bounds[param_name]
                
                # 计算调整幅度
                if param_name == 'viscosity':
                    adjustment = action[0] * 0.1  # 10%调整
                elif param_name == 'density':
                    adjustment = action[1] * 0.05  # 5%调整
                elif param_name == 'thermal_conductivity':
                    adjustment = action[2] * 0.1   # 10%调整
                else:
                    adjustment = 0.0
                
                # 应用调整
                new_param_array = param_array * (1 + adjustment)
                
                # 应用空间平滑（action[3]）
                if action[3] > 0:
                    # 简单的空间平滑
                    from scipy.ndimage import gaussian_filter
                    try:
                        new_param_array = gaussian_filter(new_param_array, sigma=action[3])
                    except:
                        pass  # 如果scipy不可用，跳过平滑
                
                # 应用时间平滑（action[4]）
                if action[4] > 0:
                    # 时间平滑：与历史值平均
                    if len(self.inversion_history['params']) > 0:
                        prev_param = self.inversion_history['params'][-1][param_name]
                        alpha = min(action[4], 0.5)  # 最大50%历史权重
                        new_param_array = (1 - alpha) * new_param_array + alpha * prev_param
                
                # 限制在边界内
                new_param_array = np.clip(new_param_array, bounds[0], bounds[1])
                new_params[param_name] = new_param_array
        
        return new_params
    
    def invert(self, obs_data: np.ndarray, init_params: Dict[str, np.ndarray], 
               iterations: int = 100) -> Dict[str, Any]:
        """执行反演优化"""
        print(f"🔄 开始RL地球物理反演...")
        print(f"   反演迭代数: {iterations}")
        print(f"   参数维度: {self.param_dim}")
        
        current_params = init_params.copy()
        best_params = init_params.copy()
        best_residual = float('inf')
        
        # 记录初始状态
        self.inversion_history['params'].append(init_params.copy())
        
        for iteration in range(iterations):
            # 正演模拟
            try:
                pred_data = self.forward_model(current_params)
            except:
                # 如果正演失败，使用模拟数据
                pred_data = obs_data + np.random.normal(0, 0.1 * np.std(obs_data), obs_data.shape)
            
            # 状态：观测残差特征
            state = self._get_residual_features(obs_data, pred_data)
            
            # 选择参数调整动作
            action = self.agent.select_action(state)
            new_params = self._adjust_params(current_params, action)
            
            # 计算新参数的正演结果
            try:
                new_pred_data = self.forward_model(new_params)
            except:
                new_pred_data = obs_data + np.random.normal(0, 0.1 * np.std(obs_data), obs_data.shape)
            
            # 奖励：残差减小+参数平滑（符合地质连续性）
            residual_reward = -np.mean(np.square(obs_data - new_pred_data))
            smooth_reward = -0.1 * np.var(list(new_params.values()))
            total_reward = residual_reward + smooth_reward
            
            # 限制奖励范围，防止梯度爆炸
            total_reward = np.clip(total_reward, -100.0, 100.0)
            
            # 更新智能体
            self.agent.update([state], [action], [total_reward], [state])
            
            # 更新参数
            current_params = new_params
            
            # 记录历史
            self.inversion_history['params'].append(current_params.copy())
            self.inversion_history['residuals'].append(residual_reward)
            self.inversion_history['rewards'].append(total_reward)
            
            # 更新最佳参数
            if residual_reward > best_residual:
                best_residual = residual_reward
                best_params = current_params.copy()
            
            if iteration % 20 == 0:
                print(f"   迭代 {iteration}: 残差={-residual_reward:.6f}, 奖励={total_reward:.4f}")
        
        print(f"✅ RL反演完成!")
        print(f"   最佳残差: {-best_residual:.6f}")
        print(f"   最终残差: {-self.inversion_history['residuals'][-1]:.6f}")
        
        return {
            'best_params': best_params,
            'final_params': current_params,
            'best_residual': -best_residual,
            'final_residual': -self.inversion_history['residuals'][-1],
            'inversion_history': self.inversion_history,
            'efficiency_improvement': self._calculate_efficiency_improvement()
        }
    
    def _calculate_efficiency_improvement(self) -> float:
        """计算反演效率提升"""
        if len(self.inversion_history['residuals']) < 2:
            return 0.0
        
        initial_residual = -self.inversion_history['residuals'][0]
        final_residual = -self.inversion_history['residuals'][-1]
        
        if initial_residual > 0:
            improvement = (initial_residual - final_residual) / initial_residual * 100
            return improvement
        return 0.0


# 在GeologicalPINN类中添加RL支持
def add_rl_support_to_pinn():
    """为GeologicalPINN类添加RL支持"""
    
    def setup_rl_time_step_optimizer(self, base_dt: float = 1e6):
        """设置RL时间步长优化器"""
        self.rl_time_optimizer = RLTimeStepOptimizer(self, base_dt)
        print(f"✅ 已设置RL时间步长优化器，基础时间步: {base_dt}")
    
    def setup_rl_inversion_agent(self, param_dim: int = 10):
        """设置RL反演智能体"""
        self.rl_inversion_agent = InversionRLAgent(self, param_dim)
        print(f"✅ 已设置RL反演智能体，参数维度: {param_dim}")
    
    def optimize_time_step_with_rl(self, state_history: List[Dict], max_steps: int = 1000):
        """使用RL优化时间步长"""
        if not hasattr(self, 'rl_time_optimizer'):
            self.setup_rl_time_step_optimizer()
        return self.rl_time_optimizer.optimize(state_history, max_steps)
    
    def invert_parameters_with_rl(self, obs_data: np.ndarray, init_params: Dict[str, np.ndarray], 
                                 iterations: int = 100):
        """使用RL进行参数反演"""
        if not hasattr(self, 'rl_inversion_agent'):
            self.setup_rl_inversion_agent()
        return self.rl_inversion_agent.invert(obs_data, init_params, iterations)
    
    # 动态添加方法到GeologicalPINN类
    GeologicalPINN.setup_rl_time_step_optimizer = setup_rl_time_step_optimizer
    GeologicalPINN.setup_rl_inversion_agent = setup_rl_inversion_agent
    GeologicalPINN.optimize_time_step_with_rl = optimize_time_step_with_rl
    GeologicalPINN.invert_parameters_with_rl = invert_parameters_with_rl


# 初始化时添加RL支持
add_rl_support_to_pinn()


if __name__ == "__main__":
    demo_geological_ml()
