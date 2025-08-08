"""
地质数值模拟ML/DL应用示例
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import matplotlib.pyplot as plt

# 导入地质ML框架
try:
    from geological_ml_framework import (
        GeologicalPINN, GeologicalSurrogateModel, GeologicalUNet,
        GeologicalMultiScaleBridge, GeologicalHybridAccelerator,
        GeologicalConfig, GeologicalPhysicsEquations
    )
    HAS_GEOLOGICAL_ML = True
except ImportError:
    HAS_GEOLOGICAL_ML = False
    warnings.warn("地质ML框架不可用，示例功能将受限")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    warnings.warn("PyTorch不可用，示例功能将受限")

try:
    import sklearn
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn不可用，示例功能将受限")


class ReservoirSimulationExample:
    """油藏模拟示例 - 用PINN求解达西方程"""
    
    def __init__(self):
        self.config = GeologicalConfig(
            porosity=0.2,
            permeability=1e-12,  # m²
            viscosity=1e-3,      # Pa·s
            density=1000.0,      # kg/m³
            compressibility=1e-9  # Pa⁻¹
        )
        
        # 油藏参数
        self.reservoir_size = (1000, 1000, 100)  # m
        self.grid_size = (50, 50, 10)  # 网格数
        self.well_positions = [(250, 250), (750, 750)]  # 井位
        self.production_rate = 100  # m³/day
        
    def generate_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """生成训练数据"""
        # 生成空间坐标和时间
        x_coords = np.random.uniform(0, self.reservoir_size[0], n_samples)
        y_coords = np.random.uniform(0, self.reservoir_size[1], n_samples)
        z_coords = np.random.uniform(0, self.reservoir_size[2], n_samples)
        t_coords = np.random.uniform(0, 365, n_samples)  # 一年时间
        
        # 组合输入
        X = np.column_stack([x_coords, y_coords, z_coords, t_coords])
        
        # 生成压力场（简化模型）
        # 压力 = 初始压力 - 生产引起的压力降
        initial_pressure = 20e6  # Pa
        pressure_drop = self._calculate_pressure_drop(x_coords, y_coords, z_coords, t_coords)
        y = initial_pressure - pressure_drop
        
        return X, y.reshape(-1, 1)
    
    def _calculate_pressure_drop(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.ndarray) -> np.ndarray:
        """计算压力降（简化模型）"""
        pressure_drop = np.zeros_like(x)
        
        for well_x, well_y in self.well_positions:
            # 距离井的距离
            distance = np.sqrt((x - well_x)**2 + (y - well_y)**2)
            # 压力降与距离和时间的关系（简化）
            pressure_drop += self.production_rate * t / (2 * np.pi * self.config.permeability * distance + 1e-6)
        
        return pressure_drop
    
    def run_pinn_simulation(self) -> Dict:
        """运行PINN油藏模拟"""
        print("🔄 开始PINN油藏模拟...")
        
        # 生成训练数据
        X, y = self.generate_training_data(n_samples=2000)
        
        # 创建PINN模型
        pinn = GeologicalPINN(
            input_dim=4,  # x, y, z, t
            hidden_dims=[128, 64, 32],
            output_dim=1,  # 压力场
            geological_config=self.config
        )
        
        # 定义达西方程
        def darcy_equation(x, y, config):
            return GeologicalPhysicsEquations.darcy_equation(x, y, config)
        
        # 训练模型
        start_time = time.time()
        result = pinn.train(X, y, epochs=500, physics_weight=1.0)
        training_time = time.time() - start_time
        
        print(f"✅ PINN训练完成，耗时: {training_time:.2f} 秒")
        print(f"   最终损失: {result['total_loss'][-1]:.6f}")
        
        return {
            'model': pinn,
            'training_result': result,
            'training_time': training_time,
            'config': self.config
        }
    
    def visualize_results(self, pinn_model: GeologicalPINN):
        """可视化结果"""
        if not HAS_PYTORCH:
            print("PyTorch不可用，跳过可视化")
            return
        
        # 生成测试网格
        x_grid = np.linspace(0, self.reservoir_size[0], 50)
        y_grid = np.linspace(0, self.reservoir_size[1], 50)
        z_mid = self.reservoir_size[2] / 2
        t_test = 180  # 半年后
        
        X_test = []
        for x in x_grid:
            for y in y_grid:
                X_test.append([x, y, z_mid, t_test])
        
        X_test = np.array(X_test)
        
        # 预测压力场
        pressure_field = pinn_model.predict(X_test)
        pressure_field = pressure_field.reshape(50, 50)
        
        # 绘制压力场
        plt.figure(figsize=(10, 8))
        plt.imshow(pressure_field, extent=[0, self.reservoir_size[0], 0, self.reservoir_size[1]], 
                   origin='lower', cmap='viridis')
        plt.colorbar(label='压力 (Pa)')
        plt.title('PINN预测的油藏压力场')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        
        # 标记井位
        for well_x, well_y in self.well_positions:
            plt.plot(well_x, well_y, 'ro', markersize=10, label='生产井')
        
        plt.legend()
        plt.show()


class SeismicInversionExample:
    """地震反演示例 - 用UNet从地震数据反演地质结构"""
    
    def __init__(self):
        self.seismic_size = (256, 256)  # 地震数据大小
        self.geological_size = (256, 256)  # 地质属性大小
        
    def generate_synthetic_data(self, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """生成合成地震和地质数据"""
        seismic_data = []
        geological_data = []
        
        for i in range(n_samples):
            # 生成随机地质结构（孔隙度场）
            porosity_field = self._generate_porosity_field()
            
            # 生成对应的地震数据（简化模型）
            seismic_field = self._generate_seismic_data(porosity_field)
            
            seismic_data.append(seismic_field)
            geological_data.append(porosity_field)
        
        return np.array(seismic_data), np.array(geological_data)
    
    def _generate_porosity_field(self) -> np.ndarray:
        """生成孔隙度场"""
        # 使用随机场生成孔隙度分布
        field = np.random.randn(*self.geological_size)
        
        # 应用高斯滤波平滑
        from scipy.ndimage import gaussian_filter
        field = gaussian_filter(field, sigma=5)
        
        # 归一化到0.1-0.3范围
        field = 0.1 + 0.2 * (field - field.min()) / (field.max() - field.min())
        
        return field
    
    def _generate_seismic_data(self, porosity_field: np.ndarray) -> np.ndarray:
        """从孔隙度场生成地震数据"""
        # 简化模型：地震振幅与孔隙度相关
        seismic_field = porosity_field * 0.5 + np.random.normal(0, 0.1, porosity_field.shape)
        
        # 添加噪声
        noise = np.random.normal(0, 0.05, seismic_field.shape)
        seismic_field += noise
        
        return seismic_field
    
    def run_unet_inversion(self) -> Dict:
        """运行UNet地震反演"""
        print("🔄 开始UNet地震反演...")
        
        # 生成训练数据
        seismic_data, geological_data = self.generate_synthetic_data(n_samples=200)
        
        # 创建UNet模型
        unet = GeologicalUNet(
            input_channels=1,
            output_channels=1,
            initial_features=64,
            depth=4
        )
        
        # 训练模型
        start_time = time.time()
        result = unet.train_model(seismic_data, geological_data, epochs=100, batch_size=8)
        training_time = time.time() - start_time
        
        print(f"✅ UNet训练完成，耗时: {training_time:.2f} 秒")
        print(f"   最终损失: {result['loss'][-1]:.6f}")
        
        return {
            'model': unet,
            'training_result': result,
            'training_time': training_time,
            'seismic_data': seismic_data,
            'geological_data': geological_data
        }
    
    def visualize_inversion_results(self, unet_model: GeologicalUNet, 
                                  seismic_data: np.ndarray, geological_data: np.ndarray):
        """可视化反演结果"""
        if not HAS_PYTORCH:
            print("PyTorch不可用，跳过可视化")
            return
        
        # 选择测试样本
        test_idx = 0
        test_seismic = seismic_data[test_idx:test_idx+1]
        test_geological = geological_data[test_idx:test_idx+1]
        
        # 预测地质属性
        predicted_geological = unet_model.predict(test_seismic)
        
        # 绘制结果
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 地震数据
        axes[0].imshow(test_seismic[0, 0], cmap='gray')
        axes[0].set_title('地震数据')
        axes[0].axis('off')
        
        # 真实地质属性
        axes[1].imshow(test_geological[0, 0], cmap='viridis')
        axes[1].set_title('真实孔隙度场')
        axes[1].axis('off')
        
        # 预测地质属性
        axes[2].imshow(predicted_geological[0, 0], cmap='viridis')
        axes[2].set_title('预测孔隙度场')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


class MultiScaleModelingExample:
    """多尺度建模示例 - 用桥接模型连接微观和宏观尺度"""
    
    def __init__(self):
        self.micro_scale_size = 1000  # 微观样本数
        self.macro_scale_size = 100   # 宏观样本数
        
    def generate_micro_macro_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成微观和宏观数据"""
        # 微观参数（孔隙结构参数）
        micro_params = np.random.randn(self.micro_scale_size, 5)
        micro_params[:, 0] = np.random.uniform(0.1, 10, self.micro_scale_size)  # 平均孔隙半径 (μm)
        micro_params[:, 1] = np.random.uniform(0.1, 1, self.micro_scale_size)   # 孔隙度
        micro_params[:, 2] = np.random.uniform(0.1, 5, self.micro_scale_size)   # 喉道密度
        micro_params[:, 3] = np.random.uniform(0.1, 2, self.micro_scale_size)   # 形状因子
        micro_params[:, 4] = np.random.uniform(0.1, 1, self.micro_scale_size)   # 连通性
        
        # 宏观参数（等效渗透率）
        macro_params = self._calculate_macro_permeability(micro_params)
        
        return micro_params, macro_params
    
    def _calculate_macro_permeability(self, micro_params: np.ndarray) -> np.ndarray:
        """计算宏观等效渗透率（简化模型）"""
        # 基于Kozeny-Carman方程
        porosity = micro_params[:, 1]
        avg_radius = micro_params[:, 0]
        shape_factor = micro_params[:, 3]
        
        # Kozeny-Carman方程：k = φ³ / (c * S² * (1-φ)²)
        # 其中c是Kozeny常数，S是比表面积
        kozeny_constant = 5.0
        specific_surface = 3 / avg_radius  # 假设球形孔隙
        
        permeability = (porosity**3) / (kozeny_constant * (specific_surface**2) * ((1-porosity)**2))
        
        # 添加随机噪声
        permeability *= np.random.uniform(0.8, 1.2, len(permeability))
        
        return permeability.reshape(-1, 1)
    
    def run_multiscale_bridge(self) -> Dict:
        """运行多尺度桥接"""
        print("🔄 开始多尺度桥接建模...")
        
        # 生成数据
        micro_data, macro_data = self.generate_micro_macro_data()
        
        # 创建桥接模型
        bridge = GeologicalMultiScaleBridge()
        bridge.setup_bridge_model(
            input_dim=5,  # 微观参数维度
            output_dim=1,  # 宏观渗透率
            model_type='neural_network'
        )
        
        # 训练桥接模型
        start_time = time.time()
        result = bridge.train_bridge(micro_data, macro_data, epochs=200)
        training_time = time.time() - start_time
        
        print(f"✅ 多尺度桥接训练完成，耗时: {training_time:.2f} 秒")
        
        return {
            'bridge': bridge,
            'training_result': result,
            'training_time': training_time,
            'micro_data': micro_data,
            'macro_data': macro_data
        }
    
    def visualize_bridge_results(self, bridge: GeologicalMultiScaleBridge, 
                               micro_data: np.ndarray, macro_data: np.ndarray):
        """可视化桥接结果"""
        # 预测宏观参数
        predicted_macro = bridge.predict_coarse_from_fine(micro_data)
        
        # 绘制散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(macro_data.flatten(), predicted_macro.flatten(), alpha=0.6)
        plt.plot([macro_data.min(), macro_data.max()], 
                [macro_data.min(), macro_data.max()], 'r--', label='理想预测')
        plt.xlabel('真实宏观渗透率')
        plt.ylabel('预测宏观渗透率')
        plt.title('多尺度桥接预测结果')
        plt.legend()
        plt.grid(True)
        plt.show()


class ParameterInversionExample:
    """参数反演示例 - 用代理模型加速参数优化"""
    
    def __init__(self):
        self.parameter_bounds = {
            'porosity': (0.1, 0.3),
            'permeability': (1e-15, 1e-12),
            'viscosity': (1e-4, 1e-2),
            'compressibility': (1e-10, 1e-8)
        }
        
    def generate_training_data(self, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """生成参数-产量训练数据"""
        parameters = []
        production_rates = []
        
        for i in range(n_samples):
            # 随机生成参数
            param = {
                'porosity': np.random.uniform(*self.parameter_bounds['porosity']),
                'permeability': np.random.uniform(*self.parameter_bounds['permeability']),
                'viscosity': np.random.uniform(*self.parameter_bounds['viscosity']),
                'compressibility': np.random.uniform(*self.parameter_bounds['compressibility'])
            }
            
            # 计算产量（简化模型）
            production_rate = self._calculate_production_rate(param)
            
            parameters.append([param['porosity'], param['permeability'], 
                             param['viscosity'], param['compressibility']])
            production_rates.append(production_rate)
        
        return np.array(parameters), np.array(production_rates)
    
    def _calculate_production_rate(self, params: Dict) -> float:
        """计算产量（简化模型）"""
        # 基于达西定律的简化产量模型
        k = params['permeability']
        mu = params['viscosity']
        phi = params['porosity']
        c = params['compressibility']
        
        # 产量与渗透率成正比，与粘度成反比
        production_rate = k / mu * phi * (1 - c * 1e6)
        
        # 添加随机噪声
        production_rate *= np.random.uniform(0.9, 1.1)
        
        return production_rate
    
    def run_parameter_inversion(self) -> Dict:
        """运行参数反演"""
        print("🔄 开始参数反演...")
        
        # 生成训练数据
        X, y = self.generate_training_data(n_samples=1000)
        
        # 创建代理模型
        surrogate = GeologicalSurrogateModel('gaussian_process')
        
        # 训练模型
        start_time = time.time()
        result = surrogate.train(X, y.flatten())
        training_time = time.time() - start_time
        
        print(f"✅ 代理模型训练完成，耗时: {training_time:.2f} 秒")
        
        # 测试参数反演
        target_production = 1000  # 目标产量
        optimal_params = self._invert_parameters(surrogate, target_production)
        
        return {
            'surrogate': surrogate,
            'training_result': result,
            'training_time': training_time,
            'optimal_params': optimal_params,
            'target_production': target_production
        }
    
    def _invert_parameters(self, surrogate: GeologicalSurrogateModel, 
                          target_production: float) -> Dict:
        """反演参数"""
        # 使用网格搜索找到最优参数
        best_params = None
        best_error = float('inf')
        
        # 在参数空间中采样
        n_samples = 1000
        param_samples = []
        
        for i in range(n_samples):
            param = [
                np.random.uniform(*self.parameter_bounds['porosity']),
                np.random.uniform(*self.parameter_bounds['permeability']),
                np.random.uniform(*self.parameter_bounds['viscosity']),
                np.random.uniform(*self.parameter_bounds['compressibility'])
            ]
            param_samples.append(param)
        
        param_samples = np.array(param_samples)
        
        # 预测产量
        predicted_productions = surrogate.predict(param_samples)
        
        # 找到最接近目标产量的参数
        errors = np.abs(predicted_productions - target_production)
        best_idx = np.argmin(errors)
        
        best_params = {
            'porosity': param_samples[best_idx, 0],
            'permeability': param_samples[best_idx, 1],
            'viscosity': param_samples[best_idx, 2],
            'compressibility': param_samples[best_idx, 3]
        }
        
        return best_params
    
    def visualize_inversion_results(self, surrogate: GeologicalSurrogateModel, 
                                  optimal_params: Dict, target_production: float):
        """可视化反演结果"""
        # 生成测试数据
        X_test, y_test = self.generate_training_data(n_samples=100)
        
        # 预测
        y_pred, y_std = surrogate.predict(X_test, return_std=True)
        
        # 绘制结果
        plt.figure(figsize=(12, 8))
        
        # 预测vs真实
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('真实产量')
        plt.ylabel('预测产量')
        plt.title('代理模型预测性能')
        
        # 预测不确定性
        plt.subplot(2, 2, 2)
        plt.scatter(y_pred, y_std, alpha=0.6)
        plt.xlabel('预测产量')
        plt.ylabel('预测标准差')
        plt.title('预测不确定性')
        
        # 参数分布
        plt.subplot(2, 2, 3)
        param_names = list(optimal_params.keys())
        param_values = list(optimal_params.values())
        plt.bar(param_names, param_values)
        plt.title('最优参数')
        plt.xticks(rotation=45)
        
        # 目标产量
        plt.subplot(2, 2, 4)
        plt.axhline(y=target_production, color='r', linestyle='--', label='目标产量')
        plt.hist(y_pred, bins=30, alpha=0.7, label='预测产量分布')
        plt.xlabel('产量')
        plt.ylabel('频次')
        plt.title('产量分布')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


def run_all_examples():
    """运行所有示例"""
    print("🤖 地质数值模拟ML/DL应用示例")
    print("=" * 60)
    
    # 1. 油藏模拟示例
    print("\n🔧 1. 油藏模拟示例（PINN）")
    try:
        reservoir_example = ReservoirSimulationExample()
        result = reservoir_example.run_pinn_simulation()
        print(f"   ✅ 完成，训练时间: {result['training_time']:.2f} 秒")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
    
    # 2. 地震反演示例
    print("\n🔧 2. 地震反演示例（UNet）")
    try:
        seismic_example = SeismicInversionExample()
        result = seismic_example.run_unet_inversion()
        print(f"   ✅ 完成，训练时间: {result['training_time']:.2f} 秒")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
    
    # 3. 多尺度建模示例
    print("\n🔧 3. 多尺度建模示例（桥接模型）")
    try:
        multiscale_example = MultiScaleModelingExample()
        result = multiscale_example.run_multiscale_bridge()
        print(f"   ✅ 完成，训练时间: {result['training_time']:.2f} 秒")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
    
    # 4. 参数反演示例
    print("\n🔧 4. 参数反演示例（代理模型）")
    try:
        inversion_example = ParameterInversionExample()
        result = inversion_example.run_parameter_inversion()
        print(f"   ✅ 完成，训练时间: {result['training_time']:.2f} 秒")
        print(f"   最优参数: {result['optimal_params']}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
    
    print("\n✅ 所有示例运行完成!")


if __name__ == "__main__":
    run_all_examples()
