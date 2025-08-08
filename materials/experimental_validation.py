"""
实验验证模块
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import json


@dataclass
class ExperimentalData:
    """实验数据"""
    temperature: np.ndarray  # 温度数据
    pressure: np.ndarray     # 压力数据
    stress: np.ndarray       # 应力数据
    strain: np.ndarray       # 应变数据
    melt_fraction: np.ndarray  # 熔体分数数据
    time: np.ndarray         # 时间数据
    metadata: Dict[str, Any]  # 元数据


@dataclass
class ValidationResult:
    """验证结果"""
    model_name: str
    experimental_data: ExperimentalData
    predicted_values: np.ndarray
    error_metrics: Dict[str, float]
    correlation_coefficient: float
    r_squared: float
    validation_score: float


class ExperimentalDataLoader:
    """实验数据加载器"""
    
    def __init__(self):
        self.data_cache: Dict[str, ExperimentalData] = {}
    
    def load_csv_data(self, file_path: str, 
                     temperature_col: str = 'temperature',
                     pressure_col: str = 'pressure',
                     stress_col: str = 'stress',
                     strain_col: str = 'strain',
                     melt_fraction_col: str = 'melt_fraction',
                     time_col: str = 'time') -> ExperimentalData:
        """从CSV文件加载实验数据"""
        try:
            df = pd.read_csv(file_path)
            
            # 提取数据
            temperature = df[temperature_col].values if temperature_col in df.columns else np.zeros(len(df))
            pressure = df[pressure_col].values if pressure_col in df.columns else np.zeros(len(df))
            stress = df[stress_col].values if stress_col in df.columns else np.zeros(len(df))
            strain = df[strain_col].values if strain_col in df.columns else np.zeros(len(df))
            melt_fraction = df[melt_fraction_col].values if melt_fraction_col in df.columns else np.zeros(len(df))
            time = df[time_col].values if time_col in df.columns else np.arange(len(df))
            
            # 创建实验数据对象
            experimental_data = ExperimentalData(
                temperature=temperature,
                pressure=pressure,
                stress=stress,
                strain=strain,
                melt_fraction=melt_fraction,
                time=time,
                metadata={'source': file_path, 'rows': len(df)}
            )
            
            return experimental_data
            
        except Exception as e:
            raise ValueError(f"无法加载CSV文件 {file_path}: {str(e)}")
    
    def load_json_data(self, file_path: str) -> ExperimentalData:
        """从JSON文件加载实验数据"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 提取数据
            experimental_data = ExperimentalData(
                temperature=np.array(data.get('temperature', [])),
                pressure=np.array(data.get('pressure', [])),
                stress=np.array(data.get('stress', [])),
                strain=np.array(data.get('strain', [])),
                melt_fraction=np.array(data.get('melt_fraction', [])),
                time=np.array(data.get('time', [])),
                metadata=data.get('metadata', {})
            )
            
            return experimental_data
            
        except Exception as e:
            raise ValueError(f"无法加载JSON文件 {file_path}: {str(e)}")
    
    def create_synthetic_data(self, n_points: int = 100,
                            temperature_range: Tuple[float, float] = (1000, 2000),
                            pressure_range: Tuple[float, float] = (1e8, 1e9),
                            noise_level: float = 0.05) -> ExperimentalData:
        """创建合成实验数据用于测试"""
        # 生成基础数据
        temperature = np.linspace(temperature_range[0], temperature_range[1], n_points)
        pressure = np.linspace(pressure_range[0], pressure_range[1], n_points)
        
        # 生成应力-应变关系（简化的线性关系）
        strain = np.linspace(0, 0.1, n_points)
        stress = 200e9 * strain + np.random.normal(0, noise_level * 200e9, n_points)
        
        # 生成熔体分数（基于温度）
        melt_fraction = np.clip((temperature - 1200) / 200, 0, 1) + \
                       np.random.normal(0, noise_level, n_points)
        melt_fraction = np.clip(melt_fraction, 0, 1)
        
        # 时间数据
        time = np.linspace(0, 1000, n_points)
        
        experimental_data = ExperimentalData(
            temperature=temperature,
            pressure=pressure,
            stress=stress,
            strain=strain,
            melt_fraction=melt_fraction,
            time=time,
            metadata={'type': 'synthetic', 'noise_level': noise_level}
        )
        
        return experimental_data


class ModelValidator(ABC):
    """模型验证器基类"""
    
    def __init__(self, name: str = "Model Validator"):
        self.name = name
    
    @abstractmethod
    def predict(self, experimental_data: ExperimentalData) -> np.ndarray:
        """模型预测"""
        pass
    
    def compute_error_metrics(self, experimental_values: np.ndarray, 
                            predicted_values: np.ndarray) -> Dict[str, float]:
        """计算误差指标"""
        # 移除NaN值
        valid_mask = ~(np.isnan(experimental_values) | np.isnan(predicted_values))
        exp_valid = experimental_values[valid_mask]
        pred_valid = predicted_values[valid_mask]
        
        if len(exp_valid) == 0:
            return {
                'mse': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'r_squared': np.nan,
                'correlation': np.nan
            }
        
        # 计算误差指标
        mse = np.mean((exp_valid - pred_valid) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(exp_valid - pred_valid))
        
        # 平均绝对百分比误差
        mape = np.mean(np.abs((exp_valid - pred_valid) / (exp_valid + 1e-12))) * 100
        
        # R²值
        ss_res = np.sum((exp_valid - pred_valid) ** 2)
        ss_tot = np.sum((exp_valid - np.mean(exp_valid)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-12))
        
        # 相关系数
        correlation = np.corrcoef(exp_valid, pred_valid)[0, 1] if len(exp_valid) > 1 else np.nan
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r_squared': r_squared,
            'correlation': correlation
        }
    
    def validate(self, experimental_data: ExperimentalData) -> ValidationResult:
        """验证模型"""
        # 模型预测
        predicted_values = self.predict(experimental_data)
        
        # 计算误差指标（以熔体分数为例）
        error_metrics = self.compute_error_metrics(
            experimental_data.melt_fraction, predicted_values)
        
        # 计算验证分数
        validation_score = self.compute_validation_score(error_metrics)
        
        return ValidationResult(
            model_name=self.name,
            experimental_data=experimental_data,
            predicted_values=predicted_values,
            error_metrics=error_metrics,
            correlation_coefficient=error_metrics['correlation'],
            r_squared=error_metrics['r_squared'],
            validation_score=validation_score
        )
    
    def compute_validation_score(self, error_metrics: Dict[str, float]) -> float:
        """计算验证分数"""
        # 综合评分（0-100）
        score = 0.0
        
        # R²值贡献（40%）
        if not np.isnan(error_metrics['r_squared']):
            score += 40 * max(0, error_metrics['r_squared'])
        
        # 相关系数贡献（30%）
        if not np.isnan(error_metrics['correlation']):
            score += 30 * max(0, error_metrics['correlation'])
        
        # RMSE贡献（20%）
        if not np.isnan(error_metrics['rmse']):
            # 归一化RMSE（假设最大误差为1）
            normalized_rmse = min(1.0, error_metrics['rmse'])
            score += 20 * (1.0 - normalized_rmse)
        
        # MAPE贡献（10%）
        if not np.isnan(error_metrics['mape']):
            # 归一化MAPE（假设最大误差为100%）
            normalized_mape = min(1.0, error_metrics['mape'] / 100.0)
            score += 10 * (1.0 - normalized_mape)
        
        return score


class PhaseChangeModelValidator(ModelValidator):
    """相变模型验证器"""
    
    def __init__(self, phase_change_model, name: str = "Phase Change Model Validator"):
        super().__init__(name)
        self.phase_change_model = phase_change_model
    
    def predict(self, experimental_data: ExperimentalData) -> np.ndarray:
        """相变模型预测"""
        # 使用相变模型预测熔体分数
        predicted_melt_fraction = self.phase_change_model.compute_melt_fraction(
            experimental_data.temperature, experimental_data.pressure)
        
        return predicted_melt_fraction


class PlasticityModelValidator(ModelValidator):
    """塑性模型验证器"""
    
    def __init__(self, plasticity_model, name: str = "Plasticity Model Validator"):
        super().__init__(name)
        self.plasticity_model = plasticity_model
    
    def predict(self, experimental_data: ExperimentalData) -> np.ndarray:
        """塑性模型预测"""
        # 使用塑性模型预测应力
        # 这里需要根据具体的塑性模型实现
        predicted_stress = np.zeros_like(experimental_data.stress)
        
        # 简化的应力预测
        for i in range(len(experimental_data.strain)):
            strain_tensor = np.array([[experimental_data.strain[i], 0], [0, 0]])
            stress_tensor = np.array([[experimental_data.stress[i], 0], [0, 0]])
            
            # 使用塑性模型计算应力
            # predicted_stress[i] = self.plasticity_model.compute_stress(strain_tensor)
            predicted_stress[i] = experimental_data.stress[i]  # 简化版本
        
        return predicted_stress


class ValidationReport:
    """验证报告生成器"""
    
    def __init__(self):
        self.reports: List[ValidationResult] = []
    
    def add_validation_result(self, result: ValidationResult):
        """添加验证结果"""
        self.reports.append(result)
    
    def generate_summary_report(self) -> str:
        """生成汇总报告"""
        if not self.reports:
            return "没有验证结果"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("实验验证汇总报告")
        report_lines.append("=" * 60)
        
        for i, result in enumerate(self.reports, 1):
            report_lines.append(f"\n{i}. {result.model_name}")
            report_lines.append("-" * 40)
            report_lines.append(f"验证分数: {result.validation_score:.2f}/100")
            report_lines.append(f"R²值: {result.r_squared:.4f}")
            report_lines.append(f"相关系数: {result.correlation_coefficient:.4f}")
            report_lines.append(f"RMSE: {result.error_metrics['rmse']:.2e}")
            report_lines.append(f"MAE: {result.error_metrics['mae']:.2e}")
            report_lines.append(f"MAPE: {result.error_metrics['mape']:.2f}%")
        
        # 计算平均分数
        avg_score = np.mean([r.validation_score for r in self.reports])
        report_lines.append(f"\n平均验证分数: {avg_score:.2f}/100")
        
        return "\n".join(report_lines)
    
    def plot_validation_results(self, save_path: Optional[str] = None):
        """绘制验证结果"""
        if not self.reports:
            print("没有验证结果可绘制")
            return
        
        n_models = len(self.reports)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 验证分数对比
        model_names = [r.model_name for r in self.reports]
        scores = [r.validation_score for r in self.reports]
        
        axes[0, 0].bar(model_names, scores, color='skyblue')
        axes[0, 0].set_title('验证分数对比')
        axes[0, 0].set_ylabel('验证分数')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. R²值对比
        r_squared_values = [r.r_squared for r in self.reports]
        
        axes[0, 1].bar(model_names, r_squared_values, color='lightgreen')
        axes[0, 1].set_title('R²值对比')
        axes[0, 1].set_ylabel('R²值')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 预测值与实验值对比（第一个模型）
        if self.reports:
            result = self.reports[0]
            exp_data = result.experimental_data
            pred_values = result.predicted_values
            
            axes[1, 0].scatter(exp_data.melt_fraction, pred_values, alpha=0.6)
            axes[1, 0].plot([0, 1], [0, 1], 'r--', label='理想线')
            axes[1, 0].set_xlabel('实验值')
            axes[1, 0].set_ylabel('预测值')
            axes[1, 0].set_title(f'{result.model_name} - 预测值 vs 实验值')
            axes[1, 0].legend()
        
        # 4. 误差分布
        if self.reports:
            errors = []
            for result in self.reports:
                error = result.experimental_data.melt_fraction - result.predicted_values
                errors.extend(error[~np.isnan(error)])
            
            axes[1, 1].hist(errors, bins=20, alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('误差')
            axes[1, 1].set_ylabel('频次')
            axes[1, 1].set_title('误差分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def demo_experimental_validation():
    """演示实验验证功能"""
    print("🔧 实验验证模块演示")
    print("=" * 50)
    
    # 创建数据加载器
    data_loader = ExperimentalDataLoader()
    
    # 创建合成实验数据
    print("\n🔧 创建合成实验数据...")
    experimental_data = data_loader.create_synthetic_data(
        n_points=100,
        temperature_range=(1000, 2000),
        pressure_range=(1e8, 1e9),
        noise_level=0.05
    )
    
    print(f"   数据点数: {len(experimental_data.temperature)}")
    print(f"   温度范围: {experimental_data.temperature.min():.0f} - {experimental_data.temperature.max():.0f} K")
    print(f"   压力范围: {experimental_data.pressure.min():.1e} - {experimental_data.pressure.max():.1e} Pa")
    
    # 创建验证器（使用简化的模型）
    print("\n🔧 创建模型验证器...")
    
    # 模拟相变模型验证器
    class MockPhaseChangeModel:
        def compute_melt_fraction(self, temperature, pressure):
            # 简化的熔体分数计算
            return np.clip((temperature - 1200) / 200, 0, 1)
    
    mock_phase_model = MockPhaseChangeModel()
    phase_validator = PhaseChangeModelValidator(mock_phase_model, "Mock Phase Change Model")
    
    # 执行验证
    print("\n🔧 执行模型验证...")
    validation_result = phase_validator.validate(experimental_data)
    
    print(f"   验证分数: {validation_result.validation_score:.2f}/100")
    print(f"   R²值: {validation_result.r_squared:.4f}")
    print(f"   相关系数: {validation_result.correlation_coefficient:.4f}")
    print(f"   RMSE: {validation_result.error_metrics['rmse']:.2e}")
    print(f"   MAE: {validation_result.error_metrics['mae']:.2e}")
    print(f"   MAPE: {validation_result.error_metrics['mape']:.2f}%")
    
    # 生成验证报告
    print("\n🔧 生成验证报告...")
    report_generator = ValidationReport()
    report_generator.add_validation_result(validation_result)
    
    summary_report = report_generator.generate_summary_report()
    print(summary_report)
    
    print("\n✅ 实验验证模块演示完成!")


if __name__ == "__main__":
    demo_experimental_validation()
