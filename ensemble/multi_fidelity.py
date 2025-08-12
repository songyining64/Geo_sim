"""
多保真度建模模块

实现"低保真度数据预训练 + 高保真度数据微调"的两阶段训练过程，
使用协同训练让低成本模型辅助高成本模型学习，减少大规模仿真的计算成本。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import warnings
import time
from dataclasses import dataclass, field
import json
from pathlib import Path

# 可选依赖
try:
    import torch.nn.functional as F
    HAS_TORCH_FUNCTIONAL = True
except ImportError:
    HAS_TORCH_FUNCTIONAL = False
    warnings.warn("torch.nn.functional not available. Some features will be limited.")

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available. Traditional ML models will be limited.")


@dataclass
class FidelityLevel:
    """保真度级别定义"""
    
    name: str
    level: int  # 0=最低保真度, 1,2,3...=递增保真度
    description: str
    computational_cost: float  # 相对计算成本
    accuracy: float  # 预期精度 (0-1)
    data_requirements: int  # 所需数据量
    training_time: float  # 预期训练时间 (秒)
    
    # 模型参数
    model_type: str  # 'neural_network', 'random_forest', 'gradient_boosting'
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # 训练参数
    training_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """设置默认值"""
        if not self.model_params:
            if self.model_type == 'neural_network':
                self.model_params = {
                    'hidden_layers': [64, 32],
                    'activation': 'relu',
                    'dropout': 0.1
                }
            elif self.model_type == 'random_forest':
                self.model_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                }
            elif self.model_type == 'gradient_boosting':
                self.model_params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
        
        if not self.training_params:
            self.training_params = {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'validation_split': 0.2
            }


@dataclass
class MultiFidelityConfig:
    """多保真度配置"""
    
    name: str
    description: str
    fidelity_levels: List[FidelityLevel]
    
    # 协同训练参数
    co_training:
        enabled: bool = True
        transfer_learning: bool = True
        knowledge_distillation: bool = True
        ensemble_method: str = 'weighted_average'  # 'weighted_average', 'stacking', 'boosting'
    
    # 训练策略
    training_strategy:
        stage1_epochs: int = 1000  # 低保真度预训练
        stage2_epochs: int = 500   # 高保真度微调
        transfer_epochs: int = 200 # 知识迁移
        distillation_epochs: int = 100  # 知识蒸馏
    
    # 性能优化
    performance:
        parallel_training: bool = True
        early_stopping: bool = True
        adaptive_learning_rate: bool = True
        memory_optimization: bool = True
    
    def get_fidelity_level(self, level: int) -> Optional[FidelityLevel]:
        """获取指定保真度级别"""
        for fidelity in self.fidelity_levels:
            if fidelity.level == level:
                return fidelity
        return None
    
    def get_lowest_fidelity(self) -> Optional[FidelityLevel]:
        """获取最低保真度级别"""
        if not self.fidelity_levels:
            return None
        return min(self.fidelity_levels, key=lambda x: x.level)
    
    def get_highest_fidelity(self) -> Optional[FidelityLevel]:
        """获取最高保真度级别"""
        if not self.fidelity_levels:
            return None
        return max(self.fidelity_levels, key=lambda x: x.level)


class BaseFidelityModel(ABC):
    """保真度模型基类"""
    
    def __init__(self, fidelity_level: FidelityLevel):
        self.fidelity_level = fidelity_level
        self.model = None
        self.is_trained = False
        self.training_history = []
        self.validation_metrics = {}
        
    @abstractmethod
    def build_model(self, input_dim: int, output_dim: int):
        """构建模型"""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """模型预测"""
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型"""
        pass
    
    def save_model(self, filepath: str):
        """保存模型"""
        if self.model is None:
            raise RuntimeError("模型未训练，无法保存")
        
        model_data = {
            'fidelity_level': self.fidelity_level.name,
            'model_type': self.fidelity_level.model_type,
            'model_params': self.fidelity_level.model_params,
            'training_history': self.training_history,
            'validation_metrics': self.validation_metrics,
            'is_trained': self.is_trained
        }
        
        # 保存模型权重
        if hasattr(self.model, 'state_dict'):
            model_data['model_weights'] = self.model.state_dict()
        
        # 保存到文件
        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)
        else:
            torch.save(model_data, filepath)
        
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        else:
            model_data = torch.load(filepath, map_location='cpu')
        
        # 恢复模型状态
        self.training_history = model_data.get('training_history', [])
        self.validation_metrics = model_data.get('validation_metrics', {})
        self.is_trained = model_data.get('is_trained', False)
        
        # 恢复模型权重
        if 'model_weights' in model_data and hasattr(self.model, 'load_state_dict'):
            self.model.load_state_dict(model_data['model_weights'])
        
        print(f"模型已从 {filepath} 加载")


class NeuralNetworkFidelityModel(BaseFidelityModel):
    """神经网络保真度模型"""
    
    def __init__(self, fidelity_level: FidelityLevel):
        super().__init__(fidelity_level)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_model(self, input_dim: int, output_dim: int):
        """构建神经网络模型"""
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in self.fidelity_level.model_params['hidden_layers']:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.fidelity_level.model_params.get('dropout', 0) > 0:
                layers.append(nn.Dropout(self.fidelity_level.model_params['dropout']))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers).to(self.device)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """训练神经网络模型"""
        if self.model is None:
            raise RuntimeError("模型未构建，请先调用build_model")
        
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # 优化器
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.fidelity_level.training_params.get('learning_rate', 0.001)
        )
        
        # 损失函数
        criterion = nn.MSELoss()
        
        # 训练参数
        epochs = kwargs.get('epochs', self.fidelity_level.training_params['epochs'])
        batch_size = self.fidelity_level.training_params.get('batch_size', 32)
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            # 批次训练
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # 记录训练历史
            epoch_info = {
                'epoch': epoch,
                'train_loss': total_loss / num_batches if num_batches > 0 else 0.0,
                'timestamp': time.time()
            }
            
            # 验证
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    epoch_info['val_loss'] = val_loss
                self.model.train()
            
            self.training_history.append(epoch_info)
            
            # 早停检查
            if (self.fidelity_level.performance.early_stopping and 
                len(self.training_history) > 10):
                recent_val_losses = [h['val_loss'] for h in self.training_history[-10:] 
                                   if 'val_loss' in h]
                if len(recent_val_losses) >= 5:
                    if all(recent_val_losses[i] <= recent_val_losses[i-1] 
                           for i in range(1, len(recent_val_losses))):
                        print(f"早停触发，在第 {epoch} 轮停止训练")
                        break
        
        self.is_trained = True
        
        # 计算最终验证指标
        if X_val is not None and y_val is not None:
            self.validation_metrics = self.evaluate(X_val, y_val)
        
        return {
            'total_epochs': len(self.training_history),
            'final_train_loss': self.training_history[-1]['train_loss'],
            'final_val_loss': self.training_history[-1].get('val_loss', 0.0),
            'validation_metrics': self.validation_metrics
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """模型预测"""
        if not self.is_trained:
            raise RuntimeError("模型未训练，无法预测")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型"""
        if not self.is_trained:
            return {}
        
        predictions = self.predict(X)
        
        # 计算评估指标
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        # 计算相对误差
        relative_error = np.mean(np.abs(predictions - y) / (np.abs(y) + 1e-8))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'relative_error': relative_error
        }


class TraditionalMLFidelityModel(BaseFidelityModel):
    """传统机器学习保真度模型"""
    
    def __init__(self, fidelity_level: FidelityLevel):
        super().__init__(fidelity_level)
        
        if not HAS_SKLEARN:
            raise ImportError("需要scikit-learn来使用传统ML模型")
    
    def build_model(self, input_dim: int, output_dim: int):
        """构建传统ML模型"""
        if self.fidelity_level.model_type == 'random_forest':
            self.model = RandomForestRegressor(**self.fidelity_level.model_params)
        elif self.fidelity_level.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**self.fidelity_level.model_params)
        else:
            raise ValueError(f"不支持的模型类型: {self.fidelity_level.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """训练传统ML模型"""
        if self.model is None:
            raise RuntimeError("模型未构建，请先调用build_model")
        
        start_time = time.time()
        
        # 训练模型
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # 记录训练历史
        self.training_history.append({
            'training_time': training_time,
            'timestamp': time.time()
        })
        
        # 计算验证指标
        if X_val is not None and y_val is not None:
            self.validation_metrics = self.evaluate(X_val, y_val)
        
        self.is_trained = True
        
        return {
            'training_time': training_time,
            'validation_metrics': self.validation_metrics
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """模型预测"""
        if not self.is_trained:
            raise RuntimeError("模型未训练，无法预测")
        
        predictions = self.model.predict(X)
        
        # 确保输出是2D数组
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型"""
        if not self.is_trained:
            return {}
        
        predictions = self.predict(X)
        
        # 确保y是2D数组
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # 计算评估指标
        metrics = {}
        for i in range(y.shape[1]):
            mse = mean_squared_error(y[:, i], predictions[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y[:, i], predictions[:, i])
            relative_error = np.mean(np.abs(predictions[:, i] - y[:, i]) / (np.abs(y[:, i]) + 1e-8))
            
            metrics[f'mse_output_{i}'] = mse
            metrics[f'rmse_output_{i}'] = rmse
            metrics[f'r2_output_{i}'] = r2
            metrics[f'relative_error_output_{i}'] = relative_error
        
        # 总体指标
        metrics['overall_mse'] = np.mean([metrics[f'mse_output_{i}'] for i in range(y.shape[1])])
        metrics['overall_r2'] = np.mean([metrics[f'r2_output_{i}'] for i in range(y.shape[1])])
        
        return metrics


class MultiFidelityTrainer:
    """多保真度训练器"""
    
    def __init__(self, config: MultiFidelityConfig):
        self.config = config
        self.models: Dict[int, BaseFidelityModel] = {}
        self.training_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.validation_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        
        # 训练状态
        self.current_stage = 1
        self.training_progress = {}
        self.ensemble_predictions = {}
        
    def add_training_data(self, fidelity_level: int, X: np.ndarray, y: np.ndarray):
        """添加训练数据"""
        self.training_data[fidelity_level] = (X, y)
        print(f"✅ 添加保真度级别 {fidelity_level} 的训练数据: {X.shape}")
    
    def add_validation_data(self, fidelity_level: int, X: np.ndarray, y: np.ndarray):
        """添加验证数据"""
        self.validation_data[fidelity_level] = (X, y)
        print(f"✅ 添加保真度级别 {fidelity_level} 的验证数据: {X.shape}")
    
    def create_models(self, input_dim: int, output_dim: int):
        """创建所有保真度级别的模型"""
        for fidelity in self.config.fidelity_levels:
            if fidelity.model_type == 'neural_network':
                model = NeuralNetworkFidelityModel(fidelity)
            elif fidelity.model_type in ['random_forest', 'gradient_boosting']:
                model = TraditionalMLFidelityModel(fidelity)
            else:
                warnings.warn(f"不支持的模型类型: {fidelity.model_type}")
                continue
            
            model.build_model(input_dim, output_dim)
            self.models[fidelity.level] = model
            
            print(f"✅ 创建保真度级别 {fidelity.level} 的模型: {fidelity.model_type}")
    
    def stage1_low_fidelity_pretraining(self) -> Dict[str, Any]:
        """阶段1：低保真度预训练"""
        print("\n=== 阶段1：低保真度预训练 ===")
        
        # 获取最低保真度级别
        lowest_fidelity = self.config.get_lowest_fidelity()
        if not lowest_fidelity:
            raise ValueError("未找到最低保真度级别")
        
        if lowest_fidelity.level not in self.training_data:
            raise ValueError(f"缺少保真度级别 {lowest_fidelity.level} 的训练数据")
        
        X_train, y_train = self.training_data[lowest_fidelity.level]
        X_val, y_val = self.validation_data.get(lowest_fidelity.level, (None, None))
        
        # 训练模型
        model = self.models[lowest_fidelity.level]
        training_result = model.train(
            X_train, y_train, X_val, y_val,
            epochs=self.config.training_strategy.stage1_epochs
        )
        
        self.training_progress['stage1'] = training_result
        print(f"✅ 低保真度预训练完成: {training_result}")
        
        return training_result
    
    def stage2_high_fidelity_finetuning(self) -> Dict[str, Any]:
        """阶段2：高保真度微调"""
        print("\n=== 阶段2：高保真度微调 ===")
        
        # 获取最高保真度级别
        highest_fidelity = self.config.get_highest_fidelity()
        if not highest_fidelity:
            raise ValueError("未找到最高保真度级别")
        
        if highest_fidelity.level not in self.training_data:
            raise ValueError(f"缺少保真度级别 {highest_fidelity.level} 的训练数据")
        
        X_train, y_train = self.training_data[highest_fidelity.level]
        X_val, y_val = self.validation_data.get(highest_fidelity.level, (None, None))
        
        # 知识迁移：从低保真度模型迁移权重
        if self.config.co_training.transfer_learning:
            self._transfer_knowledge(highest_fidelity.level)
        
        # 训练模型
        model = self.models[highest_fidelity.level]
        training_result = model.train(
            X_train, y_train, X_val, y_val,
            epochs=self.config.training_strategy.stage2_epochs
        )
        
        self.training_progress['stage2'] = training_result
        print(f"✅ 高保真度微调完成: {training_result}")
        
        return training_result
    
    def _transfer_knowledge(self, target_fidelity_level: int):
        """知识迁移：从低保真度模型迁移到高保真度模型"""
        if not self.config.co_training.transfer_learning:
            return
        
        # 找到最低保真度级别
        lowest_fidelity = self.config.get_lowest_fidelity()
        if not lowest_fidelity or lowest_fidelity.level not in self.models:
            return
        
        source_model = self.models[lowest_fidelity.level]
        target_model = self.models[target_fidelity_level]
        
        # 检查是否都是神经网络模型
        if (isinstance(source_model, NeuralNetworkFidelityModel) and 
            isinstance(target_model, NeuralNetworkFidelityModel)):
            
            print(f"🔄 从保真度级别 {lowest_fidelity.level} 迁移知识到级别 {target_fidelity_level}")
            
            # 迁移权重（如果架构兼容）
            try:
                if (hasattr(source_model.model, 'state_dict') and 
                    hasattr(target_model.model, 'state_dict')):
                    
                    source_state = source_model.model.state_dict()
                    target_state = target_model.model.state_dict()
                    
                    # 迁移兼容的层权重
                    transferred_count = 0
                    for key in source_state:
                        if key in target_state and source_state[key].shape == target_state[key].shape:
                            target_state[key] = source_state[key]
                            transferred_count += 1
                    
                    target_model.model.load_state_dict(target_state)
                    print(f"✅ 成功迁移 {transferred_count} 层权重")
                    
            except Exception as e:
                warnings.warn(f"知识迁移失败: {str(e)}")
    
    def co_training(self) -> Dict[str, Any]:
        """协同训练：让低保真度模型辅助高保真度模型学习"""
        if not self.config.co_training.enabled:
            return {}
        
        print("\n=== 协同训练 ===")
        
        # 知识蒸馏
        if self.config.co_training.knowledge_distillation:
            self._knowledge_distillation()
        
        # 集成预测
        ensemble_result = self._ensemble_prediction()
        
        self.training_progress['co_training'] = ensemble_result
        print(f"✅ 协同训练完成: {ensemble_result}")
        
        return ensemble_result
    
    def _knowledge_distillation(self):
        """知识蒸馏：从高保真度模型向低保真度模型传递知识"""
        if not self.config.co_training.knowledge_distillation:
            return
        
        print("🔄 执行知识蒸馏...")
        
        # 获取所有保真度级别
        fidelity_levels = sorted(self.config.fidelity_levels, key=lambda x: x.level)
        
        for i in range(len(fidelity_levels) - 1):
            source_level = fidelity_levels[i + 1]  # 高保真度
            target_level = fidelity_levels[i]      # 低保真度
            
            if (source_level.level in self.models and 
                target_level.level in self.models and
                source_level.level in self.training_data):
                
                source_model = self.models[source_level.level]
                target_model = self.models[target_level.level]
                
                # 使用高保真度模型的预测作为软标签
                X_train, _ = self.training_data[source_level.level]
                soft_labels = source_model.predict(X_train)
                
                # 在低保真度数据上微调
                if target_level.level in self.training_data:
                    X_target, y_target = self.training_data[target_level.level]
                    
                    # 混合硬标签和软标签
                    alpha = 0.7  # 软标签权重
                    mixed_labels = alpha * soft_labels[:len(y_target)] + (1 - alpha) * y_target
                    
                    # 微调模型
                    target_model.train(
                        X_target, mixed_labels,
                        epochs=self.config.training_strategy.distillation_epochs
                    )
                    
                    print(f"✅ 完成从级别 {source_level.level} 到级别 {target_level.level} 的知识蒸馏")
    
    def _ensemble_prediction(self) -> Dict[str, Any]:
        """集成预测：结合多个保真度级别的预测"""
        print("🔄 执行集成预测...")
        
        if not self.models:
            return {}
        
        # 获取验证数据
        highest_fidelity = self.config.get_highest_fidelity()
        if not highest_fidelity or highest_fidelity.level not in self.validation_data:
            return {}
        
        X_val, y_val = self.validation_data[highest_fidelity.level]
        
        # 收集所有模型的预测
        predictions = {}
        for level, model in self.models.items():
            if model.is_trained:
                try:
                    pred = model.predict(X_val)
                    predictions[level] = pred
                except Exception as e:
                    warnings.warn(f"保真度级别 {level} 预测失败: {str(e)}")
        
        if not predictions:
            return {}
        
        # 集成预测
        if self.config.co_training.ensemble_method == 'weighted_average':
            ensemble_pred = self._weighted_average_ensemble(predictions)
        elif self.config.co_training.ensemble_method == 'stacking':
            ensemble_pred = self._stacking_ensemble(predictions, y_val)
        else:
            ensemble_pred = self._simple_average_ensemble(predictions)
        
        # 评估集成性能
        ensemble_metrics = self._evaluate_ensemble(ensemble_pred, y_val)
        
        # 保存集成预测结果
        self.ensemble_predictions = {
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'ensemble_metrics': ensemble_metrics
        }
        
        return ensemble_metrics
    
    def _weighted_average_ensemble(self, predictions: Dict[int, np.ndarray]) -> np.ndarray:
        """加权平均集成"""
        # 基于保真度级别分配权重
        total_weight = 0
        weighted_sum = None
        
        for level, pred in predictions.items():
            fidelity = self.config.get_fidelity_level(level)
            if fidelity:
                weight = fidelity.accuracy  # 使用精度作为权重
                total_weight += weight
                
                if weighted_sum is None:
                    weighted_sum = weight * pred
                else:
                    weighted_sum += weight * pred
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return np.mean(list(predictions.values()), axis=0)
    
    def _stacking_ensemble(self, predictions: Dict[int, np.ndarray], y_true: np.ndarray) -> np.ndarray:
        """堆叠集成"""
        # 使用线性回归作为元学习器
        if not HAS_SKLEARN:
            return self._simple_average_ensemble(predictions)
        
        try:
            from sklearn.linear_model import LinearRegression
            
            # 准备元特征
            meta_features = np.column_stack(list(predictions.values()))
            
            # 训练元学习器
            meta_learner = LinearRegression()
            meta_learner.fit(meta_features, y_true)
            
            # 预测
            ensemble_pred = meta_learner.predict(meta_features)
            
            return ensemble_pred
            
        except Exception as e:
            warnings.warn(f"堆叠集成失败: {str(e)}")
            return self._simple_average_ensemble(predictions)
    
    def _simple_average_ensemble(self, predictions: Dict[int, np.ndarray]) -> np.ndarray:
        """简单平均集成"""
        return np.mean(list(predictions.values()), axis=0)
    
    def _evaluate_ensemble(self, ensemble_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """评估集成性能"""
        if not HAS_SKLEARN:
            return {}
        
        try:
            mse = mean_squared_error(y_true, ensemble_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, ensemble_pred)
            relative_error = np.mean(np.abs(ensemble_pred - y_true) / (np.abs(y_true) + 1e-8))
            
            return {
                'ensemble_mse': mse,
                'ensemble_rmse': rmse,
                'ensemble_r2': r2,
                'ensemble_relative_error': relative_error
            }
        except Exception as e:
            warnings.warn(f"集成评估失败: {str(e)}")
            return {}
    
    def run_full_training(self, input_dim: int, output_dim: int) -> Dict[str, Any]:
        """运行完整的多保真度训练流程"""
        print("🚀 开始多保真度训练流程")
        
        # 1. 创建模型
        self.create_models(input_dim, output_dim)
        
        # 2. 阶段1：低保真度预训练
        stage1_result = self.stage1_low_fidelity_pretraining()
        
        # 3. 阶段2：高保真度微调
        stage2_result = self.stage2_high_fidelity_finetuning()
        
        # 4. 协同训练
        co_training_result = self.co_training()
        
        # 5. 最终评估
        final_evaluation = self._final_evaluation()
        
        # 汇总结果
        training_summary = {
            'stage1': stage1_result,
            'stage2': stage2_result,
            'co_training': co_training_result,
            'final_evaluation': final_evaluation,
            'total_training_time': sum([
                stage1_result.get('training_time', 0),
                stage2_result.get('training_time', 0)
            ]),
            'model_performance': self._get_model_performance_summary()
        }
        
        print("\n🎉 多保真度训练完成！")
        print(f"训练摘要: {training_summary}")
        
        return training_summary
    
    def _final_evaluation(self) -> Dict[str, Any]:
        """最终评估"""
        print("\n=== 最终评估 ===")
        
        evaluation_results = {}
        
        # 评估每个模型
        for level, model in self.models.items():
            if model.is_trained and level in self.validation_data:
                X_val, y_val = self.validation_data[level]
                metrics = model.evaluate(X_val, y_val)
                evaluation_results[f'level_{level}'] = metrics
                
                print(f"保真度级别 {level} 性能: {metrics}")
        
        # 集成模型性能
        if self.ensemble_predictions:
            evaluation_results['ensemble'] = self.ensemble_predictions['ensemble_metrics']
            print(f"集成模型性能: {self.ensemble_predictions['ensemble_metrics']}")
        
        return evaluation_results
    
    def _get_model_performance_summary(self) -> Dict[str, Any]:
        """获取模型性能摘要"""
        summary = {}
        
        for level, model in self.models.items():
            if model.is_trained:
                summary[f'level_{level}'] = {
                    'model_type': model.fidelity_level.model_type,
                    'training_completed': model.is_trained,
                    'validation_metrics': model.validation_metrics,
                    'training_history_length': len(model.training_history)
                }
        
        return summary
    
    def predict_with_ensemble(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """使用集成模型进行预测"""
        if not self.models:
            raise RuntimeError("没有可用的模型")
        
        predictions = {}
        
        # 获取各个模型的预测
        for level, model in self.models.items():
            if model.is_trained:
                try:
                    pred = model.predict(X)
                    predictions[f'level_{level}'] = pred
                except Exception as e:
                    warnings.warn(f"保真度级别 {level} 预测失败: {str(e)}")
        
        # 集成预测
        if len(predictions) > 1:
            if self.config.co_training.ensemble_method == 'weighted_average':
                ensemble_pred = self._weighted_average_ensemble(predictions)
            elif self.config.co_training.ensemble_method == 'stacking':
                # 对于新数据，使用简单平均
                ensemble_pred = self._simple_average_ensemble(predictions)
            else:
                ensemble_pred = self._simple_average_ensemble(predictions)
            
            predictions['ensemble'] = ensemble_pred
        
        return predictions


def create_multi_fidelity_system(config: MultiFidelityConfig) -> MultiFidelityTrainer:
    """创建多保真度系统"""
    return MultiFidelityTrainer(config)


def demo_multi_fidelity():
    """演示多保真度建模功能"""
    print("=== 多保真度建模演示 ===")
    
    # 1. 创建配置
    try:
        # 定义保真度级别
        low_fidelity = FidelityLevel(
            name="低精度快速仿真",
            level=0,
            description="使用简化PDE的快速仿真",
            computational_cost=1.0,
            accuracy=0.7,
            data_requirements=1000,
            training_time=60.0,
            model_type="neural_network",
            model_params={
                'hidden_layers': [32, 16],
                'activation': 'relu',
                'dropout': 0.1
            },
            training_params={
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        )
        
        high_fidelity = FidelityLevel(
            name="高精度完整仿真",
            level=1,
            description="使用完整物理模型的精确仿真",
            computational_cost=10.0,
            accuracy=0.95,
            data_requirements=5000,
            training_time=300.0,
            model_type="neural_network",
            model_params={
                'hidden_layers': [128, 64, 32],
                'activation': 'relu',
                'dropout': 0.2
            },
            training_params={
                'epochs': 200,
                'batch_size': 16,
                'learning_rate': 0.0005
            }
        )
        
        config = MultiFidelityConfig(
            name="油藏模拟多保真度系统",
            description="结合快速和精确仿真的油藏预测系统",
            fidelity_levels=[low_fidelity, high_fidelity],
            co_training=MultiFidelityConfig.co_training(
                enabled=True,
                transfer_learning=True,
                knowledge_distillation=True,
                ensemble_method='weighted_average'
            ),
            training_strategy=MultiFidelityConfig.training_strategy(
                stage1_epochs=100,
                stage2_epochs=50,
                transfer_epochs=20,
                distillation_epochs=10
            )
        )
        
        print("✅ 创建多保真度配置")
        
    except Exception as e:
        print(f"❌ 创建配置失败: {e}")
        return
    
    # 2. 创建系统
    try:
        trainer = create_multi_fidelity_system(config)
        print("✅ 创建多保真度训练器")
        
    except Exception as e:
        print(f"❌ 创建训练器失败: {e}")
        return
    
    # 3. 生成模拟数据
    try:
        np.random.seed(42)
        
        # 低保真度数据
        X_low = np.random.randn(1000, 5)  # 5个输入特征
        y_low = np.random.randn(1000, 2)  # 2个输出目标
        
        # 高保真度数据
        X_high = np.random.randn(5000, 5)
        y_high = np.random.randn(5000, 2)
        
        # 添加验证数据
        X_val_low = np.random.randn(200, 5)
        y_val_low = np.random.randn(200, 2)
        
        X_val_high = np.random.randn(1000, 5)
        y_val_high = np.random.randn(1000, 2)
        
        # 添加到训练器
        trainer.add_training_data(0, X_low, y_low)
        trainer.add_training_data(1, X_high, y_high)
        trainer.add_validation_data(0, X_val_low, y_val_low)
        trainer.add_validation_data(1, X_val_high, y_val_high)
        
        print("✅ 添加训练和验证数据")
        
    except Exception as e:
        print(f"❌ 添加数据失败: {e}")
        return
    
    # 4. 运行训练
    try:
        training_summary = trainer.run_full_training(input_dim=5, output_dim=2)
        print("✅ 训练完成")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return
    
    # 5. 测试预测
    try:
        X_test = np.random.randn(100, 5)
        predictions = trainer.predict_with_ensemble(X_test)
        
        print(f"✅ 集成预测完成，输出形状: {predictions['ensemble'].shape}")
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
    
    print("\n🎉 多保真度建模演示完成！")


if __name__ == "__main__":
    demo_multi_fidelity()
