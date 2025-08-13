"""
统一API接口模块

为所有核心模块（有限元、多物理场耦合、ML模型）提供统一的接口，
包括setup()、run()、visualize()等方法，减少用户学习成本。
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import yaml
from pathlib import Path

# 可选依赖
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Visualization features will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("plotly not available. Interactive visualization features will be limited.")


@dataclass
class SimulationConfig:
    """统一仿真配置类"""
    
    # 基本信息
    name: str = "simulation"
    description: str = ""
    version: str = "1.0.0"
    
    # 物理参数
    physics_params: Dict[str, Any] = field(default_factory=dict)
    
    # 数值参数
    numerical_params: Dict[str, Any] = field(default_factory=dict)
    
    # 边界条件
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # 输出设置
    output_params: Dict[str, Any] = field(default_factory=dict)
    
    # 可视化设置
    visualization_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """设置默认值"""
        if not self.numerical_params:
            self.numerical_params = {
                'time_steps': 100,
                'dt': 0.01,
                'tolerance': 1e-6,
                'max_iterations': 1000
            }
        
        if not self.output_params:
            self.output_params = {
                'save_frequency': 10,
                'output_dir': './output',
                'save_format': 'h5'
            }
        
        if not self.visualization_params:
            self.visualization_params = {
                'plot_frequency': 5,
                'show_plots': True,
                'save_plots': True,
                'plot_style': 'default'
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'physics_params': self.physics_params,
            'numerical_params': self.numerical_params,
            'boundary_conditions': self.boundary_conditions,
            'output_params': self.output_params,
            'visualization_params': self.visualization_params
        }
    
    def to_yaml(self, filepath: str):
        """保存为YAML文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def to_json(self, filepath: str):
        """保存为JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'SimulationConfig':
        """从YAML文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'SimulationConfig':
        """从JSON文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class SimulationResult:
    """仿真结果类"""
    
    # 基本信息
    config: SimulationConfig
    start_time: float
    end_time: float
    
    # 结果数据
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 性能指标
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 收敛信息
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    
    # 错误信息
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """仿真持续时间"""
        return self.end_time - self.start_time
    
    @property
    def success(self) -> bool:
        """仿真是否成功"""
        return len(self.errors) == 0
    
    def add_data(self, key: str, value: Any):
        """添加结果数据"""
        self.data[key] = value
    
    def add_error(self, error: str):
        """添加错误信息"""
        self.errors.append(error)
    
    def add_performance_metric(self, key: str, value: Any):
        """添加性能指标"""
        self.performance_metrics[key] = value
    
    def add_convergence_info(self, key: str, value: Any):
        """添加收敛信息"""
        self.convergence_info[key] = value


class BaseSimulator(ABC):
    """统一仿真器基类"""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.result: Optional[SimulationResult] = None
        self.is_initialized = False
        self.is_running = False
        
        # 状态跟踪
        self._current_step = 0
        self._current_time = 0.0
        
        # 回调函数
        self._callbacks: Dict[str, List[Callable]] = {
            'step_completed': [],
            'convergence_check': [],
            'visualization_update': []
        }
    
    @abstractmethod
    def setup(self, **kwargs) -> bool:
        """
        设置仿真环境
        
        Returns:
            bool: 设置是否成功
        """
        pass
    
    @abstractmethod
    def run(self, **kwargs) -> SimulationResult:
        """
        运行仿真
        
        Returns:
            SimulationResult: 仿真结果
        """
        pass
    
    @abstractmethod
    def visualize(self, **kwargs):
        """
        可视化结果
        """
        pass
    
    def initialize(self, **kwargs) -> bool:
        """
        初始化仿真器
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            success = self.setup(**kwargs)
            if success:
                self.is_initialized = True
                self._on_initialized()
            return success
        except Exception as e:
            self._handle_error(f"初始化失败: {str(e)}")
            return False
    
    def execute(self, **kwargs) -> SimulationResult:
        """
        执行完整的仿真流程
        
        Returns:
            SimulationResult: 仿真结果
        """
        if not self.is_initialized:
            if not self.initialize(**kwargs):
                raise RuntimeError("仿真器初始化失败")
        
        try:
            self.is_running = True
            self._on_start()
            
            result = self.run(**kwargs)
            self.result = result
            
            self._on_complete()
            return result
            
        except Exception as e:
            self._handle_error(f"仿真执行失败: {str(e)}")
            raise
        finally:
            self.is_running = False
    
    def step(self, **kwargs) -> bool:
        """
        执行单个仿真步骤
        
        Returns:
            bool: 步骤是否成功
        """
        if not self.is_initialized:
            raise RuntimeError("仿真器未初始化")
        
        try:
            success = self._execute_step(**kwargs)
            if success:
                self._current_step += 1
                self._on_step_completed()
            return success
        except Exception as e:
            self._handle_error(f"步骤执行失败: {str(e)}")
            return False
    
    def reset(self):
        """重置仿真器状态"""
        self._current_step = 0
        self._current_time = 0.0
        self.is_initialized = False
        self.is_running = False
        self.result = None
        self._on_reset()
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'current_step': self._current_step,
            'current_time': self._current_time,
            'has_result': self.result is not None,
            'success': self.result.success if self.result else None
        }
    
    def add_callback(self, event: str, callback: Callable):
        """添加回调函数"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            warnings.warn(f"未知事件类型: {event}")
    
    def remove_callback(self, event: str, callback: Callable):
        """移除回调函数"""
        if event in self._callbacks:
            try:
                self._callbacks[event].remove(callback)
            except ValueError:
                pass
    
    def _execute_step(self, **kwargs) -> bool:
        """执行单个步骤的抽象方法"""
        raise NotImplementedError("子类必须实现_execute_step方法")
    
    def _on_initialized(self):
        """初始化完成回调"""
        pass
    
    def _on_start(self):
        """仿真开始回调"""
        pass
    
    def _on_step_completed(self):
        """步骤完成回调"""
        for callback in self._callbacks['step_completed']:
            try:
                callback(self)
            except Exception as e:
                warnings.warn(f"回调函数执行失败: {str(e)}")
    
    def _on_complete(self):
        """仿真完成回调"""
        pass
    
    def _on_reset(self):
        """重置回调"""
        pass
    
    def _handle_error(self, error_msg: str):
        """错误处理"""
        if self.result:
            self.result.add_error(error_msg)
        warnings.warn(error_msg)
    
    def save_result(self, filepath: str, format: str = 'auto'):
        """保存仿真结果"""
        if not self.result:
            raise RuntimeError("没有可保存的结果")
        
        if format == 'auto':
            format = Path(filepath).suffix[1:] or 'json'
        
        if format.lower() in ['json', 'js']:
            self.result.config.to_json(filepath)
        elif format.lower() in ['yaml', 'yml']:
            self.result.config.to_yaml(filepath)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        print(f"结果已保存到: {filepath}")


class FiniteElementSimulator(BaseSimulator):
    """有限元仿真器"""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        super().__init__(config)
        self.mesh = None
        self.basis_functions = None
        self.assembly = None
        
    def setup(self, **kwargs) -> bool:
        """设置有限元仿真环境"""
        try:
            # 设置网格
            self.mesh = kwargs.get('mesh')
            if not self.mesh:
                raise ValueError("必须提供网格")
            
            # 设置基函数
            self.basis_functions = kwargs.get('basis_functions')
            
            # 设置组装器
            self.assembly = kwargs.get('assembly')
            
            return True
        except Exception as e:
            self._handle_error(f"有限元设置失败: {str(e)}")
            return False
    
    def run(self, **kwargs) -> SimulationResult:
        """运行有限元仿真"""
        start_time = time.time()
        
        # 创建结果对象
        result = SimulationResult(
            config=self.config,
            start_time=start_time,
            end_time=start_time
        )
        
        try:
            # 执行仿真步骤
            while self._current_step < self.config.numerical_params['time_steps']:
                if not self.step(**kwargs):
                    break
            
            result.end_time = time.time()
            return result
            
        except Exception as e:
            result.add_error(str(e))
            result.end_time = time.time()
            return result
    
    def visualize(self, **kwargs):
        """可视化有限元结果"""
        if not self.result or not self.result.success:
            print("没有可可视化的结果")
            return
        
        # 实现有限元特定的可视化
        pass
    
    def _execute_step(self, **kwargs) -> bool:
        """执行有限元步骤"""
        # 实现有限元计算步骤
        return True


class MultiPhysicsSimulator(BaseSimulator):
    """多物理场耦合仿真器"""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        super().__init__(config)
        self.couplers = []
        self.physics_simulators = {}
        
    def setup(self, **kwargs) -> bool:
        """设置多物理场仿真环境"""
        try:
            # 设置物理场仿真器
            self.physics_simulators = kwargs.get('physics_simulators', {})
            
            # 设置耦合器
            self.couplers = kwargs.get('couplers', [])
            
            return True
        except Exception as e:
            self._handle_error(f"多物理场设置失败: {str(e)}")
            return False
    
    def run(self, **kwargs) -> SimulationResult:
        """运行多物理场仿真"""
        start_time = time.time()
        
        result = SimulationResult(
            config=self.config,
            start_time=start_time,
            end_time=start_time
        )
        
        try:
            # 执行耦合仿真
            while self._current_step < self.config.numerical_params['time_steps']:
                if not self.step(**kwargs):
                    break
            
            result.end_time = time.time()
            return result
            
        except Exception as e:
            result.add_error(str(e))
            result.end_time = time.time()
            return result
    
    def visualize(self, **kwargs):
        """可视化多物理场结果"""
        if not self.result or not self.result.success:
            print("没有可可视化的结果")
            return
        
        # 实现多物理场特定的可视化
        pass
    
    def _execute_step(self, **kwargs) -> bool:
        """执行多物理场步骤"""
        # 实现耦合计算步骤
        return True


class MLSimulator(BaseSimulator):
    """机器学习仿真器"""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        super().__init__(config)
        self.model = None
        self.trainer = None
        self.data_loader = None
        
    def setup(self, **kwargs) -> bool:
        """设置ML仿真环境"""
        try:
            # 设置模型
            self.model = kwargs.get('model')
            if not self.model:
                raise ValueError("必须提供模型")
            
            # 设置训练器
            self.trainer = kwargs.get('trainer')
            
            # 设置数据加载器
            self.data_loader = kwargs.get('data_loader')
            
            return True
        except Exception as e:
            self._handle_error(f"ML设置失败: {str(e)}")
            return False
    
    def run(self, **kwargs) -> SimulationResult:
        """运行ML仿真"""
        start_time = time.time()
        
        result = SimulationResult(
            config=self.config,
            start_time=start_time,
            end_time=start_time
        )
        
        try:
            # 执行训练或推理
            if kwargs.get('mode') == 'training':
                self._train_model(**kwargs)
            else:
                self._inference(**kwargs)
            
            result.end_time = time.time()
            return result
            
        except Exception as e:
            result.add_error(str(e))
            result.end_time = time.time()
            return result
    
    def visualize(self, **kwargs):
        """可视化ML结果"""
        if not self.result or not self.result.success:
            print("没有可可视化的结果")
            return
        
        # 实现ML特定的可视化
        pass
    
    def _execute_step(self, **kwargs) -> bool:
        """执行ML步骤"""
        # 实现ML计算步骤
        return True
    
    def _train_model(self, **kwargs):
        """训练模型"""
        if not self.trainer:
            raise RuntimeError("训练器未设置")
        
        # 执行训练
        pass
    
    def _inference(self, **kwargs):
        """模型推理"""
        if not self.model:
            raise RuntimeError("模型未设置")
        
        # 执行推理
        pass


def create_simulator(simulator_type: str, config: Optional[SimulationConfig] = None) -> BaseSimulator:
    """创建仿真器工厂函数"""
    
    simulators = {
        'finite_element': FiniteElementSimulator,
        'multi_physics': MultiPhysicsSimulator,
        'ml': MLSimulator
    }
    
    if simulator_type not in simulators:
        raise ValueError(f"不支持的仿真器类型: {simulator_type}")
    
    return simulators[simulator_type](config)


def load_config_from_template(template_name: str) -> SimulationConfig:
    """从模板加载配置"""
    template_path = Path(__file__).parent / 'configs' / f'{template_name}.yaml'
    
    if not template_path.exists():
        raise FileNotFoundError(f"模板文件不存在: {template_path}")
    
    return SimulationConfig.from_yaml(str(template_path))

