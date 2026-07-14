"""
增强调试与可视化工具

提供实时监控功能、错误诊断工具和详细的性能分析，
帮助用户快速定位模型问题和优化仿真性能。
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import json
import traceback
from pathlib import Path

# 可选依赖
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider, Button, CheckButtons
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Visualization features will be limited.", ImportWarning)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("plotly not available. Interactive visualization features will be limited.", ImportWarning)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


@dataclass
class DebugConfig:
    """调试配置类"""
    
    monitoring_enabled: bool = True
    monitoring_update_interval: float = 1.0
    monitoring_save_history: bool = True
    monitoring_max_history_size: int = 1000
    
    vis_realtime_plots: bool = True
    vis_save_plots: bool = True
    vis_plot_format: str = "png"
    vis_dpi: int = 300
    
    error_diagnosis_enabled: bool = True
    error_detailed_traceback: bool = True
    error_save_error_logs: bool = True
    error_log_dir: str = "./error_logs"
    
    perf_enabled: bool = True
    perf_profile_functions: bool = True
    perf_memory_tracking: bool = True
    perf_timing_breakdown: bool = True


@dataclass
class PhysicalConstraint:
    """物理约束类"""
    
    name: str
    equation: Callable
    weight: float = 1.0
    tolerance: float = 1e-6
    description: str = ""
    
    def compute_residual(self, *args, **kwargs) -> float:
        """计算约束残差"""
        try:
            residual = self.equation(*args, **kwargs)
            return float(residual)
        except Exception as e:
            warnings.warn(f"计算约束 {self.name} 残差失败: {str(e)}")
            return np.inf


@dataclass
class ConvergenceMetrics:
    """收敛指标类"""
    
    iteration: int
    timestamp: float
    residual_norm: float
    relative_change: float
    constraint_violations: Dict[str, float]
    performance_metrics: Dict[str, float]


class RealTimeMonitor:
    """实时监控器"""
    
    def __init__(self, config: Optional[DebugConfig] = None):
        self.config = config or DebugConfig()
        self.constraints: List[PhysicalConstraint] = []
        self.convergence_history: List[ConvergenceMetrics] = []
        self.performance_history: List[Dict[str, Any]] = []
        
        # 实时监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.last_update = time.time()
        
        # 回调函数
        self._callbacks: Dict[str, List[Callable]] = {
            'constraint_violation': [],
            'convergence_issue': [],
            'performance_degradation': []
        }
    
    def add_constraint(self, constraint: PhysicalConstraint):
        """添加物理约束"""
        self.constraints.append(constraint)
    
    def add_callback(self, event: str, callback: Callable):
        """添加回调函数"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def start_monitoring(self):
        """开始实时监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_loop()
    
    def stop_monitoring(self):
        """停止实时监控"""
        self.is_monitoring = False
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                self._update_metrics()
                time.sleep(self.config.monitoring_update_interval)
            except Exception as e:
                warnings.warn(f"监控循环出错: {str(e)}")
                break
    
    def _update_metrics(self):
        """更新监控指标"""
        current_time = time.time()
        
        # 检查约束违反
        constraint_violations = {}
        for constraint in self.constraints:
            try:
                residual = constraint.compute_residual()
                if abs(residual) > constraint.tolerance:
                    constraint_violations[constraint.name] = residual
                    self._trigger_callback('constraint_violation', constraint, residual)
            except Exception as e:
                warnings.warn(f"检查约束 {constraint.name} 失败: {str(e)}")
        
        # 更新性能指标
        performance_metrics = self._collect_performance_metrics()
        
        # 记录历史
        if self.config.monitoring_save_history:
            self.convergence_history.append(ConvergenceMetrics(
                iteration=len(self.convergence_history),
                timestamp=current_time,
                residual_norm=0.0,  # 需要从求解器获取
                relative_change=0.0,  # 需要从求解器获取
                constraint_violations=constraint_violations,
                performance_metrics=performance_metrics
            ))
            
            # 限制历史记录大小
            if len(self.convergence_history) > self.config.monitoring_max_history_size:
                self.convergence_history.pop(0)
        
        self.last_update = current_time
    
    def _collect_performance_metrics(self) -> Dict[str, float]:
        """收集性能指标"""
        metrics = {}
        
        try:
            import psutil
            
            # 系统资源使用
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            metrics.update({
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available': memory.available / (1024**3),  # GB
                'timestamp': time.time()
            })
            
        except ImportError:
            metrics['timestamp'] = time.time()
        
        return metrics
    
    def _trigger_callback(self, event: str, *args):
        """触发回调函数"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                warnings.warn(f"回调函数执行失败: {str(e)}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        if not self.convergence_history:
            return {}
        
        latest = self.convergence_history[-1]
        
        return {
            'total_iterations': len(self.convergence_history),
            'last_update': latest.timestamp,
            'constraint_violations': latest.constraint_violations,
            'performance_metrics': latest.performance_metrics,
            'monitoring_active': self.is_monitoring
        }


class ErrorDiagnostic:
    """错误诊断工具"""
    
    def __init__(self, config: Optional[DebugConfig] = None):
        self.config = config or DebugConfig()
        self.error_logs: List[Dict[str, Any]] = []
        self.diagnosis_rules: List[Callable] = []
        
        # 常见问题模式
        self._setup_diagnosis_rules()
    
    def _setup_diagnosis_rules(self):
        """设置诊断规则"""
        
        # 网格质量检查
        def check_mesh_quality(error_msg: str) -> Optional[str]:
            if any(keyword in error_msg.lower() for keyword in ['mesh', 'grid', 'element']):
                return "网格质量问题 - 建议检查网格质量、长宽比、角度等"
            return None
        
        # 数值稳定性检查
        def check_numerical_stability(error_msg: str) -> Optional[str]:
            if any(keyword in error_msg.lower() for keyword in ['convergence', 'divergence', 'nan', 'inf']):
                return "数值稳定性问题 - 建议减小时间步长、调整收敛容差、检查初始条件"
            return None
        
        # 物理参数检查
        def check_physics_parameters(error_msg: str) -> Optional[str]:
            if any(keyword in error_msg.lower() for keyword in ['parameter', 'boundary', 'initial']):
                return "物理参数问题 - 建议检查边界条件、初始条件、材料参数范围"
            return None
        
        # 内存问题检查
        def check_memory_issues(error_msg: str) -> Optional[str]:
            if any(keyword in error_msg.lower() for keyword in ['memory', 'out of memory', 'oom']):
                return "内存不足问题 - 建议减少网格规模、启用内存优化、使用分块计算"
            return None
        
        # 并行计算问题检查
        def check_parallel_issues(error_msg: str) -> Optional[str]:
            if any(keyword in error_msg.lower() for keyword in ['mpi', 'parallel', 'communication']):
                return "并行计算问题 - 建议检查MPI配置、减少进程数、调整通信参数"
            return None
        
        self.diagnosis_rules = [
            check_mesh_quality,
            check_numerical_stability,
            check_physics_parameters,
            check_memory_issues,
            check_parallel_issues
        ]
    
    def diagnose_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """诊断错误"""
        error_info = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc() if self.config.error_detailed_traceback else None,
            'context': context or {},
            'diagnosis': self._run_diagnosis(str(error)),
            'suggestions': self._generate_suggestions(error, context)
        }
        
        # 记录错误日志
        self.error_logs.append(error_info)
        
        # 保存错误日志
        if self.config.error_save_error_logs:
            self._save_error_log(error_info)
        
        return error_info
    
    def _run_diagnosis(self, error_msg: str) -> List[str]:
        """运行诊断规则"""
        diagnoses = []
        
        for rule in self.diagnosis_rules:
            try:
                result = rule(error_msg)
                if result:
                    diagnoses.append(result)
            except Exception as e:
                warnings.warn(f"诊断规则执行失败: {str(e)}")
        
        return diagnoses
    
    def _generate_suggestions(self, error: Exception, context: Optional[Dict[str, Any]]) -> List[str]:
        """生成修复建议"""
        suggestions = []
        
        # 基于错误类型的通用建议
        if isinstance(error, ValueError):
            suggestions.append("检查输入参数的有效性和范围")
        elif isinstance(error, RuntimeError):
            suggestions.append("检查运行时环境和依赖库版本")
        elif isinstance(error, MemoryError):
            suggestions.append("减少问题规模或启用内存优化")
        elif isinstance(error, ImportError):
            suggestions.append("检查依赖库安装和版本兼容性")
        
        # 基于上下文的特定建议
        if context:
            if 'mesh' in context:
                suggestions.append("检查网格文件格式和完整性")
            if 'solver' in context:
                suggestions.append("调整求解器参数和收敛条件")
            if 'boundary_conditions' in context:
                suggestions.append("验证边界条件的物理合理性")
        
        return suggestions
    
    def _save_error_log(self, error_info: Dict[str, Any]):
        """保存错误日志"""
        try:
            log_dir = Path(self.config.error_log_dir)
            log_dir.mkdir(exist_ok=True)
            
            timestamp = int(error_info['timestamp'])
            log_file = log_dir / f"error_log_{timestamp}.json"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            warnings.warn(f"保存错误日志失败: {str(e)}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        if not self.error_logs:
            return {}
        
        # 统计错误类型
        error_types = {}
        for log in self.error_logs:
            error_type = log['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_logs),
            'error_types': error_types,
            'recent_errors': self.error_logs[-5:],  # 最近5个错误
            'common_issues': self._identify_common_issues()
        }
    
    def _identify_common_issues(self) -> List[str]:
        """识别常见问题"""
        common_issues = []
        
        # 分析错误日志，识别重复出现的问题
        error_messages = [log['error_message'] for log in self.error_logs]
        
        # 简单的关键词统计
        keywords = ['mesh', 'convergence', 'memory', 'parameter', 'boundary']
        for keyword in keywords:
            count = sum(1 for msg in error_messages if keyword.lower() in msg.lower())
            if count > 1:
                common_issues.append(f"{keyword}相关问题 ({count}次)")
        
        return common_issues


class AdvancedVisualizer:
    """高级可视化器"""
    
    def __init__(self, config: Optional[DebugConfig] = None):
        self.config = config or DebugConfig()
        self.figures = {}
        self.animations = {}
        
        if HAS_MATPLOTLIB:
            plt.style.use('default')
            if HAS_SEABORN:
                sns.set_theme()
    
    def create_convergence_dashboard(self, monitor: RealTimeMonitor) -> 'matplotlib.figure.Figure':
        """创建收敛监控仪表板"""
        if not HAS_MATPLOTLIB:
            raise ImportError("需要matplotlib来创建可视化")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('仿真收敛监控仪表板', fontsize=16)
        
        # 子图1: 约束残差
        ax1 = axes[0, 0]
        ax1.set_title('物理约束残差')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('残差')
        ax1.grid(True)
        
        # 子图2: 收敛历史
        ax2 = axes[0, 1]
        ax2.set_title('收敛历史')
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('相对变化')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        # 子图3: 性能指标
        ax3 = axes[1, 0]
        ax3.set_title('系统性能')
        ax3.set_xlabel('时间')
        ax3.set_ylabel('使用率 (%)')
        ax3.grid(True)
        
        # 子图4: 约束违反统计
        ax4 = axes[1, 1]
        ax4.set_title('约束违反统计')
        ax4.set_xlabel('约束名称')
        ax4.set_ylabel('违反次数')
        
        # 添加实时更新功能
        def update_plots(frame):
            try:
                # 更新约束残差
                if monitor.convergence_history:
                    iterations = [m.iteration for m in monitor.convergence_history]
                    
                    # 约束残差
                    constraint_names = list(monitor.constraints)
                    for i, constraint in enumerate(constraint_names):
                        residuals = [m.constraint_violations.get(constraint.name, 0) 
                                   for m in monitor.convergence_history]
                        ax1.plot(iterations, residuals, label=constraint.name, marker='o')
                    
                    ax1.legend()
                    
                    # 收敛历史
                    relative_changes = [m.relative_change for m in monitor.convergence_history]
                    ax2.plot(iterations, relative_changes, 'b-o')
                    
                    # 性能指标
                    timestamps = [m.timestamp for m in monitor.convergence_history]
                    cpu_usage = [m.performance_metrics.get('cpu_usage', 0) 
                               for m in monitor.convergence_history]
                    memory_usage = [m.performance_metrics.get('memory_usage', 0) 
                                  for m in monitor.convergence_history]
                    
                    ax3.plot(timestamps, cpu_usage, 'r-', label='CPU')
                    ax3.plot(timestamps, memory_usage, 'b-', label='Memory')
                    ax3.legend()
                    
                    # 约束违反统计
                    violation_counts = {}
                    for m in monitor.convergence_history:
                        for constraint_name in m.constraint_violations:
                            violation_counts[constraint_name] = violation_counts.get(constraint_name, 0) + 1
                    
                    if violation_counts:
                        names = list(violation_counts.keys())
                        counts = list(violation_counts.values())
                        ax4.bar(names, counts)
                        ax4.tick_params(axis='x', rotation=45)
                
            except Exception as e:
                warnings.warn(f"更新图表失败: {str(e)}")
        
        # 创建动画
        ani = animation.FuncAnimation(fig, update_plots, interval=1000, blit=False)
        self.animations['convergence_dashboard'] = ani
        
        return fig
    
    def create_physics_constraint_plot(self, monitor: RealTimeMonitor) -> 'matplotlib.figure.Figure':
        """创建物理约束监控图"""
        if not HAS_MATPLOTLIB:
            raise ImportError("需要matplotlib来创建可视化")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('物理约束监控', fontsize=16)
        
        # 上图: 约束残差时间序列
        ax1 = axes[0]
        ax1.set_title('约束残差时间序列')
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('残差')
        ax1.set_yscale('log')
        ax1.grid(True)
        
        # 下图: 约束违反热力图
        ax2 = axes[1]
        ax2.set_title('约束违反热力图')
        ax2.set_xlabel('约束名称')
        ax2.set_ylabel('时间步')
        
        def update_constraint_plots(frame):
            try:
                if monitor.convergence_history:
                    # 时间序列
                    timestamps = [m.timestamp for m in monitor.convergence_history]
                    
                    for constraint in monitor.constraints:
                        residuals = [m.constraint_violations.get(constraint.name, 0) 
                                   for m in monitor.convergence_history]
                        ax1.plot(timestamps, residuals, label=constraint.name, marker='o')
                    
                    ax1.legend()
                    
                    # 热力图
                    constraint_names = [c.name for c in monitor.constraints]
                    if constraint_names:
                        violation_matrix = []
                        for m in monitor.convergence_history:
                            row = [1 if m.constraint_violations.get(name, 0) > 0 else 0 
                                  for name in constraint_names]
                            violation_matrix.append(row)
                        
                        if violation_matrix:
                            im = ax2.imshow(violation_matrix, cmap='Reds', aspect='auto')
                            ax2.set_xticks(range(len(constraint_names)))
                            ax2.set_xticklabels(constraint_names, rotation=45)
                            ax2.set_yticks(range(0, len(violation_matrix), max(1, len(violation_matrix)//10)))
                            
                            # 添加颜色条
                            if not hasattr(self, '_colorbar_added'):
                                plt.colorbar(im, ax=ax2, label='违反状态')
                                self._colorbar_added = True
                
            except Exception as e:
                warnings.warn(f"更新约束图表失败: {str(e)}")
        
        # 创建动画
        ani = animation.FuncAnimation(fig, update_constraint_plots, interval=1000, blit=False)
        self.animations['physics_constraint_plot'] = ani
        
        return fig
    
    def create_performance_dashboard(self, monitor: RealTimeMonitor) -> 'matplotlib.figure.Figure':
        """创建性能监控仪表板"""
        if not HAS_MATPLOTLIB:
            raise ImportError("需要matplotlib来创建可视化")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('系统性能监控', fontsize=16)
        
        # 子图1: CPU和内存使用率
        ax1 = axes[0, 0]
        ax1.set_title('系统资源使用率')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('使用率 (%)')
        ax1.grid(True)
        
        # 子图2: 内存使用量
        ax2 = axes[0, 1]
        ax2.set_title('内存使用量')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('内存 (GB)')
        ax2.grid(True)
        
        # 子图3: 性能趋势
        ax3 = axes[1, 0]
        ax3.set_title('性能趋势')
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('性能指标')
        ax3.grid(True)
        
        # 子图4: 资源分布饼图
        ax4 = axes[1, 1]
        ax4.set_title('当前资源分布')
        
        def update_performance_plots(frame):
            try:
                if monitor.convergence_history:
                    timestamps = [m.timestamp for m in monitor.convergence_history]
                    
                    # CPU和内存使用率
                    cpu_usage = [m.performance_metrics.get('cpu_usage', 0) 
                               for m in monitor.convergence_history]
                    memory_usage = [m.performance_metrics.get('memory_usage', 0) 
                                  for m in monitor.convergence_history]
                    
                    ax1.plot(timestamps, cpu_usage, 'r-', label='CPU', linewidth=2)
                    ax1.plot(timestamps, memory_usage, 'b-', label='Memory', linewidth=2)
                    ax1.legend()
                    
                    # 内存使用量
                    memory_available = [m.performance_metrics.get('memory_available', 0) 
                                      for m in monitor.convergence_history]
                    ax2.plot(timestamps, memory_available, 'g-', linewidth=2)
                    
                    # 性能趋势
                    iterations = [m.iteration for m in monitor.convergence_history]
                    if len(iterations) > 1:
                        # 计算性能指标（这里用简单的迭代速度作为示例）
                        performance_metrics = [1.0 / (m.timestamp - monitor.convergence_history[i-1].timestamp + 1e-6)
                                             for i, m in enumerate(monitor.convergence_history[1:], 1)]
                        ax3.plot(iterations[1:], performance_metrics, 'purple', linewidth=2)
                    
                    # 资源分布饼图
                    ax4.clear()
                    if monitor.convergence_history:
                        latest = monitor.convergence_history[-1]
                        cpu = latest.performance_metrics.get('cpu_usage', 0)
                        memory = latest.performance_metrics.get('memory_usage', 0)
                        available = latest.performance_metrics.get('memory_available', 0)
                        
                        labels = ['CPU使用', '内存使用', '内存可用']
                        sizes = [cpu, memory, available]
                        colors = ['#ff9999', '#66b3ff', '#99ff99']
                        
                        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                        ax4.set_title('当前资源分布')
                
            except Exception as e:
                warnings.warn(f"更新性能图表失败: {str(e)}")
        
        # 创建动画
        ani = animation.FuncAnimation(fig, update_performance_plots, interval=1000, blit=False)
        self.animations['performance_dashboard'] = ani
        
        return fig
    
    def save_all_plots(self, output_dir: str = "./debug_plots"):
        """保存所有图表"""
        if not self.figures:
            print("没有可保存的图表")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for name, fig in self.figures.items():
            try:
                filepath = output_path / f"{name}.{self.config.vis_plot_format}"
                fig.savefig(filepath, dpi=self.config.vis_dpi, 
                           bbox_inches='tight')
                print(f"图表已保存: {filepath}")
            except Exception as e:
                warnings.warn(f"保存图表 {name} 失败: {str(e)}")
    
    def close_all_figures(self):
        """关闭所有图表"""
        for fig in self.figures.values():
            try:
                plt.close(fig)
            except:
                pass
        self.figures.clear()
        
        for ani in self.animations.values():
            try:
                ani.event_source.stop()
            except:
                pass
        self.animations.clear()


class DebugManager:
    """调试管理器 - 统一管理所有调试工具"""
    
    def __init__(self, config: Optional[DebugConfig] = None):
        self.config = config or DebugConfig()
        
        # 初始化各个工具
        self.monitor = RealTimeMonitor(config)
        self.diagnostic = ErrorDiagnostic(config)
        self.visualizer = AdvancedVisualizer(config)
        
        # 状态跟踪
        self.is_active = False
        self.start_time = None
    
    def start_debugging(self):
        """开始调试"""
        if self.is_active:
            return
        
        self.is_active = True
        self.start_time = time.time()
        
        # 启动监控
        self.monitor.start_monitoring()
        
        print("🔍 调试模式已启动")
    
    def stop_debugging(self):
        """停止调试"""
        if not self.is_active:
            return
        
        self.is_active = False
        
        # 停止监控
        self.monitor.stop_monitoring()
        
        # 保存所有图表
        if self.config.vis_save_plots:
            self.visualizer.save_all_plots()
        
        print("🔍 调试模式已停止")
    
    def add_physical_constraint(self, name: str, equation: Callable, 
                               weight: float = 1.0, tolerance: float = 1e-6,
                               description: str = ""):
        """添加物理约束"""
        constraint = PhysicalConstraint(
            name=name,
            equation=equation,
            weight=weight,
            tolerance=tolerance,
            description=description
        )
        self.monitor.add_constraint(constraint)
    
    def diagnose_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """诊断错误"""
        return self.diagnostic.diagnose_error(error, context)
    
    def create_dashboards(self):
        """创建监控仪表板"""
        if not HAS_MATPLOTLIB:
            print("需要matplotlib来创建可视化仪表板")
            return
        
        # 创建各种仪表板
        self.visualizer.figures['convergence_dashboard'] = \
            self.visualizer.create_convergence_dashboard(self.monitor)
        
        self.visualizer.figures['physics_constraint_plot'] = \
            self.visualizer.create_physics_constraint_plot(self.monitor)
        
        self.visualizer.figures['performance_dashboard'] = \
            self.visualizer.create_performance_dashboard(self.monitor)
        
        print("📊 监控仪表板已创建")
    
    def get_debug_summary(self) -> Dict[str, Any]:
        """获取调试摘要"""
        return {
            'debug_active': self.is_active,
            'start_time': self.start_time,
            'monitoring_summary': self.monitor.get_monitoring_summary(),
            'error_summary': self.diagnostic.get_error_summary(),
            'total_constraints': len(self.monitor.constraints),
            'total_errors': len(self.diagnostic.error_logs)
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_debugging()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if exc_type:
            # 如果有异常，进行诊断
            self.diagnose_error(exc_val, {'context': 'debug_manager_exit'})
        
        self.stop_debugging()


# 便捷函数
def create_debug_manager(config: Optional[DebugConfig] = None) -> DebugManager:
    """创建调试管理器"""
    return DebugManager(config)


def quick_debug_setup() -> DebugManager:
    """快速调试设置"""
    config = DebugConfig(
        monitoring=DebugConfig.monitoring(enabled=True, update_interval=0.5),
        visualization=DebugConfig.visualization(realtime_plots=True, save_plots=True),
        error_diagnosis=DebugConfig.error_diagnosis(enabled=True, detailed_traceback=True)
    )
    
    return DebugManager(config)
