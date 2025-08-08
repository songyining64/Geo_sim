"""
实时可视化系统

包含完整的实时可视化功能：
1. 实时数据更新
2. 交互式操作
3. 动画生成
4. 多视图同步
5. 性能优化
"""

import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider, Button, RadioButtons
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Real-time visualization will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("plotly not available. Interactive visualization will be limited.")


@dataclass
class VisualizationConfig:
    """可视化配置"""
    backend: str = 'matplotlib'  # 'matplotlib' or 'plotly'
    update_interval: float = 0.1  # 更新间隔（秒）
    max_fps: int = 30  # 最大帧率
    auto_scale: bool = True  # 自动缩放
    show_legend: bool = True  # 显示图例
    colormap: str = 'viridis'  # 颜色映射
    figure_size: Tuple[int, int] = (12, 8)  # 图像大小
    dpi: int = 100  # 分辨率


class BaseRealTimeVisualizer(ABC):
    """实时可视化器基类"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.is_running = False
        self.update_thread = None
        self.data_queue = []
        self.callbacks = []
        self.last_update_time = 0.0
        
    @abstractmethod
    def setup_display(self):
        """设置显示"""
        pass
    
    @abstractmethod
    def update_display(self, data: Dict):
        """更新显示"""
        pass
    
    @abstractmethod
    def add_interaction(self, interaction_type: str, **kwargs):
        """添加交互功能"""
        pass
    
    def start(self):
        """启动实时可视化"""
        if self.is_running:
            return
        
        self.is_running = True
        self.setup_display()
        
        # 启动更新线程
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        print("🔄 实时可视化已启动")
    
    def stop(self):
        """停止实时可视化"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
        print("🛑 实时可视化已停止")
    
    def _update_loop(self):
        """更新循环"""
        while self.is_running:
            current_time = time.time()
            
            # 检查是否需要更新
            if current_time - self.last_update_time >= self.config.update_interval:
                if self.data_queue:
                    data = self.data_queue.pop(0)
                    self.update_display(data)
                    self.last_update_time = current_time
            
            time.sleep(0.01)  # 避免过度占用CPU
    
    def add_data(self, data: Dict):
        """添加数据"""
        self.data_queue.append(data)
        
        # 限制队列大小
        if len(self.data_queue) > 100:
            self.data_queue.pop(0)
    
    def add_callback(self, callback: Callable):
        """添加回调函数"""
        self.callbacks.append(callback)


class MatplotlibRealTimeVisualizer(BaseRealTimeVisualizer):
    """Matplotlib实时可视化器"""
    
    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)
        if not HAS_MATPLOTLIB:
            raise ImportError("需要安装matplotlib来使用MatplotlibRealTimeVisualizer")
        
        self.fig = None
        self.ax = None
        self.lines = []
        self.scatters = []
        self.contours = []
        self.images = []
        self.animation = None
    
    def setup_display(self):
        """设置显示"""
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        self.ax.set_title("实时可视化")
        self.ax.grid(True)
        
        # 设置自动缩放
        if self.config.auto_scale:
            self.ax.autoscale(enable=True)
        
        plt.show(block=False)
    
    def update_display(self, data: Dict):
        """更新显示"""
        if not self.fig or not self.ax:
            return
        
        # 清除之前的图形
        self.ax.clear()
        
        # 根据数据类型更新显示
        if 'contour' in data:
            self._update_contour(data['contour'])
        elif 'scatter' in data:
            self._update_scatter(data['scatter'])
        elif 'line' in data:
            self._update_line(data['line'])
        elif 'image' in data:
            self._update_image(data['image'])
        elif 'vector_field' in data:
            self._update_vector_field(data['vector_field'])
        
        # 更新标题和时间戳
        if 'title' in data:
            self.ax.set_title(data['title'])
        
        if 'timestamp' in data:
            self.ax.text(0.02, 0.98, f"时间: {data['timestamp']:.3f}s", 
                        transform=self.ax.transAxes, verticalalignment='top')
        
        # 刷新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _update_contour(self, contour_data: Dict):
        """更新等值线图"""
        x = contour_data.get('x')
        y = contour_data.get('y')
        z = contour_data.get('z')
        
        if x is not None and y is not None and z is not None:
            contour = self.ax.contour(x, y, z, levels=20, cmap=self.config.colormap)
            if self.config.show_legend:
                self.fig.colorbar(contour, ax=self.ax)
    
    def _update_scatter(self, scatter_data: Dict):
        """更新散点图"""
        x = scatter_data.get('x')
        y = scatter_data.get('y')
        c = scatter_data.get('c')
        
        if x is not None and y is not None:
            scatter = self.ax.scatter(x, y, c=c, cmap=self.config.colormap)
            if self.config.show_legend and c is not None:
                self.fig.colorbar(scatter, ax=self.ax)
    
    def _update_line(self, line_data: Dict):
        """更新线图"""
        x = line_data.get('x')
        y = line_data.get('y')
        
        if x is not None and y is not None:
            self.ax.plot(x, y, '-o', markersize=3)
    
    def _update_image(self, image_data: Dict):
        """更新图像"""
        image = image_data.get('data')
        
        if image is not None:
            im = self.ax.imshow(image, cmap=self.config.colormap, aspect='auto')
            if self.config.show_legend:
                self.fig.colorbar(im, ax=self.ax)
    
    def _update_vector_field(self, vector_data: Dict):
        """更新向量场"""
        x = vector_data.get('x')
        y = vector_data.get('y')
        u = vector_data.get('u')
        v = vector_data.get('v')
        
        if x is not None and y is not None and u is not None and v is not None:
            self.ax.quiver(x, y, u, v, scale=50)
    
    def add_interaction(self, interaction_type: str, **kwargs):
        """添加交互功能"""
        if interaction_type == 'slider':
            self._add_slider(**kwargs)
        elif interaction_type == 'button':
            self._add_button(**kwargs)
        elif interaction_type == 'radio':
            self._add_radio_buttons(**kwargs)
    
    def _add_slider(self, **kwargs):
        """添加滑块"""
        if not self.fig:
            return
        
        # 创建滑块
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
        slider = Slider(ax_slider, '参数', kwargs.get('min_val', 0), 
                       kwargs.get('max_val', 1), kwargs.get('init_val', 0.5))
        
        def update_slider(val):
            # 滑块更新回调
            if 'callback' in kwargs:
                kwargs['callback'](val)
        
        slider.on_changed(update_slider)
    
    def _add_button(self, **kwargs):
        """添加按钮"""
        if not self.fig:
            return
        
        # 创建按钮
        ax_button = plt.axes([0.8, 0.02, 0.1, 0.03])
        button = Button(ax_button, kwargs.get('label', '按钮'))
        
        def button_callback(event):
            if 'callback' in kwargs:
                kwargs['callback']()
        
        button.on_clicked(button_callback)
    
    def _add_radio_buttons(self, **kwargs):
        """添加单选按钮"""
        if not self.fig:
            return
        
        # 创建单选按钮
        ax_radio = plt.axes([0.02, 0.7, 0.15, 0.15])
        radio = RadioButtons(ax_radio, kwargs.get('labels', ['选项1', '选项2']))
        
        def radio_callback(label):
            if 'callback' in kwargs:
                kwargs['callback'](label)
        
        radio.on_clicked(radio_callback)
    
    def create_animation(self, data_sequence: List[Dict], filename: str = None):
        """创建动画"""
        if not data_sequence:
            return
        
        def animate(frame):
            self.update_display(data_sequence[frame])
            return self.ax,
        
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=len(data_sequence),
            interval=self.config.update_interval * 1000, blit=False
        )
        
        if filename:
            self.animation.save(filename, writer='pillow')
        
        return self.animation


class PlotlyRealTimeVisualizer(BaseRealTimeVisualizer):
    """Plotly实时可视化器"""
    
    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)
        if not HAS_PLOTLY:
            raise ImportError("需要安装plotly来使用PlotlyRealTimeVisualizer")
        
        self.fig = None
        self.data_traces = []
        self.layout = {}
    
    def setup_display(self):
        """设置显示"""
        self.fig = go.Figure()
        self.layout = {
            'title': '实时可视化',
            'xaxis': {'title': 'X轴'},
            'yaxis': {'title': 'Y轴'},
            'showlegend': self.config.show_legend,
            'width': self.config.figure_size[0] * 100,
            'height': self.config.figure_size[1] * 100
        }
        
        self.fig.update_layout(**self.layout)
        self.fig.show()
    
    def update_display(self, data: Dict):
        """更新显示"""
        if not self.fig:
            return
        
        # 清除之前的轨迹
        self.fig.data = []
        
        # 根据数据类型更新显示
        if 'contour' in data:
            self._update_contour(data['contour'])
        elif 'scatter' in data:
            self._update_scatter(data['scatter'])
        elif 'line' in data:
            self._update_line(data['line'])
        elif 'image' in data:
            self._update_image(data['image'])
        elif 'vector_field' in data:
            self._update_vector_field(data['vector_field'])
        
        # 更新标题和时间戳
        if 'title' in data:
            self.fig.update_layout(title=data['title'])
        
        # 刷新显示
        self.fig.show()
    
    def _update_contour(self, contour_data: Dict):
        """更新等值线图"""
        x = contour_data.get('x')
        y = contour_data.get('y')
        z = contour_data.get('z')
        
        if x is not None and y is not None and z is not None:
            contour = go.Contour(x=x, y=y, z=z, colorscale=self.config.colormap)
            self.fig.add_trace(contour)
    
    def _update_scatter(self, scatter_data: Dict):
        """更新散点图"""
        x = scatter_data.get('x')
        y = scatter_data.get('y')
        c = scatter_data.get('c')
        
        if x is not None and y is not None:
            scatter = go.Scatter(x=x, y=y, mode='markers', 
                               marker=dict(color=c, colorscale=self.config.colormap))
            self.fig.add_trace(scatter)
    
    def _update_line(self, line_data: Dict):
        """更新线图"""
        x = line_data.get('x')
        y = line_data.get('y')
        
        if x is not None and y is not None:
            line = go.Scatter(x=x, y=y, mode='lines+markers')
            self.fig.add_trace(line)
    
    def _update_image(self, image_data: Dict):
        """更新图像"""
        image = image_data.get('data')
        
        if image is not None:
            img = go.Image(z=image, colorscale=self.config.colormap)
            self.fig.add_trace(img)
    
    def _update_vector_field(self, vector_data: Dict):
        """更新向量场"""
        x = vector_data.get('x')
        y = vector_data.get('y')
        u = vector_data.get('u')
        v = vector_data.get('v')
        
        if x is not None and y is not None and u is not None and v is not None:
            # 创建向量场
            vectors = go.Scatter(x=x.flatten(), y=y.flatten(), 
                               mode='markers', marker=dict(size=1))
            self.fig.add_trace(vectors)
    
    def add_interaction(self, interaction_type: str, **kwargs):
        """添加交互功能"""
        if interaction_type == 'slider':
            self._add_slider(**kwargs)
        elif interaction_type == 'button':
            self._add_button(**kwargs)
    
    def _add_slider(self, **kwargs):
        """添加滑块"""
        if not self.fig:
            return
        
        # 创建滑块
        slider = dict(
            active=0,
            currentvalue={"prefix": "参数: "},
            len=0.9,
            x=0.1,
            xanchor="left",
            y=0,
            yanchor="bottom",
            steps=[]
        )
        
        # 添加滑块步骤
        min_val = kwargs.get('min_val', 0)
        max_val = kwargs.get('max_val', 1)
        n_steps = kwargs.get('n_steps', 10)
        
        for i in range(n_steps + 1):
            val = min_val + (max_val - min_val) * i / n_steps
            step = dict(
                method="restyle",
                args=[{"visible": [False] * len(self.fig.data)}],
                label=f"{val:.2f}"
            )
            slider["steps"].append(step)
        
        self.fig.update_layout(sliders=[slider])
    
    def _add_button(self, **kwargs):
        """添加按钮"""
        if not self.fig:
            return
        
        # 创建按钮
        button = dict(
            type="buttons",
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=False,
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top"
        )
        
        self.fig.update_layout(updatemenus=[button])


class RealTimeVisualizationManager:
    """实时可视化管理器"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.visualizers = {}
        self.data_sources = {}
        self.is_running = False
    
    def add_visualizer(self, name: str, visualizer: BaseRealTimeVisualizer):
        """添加可视化器"""
        self.visualizers[name] = visualizer
        print(f"✅ 添加可视化器: {name}")
    
    def add_data_source(self, name: str, data_source: Callable):
        """添加数据源"""
        self.data_sources[name] = data_source
        print(f"✅ 添加数据源: {name}")
    
    def start_all(self):
        """启动所有可视化器"""
        for name, visualizer in self.visualizers.items():
            visualizer.start()
        
        self.is_running = True
        print("🔄 所有可视化器已启动")
    
    def stop_all(self):
        """停止所有可视化器"""
        for name, visualizer in self.visualizers.items():
            visualizer.stop()
        
        self.is_running = False
        print("🛑 所有可视化器已停止")
    
    def update_all(self, data: Dict):
        """更新所有可视化器"""
        for name, visualizer in self.visualizers.items():
            visualizer.add_data(data)
    
    def get_visualizer(self, name: str) -> Optional[BaseRealTimeVisualizer]:
        """获取可视化器"""
        return self.visualizers.get(name)
    
    def remove_visualizer(self, name: str):
        """移除可视化器"""
        if name in self.visualizers:
            self.visualizers[name].stop()
            del self.visualizers[name]
            print(f"🗑️ 移除可视化器: {name}")


# 工厂函数
def create_realtime_visualizer(backend: str = 'matplotlib', 
                              config: VisualizationConfig = None) -> BaseRealTimeVisualizer:
    """创建实时可视化器"""
    if backend == 'matplotlib':
        return MatplotlibRealTimeVisualizer(config)
    elif backend == 'plotly':
        return PlotlyRealTimeVisualizer(config)
    else:
        raise ValueError(f"不支持的后端: {backend}")


def create_visualization_manager(config: VisualizationConfig = None) -> RealTimeVisualizationManager:
    """创建可视化管理器"""
    return RealTimeVisualizationManager(config)
