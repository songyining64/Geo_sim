"""
体渲染模块
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union, Dict, Any
import warnings

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Volume rendering will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive volume rendering will be limited.")


class VolumeRenderer(ABC):
    """体渲染基类"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        初始化体渲染器
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            图形大小 (宽度, 高度)
        dpi : int
            分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        self.current_figure = None
    
    @abstractmethod
    def render_volume(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                     values: np.ndarray, opacity: float = 0.3,
                     colormap: str = 'viridis', title: str = 'Volume Rendering') -> Any:
        """体渲染"""
        pass
    
    @abstractmethod
    def render_slice(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    values: np.ndarray, slice_axis: str = 'z', slice_value: float = 0.0,
                    colormap: str = 'viridis', title: str = 'Slice') -> Any:
        """切片渲染"""
        pass
    
    @abstractmethod
    def render_isosurface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         values: np.ndarray, level: float = 0.5,
                         colormap: str = 'viridis', title: str = 'Isosurface') -> Any:
        """等值面渲染"""
        pass
    
    @abstractmethod
    def show(self) -> None:
        """显示图形"""
        pass
    
    @abstractmethod
    def save(self, filename: str, **kwargs) -> None:
        """保存图形"""
        pass


class MatplotlibVolumeRenderer(VolumeRenderer):
    """基于Matplotlib的体渲染器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        super().__init__(figsize, dpi)
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for MatplotlibVolumeRenderer")
    
    def render_volume(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                     values: np.ndarray, opacity: float = 0.3,
                     colormap: str = 'viridis', title: str = 'Volume Rendering') -> plt.Figure:
        """体渲染"""
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.current_ax = self.current_figure.add_subplot(111, projection='3d')
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 为了性能，只显示部分点
        step = max(1, min(x.size // 1000, y.size // 1000, z.size // 1000))
        
        # 创建体渲染（使用散点图模拟）
        scatter = self.current_ax.scatter(x[::step], y[::step], z[::step], 
                                        c=values[::step], alpha=opacity, 
                                        cmap=colormap, s=1)
        
        # 添加颜色条
        self.current_figure.colorbar(scatter, ax=self.current_ax)
        
        # 设置标题和标签
        self.current_ax.set_title(title)
        self.current_ax.set_xlabel('X')
        self.current_ax.set_ylabel('Y')
        self.current_ax.set_zlabel('Z')
        
        return self.current_figure
    
    def render_slice(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    values: np.ndarray, slice_axis: str = 'z', slice_value: float = 0.0,
                    colormap: str = 'viridis', title: str = 'Slice') -> plt.Figure:
        """切片渲染"""
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.current_ax = self.current_figure.add_subplot(111, projection='3d')
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 根据切片轴选择切片
        if slice_axis == 'z':
            z_idx = np.argmin(np.abs(z - slice_value))
            if z_idx < values.shape[2]:
                X, Y = np.meshgrid(x[:, :, z_idx], y[:, :, z_idx])
                Z = np.full_like(X, slice_value)
                V = values[:, :, z_idx]
        elif slice_axis == 'y':
            y_idx = np.argmin(np.abs(y - slice_value))
            if y_idx < values.shape[1]:
                X, Z = np.meshgrid(x[:, y_idx, :], z[:, y_idx, :])
                Y = np.full_like(X, slice_value)
                V = values[:, y_idx, :]
        elif slice_axis == 'x':
            x_idx = np.argmin(np.abs(x - slice_value))
            if x_idx < values.shape[0]:
                Y, Z = np.meshgrid(y[x_idx, :, :], z[x_idx, :, :])
                X = np.full_like(Y, slice_value)
                V = values[x_idx, :, :]
        else:
            raise ValueError("slice_axis must be 'x', 'y', or 'z'")
        
        # 绘制切片
        surf = self.current_ax.plot_surface(X, Y, Z, facecolors=plt.cm.get_cmap(colormap)(V), 
                                          alpha=0.8)
        
        # 设置标题和标签
        self.current_ax.set_title(title)
        self.current_ax.set_xlabel('X')
        self.current_ax.set_ylabel('Y')
        self.current_ax.set_zlabel('Z')
        
        return self.current_figure
    
    def render_isosurface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         values: np.ndarray, level: float = 0.5,
                         colormap: str = 'viridis', title: str = 'Isosurface') -> plt.Figure:
        """等值面渲染"""
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.current_ax = self.current_figure.add_subplot(111, projection='3d')
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 使用contour3D绘制等值面
        # 为了简化，我们使用多个2D切片来近似3D等值面
        z_levels = np.linspace(z.min(), z.max(), 10)
        
        for z_level in z_levels:
            z_idx = np.argmin(np.abs(z - z_level))
            if z_idx < values.shape[2]:
                contour = self.current_ax.contour(x[:, :, z_idx], y[:, :, z_idx], 
                                                values[:, :, z_idx], 
                                                levels=[level], alpha=0.3)
        
        # 设置标题和标签
        self.current_ax.set_title(title)
        self.current_ax.set_xlabel('X')
        self.current_ax.set_ylabel('Y')
        self.current_ax.set_zlabel('Z')
        
        return self.current_figure
    
    def show(self) -> None:
        """显示图形"""
        if self.current_figure is not None:
            plt.tight_layout()
            plt.show()
    
    def save(self, filename: str, **kwargs) -> None:
        """保存图形"""
        if self.current_figure is not None:
            self.current_figure.savefig(filename, dpi=self.dpi, bbox_inches='tight', **kwargs)
            print(f"📁 图片已保存: {filename}")


class PlotlyVolumeRenderer(VolumeRenderer):
    """基于Plotly的体渲染器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        super().__init__(figsize, dpi)
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for PlotlyVolumeRenderer")
    
    def render_volume(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                     values: np.ndarray, opacity: float = 0.3,
                     colormap: str = 'viridis', title: str = 'Volume Rendering') -> go.Figure:
        """体渲染"""
        fig = go.Figure()
        
        # 添加体渲染
        fig.add_trace(go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=values.flatten(),
            opacity=opacity,
            colorscale=colormap,
            name='Volume'
        ))
        
        # 设置布局
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=self.figsize[0] * 100,
            height=self.figsize[1] * 100
        )
        
        self.current_figure = fig
        return fig
    
    def render_slice(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    values: np.ndarray, slice_axis: str = 'z', slice_value: float = 0.0,
                    colormap: str = 'viridis', title: str = 'Slice') -> go.Figure:
        """切片渲染"""
        fig = go.Figure()
        
        # 根据切片轴选择切片
        if slice_axis == 'z':
            z_idx = np.argmin(np.abs(z - slice_value))
            if z_idx < values.shape[2]:
                fig.add_trace(go.Surface(
                    x=x[:, :, z_idx],
                    y=y[:, :, z_idx],
                    z=np.full_like(x[:, :, z_idx], slice_value),
                    surfacecolor=values[:, :, z_idx],
                    colorscale=colormap,
                    name='Slice'
                ))
        elif slice_axis == 'y':
            y_idx = np.argmin(np.abs(y - slice_value))
            if y_idx < values.shape[1]:
                fig.add_trace(go.Surface(
                    x=x[:, y_idx, :],
                    z=z[:, y_idx, :],
                    y=np.full_like(x[:, y_idx, :], slice_value),
                    surfacecolor=values[:, y_idx, :],
                    colorscale=colormap,
                    name='Slice'
                ))
        elif slice_axis == 'x':
            x_idx = np.argmin(np.abs(x - slice_value))
            if x_idx < values.shape[0]:
                fig.add_trace(go.Surface(
                    y=y[x_idx, :, :],
                    z=z[x_idx, :, :],
                    x=np.full_like(y[x_idx, :, :], slice_value),
                    surfacecolor=values[x_idx, :, :],
                    colorscale=colormap,
                    name='Slice'
                ))
        else:
            raise ValueError("slice_axis must be 'x', 'y', or 'z'")
        
        # 设置布局
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=self.figsize[0] * 100,
            height=self.figsize[1] * 100
        )
        
        self.current_figure = fig
        return fig
    
    def render_isosurface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         values: np.ndarray, level: float = 0.5,
                         colormap: str = 'viridis', title: str = 'Isosurface') -> go.Figure:
        """等值面渲染"""
        fig = go.Figure()
        
        # 添加等值面
        fig.add_trace(go.Isosurface(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=values.flatten(),
            isomin=level,
            isomax=level,
            colorscale=colormap,
            name='Isosurface'
        ))
        
        # 设置布局
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=self.figsize[0] * 100,
            height=self.figsize[1] * 100
        )
        
        self.current_figure = fig
        return fig
    
    def show(self) -> None:
        """显示图形"""
        if self.current_figure is not None:
            self.current_figure.show()
    
    def save(self, filename: str, **kwargs) -> None:
        """保存图形"""
        if self.current_figure is not None:
            try:
                # 尝试保存为图片格式
                self.current_figure.write_image(filename, **kwargs)
                print(f"📁 图片已保存: {filename}")
            except ValueError as e:
                if "kaleido" in str(e):
                    # 如果没有kaleido，保存为HTML格式
                    html_filename = filename.replace('.png', '.html').replace('.jpg', '.html')
                    self.current_figure.write_html(html_filename)
                    print(f"📁 HTML文件已保存: {html_filename} (需要kaleido包来保存图片格式)")
                else:
                    print(f"❌ 保存失败: {e}")
            except Exception as e:
                print(f"❌ 保存失败: {e}")


def demo_volume_rendering():
    """演示体渲染功能"""
    print("🎨 体渲染演示")
    print("=" * 50)
    
    # 生成示例3D数据
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(-2, 2, 20)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # 3D标量场
    values = np.sin(X) * np.cos(Y) * np.exp(-Z**2)
    
    try:
        # Matplotlib体渲染
        print("📊 使用Matplotlib进行体渲染...")
        renderer_mpl = MatplotlibVolumeRenderer()
        
        # 体渲染
        renderer_mpl.render_volume(X, Y, Z, values, opacity=0.3, title='体渲染 - Matplotlib')
        renderer_mpl.save('volume_rendering_mpl.png')
        
        # 切片渲染
        renderer_mpl.render_slice(X, Y, Z, values, slice_axis='z', slice_value=0.0, title='切片 - Matplotlib')
        renderer_mpl.save('slice_rendering_mpl.png')
        
        # 等值面渲染
        renderer_mpl.render_isosurface(X, Y, Z, values, level=0.5, title='等值面 - Matplotlib')
        renderer_mpl.save('isosurface_rendering_mpl.png')
        
        print("✅ Matplotlib体渲染完成")
        
    except ImportError as e:
        print(f"❌ Matplotlib不可用: {e}")
    
    try:
        # Plotly体渲染
        print("\n📊 使用Plotly进行体渲染...")
        renderer_pl = PlotlyVolumeRenderer()
        
        # 体渲染
        renderer_pl.render_volume(X, Y, Z, values, opacity=0.3, title='体渲染 - Plotly')
        renderer_pl.save('volume_rendering_plotly.png')
        
        # 切片渲染
        renderer_pl.render_slice(X, Y, Z, values, slice_axis='z', slice_value=0.0, title='切片 - Plotly')
        renderer_pl.save('slice_rendering_plotly.png')
        
        # 等值面渲染
        renderer_pl.render_isosurface(X, Y, Z, values, level=0.5, title='等值面 - Plotly')
        renderer_pl.save('isosurface_rendering_plotly.png')
        
        print("✅ Plotly体渲染完成")
        
    except ImportError as e:
        print(f"❌ Plotly不可用: {e}")
    
    print("\n🎯 体渲染演示完成！")


if __name__ == "__main__":
    demo_volume_rendering()
