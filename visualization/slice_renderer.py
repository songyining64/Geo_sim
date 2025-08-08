"""
切片渲染模块
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
    warnings.warn("Matplotlib not available. Slice rendering will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive slice rendering will be limited.")


class SliceRenderer(ABC):
    """切片渲染基类"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        初始化切片渲染器
        
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
    def extract_slice(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     values: np.ndarray, slice_axis: str = 'z', slice_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """提取切片"""
        pass
    
    @abstractmethod
    def render_slice(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    values: np.ndarray, slice_axis: str = 'z', slice_value: float = 0.0,
                    colormap: str = 'viridis', title: str = 'Slice') -> Any:
        """渲染切片"""
        pass
    
    @abstractmethod
    def render_multiple_slices(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                              values: np.ndarray, slice_axes: List[str], slice_values: List[float],
                              colormap: str = 'viridis', title: str = 'Multiple Slices') -> Any:
        """渲染多个切片"""
        pass
    
    @abstractmethod
    def show(self) -> None:
        """显示图形"""
        pass
    
    @abstractmethod
    def save(self, filename: str, **kwargs) -> None:
        """保存图形"""
        pass


class MatplotlibSliceRenderer(SliceRenderer):
    """基于Matplotlib的切片渲染器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        super().__init__(figsize, dpi)
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for MatplotlibSliceRenderer")
    
    def extract_slice(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     values: np.ndarray, slice_axis: str = 'z', slice_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """提取切片"""
        try:
            # 处理输入数据
            if x.ndim == 3 and y.ndim == 3 and z.ndim == 3:
                # 已经是3D网格
                X, Y, Z = x, y, z
            else:
                # 需要创建3D网格
                if x.ndim == 1 and y.ndim == 1 and z.ndim == 1:
                    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                else:
                    raise ValueError("输入数组必须是1D或3D")
            
            if slice_axis == 'z':
                # 找到最接近的z索引
                z_idx = np.argmin(np.abs(Z[0, 0, :] - slice_value))
                if z_idx < values.shape[2]:
                    z_idx = min(z_idx, values.shape[2] - 1)
                    x_slice = X[:, :, z_idx]
                    y_slice = Y[:, :, z_idx]
                    X_2d, Y_2d = np.meshgrid(x_slice[0, :], y_slice[:, 0])
                    Z_2d = np.full_like(X_2d, slice_value)
                    V_2d = values[:, :, z_idx]
                    return X_2d, Y_2d, Z_2d, V_2d
            elif slice_axis == 'y':
                # 找到最接近的y索引
                y_idx = np.argmin(np.abs(Y[0, :, 0] - slice_value))
                if y_idx < values.shape[1]:
                    y_idx = min(y_idx, values.shape[1] - 1)
                    x_slice = X[:, y_idx, :]
                    z_slice = Z[:, y_idx, :]
                    X_2d, Z_2d = np.meshgrid(x_slice[:, 0], z_slice[0, :])
                    Y_2d = np.full_like(X_2d, slice_value)
                    V_2d = values[:, y_idx, :]
                    return X_2d, Y_2d, Z_2d, V_2d
            elif slice_axis == 'x':
                # 找到最接近的x索引
                x_idx = np.argmin(np.abs(X[:, 0, 0] - slice_value))
                if x_idx < values.shape[0]:
                    x_idx = min(x_idx, values.shape[0] - 1)
                    y_slice = Y[x_idx, :, :]
                    z_slice = Z[x_idx, :, :]
                    Y_2d, Z_2d = np.meshgrid(y_slice[:, 0], z_slice[0, :])
                    X_2d = np.full_like(Y_2d, slice_value)
                    V_2d = values[x_idx, :, :]
                    return X_2d, Y_2d, Z_2d, V_2d
            else:
                raise ValueError("slice_axis must be 'x', 'y', or 'z'")
        except Exception as e:
            print(f"⚠️ 切片提取失败: {e}")
        
        # 如果没有找到切片，返回默认切片
        print("⚠️ 未找到切片，返回默认切片")
        x_range = np.linspace(-2, 2, 10)
        y_range = np.linspace(-2, 2, 10)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.full_like(X, slice_value)
        V = np.zeros_like(X)
        return X, Y, Z, V
    
    def render_slice(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    values: np.ndarray, slice_axis: str = 'z', slice_value: float = 0.0,
                    colormap: str = 'viridis', title: str = 'Slice') -> plt.Figure:
        """渲染切片"""
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.current_ax = self.current_figure.add_subplot(111, projection='3d')
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 提取切片
        X, Y, Z, V = self.extract_slice(x, y, z, values, slice_axis, slice_value)
        
        if len(X) > 0 and X.size > 0:
            try:
                # 确保数据是2D的
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                if Y.ndim == 1:
                    Y = Y.reshape(-1, 1)
                if Z.ndim == 1:
                    Z = Z.reshape(-1, 1)
                if V.ndim == 1:
                    V = V.reshape(-1, 1)
                
                # 绘制切片
                surf = self.current_ax.plot_surface(X, Y, Z, facecolors=plt.cm.get_cmap(colormap)(V), 
                                                  alpha=0.8)
            except Exception as e:
                print(f"⚠️ 切片渲染失败: {e}")
                # 如果渲染失败，显示一个简单的平面
                x_range = np.linspace(-2, 2, 10)
                y_range = np.linspace(-2, 2, 10)
                X, Y = np.meshgrid(x_range, y_range)
                Z = np.full_like(X, slice_value)
                self.current_ax.plot_surface(X, Y, Z, alpha=0.3, color='gray')
        else:
            # 如果没有切片数据，显示一个简单的平面
            x_range = np.linspace(-2, 2, 10)
            y_range = np.linspace(-2, 2, 10)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.full_like(X, slice_value)
            self.current_ax.plot_surface(X, Y, Z, alpha=0.3, color='gray')
        
        # 设置标题和标签
        self.current_ax.set_title(title)
        self.current_ax.set_xlabel('X')
        self.current_ax.set_ylabel('Y')
        self.current_ax.set_zlabel('Z')
        
        return self.current_figure
    
    def render_multiple_slices(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                              values: np.ndarray, slice_axes: List[str], slice_values: List[float],
                              colormap: str = 'viridis', title: str = 'Multiple Slices') -> plt.Figure:
        """渲染多个切片"""
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.current_ax = self.current_figure.add_subplot(111, projection='3d')
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 为每个切片分配颜色
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(slice_axes)))
        
        for i, (slice_axis, slice_value) in enumerate(zip(slice_axes, slice_values)):
            # 提取切片
            X, Y, Z, V = self.extract_slice(x, y, z, values, slice_axis, slice_value)
            
            if len(X) > 0:
                # 绘制切片
                self.current_ax.plot_surface(X, Y, Z, facecolors=colors[i], 
                                           alpha=0.6, label=f'{slice_axis}={slice_value:.2f}')
        
        # 添加图例
        self.current_ax.legend()
        
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


class PlotlySliceRenderer(SliceRenderer):
    """基于Plotly的切片渲染器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        super().__init__(figsize, dpi)
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for PlotlySliceRenderer")
    
    def extract_slice(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     values: np.ndarray, slice_axis: str = 'z', slice_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """提取切片"""
        if slice_axis == 'z':
            z_idx = np.argmin(np.abs(z - slice_value))
            if z_idx < values.shape[2]:
                X, Y = np.meshgrid(x[:, :, z_idx], y[:, :, z_idx])
                Z = np.full_like(X, slice_value)
                V = values[:, :, z_idx]
                return X, Y, Z, V
        elif slice_axis == 'y':
            y_idx = np.argmin(np.abs(y - slice_value))
            if y_idx < values.shape[1]:
                X, Z = np.meshgrid(x[:, y_idx, :], z[:, y_idx, :])
                Y = np.full_like(X, slice_value)
                V = values[:, y_idx, :]
                return X, Y, Z, V
        elif slice_axis == 'x':
            x_idx = np.argmin(np.abs(x - slice_value))
            if x_idx < values.shape[0]:
                Y, Z = np.meshgrid(y[x_idx, :, :], z[x_idx, :, :])
                X = np.full_like(Y, slice_value)
                V = values[x_idx, :, :]
                return X, Y, Z, V
        else:
            raise ValueError("slice_axis must be 'x', 'y', or 'z'")
        
        # 如果没有找到切片，返回空数组
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    def render_slice(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    values: np.ndarray, slice_axis: str = 'z', slice_value: float = 0.0,
                    colormap: str = 'viridis', title: str = 'Slice') -> go.Figure:
        """渲染切片"""
        fig = go.Figure()
        
        # 提取切片
        X, Y, Z, V = self.extract_slice(x, y, z, values, slice_axis, slice_value)
        
        if len(X) > 0:
            # 添加切片
            fig.add_trace(go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=V,
                colorscale=colormap,
                name=f'Slice ({slice_axis}={slice_value:.2f})'
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
    
    def render_multiple_slices(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                              values: np.ndarray, slice_axes: List[str], slice_values: List[float],
                              colormap: str = 'viridis', title: str = 'Multiple Slices') -> go.Figure:
        """渲染多个切片"""
        fig = go.Figure()
        
        # 为每个切片分配颜色
        colors = px.colors.sample_colorscale(colormap, len(slice_axes))
        
        for i, (slice_axis, slice_value) in enumerate(zip(slice_axes, slice_values)):
            # 提取切片
            X, Y, Z, V = self.extract_slice(x, y, z, values, slice_axis, slice_value)
            
            if len(X) > 0:
                # 添加切片
                fig.add_trace(go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    surfacecolor=V,
                    colorscale=colormap,
                    name=f'Slice ({slice_axis}={slice_value:.2f})',
                    opacity=0.7
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


def demo_slice_rendering():
    """演示切片渲染功能"""
    print("🎨 切片渲染演示")
    print("=" * 50)
    
    # 生成示例3D数据
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(-2, 2, 20)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # 3D标量场
    values = np.sin(X) * np.cos(Y) * np.exp(-Z**2)
    
    try:
        # Matplotlib切片渲染
        print("📊 使用Matplotlib进行切片渲染...")
        renderer_mpl = MatplotlibSliceRenderer()
        
        # 单个切片
        renderer_mpl.render_slice(X, Y, Z, values, slice_axis='z', slice_value=0.0, title='切片 - Matplotlib')
        renderer_mpl.save('slice_mpl.png')
        
        # 多个切片
        slice_axes = ['z', 'y', 'x']
        slice_values = [0.0, 0.0, 0.0]
        renderer_mpl.render_multiple_slices(X, Y, Z, values, slice_axes, slice_values, title='多个切片 - Matplotlib')
        renderer_mpl.save('multiple_slices_mpl.png')
        
        print("✅ Matplotlib切片渲染完成")
        
    except ImportError as e:
        print(f"❌ Matplotlib不可用: {e}")
    
    try:
        # Plotly切片渲染
        print("\n📊 使用Plotly进行切片渲染...")
        renderer_pl = PlotlySliceRenderer()
        
        # 单个切片
        renderer_pl.render_slice(X, Y, Z, values, slice_axis='z', slice_value=0.0, title='切片 - Plotly')
        renderer_pl.save('slice_plotly.png')
        
        # 多个切片
        slice_axes = ['z', 'y', 'x']
        slice_values = [0.0, 0.0, 0.0]
        renderer_pl.render_multiple_slices(X, Y, Z, values, slice_axes, slice_values, title='多个切片 - Plotly')
        renderer_pl.save('multiple_slices_plotly.png')
        
        print("✅ Plotly切片渲染完成")
        
    except ImportError as e:
        print(f"❌ Plotly不可用: {e}")
    
    print("\n🎯 切片渲染演示完成！")


if __name__ == "__main__":
    demo_slice_rendering()
