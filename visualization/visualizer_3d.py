"""
3D可视化模块 
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union, Dict, Any
import warnings

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. 3D visualization will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive 3D visualization will be limited.")


class Visualizer3D(ABC):
    """3D可视化基类"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        初始化3D可视化器
        
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
    def plot_volume(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                   values: np.ndarray, opacity: float = 0.3,
                   title: str = 'Volume Rendering', **kwargs) -> Any:
        """体渲染"""
        pass
    
    @abstractmethod
    def plot_isosurface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       values: np.ndarray, level: float = 0.5,
                       title: str = 'Isosurface', **kwargs) -> Any:
        """等值面"""
        pass
    
    @abstractmethod
    def plot_slice(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  values: np.ndarray, slice_axis: str = 'z', slice_value: float = 0.0,
                  title: str = 'Slice', **kwargs) -> Any:
        """切片显示"""
        pass
    
    @abstractmethod
    def plot_point_cloud(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        c: Optional[np.ndarray] = None,
                        title: str = 'Point Cloud', **kwargs) -> Any:
        """点云"""
        pass
    
    @abstractmethod
    def plot_vector_field_3d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            u: np.ndarray, v: np.ndarray, w: np.ndarray,
                            title: str = '3D Vector Field', **kwargs) -> Any:
        """3D向量场"""
        pass
    
    @abstractmethod
    def plot_mesh_3d(self, nodes: np.ndarray, elements: np.ndarray,
                    title: str = '3D Mesh', **kwargs) -> Any:
        """3D网格"""
        pass
    
    @abstractmethod
    def show(self) -> None:
        """显示图形"""
        pass
    
    @abstractmethod
    def save(self, filename: str, **kwargs) -> None:
        """保存图形"""
        pass


class MatplotlibVisualizer3D(Visualizer3D):
    """基于Matplotlib的3D可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        super().__init__(figsize, dpi)
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for MatplotlibVisualizer3D")
    
    def plot_volume(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                   values: np.ndarray, opacity: float = 0.3,
                   title: str = 'Volume Rendering', **kwargs) -> plt.Figure:
        """体渲染"""
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.current_ax = self.current_figure.add_subplot(111, projection='3d')
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 创建体渲染（使用散点图模拟）
        # 为了性能，只显示部分点
        step = max(1, min(x.size // 1000, y.size // 1000, z.size // 1000))
        
        scatter = self.current_ax.scatter(x[::step], y[::step], z[::step], 
                                        c=values[::step], alpha=opacity, 
                                        cmap='viridis', **kwargs)
        
        # 添加颜色条
        self.current_figure.colorbar(scatter, ax=self.current_ax)
        
        # 设置标题和标签
        self.current_ax.set_title(title)
        self.current_ax.set_xlabel('X')
        self.current_ax.set_ylabel('Y')
        self.current_ax.set_zlabel('Z')
        
        return self.current_figure
    
    def plot_isosurface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       values: np.ndarray, level: float = 0.5,
                       title: str = 'Isosurface', **kwargs) -> plt.Figure:
        """等值面"""
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
    
    def plot_slice(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                   values: np.ndarray, slice_axis: str = 'z', slice_value: float = 0.0,
                   title: str = 'Slice', **kwargs) -> plt.Figure:
        """切片显示"""
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.current_ax = self.current_figure.add_subplot(111, projection='3d')
        
        # 清除当前图形
        self.current_ax.clear()
        
        try:
            # 根据切片轴选择切片
            if slice_axis == 'z':
                z_idx = np.argmin(np.abs(z - slice_value))
                if z_idx < values.shape[2]:
                    # 创建2D网格
                    x_2d = x[0, :, z_idx] if x.ndim == 3 else x
                    y_2d = y[:, 0, z_idx] if y.ndim == 3 else y
                    X, Y = np.meshgrid(x_2d, y_2d)
                    Z = np.full_like(X, slice_value)
                    V = values[:, :, z_idx]
                    
                    # 绘制切片
                    surf = self.current_ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(V), 
                                                      alpha=0.8, **kwargs)
                    
            elif slice_axis == 'y':
                y_idx = np.argmin(np.abs(y - slice_value))
                if y_idx < values.shape[1]:
                    x_2d = x[0, y_idx, :] if x.ndim == 3 else x
                    z_2d = z[:, y_idx, 0] if z.ndim == 3 else z
                    X, Z = np.meshgrid(x_2d, z_2d)
                    Y = np.full_like(X, slice_value)
                    V = values[:, y_idx, :]
                    
                    # 绘制切片
                    surf = self.current_ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(V), 
                                                      alpha=0.8, **kwargs)
                    
            elif slice_axis == 'x':
                x_idx = np.argmin(np.abs(x - slice_value))
                if x_idx < values.shape[0]:
                    y_2d = y[x_idx, :, 0] if y.ndim == 3 else y
                    z_2d = z[x_idx, :, 0] if z.ndim == 3 else z
                    Y, Z = np.meshgrid(y_2d, z_2d)
                    X = np.full_like(Y, slice_value)
                    V = values[x_idx, :, :]
                    
                    # 绘制切片
                    surf = self.current_ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(V), 
                                                      alpha=0.8, **kwargs)
            else:
                raise ValueError("slice_axis must be 'x', 'y', or 'z'")
                
        except Exception as e:
            print(f"⚠️ 切片渲染失败: {e}")
            # 如果切片失败，显示一个简单的平面
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
    
    def plot_point_cloud(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        c: Optional[np.ndarray] = None,
                        title: str = 'Point Cloud', **kwargs) -> plt.Figure:
        """点云"""
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.current_ax = self.current_figure.add_subplot(111, projection='3d')
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 绘制点云
        scatter = self.current_ax.scatter(x, y, z, c=c, cmap='viridis', **kwargs)
        
        # 如果有颜色映射，添加颜色条
        if c is not None:
            self.current_figure.colorbar(scatter, ax=self.current_ax)
        
        # 设置标题和标签
        self.current_ax.set_title(title)
        self.current_ax.set_xlabel('X')
        self.current_ax.set_ylabel('Y')
        self.current_ax.set_zlabel('Z')
        
        return self.current_figure
    
    def plot_vector_field_3d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            u: np.ndarray, v: np.ndarray, w: np.ndarray,
                            title: str = '3D Vector Field', **kwargs) -> plt.Figure:
        """3D向量场"""
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.current_ax = self.current_figure.add_subplot(111, projection='3d')
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 为了性能，只显示部分向量
        step = max(1, min(x.size // 500, y.size // 500, z.size // 500))
        
        # 绘制3D向量场
        quiver = self.current_ax.quiver(x[::step], y[::step], z[::step],
                                       u[::step], v[::step], w[::step],
                                       length=0.1, normalize=True, **kwargs)
        
        # 设置标题和标签
        self.current_ax.set_title(title)
        self.current_ax.set_xlabel('X')
        self.current_ax.set_ylabel('Y')
        self.current_ax.set_zlabel('Z')
        
        return self.current_figure
    
    def plot_mesh_3d(self, nodes: np.ndarray, elements: np.ndarray,
                    title: str = '3D Mesh', **kwargs) -> plt.Figure:
        """3D网格"""
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.current_ax = self.current_figure.add_subplot(111, projection='3d')
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 绘制网格元素
        for element in elements:
            element_nodes = nodes[element]
            # 绘制四面体的面
            faces = [
                [element_nodes[0], element_nodes[1], element_nodes[2]],
                [element_nodes[0], element_nodes[1], element_nodes[3]],
                [element_nodes[0], element_nodes[2], element_nodes[3]],
                [element_nodes[1], element_nodes[2], element_nodes[3]]
            ]
            
            poly3d = Poly3DCollection(faces, alpha=0.3, facecolor='blue', **kwargs)
            self.current_ax.add_collection3d(poly3d)
        
        # 绘制节点
        self.current_ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], 
                               c='red', s=20, zorder=5)
        
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


class PlotlyVisualizer3D(Visualizer3D):
    """基于Plotly的3D可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        super().__init__(figsize, dpi)
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for PlotlyVisualizer3D")
    
    def plot_volume(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                   values: np.ndarray, opacity: float = 0.3,
                   title: str = 'Volume Rendering', **kwargs) -> go.Figure:
        """体渲染"""
        fig = go.Figure()
        
        # 添加体渲染
        fig.add_trace(go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=values.flatten(),
            opacity=opacity,
            colorscale='viridis',
            **kwargs
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
    
    def plot_isosurface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       values: np.ndarray, level: float = 0.5,
                       title: str = 'Isosurface', **kwargs) -> go.Figure:
        """等值面"""
        fig = go.Figure()
        
        # 添加等值面
        fig.add_trace(go.Isosurface(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=values.flatten(),
            isomin=level,
            isomax=level,
            colorscale='viridis',
            **kwargs
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
    
    def plot_slice(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  values: np.ndarray, slice_axis: str = 'z', slice_value: float = 0.0,
                  title: str = 'Slice', **kwargs) -> go.Figure:
        """切片显示"""
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
                    colorscale='viridis',
                    **kwargs
                ))
        elif slice_axis == 'y':
            y_idx = np.argmin(np.abs(y - slice_value))
            if y_idx < values.shape[1]:
                fig.add_trace(go.Surface(
                    x=x[:, y_idx, :],
                    z=z[:, y_idx, :],
                    y=np.full_like(x[:, y_idx, :], slice_value),
                    surfacecolor=values[:, y_idx, :],
                    colorscale='viridis',
                    **kwargs
                ))
        elif slice_axis == 'x':
            x_idx = np.argmin(np.abs(x - slice_value))
            if x_idx < values.shape[0]:
                fig.add_trace(go.Surface(
                    y=y[x_idx, :, :],
                    z=z[x_idx, :, :],
                    x=np.full_like(y[x_idx, :, :], slice_value),
                    surfacecolor=values[x_idx, :, :],
                    colorscale='viridis',
                    **kwargs
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
    
    def plot_point_cloud(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        c: Optional[np.ndarray] = None,
                        title: str = 'Point Cloud', **kwargs) -> go.Figure:
        """点云"""
        fig = go.Figure()
        
        # 添加点云
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                color=c,
                colorscale='viridis' if c is not None else None,
                size=3
            ),
            **kwargs
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
    
    def plot_vector_field_3d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            u: np.ndarray, v: np.ndarray, w: np.ndarray,
                            title: str = '3D Vector Field', **kwargs) -> go.Figure:
        """3D向量场"""
        fig = go.Figure()
        
        # 添加3D向量场
        fig.add_trace(go.Cone(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            u=u.flatten(),
            v=v.flatten(),
            w=w.flatten(),
            colorscale='viridis',
            **kwargs
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
    
    def plot_mesh_3d(self, nodes: np.ndarray, elements: np.ndarray,
                    title: str = '3D Mesh', **kwargs) -> go.Figure:
        """3D网格"""
        fig = go.Figure()
        
        # 绘制网格线
        for element in elements:
            element_nodes = nodes[element]
            # 绘制四面体的边
            edges = [
                [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
            ]
            
            for edge in edges:
                fig.add_trace(go.Scatter3d(
                    x=[element_nodes[edge[0], 0], element_nodes[edge[1], 0]],
                    y=[element_nodes[edge[0], 1], element_nodes[edge[1], 1]],
                    z=[element_nodes[edge[0], 2], element_nodes[edge[1], 2]],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False,
                    **kwargs
                ))
        
        # 绘制节点
        fig.add_trace(go.Scatter3d(
            x=nodes[:, 0],
            y=nodes[:, 1],
            z=nodes[:, 2],
            mode='markers',
            marker=dict(color='red', size=5),
            name='Nodes',
            **kwargs
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


def demo_3d_visualization():
    """演示3D可视化功能"""
    print("🎨 3D可视化演示")
    print("=" * 50)
    
    # 生成示例3D数据
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(-2, 2, 20)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # 3D标量场
    values = np.sin(X) * np.cos(Y) * np.exp(-Z**2)
    
    # 3D向量场
    U = -Y
    V = X
    W = np.zeros_like(Z)
    
    # 点云数据
    np.random.seed(42)
    n_points = 1000
    point_x = np.random.randn(n_points)
    point_y = np.random.randn(n_points)
    point_z = np.random.randn(n_points)
    point_c = np.random.rand(n_points)
    
    # 3D网格数据
    nodes = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    elements = np.array([
        [0, 1, 2, 4], [1, 2, 6, 4], [2, 3, 7, 6], [0, 3, 7, 4]
    ])
    
    try:
        # Matplotlib可视化
        print("📊 使用Matplotlib进行3D可视化...")
        viz_mpl = MatplotlibVisualizer3D()
        
        # 点云
        viz_mpl.plot_point_cloud(point_x, point_y, point_z, c=point_c, title='点云 - Matplotlib')
        viz_mpl.save('point_cloud_mpl.png')
        
        # 切片
        viz_mpl.plot_slice(X, Y, Z, values, slice_axis='z', slice_value=0.0, title='切片 - Matplotlib')
        viz_mpl.save('slice_mpl.png')
        
        # 3D网格
        viz_mpl.plot_mesh_3d(nodes, elements, title='3D网格 - Matplotlib')
        viz_mpl.save('mesh_3d_mpl.png')
        
        print("✅ Matplotlib 3D可视化完成")
        
    except ImportError as e:
        print(f"❌ Matplotlib不可用: {e}")
    
    try:
        # Plotly可视化
        print("\n📊 使用Plotly进行3D可视化...")
        viz_pl = PlotlyVisualizer3D()
        
        # 点云
        viz_pl.plot_point_cloud(point_x, point_y, point_z, c=point_c, title='点云 - Plotly')
        viz_pl.save('point_cloud_plotly.png')
        
        # 等值面
        viz_pl.plot_isosurface(X, Y, Z, values, level=0.5, title='等值面 - Plotly')
        viz_pl.save('isosurface_plotly.png')
        
        # 切片
        viz_pl.plot_slice(X, Y, Z, values, slice_axis='z', slice_value=0.0, title='切片 - Plotly')
        viz_pl.save('slice_plotly.png')
        
        # 3D向量场
        viz_pl.plot_vector_field_3d(X, Y, Z, U, V, W, title='3D向量场 - Plotly')
        viz_pl.save('vector_field_3d_plotly.png')
        
        print("✅ Plotly 3D可视化完成")
        
    except ImportError as e:
        print(f"❌ Plotly不可用: {e}")
    
    print("\n🎯 3D可视化演示完成！")


if __name__ == "__main__":
    demo_3d_visualization()
