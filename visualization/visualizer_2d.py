"""
2D可视化模块
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union, Dict, Any
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. 2D visualization will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualization will be limited.")


class Visualizer2D(ABC):
    """2D可视化基类"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        初始化2D可视化器
        
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
    def plot_contour(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                    levels: Optional[int] = 20, cmap: str = 'viridis',
                    title: str = 'Contour Plot', **kwargs) -> Any:
        """绘制等值线图"""
        pass
    
    @abstractmethod
    def plot_vector_field(self, x: np.ndarray, y: np.ndarray, 
                         u: np.ndarray, v: np.ndarray,
                         title: str = 'Vector Field', **kwargs) -> Any:
        """绘制向量场"""
        pass
    
    @abstractmethod
    def plot_mesh(self, nodes: np.ndarray, elements: np.ndarray,
                  title: str = 'Mesh', **kwargs) -> Any:
        """绘制网格"""
        pass
    
    @abstractmethod
    def plot_scatter(self, x: np.ndarray, y: np.ndarray, 
                    c: Optional[np.ndarray] = None,
                    title: str = 'Scatter Plot', **kwargs) -> Any:
        """绘制散点图"""
        pass
    
    @abstractmethod
    def plot_heatmap(self, data: np.ndarray, x_labels: Optional[List] = None,
                    y_labels: Optional[List] = None,
                    title: str = 'Heatmap', **kwargs) -> Any:
        """绘制热力图"""
        pass
    
    @abstractmethod
    def show(self) -> None:
        """显示图形"""
        pass
    
    @abstractmethod
    def save(self, filename: str, **kwargs) -> None:
        """保存图形"""
        pass


class MatplotlibVisualizer2D(Visualizer2D):
    """基于Matplotlib的2D可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        super().__init__(figsize, dpi)
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for MatplotlibVisualizer2D")
    
    def plot_contour(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                    levels: Optional[int] = 20, cmap: str = 'viridis',
                    title: str = 'Contour Plot', **kwargs) -> plt.Figure:
        """绘制等值线图"""
        if self.current_figure is None:
            self.current_figure, self.current_ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 绘制等值线
        contour = self.current_ax.contourf(x, y, z, levels=levels, cmap=cmap, **kwargs)
        self.current_ax.contour(x, y, z, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        # 添加颜色条
        plt.colorbar(contour, ax=self.current_ax)
        
        # 设置标题和标签
        self.current_ax.set_title(title)
        self.current_ax.set_xlabel('X')
        self.current_ax.set_ylabel('Y')
        
        return self.current_figure
    
    def plot_vector_field(self, x: np.ndarray, y: np.ndarray, 
                         u: np.ndarray, v: np.ndarray,
                         title: str = 'Vector Field', **kwargs) -> plt.Figure:
        """绘制向量场"""
        if self.current_figure is None:
            self.current_figure, self.current_ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 计算向量模量用于颜色映射
        magnitude = np.sqrt(u**2 + v**2)
        
        # 绘制向量场
        skip = max(1, min(x.shape[0] // 20, y.shape[1] // 20))
        quiver = self.current_ax.quiver(x[::skip, ::skip], y[::skip, ::skip], 
                                       u[::skip, ::skip], v[::skip, ::skip],
                                       magnitude[::skip, ::skip], 
                                       cmap='viridis', **kwargs)
        
        # 添加颜色条
        plt.colorbar(quiver, ax=self.current_ax)
        
        # 设置标题和标签
        self.current_ax.set_title(title)
        self.current_ax.set_xlabel('X')
        self.current_ax.set_ylabel('Y')
        
        return self.current_figure
    
    def plot_mesh(self, nodes: np.ndarray, elements: np.ndarray,
                  title: str = 'Mesh', **kwargs) -> plt.Figure:
        """绘制网格"""
        if self.current_figure is None:
            self.current_figure, self.current_ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 绘制网格元素
        for element in elements:
            element_nodes = nodes[element]
            self.current_ax.plot(element_nodes[:, 0], element_nodes[:, 1], 'k-', linewidth=0.5)
        
        # 绘制节点
        self.current_ax.scatter(nodes[:, 0], nodes[:, 1], c='red', s=20, zorder=5)
        
        # 设置标题和标签
        self.current_ax.set_title(title)
        self.current_ax.set_xlabel('X')
        self.current_ax.set_ylabel('Y')
        self.current_ax.set_aspect('equal')
        
        return self.current_figure
    
    def plot_scatter(self, x: np.ndarray, y: np.ndarray, 
                    c: Optional[np.ndarray] = None,
                    title: str = 'Scatter Plot', **kwargs) -> plt.Figure:
        """绘制散点图"""
        if self.current_figure is None:
            self.current_figure, self.current_ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 绘制散点图
        scatter = self.current_ax.scatter(x, y, c=c, **kwargs)
        
        # 如果有颜色映射，添加颜色条
        if c is not None:
            plt.colorbar(scatter, ax=self.current_ax)
        
        # 设置标题和标签
        self.current_ax.set_title(title)
        self.current_ax.set_xlabel('X')
        self.current_ax.set_ylabel('Y')
        
        return self.current_figure
    
    def plot_heatmap(self, data: np.ndarray, x_labels: Optional[List] = None,
                    y_labels: Optional[List] = None,
                    title: str = 'Heatmap', **kwargs) -> plt.Figure:
        """绘制热力图"""
        if self.current_figure is None:
            self.current_figure, self.current_ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 绘制热力图
        im = self.current_ax.imshow(data, cmap='viridis', aspect='auto', **kwargs)
        
        # 设置标签
        if x_labels is not None:
            self.current_ax.set_xticks(range(len(x_labels)))
            self.current_ax.set_xticklabels(x_labels, rotation=45)
        
        if y_labels is not None:
            self.current_ax.set_yticks(range(len(y_labels)))
            self.current_ax.set_yticklabels(y_labels)
        
        # 添加颜色条
        plt.colorbar(im, ax=self.current_ax)
        
        # 设置标题
        self.current_ax.set_title(title)
        
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


class PlotlyVisualizer2D(Visualizer2D):
    """基于Plotly的2D可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        super().__init__(figsize, dpi)
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for PlotlyVisualizer2D")
    
    def plot_contour(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                    levels: Optional[int] = 20, cmap: str = 'viridis',
                    title: str = 'Contour Plot', **kwargs) -> go.Figure:
        """绘制等值线图"""
        fig = go.Figure()
        
        # 添加等值线图
        fig.add_trace(go.Contour(
            x=x[0, :] if x.ndim > 1 else x,
            y=y[:, 0] if y.ndim > 1 else y,
            z=z,
            colorscale=cmap,
            ncontours=levels,
            **kwargs
        ))
        
        # 设置布局
        fig.update_layout(
            title=title,
            xaxis_title='X',
            yaxis_title='Y',
            width=self.figsize[0] * 100,
            height=self.figsize[1] * 100
        )
        
        self.current_figure = fig
        return fig
    
    def plot_vector_field(self, x: np.ndarray, y: np.ndarray, 
                         u: np.ndarray, v: np.ndarray,
                         title: str = 'Vector Field', **kwargs) -> go.Figure:
        """绘制向量场"""
        fig = go.Figure()
        
        # 计算向量模量
        magnitude = np.sqrt(u**2 + v**2)
        
        # 添加向量场
        fig.add_trace(go.Streamtube(
            x=x.flatten(),
            y=y.flatten(),
            u=u.flatten(),
            v=v.flatten(),
            colorscale='viridis',
            **kwargs
        ))
        
        # 设置布局
        fig.update_layout(
            title=title,
            xaxis_title='X',
            yaxis_title='Y',
            width=self.figsize[0] * 100,
            height=self.figsize[1] * 100
        )
        
        self.current_figure = fig
        return fig
    
    def plot_mesh(self, nodes: np.ndarray, elements: np.ndarray,
                  title: str = 'Mesh', **kwargs) -> go.Figure:
        """绘制网格"""
        fig = go.Figure()
        
        # 绘制网格线
        for element in elements:
            element_nodes = nodes[element]
            # 闭合多边形
            element_nodes = np.vstack([element_nodes, element_nodes[0]])
            
            fig.add_trace(go.Scatter(
                x=element_nodes[:, 0],
                y=element_nodes[:, 1],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                **kwargs
            ))
        
        # 绘制节点
        fig.add_trace(go.Scatter(
            x=nodes[:, 0],
            y=nodes[:, 1],
            mode='markers',
            marker=dict(color='red', size=5),
            name='Nodes',
            **kwargs
        ))
        
        # 设置布局
        fig.update_layout(
            title=title,
            xaxis_title='X',
            yaxis_title='Y',
            width=self.figsize[0] * 100,
            height=self.figsize[1] * 100
        )
        
        self.current_figure = fig
        return fig
    
    def plot_scatter(self, x: np.ndarray, y: np.ndarray, 
                    c: Optional[np.ndarray] = None,
                    title: str = 'Scatter Plot', **kwargs) -> go.Figure:
        """绘制散点图"""
        fig = go.Figure()
        
        # 绘制散点图
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                color=c,
                colorscale='viridis' if c is not None else None,
                size=5
            ),
            **kwargs
        ))
        
        # 设置布局
        fig.update_layout(
            title=title,
            xaxis_title='X',
            yaxis_title='Y',
            width=self.figsize[0] * 100,
            height=self.figsize[1] * 100
        )
        
        self.current_figure = fig
        return fig
    
    def plot_heatmap(self, data: np.ndarray, x_labels: Optional[List] = None,
                    y_labels: Optional[List] = None,
                    title: str = 'Heatmap', **kwargs) -> go.Figure:
        """绘制热力图"""
        fig = go.Figure()
        
        # 绘制热力图
        fig.add_trace(go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale='viridis',
            **kwargs
        ))
        
        # 设置布局
        fig.update_layout(
            title=title,
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


def demo_2d_visualization():
    """演示2D可视化功能"""
    print("🎨 2D可视化演示")
    print("=" * 50)
    
    # 生成示例数据
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    
    # 向量场数据
    U = -Y
    V = X
    
    # 散点数据
    np.random.seed(42)
    scatter_x = np.random.randn(100)
    scatter_y = np.random.randn(100)
    scatter_c = np.random.rand(100)
    
    # 网格数据
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    elements = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    
    # 热力图数据
    heatmap_data = np.random.rand(10, 10)
    
    try:
        # Matplotlib可视化
        print("📊 使用Matplotlib进行2D可视化...")
        viz_mpl = MatplotlibVisualizer2D()
        
        # 等值线图
        viz_mpl.plot_contour(X, Y, Z, title='等值线图 - Matplotlib')
        viz_mpl.save('contour_mpl.png')
        
        # 向量场
        viz_mpl.plot_vector_field(X, Y, U, V, title='向量场 - Matplotlib')
        viz_mpl.save('vector_field_mpl.png')
        
        # 网格
        viz_mpl.plot_mesh(nodes, elements, title='网格 - Matplotlib')
        viz_mpl.save('mesh_mpl.png')
        
        # 散点图
        viz_mpl.plot_scatter(scatter_x, scatter_y, c=scatter_c, title='散点图 - Matplotlib')
        viz_mpl.save('scatter_mpl.png')
        
        # 热力图
        viz_mpl.plot_heatmap(heatmap_data, title='热力图 - Matplotlib')
        viz_mpl.save('heatmap_mpl.png')
        
        print("✅ Matplotlib 2D可视化完成")
        
    except ImportError as e:
        print(f"❌ Matplotlib不可用: {e}")
    
    try:
        # Plotly可视化
        print("\n📊 使用Plotly进行2D可视化...")
        viz_pl = PlotlyVisualizer2D()
        
        # 等值线图
        viz_pl.plot_contour(X, Y, Z, title='等值线图 - Plotly')
        viz_pl.save('contour_plotly.png')
        
        # 散点图
        viz_pl.plot_scatter(scatter_x, scatter_y, c=scatter_c, title='散点图 - Plotly')
        viz_pl.save('scatter_plotly.png')
        
        # 网格
        viz_pl.plot_mesh(nodes, elements, title='网格 - Plotly')
        viz_pl.save('mesh_plotly.png')
        
        # 热力图
        viz_pl.plot_heatmap(heatmap_data, title='热力图 - Plotly')
        viz_pl.save('heatmap_plotly.png')
        
        print("✅ Plotly 2D可视化完成")
        
    except ImportError as e:
        print(f"❌ Plotly不可用: {e}")
    
    print("\n🎯 2D可视化演示完成！")


if __name__ == "__main__":
    demo_2d_visualization()
