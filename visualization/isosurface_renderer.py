"""
等值面渲染模块 - 专门用于等值面的提取和渲染

支持的功能：
- 等值面提取
- 等值面渲染
- 多等值面显示
- 等值面颜色映射
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
    from scipy.interpolate import griddata
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Isosurface rendering will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive isosurface rendering will be limited.")


class IsosurfaceRenderer(ABC):
    """等值面渲染基类"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        初始化等值面渲染器
        
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
    def extract_isosurface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          values: np.ndarray, level: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """提取等值面"""
        pass
    
    @abstractmethod
    def render_isosurface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         values: np.ndarray, level: float = 0.5,
                         colormap: str = 'viridis', title: str = 'Isosurface') -> Any:
        """渲染等值面"""
        pass
    
    @abstractmethod
    def render_multiple_isosurfaces(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                   values: np.ndarray, levels: List[float],
                                   colormap: str = 'viridis', title: str = 'Multiple Isosurfaces') -> Any:
        """渲染多个等值面"""
        pass
    
    @abstractmethod
    def show(self) -> None:
        """显示图形"""
        pass
    
    @abstractmethod
    def save(self, filename: str, **kwargs) -> None:
        """保存图形"""
        pass


class MatplotlibIsosurfaceRenderer(IsosurfaceRenderer):
    """基于Matplotlib的等值面渲染器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        super().__init__(figsize, dpi)
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for MatplotlibIsosurfaceRenderer")
    
    def extract_isosurface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          values: np.ndarray, level: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """提取等值面 - 基于Underworld2的设计"""
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
            
            # 使用Marching Cubes算法的简化版本
            surface_points = []
            
            # 获取数据范围
            x_min, x_max = X.min(), X.max()
            y_min, y_max = Y.min(), Y.max()
            z_min, z_max = Z.min(), Z.max()
            
            # 创建更密集的采样网格用于等值面提取
            nx, ny, nz = values.shape
            x_dense = np.linspace(x_min, x_max, nx)
            y_dense = np.linspace(y_min, y_max, ny)
            z_dense = np.linspace(z_min, z_max, nz)
            
            # 在多个z平面上提取等值线
            z_levels = np.linspace(z_min, z_max, min(20, nz))
            
            for z_level in z_levels:
                try:
                    # 找到最接近的z索引
                    z_idx = np.argmin(np.abs(z_dense - z_level))
                    if z_idx >= values.shape[2]:
                        continue
                    
                    # 提取2D切片
                    x_slice = X[:, :, z_idx]
                    y_slice = Y[:, :, z_idx]
                    values_slice = values[:, :, z_idx]
                    
                    # 找到等值线
                    contour = plt.contour(x_slice, y_slice, values_slice, levels=[level])
                    
                    # 检查是否有等值线
                    if len(contour.allsegs) > 0 and len(contour.allsegs[0]) > 0:
                        for seg in contour.allsegs[0]:
                            if len(seg) > 0:
                                seg_points = np.column_stack([seg, np.full(len(seg), z_level)])
                                surface_points.append(seg_points)
                except Exception as e:
                    # 如果等值线提取失败，跳过这个切片
                    continue
            
            if surface_points:
                all_points = np.vstack(surface_points)
                return all_points[:, 0], all_points[:, 1], all_points[:, 2]
            else:
                # 如果没有找到等值面，返回一些默认点
                print("⚠️ 未找到等值面，返回默认点")
                default_x = np.array([0.0])
                default_y = np.array([0.0])
                default_z = np.array([0.0])
                return default_x, default_y, default_z
                
        except Exception as e:
            print(f"⚠️ 等值面提取失败: {e}")
            # 返回默认点
            default_x = np.array([0.0])
            default_y = np.array([0.0])
            default_z = np.array([0.0])
            return default_x, default_y, default_z
    
    def render_isosurface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         values: np.ndarray, level: float = 0.5,
                         colormap: str = 'viridis', title: str = 'Isosurface') -> plt.Figure:
        """渲染等值面"""
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.current_ax = self.current_figure.add_subplot(111, projection='3d')
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 提取等值面
        surface_x, surface_y, surface_z = self.extract_isosurface(x, y, z, values, level)
        
        if len(surface_x) > 0 and surface_x.size > 0:
            try:
                # 绘制等值面点
                scatter = self.current_ax.scatter(surface_x, surface_y, surface_z, 
                                                c=surface_z, cmap=colormap, s=1, alpha=0.6)
                
                # 添加颜色条
                self.current_figure.colorbar(scatter, ax=self.current_ax)
            except Exception as e:
                print(f"⚠️ 等值面渲染失败: {e}")
                # 如果渲染失败，显示一个简单的点
                self.current_ax.scatter([0], [0], [0], c='red', s=100, alpha=0.8)
        else:
            # 如果没有等值面数据，显示一个简单的点
            self.current_ax.scatter([0], [0], [0], c='red', s=100, alpha=0.8)
        
        # 设置标题和标签
        self.current_ax.set_title(title)
        self.current_ax.set_xlabel('X')
        self.current_ax.set_ylabel('Y')
        self.current_ax.set_zlabel('Z')
        
        return self.current_figure
    
    def render_multiple_isosurfaces(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                   values: np.ndarray, levels: List[float],
                                   colormap: str = 'viridis', title: str = 'Multiple Isosurfaces') -> plt.Figure:
        """渲染多个等值面"""
        if self.current_figure is None:
            self.current_figure = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.current_ax = self.current_figure.add_subplot(111, projection='3d')
        
        # 清除当前图形
        self.current_ax.clear()
        
        # 为每个等值面分配颜色
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(levels)))
        
        for i, level in enumerate(levels):
            # 提取等值面
            surface_x, surface_y, surface_z = self.extract_isosurface(x, y, z, values, level)
            
            if len(surface_x) > 0:
                # 绘制等值面点
                self.current_ax.scatter(surface_x, surface_y, surface_z, 
                                      c=colors[i], s=1, alpha=0.6, 
                                      label=f'Level {level:.2f}')
        
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


class PlotlyIsosurfaceRenderer(IsosurfaceRenderer):
    """基于Plotly的等值面渲染器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        super().__init__(figsize, dpi)
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for PlotlyIsosurfaceRenderer")
    
    def extract_isosurface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                          values: np.ndarray, level: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """提取等值面"""
        # Plotly会自动处理等值面提取
        return x.flatten(), y.flatten(), z.flatten()
    
    def render_isosurface(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         values: np.ndarray, level: float = 0.5,
                         colormap: str = 'viridis', title: str = 'Isosurface') -> go.Figure:
        """渲染等值面"""
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
            name=f'Isosurface (level={level:.2f})'
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
    
    def render_multiple_isosurfaces(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                   values: np.ndarray, levels: List[float],
                                   colormap: str = 'viridis', title: str = 'Multiple Isosurfaces') -> go.Figure:
        """渲染多个等值面"""
        fig = go.Figure()
        
        # 为每个等值面分配颜色
        colors = px.colors.sample_colorscale(colormap, len(levels))
        
        for i, level in enumerate(levels):
            # 添加等值面
            fig.add_trace(go.Isosurface(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value=values.flatten(),
                isomin=level,
                isomax=level,
                colorscale=colormap,
                name=f'Isosurface (level={level:.2f})',
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


def demo_isosurface_rendering():
    """演示等值面渲染功能"""
    print("🎨 等值面渲染演示")
    print("=" * 50)
    
    # 生成示例3D数据
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(-2, 2, 20)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # 3D标量场
    values = np.sin(X) * np.cos(Y) * np.exp(-Z**2)
    
    try:
        # Matplotlib等值面渲染
        print("📊 使用Matplotlib进行等值面渲染...")
        renderer_mpl = MatplotlibIsosurfaceRenderer()
        
        # 单个等值面
        renderer_mpl.render_isosurface(X, Y, Z, values, level=0.5, title='等值面 - Matplotlib')
        renderer_mpl.save('isosurface_mpl.png')
        
        # 多个等值面
        levels = [0.3, 0.5, 0.7]
        renderer_mpl.render_multiple_isosurfaces(X, Y, Z, values, levels, title='多个等值面 - Matplotlib')
        renderer_mpl.save('multiple_isosurfaces_mpl.png')
        
        print("✅ Matplotlib等值面渲染完成")
        
    except ImportError as e:
        print(f"❌ Matplotlib不可用: {e}")
    
    try:
        # Plotly等值面渲染
        print("\n📊 使用Plotly进行等值面渲染...")
        renderer_pl = PlotlyIsosurfaceRenderer()
        
        # 单个等值面
        renderer_pl.render_isosurface(X, Y, Z, values, level=0.5, title='等值面 - Plotly')
        renderer_pl.save('isosurface_plotly.png')
        
        # 多个等值面
        levels = [0.3, 0.5, 0.7]
        renderer_pl.render_multiple_isosurfaces(X, Y, Z, values, levels, title='多个等值面 - Plotly')
        renderer_pl.save('multiple_isosurfaces_plotly.png')
        
        print("✅ Plotly等值面渲染完成")
        
    except ImportError as e:
        print(f"❌ Plotly不可用: {e}")
    
    print("\n🎯 等值面渲染演示完成！")


if __name__ == "__main__":
    demo_isosurface_rendering()
