"""
3D可视化演示脚本 - 展示Geo-Sim库的3D可视化功能

这个脚本演示了以下功能：
1. 2D可视化（等值线、向量场、网格、散点图、热力图）
2. 3D可视化（体渲染、等值面、切片、点云、向量场、网格）
3. 实时可视化（交互式显示、动画生成）
4. 体渲染（透明度控制、颜色映射）
5. 等值面渲染（等值面提取、多等值面显示）
6. 切片渲染（切片提取、多切片显示）
"""

import numpy as np
import time
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def demo_2d_visualization():
    """演示2D可视化功能"""
    print("🎨 2D可视化演示")
    print("=" * 50)
    
    try:
        from visualization import MatplotlibVisualizer2D, PlotlyVisualizer2D
        
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


def demo_3d_visualization():
    """演示3D可视化功能"""
    print("\n🎨 3D可视化演示")
    print("=" * 50)
    
    try:
        from visualization import MatplotlibVisualizer3D, PlotlyVisualizer3D
        
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


def demo_volume_rendering():
    """演示体渲染功能"""
    print("\n🎨 体渲染演示")
    print("=" * 50)
    
    try:
        from visualization import MatplotlibVolumeRenderer, PlotlyVolumeRenderer
        
        # 生成示例3D数据
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        z = np.linspace(-2, 2, 20)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # 3D标量场
        values = np.sin(X) * np.cos(Y) * np.exp(-Z**2)
        
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


def demo_isosurface_rendering():
    """演示等值面渲染功能"""
    print("\n🎨 等值面渲染演示")
    print("=" * 50)
    
    try:
        from visualization import MatplotlibIsosurfaceRenderer, PlotlyIsosurfaceRenderer
        
        # 生成示例3D数据
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        z = np.linspace(-2, 2, 20)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # 3D标量场
        values = np.sin(X) * np.cos(Y) * np.exp(-Z**2)
        
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


def demo_slice_rendering():
    """演示切片渲染功能"""
    print("\n🎨 切片渲染演示")
    print("=" * 50)
    
    try:
        from visualization import MatplotlibSliceRenderer, PlotlySliceRenderer
        
        # 生成示例3D数据
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        z = np.linspace(-2, 2, 20)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # 3D标量场
        values = np.sin(X) * np.cos(Y) * np.exp(-Z**2)
        
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


def demo_realtime_visualization():
    """演示实时可视化功能"""
    print("\n🎨 实时可视化演示")
    print("=" * 50)
    
    try:
        from visualization import MatplotlibRealTimeVisualizer, PlotlyRealTimeVisualizer
        
        # 生成示例数据生成器
        def generate_scatter_data():
            """生成散点数据"""
            np.random.seed(int(time.time()))
            n_points = 50
            x = np.random.randn(n_points)
            y = np.random.randn(n_points)
            c = np.random.rand(n_points)
            
            return {
                'scatter': {'x': x, 'y': y, 'c': c},
                'title': f'实时散点图 - {time.strftime("%H:%M:%S")}'
            }
        
        def generate_line_data():
            """生成线图数据"""
            t = time.time()
            x = np.linspace(0, 2*np.pi, 100)
            y = np.sin(x + t)
            
            return {
                'line': {'x': x, 'y': y},
                'title': f'实时线图 - {time.strftime("%H:%M:%S")}'
            }
        
        def generate_contour_data():
            """生成等值线数据"""
            t = time.time()
            x = np.linspace(-2, 2, 50)
            y = np.linspace(-2, 2, 50)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X + t) * np.cos(Y + t)
            
            return {
                'contour': {'x': X, 'y': Y, 'z': Z},
                'title': f'实时等值线图 - {time.strftime("%H:%M:%S")}'
            }
        
        # Matplotlib实时可视化
        print("📊 使用Matplotlib进行实时可视化...")
        viz_mpl = MatplotlibRealTimeVisualizer()
        
        # 演示散点图实时更新
        print("演示散点图实时更新（3秒）...")
        viz_mpl.start_realtime_update(generate_scatter_data)
        time.sleep(3)
        viz_mpl.stop_realtime_update()
        
        # 演示线图实时更新
        print("演示线图实时更新（3秒）...")
        viz_mpl.start_realtime_update(generate_line_data)
        time.sleep(3)
        viz_mpl.stop_realtime_update()
        
        print("✅ Matplotlib实时可视化完成")
        
    except ImportError as e:
        print(f"❌ Matplotlib不可用: {e}")
    
    try:
        # Plotly实时可视化
        print("\n📊 使用Plotly进行实时可视化...")
        viz_pl = PlotlyRealTimeVisualizer()
        
        # 演示等值线图实时更新
        print("演示等值线图实时更新（3秒）...")
        viz_pl.start_realtime_update(generate_contour_data)
        time.sleep(3)
        viz_pl.stop_realtime_update()
        
        print("✅ Plotly实时可视化完成")
        
    except ImportError as e:
        print(f"❌ Plotly不可用: {e}")


def main():
    """主函数"""
    print("🚀 Geo-Sim 3D可视化演示")
    print("=" * 60)
    print("这个演示展示了Geo-Sim库的完整3D可视化功能")
    print("包括2D可视化、3D可视化、实时可视化、体渲染、等值面渲染和切片渲染")
    print("=" * 60)
    
    # 演示2D可视化
    demo_2d_visualization()
    
    # 演示3D可视化
    demo_3d_visualization()
    
    # 演示体渲染
    demo_volume_rendering()
    
    # 演示等值面渲染
    demo_isosurface_rendering()
    
    # 演示切片渲染
    demo_slice_rendering()
    
    # 演示实时可视化
    demo_realtime_visualization()
    
    print("\n🎯 3D可视化演示完成！")
    print("=" * 60)
    print("所有生成的图片已保存到当前目录")
    print("您可以使用以下命令查看生成的图片：")
    print("  - Windows: start *.png")
    print("  - Linux/Mac: open *.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
