"""
3D可视化测试脚本 - 测试Geo-Sim库的3D可视化功能
"""

import numpy as np
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def test_3d_visualization():
    """测试3D可视化功能"""
    print("🎨 3D可视化测试")
    print("=" * 50)
    
    # 生成示例3D数据
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    z = np.linspace(-2, 2, 20)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # 3D标量场
    values = np.sin(X) * np.cos(Y) * np.exp(-Z**2)
    
    # 点云数据
    np.random.seed(42)
    n_points = 1000
    point_x = np.random.randn(n_points)
    point_y = np.random.randn(n_points)
    point_z = np.random.randn(n_points)
    point_c = np.random.rand(n_points)
    
    try:
        # 测试2D可视化
        print("📊 测试2D可视化...")
        from visualization import MatplotlibVisualizer2D
        
        viz_2d = MatplotlibVisualizer2D()
        viz_2d.plot_contour(X[:, :, 10], Y[:, :, 10], values[:, :, 10], title='2D等值线图')
        viz_2d.save('test_2d_contour.png')
        print("✅ 2D可视化测试完成")
        
    except Exception as e:
        print(f"❌ 2D可视化测试失败: {e}")
    
    try:
        # 测试3D可视化
        print("\n📊 测试3D可视化...")
        from visualization import MatplotlibVisualizer3D
        
        viz_3d = MatplotlibVisualizer3D()
        viz_3d.plot_point_cloud(point_x, point_y, point_z, c=point_c, title='3D点云')
        viz_3d.save('test_3d_point_cloud.png')
        print("✅ 3D可视化测试完成")
        
    except Exception as e:
        print(f"❌ 3D可视化测试失败: {e}")
    
    try:
        # 测试体渲染
        print("\n📊 测试体渲染...")
        from visualization import MatplotlibVolumeRenderer
        
        renderer = MatplotlibVolumeRenderer()
        renderer.render_volume(X, Y, Z, values, opacity=0.3, title='体渲染')
        renderer.save('test_volume_rendering.png')
        print("✅ 体渲染测试完成")
        
    except Exception as e:
        print(f"❌ 体渲染测试失败: {e}")
    
    try:
        # 测试等值面渲染
        print("\n📊 测试等值面渲染...")
        from visualization import MatplotlibIsosurfaceRenderer
        
        renderer = MatplotlibIsosurfaceRenderer()
        renderer.render_isosurface(X, Y, Z, values, level=0.5, title='等值面')
        renderer.save('test_isosurface.png')
        print("✅ 等值面渲染测试完成")
        
    except Exception as e:
        print(f"❌ 等值面渲染测试失败: {e}")
    
    try:
        # 测试切片渲染
        print("\n📊 测试切片渲染...")
        from visualization import MatplotlibSliceRenderer
        
        renderer = MatplotlibSliceRenderer()
        renderer.render_slice(X, Y, Z, values, slice_axis='z', slice_value=0.0, title='切片')
        renderer.save('test_slice.png')
        print("✅ 切片渲染测试完成")
        
    except Exception as e:
        print(f"❌ 切片渲染测试失败: {e}")
    
    print("\n🎯 3D可视化测试完成！")
    print("=" * 50)
    print("所有生成的图片已保存到当前目录")


if __name__ == "__main__":
    test_3d_visualization()
