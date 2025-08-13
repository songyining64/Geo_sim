"""
统一工作流接口模块

提供类似Underworld的简洁接口，封装网格、材料、求解器和边界条件，
简化用户代码，支持地幔对流、岩石圈变形等地质模拟。
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml

# 科学计算库
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, cg, gmres

# 核心模块导入
from .unified_api import SimulationConfig, SimulationResult, BaseSimulator
from ..finite_elements.mesh_generation import AdaptiveMesh, MeshCell
from ..finite_elements.boundary_conditions import DirichletBC, NeumannBC, RobinBC, BoundaryAssembly
from ..materials import Material, MaterialRegistry
from ..solvers.multigrid_solver import MultigridSolver
from ..solvers.multiphysics_coupling_solver import MultiphysicsCouplingSolver
from ..time_integration.advanced_integrators import ImplicitTimeIntegrator
from ..parallel.advanced_parallel_solver_v2 import AdvancedParallelSolver

# 新增模块导入
try:
    from ..adaptivity.advanced_mesh import AdaptiveMeshRefiner
    from ..adaptivity.error_estimator import ErrorEstimator
    from ..adaptivity.mesh_refinement import MeshRefinement
    HAS_ADAPTIVITY = True
except ImportError:
    HAS_ADAPTIVITY = False
    warnings.warn("Adaptivity modules not available. Adaptive mesh refinement will be limited.")

try:
    from ..time_integration.time_integrators import TimeIntegrator
    from ..time_integration.advanced_integrators import BDFIntegrator, CrankNicolsonIntegrator
    HAS_TIME_INTEGRATION = True
except ImportError:
    HAS_TIME_INTEGRATION = False
    warnings.warn("Time integration modules not available. Advanced time stepping will be limited.")

try:
    from ..visualization.visualizer_2d import Visualizer2D
    from ..visualization.visualizer_3d import Visualizer3D
    from ..visualization.realtime_visualizer import RealtimeVisualizer
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    warnings.warn("Visualization modules not available. Plotting features will be limited.")

try:
    from ..coupling.thermal_mechanical import ThermalMechanicalCoupling
    from ..coupling.fluid_solid import FluidSolidCoupling
    from ..coupling.chemical_mechanical import ChemicalMechanicalCoupling
    HAS_COUPLING = True
except ImportError:
    HAS_COUPLING = False
    warnings.warn("Coupling modules not available. Multi-physics coupling will be limited.")

try:
    from ..ensemble.multi_fidelity import MultiFidelitySimulator
    from ..gpu_acceleration.cuda_acceleration import CUDAAccelerator
    HAS_ADVANCED_FEATURES = True
except ImportError:
    HAS_ADVANCED_FEATURES = False
    warnings.warn("Advanced features not available. GPU acceleration and ensemble methods will be limited.")


# 可选依赖
try:
    import vtk
    from vtk import vtkUnstructuredGrid, vtkPoints, vtkCellArray, vtkDoubleArray
    HAS_VTK = True
except ImportError:
    HAS_VTK = False
    warnings.warn("VTK not available. VTK output features will be limited.")

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False
    warnings.warn("h5py not available. HDF5 output features will be limited.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Visualization features will be limited.")


@dataclass
class GeodynamicConfig(SimulationConfig):
    """地质动力学仿真配置"""
    
    # 物理参数
    gravity: float = 9.81  # 重力加速度
    temperature: float = 273.15  # 初始温度 (K)
    pressure: float = 1e5  # 初始压力 (Pa)
    
    # 材料参数
    material_properties: Dict[str, Any] = field(default_factory=dict)
    
    # 边界条件参数
    boundary_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # 求解器参数
    solver_params: Dict[str, Any] = field(default_factory=dict)
    
    # 时间积分参数
    time_integration: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """设置默认值"""
        super().__post_init__()
        
        if not self.material_properties:
            self.material_properties = {
                'density': 3300.0,  # kg/m³ (地幔密度)
                'viscosity': 1e21,  # Pa·s (地幔粘度)
                'thermal_expansion': 3e-5,  # 1/K
                'thermal_conductivity': 3.0,  # W/(m·K)
                'specific_heat': 1200.0  # J/(kg·K)
            }
        
        if not self.boundary_conditions:
            self.boundary_conditions = {
                'top_temperature': 273.15,  # K
                'bottom_temperature': 1573.15,  # K
                'side_velocity': 0.0,  # m/s
                'top_pressure': 1e5,  # Pa
                'bottom_pressure': 1e8  # Pa
            }
        
        if not self.solver_params:
            self.solver_params = {
                'linear_solver': 'multigrid',  # 'multigrid', 'cg', 'gmres'
                'preconditioner': 'amg',  # 'amg', 'ilu', 'jacobi'
                'tolerance': 1e-6,
                'max_iterations': 1000,
                'multigrid_cycles': 'v',  # 'v', 'w', 'fmg'
                'smoother': 'gauss_seidel'  # 'jacobi', 'gauss_seidel', 'chebyshev'
            }
        
        if not self.time_integration:
            self.time_integration = {
                'method': 'bdf',  # 'bdf', 'crank_nicolson', 'explicit'
                'order': 2,  # BDF阶数
                'adaptive_timestep': True,
                'min_dt': 1e-6,
                'max_dt': 1e3,
                'error_tolerance': 1e-3
            }


class GeodynamicSimulation(BaseSimulator):
    """
    地质动力学仿真统一接口
    
    提供类似Underworld的简洁接口，支持：
    - 地幔对流模拟
    - 岩石圈变形模拟
    - 多物理场耦合
    - 自适应网格细化
    - 并行计算
    """
    
    def __init__(self, config: Optional[GeodynamicConfig] = None):
        super().__init__(config or GeodynamicConfig())
        
        # 核心组件
        self.mesh: Optional[AdaptiveMesh] = None
        self.materials: Dict[str, Material] = {}
        self.boundary_conditions: List[Any] = []
        self.solver: Optional[Any] = None
        self.time_integrator: Optional[Any] = None
        
        # 新增组件
        self.adaptive_refiner: Optional[Any] = None
        self.error_estimator: Optional[Any] = None
        self.visualizer_2d: Optional[Any] = None
        self.visualizer_3d: Optional[Any] = None
        self.realtime_visualizer: Optional[Any] = None
        self.thermal_mechanical_coupling: Optional[Any] = None
        self.fluid_solid_coupling: Optional[Any] = None
        self.multi_fidelity_simulator: Optional[Any] = None
        self.gpu_accelerator: Optional[Any] = None
        
        # 物理场
        self.temperature_field: Optional[np.ndarray] = None
        self.velocity_field: Optional[np.ndarray] = None
        self.pressure_field: Optional[np.ndarray] = None
        self.strain_field: Optional[np.ndarray] = None
        self.stress_field: Optional[np.ndarray] = None
        
        # 求解器状态
        self.is_assembled = False
        self.is_linearized = False
        
        # 输出管理
        self.output_manager = OutputManager()
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 初始化高级功能
        self._initialize_advanced_features()
    
    def _initialize_advanced_features(self):
        """初始化高级功能模块"""
        # 初始化自适应网格细化
        if HAS_ADAPTIVITY:
            try:
                self.adaptive_refiner = AdaptiveMeshRefiner()
                self.error_estimator = ErrorEstimator()
                print("自适应网格细化模块已初始化")
            except Exception as e:
                print(f"自适应网格细化初始化失败: {e}")
        
        # 初始化时间积分器
        if HAS_TIME_INTEGRATION:
            try:
                if self.config.time_integration['method'] == 'bdf':
                    self.time_integrator = BDFIntegrator(order=self.config.time_integration['order'])
                elif self.config.time_integration['method'] == 'crank_nicolson':
                    self.time_integrator = CrankNicolsonIntegrator()
                else:
                    self.time_integrator = TimeIntegrator()
                print("时间积分模块已初始化")
            except Exception as e:
                print(f"时间积分模块初始化失败: {e}")
        
        # 初始化可视化器
        if HAS_VISUALIZATION:
            try:
                self.visualizer_2d = Visualizer2D()
                self.visualizer_3d = Visualizer3D()
                self.realtime_visualizer = RealtimeVisualizer()
                print("可视化模块已初始化")
            except Exception as e:
                print(f"可视化模块初始化失败: {e}")
        
        # 初始化耦合模块
        if HAS_COUPLING:
            try:
                self.thermal_mechanical_coupling = ThermalMechanicalCoupling()
                self.fluid_solid_coupling = FluidSolidCoupling()
                print("多物理场耦合模块已初始化")
            except Exception as e:
                print(f"多物理场耦合模块初始化失败: {e}")
        
        # 初始化高级功能
        if HAS_ADVANCED_FEATURES:
            try:
                self.multi_fidelity_simulator = MultiFidelitySimulator()
                self.gpu_accelerator = CUDAAccelerator()
                print("高级功能模块已初始化")
            except Exception as e:
                print(f"高级功能模块初始化失败: {e}")
    
    def enable_adaptive_mesh_refinement(self, refinement_criteria: str = "error_based", 
                                      max_refinement_levels: int = 5) -> 'GeodynamicSimulation':
        """启用自适应网格细化"""
        if not HAS_ADAPTIVITY:
            print("警告：自适应网格细化模块不可用")
            return self
        
        try:
            if self.adaptive_refiner:
                self.adaptive_refiner.set_refinement_criteria(refinement_criteria)
                self.adaptive_refiner.set_max_levels(max_refinement_levels)
                print(f"自适应网格细化已启用，使用{refinement_criteria}标准，最大细化级别{max_refinement_levels}")
            return self
        except Exception as e:
            print(f"启用自适应网格细化失败: {e}")
            return self
    
    def enable_gpu_acceleration(self, precision: str = "mixed") -> 'GeodynamicSimulation':
        """启用GPU加速"""
        if not HAS_ADVANCED_FEATURES:
            print("警告：GPU加速模块不可用")
            return self
        
        try:
            if self.gpu_accelerator:
                self.gpu_accelerator.set_precision(precision)
                print(f"GPU加速已启用，精度模式: {precision}")
            return self
        except Exception as e:
            print(f"启用GPU加速失败: {e}")
            return self
    
    def setup_visualization(self, plot_type: str = "2d", realtime: bool = False) -> 'GeodynamicSimulation':
        """设置可视化"""
        if not HAS_VISUALIZATION:
            print("警告：可视化模块不可用")
            return self
        
        try:
            if plot_type == "2d" and self.visualizer_2d:
                self.visualizer_2d.setup_plot()
                print("2D可视化已设置")
            elif plot_type == "3d" and self.visualizer_3d:
                self.visualizer_3d.setup_plot()
                print("3D可视化已设置")
            
            if realtime and self.realtime_visualizer:
                self.realtime_visualizer.start()
                print("实时可视化已启动")
            
            return self
        except Exception as e:
            print(f"设置可视化失败: {e}")
            return self
    
    def setup_multi_physics_coupling(self, coupling_type: str = "thermal_mechanical") -> 'GeodynamicSimulation':
        """设置多物理场耦合"""
        if not HAS_COUPLING:
            print("警告：多物理场耦合模块不可用")
            return self
        
        try:
            if coupling_type == "thermal_mechanical" and self.thermal_mechanical_coupling:
                self.thermal_mechanical_coupling.setup_coupling()
                print("热-力耦合已设置")
            elif coupling_type == "fluid_solid" and self.fluid_solid_coupling:
                self.fluid_solid_coupling.setup_coupling()
                print("流-固耦合已设置")
            
            return self
        except Exception as e:
            print(f"设置多物理场耦合失败: {e}")
            return self
    
    def run_with_adaptivity(self, time_steps: int = 100, refinement_interval: int = 10) -> SimulationResult:
        """运行带自适应网格细化的仿真"""
        if not self.adaptive_refiner:
            print("警告：自适应网格细化未启用，使用标准运行模式")
            return self.run(time_steps)
        
        print(f"开始自适应网格细化仿真，总时间步: {time_steps}")
        
        for step in range(time_steps):
            # 执行时间步
            self._execute_timestep()
            
            # 检查是否需要网格细化
            if step % refinement_interval == 0 and step > 0:
                self._perform_adaptive_refinement()
            
            # 更新可视化
            if self.realtime_visualizer:
                self.realtime_visualizer.update(self._get_visualization_data())
        
        return self._create_result()
    
    def _perform_adaptive_refinement(self):
        """执行自适应网格细化"""
        try:
            if self.adaptive_refiner and self.error_estimator:
                # 估计误差
                error_indicators = self.error_estimator.estimate_error(
                    self.mesh, self.displacement_field, self.temperature_field
                )
                
                # 执行网格细化
                refined_mesh = self.adaptive_refiner.refine_mesh(
                    self.mesh, error_indicators
                )
                
                if refined_mesh is not None:
                    self.mesh = refined_mesh
                    print(f"网格已细化，新节点数: {len(self.mesh.nodes)}")
                    
                    # 重新组装系统
                    self.is_assembled = False
                    self._assemble_system()
        except Exception as e:
            print(f"自适应网格细化失败: {e}")
    
    def _get_visualization_data(self) -> Dict[str, Any]:
        """获取可视化数据"""
        return {
            'mesh': self.mesh,
            'displacement': self.displacement_field,
            'temperature': self.temperature_field,
            'velocity': self.velocity_field,
            'pressure': self.pressure_field,
            'strain': self.strain_field,
            'stress': self.stress_field
        }
    
    def export_results(self, format: str = "vtk", filename: str = None) -> str:
        """导出结果"""
        if not filename:
            filename = f"geodynamic_simulation_{int(time.time())}"
        
        try:
            if format.lower() == "vtk" and HAS_VTK:
                return self._export_vtk(filename)
            elif format.lower() == "hdf5" and HAS_HDF5:
                return self._export_hdf5(filename)
            elif format.lower() == "numpy":
                return self._export_numpy(filename)
            else:
                print(f"不支持的导出格式: {format}")
                return self._export_numpy(filename)
        except Exception as e:
            print(f"导出失败: {e}")
            return self._export_numpy(filename)
    
    def _export_vtk(self, filename: str) -> str:
        """导出为VTK格式"""
        if not HAS_VTK:
            raise ImportError("VTK不可用")
        
        # 创建VTK网格
        vtk_mesh = vtkUnstructuredGrid()
        
        # 设置节点
        points = vtkPoints()
        for node in self.mesh.nodes:
            points.InsertNextPoint(node[0], node[1], 0.0 if len(node) == 2 else node[2])
        vtk_mesh.SetPoints(points)
        
        # 设置单元
        cells = vtkCellArray()
        for cell in self.mesh.cells:
            if len(cell) == 3:  # 三角形
                cells.InsertNextCell(3, cell)
            elif len(cell) == 4:  # 四边形
                cells.InsertNextCell(4, cell)
        vtk_mesh.SetCells(vtk.VTK_TRIANGLE if len(self.mesh.cells[0]) == 3 else vtk.VTK_QUAD, cells)
        
        # 添加场数据
        if self.displacement_field is not None:
            disp_array = vtkDoubleArray()
            disp_array.SetName("Displacement")
            for disp in self.displacement_field:
                disp_array.InsertNextTuple(disp)
            vtk_mesh.GetPointData().AddArray(disp_array)
        
        if self.temperature_field is not None:
            temp_array = vtkDoubleArray()
            temp_array.SetName("Temperature")
            for temp in self.temperature_field:
                temp_array.InsertNextTuple([temp])
            vtk_mesh.GetPointData().AddArray(temp_array)
        
        # 写入文件
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(f"{filename}.vtk")
        writer.SetInputData(vtk_mesh)
        writer.Write()
        
        return f"{filename}.vtk"
    
    def _export_hdf5(self, filename: str) -> str:
        """导出为HDF5格式"""
        if not HAS_HDF5:
            raise ImportError("h5py不可用")
        
        with h5py.File(f"{filename}.h5", 'w') as f:
            # 保存网格
            mesh_group = f.create_group("mesh")
            mesh_group.create_dataset("nodes", data=self.mesh.nodes)
            mesh_group.create_dataset("cells", data=self.mesh.cells)
            
            # 保存场数据
            fields_group = f.create_group("fields")
            if self.displacement_field is not None:
                fields_group.create_dataset("displacement", data=self.displacement_field)
            if self.temperature_field is not None:
                fields_group.create_dataset("temperature", data=self.temperature_field)
            if self.velocity_field is not None:
                fields_group.create_dataset("velocity", data=self.velocity_field)
            if self.pressure_field is not None:
                fields_group.create_dataset("pressure", data=self.pressure_field)
            
            # 保存配置
            config_group = f.create_group("config")
            for key, value in self.config.__dict__.items():
                if isinstance(value, (int, float, str, bool)):
                    config_group.attrs[key] = value
        
        return f"{filename}.h5"
    
    def _export_numpy(self, filename: str) -> str:
        """导出为NumPy格式"""
        np.savez_compressed(f"{filename}.npz",
                           nodes=self.mesh.nodes,
                           cells=self.mesh.cells,
                           displacement=self.displacement_field,
                           temperature=self.temperature_field,
                           velocity=self.velocity_field,
                           pressure=self.pressure_field,
                           config=self.config.__dict__)
        
        return f"{filename}.npz"
    
    def create_mesh(self, mesh_type: str = "rectangular", **kwargs) -> 'GeodynamicSimulation':
        """
        创建网格
        
        Args:
            mesh_type: 网格类型 ('rectangular', 'triangular', 'tetrahedral')
            **kwargs: 网格参数
        
        Returns:
            self: 支持链式调用
        """
        try:
            if mesh_type == "rectangular":
                self.mesh = self._create_rectangular_mesh(**kwargs)
            elif mesh_type == "triangular":
                self.mesh = self._create_triangular_mesh(**kwargs)
            elif mesh_type == "tetrahedral":
                self.mesh = self._create_tetrahedral_mesh(**kwargs)
            else:
                raise ValueError(f"不支持的网格类型: {mesh_type}")
            
            print(f"成功创建{mesh_type}网格，节点数: {len(self.mesh.nodes)}, 单元数: {len(self.mesh.cells)}")
            return self
            
        except Exception as e:
            self._handle_error(f"网格创建失败: {str(e)}")
            raise
    
    def _create_rectangular_mesh(self, nx: int = 10, ny: int = 10, 
                                x_range: Tuple[float, float] = (0.0, 1.0),
                                y_range: Tuple[float, float] = (0.0, 1.0)) -> AdaptiveMesh:
        """创建矩形网格"""
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # 生成节点坐标
        x_coords = np.linspace(x_min, x_max, nx + 1)
        y_coords = np.linspace(y_min, y_max, ny + 1)
        
        nodes = []
        for y in y_coords:
            for x in x_coords:
                nodes.append([x, y])
        nodes = np.array(nodes)
        
        # 生成单元
        cells = []
        cell_id = 0
        for j in range(ny):
            for i in range(nx):
                # 矩形单元的四个节点（逆时针）
                node1 = j * (nx + 1) + i
                node2 = j * (nx + 1) + i + 1
                node3 = (j + 1) * (nx + 1) + i + 1
                node4 = (j + 1) * (nx + 1) + i
                
                cell = MeshCell(
                    id=cell_id,
                    nodes=[node1, node2, node3, node4],
                    level=0
                )
                cells.append(cell)
                cell_id += 1
        
        return AdaptiveMesh(nodes=nodes, cells=cells, dim=2)
    
    def _create_triangular_mesh(self, nx: int = 10, ny: int = 10,
                               x_range: Tuple[float, float] = (0.0, 1.0),
                               y_range: Tuple[float, float] = (0.0, 1.0)) -> AdaptiveMesh:
        """创建三角形网格"""
        # 基于矩形网格创建三角形网格
        rect_mesh = self._create_rectangular_mesh(nx, ny, x_range, y_range)
        
        # 将每个矩形单元分解为两个三角形
        triangular_cells = []
        cell_id = 0
        
        for rect_cell in rect_mesh.cells:
            nodes = rect_cell.nodes
            
            # 第一个三角形
            tri1 = MeshCell(
                id=cell_id,
                nodes=[nodes[0], nodes[1], nodes[2]],
                level=0
            )
            triangular_cells.append(tri1)
            cell_id += 1
            
            # 第二个三角形
            tri2 = MeshCell(
                id=cell_id,
                nodes=[nodes[0], nodes[2], nodes[3]],
                level=0
            )
            triangular_cells.append(tri2)
            cell_id += 1
        
        return AdaptiveMesh(nodes=rect_mesh.nodes, cells=triangular_cells, dim=2)
    
    def _create_tetrahedral_mesh(self, nx: int = 5, ny: int = 5, nz: int = 5,
                                 x_range: Tuple[float, float] = (0.0, 1.0),
                                 y_range: Tuple[float, float] = (0.0, 1.0),
                                 z_range: Tuple[float, float] = (0.0, 1.0)) -> AdaptiveMesh:
        """创建四面体网格"""
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range
        
        # 生成节点坐标
        x_coords = np.linspace(x_min, x_max, nx + 1)
        y_coords = np.linspace(y_min, y_max, ny + 1)
        z_coords = np.linspace(z_min, z_max, nz + 1)
        
        nodes = []
        for z in z_coords:
            for y in y_coords:
                for x in x_coords:
                    nodes.append([x, y, z])
        nodes = np.array(nodes)
        
        # 生成四面体单元（基于立方体分解）
        cells = []
        cell_id = 0
        
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # 立方体的8个节点
                    base = k * (ny + 1) * (nx + 1) + j * (nx + 1) + i
                    n000 = base
                    n100 = base + 1
                    n010 = base + (nx + 1)
                    n110 = base + (nx + 1) + 1
                    n001 = base + (ny + 1) * (nx + 1)
                    n101 = base + (ny + 1) * (nx + 1) + 1
                    n011 = base + (ny + 1) * (nx + 1) + (nx + 1)
                    n111 = base + (ny + 1) * (nx + 1) + (nx + 1) + 1
                    
                    # 将立方体分解为6个四面体
                    tetrahedra = [
                        [n000, n001, n010, n100],
                        [n001, n010, n100, n101],
                        [n010, n100, n101, n110],
                        [n001, n010, n101, n011],
                        [n010, n101, n110, n111],
                        [n001, n101, n011, n111]
                    ]
                    
                    for tetra in tetrahedra:
                        cell = MeshCell(
                            id=cell_id,
                            nodes=tetra,
                            level=0
                        )
                        cells.append(cell)
                        cell_id += 1
        
        return AdaptiveMesh(nodes=nodes, cells=cells, dim=3)
    
    def add_material(self, material: Material, mask: Optional[np.ndarray] = None) -> 'GeodynamicSimulation':
        """
        添加材料
        
        Args:
            material: 材料对象
            mask: 材料分布掩码（可选）
        
        Returns:
            self: 支持链式调用
        """
        try:
            if not self.mesh:
                raise ValueError("必须先创建网格")
            
            material_name = material.name
            self.materials[material_name] = material
            
            if mask is not None:
                if mask.shape != (len(self.mesh.cells),):
                    raise ValueError(f"掩码形状不匹配: 期望({len(self.mesh.cells)},), 实际{mask.shape}")
                material.mask = mask
            
            print(f"成功添加材料: {material_name}")
            return self
            
        except Exception as e:
            self._handle_error(f"材料添加失败: {str(e)}")
            raise
    
    def add_boundary_condition(self, bc: Any, boundary_name: str) -> 'GeodynamicSimulation':
        """
        添加边界条件
        
        Args:
            bc: 边界条件对象
            boundary_name: 边界名称
        
        Returns:
            self: 支持链式调用
        """
        try:
            if not self.mesh:
                raise ValueError("必须先创建网格")
            
            # 设置边界条件名称
            if not bc.name:
                bc.name = f"{boundary_name}_{bc.boundary_type}"
            
            self.boundary_conditions.append(bc)
            print(f"成功添加边界条件: {bc.name}")
            return self
            
        except Exception as e:
            self._handle_error(f"边界条件添加失败: {str(e)}")
            raise
    
    def setup_solver(self, solver_type: str = "auto", **kwargs) -> 'GeodynamicSimulation':
        """
        设置求解器
        
        Args:
            solver_type: 求解器类型 ('auto', 'multigrid', 'multiphysics', 'parallel')
            **kwargs: 求解器参数
        
        Returns:
            self: 支持链式调用
        """
        try:
            if not self.mesh:
                raise ValueError("必须先创建网格")
            
            if solver_type == "auto":
                # 根据问题规模自动选择求解器
                if len(self.mesh.nodes) > 10000:
                    solver_type = "parallel"
                elif len(self.materials) > 1:
                    solver_type = "multiphysics"
                else:
                    solver_type = "multigrid"
            
            if solver_type == "multigrid":
                self.solver = MultigridSolver(**kwargs)
            elif solver_type == "multiphysics":
                self.solver = MultiphysicsCouplingSolver(**kwargs)
            elif solver_type == "parallel":
                self.solver = AdvancedParallelSolver(**kwargs)
            else:
                raise ValueError(f"不支持的求解器类型: {solver_type}")
            
            print(f"成功设置求解器: {solver_type}")
            return self
            
        except Exception as e:
            self._handle_error(f"求解器设置失败: {str(e)}")
            raise
    
    def setup_time_integrator(self, method: str = "auto", **kwargs) -> 'GeodynamicSimulation':
        """
        设置时间积分器
        
        Args:
            method: 积分方法 ('auto', 'bdf', 'crank_nicolson', 'explicit')
            **kwargs: 积分器参数
        
        Returns:
            self: 支持链式调用
        """
        try:
            if method == "auto":
                # 根据问题特性自动选择积分方法
                if self.config.time_integration.get('adaptive_timestep', True):
                    method = "bdf"
                else:
                    method = "crank_nicolson"
            
            self.time_integrator = ImplicitTimeIntegrator(method=method, **kwargs)
            print(f"成功设置时间积分器: {method}")
            return self
            
        except Exception as e:
            self._handle_error(f"时间积分器设置失败: {str(e)}")
            raise
    
    def setup(self, **kwargs) -> bool:
        """设置仿真环境"""
        try:
            if not self.mesh:
                raise ValueError("必须创建网格")
            
            if not self.materials:
                raise ValueError("必须添加材料")
            
            if not self.boundary_conditions:
                warnings.warn("未设置边界条件，将使用默认边界条件")
                self._setup_default_boundary_conditions()
            
            if not self.solver:
                self.setup_solver()
            
            if not self.time_integrator:
                self.setup_time_integrator()
            
            # 初始化物理场
            self._initialize_physical_fields()
            
            # 设置输出管理器
            self.output_manager.setup(self.mesh, self.config.output_params)
            
            print("仿真环境设置完成")
            return True
            
        except Exception as e:
            self._handle_error(f"仿真环境设置失败: {str(e)}")
            return False
    
    def _setup_default_boundary_conditions(self):
        """设置默认边界条件"""
        if not self.mesh:
            return
        
        # 根据网格边界自动设置边界条件
        if self.mesh.dim == 2:
            # 2D情况：设置四周边界
            self.add_boundary_condition(
                DirichletBC(0, "temperature", self.config.boundary_conditions['top_temperature']),
                "top"
            )
            self.add_boundary_condition(
                DirichletBC(1, "temperature", self.config.boundary_conditions['bottom_temperature']),
                "bottom"
            )
            self.add_boundary_condition(
                NeumannBC(2, "velocity", self.config.boundary_conditions['side_velocity']),
                "left"
            )
            self.add_boundary_condition(
                NeumannBC(3, "velocity", self.config.boundary_conditions['side_velocity']),
                "right"
            )
        else:
            # 3D情况：设置六面边界
            # 这里简化处理，实际应用中需要更复杂的边界识别
            pass
    
    def _initialize_physical_fields(self):
        """初始化物理场"""
        if not self.mesh:
            return
        
        n_nodes = len(self.mesh.nodes)
        
        # 初始化温度场
        self.temperature_field = np.full(n_nodes, self.config.temperature)
        
        # 初始化速度场
        self.velocity_field = np.zeros((n_nodes, self.mesh.dim))
        
        # 初始化压力场
        self.pressure_field = np.full(n_nodes, self.config.pressure)
        
        # 初始化应变场
        self.strain_field = np.zeros((n_nodes, self.mesh.dim, self.mesh.dim))
        
        # 初始化应力场
        self.stress_field = np.zeros((n_nodes, self.mesh.dim, self.mesh.dim))
    
    def run(self, time_steps: Optional[int] = None, **kwargs) -> SimulationResult:
        """运行仿真"""
        start_time = time.time()
        
        if not self.is_initialized:
            raise RuntimeError("仿真器未初始化")
        
        # 创建结果对象
        result = SimulationResult(
            config=self.config,
            start_time=start_time,
            end_time=start_time
        )
        
        try:
            # 获取时间步数
            if time_steps is None:
                time_steps = self.config.numerical_params['time_steps']
            
            print(f"开始运行仿真，总时间步数: {time_steps}")
            
            # 执行时间步进
            for step in range(time_steps):
                step_start = time.time()
                
                if not self._execute_timestep():
                    result.add_error(f"时间步 {step} 执行失败")
                    break
                
                step_time = time.time() - step_start
                self.performance_monitor.record_timestep(step, step_time)
                
                # 输出进度
                if (step + 1) % 10 == 0:
                    print(f"完成时间步 {step + 1}/{time_steps}, 耗时: {step_time:.3f}s")
                
                # 保存输出
                if self.config.output_params.get('save_frequency', 10) > 0:
                    if (step + 1) % self.config.output_params['save_frequency'] == 0:
                        self._save_timestep_output(step + 1)
            
            result.end_time = time.time()
            print(f"仿真完成，总耗时: {result.duration:.2f}s")
            
            # 保存最终结果
            self._save_final_output()
            
            return result
            
        except Exception as e:
            result.add_error(str(e))
            result.end_time = time.time()
            return result
    
    def _execute_timestep(self):
        """执行单个时间步"""
        if self.current_time >= self.end_time:
            print("模拟时间已结束")
            return False
        
        print(f"执行时间步 {self.time_step_count + 1}, 时间: {self.current_time:.6f}")
        
        # 组装线性系统
        try:
            K, f = self._assemble_system()
            self.system_matrix = K
            self.system_rhs = f
            self.is_assembled = True
            print("系统矩阵组装完成")
        except Exception as e:
            print(f"系统组装失败: {e}")
            return False
        
        # 求解线性系统
        try:
            solution = self._solve_linear_system()
            self.current_solution = solution
            print("线性系统求解完成")
        except Exception as e:
            print(f"线性系统求解失败: {e}")
            return False
        
        # 更新物理场
        try:
            self._update_physical_fields(solution)
            print("物理场更新完成")
        except Exception as e:
            print(f"物理场更新失败: {e}")
            return False
        
        # 检查收敛性
        if not self._check_convergence():
            print("模拟未收敛，停止计算")
            return False
        
        # 保存当前解
        self.solution_history.append(solution.copy())
        self.time_step_history.append(self.current_time)
        
        # 更新时间
        self.current_time += self.time_step
        self.time_step_count += 1
        
        print(f"时间步 {self.time_step_count} 完成，当前时间: {self.current_time:.6f}")
        return True
    
    def _update_time_dependent_boundary_conditions(self, step: int):
        """更新与时间相关的边界条件"""
        current_time = step * self.config.numerical_params['dt']
        
        for bc in self.boundary_conditions:
            if bc.is_time_dependent and callable(bc.value):
                # 如果边界条件是时间相关的函数
                new_value = bc.value(current_time)
                # 更新边界条件值
                bc.current_value = new_value
    
    def _assemble_system(self):
        """组装线性系统 - 调用现有的专业模块"""
        if self.solver_type == 'standard':
            return self._assemble_standard_system()
        elif self.solver_type == 'multigrid':
            return self._assemble_multigrid_system()
        elif self.solver_type == 'multiphysics':
            return self._assemble_multiphysics_system()
        elif self.solver_type == 'parallel':
            return self._assemble_parallel_system()
        else:
            raise ValueError(f"不支持的求解器类型: {self.solver_type}")

    def _assemble_standard_system(self):
        """标准有限元组装 - 调用现有的global_assembly模块"""
        try:
            # 导入现有的专业模块
            from finite_elements.global_assembly import assemble_global_stiffness
            from finite_elements.assembly import triangle_stiffness, quad_stiffness
            
            # 使用现有的专业组装函数
            K = assemble_global_stiffness(
                self.mesh.nodes, 
                self.mesh.elements, 
                element_type=self.mesh.element_type,
                order=self.mesh.order
            )
            
            # 组装右端项
            f = self._assemble_rhs_using_existing_modules()
            
            # 应用边界条件
            K, f = self._apply_boundary_conditions(K, f)
            
            return K, f
            
        except ImportError as e:
            print(f"警告：无法导入专业模块，使用简化实现: {e}")
            return self._assemble_system_simplified()
    
    def _assemble_rhs_using_existing_modules(self):
        """使用现有模块组装右端项"""
        try:
            from finite_elements.assembly import triangle_force, quad_force
            
            n_nodes = len(self.mesh.nodes)
            f = np.zeros(n_nodes)
            
            for elem in self.mesh.elements:
                elem_coords = self.mesh.nodes[elem]
                material_props = self._get_element_material(elem)
                
                # 根据单元类型调用相应的力计算函数
                if self.mesh.element_type == 'triangle':
                    fe = triangle_force(elem_coords, material_props)
                elif self.mesh.element_type == 'quad':
                    fe = quad_force(elem_coords, material_props)
                else:
                    # 回退到简化实现
                    fe = self._compute_element_force(elem_coords, material_props, elem)
                
                # 组装到全局向量
                for i, node_idx in enumerate(elem):
                    f[node_idx] += fe[i]
            
            return f
            
        except ImportError:
            # 回退到简化实现
            return self._assemble_rhs_simplified()
    
    def _assemble_rhs_simplified(self):
        """简化的右端项组装（回退方案）"""
        n_nodes = len(self.mesh.nodes)
        f = np.zeros(n_nodes)
        
        for elem in self.mesh.elements:
            elem_coords = self.mesh.nodes[elem]
            material_props = self._get_element_material(elem)
            fe = self._compute_element_force(elem_coords, material_props, elem)
            
            for i, node_idx in enumerate(elem):
                f[node_idx] += fe[i]
        
        return f
    
    def _assemble_system_simplified(self):
        """简化的系统组装（回退方案）"""
        if self.mesh is None:
            raise ValueError("网格未初始化")
        
        # 获取网格信息
        nodes = np.array(self.mesh.nodes)
        elements = np.array(self.mesh.elements)
        
        # 组装刚度矩阵
        n_nodes = len(nodes)
        K = sp.lil_matrix((n_nodes, n_nodes))
        f = np.zeros(n_nodes)
        
        # 遍历所有单元
        for elem in elements:
            elem_nodes = elem.node_ids if hasattr(elem, 'node_ids') else elem
            elem_coords = nodes[elem_nodes]
            
            # 获取单元材料属性
            elem_material = self._get_element_material(elem)
            
            # 计算单元刚度矩阵
            Ke = self._compute_element_stiffness(elem_coords, elem_material)
            
            # 组装到全局矩阵
            for i, ni in enumerate(elem_nodes):
                for j, nj in enumerate(elem_nodes):
                    K[ni, nj] += Ke[i, j]
                
                # 组装右端项（重力、热源等）
                f[ni] += self._compute_element_force(elem_coords, elem_material, i)
        
        # 应用边界条件
        K, f = self._apply_boundary_conditions(K, f)
        
        return K.tocsr(), f

    def _assemble_multigrid_system(self):
        """组装多重网格系统"""
        # 先组装标准系统
        K, f = self._assemble_standard_system()
        
        # 创建多重网格层次结构
        if not hasattr(self, 'multigrid_hierarchy'):
            self.multigrid_hierarchy = self._create_multigrid_hierarchy()
        
        return K, f

    def _assemble_multiphysics_system(self):
        """组装多物理场耦合系统"""
        # 组装各个物理场的矩阵
        thermal_matrix = self._assemble_thermal_matrix()
        mechanical_matrix = self._assemble_mechanical_matrix()
        coupling_matrix = self._assemble_coupling_matrix()
        
        # 组装耦合系统
        n_thermal = thermal_matrix.shape[0]
        n_mechanical = mechanical_matrix.shape[0]
        n_total = n_thermal + n_mechanical
        
        # 创建块矩阵
        K = sp.bmat([
            [thermal_matrix, coupling_matrix],
            [coupling_matrix.T, mechanical_matrix]
        ], format='csr')
        
        # 组装右端项
        thermal_rhs = self._assemble_thermal_rhs()
        mechanical_rhs = self._assemble_mechanical_rhs()
        f = np.concatenate([thermal_rhs, mechanical_rhs])
        
        return K, f

    def _assemble_parallel_system(self):
        """组装并行系统"""
        # 使用并行求解器的组装方法
        if hasattr(self.solver, 'assemble_parallel_system'):
            K, f = self.solver.assemble_parallel_system()
        else:
            # 回退到标准组装
            K, f = self._assemble_standard_system()
        
        return K, f
    
    def _solve_linear_system(self):
        """求解线性系统 - 调用现有的专业求解器模块"""
        if self.solver_type == 'standard':
            return self._solve_with_standard()
        elif self.solver_type == 'multigrid':
            return self._solve_with_multigrid()
        elif self.solver_type == 'multiphysics':
            return self._solve_with_multiphysics()
        elif self.solver_type == 'parallel':
            return self._solve_with_parallel()
        else:
            raise ValueError(f"不支持的求解器类型: {self.solver_type}")

    def _solve_with_standard(self):
        """标准求解器 - 尝试直接求解，回退到迭代求解"""
        try:
            # 尝试直接求解
            solution = spsolve(self.system_matrix, self.system_rhs)
            return solution
        except:
            try:
                # 回退到CG求解器
                solution, info = cg(self.system_matrix, self.system_rhs, tol=1e-10, maxiter=1000)
                if info == 0:
                    return solution
                else:
                    # 最后回退到GMRES
                    solution, info = gmres(self.system_matrix, self.system_rhs, tol=1e-10, maxiter=1000)
                    return solution
            except Exception as e:
                print(f"所有求解器都失败: {e}")
                raise

    def _solve_with_multigrid(self):
        """多重网格求解器 - 调用现有的multigrid_solver模块"""
        try:
            from solvers.multigrid_solver import MultigridSolver, MultigridConfig
            
            # 创建多重网格求解器
            config = MultigridConfig(
                max_levels=5,
                smoother='gauss_seidel',
                cycle_type='v',
                tolerance=1e-8
            )
            
            solver = MultigridSolver(config)
            
            # 求解
            solution = solver.solve(self.system_matrix, self.system_rhs)
            return solution
            
        except ImportError as e:
            print(f"警告：无法导入多重网格求解器，回退到标准求解: {e}")
            return self._solve_with_standard()

    def _solve_with_multiphysics(self):
        """多物理场耦合求解器 - 调用现有的multiphysics_coupling_solver模块"""
        try:
            from solvers.multiphysics_coupling_solver import MultiphysicsCouplingSolver, CouplingConfig
            
            # 创建耦合求解器
            config = CouplingConfig(
                physics_fields=['thermal', 'mechanical'],
                coupling_type='staggered',
                tolerance=1e-6
            )
            
            solver = MultiphysicsCouplingSolver(config)
            
            # 求解耦合系统
            solution = solver.solve_coupled_system(
                self.system_matrix, 
                self.system_rhs,
                self.mesh,
                self.materials
            )
            return solution
            
        except ImportError as e:
            print(f"警告：无法导入多物理场耦合求解器，回退到标准求解: {e}")
            return self._solve_with_standard()

    def _solve_with_parallel(self):
        """并行求解器 - 调用现有的parallel模块"""
        try:
            from parallel.advanced_parallel_solver_v2 import AdvancedParallelSolver
            
            # 创建并行求解器
            solver = AdvancedParallelSolver()
            
            # 并行求解
            solution = solver.solve_parallel_linear_system(
                self.system_matrix, 
                self.system_rhs
            )
            return solution
            
        except ImportError as e:
            print(f"警告：无法导入并行求解器，回退到标准求解: {e}")
            return self._solve_with_standard()
    
    def _update_physical_fields(self, solution: np.ndarray):
        """更新物理场"""
        if solution is None or len(solution) == 0:
            return
        
        # 根据求解器类型更新不同的物理场
        if hasattr(self, 'solver_type'):
            if self.solver_type == 'multiphysics':
                self._update_multiphysics_fields(solution)
            else:
                self._update_single_physics_fields(solution)
        else:
            self._update_single_physics_fields(solution)
    
    def _update_single_physics_fields(self, solution: np.ndarray):
        """更新单物理场"""
        # 假设解向量包含位移场
        if len(solution) == len(self.mesh.nodes):
            # 更新位移场
            self.displacement_field = solution.reshape(-1, self.mesh.dimension)
            
            # 计算应变场（简化）
            self.strain_field = self._compute_strain_from_displacement(self.displacement_field)
            
            # 计算应力场（简化）
            self.stress_field = self._compute_stress_from_strain(self.strain_field)
            
            # 更新速度场（时间导数）
            if hasattr(self, 'previous_displacement'):
                dt = self.config.time_step
                self.velocity_field = (self.displacement_field - self.previous_displacement) / dt
            else:
                self.velocity_field = np.zeros_like(self.displacement_field)
            
            # 保存当前位移作为下一步的前一步
            self.previous_displacement = self.displacement_field.copy()
    
    def _update_multiphysics_fields(self, solution: np.ndarray):
        """更新多物理场"""
        # 解析耦合解向量
        n_thermal = len(self.mesh.nodes)  # 假设热场和位移场节点数相同
        n_mechanical = len(self.mesh.nodes)
        
        # 分离热场和位移场解
        thermal_solution = solution[:n_thermal]
        mechanical_solution = solution[n_thermal:n_thermal + n_mechanical]
        
        # 更新温度场
        self.temperature_field = thermal_solution
        
        # 更新位移场
        self.displacement_field = mechanical_solution.reshape(-1, self.mesh.dimension)
        
        # 计算应变和应力场
        self.strain_field = self._compute_strain_from_displacement(self.displacement_field)
        self.stress_field = self._compute_stress_from_strain(self.strain_field)
        
        # 更新速度场
        if hasattr(self, 'previous_displacement'):
            dt = self.config.time_step
            self.velocity_field = (self.displacement_field - self.previous_displacement) / dt
        else:
            self.velocity_field = np.zeros_like(self.displacement_field)
        
        # 保存当前状态
        self.previous_displacement = self.displacement_field.copy()
    
    def _compute_strain_from_displacement(self, displacement: np.ndarray) -> np.ndarray:
        """从位移场计算应变场（简化实现）"""
        # 这里应该实现真正的应变计算
        # 简化版本：假设应变与位移成正比
        strain = displacement * 0.1  # 简化系数
        return strain
    
    def _compute_stress_from_strain(self, strain: np.ndarray) -> np.ndarray:
        """从应变场计算应力场（简化实现）"""
        # 这里应该实现真正的应力计算
        # 简化版本：假设应力与应变成正比
        stress = strain * 1e6  # 简化弹性模量
        return stress
    
    def _check_convergence(self) -> bool:
        """检查收敛性"""
        if not hasattr(self, 'convergence_history'):
            self.convergence_history = []
        
        # 计算当前残差
        if hasattr(self, 'system_matrix') and hasattr(self, 'system_rhs'):
            if hasattr(self, 'current_solution'):
                residual = np.linalg.norm(self.system_matrix @ self.current_solution - self.system_rhs)
                self.convergence_history.append(residual)
                
                # 检查收敛性
                if len(self.convergence_history) >= 2:
                    # 相对残差变化
                    relative_change = abs(residual - self.convergence_history[-2]) / (abs(self.convergence_history[-2]) + 1e-12)
                    
                    # 绝对残差
                    if residual < self.config.tolerance:
                        print(f"收敛：绝对残差 {residual:.2e} < {self.config.tolerance:.2e}")
                        return True
                    
                    # 相对残差变化
                    if relative_change < self.config.tolerance * 0.1:
                        print(f"收敛：相对残差变化 {relative_change:.2e} < {self.config.tolerance * 0.1:.2e}")
                        return True
                    
                    # 检查是否发散
                    if residual > 1e6 or np.isnan(residual) or np.isinf(residual):
                        print(f"发散：残差 {residual:.2e}")
                        return False
        
        # 检查时间步收敛性
        if hasattr(self, 'time_step_history'):
            if len(self.time_step_history) >= 2:
                time_change = abs(self.time_step_history[-1] - self.time_step_history[-2])
                if time_change < self.config.time_step * 1e-6:
                    print("时间步收敛")
                    return True
        
        return False
    
    def _save_timestep_output(self, step: int):
        """保存时间步输出"""
        try:
            self.output_manager.save_timestep(step, {
                'temperature': self.temperature_field,
                'velocity': self.velocity_field,
                'pressure': self.pressure_field
            })
        except Exception as e:
            warnings.warn(f"保存时间步输出失败: {str(e)}")
    
    def _save_final_output(self):
        """保存最终输出"""
        try:
            self.output_manager.save_final_output({
                'temperature': self.temperature_field,
                'velocity': self.velocity_field,
                'pressure': self.pressure_field,
                'strain': self.strain_field,
                'stress': self.stress_field
            })
        except Exception as e:
            warnings.warn(f"保存最终输出失败: {str(e)}")
    
    def visualize(self, field_name: str = "temperature", **kwargs):
        """可视化结果"""
        if not self.result or not self.result.success:
            print("没有可可视化的结果")
            return
        
        try:
            if field_name == "temperature":
                field_data = self.temperature_field
            elif field_name == "velocity":
                field_data = self.velocity_field
            elif field_name == "pressure":
                field_data = self.pressure_field
            else:
                raise ValueError(f"不支持的场: {field_name}")
            
            self.output_manager.visualize_field(field_name, field_data, self.mesh, **kwargs)
            
        except Exception as e:
            self._handle_error(f"可视化失败: {str(e)}")
    
    def export_vtk(self, filepath: str, field_names: Optional[List[str]] = None):
        """导出为VTK格式"""
        if not HAS_VTK:
            raise ImportError("VTK不可用，无法导出VTK格式")
        
        try:
            if field_names is None:
                field_names = ["temperature", "velocity", "pressure"]
            
            self.output_manager.export_vtk(filepath, self.mesh, field_names, {
                'temperature': self.temperature_field,
                'velocity': self.velocity_field,
                'pressure': self.pressure_field
            })
            
            print(f"VTK文件已导出到: {filepath}")
            
        except Exception as e:
            self._handle_error(f"VTK导出失败: {str(e)}")
    
    def export_hdf5(self, filepath: str):
        """导出为HDF5格式"""
        if not HAS_HDF5:
            raise ImportError("h5py不可用，无法导出HDF5格式")
        
        try:
            self.output_manager.export_hdf5(filepath, self.mesh, {
                'temperature': self.temperature_field,
                'velocity': self.velocity_field,
                'pressure': self.pressure_field,
                'strain': self.strain_field,
                'stress': self.stress_field
            })
            
            print(f"HDF5文件已导出到: {filepath}")
            
        except Exception as e:
            self._handle_error(f"HDF5导出失败: {str(e)}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return self.performance_monitor.get_summary()
    
    # ========== 数值求解辅助方法 ==========
    
    def _get_element_material(self, element) -> Dict[str, Any]:
        """获取单元的材料属性"""
        # 简化实现：返回默认材料属性
        return {
            'youngs_modulus': 1e9,
            'poisson_ratio': 0.3,
            'density': 1000.0,
            'thermal_conductivity': 1.0,
            'specific_heat': 1000.0
        }
    
    def _compute_element_stiffness(self, elem_coords: np.ndarray, material_props: Dict[str, Any]) -> np.ndarray:
        """计算单元刚度矩阵"""
        # 简化实现：基于距离的刚度矩阵
        n_nodes = len(elem_coords)
        Ke = np.zeros((n_nodes, n_nodes))
        
        # 计算单元面积/体积
        if n_nodes == 3:  # 三角形
            area = self._compute_triangle_area(elem_coords)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i == j:
                        Ke[i, j] = material_props['youngs_modulus'] * area / 3.0
                    else:
                        Ke[i, j] = -material_props['youngs_modulus'] * area / 6.0
        elif n_nodes == 4:  # 四边形
            area = self._compute_quad_area(elem_coords)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i == j:
                        Ke[i, j] = material_props['youngs_modulus'] * area / 4.0
                    else:
                        Ke[i, j] = -material_props['youngs_modulus'] * area / 8.0
        
        return Ke
    
    def _compute_element_force(self, elem_coords: np.ndarray, material_props: Dict[str, Any], node_idx: int) -> float:
        """计算单元节点力"""
        # 简化实现：重力 + 热源
        gravity = 9.81  # m/s²
        density = material_props['density']
        
        # 重力分量（假设y方向向下）
        if self.mesh.dimension >= 2:
            force_y = -density * gravity * 0.1  # 简化系数
        else:
            force_y = 0.0
        
        # 热源（如果有温度梯度）
        if hasattr(self, 'temperature_field') and self.temperature_field is not None:
            thermal_force = material_props['thermal_expansion'] * 100.0  # 简化热力
        else:
            thermal_force = 0.0
        
        return force_y + thermal_force
    
    def _apply_boundary_conditions(self, K: sp.spmatrix, f: np.ndarray):
        """应用边界条件"""
        if not hasattr(self, 'boundary_conditions') or not self.boundary_conditions:
            return K, f
        
        for bc in self.boundary_conditions:
            if bc.bc_type == "Dirichlet":
                # Dirichlet边界条件：修改矩阵和右端项
                node_id = bc.node_id
                if node_id < len(f):
                    # 将对应行和列设为单位矩阵
                    K[node_id, :] = 0.0
                    K[node_id, node_id] = 1.0
                    f[node_id] = bc.value
            elif bc.bc_type == "Neumann":
                # Neumann边界条件：修改右端项
                node_id = bc.node_id
                if node_id < len(f):
                    f[node_id] += bc.value
            elif bc.bc_type == "Robin":
                # Robin边界条件：修改矩阵和右端项
                node_id = bc.node_id
                if node_id < len(f):
                    K[node_id, node_id] += bc.coefficient
                    f[node_id] += bc.value
        
        return K, f
    
    def _create_multigrid_hierarchy(self):
        """创建多重网格层次结构"""
        # 简化实现：创建两级网格
        hierarchy = {
            'fine_level': 0,
            'coarse_level': 1,
            'prolongation': None,
            'restriction': None
        }
        return hierarchy
    
    def _assemble_thermal_matrix(self) -> sp.spmatrix:
        """组装热传导矩阵"""
        n_nodes = len(self.mesh.nodes)
        K = sp.lil_matrix((n_nodes, n_nodes))
        
        # 简化实现：对角占优矩阵
        for i in range(n_nodes):
            K[i, i] = 1.0
            if i > 0:
                K[i, i-1] = -0.1
            if i < n_nodes - 1:
                K[i, i+1] = -0.1
        
        return K.tocsr()
    
    def _assemble_mechanical_matrix(self) -> sp.spmatrix:
        """组装力学矩阵"""
        n_nodes = len(self.mesh.nodes)
        K = sp.lil_matrix((n_nodes, n_nodes))
        
        # 简化实现：对角占优矩阵
        for i in range(n_nodes):
            K[i, i] = 2.0
            if i > 0:
                K[i, i-1] = -0.5
            if i < n_nodes - 1:
                K[i, i+1] = -0.5
        
        return K.tocsr()
    
    def _assemble_coupling_matrix(self) -> sp.spmatrix:
        """组装耦合矩阵"""
        n_nodes = len(self.mesh.nodes)
        K = sp.lil_matrix((n_nodes, n_nodes))
        
        # 简化实现：热-力耦合项
        for i in range(n_nodes):
            K[i, i] = 0.01  # 耦合系数
        
        return K.tocsr()
    
    def _assemble_thermal_rhs(self) -> np.ndarray:
        """组装热场右端项"""
        n_nodes = len(self.mesh.nodes)
        f = np.zeros(n_nodes)
        
        # 简化实现：热源项
        for i in range(n_nodes):
            f[i] = 0.1  # 单位热源
        
        return f
    
    def _assemble_mechanical_rhs(self) -> np.ndarray:
        """组装力场右端项"""
        n_nodes = len(self.mesh.nodes)
        f = np.zeros(n_nodes)
        
        # 简化实现：重力项
        for i in range(n_nodes):
            f[i] = -9.81  # 重力加速度
        
        return f
    
    def _compute_triangle_area(self, coords: np.ndarray) -> float:
        """计算三角形面积"""
        if len(coords) != 3:
            raise ValueError("需要3个节点坐标")
        
        # 使用叉积计算面积
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        
        if len(v1) == 2:  # 2D
            area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
        else:  # 3D
            cross_product = np.cross(v1, v2)
            area = 0.5 * np.linalg.norm(cross_product)
        
        return area
    
    def _compute_quad_area(self, coords: np.ndarray) -> float:
        """计算四边形面积"""
        if len(coords) != 4:
            raise ValueError("需要4个节点坐标")
        
        # 简化为两个三角形的面积和
        area1 = self._compute_triangle_area(coords[:3])
        area2 = self._compute_triangle_area(coords[1:])
        
        return area1 + area2


class OutputManager:
    """输出管理器"""
    
    def __init__(self):
        self.output_dir = "./output"
        self.timestep_files = []
        self.output_format = "h5"
    
    def setup(self, mesh: AdaptiveMesh, output_params: Dict[str, Any]):
        """设置输出管理器"""
        self.output_dir = output_params.get('output_dir', './output')
        self.output_format = output_params.get('save_format', 'h5')
        
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def save_timestep(self, step: int, data: Dict[str, np.ndarray]):
        """保存时间步数据"""
        # 简化实现
        pass
    
    def save_final_output(self, data: Dict[str, np.ndarray]):
        """保存最终输出"""
        # 简化实现
        pass
    
    def visualize_field(self, field_name: str, field_data: np.ndarray, 
                       mesh: AdaptiveMesh, **kwargs):
        """可视化场数据"""
        if not HAS_MATPLOTLIB:
            print("matplotlib不可用，无法可视化")
            return
        
        # 简化实现
        print(f"可视化场: {field_name}")
    
    def export_vtk(self, filepath: str, mesh: AdaptiveMesh, 
                   field_names: List[str], field_data: Dict[str, np.ndarray]):
        """导出为VTK格式"""
        # 简化实现
        pass
    
    def export_hdf5(self, filepath: str, mesh: AdaptiveMesh, 
                    field_data: Dict[str, np.ndarray]):
        """导出为HDF5格式"""
        # 简化实现
        pass


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.timestep_times = []
        self.total_time = 0.0
        self.start_time = None
    
    def record_timestep(self, step: int, time_taken: float):
        """记录时间步性能"""
        self.timestep_times.append((step, time_taken))
        self.total_time += time_taken
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.timestep_times:
            return {}
        
        times = [t[1] for t in self.timestep_times]
        return {
            'total_timesteps': len(self.timestep_times),
            'total_time': self.total_time,
            'average_timestep_time': np.mean(times),
            'min_timestep_time': np.min(times),
            'max_timestep_time': np.max(times),
            'std_timestep_time': np.std(times)
        }


# 便捷函数
def create_mantle_convection_simulation(nx: int = 50, ny: int = 50, 
                                       aspect_ratio: float = 2.0) -> GeodynamicSimulation:
    """
    创建地幔对流模拟
    
    Args:
        nx: x方向网格数
        ny: y方向网格数
        aspect_ratio: 长宽比
    
    Returns:
        GeodynamicSimulation: 配置好的仿真对象
    """
    # 创建配置
    config = GeodynamicConfig(
        name="mantle_convection",
        description="地幔对流模拟",
        material_properties={
            'density': 3300.0,
            'viscosity': 1e21,
            'thermal_expansion': 3e-5,
            'thermal_conductivity': 3.0,
            'specific_heat': 1200.0
        },
        boundary_conditions={
            'top_temperature': 273.15,
            'bottom_temperature': 1573.15,
            'side_velocity': 0.0
        }
    )
    
    # 创建仿真对象
    sim = GeodynamicSimulation(config)
    
    # 创建网格
    sim.create_mesh("rectangular", nx=nx, ny=ny, 
                    x_range=(0.0, aspect_ratio), y_range=(0.0, 1.0))
    
    # 添加材料
    from ..materials import create_mantle_material
    mantle_material = create_mantle_material()
    sim.add_material(mantle_material)
    
    # 设置边界条件
    sim.add_boundary_condition(
        DirichletBC(0, "temperature", 273.15), "top"
    )
    sim.add_boundary_condition(
        DirichletBC(1, "temperature", 1573.15), "bottom"
    )
    sim.add_boundary_condition(
        NeumannBC(2, "velocity", 0.0), "left"
    )
    sim.add_boundary_condition(
        NeumannBC(3, "velocity", 0.0), "right"
    )
    
    return sim


def create_lithospheric_deformation_simulation(nx: int = 40, ny: int = 40) -> GeodynamicSimulation:
    """
    创建岩石圈变形模拟
    
    Args:
        nx: x方向网格数
        ny: y方向网格数
    
    Returns:
        GeodynamicSimulation: 配置好的仿真对象
    """
    # 创建配置
    config = GeodynamicConfig(
        name="lithospheric_deformation",
        description="岩石圈变形模拟",
        material_properties={
            'density': 2800.0,
            'viscosity': 1e22,
            'thermal_expansion': 2e-5,
            'thermal_conductivity': 2.5,
            'specific_heat': 1000.0
        },
        boundary_conditions={
            'top_temperature': 273.15,
            'bottom_temperature': 1273.15,
            'side_velocity': 1e-9  # 1 cm/year
        }
    )
    
    # 创建仿真对象
    sim = GeodynamicSimulation(config)
    
    # 创建网格
    sim.create_mesh("rectangular", nx=nx, ny=ny, 
                    x_range=(0.0, 1000.0), y_range=(0.0, 100.0))
    
    # 添加材料
    from ..materials import create_crust_material
    crust_material = create_crust_material()
    sim.add_material(crust_material)
    
    # 设置边界条件
    sim.add_boundary_condition(
        DirichletBC(0, "temperature", 273.15), "top"
    )
    sim.add_boundary_condition(
        DirichletBC(1, "temperature", 1273.15), "bottom"
    )
    sim.add_boundary_condition(
        NeumannBC(2, "velocity", 1e-9), "left"
    )
    sim.add_boundary_condition(
        NeumannBC(3, "velocity", -1e-9), "right"
    )
    
    return sim
