# GeoSim API 参考文档

## 目录

1. [并行计算模块](#并行计算模块)
2. [多物理场耦合模块](#多物理场耦合模块)
3. [材料模型模块](#材料模型模块)
4. [有限元模块](#有限元模块)
5. [GPU加速模块](#gpu加速模块)
6. [可视化模块](#可视化模块)
7. [时间积分模块](#时间积分模块)
8. [自适应模块](#自适应模块)

## 并行计算模块

### parallel.advanced_parallel_solver

#### ParallelConfig

并行求解器配置类。

```python
class ParallelConfig:
    def __init__(self, 
                 solver_type: str = 'cg',
                 max_iterations: int = 1000,
                 tolerance: float = 1e-8,
                 communication_optimization: bool = True,
                 load_balancing: bool = True,
                 use_nonblocking: bool = True,
                 buffer_size: int = 1024,
                 overlap_communication: bool = True,
                 adaptive_tolerance: bool = True,
                 preconditioner: str = 'jacobi',
                 restart_frequency: int = 50,
                 max_subdomain_iterations: int = 10)
```

**参数：**
- `solver_type`: 求解器类型 ('cg', 'gmres', 'schwarz')
- `max_iterations`: 最大迭代次数
- `tolerance`: 收敛容差
- `communication_optimization`: 是否启用通信优化
- `load_balancing`: 是否启用负载均衡
- `use_nonblocking`: 是否使用非阻塞通信
- `buffer_size`: 通信缓冲区大小
- `overlap_communication`: 是否启用通信计算重叠
- `adaptive_tolerance`: 是否启用自适应容差
- `preconditioner`: 预处理器类型 ('jacobi', 'ilu')
- `restart_frequency`: GMRES重启频率
- `max_subdomain_iterations`: 子域最大迭代次数

#### PerformanceStats

性能统计类。

```python
class PerformanceStats:
    def __init__(self):
        self.solve_time: float = 0.0
        self.iterations: int = 0
        self.residual_norm: float = 0.0
        self.communication_time: float = 0.0
        self.computation_time: float = 0.0
        self.memory_usage: float = 0.0
        self.cache_hit_rate: float = 0.0
        self.parallel_efficiency: float = 0.0
```

#### AdvancedParallelSolver

高级并行求解器基类。

```python
class AdvancedParallelSolver(ABC):
    def __init__(self, config: ParallelConfig = None)
    
    @abstractmethod
    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        """求解线性系统"""
        pass
    
    def optimize_communication(self, partition_info: Dict) -> Dict:
        """优化通信模式"""
        
    def balance_load(self, partition_info: Dict, load_weights: np.ndarray) -> Dict:
        """负载均衡"""
        
    def get_performance_stats(self) -> PerformanceStats:
        """获取性能统计"""
```

#### CommunicationOptimizer

通信优化器。

```python
class CommunicationOptimizer:
    def __init__(self, comm=None)
    
    def optimize_communication_schedule(self, partition_info: Dict) -> Dict:
        """优化通信调度"""
        
    def _analyze_communication_pattern(self, partition_info: Dict) -> Dict:
        """分析通信模式"""
        
    def _generate_optimized_schedule(self, pattern: Dict) -> Dict:
        """生成优化调度"""
        
    def _preallocate_buffers(self, schedule: Dict):
        """预分配缓冲区"""
```

#### LoadBalancer

负载均衡器。

```python
class LoadBalancer:
    def __init__(self, comm=None, balance_threshold: float = 0.1)
    
    def balance_load(self, partition_info: Dict, load_weights: np.ndarray) -> Dict:
        """负载均衡"""
        
    def _compute_partition_loads(self, partition_info: Dict, load_weights: np.ndarray) -> Dict:
        """计算分区负载"""
        
    def _compute_load_imbalance(self, loads: Dict) -> float:
        """计算负载不平衡度"""
        
    def _redistribute_load(self, partition_info: Dict, current_loads: Dict) -> Dict:
        """重新分配负载"""
```

#### PerformanceMonitor

性能监控器。

```python
class PerformanceMonitor:
    def __init__(self)
    
    def start_timer(self, name: str):
        """开始计时"""
        
    def end_timer(self, name: str):
        """结束计时"""
        
    def get_metrics(self) -> Dict:
        """获取性能指标"""
```

#### 预处理器

##### JacobiPreconditioner

```python
class JacobiPreconditioner:
    def setup(self, A: sp.spmatrix):
        """设置预处理器"""
        
    def apply(self, x: np.ndarray) -> np.ndarray:
        """应用预处理器"""
```

##### ILUPreconditioner

```python
class ILUPreconditioner:
    def setup(self, A: sp.spmatrix):
        """设置预处理器"""
        
    def apply(self, x: np.ndarray) -> np.ndarray:
        """应用预处理器"""
```

#### 并行求解器

##### ParallelCGSolver

```python
class ParallelCGSolver(AdvancedParallelSolver):
    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        """并行CG求解"""
```

##### ParallelGMRESSolver

```python
class ParallelGMRESSolver(AdvancedParallelSolver):
    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        """并行GMRES求解"""
```

##### ParallelSchwarzSolver

```python
class ParallelSchwarzSolver(AdvancedParallelSolver):
    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        """并行Schwarz求解"""
```

#### 工厂函数

```python
def create_parallel_solver(solver_type: str, config: ParallelConfig) -> AdvancedParallelSolver:
    """创建并行求解器"""
```

### parallel.domain_decomposition

#### DomainDecomposer

域分解器基类。

```python
class DomainDecomposer(ABC):
    def __init__(self, method: str = 'metis')
    
    @abstractmethod
    def decompose_domain(self, mesh_data: Dict, num_partitions: int) -> Dict:
        """域分解"""
        pass
    
    def handle_communication(self, local_data: np.ndarray, partition_info: Dict) -> np.ndarray:
        """处理通信"""
        
    def balance_load(self, partition_info: Dict, load_weights: np.ndarray) -> Dict:
        """负载均衡"""
```

#### METISDecomposer

```python
class METISDecomposer(DomainDecomposer):
    def decompose_domain(self, mesh_data: Dict, num_partitions: int) -> Dict:
        """使用METIS进行域分解"""
```

#### RecursiveBisectionDecomposer

```python
class RecursiveBisectionDecomposer(DomainDecomposer):
    def decompose_domain(self, mesh_data: Dict, num_partitions: int) -> Dict:
        """使用递归二分进行域分解"""
```

#### 工厂函数

```python
def create_domain_decomposer(method: str = 'metis') -> DomainDecomposer:
    """创建域分解器"""
```

## 多物理场耦合模块

### coupling.thermal_mechanical

#### CouplingConfig

热-力学耦合配置类。

```python
class CouplingConfig:
    def __init__(self,
                 solver_type: str = 'iterative',
                 max_iterations: int = 10,
                 tolerance: float = 1e-6,
                 relaxation_factor: float = 1.0,
                 adaptive_timestep: bool = True,
                 min_timestep: float = 1e-6,
                 max_timestep: float = 1e-3,
                 cfl_number: float = 0.5,
                 parallel_solver: bool = True,
                 preconditioner: str = 'jacobi')
```

#### CouplingState

耦合状态类。

```python
class CouplingState:
    def __init__(self,
                 temperature: np.ndarray,
                 displacement: np.ndarray,
                 stress: np.ndarray,
                 strain: np.ndarray,
                 thermal_strain: np.ndarray,
                 mechanical_heat: np.ndarray,
                 time: float = 0.0,
                 iteration: int = 0,
                 residual_norm: float = 0.0)
```

#### ThermoMechanicalCoupling

热-力学耦合求解器。

```python
class ThermoMechanicalCoupling:
    def __init__(self, mesh, thermal_conductivity: float = 3.0,
                 heat_capacity: float = 1000.0, density: float = 2700.0,
                 thermal_expansion: float = 2.4e-5, young_modulus: float = 70e9,
                 poisson_ratio: float = 0.3, coupling_parameter: float = 1.0,
                 config: CouplingConfig = None)
    
    def solve_coupled_system(self, 
                           initial_temperature: np.ndarray,
                           initial_displacement: np.ndarray,
                           boundary_conditions: Dict,
                           time_steps: int,
                           dt: float,
                           heat_source: Optional[Callable] = None,
                           body_force: Optional[Callable] = None) -> List[CouplingState]:
        """求解耦合系统"""
        
    def compute_thermal_stress(self, temperature: np.ndarray) -> np.ndarray:
        """计算热应力"""
        
    def compute_mechanical_heating(self, stress: np.ndarray, strain_rate: np.ndarray) -> np.ndarray:
        """计算机械功产热"""
        
    def visualize_coupling_results(self, solution_history: List[CouplingState]):
        """可视化耦合结果"""
        
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
```

### coupling.fluid_solid

#### FSIConfig

流体-固体耦合配置类。

```python
class FSIConfig:
    def __init__(self,
                 solver_type: str = 'partitioned',
                 max_iterations: int = 10,
                 tolerance: float = 1e-6,
                 relaxation_factor: float = 0.5,
                 adaptive_timestep: bool = True,
                 min_timestep: float = 1e-6,
                 max_timestep: float = 1e-3,
                 cfl_number: float = 0.5,
                 interface_tracking: bool = True,
                 mesh_deformation: bool = True,
                 parallel_solver: bool = True,
                 preconditioner: str = 'jacobi')
```

#### FluidSolidState

流体-固体状态类。

```python
class FluidSolidState:
    def __init__(self,
                 fluid_velocity: np.ndarray,
                 fluid_pressure: np.ndarray,
                 fluid_temperature: np.ndarray,
                 solid_displacement: np.ndarray,
                 solid_velocity: np.ndarray,
                 solid_stress: np.ndarray,
                 interface_force: np.ndarray,
                 interface_displacement: np.ndarray,
                 mesh_deformation: np.ndarray,
                 time: float = 0.0,
                 iteration: int = 0,
                 residual_norm: float = 0.0,
                 convergence_status: bool = False)
```

#### FluidSolver

流体求解器基类。

```python
class FluidSolver(ABC):
    def __init__(self, mesh, fluid_properties: Dict)
    
    @abstractmethod
    def solve_fluid_step(self, current_state: FluidSolidState, dt: float) -> np.ndarray:
        """求解流体步骤"""
        pass
    
    def compute_interface_force(self, fluid_state: np.ndarray) -> np.ndarray:
        """计算界面力"""
        
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
```

#### SolidSolver

固体求解器基类。

```python
class SolidSolver(ABC):
    def __init__(self, mesh, solid_properties: Dict)
    
    @abstractmethod
    def solve_solid_step(self, current_state: FluidSolidState, dt: float) -> np.ndarray:
        """求解固体步骤"""
        pass
    
    def compute_interface_displacement(self, solid_state: np.ndarray) -> np.ndarray:
        """计算界面位移"""
        
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
```

#### NavierStokesSolver

Navier-Stokes求解器。

```python
class NavierStokesSolver(FluidSolver):
    def __init__(self, mesh, fluid_properties: Dict)
    
    def solve_fluid_step(self, current_state: FluidSolidState, dt: float) -> np.ndarray:
        """求解Navier-Stokes方程"""
        
    def _predict_velocity(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """预测速度"""
        
    def _solve_pressure_correction(self, predicted_velocity: np.ndarray, dt: float) -> np.ndarray:
        """求解压力修正"""
        
    def _solve_energy_equation(self, temperature: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
        """求解能量方程"""
```

#### ElasticSolidSolver

弹性固体求解器。

```python
class ElasticSolidSolver(SolidSolver):
    def __init__(self, mesh, solid_properties: Dict)
    
    def solve_solid_step(self, current_state: FluidSolidState, dt: float) -> np.ndarray:
        """求解弹性固体动力学"""
        
    def _assemble_mass_matrix(self) -> sp.spmatrix:
        """组装质量矩阵"""
        
    def _assemble_stiffness_matrix(self) -> sp.spmatrix:
        """组装刚度矩阵"""
        
    def _compute_external_force(self, interface_force: np.ndarray) -> np.ndarray:
        """计算外力"""
```

#### FluidSolidCoupling

流体-固体耦合求解器。

```python
class FluidSolidCoupling:
    def __init__(self, fluid_solver: FluidSolver, solid_solver: SolidSolver,
                 interface_nodes: np.ndarray, config: FSIConfig = None)
    
    def solve_coupled_system(self,
                           initial_state: FluidSolidState,
                           boundary_conditions: Dict,
                           time_steps: int,
                           dt: float,
                           fluid_source: Optional[Callable] = None,
                           solid_source: Optional[Callable] = None) -> List[FluidSolidState]:
        """求解耦合系统"""
        
    def _solve_partitioned_step(self, current_state: FluidSolidState,
                               boundary_conditions: Dict, dt: float) -> FluidSolidState:
        """分区求解步骤"""
        
    def _solve_monolithic_step(self, current_state: FluidSolidState,
                              boundary_conditions: Dict, dt: float) -> FluidSolidState:
        """整体求解步骤"""
        
    def _solve_staggered_step(self, current_state: FluidSolidState,
                             boundary_conditions: Dict, dt: float) -> FluidSolidState:
        """交错求解步骤"""
        
    def _compute_adaptive_timestep(self, current_state: FluidSolidState, dt: float) -> float:
        """计算自适应时间步长"""
        
    def _update_mesh_deformation(self, solid_displacement: np.ndarray) -> np.ndarray:
        """更新网格变形"""
        
    def visualize_coupling_results(self, solution_history: List[FluidSolidState]):
        """可视化耦合结果"""
```

#### 工厂函数

```python
def create_fluid_solid_coupling(fluid_mesh, solid_mesh, interface_nodes: np.ndarray,
                               config: FSIConfig = None) -> FluidSolidCoupling:
    """创建流体-固体耦合求解器"""
```

### coupling.chemical_mechanical

#### ChemicalMechanicalConfig

化学-力学耦合配置类。

```python
class ChemicalMechanicalConfig:
    def __init__(self,
                 solver_type: str = 'iterative',
                 max_iterations: int = 10,
                 tolerance: float = 1e-6,
                 relaxation_factor: float = 1.0,
                 adaptive_timestep: bool = True,
                 min_timestep: float = 1e-6,
                 max_timestep: float = 1e-3,
                 cfl_number: float = 0.5,
                 parallel_solver: bool = True,
                 preconditioner: str = 'jacobi')
```

#### ChemicalMechanicalCoupling

化学-力学耦合求解器。

```python
class ChemicalMechanicalCoupling:
    def __init__(self, mesh, config: ChemicalMechanicalConfig = None)
    
    def solve_coupled_system(self,
                           initial_concentration: np.ndarray,
                           initial_displacement: np.ndarray,
                           boundary_conditions: Dict,
                           time_steps: int,
                           dt: float,
                           temperature: np.ndarray,
                           pressure: np.ndarray,
                           chemical_source: Optional[Callable] = None,
                           body_force: Optional[Callable] = None) -> List[Dict]:
        """求解耦合系统"""
        
    def compute_chemical_strain(self, concentration: np.ndarray) -> np.ndarray:
        """计算化学应变"""
        
    def compute_stress_chemical_coupling(self, stress: np.ndarray, concentration: np.ndarray) -> np.ndarray:
        """计算应力-化学耦合"""
        
    def visualize_coupling_results(self, solution_history: List[Dict]):
        """可视化耦合结果"""
```

#### 工厂函数

```python
def create_chemical_mechanical_coupling(mesh, config: ChemicalMechanicalConfig = None) -> ChemicalMechanicalCoupling:
    """创建化学-力学耦合求解器"""
```

## 材料模型模块

### materials.elastic_models

#### LinearElasticModel

线性弹性材料模型。

```python
class LinearElasticModel:
    def __init__(self, young_modulus: float, poisson_ratio: float, density: float = 2700.0)
    
    def compute_stress(self, strain: np.ndarray) -> np.ndarray:
        """计算应力"""
        
    def compute_strain(self, stress: np.ndarray) -> np.ndarray:
        """计算应变"""
        
    def get_material_properties(self) -> Dict:
        """获取材料属性"""
```

#### NonlinearElasticModel

非线性弹性材料模型。

```python
class NonlinearElasticModel:
    def __init__(self, young_modulus: float, poisson_ratio: float, 
                 nonlinear_parameter: float, density: float = 2700.0)
    
    def compute_stress(self, strain: np.ndarray) -> np.ndarray:
        """计算应力"""
        
    def compute_tangent_stiffness(self, strain: np.ndarray) -> np.ndarray:
        """计算切线刚度"""
```

### materials.plastic_models

#### VonMisesPlasticModel

von Mises塑性材料模型。

```python
class VonMisesPlasticModel:
    def __init__(self, young_modulus: float, poisson_ratio: float,
                 yield_stress: float, hardening_modulus: float = 0.0,
                 density: float = 2700.0)
    
    def compute_stress(self, strain: np.ndarray, plastic_strain: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算应力"""
        
    def compute_yield_function(self, stress: np.ndarray) -> float:
        """计算屈服函数"""
        
    def compute_plastic_strain_increment(self, stress: np.ndarray, strain_increment: np.ndarray) -> np.ndarray:
        """计算塑性应变增量"""
```

#### DruckerPragerPlasticModel

Drucker-Prager塑性材料模型。

```python
class DruckerPragerPlasticModel:
    def __init__(self, young_modulus: float, poisson_ratio: float,
                 cohesion: float, friction_angle: float,
                 density: float = 2700.0)
    
    def compute_stress(self, strain: np.ndarray, plastic_strain: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算应力"""
        
    def compute_yield_function(self, stress: np.ndarray) -> float:
        """计算屈服函数"""
```

### materials.viscoelastic_models

#### MaxwellViscoelasticModel

Maxwell粘弹性材料模型。

```python
class MaxwellViscoelasticModel:
    def __init__(self, young_modulus: float, poisson_ratio: float,
                 viscosity: float, density: float = 2700.0)
    
    def compute_stress(self, strain: np.ndarray, time: float) -> np.ndarray:
        """计算应力"""
        
    def compute_relaxation_function(self, time: float) -> float:
        """计算松弛函数"""
```

#### KelvinVoigtViscoelasticModel

Kelvin-Voigt粘弹性材料模型。

```python
class KelvinVoigtViscoelasticModel:
    def __init__(self, young_modulus: float, poisson_ratio: float,
                 viscosity: float, density: float = 2700.0)
    
    def compute_stress(self, strain: np.ndarray, strain_rate: np.ndarray) -> np.ndarray:
        """计算应力"""
        
    def compute_creep_function(self, time: float) -> float:
        """计算蠕变函数"""
```

## 有限元模块

### finite_elements.mesh_generation

#### Mesh

网格基类。

```python
class Mesh:
    def __init__(self, nodes: np.ndarray, elements: List[Dict])
    
    def get_node_coordinates(self, node_id: int) -> np.ndarray:
        """获取节点坐标"""
        
    def get_element_nodes(self, element_id: int) -> List[int]:
        """获取单元节点"""
        
    def get_element_volume(self, element_id: int) -> float:
        """获取单元体积"""
        
    def refine_element(self, element_id: int):
        """细化单元"""
```

#### 工厂函数

```python
def create_rectangular_mesh(x_min: float, x_max: float, nx: int,
                           y_min: float, y_max: float, ny: int) -> Mesh:
    """创建矩形网格"""
    
def create_triangular_mesh(x_min: float, x_max: float, nx: int,
                          y_min: float, y_max: float, ny: int) -> Mesh:
    """创建三角形网格"""
    
def create_tetrahedral_mesh(x_min: float, x_max: float, nx: int,
                           y_min: float, y_max: float, ny: int,
                           z_min: float, z_max: float, nz: int) -> Mesh:
    """创建四面体网格"""
```

### finite_elements.assembly

#### 组装函数

```python
def assemble_stiffness_matrix(mesh: Mesh, material_model) -> sp.spmatrix:
    """组装刚度矩阵"""
    
def assemble_mass_matrix(mesh: Mesh, material_model) -> sp.spmatrix:
    """组装质量矩阵"""
    
def assemble_damping_matrix(mesh: Mesh, material_model) -> sp.spmatrix:
    """组装阻尼矩阵"""
    
def assemble_force_vector(mesh: Mesh, force_function: Callable) -> np.ndarray:
    """组装力向量"""
```

### finite_elements.boundary_conditions

#### 边界条件函数

```python
def apply_dirichlet_bc(stiffness_matrix: sp.spmatrix, force_vector: np.ndarray,
                      fixed_nodes: List[int], fixed_values: List[float]) -> Tuple[sp.spmatrix, np.ndarray]:
    """应用Dirichlet边界条件"""
    
def apply_neumann_bc(force_vector: np.ndarray, force_nodes: List[int],
                    force_values: List[float]) -> np.ndarray:
    """应用Neumann边界条件"""
    
def apply_periodic_bc(stiffness_matrix: sp.spmatrix, force_vector: np.ndarray,
                     periodic_pairs: List[Tuple[int, int]]) -> Tuple[sp.spmatrix, np.ndarray]:
    """应用周期性边界条件"""
```

## GPU加速模块

### gpu_acceleration.cuda_acceleration

#### CUDASolver

CUDA求解器。

```python
class CUDASolver:
    def __init__(self)
    
    def is_available(self) -> bool:
        """检查CUDA是否可用"""
        
    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        """在GPU上求解线性系统"""
        
    def get_gpu_info(self) -> Dict:
        """获取GPU信息"""
        
    def transfer_to_gpu(self, data: np.ndarray) -> 'cupy.ndarray':
        """传输数据到GPU"""
        
    def transfer_to_cpu(self, data: 'cupy.ndarray') -> np.ndarray:
        """传输数据到CPU"""
```

### gpu_acceleration.opencl_acceleration

#### OpenCLSolver

OpenCL求解器。

```python
class OpenCLSolver:
    def __init__(self)
    
    def is_available(self) -> bool:
        """检查OpenCL是否可用"""
        
    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        """在GPU上求解线性系统"""
        
    def get_devices(self) -> List[Dict]:
        """获取可用设备"""
        
    def create_context(self, device_type: str = 'GPU'):
        """创建OpenCL上下文"""
```

## 可视化模块

### visualization.plotting

#### 2D绘图函数

```python
def plot_2d_field(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                  title: str = "", save_path: str = None, backend: str = 'matplotlib'):
    """绘制2D场"""
    
def plot_contour(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                 levels: int = 20, title: str = "", save_path: str = None):
    """绘制等高线"""
    
def plot_vector_field(X: np.ndarray, Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                      title: str = "", save_path: str = None):
    """绘制向量场"""
    
def plot_mesh(nodes: np.ndarray, elements: List[Dict],
              title: str = "", save_path: str = None):
    """绘制网格"""
```

#### 3D绘图函数

```python
def plot_3d_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                    title: str = "", save_path: str = None):
    """绘制3D表面"""
    
def plot_3d_scatter(points: np.ndarray, values: np.ndarray,
                    title: str = "", save_path: str = None):
    """绘制3D散点图"""
    
def plot_isosurface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, V: np.ndarray,
                    level: float = 0.5, title: str = "", save_path: str = None):
    """绘制等值面"""
    
def plot_slice(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, V: np.ndarray,
               slice_axis: str = 'z', slice_value: float = 0.0,
               title: str = "", save_path: str = None):
    """绘制切片"""
```

### visualization.interactive

#### 交互式绘图函数

```python
def create_interactive_plot(data: np.ndarray, x: np.ndarray = None, y: np.ndarray = None,
                           plot_type: str = 'surface', title: str = "") -> 'plotly.graph_objects.Figure':
    """创建交互式图"""
    
def create_3d_interactive_plot(data: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                              plot_type: str = 'isosurface', title: str = "") -> 'plotly.graph_objects.Figure':
    """创建3D交互式图"""
```

## 时间积分模块

### time_integration.time_integrators

#### 显式积分器

```python
class ExplicitEuler:
    def __init__(self, dt: float)
    
    def integrate(self, f: Callable, t_span: Tuple[float, float], y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """显式Euler积分"""
    
class RungeKutta4:
    def __init__(self, dt: float)
    
    def integrate(self, f: Callable, t_span: Tuple[float, float], y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """四阶Runge-Kutta积分"""
```

#### 隐式积分器

```python
class ImplicitEuler:
    def __init__(self, dt: float)
    
    def integrate(self, f: Callable, t_span: Tuple[float, float], y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """隐式Euler积分"""
    
class CrankNicolson:
    def __init__(self, dt: float)
    
    def integrate(self, f: Callable, t_span: Tuple[float, float], y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crank-Nicolson积分"""
```

## 自适应模块

### adaptivity.mesh_refinement

#### AdaptiveMeshRefiner

自适应网格细化器。

```python
class AdaptiveMeshRefiner:
    def __init__(self, max_elements: int = 10000, refinement_ratio: float = 0.3,
                 coarsening_ratio: float = 0.1)
    
    def refine_mesh(self, mesh: Mesh, error_indicator: Callable) -> Mesh:
        """细化网格"""
        
    def coarsen_mesh(self, mesh: Mesh, error_indicator: Callable) -> Mesh:
        """粗化网格"""
        
    def adapt_mesh(self, mesh: Mesh, error_indicator: Callable) -> Mesh:
        """自适应网格"""
```

### adaptivity.time_adaptivity

#### AdaptiveTimeStepper

自适应时间步进器。

```python
class AdaptiveTimeStepper:
    def __init__(self, initial_dt: float, min_dt: float, max_dt: float, tolerance: float)
    
    def integrate(self, f: Callable, t_span: Tuple[float, float], y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """自适应时间积分"""
        
    def compute_optimal_timestep(self, error_estimate: float) -> float:
        """计算最优时间步长"""
```

---

## 数据类型说明

### 常用数据类型

- `np.ndarray`: NumPy数组，用于存储数值数据
- `sp.spmatrix`: SciPy稀疏矩阵
- `Dict`: 字典类型，用于存储配置和结果
- `List`: 列表类型，用于存储序列数据
- `Tuple`: 元组类型，用于存储固定长度的数据
- `Callable`: 可调用对象，如函数
- `Optional[T]`: 可选类型，可能为None
- `ABC`: 抽象基类

### 物理单位

- 长度: 米 (m)
- 时间: 秒 (s)
- 质量: 千克 (kg)
- 温度: 开尔文 (K)
- 压力: 帕斯卡 (Pa)
- 应力: 帕斯卡 (Pa)
- 应变: 无量纲
- 浓度: 摩尔/立方米 (mol/m³)
- 热导率: 瓦特/米/开尔文 (W/m/K)
- 比热容: 焦耳/千克/开尔文 (J/kg/K)
- 密度: 千克/立方米 (kg/m³)

---

**注意**: 本API参考文档提供了所有主要类和函数的详细说明。具体的使用示例请参考使用文档和快速入门指南。
