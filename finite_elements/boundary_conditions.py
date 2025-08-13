"""
边界条件处理模块 - Underworld风格
支持Dirichlet、Neumann、Robin边界条件
支持复杂几何边界、自适应边界等
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# 可选依赖
try:
    from scipy.sparse import csr_matrix, lil_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    csr_matrix = lil_matrix = None

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None


@dataclass
class BoundaryCondition:
    """边界条件基类"""
    name: str
    boundary_type: str  # 'dirichlet', 'neumann', 'robin'
    boundary_id: int
    field_name: str
    value: Union[float, Callable, np.ndarray]
    is_time_dependent: bool = False


class DirichletBC(BoundaryCondition):
    """Dirichlet边界条件"""
    
    def __init__(self, boundary_id: int, field_name: str, 
                 value: Union[float, Callable, np.ndarray], 
                 name: str = None):
        super().__init__(
            name=name or f"Dirichlet_{boundary_id}_{field_name}",
            boundary_type="dirichlet",
            boundary_id=boundary_id,
            field_name=field_name,
            value=value
        )


class NeumannBC(BoundaryCondition):
    """Neumann边界条件"""
    
    def __init__(self, boundary_id: int, field_name: str,
                 flux: Union[float, Callable, np.ndarray],
                 name: str = None):
        super().__init__(
            name=name or f"Neumann_{boundary_id}_{field_name}",
            boundary_type="neumann",
            boundary_id=boundary_id,
            field_name=field_name,
            value=flux
        )


class RobinBC(BoundaryCondition):
    """Robin边界条件 (混合边界条件)"""
    
    def __init__(self, boundary_id: int, field_name: str,
                 alpha: float, beta: float, gamma: float,
                 name: str = None):
        # alpha * u + beta * du/dn = gamma
        super().__init__(
            name=name or f"Robin_{boundary_id}_{field_name}",
            boundary_type="robin",
            boundary_id=boundary_id,
            field_name=field_name,
            value={"alpha": alpha, "beta": beta, "gamma": gamma}
        )


class BoundaryAssembly:
    """边界条件组装器 - Underworld风格"""
    
    def __init__(self, mesh):
        self.mesh = mesh
        self.boundary_conditions = []
        self.boundary_nodes = {}
        self.boundary_faces = {}
        
    def add_boundary_condition(self, bc: BoundaryCondition):
        """添加边界条件"""
        self.boundary_conditions.append(bc)
        
    def identify_boundary_nodes(self):
        """识别边界节点"""
        # 这里需要根据具体的网格结构来识别边界节点
        # 对于结构化网格，可以通过坐标判断
        # 对于非结构化网格，需要通过拓扑关系判断
        
        if hasattr(self.mesh, 'identify_boundary_nodes'):
            self.boundary_nodes = self.mesh.identify_boundary_nodes()
        else:
            # 默认实现：基于坐标识别边界
            self._identify_boundary_by_coordinates()
    
    def _identify_boundary_by_coordinates(self):
        """基于坐标识别边界节点"""
        coords = self.mesh.get_all_coordinates()
        tolerance = 1e-10
        
        # 识别各个边界
        boundaries = {
            'left': coords[:, 0] <= self.mesh.bounds[0] + tolerance,
            'right': coords[:, 0] >= self.mesh.bounds[1] - tolerance,
            'bottom': coords[:, 1] <= self.mesh.bounds[2] + tolerance,
            'top': coords[:, 1] >= self.mesh.bounds[3] - tolerance
        }
        
        if self.mesh.dimension == 3:
            boundaries.update({
                'front': coords[:, 2] <= self.mesh.bounds[4] + tolerance,
                'back': coords[:, 2] >= self.mesh.bounds[5] - tolerance
            })
        
        self.boundary_nodes = boundaries
    
    def apply_dirichlet_conditions(self, K: csr_matrix, f: np.ndarray, 
                                 time: float = 0.0) -> Tuple[csr_matrix, np.ndarray]:
        """应用Dirichlet边界条件"""
        if not HAS_SCIPY:
            raise ImportError("scipy is required for boundary condition application")
        
        K_modified = K.copy()
        f_modified = f.copy()
        
        for bc in self.boundary_conditions:
            if bc.boundary_type == "dirichlet":
                # 获取边界节点
                boundary_nodes = self._get_boundary_nodes(bc.boundary_id)
                
                # 计算边界值
                if callable(bc.value):
                    if bc.is_time_dependent:
                        boundary_values = bc.value(boundary_nodes, time)
                    else:
                        boundary_values = bc.value(boundary_nodes)
                else:
                    boundary_values = np.full(len(boundary_nodes), bc.value)
                
                # 应用边界条件
                for i, node_id in enumerate(boundary_nodes):
                    # 修改刚度矩阵
                    K_modified[node_id, :] = 0
                    K_modified[node_id, node_id] = 1
                    
                    # 修改右端项
                    f_modified[node_id] = boundary_values[i]
        
        return K_modified, f_modified
    
    def apply_neumann_conditions(self, f: np.ndarray, time: float = 0.0) -> np.ndarray:
        """应用Neumann边界条件"""
        f_modified = f.copy()
        
        for bc in self.boundary_conditions:
            if bc.boundary_type == "neumann":
                # 获取边界节点和面
                boundary_faces = self._get_boundary_faces(bc.boundary_id)
                
                # 计算边界积分
                for face in boundary_faces:
                    face_nodes = face['nodes']
                    face_area = face['area']
                    face_normal = face['normal']
                    
                    # 计算面力
                    if callable(bc.value):
                        if bc.is_time_dependent:
                            face_force = bc.value(face_nodes, time)
                        else:
                            face_force = bc.value(face_nodes)
                    else:
                        face_force = np.full(len(face_nodes), bc.value)
                    
                    # 组装到右端项
                    for i, node_id in enumerate(face_nodes):
                        f_modified[node_id] += face_force[i] * face_area / len(face_nodes)
        
        return f_modified
    
    def apply_robin_conditions(self, K: csr_matrix, f: np.ndarray,
                             time: float = 0.0) -> Tuple[csr_matrix, np.ndarray]:
        """应用Robin边界条件"""
        if not HAS_SCIPY:
            raise ImportError("scipy is required for Robin boundary conditions")
        
        K_modified = K.copy()
        f_modified = f.copy()
        
        for bc in self.boundary_conditions:
            if bc.boundary_type == "robin":
                alpha = bc.value["alpha"]
                beta = bc.value["beta"]
                gamma = bc.value["gamma"]
                
                boundary_faces = self._get_boundary_faces(bc.boundary_id)
                
                for face in boundary_faces:
                    face_nodes = face['nodes']
                    face_area = face['area']
                    
                    # 组装Robin边界项
                    for i, node_i in enumerate(face_nodes):
                        for j, node_j in enumerate(face_nodes):
                            # 刚度矩阵项
                            K_modified[node_i, node_j] += alpha * face_area / len(face_nodes)
                        
                        # 右端项
                        f_modified[node_i] += gamma * face_area / len(face_nodes)
        
        return K_modified, f_modified
    
    def _get_boundary_nodes(self, boundary_id: int) -> List[int]:
        """获取指定边界的节点"""
        # 这里需要根据具体的边界ID映射来获取节点
        # 简化实现：假设boundary_id对应边界名称
        boundary_names = ['left', 'right', 'bottom', 'top', 'front', 'back']
        if boundary_id < len(boundary_names):
            boundary_name = boundary_names[boundary_id]
            return np.where(self.boundary_nodes[boundary_name])[0].tolist()
        else:
            return []
    
    def _get_boundary_faces(self, boundary_id: int) -> List[Dict]:
        """获取指定边界的面"""
        # 实现边界面的几何信息计算
        if not hasattr(self.mesh, 'boundary_faces'):
            # 如果网格没有边界拓扑信息，尝试构建
            self._build_mesh_boundary_topology()
        
        # 根据边界ID获取对应的边界面
        boundary_faces = []
        
        if hasattr(self.mesh, 'boundary_faces'):
            # 使用网格的边界拓扑信息
            for face in self.mesh.boundary_faces:
                # 检查面是否属于指定的边界
                if self._face_belongs_to_boundary(face, boundary_id):
                    boundary_faces.append(face)
        else:
            # 回退到基于坐标的边界识别
            boundary_faces = self._get_boundary_faces_by_coordinates(boundary_id)
        
        return boundary_faces
    
    def _build_mesh_boundary_topology(self):
        """构建网格边界拓扑信息"""
        if hasattr(self.mesh, '_build_boundary_topology'):
            self.mesh._build_boundary_topology()
        elif hasattr(self.mesh, 'identify_boundary_faces'):
            self.mesh.identify_boundary_faces()
    
    def _face_belongs_to_boundary(self, face: Dict, boundary_id: int) -> bool:
        """判断面是否属于指定边界"""
        # 这里需要根据具体的边界ID映射逻辑来判断
        # 简化实现：假设boundary_id对应边界名称
        boundary_names = ['left', 'right', 'bottom', 'top', 'front', 'back']
        if boundary_id < len(boundary_names):
            boundary_name = boundary_names[boundary_id]
            return self._face_belongs_to_named_boundary(face, boundary_name)
        return False
    
    def _face_belongs_to_named_boundary(self, face: Dict, boundary_name: str) -> bool:
        """判断面是否属于指定名称的边界"""
        if not hasattr(self.mesh, 'boundary_nodes'):
            return False
        
        # 检查面的节点是否都在指定边界上
        face_nodes = face['nodes']
        boundary_nodes = self.mesh.boundary_nodes.get(boundary_name, [])
        
        if isinstance(boundary_nodes, np.ndarray):
            boundary_node_indices = np.where(boundary_nodes)[0]
        else:
            boundary_node_indices = boundary_nodes
        
        # 检查面的所有节点是否都在边界上
        return all(node_id in boundary_node_indices for node_id in face_nodes)
    
    def _get_boundary_faces_by_coordinates(self, boundary_id: int) -> List[Dict]:
        """基于坐标获取边界面（回退方法）"""
        boundary_faces = []
        
        # 获取边界名称
        boundary_names = ['left', 'right', 'bottom', 'top', 'front', 'back']
        if boundary_id >= len(boundary_names):
            return boundary_faces
        
        boundary_name = boundary_names[boundary_id]
        
        # 基于坐标识别边界节点
        if hasattr(self.mesh, 'boundary_nodes') and boundary_name in self.mesh.boundary_nodes:
            boundary_node_indices = np.where(self.mesh.boundary_nodes[boundary_name])[0]
            
            # 构建边界边（2D）或面（3D）
            if self.mesh.dimension == 2:
                # 2D：构建边界边
                for i in range(len(boundary_node_indices) - 1):
                    node1 = boundary_node_indices[i]
                    node2 = boundary_node_indices[i + 1]
                    
                    # 计算边的几何信息
                    p1 = self.mesh.get_node_coordinates(node1)
                    p2 = self.mesh.get_node_coordinates(node2)
                    
                    edge = {
                        'nodes': [node1, node2],
                        'area': np.linalg.norm(p2 - p1),  # 2D中边长度作为"面积"
                        'normal': self._compute_2d_edge_normal(p1, p2),
                        'center': (p1 + p2) / 2
                    }
                    boundary_faces.append(edge)
            else:
                # 3D：构建边界面（简化实现）
                # 这里需要更复杂的逻辑来构建3D边界面
                pass
        
        return boundary_faces
    
    def _compute_2d_edge_normal(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """计算2D边的法向量"""
        edge = p2 - p1
        # 逆时针90度旋转
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 0:
            return normal / norm
        return np.array([0.0, 1.0])  # 默认法向量


class AdaptiveBoundaryConditions:
    """自适应边界条件 - Underworld风格"""
    
    def __init__(self, mesh, base_bc: BoundaryAssembly):
        self.mesh = mesh
        self.base_bc = base_bc
        self.adaptive_conditions = []
        
    def add_adaptive_condition(self, condition: Callable):
        """添加自适应边界条件"""
        self.adaptive_conditions.append(condition)
    
    def update_boundary_conditions(self, current_solution: np.ndarray, 
                                 time: float = 0.0):
        """根据当前解更新边界条件"""
        for condition in self.adaptive_conditions:
            new_bc = condition(current_solution, time)
            if new_bc:
                self.base_bc.add_boundary_condition(new_bc)


class ImmersedBoundaryMethod:
    """沉浸边界法 - Underworld风格"""
    
    def __init__(self, mesh, boundary_function: Callable):
        self.mesh = mesh
        self.boundary_function = boundary_function
        self.immersed_nodes = []
        self.immersed_elements = []
        
    def identify_immersed_boundary(self):
        """识别沉浸边界"""
        # 遍历所有节点，找到边界附近的节点
        coords = self.mesh.get_all_coordinates()
        
        for i, coord in enumerate(coords):
            distance = self.boundary_function(coord)
            if abs(distance) < self.mesh.characteristic_length * 0.1:
                self.immersed_nodes.append(i)
        
        # 识别包含沉浸边界的单元
        for elem_id in range(self.mesh.n_elements):
            elem_nodes = self.mesh.get_element_nodes(elem_id)
            elem_coords = self.mesh.get_node_coordinates(elem_nodes)
            
            # 检查单元是否与边界相交
            distances = [self.boundary_function(coord) for coord in elem_coords]
            if any(d < 0 for d in distances) and any(d > 0 for d in distances):
                self.immersed_elements.append(elem_id)
    
    def modify_element_matrices(self, element_matrices: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """修改沉浸边界单元矩阵"""
        modified_matrices = element_matrices.copy()
        
        for elem_id in self.immersed_elements:
            if elem_id in modified_matrices:
                # 计算切割系数
                elem_nodes = self.mesh.get_element_nodes(elem_id)
                elem_coords = self.mesh.get_node_coordinates(elem_nodes)
                
                # 简化的切割系数计算
                distances = [self.boundary_function(coord) for coord in elem_coords]
                min_dist = min(distances)
                max_dist = max(distances)
                
                if max_dist > 0 and min_dist < 0:
                    # 单元被边界切割
                    cut_ratio = abs(min_dist) / (abs(min_dist) + max_dist)
                    
                    # 修改单元矩阵
                    modified_matrices[elem_id] *= cut_ratio
        
        return modified_matrices


class PeriodicBC(BoundaryCondition):
    """周期性边界条件"""
    
    def __init__(self, boundary_id: int, field_name: str,
                 paired_boundary_id: int, periodicity_type: str = 'translational',
                 offset: np.ndarray = None, name: str = None):
        super().__init__(
            name=name or f"Periodic_{boundary_id}_{field_name}",
            boundary_type="periodic",
            boundary_id=boundary_id,
            field_name=field_name,
            value={"paired_boundary": paired_boundary_id, "type": periodicity_type, "offset": offset}
        )


class SlidingBC(BoundaryCondition):
    """滑动边界条件（如断层）"""
    
    def __init__(self, boundary_id: int, field_name: str,
                 slip_direction: np.ndarray, slip_rate: Union[float, Callable],
                 friction_coefficient: float = 0.6, name: str = None):
        super().__init__(
            name=name or f"Sliding_{boundary_id}_{field_name}",
            boundary_type="sliding",
            boundary_id=boundary_id,
            field_name=field_name,
            value={"slip_direction": slip_direction, "slip_rate": slip_rate, 
                   "friction": friction_coefficient}
        )


class FreeSurfaceBC(BoundaryCondition):
    """自由表面边界条件（如地形演化）"""
    
    def __init__(self, boundary_id: int, field_name: str,
                 surface_tension: float = 0.0, gravity: float = 9.81,
                 erosion_rate: Union[float, Callable] = 0.0, name: str = None):
        super().__init__(
            name=name or f"FreeSurface_{boundary_id}_{field_name}",
            boundary_type="free_surface",
            boundary_id=boundary_id,
            field_name=field_name,
            value={"surface_tension": surface_tension, "gravity": gravity, 
                   "erosion_rate": erosion_rate}
        )


class AdvancedBoundaryAssembly(BoundaryAssembly):
    """高级边界条件组装器 - 支持复杂边界条件"""
    
    def __init__(self, mesh):
        super().__init__(mesh)
        self.periodic_pairs = {}  # 周期性边界对
        self.sliding_boundaries = {}  # 滑动边界
        self.free_surfaces = {}  # 自由表面
    
    def add_periodic_boundary(self, bc: PeriodicBC):
        """添加周期性边界条件"""
        self.add_boundary_condition(bc)
        paired_id = bc.value["paired_boundary"]
        self.periodic_pairs[bc.boundary_id] = paired_id
    
    def add_sliding_boundary(self, bc: SlidingBC):
        """添加滑动边界条件"""
        self.add_boundary_condition(bc)
        self.sliding_boundaries[bc.boundary_id] = bc
    
    def add_free_surface(self, bc: FreeSurfaceBC):
        """添加自由表面边界条件"""
        self.add_boundary_condition(bc)
        self.free_surfaces[bc.boundary_id] = bc
    
    def apply_periodic_conditions(self, K: csr_matrix, f: np.ndarray,
                                time: float = 0.0) -> Tuple[csr_matrix, np.ndarray]:
        """应用周期性边界条件"""
        if not HAS_SCIPY:
            raise ImportError("scipy is required for periodic boundary conditions")
        
        K_modified = K.copy()
        f_modified = f.copy()
        
        for bc in self.boundary_conditions:
            if bc.boundary_type == "periodic":
                paired_id = bc.value["paired_boundary"]
                periodicity_type = bc.value["type"]
                offset = bc.value.get("offset", np.zeros(self.mesh.dimension))
                
                # 获取边界节点
                boundary_nodes = self._get_boundary_nodes(bc.boundary_id)
                paired_nodes = self._get_boundary_nodes(paired_id)
                
                if len(boundary_nodes) == len(paired_nodes):
                    # 应用周期性约束
                    for i, (node1, node2) in enumerate(zip(boundary_nodes, paired_nodes)):
                        if periodicity_type == 'translational':
                            # 平移周期性：u1 = u2 + offset
                            K_modified[node1, node1] = 1.0
                            K_modified[node1, node2] = -1.0
                            f_modified[node1] = offset[i] if i < len(offset) else 0.0
                        elif periodicity_type == 'rotational':
                            # 旋转周期性（简化实现）
                            K_modified[node1, node1] = 1.0
                            K_modified[node1, node2] = -1.0
                            f_modified[node1] = 0.0
        
        return K_modified, f_modified
    
    def apply_sliding_conditions(self, K: csr_matrix, f: np.ndarray,
                               time: float = 0.0) -> Tuple[csr_matrix, np.ndarray]:
        """应用滑动边界条件"""
        if not HAS_SCIPY:
            raise ImportError("scipy is required for sliding boundary conditions")
        
        K_modified = K.copy()
        f_modified = f.copy()
        
        for bc in self.boundary_conditions:
            if bc.boundary_type == "sliding":
                slip_direction = bc.value["slip_direction"]
                slip_rate = bc.value["slip_rate"]
                friction = bc.value["friction"]
                
                # 获取边界节点
                boundary_nodes = self._get_boundary_nodes(bc.boundary_id)
                
                # 计算滑动速度
                if callable(slip_rate):
                    if bc.is_time_dependent:
                        slip_velocity = slip_rate(boundary_nodes, time)
                    else:
                        slip_velocity = slip_rate(boundary_nodes)
                else:
                    slip_velocity = np.full(len(boundary_nodes), slip_rate)
                
                # 应用滑动约束
                for i, node_id in enumerate(boundary_nodes):
                    # 在滑动方向上的约束
                    # 这里需要根据具体的有限元实现来设置约束
                    # 简化实现：设置对角项
                    K_modified[node_id, node_id] = 1.0
                    
                    # 添加摩擦力项
                    if friction > 0:
                        # 摩擦力与法向力成正比
                        # 这里需要更复杂的实现
                        pass
        
        return K_modified, f_modified
    
    def apply_free_surface_conditions(self, K: csr_matrix, f: np.ndarray,
                                    time: float = 0.0) -> Tuple[csr_matrix, np.ndarray]:
        """应用自由表面边界条件"""
        if not HAS_SCIPY:
            raise ImportError("scipy is required for free surface boundary conditions")
        
        K_modified = K.copy()
        f_modified = f.copy()
        
        for bc in self.boundary_conditions:
            if bc.boundary_type == "free_surface":
                surface_tension = bc.value["surface_tension"]
                gravity = bc.value["gravity"]
                erosion_rate = bc.value["erosion_rate"]
                
                # 获取边界节点
                boundary_nodes = self._get_boundary_nodes(bc.boundary_id)
                boundary_faces = self._get_boundary_faces(bc.boundary_id)
                
                # 应用自由表面条件
                for face in boundary_faces:
                    face_nodes = face['nodes']
                    face_area = face['area']
                    face_normal = face['normal']
                    
                    # 表面张力项
                    if surface_tension > 0:
                        # 表面张力与曲率成正比
                        # 这里需要计算表面曲率
                        curvature = self._compute_surface_curvature(face_nodes)
                        
                        for node_id in face_nodes:
                            # 添加表面张力项到右端项
                            f_modified[node_id] += surface_tension * curvature * face_area / len(face_nodes)
                    
                    # 重力项
                    if gravity > 0:
                        for node_id in face_nodes:
                            # 添加重力项
                            f_modified[node_id] += gravity * face_area / len(face_nodes)
                    
                    # 侵蚀项
                    if callable(erosion_rate):
                        if bc.is_time_dependent:
                            erosion = erosion_rate(face_nodes, time)
                        else:
                            erosion = erosion_rate(face_nodes)
                        
                        for i, node_id in enumerate(face_nodes):
                            f_modified[node_id] += erosion[i] * face_area / len(face_nodes)
        
        return K_modified, f_modified
    
    def _compute_surface_curvature(self, face_nodes: List[int]) -> float:
        """计算表面曲率（简化实现）"""
        # 这里需要实现真正的曲率计算
        # 简化：返回0
        return 0.0
    
    def apply_all_advanced_conditions(self, K: csr_matrix, f: np.ndarray,
                                    time: float = 0.0) -> Tuple[csr_matrix, np.ndarray]:
        """应用所有高级边界条件"""
        # 应用标准边界条件
        K, f = self.apply_dirichlet_conditions(K, f, time)
        f = self.apply_neumann_conditions(f, time)
        K, f = self.apply_robin_conditions(K, f, time)
        
        # 应用高级边界条件
        K, f = self.apply_periodic_conditions(K, f, time)
        K, f = self.apply_sliding_conditions(K, f, time)
        K, f = self.apply_free_surface_conditions(K, f, time)
        
        return K, f


# 便捷函数
def create_dirichlet_bc(boundary_id: int, field_name: str, 
                       value: Union[float, Callable], name: str = None) -> DirichletBC:
    """创建Dirichlet边界条件"""
    return DirichletBC(boundary_id, field_name, value, name)

def create_neumann_bc(boundary_id: int, field_name: str,
                     flux: Union[float, Callable], name: str = None) -> NeumannBC:
    """创建Neumann边界条件"""
    return NeumannBC(boundary_id, field_name, flux, name)

def create_robin_bc(boundary_id: int, field_name: str,
                   alpha: float, beta: float, gamma: float, name: str = None) -> RobinBC:
    """创建Robin边界条件"""
    return RobinBC(boundary_id, field_name, alpha, beta, gamma, name) 

def create_periodic_bc(boundary_id: int, field_name: str,
                      paired_boundary_id: int, periodicity_type: str = 'translational',
                      offset: np.ndarray = None, name: str = None) -> PeriodicBC:
    """创建周期性边界条件"""
    return PeriodicBC(boundary_id, field_name, paired_boundary_id, periodicity_type, offset, name)

def create_sliding_bc(boundary_id: int, field_name: str,
                     slip_direction: np.ndarray, slip_rate: Union[float, Callable],
                     friction_coefficient: float = 0.6, name: str = None) -> SlidingBC:
    """创建滑动边界条件"""
    return SlidingBC(boundary_id, field_name, slip_direction, slip_rate, friction_coefficient, name)

def create_free_surface_bc(boundary_id: int, field_name: str,
                          surface_tension: float = 0.0, gravity: float = 9.81,
                          erosion_rate: Union[float, Callable] = 0.0, name: str = None) -> FreeSurfaceBC:
    """创建自由表面边界条件"""
    return FreeSurfaceBC(boundary_id, field_name, surface_tension, gravity, erosion_rate, name) 