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
        # 这里需要根据具体的网格结构来获取边界面
        # 简化实现：返回空列表
        return []


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