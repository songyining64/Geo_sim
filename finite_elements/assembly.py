"""
单元刚度矩阵装配
"""

import numpy as np
from .basis_functions import LagrangeTriangle, LagrangeQuad, LagrangeTetra, LagrangeHex
from .quadrature import triangle_points_weights, quad_points_weights, tetra_points_weights, hex_points_weights
from .transformations import jacobian_matrix, jacobian_det, dN_dx
import sympy as sp
from scipy.sparse import csr_matrix, lil_matrix
from typing import Dict, List, Tuple, Optional, Callable

# 可选依赖
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None

try:
    from petsc4py import PETSc
    HAS_PETSC = True
except ImportError:
    HAS_PETSC = False
    PETSc = None


class ElementAssembly:
    """单元矩阵组装器 - Underworld风格"""
    
    def __init__(self, element_type: str, order: int = 1):
        self.element_type = element_type
        self.order = order
        self.basis = self._create_basis()
        self.quad_pts, self.quad_wts = self._get_quadrature()
        
    def _create_basis(self):
        """创建基函数"""
        if self.element_type == "triangle":
            return LagrangeTriangle(self.order)
        elif self.element_type == "quad":
            return LagrangeQuad(self.order)
        elif self.element_type == "tetra":
            return LagrangeTetra(self.order)
        elif self.element_type == "hex":
            return LagrangeHex(self.order)
        else:
            raise ValueError(f"不支持的单元类型: {self.element_type}")
    
    def _get_quadrature(self):
        """获取积分点和权重"""
        if self.element_type == "triangle":
            return triangle_points_weights(self.order)
        elif self.element_type == "quad":
            return quad_points_weights(self.order)
        elif self.element_type == "tetra":
            return tetra_points_weights(self.order)
        elif self.element_type == "hex":
            return hex_points_weights(self.order)
    
    def assemble_stiffness_matrix(self, node_coords: np.ndarray, 
                                 material_props: Dict[str, float]) -> np.ndarray:
        """组装刚度矩阵"""
        n_nodes = node_coords.shape[0]
        Ke = np.zeros((n_nodes, n_nodes))
        
        for q, w in zip(self.quad_pts, self.quad_wts):
            # 计算基函数值和导数
            N = self.basis.evaluate(q.reshape(1, -1))[0]
            dN_dxi = self.basis.evaluate_derivatives(q.reshape(1, -1))[0]
            
            # 计算雅可比矩阵
            J = jacobian_matrix(node_coords, dN_dxi)
            detJ = jacobian_det(J)
            J_inv = np.linalg.inv(J)
            
            # 计算物理坐标下的导数
            dN_dx_ = dN_dx(dN_dxi, J_inv)
            
            # 组装刚度矩阵
            Ke += material_props.get('lambda', 1.0) * (dN_dx_ @ dN_dx_.T) * detJ * w
            
        return Ke
    
    def assemble_mass_matrix(self, node_coords: np.ndarray, 
                           density: float = 1.0) -> np.ndarray:
        """组装质量矩阵"""
        n_nodes = node_coords.shape[0]
        Me = np.zeros((n_nodes, n_nodes))
        
        for q, w in zip(self.quad_pts, self.quad_wts):
            N = self.basis.evaluate(q.reshape(1, -1))[0]
            dN_dxi = self.basis.evaluate_derivatives(q.reshape(1, -1))[0]
            
            J = jacobian_matrix(node_coords, dN_dxi)
            detJ = jacobian_det(J)
            
            # 组装质量矩阵
            Me += density * np.outer(N, N) * detJ * w
            
        return Me
    
    def assemble_elasticity_matrix(self, node_coords: np.ndarray,
                                 lame_lambda: float, lame_mu: float) -> np.ndarray:
        """组装弹性矩阵（2D/3D）"""
        dim = node_coords.shape[1]
        n_nodes = node_coords.shape[0]
        n_dofs = dim * n_nodes
        Ke = np.zeros((n_dofs, n_dofs))
        
        for q, w in zip(self.quad_pts, self.quad_wts):
            N = self.basis.evaluate(q.reshape(1, -1))[0]
            dN_dxi = self.basis.evaluate_derivatives(q.reshape(1, -1))[0]
            
            J = jacobian_matrix(node_coords, dN_dxi)
            detJ = jacobian_det(J)
            J_inv = np.linalg.inv(J)
            dN_dx_ = dN_dx(dN_dxi, J_inv)
            
            # 构建应变-位移矩阵B
            B = self._build_strain_displacement_matrix(dN_dx_, dim)
            
            # 构建弹性矩阵D
            D = self._build_elasticity_matrix(lame_lambda, lame_mu, dim)
            
            # 组装弹性矩阵
            Ke += B.T @ D @ B * detJ * w
            
        return Ke
    
    def _build_strain_displacement_matrix(self, dN_dx: np.ndarray, dim: int) -> np.ndarray:
        """构建应变-位移矩阵"""
        n_nodes = dN_dx.shape[0]
        
        if dim == 2:
            # 平面应变
            B = np.zeros((3, 2 * n_nodes))
            for i in range(n_nodes):
                B[0, 2*i] = dN_dx[i, 0]     # du/dx
                B[1, 2*i+1] = dN_dx[i, 1]   # dv/dy
                B[2, 2*i] = dN_dx[i, 1]     # du/dy
                B[2, 2*i+1] = dN_dx[i, 0]   # dv/dx
        elif dim == 3:
            # 3D
            B = np.zeros((6, 3 * n_nodes))
            for i in range(n_nodes):
                B[0, 3*i] = dN_dx[i, 0]     # du/dx
                B[1, 3*i+1] = dN_dx[i, 1]   # dv/dy
                B[2, 3*i+2] = dN_dx[i, 2]   # dw/dz
                B[3, 3*i] = dN_dx[i, 1]     # du/dy
                B[3, 3*i+1] = dN_dx[i, 0]   # dv/dx
                B[4, 3*i+1] = dN_dx[i, 2]   # dv/dz
                B[4, 3*i+2] = dN_dx[i, 1]   # dw/dy
                B[5, 3*i] = dN_dx[i, 2]     # du/dz
                B[5, 3*i+2] = dN_dx[i, 0]   # dw/dx
        else:
            raise ValueError(f"不支持的维度: {dim}")
            
        return B
    
    def _build_elasticity_matrix(self, lame_lambda: float, lame_mu: float, dim: int) -> np.ndarray:
        """构建弹性矩阵"""
        if dim == 2:
            # 平面应变
            D = np.array([
                [lame_lambda + 2*lame_mu, lame_lambda, 0],
                [lame_lambda, lame_lambda + 2*lame_mu, 0],
                [0, 0, lame_mu]
            ])
        elif dim == 3:
            # 3D
            D = np.zeros((6, 6))
            D[0:3, 0:3] = lame_lambda
            D[0, 0] += 2*lame_mu
            D[1, 1] += 2*lame_mu
            D[2, 2] += 2*lame_mu
            D[3, 3] = lame_mu
            D[4, 4] = lame_mu
            D[5, 5] = lame_mu
        else:
            raise ValueError(f"不支持的维度: {dim}")
            
        return D


class GlobalAssembly:
    """全局矩阵组装器 - Underworld风格"""
    
    def __init__(self, mesh, element_assembly: ElementAssembly):
        self.mesh = mesh
        self.element_assembly = element_assembly
        self.n_nodes = mesh.n_nodes
        self.n_elements = mesh.n_elements
        
        # 并行设置
        self.comm = MPI.COMM_WORLD if HAS_MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1
        
    def assemble_global_stiffness_matrix(self, material_props: Dict[str, float]) -> csr_matrix:
        """组装全局刚度矩阵"""
        # 使用LIL矩阵进行组装（适合增量组装）
        K_global = lil_matrix((self.n_nodes, self.n_nodes))
        
        for elem_id in range(self.n_elements):
            # 获取单元节点坐标
            elem_nodes = self.mesh.get_element_nodes(elem_id)
            node_coords = self.mesh.get_node_coordinates(elem_nodes)
            
            # 组装单元矩阵
            Ke = self.element_assembly.assemble_stiffness_matrix(node_coords, material_props)
            
            # 组装到全局矩阵
            for i, node_i in enumerate(elem_nodes):
                for j, node_j in enumerate(elem_nodes):
                    K_global[node_i, node_j] += Ke[i, j]
        
        return K_global.tocsr()
    
    def assemble_global_mass_matrix(self, density: float = 1.0) -> csr_matrix:
        """组装全局质量矩阵"""
        M_global = lil_matrix((self.n_nodes, self.n_nodes))
        
        for elem_id in range(self.n_elements):
            elem_nodes = self.mesh.get_element_nodes(elem_id)
            node_coords = self.mesh.get_node_coordinates(elem_nodes)
            
            Me = self.element_assembly.assemble_mass_matrix(node_coords, density)
            
            for i, node_i in enumerate(elem_nodes):
                for j, node_j in enumerate(elem_nodes):
                    M_global[node_i, node_j] += Me[i, j]
        
        return M_global.tocsr()
    
    def assemble_global_elasticity_matrix(self, lame_lambda: float, lame_mu: float) -> csr_matrix:
        """组装全局弹性矩阵"""
        dim = self.mesh.dimension
        n_dofs = dim * self.n_nodes
        K_global = lil_matrix((n_dofs, n_dofs))
        
        for elem_id in range(self.n_elements):
            elem_nodes = self.mesh.get_element_nodes(elem_id)
            node_coords = self.mesh.get_node_coordinates(elem_nodes)
            
            Ke = self.element_assembly.assemble_elasticity_matrix(
                node_coords, lame_lambda, lame_mu)
            
            # 组装到全局矩阵
            for i, node_i in enumerate(elem_nodes):
                for j, node_j in enumerate(elem_nodes):
                    for d1 in range(dim):
                        for d2 in range(dim):
                            global_i = dim * node_i + d1
                            global_j = dim * node_j + d2
                            local_i = dim * i + d1
                            local_j = dim * j + d2
                            K_global[global_i, global_j] += Ke[local_i, local_j]
        
        return K_global.tocsr()


class ParallelAssembly:
    """并行矩阵组装器 - Underworld风格"""
    
    def __init__(self, mesh, element_assembly: ElementAssembly):
        self.mesh = mesh
        self.element_assembly = element_assembly
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # 域分解
        self.local_elements = self._partition_elements()
        
    def _partition_elements(self) -> List[int]:
        """分区元素"""
        if self.size == 1:
            return list(range(self.mesh.n_elements))
        
        # 简单的均匀分区
        elements_per_proc = self.mesh.n_elements // self.size
        start = self.rank * elements_per_proc
        end = start + elements_per_proc if self.rank < self.size - 1 else self.mesh.n_elements
        
        return list(range(start, end))
    
    def assemble_parallel_stiffness_matrix(self, material_props: Dict[str, float]) -> csr_matrix:
        """并行组装刚度矩阵"""
        # 本地组装
        local_K = lil_matrix((self.mesh.n_nodes, self.mesh.n_nodes))
        
        for elem_id in self.local_elements:
            elem_nodes = self.mesh.get_element_nodes(elem_id)
            node_coords = self.mesh.get_node_coordinates(elem_nodes)
            
            Ke = self.element_assembly.assemble_stiffness_matrix(node_coords, material_props)
            
            for i, node_i in enumerate(elem_nodes):
                for j, node_j in enumerate(elem_nodes):
                    local_K[node_i, node_j] += Ke[i, j]
        
        # 全局归约
        local_K = local_K.tocsr()
        global_K = self.comm.allreduce(local_K, op=MPI.SUM)
        
        return global_K


# 便捷函数
def triangle_stiffness(node_coords, order=1, material_lambda=1.0):
    """三角形单元刚度矩阵"""
    assembly = ElementAssembly("triangle", order)
    return assembly.assemble_stiffness_matrix(node_coords, {"lambda": material_lambda})

def quad_stiffness(node_coords, order=1, material_lambda=1.0):
    """四边形单元刚度矩阵"""
    assembly = ElementAssembly("quad", order)
    return assembly.assemble_stiffness_matrix(node_coords, {"lambda": material_lambda})

def tetra_stiffness(node_coords, order=1, material_lambda=1.0):
    """四面体单元刚度矩阵"""
    assembly = ElementAssembly("tetra", order)
    return assembly.assemble_stiffness_matrix(node_coords, {"lambda": material_lambda})

def hex_stiffness(node_coords, order=1, material_lambda=1.0):
    """六面体单元刚度矩阵"""
    assembly = ElementAssembly("hex", order)
    return assembly.assemble_stiffness_matrix(node_coords, {"lambda": material_lambda})

def elasticity_triangle_stiffness(node_coords, order=1, lame_lambda=1.0, lame_mu=1.0):
    """三角形弹性矩阵"""
    assembly = ElementAssembly("triangle", order)
    return assembly.assemble_elasticity_matrix(node_coords, lame_lambda, lame_mu)

def stokes_heat_element_matrix(node_coords, params):
    """Stokes-热耦合单元矩阵"""
    # 这里可以实现Stokes-热耦合的单元矩阵组装
    # 包括速度、压力和温度的耦合项
    pass
