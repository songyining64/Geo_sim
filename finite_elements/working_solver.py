"""
有限元求解器
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, gmres
from scipy.linalg import solve
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

@dataclass
class Mesh:
    """简单但完整的网格类"""
    coordinates: np.ndarray  # (n_nodes, dim)
    elements: np.ndarray     # (n_elements, n_nodes_per_element)
    element_type: str = "triangle"
    
    def __post_init__(self):
        self.n_nodes = len(self.coordinates)
        self.n_elements = len(self.elements)
        self.dim = self.coordinates.shape[1]
        
        # 计算网格质量
        self.quality = self._compute_mesh_quality()
    
    def _compute_mesh_quality(self) -> float:
        """计算网格质量"""
        qualities = []
        for element in self.elements:
            coords = self.coordinates[element]
            if self.element_type == "triangle":
                # 三角形质量：面积/最长边长的平方
                edges = []
                for i in range(3):
                    edge = coords[(i+1)%3] - coords[i]
                    edges.append(np.linalg.norm(edge))
                
                # 计算面积
                v1 = coords[1] - coords[0]
                v2 = coords[2] - coords[0]
                area = 0.5 * abs(np.cross(v1, v2))
                
                max_edge = max(edges)
                quality = area / (max_edge ** 2)
                qualities.append(quality)
        
        return np.mean(qualities) if qualities else 0.0

class WorkingFiniteElementSolver:
    """完整可工作的有限元求解器"""
    
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.solution = None
        self.solve_time = 0.0
        
        # 预计算基函数和积分点
        self._setup_basis_functions()
        self._setup_quadrature()
    
    def _setup_basis_functions(self):
        """设置基函数"""
        if self.mesh.element_type == "triangle":
            # 线性三角形基函数
            self.basis_functions = {
                'N': lambda xi, eta: np.array([1-xi-eta, xi, eta]),
                'dN_dxi': lambda xi, eta: np.array([[-1, 1, 0], [-1, 0, 1]]).T
            }
        else:
            raise ValueError(f"不支持的单元类型: {self.mesh.element_type}")
    
    def _setup_quadrature(self):
        """设置积分点"""
        if self.mesh.element_type == "triangle":
            # 3点高斯积分
            self.quad_points = np.array([
                [1/6, 1/6],
                [2/3, 1/6], 
                [1/6, 2/3]
            ])
            self.quad_weights = np.array([1/6, 1/6, 1/6])
    
    def solve_heat_conduction(self, 
                            thermal_conductivity: float = 1.0,
                            heat_source: Optional[Callable] = None,
                            boundary_conditions: Optional[Dict] = None) -> np.ndarray:
        """
        求解热传导问题
        
        Args:
            thermal_conductivity: 热导率
            heat_source: 热源函数 f(x, y)
            boundary_conditions: 边界条件
            
        Returns:
            temperature: 温度场解
        """
        print("🔥 求解热传导问题...")
        start_time = time.time()
        
        # 组装刚度矩阵和右端项
        K, f = self._assemble_heat_conduction_system(thermal_conductivity, heat_source)
        
        # 应用边界条件
        if boundary_conditions:
            K, f = self._apply_boundary_conditions(K, f, boundary_conditions)
        
        # 求解线性系统
        temperature = spsolve(K, f)
        
        self.solution = temperature
        self.solve_time = time.time() - start_time
        
        print(f"✅ 热传导求解完成，耗时: {self.solve_time:.3f}秒")
        return temperature
    
    def solve_elasticity(self,
                        young_modulus: float = 70e9,
                        poisson_ratio: float = 0.3,
                        body_force: Optional[Callable] = None,
                        boundary_conditions: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解弹性力学问题
        
        Args:
            young_modulus: 杨氏模量
            poisson_ratio: 泊松比
            body_force: 体积力函数
            boundary_conditions: 边界条件
            
        Returns:
            displacement: 位移场
            stress: 应力场
        """
        print("🏗️ 求解弹性力学问题...")
        start_time = time.time()
        
        # 计算拉梅常数
        mu = young_modulus / (2 * (1 + poisson_ratio))
        lambda_val = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        
        # 组装刚度矩阵和右端项
        K, f = self._assemble_elasticity_system(mu, lambda_val, body_force)
        
        # 应用边界条件
        if boundary_conditions:
            K, f = self._apply_elasticity_boundary_conditions(K, f, boundary_conditions)
        
        # 求解线性系统
        displacement = spsolve(K, f)
        
        # 计算应力
        stress = self._compute_stress(displacement, mu, lambda_val)
        
        self.solution = displacement
        self.solve_time = time.time() - start_time
        
        print(f"✅ 弹性力学求解完成，耗时: {self.solve_time:.3f}秒")
        return displacement, stress
    
    def solve_stokes_flow(self,
                         viscosity: float = 1.0,
                         body_force: Optional[Callable] = None,
                         boundary_conditions: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解Stokes流问题
        
        Args:
            viscosity: 粘度
            body_force: 体积力
            boundary_conditions: 边界条件
            
        Returns:
            velocity: 速度场
            pressure: 压力场
        """
        print("🌊 求解Stokes流问题...")
        start_time = time.time()
        
        # 组装Stokes系统
        A, b = self._assemble_stokes_system(viscosity, body_force)
        
        # 应用边界条件
        if boundary_conditions:
            A, b = self._apply_stokes_boundary_conditions(A, b, boundary_conditions)
        
        # 求解线性系统
        solution = spsolve(A, b)
        
        # 分离速度和压力
        n_velocity_dofs = 2 * self.mesh.n_nodes
        velocity = solution[:n_velocity_dofs]
        pressure = solution[n_velocity_dofs:]
        
        self.solution = solution
        self.solve_time = time.time() - start_time
        
        print(f"✅ Stokes流求解完成，耗时: {self.solve_time:.3f}秒")
        return velocity, pressure
    
    def _assemble_heat_conduction_system(self, k: float, heat_source: Optional[Callable]) -> Tuple[csr_matrix, np.ndarray]:
        """组装热传导系统"""
        n_nodes = self.mesh.n_nodes
        
        # 初始化矩阵
        K = lil_matrix((n_nodes, n_nodes))
        f = np.zeros(n_nodes)
        
        # 遍历所有单元
        for element in self.mesh.elements:
            element_coords = self.mesh.coordinates[element]
            
            # 计算单元矩阵
            Ke, fe = self._compute_heat_element_matrix(element_coords, k, heat_source)
            
            # 组装到全局矩阵
            for i, node_i in enumerate(element):
                for j, node_j in enumerate(element):
                    K[node_i, node_j] += Ke[i, j]
                f[node_i] += fe[i]
        
        return K.tocsr(), f
    
    def _compute_heat_element_matrix(self, element_coords: np.ndarray, k: float, heat_source: Optional[Callable]) -> Tuple[np.ndarray, np.ndarray]:
        """计算热传导单元矩阵"""
        n_nodes = len(element_coords)
        Ke = np.zeros((n_nodes, n_nodes))
        fe = np.zeros(n_nodes)
        
        # 计算雅可比矩阵
        J = self._compute_jacobian(element_coords)
        J_inv = np.linalg.inv(J)
        det_J = abs(np.linalg.det(J))
        
        # 在积分点上计算
        for q, (xi, eta) in enumerate(self.quad_points):
            weight = self.quad_weights[q]
            
            # 基函数值
            N = self.basis_functions['N'](xi, eta)
            
            # 基函数导数
            dN_dxi = self.basis_functions['dN_dxi'](xi, eta)
            dN_dx = dN_dxi @ J_inv.T
            
            # 刚度矩阵项
            for i in range(n_nodes):
                for j in range(n_nodes):
                    Ke[i, j] += k * np.dot(dN_dx[i], dN_dx[j]) * det_J * weight
            
            # 右端项（热源）
            if heat_source:
                x_phys = np.sum(N[:, None] * element_coords, axis=0)
                source_value = heat_source(x_phys[0], x_phys[1])
                for i in range(n_nodes):
                    fe[i] += N[i] * source_value * det_J * weight
        
        return Ke, fe
    
    def _assemble_elasticity_system(self, mu: float, lambda_val: float, body_force: Optional[Callable]) -> Tuple[csr_matrix, np.ndarray]:
        """组装弹性力学系统"""
        n_nodes = self.mesh.n_nodes
        n_dofs = 2 * n_nodes
        
        # 初始化矩阵
        K = lil_matrix((n_dofs, n_dofs))
        f = np.zeros(n_dofs)
        
        # 遍历所有单元
        for element in self.mesh.elements:
            element_coords = self.mesh.coordinates[element]
            
            # 计算单元矩阵
            Ke, fe = self._compute_elasticity_element_matrix(element_coords, mu, lambda_val, body_force)
            
            # 组装到全局矩阵
            for i, node_i in enumerate(element):
                for j, node_j in enumerate(element):
                    # 2x2块矩阵
                    K[2*node_i:2*node_i+2, 2*node_j:2*node_j+2] += Ke[2*i:2*i+2, 2*j:2*j+2]
                f[2*node_i:2*node_i+2] += fe[2*i:2*i+2]
        
        return K.tocsr(), f
    
    def _compute_elasticity_element_matrix(self, element_coords: np.ndarray, mu: float, lambda_val: float, body_force: Optional[Callable]) -> Tuple[np.ndarray, np.ndarray]:
        """计算弹性力学单元矩阵"""
        n_nodes = len(element_coords)
        n_dofs = 2 * n_nodes
        Ke = np.zeros((n_dofs, n_dofs))
        fe = np.zeros(n_dofs)
        
        # 计算雅可比矩阵
        J = self._compute_jacobian(element_coords)
        J_inv = np.linalg.inv(J)
        det_J = abs(np.linalg.det(J))
        
        # 在积分点上计算
        for q, (xi, eta) in enumerate(self.quad_points):
            weight = self.quad_weights[q]
            
            # 基函数值
            N = self.basis_functions['N'](xi, eta)
            
            # 基函数导数
            dN_dxi = self.basis_functions['dN_dxi'](xi, eta)
            dN_dx = dN_dxi @ J_inv.T
            
            # 构建B矩阵（应变-位移矩阵）
            B = np.zeros((3, n_dofs))
            for i in range(n_nodes):
                B[0, 2*i] = dN_dx[i, 0]  # du/dx
                B[1, 2*i+1] = dN_dx[i, 1]  # dv/dy
                B[2, 2*i] = dN_dx[i, 1]  # du/dy
                B[2, 2*i+1] = dN_dx[i, 0]  # dv/dx
            
            # D矩阵（弹性矩阵）
            D = np.array([
                [lambda_val + 2*mu, lambda_val, 0],
                [lambda_val, lambda_val + 2*mu, 0],
                [0, 0, mu]
            ])
            
            # 刚度矩阵
            Ke += B.T @ D @ B * det_J * weight
            
            # 右端项（体积力）
            if body_force:
                x_phys = np.sum(N[:, None] * element_coords, axis=0)
                force = body_force(x_phys[0], x_phys[1])
                for i in range(n_nodes):
                    fe[2*i:2*i+2] += N[i] * force * det_J * weight
        
        return Ke, fe
    
    def _assemble_stokes_system(self, viscosity: float, body_force: Optional[Callable]) -> Tuple[csr_matrix, np.ndarray]:
        """组装Stokes系统"""
        n_nodes = self.mesh.n_nodes
        n_velocity_dofs = 2 * n_nodes
        n_pressure_dofs = n_nodes
        n_total_dofs = n_velocity_dofs + n_pressure_dofs
        
        # 初始化矩阵
        A = lil_matrix((n_total_dofs, n_total_dofs))
        b = np.zeros(n_total_dofs)
        
        # 遍历所有单元
        for element in self.mesh.elements:
            element_coords = self.mesh.coordinates[element]
            
            # 计算单元矩阵
            Ae, be = self._compute_stokes_element_matrix(element_coords, viscosity, body_force)
            
            # 组装到全局矩阵
            for i, node_i in enumerate(element):
                for j, node_j in enumerate(element):
                    # 速度-速度块
                    A[2*node_i:2*node_i+2, 2*node_j:2*node_j+2] += Ae[2*i:2*i+2, 2*j:2*j+2]
                    # 速度-压力块
                    A[2*node_i:2*node_i+2, n_velocity_dofs + node_j] += Ae[2*i:2*i+2, 2*len(element) + j]
                    A[n_velocity_dofs + node_i, 2*node_j:2*node_j+2] += Ae[2*len(element) + i, 2*j:2*j+2]
                
                b[2*node_i:2*node_i+2] += be[2*i:2*i+2]
                b[n_velocity_dofs + node_i] += be[2*len(element) + i]
        
        return A.tocsr(), b
    
    def _compute_stokes_element_matrix(self, element_coords: np.ndarray, viscosity: float, body_force: Optional[Callable]) -> Tuple[np.ndarray, np.ndarray]:
        """计算Stokes单元矩阵"""
        n_nodes = len(element_coords)
        n_velocity_dofs = 2 * n_nodes
        n_pressure_dofs = n_nodes
        n_total_dofs = n_velocity_dofs + n_pressure_dofs
        
        Ae = np.zeros((n_total_dofs, n_total_dofs))
        be = np.zeros(n_total_dofs)
        
        # 计算雅可比矩阵
        J = self._compute_jacobian(element_coords)
        J_inv = np.linalg.inv(J)
        det_J = abs(np.linalg.det(J))
        
        # 在积分点上计算
        for q, (xi, eta) in enumerate(self.quad_points):
            weight = self.quad_weights[q]
            
            # 基函数值
            N = self.basis_functions['N'](xi, eta)
            
            # 基函数导数
            dN_dxi = self.basis_functions['dN_dxi'](xi, eta)
            dN_dx = dN_dxi @ J_inv.T
            
            # 构建B矩阵（速度梯度矩阵）
            B = np.zeros((4, n_velocity_dofs))
            for i in range(n_nodes):
                B[0, 2*i] = dN_dx[i, 0]  # du/dx
                B[1, 2*i+1] = dN_dx[i, 1]  # dv/dy
                B[2, 2*i] = dN_dx[i, 1]  # du/dy
                B[3, 2*i+1] = dN_dx[i, 0]  # dv/dx
            
            # 粘度矩阵
            D = viscosity * np.array([
                [2, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            # 速度-速度块
            Ae[:n_velocity_dofs, :n_velocity_dofs] += B.T @ D @ B * det_J * weight
            
            # 压力-速度块（散度项）
            for i in range(n_nodes):
                for j in range(n_nodes):
                    # 散度项
                    div_term = dN_dx[i, 0] * N[j] + dN_dx[i, 1] * N[j]
                    Ae[n_velocity_dofs + i, 2*j] += div_term * det_J * weight
                    Ae[2*j, n_velocity_dofs + i] += div_term * det_J * weight
            
            # 右端项（体积力）
            if body_force:
                x_phys = np.sum(N[:, None] * element_coords, axis=0)
                force = body_force(x_phys[0], x_phys[1])
                for i in range(n_nodes):
                    be[2*i:2*i+2] += N[i] * force * det_J * weight
        
        return Ae, be
    
    def _compute_jacobian(self, element_coords: np.ndarray) -> np.ndarray:
        """计算雅可比矩阵"""
        dN_dxi = self.basis_functions['dN_dxi'](0, 0)  # 在参考单元中心
        return dN_dxi.T @ element_coords
    
    def _apply_boundary_conditions(self, K: csr_matrix, f: np.ndarray, boundary_conditions: Dict) -> Tuple[csr_matrix, np.ndarray]:
        """应用边界条件"""
        K_modified = K.copy()
        f_modified = f.copy()
        
        # Dirichlet边界条件
        if 'dirichlet' in boundary_conditions:
            for node_id, value in boundary_conditions['dirichlet'].items():
                K_modified[node_id, :] = 0
                K_modified[node_id, node_id] = 1
                f_modified[node_id] = value
        
        return K_modified, f_modified
    
    def _apply_elasticity_boundary_conditions(self, K: csr_matrix, f: np.ndarray, boundary_conditions: Dict) -> Tuple[csr_matrix, np.ndarray]:
        """应用弹性力学边界条件"""
        K_modified = K.copy()
        f_modified = f.copy()
        
        # Dirichlet边界条件
        if 'dirichlet' in boundary_conditions:
            for node_id, values in boundary_conditions['dirichlet'].items():
                for dof, value in enumerate(values):
                    dof_id = 2 * node_id + dof
                    K_modified[dof_id, :] = 0
                    K_modified[dof_id, dof_id] = 1
                    f_modified[dof_id] = value
        
        return K_modified, f_modified
    
    def _apply_stokes_boundary_conditions(self, A: csr_matrix, b: np.ndarray, boundary_conditions: Dict) -> Tuple[csr_matrix, np.ndarray]:
        """应用Stokes边界条件"""
        A_modified = A.copy()
        b_modified = b.copy()
        
        n_velocity_dofs = 2 * self.mesh.n_nodes
        
        # 速度边界条件
        if 'velocity' in boundary_conditions:
            for node_id, values in boundary_conditions['velocity'].items():
                for dof, value in enumerate(values):
                    dof_id = 2 * node_id + dof
                    A_modified[dof_id, :] = 0
                    A_modified[dof_id, dof_id] = 1
                    b_modified[dof_id] = value
        
        return A_modified, b_modified
    
    def _compute_stress(self, displacement: np.ndarray, mu: float, lambda_val: float) -> np.ndarray:
        """计算应力"""
        n_nodes = self.mesh.n_nodes
        stress = np.zeros((n_nodes, 3))  # [σxx, σyy, σxy]
        
        # 简化的应力计算
        for i, element in enumerate(self.mesh.elements):
            element_coords = self.mesh.coordinates[element]
            element_displacement = displacement[2*element[0]:2*element[0]+2]
            
            # 计算应变
            J = self._compute_jacobian(element_coords)
            J_inv = np.linalg.inv(J)
            dN_dxi = self.basis_functions['dN_dxi'](1/3, 1/3)  # 在重心
            dN_dx = dN_dxi @ J_inv.T
            
            # 计算应变
            strain = np.zeros(3)
            for j, node in enumerate(element):
                strain[0] += dN_dx[j, 0] * displacement[2*node]  # εxx
                strain[1] += dN_dx[j, 1] * displacement[2*node+1]  # εyy
                strain[2] += dN_dx[j, 1] * displacement[2*node] + dN_dx[j, 0] * displacement[2*node+1]  # εxy
            
            # 计算应力
            D = np.array([
                [lambda_val + 2*mu, lambda_val, 0],
                [lambda_val, lambda_val + 2*mu, 0],
                [0, 0, mu]
            ])
            element_stress = D @ strain
            
            # 分配到节点
            for node in element:
                stress[node] += element_stress / len(element)
        
        return stress
    
    def visualize_solution(self, solution_type: str = "temperature", 
                          show_mesh: bool = True, 
                          save_plot: bool = False,
                          filename: str = "solution.png"):
        """可视化解"""
        if self.solution is None:
            print("❌ 没有可用的解")
            return
        
        plt.figure(figsize=(12, 8))
        
        if solution_type == "temperature":
            # 温度场可视化
            plt.subplot(1, 2, 1)
            plt.tricontourf(self.mesh.coordinates[:, 0], 
                           self.mesh.coordinates[:, 1], 
                           self.solution, levels=20, cmap='hot')
            plt.colorbar(label='Temperature')
            plt.title('Temperature Field')
            plt.xlabel('x')
            plt.ylabel('y')
            
            if show_mesh:
                plt.triplot(self.mesh.coordinates[:, 0], 
                           self.mesh.coordinates[:, 1], 
                           self.mesh.elements, 'k-', alpha=0.3)
        
        elif solution_type == "displacement":
            # 位移场可视化
            displacement = self.solution
            magnitude = np.sqrt(displacement[::2]**2 + displacement[1::2]**2)
            
            plt.subplot(1, 2, 1)
            plt.tricontourf(self.mesh.coordinates[:, 0], 
                           self.mesh.coordinates[:, 1], 
                           magnitude, levels=20, cmap='viridis')
            plt.colorbar(label='Displacement Magnitude')
            plt.title('Displacement Field')
            plt.xlabel('x')
            plt.ylabel('y')
            
            # 位移向量场
            plt.subplot(1, 2, 2)
            skip = max(1, self.mesh.n_nodes // 100)  # 只显示部分向量
            plt.quiver(self.mesh.coordinates[::skip, 0], 
                      self.mesh.coordinates[::skip, 1],
                      displacement[::2*skip], 
                      displacement[1::2*skip],
                      scale=1.0, alpha=0.7)
            plt.title('Displacement Vectors')
            plt.xlabel('x')
            plt.ylabel('y')
        
        elif solution_type == "velocity":
            # 速度场可视化
            velocity = self.solution[:2*self.mesh.n_nodes]
            magnitude = np.sqrt(velocity[::2]**2 + velocity[1::2]**2)
            
            plt.subplot(1, 2, 1)
            plt.tricontourf(self.mesh.coordinates[:, 0], 
                           self.mesh.coordinates[:, 1], 
                           magnitude, levels=20, cmap='plasma')
            plt.colorbar(label='Velocity Magnitude')
            plt.title('Velocity Field')
            plt.xlabel('x')
            plt.ylabel('y')
            
            # 速度向量场
            plt.subplot(1, 2, 2)
            skip = max(1, self.mesh.n_nodes // 100)
            plt.quiver(self.mesh.coordinates[::skip, 0], 
                      self.mesh.coordinates[::skip, 1],
                      velocity[::2*skip], 
                      velocity[1::2*skip],
                      scale=1.0, alpha=0.7)
            plt.title('Velocity Vectors')
            plt.xlabel('x')
            plt.ylabel('y')
        
        if show_mesh:
            plt.triplot(self.mesh.coordinates[:, 0], 
                       self.mesh.coordinates[:, 1], 
                       self.mesh.elements, 'k-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📁 图片已保存: {filename}")
        
        plt.show()

def create_simple_mesh(nx: int = 10, ny: int = 10, 
                      x_min: float = 0.0, x_max: float = 1.0,
                      y_min: float = 0.0, y_max: float = 1.0) -> Mesh:
    """创建简单的矩形网格"""
    # 生成节点坐标
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    coordinates = np.column_stack([X.ravel(), Y.ravel()])
    
    # 生成三角形单元
    elements = []
    for j in range(ny-1):
        for i in range(nx-1):
            # 左下角节点
            n0 = j * nx + i
            n1 = j * nx + (i + 1)
            n2 = (j + 1) * nx + i
            n3 = (j + 1) * nx + (i + 1)
            
            # 两个三角形
            elements.append([n0, n1, n2])
            elements.append([n1, n3, n2])
    
    return Mesh(coordinates, np.array(elements))

def create_circle_mesh(radius: float = 1.0, n_radial: int = 10, n_angular: int = 20) -> Mesh:
    """创建圆形网格"""
    # 生成极坐标网格
    r = np.linspace(0, radius, n_radial)
    theta = np.linspace(0, 2*np.pi, n_angular, endpoint=False)
    
    # 转换为笛卡尔坐标
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    coordinates = np.column_stack([X.ravel(), Y.ravel()])
    
    # 生成三角形单元（简化实现）
    elements = []
    for j in range(n_angular-1):
        for i in range(n_radial-1):
            n0 = j * n_radial + i
            n1 = j * n_radial + (i + 1)
            n2 = (j + 1) * n_radial + i
            n3 = (j + 1) * n_radial + (i + 1)
            
            elements.append([n0, n1, n2])
            elements.append([n1, n3, n2])
    
    return Mesh(coordinates, np.array(elements)) 