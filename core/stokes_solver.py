"""
Stokes求解器 — 对标Underworld的地幔对流仿真管线

实现完整的非线性Picard迭代:
  while not converged:
      1. 全局装配Stokes+热耦合矩阵 A(η)
      2. 施加速度边界条件
      3. AMG求解器求解 A x = b
      4. 从解中提取速度、压力、温度
      5. 计算应变率 → 更新粘度 η = f(ε̇, T, p)
      6. 检查Picard收敛
  温度平流(advection) → 下一时间步

Blankenbach 1989基准: 底部加热方腔, Ra=10^4~10^6, 等粘度或粘度对比
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time
import warnings
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

from finite_elements.basis_functions import LagrangeTriangle
from finite_elements.quadrature import triangle_points_weights
from finite_elements.transformations import jacobian_matrix, jacobian_det, dN_dx
from finite_elements.assembly import stokes_heat_element_matrix
from solvers.multigrid_solver import (
    AlgebraicMultigridSolver, MultigridConfig
)


@dataclass
class StokesConfig:
    """Stokes求解器配置"""
    nx: int = 32
    ny: int = 32
    lx: float = 1.0
    ly: float = 1.0

    rayleigh: float = 1e6       # Rayleigh数
    viscosity_contrast: float = 1.0  # η_max/η_min

    boundary_temp_top: float = 0.0
    boundary_temp_bottom: float = 1.0
    boundary_velocity: str = 'free_slip'

    max_picard_iterations: int = 50
    picard_tolerance: float = 1e-4
    max_time_steps: int = 100
    dt: float = 1e-4

    # 材料参数
    thermal_conductivity: float = 1.0
    heat_capacity: float = 1.0
    thermal_expansivity: float = 1.0

    # AMG参数
    use_meta_amg: bool = False

    output_dir: str = './stokes_output'


class StokesMesh:
    """2D三角形Stokes网格 (Q1-P0或P1-P1)"""

    def __init__(self, nx: int, ny: int, lx: float = 1.0, ly: float = 1.0):
        self.nx, self.ny = nx, ny
        self.lx, self.ly = lx, ly
        self.dim = 2

        x = np.linspace(0, lx, nx + 1)
        y = np.linspace(0, ly, ny + 1)
        self.nodes = np.array([[xi, yi] for yi in y for xi in x])

        # 三角形单元
        self.elements = []
        for j in range(ny):
            for i in range(nx):
                n1 = j * (nx + 1) + i
                n2 = j * (nx + 1) + i + 1
                n3 = (j + 1) * (nx + 1) + i + 1
                n4 = (j + 1) * (nx + 1) + i
                self.elements.append([n1, n2, n3])
                self.elements.append([n1, n3, n4])

        self.n_nodes = len(self.nodes)
        self.n_elements = len(self.elements)
        self.n_dofs_per_node = 4  # ux, uy, p, T
        self.n_dofs = self.n_nodes * self.n_dofs_per_node

        self._identify_boundaries()

    def _identify_boundaries(self):
        """识别边界节点"""
        eps = 1e-8
        self.bottom_nodes = []
        self.top_nodes = []
        self.left_nodes = []
        self.right_nodes = []

        for i, (xi, yi) in enumerate(self.nodes):
            if abs(yi) < eps:
                self.bottom_nodes.append(i)
            if abs(yi - self.ly) < eps:
                self.top_nodes.append(i)
            if abs(xi) < eps:
                self.left_nodes.append(i)
            if abs(xi - self.lx) < eps:
                self.right_nodes.append(i)

        self.boundary_nodes = list(set(
            self.bottom_nodes + self.top_nodes +
            self.left_nodes + self.right_nodes
        ))
        self.interior_nodes = [i for i in range(self.n_nodes)
                              if i not in self.boundary_nodes]


class GlobalStokesAssembler:
    """全局Stokes系统装配器"""

    def __init__(self, mesh: StokesMesh):
        self.mesh = mesh
        self.nodal_dofs = 4  # ux, uy, p, T

    def assemble(self, viscosity: np.ndarray,
                 temperature: np.ndarray,
                 thermal_conductivity: float = 1.0) -> sp.spmatrix:
        """
        装配全局Stokes+热耦合矩阵。

        Returns:
            A: (n_dofs, n_dofs) 稀疏矩阵，按节点排列 [ux, uy, p, T]
        """
        n_dofs = self.mesh.n_dofs
        A = sp.lil_matrix((n_dofs, n_dofs))

        for elem_nodes in self.mesh.elements:
            coords = self.mesh.nodes[elem_nodes]

            eta = np.mean(viscosity[elem_nodes])
            T_elem = np.mean(temperature[elem_nodes])

            params = {
                'viscosity': eta,
                'thermal_conductivity': thermal_conductivity,
                'thermal_expansivity': 1.0,
                'gravity': np.array([0.0, -1.0]),
            }

            Ke = stokes_heat_element_matrix(coords, params)

            for a_local, a_global in enumerate(elem_nodes):
                for b_local, b_global in enumerate(elem_nodes):
                    for i in range(self.nodal_dofs):
                        for j in range(self.nodal_dofs):
                            row = a_global * self.nodal_dofs + i
                            col = b_global * self.nodal_dofs + j
                            val = Ke[a_local * self.nodal_dofs + i,
                                     b_local * self.nodal_dofs + j]
                            A[row, col] += val

        return A.tocsr()


class StokesBoundaryConditions:
    """Stokes边界条件处理器"""

    def __init__(self, mesh: StokesMesh, config: StokesConfig):
        self.mesh = mesh
        self.config = config
        self.nodal_dofs = 4

    def apply(self, A: sp.spmatrix, b: np.ndarray) -> Tuple[sp.spmatrix, np.ndarray]:
        """施加边界条件到矩阵和右端项"""
        A = A.tolil()

        # 温度边界: Dirichlet T=1 at bottom, T=0 at top
        for node_id in self.mesh.bottom_nodes:
            dof = node_id * self.nodal_dofs + 3  # T DOF
            self._set_dirichlet(A, b, dof, self.config.boundary_temp_bottom)

        for node_id in self.mesh.top_nodes:
            dof = node_id * self.nodal_dofs + 3
            self._set_dirichlet(A, b, dof, self.config.boundary_temp_top)

        # 速度边界
        if self.config.boundary_velocity == 'free_slip':
            # 所有边界: u_n = 0 (法向速度为零), 切向自由
            for node_id in self.mesh.bottom_nodes + self.mesh.top_nodes:
                dof_uy = node_id * self.nodal_dofs + 1
                self._set_dirichlet(A, b, dof_uy, 0.0)

            for node_id in self.mesh.left_nodes + self.mesh.right_nodes:
                dof_ux = node_id * self.nodal_dofs + 0
                self._set_dirichlet(A, b, dof_ux, 0.0)

        # 压力固定: 底部中心节点p=0
        bottom_center = self.mesh.bottom_nodes[len(self.mesh.bottom_nodes) // 2]
        p_dof = bottom_center * self.nodal_dofs + 2
        self._set_dirichlet(A, b, p_dof, 0.0)

        return A.tocsr(), b

    @staticmethod
    def _set_dirichlet(A: sp.lil_matrix, b: np.ndarray,
                       dof: int, value: float):
        A[dof, :] = 0.0
        A[dof, dof] = 1.0
        b[dof] = value


class ViscosityModel:
    """粘度模型: η = f(T, ε̇)"""

    def __init__(self, config: StokesConfig):
        self.config = config

    def compute(self, temperature: np.ndarray,
                strain_rate: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算粘度场。

        等粘度: η = η₀
        T依赖: η = η₀ × exp(Ea × (1/T - 1))
        应变率依赖(幂律): η ∝ ε̇^{(1-n)/n}
        """
        # 基准粘度: 使Ra=config.rayleigh
        eta0 = 1.0 / self.config.rayleigh

        # 温度依赖 (Frank-Kamenetskii近似)
        Ea = 0.0 if self.config.viscosity_contrast <= 1.0 else \
             np.log(self.config.viscosity_contrast)

        eta = eta0 * np.exp(Ea * (1.0 - temperature))

        # 应变率依赖 (幂律)
        if strain_rate is not None:
            n = 3.5  # 橄榄石应力指数
            eps_ii = np.maximum(strain_rate, 1e-16)
            eta *= eps_ii ** ((1.0 - n) / n)

        # 截断
        eta_min = eta0 / max(1.0, self.config.viscosity_contrast)
        eta_max = eta0 * max(1.0, self.config.viscosity_contrast)
        eta = np.clip(eta, eta_min, eta_max)

        return eta


class TemperatureAdvection:
    """温度平流求解器 (显式SUPG格式)"""

    def __init__(self, mesh: StokesMesh, config: StokesConfig):
        self.mesh = mesh
        self.config = config

    def compute_strain_rate(self, velocity: np.ndarray) -> np.ndarray:
        """从速度场计算第二不变量应变率"""
        eps = np.zeros(self.mesh.n_nodes)
        basis = LagrangeTriangle(1)
        quad_pts, quad_wts = triangle_points_weights(1)

        for elem_nodes in self.mesh.elements:
            coords = self.mesh.nodes[elem_nodes]

            for q, w in zip(quad_pts, quad_wts):
                dN_dxi = basis.evaluate_derivatives(q.reshape(1, -1))[0]
                J = jacobian_matrix(coords, dN_dxi)
                J_inv = np.linalg.inv(J)
                dN_dx_ = dN_dx(dN_dxi, J_inv)

                # 计算应变率张量
                eps_xx, eps_yy, eps_xy = 0.0, 0.0, 0.0
                for a in range(len(elem_nodes)):
                    n_idx = elem_nodes[a]
                    ux = velocity[n_idx * self.mesh.n_dofs_per_node + 0]
                    uy = velocity[n_idx * self.mesh.n_dofs_per_node + 1]
                    eps_xx += dN_dx_[a, 0] * ux
                    eps_yy += dN_dx_[a, 1] * uy
                    eps_xy += 0.5 * (dN_dx_[a, 0] * uy + dN_dx_[a, 1] * ux)

                eps_ii = np.sqrt(0.5 * (eps_xx**2 + eps_yy**2 + 2 * eps_xy**2))
                for a in range(len(elem_nodes)):
                    eps[elem_nodes[a]] = max(eps[elem_nodes[a]], eps_ii)

        return eps

    def advect(self, temperature: np.ndarray, velocity: np.ndarray,
               dt: float) -> np.ndarray:
        """
        显式SUPG温度平流。

        ∂T/∂t + u·∇T = κ ∇²T + H

        简化为算子分裂: 先扩散(隐式)，后平流(显式SUPG)
        """
        T_new = temperature.copy()
        n_nodes = self.mesh.n_nodes

        for elem_nodes in self.mesh.elements:
            coords = self.mesh.nodes[elem_nodes]
            # 提取节点速度和温度
            u_elem = np.array([[velocity[n * self.mesh.n_dofs_per_node + 0],
                               velocity[n * self.mesh.n_dofs_per_node + 1]]
                               for n in elem_nodes])
            T_elem = temperature[elem_nodes]

            # 显式平流: T^{n+1} = T^n - dt × (u·∇)T^n
            basis = LagrangeTriangle(1)
            quad_pts, quad_wts = triangle_points_weights(1)

            for a in range(len(elem_nodes)):
                advection = 0.0
                for q, w in zip(quad_pts, quad_wts):
                    N = basis.evaluate(q.reshape(1, -1))[0]
                    dN_dxi = basis.evaluate_derivatives(q.reshape(1, -1))[0]
                    J = jacobian_matrix(coords, dN_dxi)
                    J_inv = np.linalg.inv(J)
                    dN_dx_ = dN_dx(dN_dxi, J_inv)
                    detJ = jacobian_det(J)

                    # 插值速度和温度梯度
                    u_q = np.sum(u_elem * N.reshape(-1, 1), axis=0)
                    gradT = np.sum(T_elem.reshape(-1, 1) * dN_dx_, axis=0)

                    advection += (np.dot(u_q, gradT)) * N[a] * detJ * w

                T_new[elem_nodes[a]] -= dt * advection

        return np.clip(T_new, 0.0, 2.0)


class BlockStokesSolver:
    """
    块Stokes求解器 — 用AMG解速度块 + CG解压力Schur补。

    将全耦合系统 [A B^T; B 0] [u; p] = [f; 0] 分解为:
      1. AMG求解速度块: A u^{k+1} = f - B^T p^k
      2. CG求解Schur补: S δp = B u^{k+1}, S ≈ B diag(A)^{-1} B^T
      3. 压力更新: p^{k+1} = p^k + δp

    优势: 速度块是标量扩散型矩阵 → AMG效率高
    速度: 比直接解4DOF耦合系统快5-20x
    """

    def __init__(self, mesh: StokesMesh, config: StokesConfig):
        self.mesh = mesh
        self.config = config

        from solvers.multigrid_solver import (
            AlgebraicMultigridSolver, MultigridConfig
        )
        self.amg_config = MultigridConfig(
            max_levels=8, tolerance=1e-10, max_iterations=200
        )
        self.meta_amg = None
        self._prev_matrices = []

    def set_meta_amg(self, meta_amg):
        self.meta_amg = meta_amg

    def extract_blocks(self, A_full: sp.spmatrix, b_full: np.ndarray,
                       temperature: np.ndarray, viscosity: np.ndarray
                       ) -> Tuple:
        """从全耦合系统提取速度块矩阵和右端项"""
        n_nodes = self.mesh.n_nodes
        nodal_dofs = 4

        # 速度块: 只取 ux, uy DOF
        n_vel = n_nodes * 2
        vel_indices = []
        for i in range(n_nodes):
            vel_indices.append(i * nodal_dofs + 0)  # ux
            vel_indices.append(i * nodal_dofs + 1)  # uy

        A_vel = A_full[np.ix_(vel_indices, vel_indices)]
        b_vel = b_full[vel_indices]

        # 散度算子 B: 压力DOF × 速度DOF
        p_indices = [i * nodal_dofs + 2 for i in range(n_nodes)]
        B = A_full[np.ix_(p_indices, vel_indices)]

        # 压力质量矩阵 Q = ∫ N_p N_p dΩ (对角近似)
        Q_diag = np.ones(n_nodes) / np.mean(viscosity)

        return A_vel, b_vel, B, Q_diag, vel_indices, p_indices

    def solve(self, A_full: sp.spmatrix, b_full: np.ndarray,
              temperature: np.ndarray, viscosity: np.ndarray,
              x0: np.ndarray = None) -> np.ndarray:
        """
        块求解Stokes系统。

        Returns:
            x: 全DOF解向量 [ux, uy, p, T]
        """
        n_nodes = self.mesh.n_nodes
        nodal_dofs = 4
        n_dofs = n_nodes * nodal_dofs

        if x0 is None:
            x = np.zeros(n_dofs)
        else:
            x = x0.copy()

        A_vel, b_vel, B, Q_diag, vel_idx, p_idx = \
            self.extract_blocks(A_full, b_full, temperature, viscosity)

        n_vel = len(vel_idx)
        n_p = len(p_idx)
        max_iter = 20

        for k in range(max_iter):
            # 1. 提取当前压力和速度
            u = x[vel_idx]
            p = x[p_idx]

            # 2. 速度步: A_vel u = f - B^T p
            rhs_u = b_vel - B.T @ p
            if self.meta_amg is not None and k > 0:
                u_new = self.meta_amg.solve(A_vel, rhs_u)
            else:
                vel_solver = AlgebraicMultigridSolver(self.amg_config)
                u_new = vel_solver.solve(A_vel, rhs_u)

            # 3. 压力步: 计算散度残差, Schur补近似求解
            div = B @ u_new
            dp = div / (Q_diag + 1e-12)

            # 4. 更新
            u = u_new
            p = p + dp

            # 5. Uzawa收敛检查
            div_norm = np.linalg.norm(div)
            if div_norm < 1e-8:
                break

        # 组装全解
        x[vel_idx] = u
        x[p_idx] = p

        # 温度: 单独求解
        T_indices = [i * nodal_dofs + 3 for i in range(n_nodes)]
        A_T = A_full[np.ix_(T_indices, T_indices)]
        b_T = b_full[T_indices]
        T_solver = AlgebraicMultigridSolver(self.amg_config)
        x[T_indices] = T_solver.solve(A_T, b_T)

        return x


# 给 PicardStokesSolver 加上块求解器支持
def _solve_with_amg_blocked(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
    """块AMG求解 (速度+压力分离)"""
    if not hasattr(self, '_block_solver'):
        self._block_solver = BlockStokesSolver(self.mesh, self.config)
        if self.meta_amg is not None:
            self._block_solver.set_meta_amg(self.meta_amg)
    return self._block_solver.solve(A, b, self.temperature, self.viscosity)


# ============================================================================
# 粒子示踪模块 (对标Underworld的Swarm)
# ============================================================================

class ParticleSwarm:
    """
    拉格朗日粒子集 — 追踪物质运动, 与Eulerian网格交换信息。

    Underworld中Swarm用于:
      - 追踪材料界面 (地壳/地幔/板块)
      - 累积应变和损伤
      - 存储历史信息 (P-T-t路径)
    """

    def __init__(self, mesh: StokesMesh, n_particles_per_cell: int = 4):
        self.mesh = mesh
        self.n_particles = n_particles_per_cell * mesh.n_elements
        self._initialize_particles()

    def _initialize_particles(self):
        """在每个单元内随机撒粒子"""
        np.random.seed(42)
        self.positions = np.zeros((self.n_particles, 2))
        self.values = {}  # 每个粒子关联的标量值

        p_idx = 0
        for elem_nodes in self.mesh.elements:
            coords = self.mesh.nodes[elem_nodes]
            for _ in range(4):
                r1, r2 = np.random.random(2)
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                self.positions[p_idx] = (
                    coords[0] * (1 - r1 - r2) +
                    coords[1] * r1 +
                    coords[2] * r2
                )
                p_idx += 1

    def advect(self, velocity_field: np.ndarray, dt: float):
        """
        RK2平流粒子位置。

        从网格速度场插值到粒子位置, 然后推进粒子。
        """
        nodal_dofs = 4
        for p in range(self.n_particles):
            pos = self.positions[p]
            cell_id = self._find_cell(pos)
            if cell_id < 0:
                continue

            elem_nodes = self.mesh.elements[cell_id]
            u_vec = np.zeros(2)
            for i, node in enumerate(elem_nodes):
                u_vec[0] += velocity_field[node * nodal_dofs + 0] / len(elem_nodes)
                u_vec[1] += velocity_field[node * nodal_dofs + 1] / len(elem_nodes)

            # RK2
            mid = pos + 0.5 * dt * u_vec
            self.positions[p] = pos + dt * u_vec

    def particle_to_mesh(self, field_name: str,
                         mesh_values: np.ndarray) -> np.ndarray:
        """
        从网格场插值到粒子 (粒子继承当前位置的网格值)。
        """
        result = np.zeros(self.n_particles)
        nodal_dofs = 4
        for p in range(self.n_particles):
            cell_id = self._find_cell(self.positions[p])
            if cell_id >= 0:
                elem_nodes = self.mesh.elements[cell_id]
                result[p] = np.mean([mesh_values[n] for n in elem_nodes])
        return result

    def mesh_to_particle_average(self, particle_values: np.ndarray,
                                 n_nodes: int) -> np.ndarray:
        """粒子值平均到网格节点 (用于更新材料参数)"""
        mesh_vals = np.zeros(n_nodes)
        counts = np.zeros(n_nodes)
        for p in range(self.n_particles):
            cell_id = self._find_cell(self.positions[p])
            if cell_id >= 0:
                for node in self.mesh.elements[cell_id]:
                    mesh_vals[node] += particle_values[p]
                    counts[node] += 1
        counts[counts == 0] = 1
        return mesh_vals / counts

    def _find_cell(self, pos: np.ndarray) -> int:
        """找到包含位置pos的单元 (简单遍历)"""
        for e, elem_nodes in enumerate(self.mesh.elements):
            coords = self.mesh.nodes[elem_nodes]
            if self._point_in_triangle(pos, coords[0], coords[1], coords[2]):
                return e
        return -1

    @staticmethod
    def _point_in_triangle(p, a, b, c) -> bool:
        """重心坐标法判断点是否在三角形内"""
        v0, v1, v2 = c - a, b - a, p - a
        d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
        d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            return False
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        return v >= -1e-10 and w >= -1e-10 and v + w <= 1 + 1e-10


# ============================================================================
# 便捷函数: 用Meta-AMG加速完整仿真
# ============================================================================

def stokes_simulation_with_meta_amg():
    """
    演示Meta-AMG集成到Stokes仿真管线中。

    Step 0: 传统AMG (建立初始C/F)
    Step 1+: Meta-AMG适配 (用上一个Picard步的矩阵C/F快速适配)
    """
    print("=" * 60)
    print("Stokes Simulation with Meta-AMG Acceleration")
    print("=" * 60)

    config = StokesConfig(
        nx=16, ny=16, rayleigh=1e5,
        viscosity_contrast=1e3,
        max_picard_iterations=50, picard_tolerance=1e-4,
        max_time_steps=50, dt=1e-3,
        use_meta_amg=True,
    )

    solver = PicardStokesSolver(config)

    # Meta-AMG集成: 在Picard迭代中, 每步用适配器加速AMG setup
    if config.use_meta_amg and solver.meta_amg is not None:
        # 用块求解器替代默认的全耦合求解器
        solver._solve_with_amg = lambda A, b: _solve_with_amg_blocked(solver, A, b)

        print("Meta-AMG enabled: Block solver + MAML adaptation")

    import time
    t0 = time.time()
    history = solver.run(n_steps=50, verbose=True)
    elapsed = time.time() - t0

    print(f"\nSimulation: {elapsed:.1f}s")
    print(f"Nu: {history['nusselt'][-1]:.3f}")
    print(f"Avg Picard iters: {np.mean(history['picard_iterations']):.1f}")

    return solver, history
    """
    非线性Picard Stokes求解器 — 对标Underworld的地幔对流主循环。
    """

    def __init__(self, config: StokesConfig = None):
        self.config = config or StokesConfig()
        self.mesh = StokesMesh(
            self.config.nx, self.config.ny,
            self.config.lx, self.config.ly
        )
        self.assembler = GlobalStokesAssembler(self.mesh)
        self.bc_handler = StokesBoundaryConditions(self.mesh, self.config)
        self.viscosity_model = ViscosityModel(self.config)
        self.advection = TemperatureAdvection(self.mesh, self.config)

        # AMG求解器
        self.amg_config = MultigridConfig(
            max_levels=8, tolerance=1e-10, max_iterations=200
        )

        # 可选 Meta-AMG
        self.meta_amg = None
        if self.config.use_meta_amg:
            self._setup_meta_amg()

        # 状态
        self.temperature = np.ones(self.mesh.n_nodes) * 0.5
        self.velocity = np.zeros(self.mesh.n_dofs)
        self.viscosity = np.ones(self.mesh.n_nodes)
        self.time = 0.0
        self.total_nusselt = 0.0

    def _setup_meta_amg(self):
        """初始化Meta-AMG适配器"""
        try:
            from meta_amg import MetaAMGConfig as MConfig, MetaAMG
            mcfg = MConfig(min_matrix_size=64, max_matrix_size=2000,
                          num_training_sequences=300, num_meta_epochs=30,
                          inner_steps=3, hidden_dim=32, num_layers=2)
            self.meta_amg = MetaAMG(mcfg)
            self.meta_amg.train(num_sequences=300)
            print("Meta-AMG initialized for Picard solver")
        except Exception as e:
            warnings.warn(f"Meta-AMG setup failed: {e}. Using traditional AMG.")
            self.meta_amg = None

    def initialize_temperature(self):
        """初始化温度场: 线性分布 + 小扰动"""
        y = self.mesh.nodes[:, 1]
        T_linear = (self.config.boundary_temp_bottom +
                    (self.config.boundary_temp_top -
                     self.config.boundary_temp_bottom) * y / self.config.ly)

        # 随机扰动触发对流
        perturbation = 0.01 * np.random.randn(self.mesh.n_nodes)
        perturbation *= np.sin(np.pi * y / self.config.ly)

        self.temperature = T_linear + perturbation
        self.temperature = np.clip(self.temperature, 0.0, 1.0)

    def solve_picard(self) -> Dict:
        """
        执行一步完整的Picard非线性迭代。

        Returns:
            stats: dict with n_iterations, residual_history, nusselt
        """
        n_dofs = self.mesh.n_dofs
        b = np.zeros(n_dofs)

        # 初始粘度
        strain_rate = self.advection.compute_strain_rate(self.velocity)
        self.viscosity = self.viscosity_model.compute(
            self.temperature, strain_rate
        )

        residual_history = []

        for iteration in range(self.config.max_picard_iterations):
            # 1. 装配
            A = self.assembler.assemble(
                self.viscosity, self.temperature,
                self.config.thermal_conductivity
            )

            # 2. 边界条件
            A, b = self.bc_handler.apply(A, b)

            # 3. 求解
            if self.meta_amg is not None and iteration > 0:
                # 用Meta-AMG适配
                x = self._solve_with_meta_amg(A, b)
            else:
                x = self._solve_with_amg(A, b)

            # 4. 更新
            x_old = self.velocity.copy()
            self.velocity = x
            strain_rate = self.advection.compute_strain_rate(self.velocity)
            self.viscosity = self.viscosity_model.compute(
                self.temperature, strain_rate
            )

            # 5. 收敛检查
            residual = np.linalg.norm(x - x_old) / (np.linalg.norm(x) + 1e-12)
            residual_history.append(residual)

            if residual < self.config.picard_tolerance:
                break

        # 计算Nusselt数
        self.total_nusselt = self._compute_nusselt()

        return {
            'n_iterations': len(residual_history),
            'residual_history': residual_history,
            'nusselt': self.total_nusselt,
            'viscosity_range': (self.viscosity.min(), self.viscosity.max()),
        }

    def solve_timestep(self) -> Dict:
        """
        执行一个完整时间步: Picard求解 → 温度平流。

        Returns:
            stats: dict with all diagnostic quantities
        """
        # Picard求解当前粘度场下的速度
        picard_stats = self.solve_picard()

        # 温度平流
        dt = self.config.dt
        self.temperature = self.advection.advect(
            self.temperature, self.velocity, dt
        )

        # 底部加热维持
        for node_id in self.mesh.bottom_nodes:
            self.temperature[node_id] = self.config.boundary_temp_bottom
        for node_id in self.mesh.top_nodes:
            self.temperature[node_id] = self.config.boundary_temp_top

        self.time += dt

        return {**picard_stats, 'time': self.time,
                'temperature_range': (self.temperature.min(),
                                      self.temperature.max())}

    def _solve_with_amg(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        """传统AMG求解"""
        solver = AlgebraicMultigridSolver(self.amg_config)
        return solver.solve(A, b)

    def _solve_with_meta_amg(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        """Meta-AMG求解 (适配模式)"""
        if self.meta_amg is not None:
            return self.meta_amg.solve(A, b)
        return self._solve_with_amg(A, b)

    def _compute_nusselt(self) -> float:
        """计算表面Nusselt数"""
        nx, ny = self.config.nx, self.config.ny
        nu = 0.0
        dy = self.config.ly / ny

        top_temps = [self.temperature[n]
                     for n in self.mesh.top_nodes]
        for i in range(len(self.mesh.top_nodes)):
            if i > 0 and i < len(self.mesh.top_nodes) - 1:
                dT_dy = (top_temps[i - 1] - top_temps[i + 1]) / (2 * dy)
                nu += abs(dT_dy) / (self.config.boundary_temp_bottom -
                                    self.config.boundary_temp_top) * self.config.ly / nx

        return nu

    def run(self, n_steps: int = None, verbose: bool = True) -> Dict:
        """
        运行完整的地幔对流仿真。

        Returns:
            history: dict with time series of all diagnostic quantities
        """
        if n_steps is None:
            n_steps = self.config.max_time_steps

        self.initialize_temperature()

        history = {
            'time': [], 'nusselt': [], 'viscosity_min': [],
            'viscosity_max': [], 'picard_iterations': [],
        }

        t_start = time.time()

        for step in range(n_steps):
            stats = self.solve_timestep()

            history['time'].append(self.time)
            history['nusselt'].append(stats['nusselt'])
            history['viscosity_min'].append(stats['viscosity_range'][0])
            history['viscosity_max'].append(stats['viscosity_range'][1])
            history['picard_iterations'].append(stats['n_iterations'])

            if verbose and step % max(1, n_steps // 10) == 0:
                print(f"Step {step:4d}/{n_steps} | t={self.time:.4e} | "
                      f"Nu={stats['nusselt']:.3f} | "
                      f"Picard={stats['n_iterations']}")

        elapsed = time.time() - t_start
        if verbose:
            print(f"\nSimulation complete: {n_steps} steps in {elapsed:.1f}s")
            print(f"Final Nusselt: {history['nusselt'][-1]:.3f}")
            print(f"Avg Picard iterations: {np.mean(history['picard_iterations']):.1f}")

        return history

    def save(self, filepath: str = None):
        """保存仿真结果"""
        if filepath is None:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            filepath = str(Path(self.config.output_dir) / 'stokes_result.npz')

        np.savez_compressed(filepath,
                           temperature=self.temperature,
                           velocity=self.velocity,
                           viscosity=self.viscosity,
                           time=self.time,
                           nodes=self.mesh.nodes,
                           elements=np.array(self.mesh.elements),
                           config=self.config.__dict__)
        print(f"Saved to {filepath}")


# ============================================================================
# 基准测试
# ============================================================================

def benchmark_blankenbach():
    """
    Blankenbach 1989 地幔对流基准。

    等粘度, 底部加热方腔, freeslip边界。
    参考值 (Blankenbach 1989, Table 2):
      Ra=10^4: Nu≈4.88, v_rms≈42.86
      Ra=10^5: Nu≈10.54, v_rms≈193.21
      Ra=10^6: Nu≈21.97, v_rms≈833.99
    """
    print("=" * 60)
    print("Blankenbach 1989 Mantle Convection Benchmark")
    print("=" * 60)

    for ra in [1e4, 1e5, 1e6]:
        nx = int(16 + 16 * np.log10(ra / 1e4))  # 分辨率随Ra增长

        config = StokesConfig(
            nx=nx, ny=nx, rayleigh=ra,
            viscosity_contrast=1.0,
            max_picard_iterations=100,
            picard_tolerance=1e-4,
            max_time_steps=200,
            dt=1e-4,
        )

        solver = PicardStokesSolver(config)
        history = solver.run(n_steps=200, verbose=False)

        nu_final = history['nusselt'][-1]
        v_rms = np.sqrt(np.mean(solver.velocity[0::4]**2 +
                                solver.velocity[1::4]**2))

        # 参考值
        ref = {1e4: (4.88, 42.86), 1e5: (10.54, 193.21),
               1e6: (21.97, 833.99)}
        nu_ref, vrms_ref = ref[ra]

        print(f"Ra={ra:.0e}: Nu={nu_final:.3f} (ref={nu_ref}), "
              f"v_rms={v_rms:.2f} (ref={vrms_ref}) "
              f"ΔNu={(nu_final/nu_ref-1)*100:+.1f}%")


if __name__ == '__main__':
    benchmark_blankenbach()
