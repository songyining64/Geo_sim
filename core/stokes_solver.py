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
from scipy.sparse.linalg import spsolve, lsmr, cg, gmres, splu, LinearOperator
import time
import warnings
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

from finite_elements.basis_functions import LagrangeTriangle
from finite_elements.quadrature import triangle_points_weights
from finite_elements.transformations import jacobian_matrix, jacobian_det, dN_dx
from finite_elements.assembly import stokes_heat_element_matrix, stokes_heat_element_matrix_3d
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
    meta_training_sequences: int = 20
    meta_training_epochs: int = 20
    meta_model_path: Optional[str] = None

    # Blocked Stokes solver stabilization
    meta_adapt_steps: int = 3
    schur_scale: float = 0.5
    velocity_solver_rtol: float = 1e-8
    velocity_acceptance_rtol: float = 1e-6
    full_blocked_acceptance_rtol: float = 1e-4
    pressure_preconditioner: str = 'diag_schur'
    velocity_jacobi_relaxation: float = 0.5
    velocity_smoothing_steps: int = 1

    output_dir: str = './stokes_output'


class StokesMesh:
    """2D/3D Stokes网格"""

    def __init__(self, nx: int, ny: int, nz: int = None,
                 lx: float = 1.0, ly: float = 1.0, lz: float = 1.0):
        self.nx, self.ny = nx, ny
        self.lx, self.ly = lx, ly
        self.nz = nz
        self.lz = lz
        self.dim = 2 if nz is None else 3
        self.n_dofs_per_node = 5 if self.dim == 3 else 4  # ux,uy,(uz),p,T

        if self.dim == 2:
            self._build_2d_mesh()
        else:
            self._build_3d_mesh()

        self._identify_boundaries()

    def _build_2d_mesh(self):
        nx, ny, lx, ly = self.nx, self.ny, self.lx, self.ly
        x = np.linspace(0, lx, nx + 1)
        y = np.linspace(0, ly, ny + 1)
        self.nodes = np.array([[xi, yi] for yi in y for xi in x])
        self.elements = []
        for j in range(ny):
            for i in range(nx):
                n1 = j * (nx + 1) + i
                n2 = n1 + 1; n3 = n1 + nx + 2; n4 = n1 + nx + 1
                self.elements.append([n1, n2, n3])
                self.elements.append([n1, n3, n4])
        self.n_nodes = len(self.nodes)
        self.n_elements = len(self.elements)
        self.n_dofs = self.n_nodes * self.n_dofs_per_node

    def _build_3d_mesh(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        lx, ly, lz = self.lx, self.ly, self.lz
        x = np.linspace(0, lx, nx + 1)
        y = np.linspace(0, ly, ny + 1)
        z = np.linspace(0, lz, nz + 1)
        self.nodes = np.array([[xi, yi, zi] for zi in z for yi in y for xi in x])
        self.elements = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    nxy = (nx + 1) * (ny + 1)
                    base = k * nxy + j * (nx + 1) + i
                    n000, n100 = base, base + 1
                    n010, n110 = base + nx + 1, base + nx + 2
                    n001, n101 = base + nxy, base + nxy + 1
                    n011, n111 = base + nxy + nx + 1, base + nxy + nx + 2
                    tets = [
                        [n000, n100, n010, n001], [n100, n110, n010, n001],
                        [n100, n110, n111, n001], [n100, n101, n111, n001],
                        [n010, n110, n111, n011], [n010, n111, n001, n011]
                    ]
                    self.elements.extend(tets)
        self.n_nodes = len(self.nodes)
        self.n_elements = len(self.elements)
        self.n_dofs = self.n_nodes * self.n_dofs_per_node

    def _identify_boundaries(self):
        eps = 1e-8
        self.bottom_nodes = [i for i, n in enumerate(self.nodes) if abs(n[-1]) < eps]
        self.top_nodes = [i for i, n in enumerate(self.nodes)
                         if abs(n[-1] - (self.lz if self.dim == 3 else self.ly)) < eps]
        self.left_nodes = [i for i, n in enumerate(self.nodes) if abs(n[0]) < eps]
        self.right_nodes = [i for i, n in enumerate(self.nodes)
                           if abs(n[0] - self.lx) < eps]
        if self.dim == 3:
            self.front_nodes = [i for i, n in enumerate(self.nodes) if abs(n[1]) < eps]
            self.back_nodes = [i for i, n in enumerate(self.nodes)
                              if abs(n[1] - self.ly) < eps]
        self.boundary_nodes = list(set(
            self.bottom_nodes + self.top_nodes +
            self.left_nodes + self.right_nodes +
            (self.front_nodes if self.dim == 3 else []) +
            (self.back_nodes if self.dim == 3 else [])
        ))
        self.interior_nodes = [i for i in range(self.n_nodes)
                              if i not in self.boundary_nodes]


class GlobalStokesAssembler:
    """全局Stokes系统装配器"""

    def __init__(self, mesh: StokesMesh):
        self.mesh = mesh

    def assemble_pressure_mass_matrix(self, viscosity: np.ndarray) -> sp.spmatrix:
        """
        真实FEM装配压力质量矩阵: Q[i,j] = ∫ (1/η) N_i N_j dΩ

        用于Schur补近似 S ≈ B diag(A)^{-1} B^T 的改进版。
        """
        n_nodes = self.mesh.n_nodes
        Q = sp.lil_matrix((n_nodes, n_nodes))

        from finite_elements.basis_functions import LagrangeTriangle, LagrangeTetra
        from finite_elements.quadrature import triangle_points_weights, tetra_points_weights
        from finite_elements.transformations import jacobian_matrix, jacobian_det

        BasisClass = LagrangeTetra if self.mesh.dim == 3 else LagrangeTriangle
        quad_func = tetra_points_weights if self.mesh.dim == 3 else triangle_points_weights
        basis = BasisClass(1)
        quad_pts, quad_wts = quad_func(2)

        for elem_nodes in self.mesh.elements:
            coords = self.mesh.nodes[elem_nodes]
            eta_inv = 1.0 / (np.mean(viscosity[elem_nodes]) + 1e-12)

            for q, w in zip(quad_pts, quad_wts):
                N = basis.evaluate(q.reshape(1, -1))[0]
                dN_dxi = basis.evaluate_derivatives(q.reshape(1, -1))[0]
                J = jacobian_matrix(coords, dN_dxi)
                detJ = jacobian_det(J)
                dV = detJ * w

                for a in range(len(elem_nodes)):
                    for b in range(len(elem_nodes)):
                        Q[elem_nodes[a], elem_nodes[b]] += eta_inv * N[a] * N[b] * dV

        return Q.tocsr()

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
        ndpn = self.mesh.n_dofs_per_node

        for elem_nodes in self.mesh.elements:
            coords = self.mesh.nodes[elem_nodes]
            eta = np.mean(viscosity[elem_nodes])

            params = {'viscosity': eta,
                     'thermal_conductivity': thermal_conductivity,
                     'thermal_expansivity': 1.0,
                     'gravity': np.array([0.0, -1.0])}

            # 根据维度选择单元矩阵
            if self.mesh.dim == 3:
                Ke = stokes_heat_element_matrix_3d(coords, params)
            else:
                Ke = stokes_heat_element_matrix(coords, params)

            for a_local, a_global in enumerate(elem_nodes):
                for b_local, b_global in enumerate(elem_nodes):
                    for i in range(ndpn):
                        for j in range(ndpn):
                            row = a_global * ndpn + i
                            col = b_global * ndpn + j
                            A[row, col] += Ke[a_local * ndpn + i,
                                              b_local * ndpn + j]

        return A.tocsr()

    def assemble_rhs(self, temperature: np.ndarray,
                     rayleigh: float,
                     thermal_expansivity: float = 1.0) -> np.ndarray:
        """Assemble buoyancy forcing for the coupled 2D Stokes system."""
        if self.mesh.dim != 2:
            raise NotImplementedError("Buoyancy RHS currently implemented for 2D Stokes only")

        n_dofs = self.mesh.n_dofs
        rhs = np.zeros(n_dofs)
        ndpn = self.mesh.n_dofs_per_node
        basis = LagrangeTriangle(1)
        quad_pts, quad_wts = triangle_points_weights(2)
        gravity = np.array([0.0, -1.0])

        for elem_nodes in self.mesh.elements:
            coords = self.mesh.nodes[elem_nodes]
            T_elem = temperature[elem_nodes]

            for q, w in zip(quad_pts, quad_wts):
                N = basis.evaluate(q.reshape(1, -1))[0]
                dN_dxi = basis.evaluate_derivatives(q.reshape(1, -1))[0]
                J = jacobian_matrix(coords, dN_dxi)
                detJ = jacobian_det(J)
                dV = detJ * w
                # Remove the hydrostatic conductive reference profile; only the
                # temperature anomaly should drive convection in Boussinesq form.
                T_q = float(np.dot(N, T_elem) - 0.5)
                body_force = rayleigh * thermal_expansivity * T_q * gravity

                for a_local, a_global in enumerate(elem_nodes):
                    for i in range(self.mesh.dim):
                        row = a_global * ndpn + i
                        rhs[row] += N[a_local] * body_force[i] * dV

        return rhs


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
        # 基准粘度采用无量纲 O(1) 标度；Rayleigh 数通过浮力项进入动量方程。
        eta0 = 1.0

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

    支持Meta-AMG在线适配:
      - 第一个Picard迭代: 传统AMG (建立初始C/F)
      - 后续迭代: Meta-AMG从上一矩阵的C/F适配到当前矩阵
    """

    def __init__(self, mesh: StokesMesh, config: StokesConfig):
        self.mesh = mesh
        self.config = config
        self.assembler = GlobalStokesAssembler(mesh)

        from solvers.multigrid_solver import (
            AlgebraicMultigridSolver, MultigridConfig
        )
        self.amg_config = MultigridConfig(
            max_levels=8, tolerance=1e-10, max_iterations=200,
            smoother='jacobi', relaxation_factor=config.velocity_jacobi_relaxation,
            pre_smoothing_steps=config.velocity_smoothing_steps,
            post_smoothing_steps=config.velocity_smoothing_steps,
        )
        self.meta_amg = None
        self._prev_A_vel = None          # 上一个Picard步的速度块矩阵
        self._prev_cf_labels = None      # 上一个速度块的C/F标签
        self._Q_matrix = None
        self._prev_viscosity = None
        self._picard_step = 0
        self.velocity_direct_fallbacks = 0
        self.velocity_solve_calls = 0
        self.velocity_cg_iterations = 0
        self.full_direct_fallbacks = 0
        self.velocity_direct_fallback_time = 0.0
        self.full_direct_fallback_time = 0.0
        self.pressure_krylov_iterations = 0
        self.pressure_solver_fallbacks = 0
        self.pressure_preconditioner_fallbacks = 0
        self.traditional_hierarchy_setups = 0
        self.adapted_hierarchy_setups = 0
        self.traditional_hierarchy_setup_time = 0.0
        self.adapted_hierarchy_setup_time = 0.0
        self.meta_adaptation_time = 0.0
        self.traditional_velocity_krylov_iterations = 0
        self.adapted_velocity_krylov_iterations = 0
        self.traditional_velocity_solve_calls = 0
        self.adapted_velocity_solve_calls = 0
        self.velocity_solver_breakdown = {
            'rhs': 0,
            'schur_columns': 0,
            'back_substitution': 0,
        }

    def reset_stats(self):
        self.velocity_direct_fallbacks = 0
        self.velocity_solve_calls = 0
        self.velocity_cg_iterations = 0
        self.full_direct_fallbacks = 0
        self.velocity_direct_fallback_time = 0.0
        self.full_direct_fallback_time = 0.0
        self.pressure_krylov_iterations = 0
        self.pressure_solver_fallbacks = 0
        self.pressure_preconditioner_fallbacks = 0
        self.traditional_hierarchy_setups = 0
        self.adapted_hierarchy_setups = 0
        self.traditional_hierarchy_setup_time = 0.0
        self.adapted_hierarchy_setup_time = 0.0
        self.meta_adaptation_time = 0.0
        self.traditional_velocity_krylov_iterations = 0
        self.adapted_velocity_krylov_iterations = 0
        self.traditional_velocity_solve_calls = 0
        self.adapted_velocity_solve_calls = 0
        self.velocity_solver_breakdown = {
            'rhs': 0,
            'schur_columns': 0,
            'back_substitution': 0,
        }

    def get_stats(self) -> Dict:
        return {
            'velocity_solve_calls': self.velocity_solve_calls,
            'velocity_direct_fallbacks': self.velocity_direct_fallbacks,
            'velocity_cg_iterations': self.velocity_cg_iterations,
            'velocity_krylov_iterations': self.velocity_cg_iterations,
            'full_direct_fallbacks': self.full_direct_fallbacks,
            'velocity_direct_fallback_time': self.velocity_direct_fallback_time,
            'full_direct_fallback_time': self.full_direct_fallback_time,
            'pressure_krylov_iterations': self.pressure_krylov_iterations,
            'pressure_solver_fallbacks': self.pressure_solver_fallbacks,
            'pressure_preconditioner_fallbacks': self.pressure_preconditioner_fallbacks,
            'traditional_hierarchy_setups': self.traditional_hierarchy_setups,
            'adapted_hierarchy_setups': self.adapted_hierarchy_setups,
            'traditional_hierarchy_setup_time': self.traditional_hierarchy_setup_time,
            'adapted_hierarchy_setup_time': self.adapted_hierarchy_setup_time,
            'meta_adaptation_time': self.meta_adaptation_time,
            'traditional_velocity_krylov_iterations': self.traditional_velocity_krylov_iterations,
            'adapted_velocity_krylov_iterations': self.adapted_velocity_krylov_iterations,
            'traditional_velocity_solve_calls': self.traditional_velocity_solve_calls,
            'adapted_velocity_solve_calls': self.adapted_velocity_solve_calls,
            'velocity_solver_breakdown': dict(self.velocity_solver_breakdown),
        }

    def set_meta_amg(self, meta_amg):
        self.meta_amg = meta_amg

    def _solve_velocity_system(self, A_vel: sp.spmatrix, rhs: np.ndarray, solver,
                               phase: str = 'rhs') -> np.ndarray:
        """Solve the SPD velocity block with CG + AMG preconditioning.

        The AMG hierarchy is used as a preconditioner via one V-cycle per GMRES
        application, not as a standalone approximate direct solver.
        """
        rhs = np.asarray(rhs).reshape(-1)
        self.velocity_solve_calls += 1
        if getattr(solver, '_hierarchy_source', 'traditional') == 'adapted':
            self.adapted_velocity_solve_calls += 1
        else:
            self.traditional_velocity_solve_calls += 1
        self.velocity_solver_breakdown[phase] = self.velocity_solver_breakdown.get(phase, 0) + 1
        try:
            if not getattr(solver, 'is_setup', False):
                setup_start = time.time()
                solver.setup(A_vel, rhs)
                setup_time = time.time() - setup_start
                if getattr(solver, '_hierarchy_source', 'traditional') == 'adapted':
                    self.adapted_hierarchy_setups += 1
                    self.adapted_hierarchy_setup_time += setup_time
                else:
                    self.traditional_hierarchy_setups += 1
                    self.traditional_hierarchy_setup_time += setup_time

            def precondition(vec):
                return solver.v_cycle(0, vec, np.zeros_like(vec))

            M = LinearOperator(A_vel.shape, matvec=precondition, dtype=float)
            krylov_iters = {'count': 0}

            def callback(_):
                krylov_iters['count'] += 1

            try:
                x, info = gmres(
                    A_vel, rhs, M=M,
                    restart=min(50, A_vel.shape[0]),
                    maxiter=max(50, min(500, 2 * A_vel.shape[0])),
                    rtol=self.config.velocity_solver_rtol, atol=1e-10,
                    callback=callback,
                    callback_type='legacy',
                )
            except TypeError:
                x, info = gmres(
                    A_vel, rhs, M=M,
                    restart=min(50, A_vel.shape[0]),
                    tol=self.config.velocity_solver_rtol,
                    maxiter=max(50, min(500, 2 * A_vel.shape[0])),
                    callback=callback,
                )

            residual = np.linalg.norm(A_vel @ x - rhs)
            rhs_norm = np.linalg.norm(rhs)
            if info == 0 and np.isfinite(residual) and residual <= self.config.velocity_acceptance_rtol * max(rhs_norm, 1.0):
                if hasattr(solver, 'performance_stats'):
                    solver.performance_stats['total_iterations'] = krylov_iters['count']
                    solver.performance_stats['final_residual'] = residual
                self.velocity_cg_iterations += krylov_iters['count']
                if getattr(solver, '_hierarchy_source', 'traditional') == 'adapted':
                    self.adapted_velocity_krylov_iterations += krylov_iters['count']
                else:
                    self.traditional_velocity_krylov_iterations += krylov_iters['count']
                return x
        except Exception:
            pass

        self.velocity_direct_fallbacks += 1
        fallback_start = time.time()
        x = spsolve(A_vel, rhs)
        self.velocity_direct_fallback_time += time.time() - fallback_start
        return x

    def _viscosity_changed_significantly(self, viscosity: np.ndarray) -> bool:
        if self._prev_viscosity is None:
            return True
        rel_change = np.max(np.abs(viscosity - self._prev_viscosity) /
                           (np.abs(self._prev_viscosity) + 1e-12))
        return rel_change > 0.1

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

        # 压力质量矩阵: 真实FEM装配 (缓存, 只在粘度变化>10%时重算)
        if self._Q_matrix is None or self._viscosity_changed_significantly(viscosity):
            self._Q_matrix = self.assembler.assemble_pressure_mass_matrix(viscosity)
            self._prev_viscosity = viscosity.copy()

        return A_vel, b_vel, B, self._Q_matrix, vel_indices, p_indices

    def solve(self, A_full: sp.spmatrix, b_full: np.ndarray,
              temperature: np.ndarray, viscosity: np.ndarray,
              x0: np.ndarray = None, A_vel_prebuilt: sp.spmatrix = None) -> np.ndarray:
        """
        块求解Stokes系统。
        可选传入预装的速度块 (从Numba装配) 跳过extract_blocks的提取步骤。
        """
        return self.solve_with_velocity_block(
            A_full, b_full,
            A_vel_prebuilt if A_vel_prebuilt is not None else None,
            temperature, viscosity, x0)

    def solve_with_velocity_block(self, A_full, b_full, A_vel_prebuilt,
                                  temperature, viscosity, x0=None):
        """块求解, 接受预装的速度块 + Meta-AMG在线适配.

        Use block elimination rather than Uzawa iteration. The previous Uzawa
        path was numerically unstable for the current saddle-point scaling and
        corrupted the physical benchmark.  Here we reuse the velocity solver as
        an approximate inverse inside the Schur complement construction.
        """
        n_nodes = self.mesh.n_nodes
        nodal_dofs = 4
        n_dofs = n_nodes * nodal_dofs
        if x0 is None:
            x = np.zeros(n_dofs)
        else:
            x = x0.copy()

        # 提取速度块
        if A_vel_prebuilt is not None:
            A_vel = A_vel_prebuilt
            vel_idx = []; [vel_idx.extend([i*nodal_dofs, i*nodal_dofs+1]) for i in range(n_nodes)]
            b_vel = b_full[vel_idx]
        else:
            vel_idx = []; [vel_idx.extend([i*nodal_dofs, i*nodal_dofs+1]) for i in range(n_nodes)]
            A_vel = A_full[np.ix_(vel_idx, vel_idx)]
            b_vel = b_full[vel_idx]

        p_idx = [i * nodal_dofs + 2 for i in range(n_nodes)]
        t_idx = [i * nodal_dofs + 3 for i in range(n_nodes)]
        B = A_full[np.ix_(p_idx, vel_idx)].tocsr()
        G = A_full[np.ix_(vel_idx, p_idx)].tocsr()
        b_p = b_full[p_idx]

        # The assembled system is block upper triangular in temperature:
        # solve K_T T = f_T first, then move velocity-temperature coupling to
        # the momentum right-hand side before applying the Stokes block solve.
        temperature_block = A_full[np.ix_(t_idx, t_idx)].tocsr()
        velocity_temperature = A_full[np.ix_(vel_idx, t_idx)].tocsr()
        temperature_solution = spsolve(temperature_block, b_full[t_idx])
        if not np.isfinite(temperature_solution).all():
            temperature_solution = lsmr(
                temperature_block, b_full[t_idx], atol=1e-10, btol=1e-10)[0]
        b_vel = b_vel - velocity_temperature @ temperature_solution
        x[t_idx] = temperature_solution

        if self._Q_matrix is None:
            self._Q_matrix = self.assembler.assemble_pressure_mass_matrix(viscosity)
        Q = self._Q_matrix

        # ── Meta-AMG 在线适配 (核心) ──
        # 第一个Picard迭代: 传统AMG → 记录C/F
        # 后续迭代: Meta-AMG 从上一矩阵适配 → 快速获取C/F
        if self.meta_amg is not None and self._prev_A_vel is not None:
            try:
                adapt_t0 = time.time()
                coarse, fine = self.meta_amg.adapter.adapt(
                    self._prev_A_vel, self._prev_cf_labels, A_vel,
                    adapt_steps=self.config.meta_adapt_steps
                )
                self.meta_amg.solver.n_adapted += 1
                adapt_time = time.time() - adapt_t0
                self.meta_amg.solver.total_adapt_time += adapt_time
                self.meta_adaptation_time += adapt_time
            except Exception:
                coarse, fine = None, None
        else:
            coarse, fine = None, None

        # ── AMG求解速度块 ──
        if coarse is not None and len(coarse) > 0:
            # 用适配的C/F构建AMG
            hierarchy_start = time.time()
            vel_solver = self._build_amg_with_cf(A_vel, coarse, fine)
            self.adapted_hierarchy_setups += 1
            self.adapted_hierarchy_setup_time += time.time() - hierarchy_start
            vel_solver._hierarchy_source = 'adapted'
        else:
            if self.meta_amg is not None:
                self.meta_amg.solver.n_traditional += 1
            vel_solver = AlgebraicMultigridSolver(self.amg_config)
            vel_solver._hierarchy_source = 'traditional'

        # 保存当前状态供下一步Meta-AMG适配
        self._prev_A_vel = A_vel.copy()
        if coarse is not None:
            self._prev_cf_labels = np.zeros(A_vel.shape[0], dtype=np.float32)
            self._prev_cf_labels[coarse] = 1.0
        else:
            # 从传统AMG提取C/F
            self._prev_cf_labels = self._extract_cf_from_solver(vel_solver, A_vel)
        self._picard_step += 1

        def solve_velocity(rhs_u, phase='rhs'):
            return self._solve_velocity_system(A_vel, rhs_u, vel_solver, phase=phase)

        # Matrix-free Schur complement: S = C - B A_u^{-1} G.  The velocity
        # inverse is supplied by the same AMG-preconditioned Krylov path used for
        # the final back-substitution, so no dense Schur matrix is assembled.
        Ainv_b = solve_velocity(b_vel, phase='rhs')
        pressure_block = A_full[np.ix_(p_idx, p_idx)].tocsr()
        rhs_p = b_p - B @ Ainv_b

        def schur_matvec(p_vec):
            return pressure_block @ p_vec - B @ solve_velocity(
                G @ p_vec, phase='schur_columns')

        schur_operator = LinearOperator(
            (len(p_idx), len(p_idx)), matvec=schur_matvec, dtype=float)
        pressure_regularization = max(1e-10, 1e-8 * self.config.schur_scale)
        Q_regularized = (Q + pressure_regularization * sp.eye(Q.shape[0], format='csr')).tocsc()
        try:
            if self.config.pressure_preconditioner == 'diag_schur':
                velocity_diagonal = np.asarray(A_vel.diagonal(), dtype=float)
                inverse_diagonal = np.zeros_like(velocity_diagonal)
                usable = np.abs(velocity_diagonal) > 1e-12
                inverse_diagonal[usable] = 1.0 / velocity_diagonal[usable]
                approximate_schur = (
                    pressure_block - B @ sp.diags(inverse_diagonal) @ G +
                    pressure_regularization * Q
                ).tocsc()
                try:
                    pressure_factor = splu(approximate_schur)
                except Exception:
                    self.pressure_preconditioner_fallbacks += 1
                    pressure_factor = splu(Q_regularized)
            else:
                pressure_factor = splu(Q_regularized)
            pressure_preconditioner = LinearOperator(
                Q.shape, matvec=pressure_factor.solve, dtype=float)
            pressure_iters = {'count': 0}

            def pressure_callback(_):
                pressure_iters['count'] += 1

            try:
                p, pressure_info = gmres(
                    schur_operator, rhs_p, M=pressure_preconditioner,
                    restart=min(50, len(p_idx)), maxiter=max(50, len(p_idx)),
                    rtol=1e-8, atol=1e-10, callback=pressure_callback,
                    callback_type='legacy')
            except TypeError:
                p, pressure_info = gmres(
                    schur_operator, rhs_p, M=pressure_preconditioner,
                    restart=min(50, len(p_idx)), maxiter=max(50, len(p_idx)),
                    tol=1e-8, callback=pressure_callback)
            self.pressure_krylov_iterations += pressure_iters['count']
            pressure_residual = np.linalg.norm(schur_matvec(p) - rhs_p)
            if (pressure_info != 0 or not np.isfinite(p).all() or
                    pressure_residual > 1e-6 * max(np.linalg.norm(rhs_p), 1.0)):
                raise ValueError("matrix-free Schur solve did not converge")
        except Exception:
            self.pressure_solver_fallbacks += 1
            approximate_S = (pressure_block + self.config.schur_scale * Q).tocsr()
            p = spsolve(approximate_S, rhs_p)

        u = solve_velocity(b_vel - G @ p, phase='back_substitution')
        x[vel_idx] = u
        x[p_idx] = p

        full_residual = np.linalg.norm(A_full @ x - b_full)
        rhs_norm = np.linalg.norm(b_full)
        if not np.isfinite(full_residual) or full_residual > self.config.full_blocked_acceptance_rtol * max(rhs_norm, 1.0):
            self.full_direct_fallbacks += 1
            fallback_start = time.time()
            x = spsolve(A_full, b_full)
            self.full_direct_fallback_time += time.time() - fallback_start

        return x

    def _build_amg_with_cf(self, A_vel, coarse, fine):
        """用给定的C/F递归构建完整多层级AMG层次"""
        from solvers.multigrid_solver import AdaptiveCoarsening

        vel_solver = AlgebraicMultigridSolver(self.amg_config)
        vel_solver.levels = []
        vel_solver.interpolation_operators = []
        vel_solver.restriction_operators = []

        current_A = A_vel.copy()
        current_coarse = list(coarse)
        current_fine = list(fine)
        level = 0

        while (level < vel_solver.config.max_levels and
               current_A.shape[0] > vel_solver.config.max_coarse_size and
               len(current_coarse) > 0):

            vel_solver.levels.append({
                'matrix': current_A, 'size': current_A.shape[0], 'level': level
            })

            P = vel_solver._build_advanced_interpolation_operator(
                current_A, current_coarse, current_fine)
            R = P.T
            coarse_A = R @ current_A @ P

            vel_solver.interpolation_operators.append(P)
            vel_solver.restriction_operators.append(R)

            # 下一层: 对粗矩阵继续C/F分裂
            current_A = coarse_A
            try:
                current_coarse, current_fine = \
                    AdaptiveCoarsening.algebraic_coarsening(
                        current_A, vel_solver.config.strong_threshold)
            except Exception:
                break
            level += 1

        # 最粗层
        if current_A.shape[0] > 0:
            vel_solver.levels.append({
                'matrix': current_A, 'size': current_A.shape[0],
                'level': level
            })

        vel_solver.is_setup = True
        return vel_solver

    @staticmethod
    def _extract_cf_from_solver(solver, A):
        """从刚求解完的AMG求解器提取C/F标记, 不重复跑AMG setup"""
        n = A.shape[0]
        labels = np.zeros(n, dtype=np.float32)

        # 优先从solver的内部算子提取 — solver刚刚setup过, 不浪费
        if hasattr(solver, 'interpolation_operators') and solver.interpolation_operators:
            P = solver.interpolation_operators[0]  # 最细层的P
            for i in range(min(P.shape[0], n)):
                row = P[i].toarray().flatten()
                # 粗点是P矩阵中有一行≈单位向量的行 (直接在粗网格上)
                dominant = np.max(np.abs(row))
                if dominant > 0.9 and np.sum(np.abs(row) > 0.1) == 1:
                    labels[i] = 1.0
            if np.any(labels > 0):
                return labels

        # 回退: 从新solver的levels中提取
        if hasattr(solver, 'levels') and len(solver.levels) > 1:
            coarse_size = solver.levels[1]['size']
            labels[:coarse_size] = 1.0
            return labels

        # 最后回退: 传统AMG (仅当以上都失败时)
        from solvers.multigrid_solver import AdaptiveCoarsening
        try:
            coarse, fine = AdaptiveCoarsening.algebraic_coarsening(A, 0.25)
            labels[coarse] = 1.0
        except Exception:
            labels[:n//3] = 1.0
        return labels
        n_dofs = n_nodes * nodal_dofs

        if x0 is None:
            x = np.zeros(n_dofs)
        else:
            x = x0.copy()

        A_vel, b_vel, B, Q, vel_idx, p_idx = \
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

            # 3. 压力步: Schur补 Q dp = B u (真实FEM质量矩阵)
            div = B @ u_new
            dp = spsolve(Q, div)

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
    """块AMG求解 — 自动用Numba装配速度块 + Meta-AMG在线适配"""
    if not hasattr(self, '_block_solver'):
        self._block_solver = BlockStokesSolver(self.mesh, self.config)

    # 每次Picard迭代前, 注入最新的Meta-AMG引用
    if self.meta_amg is not None:
        self._block_solver.set_meta_amg(self.meta_amg)

    # The standalone Numba velocity-block assembly is not yet guaranteed to
    # match the monolithic Stokes assembly exactly. For benchmark correctness,
    # always extract the velocity block from the current full operator.
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


class PicardStokesSolver:
    """
    非线性Picard Stokes求解器 — 对标Underworld的地幔对流主循环。
    """

    def __init__(self, config: StokesConfig = None):
        self.config = config or StokesConfig()
        self.mesh = StokesMesh(
            nx=self.config.nx, ny=self.config.ny,
            lx=self.config.lx, ly=self.config.ly
        )
        self.assembler = GlobalStokesAssembler(self.mesh)
        self.bc_handler = StokesBoundaryConditions(self.mesh, self.config)
        self.viscosity_model = ViscosityModel(self.config)
        self.advection = TemperatureAdvection(self.mesh, self.config)

        # Numba加速 (如果可用)
        try:
            from core.numba_accelerator import NumbaStokesAssembler
            self._numba_asm = NumbaStokesAssembler(self.mesh)
        except Exception:
            self._numba_asm = None

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
        """初始化Meta-AMG — 训练或加载预训练模型"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                'meta_amg', 'gpu_acceleration/meta_amg.py')
            meta_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(meta_mod)
            MetaAMGConfig = meta_mod.MetaAMGConfig
            MetaAMG = meta_mod.MetaAMG

            n_nodes = self.mesh.n_nodes
            mcfg = MetaAMGConfig(
                min_matrix_size=max(64, n_nodes // 4),
                max_matrix_size=max(200, n_nodes * 2),
                num_training_sequences=self.config.meta_training_sequences,
                num_meta_epochs=self.config.meta_training_epochs,
                inner_steps=3, hidden_dim=32, num_layers=2,
            )
            self.meta_amg = MetaAMG(mcfg)
            if self.config.meta_model_path:
                self.meta_amg.load(self.config.meta_model_path)
                print(f"Meta-AMG loaded from {self.config.meta_model_path}")
            else:
                self.meta_amg.train(
                    num_sequences=self.config.meta_training_sequences)
                print("Meta-AMG trained and ready")
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

    def collect_picard_sequence(self, max_iterations: int = None,
                                seed: int = 0) -> List[Dict]:
        """Assemble one real nonlinear Stokes/Picard matrix trajectory.

        The full constrained Stokes--temperature matrix is retained for
        diagnostics.  ``matrix`` is its velocity block, the SPD elliptic block
        to which AMG is actually applied in the block Stokes preconditioner.
        This avoids training a scalar AMG model on the indefinite saddle-point
        matrix while preserving the physical Picard evolution that produced it.
        """
        max_iterations = max_iterations or self.config.max_picard_iterations
        rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            self.initialize_temperature()
        finally:
            np.random.set_state(rng_state)

        n_dofs = self.mesh.n_dofs
        vel_indices = np.array(
            [node * self.mesh.n_dofs_per_node + component
             for node in range(self.mesh.n_nodes) for component in (0, 1)],
            dtype=np.int64,
        )
        records = []
        self.velocity = np.zeros(n_dofs)
        strain_rate = self.advection.compute_strain_rate(self.velocity)
        self.viscosity = self.viscosity_model.compute(self.temperature, strain_rate)

        for iteration in range(max_iterations):
            A_full = self.assembler.assemble(
                self.viscosity, self.temperature, self.config.thermal_conductivity)
            rhs = self.assembler.assemble_rhs(
                self.temperature,
                rayleigh=self.config.rayleigh,
                thermal_expansivity=self.config.thermal_expansivity,
            )
            A_full, rhs = self.bc_handler.apply(A_full, rhs)
            A_velocity = A_full[vel_indices][:, vel_indices].tocsr()

            x_old = self.velocity.copy()
            try:
                x = spsolve(A_full, rhs)
                if not np.isfinite(x).all():
                    raise ValueError("non-finite direct Stokes solution")
            except Exception:
                # Collection must be robust on marginal small meshes; LSMR
                # still gives a physically coupled Picard update for the next
                # assembled matrix.
                x = lsmr(A_full, rhs, atol=1e-10, btol=1e-10)[0]

            self.velocity = np.asarray(x)
            strain_rate = self.advection.compute_strain_rate(self.velocity)
            next_viscosity = self.viscosity_model.compute(
                self.temperature, strain_rate)
            residual = np.linalg.norm(self.velocity - x_old) / (
                np.linalg.norm(self.velocity) + 1e-12)
            records.append({
                'matrix': A_velocity,
                'full_matrix': A_full,
                'rhs': rhs,
                'solution': self.velocity.copy(),
                'viscosity': self.viscosity.copy(),
                'temperature': self.temperature.copy(),
                'strain_rate': strain_rate.copy(),
                'velocity_indices': vel_indices.copy(),
                'iteration': iteration,
                'picard_residual': float(residual),
                'rayleigh': self.config.rayleigh,
                'viscosity_contrast': self.config.viscosity_contrast,
            })
            self.viscosity = next_viscosity
            if residual < self.config.picard_tolerance and iteration > 0:
                break
        return records

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
        if hasattr(self, '_block_solver'):
            self._block_solver.reset_stats()

        for iteration in range(self.config.max_picard_iterations):
            # 1. 装配
            A = self.assembler.assemble(
                self.viscosity, self.temperature,
                self.config.thermal_conductivity
            )
            b = self.assembler.assemble_rhs(
                self.temperature,
                rayleigh=self.config.rayleigh,
                thermal_expansivity=self.config.thermal_expansivity,
            )

            # 2. 边界条件
            A, b = self.bc_handler.apply(A, b)

            # 3. 求解
            if self.meta_amg is not None and iteration > 0:
                # 用Meta-AMG适配
                x = self._solve_with_meta_amg(A, b)
            else:
                x = self._solve_with_amg(A, b)

            self._last_linear_matrix = A
            self._last_linear_rhs = b.copy()

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
            'linear_relative_residual': float(
                np.linalg.norm(self._last_linear_rhs - self._last_linear_matrix @ self.velocity) /
                max(np.linalg.norm(self._last_linear_rhs), 1e-14)
            ),
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
        """AMG求解 — 自动用Numba装配的速度块加速"""
        # Full Stokes is indefinite; AMG belongs on the SPD velocity block.
        return _solve_with_amg_blocked(self, A, b)

    def _solve_with_meta_amg(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        """Meta-AMG求解 (适配模式)"""
        return _solve_with_amg_blocked(self, A, b)

    def _compute_nusselt(self) -> float:
        """计算表面Nusselt数"""
        nx, ny = self.config.nx, self.config.ny
        dy = self.config.ly / ny
        delta_T = self.config.boundary_temp_bottom - self.config.boundary_temp_top
        if abs(delta_T) < 1e-12:
            return 0.0

        top_fluxes = []
        row_stride = nx + 1
        for top_node in self.mesh.top_nodes:
            below_node = top_node - row_stride
            if below_node < 0:
                continue
            dT_dy = (self.temperature[top_node] - self.temperature[below_node]) / dy
            top_fluxes.append(-dT_dy / delta_T)

        if not top_fluxes:
            return 0.0
        return float(np.mean(top_fluxes))




    def run(self, n_steps: int = None, verbose: bool = True,
            adaptive: bool = False, refine_interval: int = 20,
            show_progress: bool = True,
            checkpoint_every: int = 0,
            resume: bool = False) -> Dict:
        """
        运行完整的地幔对流仿真。

        Args:
            n_steps: 时间步数
            verbose: 是否打印进度
            adaptive: 是否启用自适应网格细化
            refine_interval: 自适应细化间隔
            show_progress: 是否显示tqdm进度条
            checkpoint_every: 每N步自动保存断点 (0=不保存)
            resume: 是否从上次断点继续 (checkpoint_every>0时生效)
        """
        if n_steps is None:
            n_steps = self.config.max_time_steps

        self.initialize_temperature()

        history = {
            'time': [], 'nusselt': [], 'viscosity_min': [],
            'viscosity_max': [], 'picard_iterations': [],
        }

        start_step = 0

        # 断点续跑
        if resume and checkpoint_every > 0:
            ckpt_path = Path(self.config.output_dir) / 'checkpoint.npz'
            if ckpt_path.exists():
                try:
                    ckpt = np.load(ckpt_path, allow_pickle=True)
                    self.temperature = ckpt['temperature']
                    self.velocity = ckpt['velocity']
                    self.viscosity = ckpt['viscosity']
                    self.time = float(ckpt['time'])
                    start_step = int(ckpt['step']) + 1
                    # 恢复历史
                    if 'history' in ckpt:
                        hist = ckpt['history'].item()
                        history = hist
                    if verbose:
                        print(f"  [Resume] 从步骤 {start_step}/{n_steps} 继续, "
                              f"t={self.time:.4e}")
                except Exception as e:
                    if verbose:
                        print(f"  [Resume] 断点加载失败: {e}, 从头开始")

        t_start = time.time()

        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=n_steps, initial=start_step, desc='Simulating',
                            unit='step', ncols=80)
            except ImportError:
                pass

        for step in range(start_step, n_steps):
            stats = self.solve_timestep()

            history['time'].append(self.time)
            history['nusselt'].append(stats['nusselt'])
            history['viscosity_min'].append(stats['viscosity_range'][0])
            history['viscosity_max'].append(stats['viscosity_range'][1])
            history['picard_iterations'].append(stats['n_iterations'])

            if pbar is not None:
                pbar.set_postfix({'Nu': f"{stats['nusselt']:.2f}",
                                  'Picard': stats['n_iterations']})
                pbar.update(1)

            # 自动断点
            if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                self._save_checkpoint(step, history)
                if verbose and pbar is not None:
                    pbar.write(f"  [Checkpoint] step {step+1}")

            if verbose and step % max(1, n_steps // 10) == 0:
                msg = (f"Step {step:4d}/{n_steps} | t={self.time:.4e} | "
                       f"Nu={stats['nusselt']:.3f} | Picard={stats['n_iterations']}")
                if pbar is not None:
                    pbar.write(msg)
                else:
                    print(msg)

        if pbar is not None:
            pbar.close()

        elapsed = time.time() - t_start
        if verbose:
            print(f"\nSimulation complete: {n_steps - start_step} steps "
                  f"in {elapsed:.1f}s")
            print(f"Final Nusselt: {history['nusselt'][-1]:.3f}")
            print(f"Avg Picard iterations: "
                  f"{np.mean(history['picard_iterations']):.1f}")

        return history

    def _save_checkpoint(self, step: int, history: Dict):
        """保存断点"""
        ckpt_path = Path(self.config.output_dir) / 'checkpoint.npz'
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ckpt_path,
            temperature=self.temperature,
            velocity=self.velocity,
            viscosity=self.viscosity,
            time=self.time,
            step=step,
            history=np.array([history], dtype=object),
        )

    def _interpolate_field(self, old_field: np.ndarray,
                           n_new_nodes: int) -> np.ndarray:
        """将旧网格上的场插值到新网格 (最近邻)"""
        new_field = np.zeros(n_new_nodes)
        n_old = min(len(old_field), n_new_nodes)
        # 简单的截断/填充策略
        new_field[:n_old] = old_field[:n_old]
        if n_new_nodes > n_old:
            new_field[n_old:] = np.mean(old_field)
        return new_field

    def save(self, filepath: str = None):
        """保存仿真结果 (NPZ + VTK)"""
        if filepath is None:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            filepath = str(Path(self.config.output_dir) / 'stokes_result')

        # NPZ (总是可用)
        np.savez_compressed(filepath + '.npz',
                           temperature=self.temperature,
                           velocity=self.velocity,
                           viscosity=self.viscosity,
                           time=self.time,
                           nodes=self.mesh.nodes,
                           elements=np.array(self.mesh.elements))
        print(f"Saved {filepath}.npz")

        # VTK (如果meshio可用)
        try:
            import meshio
            cells = [("triangle", np.array(self.mesh.elements))]
            point_data = {
                "temperature": self.temperature,
                "viscosity": self.viscosity,
            }
            vel_mag = np.sqrt(self.velocity[0::self.mesh.n_dofs_per_node]**2 +
                             self.velocity[1::self.mesh.n_dofs_per_node]**2)
            point_data["velocity_magnitude"] = vel_mag
            meshio.write_points_cells(filepath + '.vtk',
                                     self.mesh.nodes, cells,
                                     point_data=point_data)
            print(f"Saved {filepath}.vtk")
        except ImportError:
            pass

    def plot(self, field='temperature', save=True, show=False):
        """
        自动画图 — 温度场/粘度场/速度矢量。

        Args:
            field: 'temperature', 'viscosity', 'velocity', 'all'
            save: 是否保存PNG
            show: 是否弹出窗口显示
        """
        try:
            import matplotlib.pyplot as plt

            fields = {
                'temperature': (self.temperature, 'inferno', 'Temperature'),
                'viscosity': (np.log10(self.viscosity + 1e-12), 'viridis', 'log₁₀(Viscosity)'),
                'velocity': (np.sqrt(self.velocity[0::self.mesh.n_dofs_per_node]**2 +
                                    self.velocity[1::self.mesh.n_dofs_per_node]**2),
                            'plasma', 'Velocity Magnitude'),
            }

            if field == 'all':
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                for ax, (name, (data, cmap, label)) in zip(axes, fields.items()):
                    im = ax.tripcolor(self.mesh.nodes[:, 0], self.mesh.nodes[:, 1],
                                     self.mesh.elements, data, cmap=cmap, shading='gouraud')
                    plt.colorbar(im, ax=ax, label=label)
                    ax.set_title(name)
                    ax.set_xlabel('x'); ax.set_ylabel('y')
                    ax.set_aspect('equal')
            else:
                if field not in fields:
                    field = 'temperature'
                data, cmap, label = fields[field]
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.tripcolor(self.mesh.nodes[:, 0], self.mesh.nodes[:, 1],
                                 self.mesh.elements, data, cmap=cmap, shading='gouraud')
                plt.colorbar(im, ax=ax, label=label)
                ax.set_title(f'{field} at t={self.time:.4e}')
                ax.set_xlabel('x'); ax.set_ylabel('y')
                ax.set_aspect('equal')

            plt.tight_layout()

            if save:
                Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
                fpath = Path(self.config.output_dir) / f'{field}.png'
                plt.savefig(fpath, dpi=150, bbox_inches='tight')
                print(f"Plot saved: {fpath}")

            if show:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"Plot error: {e}")


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


# ── StokesConfig YAML序列化 ──

def _stokes_config_from_yaml(filepath: str):
    import yaml
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return StokesConfig(**{k: v for k, v in data.items()
                           if k in StokesConfig.__dataclass_fields__})

StokesConfig.from_yaml = staticmethod(_stokes_config_from_yaml)


def _stokes_config_to_yaml(self, filepath: str):
    import yaml
    data = {k: getattr(self, k) for k in self.__dataclass_fields__}
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

StokesConfig.to_yaml = _stokes_config_to_yaml
