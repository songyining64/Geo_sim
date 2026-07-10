"""
Numba JIT加速FEM装配 — 直接查找表, O(n_elements) 并行
"""

import numpy as np

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def precompute_lookup(mesh):
    """预计算: 速度块COO映射表 + 网格数据平面"""
    n_elem = len(mesh.elements)
    n_local = 3
    ndpn = mesh.n_dofs_per_node

    elem_nodes = np.array(mesh.elements, dtype=np.int32)
    nodes_flat = mesh.nodes.ravel().astype(np.float64)

    # 构建速度块COO: (ux, uy) 各一份
    coo_map = {}
    coo_rows, coo_cols = [], []
    n_nodes = mesh.n_nodes
    for e_nodes in mesh.elements:
        for ai, a in enumerate(e_nodes):
            for bi, b in enumerate(e_nodes):
                key = (a, b)
                if key not in coo_map:
                    # 速度块: 节点a到b的2×2块
                    idx_ux = len(coo_rows)
                    coo_rows.append(a * 2)      # ux_a
                    coo_cols.append(b * 2)      # ux_b
                    idx_uy = len(coo_rows)
                    coo_rows.append(a * 2 + 1)  # uy_a
                    coo_cols.append(b * 2 + 1)  # uy_b
                    coo_map[key] = (idx_ux, idx_uy)

    coo_rows_arr = np.array(coo_rows, dtype=np.int32)
    coo_cols_arr = np.array(coo_cols, dtype=np.int32)
    n_coo = len(coo_rows_arr)

    # 查找表: elem_coo_map_ux[e,a,b] → COO index for ux; _uy for uy
    elem_coo_ux = np.zeros((n_elem, n_local, n_local), dtype=np.int32)
    elem_coo_uy = np.zeros((n_elem, n_local, n_local), dtype=np.int32)
    for e, e_nodes in enumerate(mesh.elements):
        for ai, a in enumerate(e_nodes):
            for bi, b in enumerate(e_nodes):
                ux_idx, uy_idx = coo_map[(a, b)]
                elem_coo_ux[e, ai, bi] = ux_idx
                elem_coo_uy[e, ai, bi] = uy_idx

    return (nodes_flat, elem_nodes, coo_rows_arr, coo_cols_arr,
            elem_coo_ux, elem_coo_uy, n_elem, n_coo, ndpn)


# ═══════════════════════════════════════════
# JIT内核: O(n_elements × 9) — 无搜索
# ═══════════════════════════════════════════

@jit(nopython=True, parallel=True, cache=True)
def _assemble_stokes_jit(
    nodes_flat, elem_nodes,
    elem_coo_ux, elem_coo_uy,
    viscosity, ndpn, n_elem, n_coo, data_out
):
    for i in range(n_coo):
        data_out[i] = 0.0

    for e in prange(n_elem):
        n0 = elem_nodes[e, 0]
        n1 = elem_nodes[e, 1]
        n2 = elem_nodes[e, 2]

        x0 = nodes_flat[n0 * 2];      y0 = nodes_flat[n0 * 2 + 1]
        x1 = nodes_flat[n1 * 2];      y1 = nodes_flat[n1 * 2 + 1]
        x2 = nodes_flat[n2 * 2];      y2 = nodes_flat[n2 * 2 + 1]

        detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if detJ <= 0.0:
            detJ = 1e-12
        inv_det = 1.0 / detJ
        area = 0.5 * detJ

        eta = (viscosity[n0] + viscosity[n1] + viscosity[n2]) / 3.0

        d0x = (y1 - y2) * inv_det;  d0y = (x2 - x1) * inv_det
        d1x = (y2 - y0) * inv_det;  d1y = (x0 - x2) * inv_det
        d2x = (y0 - y1) * inv_det;  d2y = (x1 - x0) * inv_det

        g00 = d0x*d0x + d0y*d0y;  g01 = d0x*d1x + d0y*d1y;  g02 = d0x*d2x + d0y*d2y
        g11 = d1x*d1x + d1y*d1y;  g12 = d1x*d2x + d1y*d2y;  g22 = d2x*d2x + d2y*d2y

        kv = eta * area

        # 直接写速度块COO (ux和uy各一份)
        data_out[elem_coo_ux[e, 0, 0]] += kv * g00
        data_out[elem_coo_ux[e, 0, 1]] += kv * g01
        data_out[elem_coo_ux[e, 0, 2]] += kv * g02
        data_out[elem_coo_ux[e, 1, 0]] += kv * g01
        data_out[elem_coo_ux[e, 1, 1]] += kv * g11
        data_out[elem_coo_ux[e, 1, 2]] += kv * g12
        data_out[elem_coo_ux[e, 2, 0]] += kv * g02
        data_out[elem_coo_ux[e, 2, 1]] += kv * g12
        data_out[elem_coo_ux[e, 2, 2]] += kv * g22

        data_out[elem_coo_uy[e, 0, 0]] += kv * g00
        data_out[elem_coo_uy[e, 0, 1]] += kv * g01
        data_out[elem_coo_uy[e, 0, 2]] += kv * g02
        data_out[elem_coo_uy[e, 1, 0]] += kv * g01
        data_out[elem_coo_uy[e, 1, 1]] += kv * g11
        data_out[elem_coo_uy[e, 1, 2]] += kv * g12
        data_out[elem_coo_uy[e, 2, 0]] += kv * g02
        data_out[elem_coo_uy[e, 2, 1]] += kv * g12
        data_out[elem_coo_uy[e, 2, 2]] += kv * g22


class NumbaStokesAssembler:
    """Numba JIT加速装配 — 直接产出速度块CSR, 无后处理"""

    def __init__(self, mesh):
        self.mesh = mesh
        self._ready = HAS_NUMBA
        if not self._ready:
            return

        (self.nodes_flat, self.elem_nodes, self.coo_rows, self.coo_cols,
         self.elem_coo_ux, self.elem_coo_uy,
         self.n_elem, self.n_coo, self.ndpn) = precompute_lookup(mesh)

    def assemble_velocity_block(self, viscosity):
        if not self._ready:
            from core.stokes_solver import GlobalStokesAssembler
            asm = GlobalStokesAssembler(self.mesh)
            A_full = asm.assemble(viscosity, np.ones(self.mesh.n_nodes) * 0.5)
            n = self.mesh.n_nodes
            vi = []; [vi.extend([i*4, i*4+1]) for i in range(n)]
            return A_full[np.ix_(vi, vi)]

        data = np.zeros(self.n_coo, dtype=np.float64)
        _assemble_stokes_jit(
            self.nodes_flat, self.elem_nodes,
            self.elem_coo_ux, self.elem_coo_uy,
            viscosity.astype(np.float64),
            self.ndpn, self.n_elem, self.n_coo, data
        )
        from scipy.sparse import csr_matrix
        n_vel = self.mesh.n_nodes * 2
        return csr_matrix((data, (self.coo_rows, self.coo_cols)),
                         shape=(n_vel, n_vel))


# ═══════════════════════════════════════════
# 集成到 PicardStokesSolver
# ═══════════════════════════════════════════

def patch_solver_for_numba(solver):
    """给PicardStokesSolver加速: Numba装配速度块 + Python装配其余"""
    if not HAS_NUMBA:
        return solver

    nb_asm = NumbaStokesAssembler(solver.mesh)
    solver._numba_assembler = nb_asm
    _original = solver.assembler.assemble
    ndpn = solver.mesh.n_dofs_per_node

    def _hybrid_assemble(viscosity, temperature, kappa=1.0):
        A_vel = nb_asm.assemble_velocity_block(viscosity)
        A_full = _original(viscosity, temperature, kappa)
        n = solver.mesh.n_nodes
        # 嵌入速度块
        for a in range(n):
            for b in range(n):
                kv_ux = A_vel[a * 2, b * 2]
                kv_uy = A_vel[a * 2 + 1, b * 2 + 1]
                if kv_ux != 0:
                    A_full[a * ndpn + 0, b * ndpn + 0] = kv_ux
                    A_full[a * ndpn + 1, b * ndpn + 1] = kv_uy
        return A_full

    solver.assembler.assemble = _hybrid_assemble
    return solver
