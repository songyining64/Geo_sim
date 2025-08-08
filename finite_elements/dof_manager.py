"""
自由度管理模块
"""
import numpy as np

class MultiFieldDofManager:
    def __init__(self, mesh, field_dims):
        # field_dims: {'velocity': 2, 'pressure': 1, 'temperature': 1}
        self.mesh = mesh
        self.field_dims = field_dims
        self.n_nodes = mesh.coordinates.shape[0]
        self.offsets = self._compute_offsets()
    def _compute_offsets(self):
        offsets = {}
        offset = 0
        for name, dim in self.field_dims.items():
            offsets[name] = offset
            offset += self.n_nodes * dim
        return offsets
    def get_dof(self, node_id, field, comp=0):
        return self.offsets[field] + node_id * self.field_dims[field] + comp
    def total_dof(self):
        return sum(self.n_nodes * d for d in self.field_dims.values())
    def get_element_dofs(self, elem):
        dofs = []
        for field, dim in self.field_dims.items():
            for nid in elem.node_ids[field]:
                for d in range(dim):
                    dofs.append(self.get_dof(nid, field, d))
        return dofs

def stokes_heat_element_matrix(node_coords, params):
    # node_coords: {'velocity': (n_v, 2), 'pressure': (n_p, 2), 'temperature': (n_T, 2)}
    # params: {'viscosity': ..., 'conductivity': ..., 'buoyancy': ..., ...}
    n_v = node_coords['velocity'].shape[0] * 2  # 2D
    n_p = node_coords['pressure'].shape[0]
    n_T = node_coords['temperature'].shape[0]
    Ke = np.zeros((n_v + n_p + n_T, n_v + n_p + n_T))
    fe = np.zeros(n_v + n_p + n_T)
    # 1. 组装Stokes块（A_vv, A_vp, A_pv）
    # 2. 组装温度块（A_TT, A_Tv，对流项）
    # 3. 组装耦合项（如浮力项A_vT）
    # 4. 右端项
    # 这里只给出结构，具体实现需结合基函数和物理模型
    return Ke, fe

from scipy.sparse import lil_matrix

def assemble_global_stokes_heat(mesh, dof_manager, params):
    n_dof = dof_manager.total_dof()
    K = lil_matrix((n_dof, n_dof))
    F = np.zeros(n_dof)
    for elem in mesh.elements:
        node_coords = {field: mesh.coordinates[elem.node_ids[field]] for field in dof_manager.field_dims}
        Ke, fe = stokes_heat_element_matrix(node_coords, params)
        dofs = dof_manager.get_element_dofs(elem)
        for i, gi in enumerate(dofs):
            F[gi] += fe[i]
            for j, gj in enumerate(dofs):
                K[gi, gj] += Ke[i, j]
    return K, F

# 施加多场Dirichlet/Neumann/Robin边界
def apply_multifield_dirichlet(K, F, dof_manager, field, node_ids, values, comp=0):
    for nid, v in zip(node_ids, values):
        dof = dof_manager.get_dof(nid, field, comp)
        K[dof, :] = 0
        K[dof, dof] = 1
        F[dof] = v

# 结果拆分
def split_solution(U, dof_manager):
    n_nodes = dof_manager.n_nodes
    velocity = U[:n_nodes*2].reshape(n_nodes, 2)
    pressure = U[n_nodes*2:n_nodes*2+n_nodes]
    temperature = U[-n_nodes:]
    return velocity, pressure, temperature 