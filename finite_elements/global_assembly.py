"""
全局刚度矩阵组装（稀疏）
"""
import numpy as np
from scipy.sparse import lil_matrix
from .assembly import triangle_stiffness, quad_stiffness, tetra_stiffness, hex_stiffness

def assemble_global_stiffness(mesh_nodes, mesh_elements, element_type='triangle', order=1):
    n_nodes = mesh_nodes.shape[0]
    K = lil_matrix((n_nodes, n_nodes))
    for elem in mesh_elements:
        node_ids = elem
        node_coords = mesh_nodes[node_ids]
        if element_type == 'triangle':
            Ke = triangle_stiffness(node_coords, order=order)
        elif element_type == 'quad':
            Ke = quad_stiffness(node_coords, order=order)
        elif element_type == 'tetra':
            Ke = tetra_stiffness(node_coords, order=order)
        elif element_type == 'hex':
            Ke = hex_stiffness(node_coords, order=order)
        else:
            raise NotImplementedError
        for i in range(len(node_ids)):
            for j in range(len(node_ids)):
                K[node_ids[i], node_ids[j]] += Ke[i, j]
    return K
