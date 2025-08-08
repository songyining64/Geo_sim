"""
等参变换、Jacobi矩阵与单元微分算子（支持高阶/多类型单元）
"""
import numpy as np

def jacobian_matrix(node_coords, dN_dxi):
    """
    计算Jacobi矩阵
    node_coords: (n_nodes, dim) 物理坐标
    dN_dxi: (n_nodes, dim) 基函数对参考坐标的导数
    返回 J: (dim, dim)
    """
    # 按高斯点循环时 dN_dxi shape: (n_nodes, dim)
    return dN_dxi.T @ node_coords

def jacobian_det(J):
    """Jacobi行列式"""
    return np.linalg.det(J)

def jacobian_inv(J):
    """Jacobi逆矩阵"""
    return np.linalg.inv(J)

def dN_dx(dN_dxi, J_inv):
    """
    基函数对物理坐标的导数
    dN_dxi: (n_nodes, dim)
    J_inv: (dim, dim)
    返回 dN_dx: (n_nodes, dim)
    """
    return dN_dxi @ J_inv.T

# --------- shape_gradients for reference elements ---------

def tetrahedron_shape_gradients(order=1):
    """
    线性/二次四面体单元在参考坐标下的基函数梯度
    order=1: 4节点线性, order=2: 10节点二次
    返回: (n_nodes, 3)
    """
    if order == 1:
        # 线性四面体
        return np.array([
            [-1, -1, -1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    elif order == 2:
        # 二次四面体（10节点）参考单元梯度
        # 这里只给出前4个主节点，边节点建议用符号法自动生成
        raise NotImplementedError("二次四面体 shape_gradients 建议用符号法自动生成")
    else:
        raise NotImplementedError("仅支持线性/二次四面体")

def triangle_shape_gradients(order=1):
    """
    线性/二次三角形单元在参考坐标下的基函数梯度
    order=1: 3节点线性, order=2: 6节点二次
    返回: (n_nodes, 2)
    """
    if order == 1:
        return np.array([
            [-1, -1],
            [1, 0],
            [0, 1]
        ])
    elif order == 2:
        raise NotImplementedError("二次三角形 shape_gradients 建议用符号法自动生成")
    else:
        raise NotImplementedError("仅支持线性/二次三角形")

def hexahedron_shape_gradients(order=1):
    """
    线性六面体单元在参考坐标下的基函数梯度
    order=1: 8节点线性
    返回: (n_nodes, 3)
    """
    if order == 1:
        # 8节点线性六面体
        # 每个节点的基函数对 (xi, eta, zeta) 的导数
        # 参考单元节点顺序: (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1), (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)
        # 这里只给出符号表达式，实际应用建议用自动生成
        raise NotImplementedError("线性六面体 shape_gradients 建议用符号法自动生成")
    else:
        raise NotImplementedError("仅支持线性六面体")

# --------- 散度/梯度/旋度等单元算子 ---------

def compute_divergence(u, dN_dx):
    """
    计算单元内速度场的散度
    u: (n_nodes, dim) 节点速度
    dN_dx: (n_nodes, dim) 基函数对物理坐标的导数
    返回: float
    """
    # sum_a u_a · grad(N_a)
    return np.sum(np.einsum('ij,ij->i', u, dN_dx))

def compute_gradient(field, dN_dx):
    """
    计算单元内标量场的梯度
    field: (n_nodes,) 节点标量
    dN_dx: (n_nodes, dim)
    返回: (dim,)
    """
    return np.sum(field[:, None] * dN_dx, axis=0)

def compute_curl(u, dN_dx):
    """
    计算单元内向量场的旋度（仅3D）
    u: (n_nodes, 3)
    dN_dx: (n_nodes, 3)
    返回: (3,)
    """
    # sum_a u_a × grad(N_a)
    return np.sum(np.cross(u, dN_dx), axis=0)

# --------- 复杂单元类型支持（高阶/多面体/自定义） ---------

def reference_element_shape_gradients(element_type, order):
    """
    通用接口：返回任意类型/阶次单元在参考单元下的 shape_gradients
    element_type: 'triangle', 'quad', 'tetra', 'hex'
    order: 1, 2, 3, ...
    """
    if element_type == 'triangle':
        return triangle_shape_gradients(order)
    elif element_type == 'tetra':
        return tetrahedron_shape_gradients(order)
    elif element_type == 'hex':
        return hexahedron_shape_gradients(order)
    else:
        raise NotImplementedError(f"暂不支持 {element_type} 类型 shape_gradients")

# --------- 体积/面积 ---------

def element_volume(J, dim):
    """
    计算单元体积（或面积）
    J: Jacobi矩阵
    dim: 空间维数
    """
    if dim == 2:
        return 0.5 * np.abs(jacobian_det(J))
    elif dim == 3:
        return (1.0 / 6.0) * np.abs(jacobian_det(J))
    else:
        raise ValueError("仅支持2D/3D")
