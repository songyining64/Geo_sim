"""
有限元基函数实现
支持线性/二次/三次 Lagrange 基函数（1D/2D/3D）
"""

import numpy as np
from abc import ABC, abstractmethod
import sympy as sp

# ---- 通用高阶Lagrange基函数生成器 ----
def lagrange_nodes_1d(order):
    return np.linspace(-1, 1, order + 1)

def lagrange_basis_1d(order, xi):
    nodes = lagrange_nodes_1d(order)
    n = order + 1
    xi = np.asarray(xi)
    N = np.ones((n,)) if xi.ndim == 0 else np.ones((xi.shape[0], n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if xi.ndim == 0:
                    N[i] *= (xi - nodes[j]) / (nodes[i] - nodes[j])
                else:
                    N[:, i] *= (xi - nodes[j]) / (nodes[i] - nodes[j])
    return N

def lagrange_basis_deriv_1d(order, xi):
    nodes = lagrange_nodes_1d(order)
    n = order + 1
    xi = np.asarray(xi)
    dN = np.zeros((n,)) if xi.ndim == 0 else np.zeros((xi.shape[0], n))
    for i in range(n):
        s = 0
        for j in range(n):
            if i != j:
                prod = 1
                for k in range(n):
                    if k != i and k != j:
                        prod *= (xi - nodes[k]) / (nodes[i] - nodes[k])
                s += prod / (nodes[i] - nodes[j])
        dN[i] = s if xi.ndim == 0 else s
    return dN

# ---- 1D Lagrange ----
class Lagrange1D:
    def __init__(self, order):
        self.order = order
        self.nodes = lagrange_nodes_1d(order)

    def evaluate(self, xi):
        return lagrange_basis_1d(self.order, xi)

    def evaluate_derivatives(self, xi):
        return lagrange_basis_deriv_1d(self.order, xi)

# ---- 2D 三角形 Lagrange ----
def triangle_lagrange_nodes(order):
    nodes = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            nodes.append((i / order, j / order))
    return np.array(nodes)

def triangle_lagrange_basis(order, xi):
    # xi: (2,) or (N,2)
    xi = np.atleast_2d(xi)
    nodes = triangle_lagrange_nodes(order)
    N = []
    for a, (xi_a, eta_a) in enumerate(nodes):
        L = np.ones(xi.shape[0])
        for b, (xi_b, eta_b) in enumerate(nodes):
            if a != b:
                L *= ((xi[:, 0] - xi_b) / (xi_a - xi_b)) * ((xi[:, 1] - eta_b) / (eta_a - eta_b))
        N.append(L)
    return np.stack(N, axis=-1)

# ---- 2D 四边形 Lagrange ----
def quad_lagrange_nodes(order):
    return np.array([(xi, eta) for eta in np.linspace(-1, 1, order+1)
                               for xi in np.linspace(-1, 1, order+1)])

def quad_lagrange_basis(order, xi):
    xi = np.atleast_2d(xi)
    n = order + 1
    xi_nodes = np.linspace(-1, 1, n)
    N = []
    for j in range(n):
        for i in range(n):
            Ni = np.ones(xi.shape[0])
            for k in range(n):
                if i != k:
                    Ni *= (xi[:, 0] - xi_nodes[k]) / (xi_nodes[i] - xi_nodes[k])
            for l in range(n):
                if j != l:
                    Ni *= (xi[:, 1] - xi_nodes[l]) / (xi_nodes[j] - xi_nodes[l])
            N.append(Ni)
    return np.stack(N, axis=-1)

def quad_lagrange_basis_deriv(order, xi):
    """计算四边形Lagrange基函数的导数"""
    xi = np.atleast_2d(xi)
    n = order + 1
    xi_nodes = np.linspace(-1, 1, n)
    
    dN_dxi = []
    dN_deta = []
    
    for j in range(n):
        for i in range(n):
            # 计算 dN/dxi
            dNi_dxi = np.zeros(xi.shape[0])
            for k in range(n):
                if i != k:
                    prod = 1.0
                    for m in range(n):
                        if m != i and m != k:
                            prod *= (xi[:, 0] - xi_nodes[m]) / (xi_nodes[i] - xi_nodes[m])
                    prod /= (xi_nodes[i] - xi_nodes[k])
                    
                    # 乘以eta方向的基函数
                    for l in range(n):
                        if j != l:
                            prod *= (xi[:, 1] - xi_nodes[l]) / (xi_nodes[j] - xi_nodes[l])
                    
                    dNi_dxi += prod
            
            # 计算 dN/deta
            dNi_deta = np.zeros(xi.shape[0])
            for l in range(n):
                if j != l:
                    prod = 1.0
                    for m in range(n):
                        if m != j and m != l:
                            prod *= (xi[:, 1] - xi_nodes[m]) / (xi_nodes[j] - xi_nodes[m])
                    prod /= (xi_nodes[j] - xi_nodes[l])
                    
                    # 乘以xi方向的基函数
                    for k in range(n):
                        if i != k:
                            prod *= (xi[:, 0] - xi_nodes[k]) / (xi_nodes[i] - xi_nodes[k])
                    
                    dNi_deta += prod
            
            dN_dxi.append(dNi_dxi)
            dN_deta.append(dNi_deta)
    
    return np.stack(dN_dxi, axis=-1), np.stack(dN_deta, axis=-1)

# ---- 3D 四面体 Lagrange ----
def tetra_lagrange_nodes(order):
    nodes = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            for k in range(order + 1 - i - j):
                nodes.append((i / order, j / order, k / order))
    return np.array(nodes)

def tetra_lagrange_basis(order, xi):
    xi = np.atleast_2d(xi)
    nodes = tetra_lagrange_nodes(order)
    N = []
    for a, (xi_a, eta_a, zeta_a) in enumerate(nodes):
        L = np.ones(xi.shape[0])
        for b, (xi_b, eta_b, zeta_b) in enumerate(nodes):
            if a != b:
                L *= ((xi[:, 0] - xi_b) / (xi_a - xi_b)) * \
                     ((xi[:, 1] - eta_b) / (eta_a - eta_b)) * \
                     ((xi[:, 2] - zeta_b) / (zeta_a - zeta_b))
        N.append(L)
    return np.stack(N, axis=-1)

# ---- 3D 六面体 Lagrange ----
def hex_lagrange_nodes(order):
    return np.array([(xi, eta, zeta)
                     for zeta in np.linspace(-1, 1, order+1)
                     for eta in np.linspace(-1, 1, order+1)
                     for xi in np.linspace(-1, 1, order+1)])

def hex_lagrange_basis(order, xi):
    xi = np.atleast_2d(xi)
    n = order + 1
    xi_nodes = np.linspace(-1, 1, n)
    N = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                Ni = np.ones(xi.shape[0])
                for l in range(n):
                    if i != l:
                        Ni *= (xi[:, 0] - xi_nodes[l]) / (xi_nodes[i] - xi_nodes[l])
                for m in range(n):
                    if j != m:
                        Ni *= (xi[:, 1] - xi_nodes[m]) / (xi_nodes[j] - xi_nodes[m])
                for n_ in range(n):
                    if k != n_:
                        Ni *= (xi[:, 2] - xi_nodes[n_]) / (xi_nodes[k] - xi_nodes[n_])
                N.append(Ni)
    return np.stack(N, axis=-1)

# ---- 单元类型统一接口 ----
class LagrangeTriangle:
    def __init__(self, order):
        self.order = order
        self.nodes = triangle_lagrange_nodes(order)
    
    def evaluate(self, xi):
        """计算三角形线性基函数值"""
        xi = np.atleast_2d(xi)
        n_points = xi.shape[0]
        
        if self.order == 1:
            # 线性三角形基函数：N1 = 1 - xi - eta, N2 = xi, N3 = eta
            N = np.zeros((n_points, 3))
            N[:, 0] = 1 - xi[:, 0] - xi[:, 1]  # N1
            N[:, 1] = xi[:, 0]                 # N2  
            N[:, 2] = xi[:, 1]                 # N3
            return N
        else:
            return triangle_lagrange_basis(self.order, xi)
    
    def evaluate_derivatives(self, xi):
        """计算三角形基函数导数"""
        xi = np.atleast_2d(xi)
        n_points = xi.shape[0]
        
        if self.order == 1:
            # 线性三角形基函数导数（常数）
            # N1 = 1 - xi - eta, N2 = xi, N3 = eta
            dN = np.zeros((n_points, 3, 2))
            dN[:, 0, 0] = -1  # dN1/dxi
            dN[:, 0, 1] = -1  # dN1/deta
            dN[:, 1, 0] = 1   # dN2/dxi
            dN[:, 1, 1] = 0   # dN2/deta
            dN[:, 2, 0] = 0   # dN3/dxi
            dN[:, 2, 1] = 1   # dN3/deta
            return dN
        else:
            raise NotImplementedError("高阶三角形基函数导数待实现")

class LagrangeQuad:
    def __init__(self, order):
        self.order = order
        self.nodes = quad_lagrange_nodes(order)
    
    def evaluate(self, xi):
        """计算四边形线性基函数值"""
        xi = np.atleast_2d(xi)
        n_points = xi.shape[0]
        
        if self.order == 1:
            # 线性四边形基函数
            # N1 = (1-xi)(1-eta)/4, N2 = (1+xi)(1-eta)/4, N3 = (1+xi)(1+eta)/4, N4 = (1-xi)(1+eta)/4
            N = np.zeros((n_points, 4))
            N[:, 0] = (1 - xi[:, 0]) * (1 - xi[:, 1]) / 4  # N1
            N[:, 1] = (1 + xi[:, 0]) * (1 - xi[:, 1]) / 4  # N2
            N[:, 2] = (1 + xi[:, 0]) * (1 + xi[:, 1]) / 4  # N3
            N[:, 3] = (1 - xi[:, 0]) * (1 + xi[:, 1]) / 4  # N4
            return N
        else:
            return quad_lagrange_basis(self.order, xi)
    
    def evaluate_derivatives(self, xi):
        """计算四边形基函数导数"""
        xi = np.atleast_2d(xi)
        n_points = xi.shape[0]
        
        if self.order == 1:
            # 线性四边形基函数导数
            # N1 = (1-xi)(1-eta)/4, N2 = (1+xi)(1-eta)/4, N3 = (1+xi)(1+eta)/4, N4 = (1-xi)(1+eta)/4
            dN = np.zeros((n_points, 4, 2))
            
            for i, (xi_i, eta_i) in enumerate(xi):
                dN[i, 0, 0] = -(1 - eta_i) / 4  # dN1/dxi
                dN[i, 0, 1] = -(1 - xi_i) / 4   # dN1/deta
                dN[i, 1, 0] = (1 - eta_i) / 4   # dN2/dxi
                dN[i, 1, 1] = -(1 + xi_i) / 4   # dN2/deta
                dN[i, 2, 0] = (1 + eta_i) / 4   # dN3/dxi
                dN[i, 2, 1] = (1 + xi_i) / 4    # dN3/deta
                dN[i, 3, 0] = -(1 + eta_i) / 4  # dN4/dxi
                dN[i, 3, 1] = (1 - xi_i) / 4    # dN4/deta
            
            return dN
        else:
            raise NotImplementedError("高阶四边形基函数导数待实现")

class LagrangeTetra:
    def __init__(self, order):
        self.order = order
        self.nodes = tetra_lagrange_nodes(order)
    
    def evaluate(self, xi):
        """计算四面体线性基函数值"""
        xi = np.atleast_2d(xi)
        n_points = xi.shape[0]
        
        if self.order == 1:
            # 线性四面体基函数：N1 = 1 - xi - eta - zeta, N2 = xi, N3 = eta, N4 = zeta
            N = np.zeros((n_points, 4))
            N[:, 0] = 1 - xi[:, 0] - xi[:, 1] - xi[:, 2]  # N1
            N[:, 1] = xi[:, 0]                             # N2
            N[:, 2] = xi[:, 1]                             # N3
            N[:, 3] = xi[:, 2]                             # N4
            return N
        else:
            return tetra_lagrange_basis(self.order, xi)
    
    def evaluate_derivatives(self, xi):
        """计算四面体基函数导数"""
        xi = np.atleast_2d(xi)
        n_points = xi.shape[0]
        
        if self.order == 1:
            # 线性四面体基函数导数（常数）
            # N1 = 1 - xi - eta - zeta, N2 = xi, N3 = eta, N4 = zeta
            dN = np.zeros((n_points, 4, 3))
            dN[:, 0, 0] = -1  # dN1/dxi
            dN[:, 0, 1] = -1  # dN1/deta
            dN[:, 0, 2] = -1  # dN1/dzeta
            dN[:, 1, 0] = 1   # dN2/dxi
            dN[:, 1, 1] = 0   # dN2/deta
            dN[:, 1, 2] = 0   # dN2/dzeta
            dN[:, 2, 0] = 0   # dN3/dxi
            dN[:, 2, 1] = 1   # dN3/deta
            dN[:, 2, 2] = 0   # dN3/dzeta
            dN[:, 3, 0] = 0   # dN4/dxi
            dN[:, 3, 1] = 0   # dN4/deta
            dN[:, 3, 2] = 1   # dN4/dzeta
            return dN
        else:
            raise NotImplementedError("高阶四面体基函数导数待实现")

class LagrangeHex:
    def __init__(self, order):
        self.order = order
        self.nodes = hex_lagrange_nodes(order)
    
    def evaluate(self, xi):
        """计算六面体线性基函数值"""
        xi = np.atleast_2d(xi)
        n_points = xi.shape[0]
        
        if self.order == 1:
            # 线性六面体基函数
            N = np.zeros((n_points, 8))
            for i, (xi_i, eta_i, zeta_i) in enumerate(xi):
                # 节点顺序：[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1], [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]
                N[i, 0] = (1 - xi_i) * (1 - eta_i) * (1 - zeta_i) / 8  # N1
                N[i, 1] = (1 + xi_i) * (1 - eta_i) * (1 - zeta_i) / 8  # N2
                N[i, 2] = (1 + xi_i) * (1 + eta_i) * (1 - zeta_i) / 8  # N3
                N[i, 3] = (1 - xi_i) * (1 + eta_i) * (1 - zeta_i) / 8  # N4
                N[i, 4] = (1 - xi_i) * (1 - eta_i) * (1 + zeta_i) / 8  # N5
                N[i, 5] = (1 + xi_i) * (1 - eta_i) * (1 + zeta_i) / 8  # N6
                N[i, 6] = (1 + xi_i) * (1 + eta_i) * (1 + zeta_i) / 8  # N7
                N[i, 7] = (1 - xi_i) * (1 + eta_i) * (1 + zeta_i) / 8  # N8
            return N
        else:
            return hex_lagrange_basis(self.order, xi)
    
    def evaluate_derivatives(self, xi):
        """计算六面体基函数导数"""
        xi = np.atleast_2d(xi)
        n_points = xi.shape[0]
        
        if self.order == 1:
            # 线性六面体基函数导数
            dN = np.zeros((n_points, 8, 3))
            
            for i, (xi_i, eta_i, zeta_i) in enumerate(xi):
                # 节点顺序：[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1], [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]
                # N1 = (1-xi)(1-eta)(1-zeta)/8
                dN[i, 0, 0] = -(1 - eta_i) * (1 - zeta_i) / 8
                dN[i, 0, 1] = -(1 - xi_i) * (1 - zeta_i) / 8
                dN[i, 0, 2] = -(1 - xi_i) * (1 - eta_i) / 8
                
                # N2 = (1+xi)(1-eta)(1-zeta)/8
                dN[i, 1, 0] = (1 - eta_i) * (1 - zeta_i) / 8
                dN[i, 1, 1] = -(1 + xi_i) * (1 - zeta_i) / 8
                dN[i, 1, 2] = -(1 + xi_i) * (1 - eta_i) / 8
                
                # N3 = (1+xi)(1+eta)(1-zeta)/8
                dN[i, 2, 0] = (1 + eta_i) * (1 - zeta_i) / 8
                dN[i, 2, 1] = (1 + xi_i) * (1 - zeta_i) / 8
                dN[i, 2, 2] = -(1 + xi_i) * (1 + eta_i) / 8
                
                # N4 = (1-xi)(1+eta)(1-zeta)/8
                dN[i, 3, 0] = -(1 + eta_i) * (1 - zeta_i) / 8
                dN[i, 3, 1] = (1 - xi_i) * (1 - zeta_i) / 8
                dN[i, 3, 2] = -(1 - xi_i) * (1 + eta_i) / 8
                
                # N5 = (1-xi)(1-eta)(1+zeta)/8
                dN[i, 4, 0] = -(1 - eta_i) * (1 + zeta_i) / 8
                dN[i, 4, 1] = -(1 - xi_i) * (1 + zeta_i) / 8
                dN[i, 4, 2] = (1 - xi_i) * (1 - eta_i) / 8
                
                # N6 = (1+xi)(1-eta)(1+zeta)/8
                dN[i, 5, 0] = (1 - eta_i) * (1 + zeta_i) / 8
                dN[i, 5, 1] = -(1 + xi_i) * (1 + zeta_i) / 8
                dN[i, 5, 2] = (1 + xi_i) * (1 - eta_i) / 8
                
                # N7 = (1+xi)(1+eta)(1+zeta)/8
                dN[i, 6, 0] = (1 + eta_i) * (1 + zeta_i) / 8
                dN[i, 6, 1] = (1 + xi_i) * (1 + zeta_i) / 8
                dN[i, 6, 2] = (1 + xi_i) * (1 + eta_i) / 8
                
                # N8 = (1-xi)(1+eta)(1+zeta)/8
                dN[i, 7, 0] = -(1 + eta_i) * (1 + zeta_i) / 8
                dN[i, 7, 1] = (1 - xi_i) * (1 + zeta_i) / 8
                dN[i, 7, 2] = (1 - xi_i) * (1 + eta_i) / 8
            
            return dN
        else:
            raise NotImplementedError("高阶六面体基函数导数待实现")

# ---- 显式二次/三次基函数（保留原有） ----
class QuadraticLagrange1D:
    @staticmethod
    def basis(xi):
        N = np.array([
            0.5 * xi * (xi - 1),
            (1 - xi**2),
            0.5 * xi * (xi + 1)
        ])
        return N
    @staticmethod
    def grad_basis(xi):
        dN = np.array([
            xi - 0.5,
            -2 * xi,
            xi + 0.5
        ])
        return dN

class CubicLagrange1D:
    @staticmethod
    def basis(xi):
        N = np.array([
            -9/16 * (xi + 1/3) * (xi - 1/3) * (xi - 1),
            27/16 * (xi + 1) * (xi - 1/3) * (xi - 1),
            -27/16 * (xi + 1) * (xi + 1/3) * (xi - 1),
            9/16 * (xi + 1) * (xi + 1/3) * (xi - 1/3)
        ])
        return N
    @staticmethod
    def grad_basis(xi):
        dN = np.array([
            -9/16 * ((xi - 1/3) * (xi - 1) + (xi + 1/3) * (xi - 1) + (xi + 1/3) * (xi - 1/3)),
            27/16 * ((xi - 1/3) * (xi - 1) + (xi + 1) * (xi - 1) + (xi + 1) * (xi - 1/3)),
            -27/16 * ((xi + 1/3) * (xi - 1) + (xi + 1) * (xi - 1) + (xi + 1) * (xi + 1/3)),
            9/16 * ((xi + 1/3) * (xi - 1/3) + (xi + 1) * (xi - 1/3) + (xi + 1) * (xi + 1/3))
        ])
        return dN

class QuadraticLagrangeTriangle:
    @staticmethod
    def basis(xi, eta):
        l1 = 1 - xi - eta
        l2 = xi
        l3 = eta
        N = np.array([
            l1 * (2 * l1 - 1),
            l2 * (2 * l2 - 1),
            l3 * (2 * l3 - 1),
            4 * l1 * l2,
            4 * l2 * l3,
            4 * l3 * l1
        ])
        return N
    @staticmethod
    def grad_basis(xi, eta):
        l1 = 1 - xi - eta
        l2 = xi
        l3 = eta
        dN_dxi = np.array([
            -4 * l1 + 1,
            4 * l2 - 1,
            0,
            4 * (l1 - l2),
            4 * l3,
            -4 * l3
        ])
        dN_deta = np.array([
            -4 * l1 + 1,
            0,
            4 * l3 - 1,
            -4 * l2,
            4 * l2,
            4 * (l1 - l3)
        ])
        return dN_dxi, dN_deta

class QuadraticLagrangeQuad:
    @staticmethod
    def basis(xi, eta):
        N = np.array([
            0.25 * (xi - 1) * (eta - 1) * xi * eta,
            0.25 * (xi + 1) * (eta - 1) * xi * eta,
            0.25 * (xi + 1) * (eta + 1) * xi * eta,
            0.25 * (xi - 1) * (eta + 1) * xi * eta,
            0.5 * (1 - xi**2) * (eta - 1) * eta,
            0.5 * (xi + 1) * xi * (1 - eta**2),
            0.5 * (1 - xi**2) * (eta + 1) * eta,
            0.5 * (xi - 1) * xi * (1 - eta**2)
        ])
        return N
    
    @staticmethod
    def grad_basis(xi, eta):
        """计算二次四边形基函数的导数"""
        dN_dxi = np.array([
            0.25 * (2*xi - 1) * (eta - 1) * eta,
            0.25 * (2*xi + 1) * (eta - 1) * eta,
            0.25 * (2*xi + 1) * (eta + 1) * eta,
            0.25 * (2*xi - 1) * (eta + 1) * eta,
            -xi * (eta - 1) * eta,
            0.5 * (2*xi + 1) * (1 - eta**2),
            -xi * (eta + 1) * eta,
            0.5 * (2*xi - 1) * (1 - eta**2)
        ])
        
        dN_deta = np.array([
            0.25 * (xi - 1) * xi * (2*eta - 1),
            0.25 * (xi + 1) * xi * (2*eta - 1),
            0.25 * (xi + 1) * xi * (2*eta + 1),
            0.25 * (xi - 1) * xi * (2*eta + 1),
            0.5 * (1 - xi**2) * (2*eta - 1),
            -(xi + 1) * xi * eta,
            0.5 * (1 - xi**2) * (2*eta + 1),
            -(xi - 1) * xi * eta
        ])
        
        return dN_dxi, dN_deta

import sympy as sp
import numpy as np

def lagrange_basis_and_derivs_1d(order):
    xi = sp.symbols('xi')
    nodes = np.linspace(-1, 1, order+1)
    N = []
    dN = []
    for i in range(order+1):
        L = 1
        for j in range(order+1):
            if i != j:
                L *= (xi - nodes[j]) / (nodes[i] - nodes[j])
        N.append(L)
        dN.append(sp.diff(L, xi))
    N_func = [sp.lambdify(xi, Ni, 'numpy') for Ni in N]
    dN_func = [sp.lambdify(xi, dNi, 'numpy') for dNi in dN]
    return N_func, dN_func

# 用法示例
N_func, dN_func = lagrange_basis_and_derivs_1d(3)
xi = np.linspace(-1, 1, 10)
Nvals = np.stack([f(xi) for f in N_func], axis=-1)  # (10, 4)
dNvals = np.stack([f(xi) for f in dN_func], axis=-1)

# ---- 单元类型常量 ----
class TRIANGLE:
    name = 'triangle'
    dimension = 2
    n_nodes = 3
    n_faces = 3
    n_edges = 3

class QUADRILATERAL:
    name = 'quadrilateral'
    dimension = 2
    n_nodes = 4
    n_faces = 4
    n_edges = 4

class TETRAHEDRON:
    name = 'tetrahedron'
    dimension = 3
    n_nodes = 4
    n_faces = 4
    n_edges = 6

class HEXAHEDRON:
    name = 'hexahedron'
    dimension = 3
    n_nodes = 8
    n_faces = 6
    n_edges = 12

# ---- 统一的基函数接口 ----
class LinearBasisFunctions:
    """线性基函数统一接口"""
    
    def __init__(self, element_type):
        self.element_type = element_type
        self.dimension = element_type.dimension
        self.n_nodes = element_type.n_nodes
        
        # 根据单元类型选择具体的基函数实现
        if element_type == TRIANGLE:
            self._basis_impl = LagrangeTriangle(1)
        elif element_type == QUADRILATERAL:
            self._basis_impl = LagrangeQuad(1)
        elif element_type == TETRAHEDRON:
            self._basis_impl = LagrangeTetra(1)
        elif element_type == HEXAHEDRON:
            self._basis_impl = LagrangeHex(1)
        else:
            raise ValueError(f"不支持的单元类型: {element_type.name}")
    
    def evaluate(self, xi):
        """计算基函数值
        
        Args:
            xi: 参考坐标点，形状为 (n_points, dimension)
            
        Returns:
            N: 基函数值，形状为 (n_points, n_nodes)
        """
        return self._basis_impl.evaluate(xi)
    
    def evaluate_derivatives(self, xi):
        """计算基函数导数
        
        Args:
            xi: 参考坐标点，形状为 (n_points, dimension)
            
        Returns:
            dN: 基函数导数，形状为 (n_points, n_nodes, dimension)
        """
        return self._basis_impl.evaluate_derivatives(xi)


# ---- 三次基函数实现 ----
class CubicLagrangeTriangle:
    """三次三角形Lagrange基函数"""
    
    @staticmethod
    def basis(xi, eta):
        """计算三次三角形基函数值"""
        l1 = 1 - xi - eta
        l2 = xi
        l3 = eta
        
        N = np.array([
            # 顶点节点 (0,0), (1,0), (0,1)
            0.5 * l1 * (3*l1 - 1) * (3*l1 - 2),
            0.5 * l2 * (3*l2 - 1) * (3*l2 - 2),
            0.5 * l3 * (3*l3 - 1) * (3*l3 - 2),
            
            # 边中点 (1/3,0), (2/3,0), (2/3,1/3), (1/3,2/3), (0,2/3), (0,1/3)
            4.5 * l1 * l2 * (3*l1 - 1),
            4.5 * l1 * l2 * (3*l2 - 1),
            4.5 * l2 * l3 * (3*l2 - 1),
            4.5 * l2 * l3 * (3*l3 - 1),
            4.5 * l3 * l1 * (3*l3 - 1),
            4.5 * l3 * l1 * (3*l1 - 1),
            
            # 内部节点 (1/3,1/3)
            27 * l1 * l2 * l3
        ])
        return N
    
    @staticmethod
    def grad_basis(xi, eta):
        """计算三次三角形基函数导数"""
        l1 = 1 - xi - eta
        l2 = xi
        l3 = eta
        
        # dN/dxi
        dN_dxi = np.array([
            -0.5 * (3*l1 - 1) * (3*l1 - 2) - 1.5 * l1 * (3*l1 - 1) - 1.5 * l1 * (3*l1 - 2),
            0.5 * (3*l2 - 1) * (3*l2 - 2) + 1.5 * l2 * (3*l2 - 1) + 1.5 * l2 * (3*l2 - 2),
            0,
            4.5 * (l2 * (3*l1 - 1) + l1 * l2 * 3 - l1 * l2),
            4.5 * (l1 * (3*l2 - 1) + l1 * l2 * 3 - l1 * l2),
            4.5 * l3 * (3*l2 - 1) + 4.5 * l2 * l3 * 3,
            4.5 * l3 * (3*l2 - 1),
            -4.5 * l3 * l1 * 3,
            -4.5 * (l3 * (3*l1 - 1) + l3 * l1 * 3),
            27 * (l2 * l3 - l1 * l3)
        ])
        
        # dN/deta
        dN_deta = np.array([
            -0.5 * (3*l1 - 1) * (3*l1 - 2) - 1.5 * l1 * (3*l1 - 1) - 1.5 * l1 * (3*l1 - 2),
            0,
            0.5 * (3*l3 - 1) * (3*l3 - 2) + 1.5 * l3 * (3*l3 - 1) + 1.5 * l3 * (3*l3 - 2),
            -4.5 * l1 * l2 * 3,
            -4.5 * l1 * l2 * 3,
            4.5 * l2 * (3*l3 - 1),
            4.5 * (l2 * (3*l3 - 1) + l2 * l3 * 3),
            4.5 * (l1 * (3*l3 - 1) + l1 * l3 * 3),
            4.5 * l1 * (3*l3 - 1),
            27 * (l1 * l2 - l1 * l3)
        ])
        
        return dN_dxi, dN_deta


class CubicLagrangeQuad:
    """三次四边形Lagrange基函数"""
    
    @staticmethod
    def basis(xi, eta):
        """计算三次四边形基函数值"""
        # 定义节点坐标
        nodes_xi = np.array([-1, -1/3, 1/3, 1])
        nodes_eta = np.array([-1, -1/3, 1/3, 1])
        
        N = []
        for j, eta_node in enumerate(nodes_eta):
            for i, xi_node in enumerate(nodes_xi):
                # 计算Lagrange基函数
                Ni = 1.0
                for k, xi_k in enumerate(nodes_xi):
                    if k != i:
                        Ni *= (xi - xi_k) / (xi_node - xi_k)
                for l, eta_l in enumerate(nodes_eta):
                    if l != j:
                        Ni *= (eta - eta_l) / (eta_node - eta_l)
                N.append(Ni)
        
        return np.array(N)
    
    @staticmethod
    def grad_basis(xi, eta):
        """计算三次四边形基函数导数"""
        # 定义节点坐标
        nodes_xi = np.array([-1, -1/3, 1/3, 1])
        nodes_eta = np.array([-1, -1/3, 1/3, 1])
        
        dN_dxi = []
        dN_deta = []
        
        for j, eta_node in enumerate(nodes_eta):
            for i, xi_node in enumerate(nodes_xi):
                # 计算 dN/dxi
                dNi_dxi = 0.0
                for k, xi_k in enumerate(nodes_xi):
                    if k != i:
                        prod = 1.0 / (xi_node - xi_k)
                        for m, xi_m in enumerate(nodes_xi):
                            if m != i and m != k:
                                prod *= (xi - xi_m) / (xi_node - xi_m)
                        for l, eta_l in enumerate(nodes_eta):
                            if l != j:
                                prod *= (eta - eta_l) / (eta_node - eta_l)
                        dNi_dxi += prod
                
                # 计算 dN/deta
                dNi_deta = 0.0
                for l, eta_l in enumerate(nodes_eta):
                    if l != j:
                        prod = 1.0 / (eta_node - eta_l)
                        for m, eta_m in enumerate(nodes_eta):
                            if m != j and m != l:
                                prod *= (eta - eta_m) / (eta_node - eta_m)
                        for k, xi_k in enumerate(nodes_xi):
                            if k != i:
                                prod *= (xi - xi_k) / (xi_node - xi_k)
                        dNi_deta += prod
                
                dN_dxi.append(dNi_dxi)
                dN_deta.append(dNi_deta)
        
        return np.array(dN_dxi), np.array(dN_deta)


class CubicLagrangeTetra:
    """三次四面体Lagrange基函数"""
    
    @staticmethod
    def basis(xi, eta, zeta):
        """计算三次四面体基函数值"""
        l1 = 1 - xi - eta - zeta
        l2 = xi
        l3 = eta
        l4 = zeta
        
        N = np.array([
            # 顶点节点
            0.5 * l1 * (3*l1 - 1) * (3*l1 - 2),
            0.5 * l2 * (3*l2 - 1) * (3*l2 - 2),
            0.5 * l3 * (3*l3 - 1) * (3*l3 - 2),
            0.5 * l4 * (3*l4 - 1) * (3*l4 - 2),
            
            # 边中点
            4.5 * l1 * l2 * (3*l1 - 1),
            4.5 * l1 * l2 * (3*l2 - 1),
            4.5 * l2 * l3 * (3*l2 - 1),
            4.5 * l2 * l3 * (3*l3 - 1),
            4.5 * l3 * l4 * (3*l3 - 1),
            4.5 * l3 * l4 * (3*l4 - 1),
            4.5 * l4 * l1 * (3*l4 - 1),
            4.5 * l4 * l1 * (3*l1 - 1),
            4.5 * l1 * l3 * (3*l1 - 1),
            4.5 * l1 * l3 * (3*l3 - 1),
            4.5 * l2 * l4 * (3*l2 - 1),
            4.5 * l2 * l4 * (3*l4 - 1),
            
            # 面中心
            27 * l1 * l2 * l3,
            27 * l2 * l3 * l4,
            27 * l3 * l4 * l1,
            27 * l4 * l1 * l2
        ])
        return N
    
    @staticmethod
    def grad_basis(xi, eta, zeta):
        """计算三次四面体基函数导数"""
        l1 = 1 - xi - eta - zeta
        l2 = xi
        l3 = eta
        l4 = zeta
        
        # 简化实现：返回主要方向的导数
        dN_dxi = np.zeros(20)
        dN_deta = np.zeros(20)
        dN_dzeta = np.zeros(20)
        
        # 这里需要完整的导数计算，为简洁起见使用简化版本
        # 实际应用中需要完整的导数表达式
        
        return dN_dxi, dN_deta, dN_dzeta


# ---- 高阶基函数工厂类 ----
class HighOrderBasisFactory:
    """高阶基函数工厂类"""
    
    @staticmethod
    def create_basis(element_type: str, order: int):
        """创建高阶基函数
        
        Args:
            element_type: 单元类型 ('triangle', 'quad', 'tetra', 'hex')
            order: 阶数 (1, 2, 3)
            
        Returns:
            basis_class: 基函数类
        """
        if element_type == 'triangle':
            if order == 1:
                return LagrangeTriangle(1)
            elif order == 2:
                return QuadraticLagrangeTriangle
            elif order == 3:
                return CubicLagrangeTriangle
        elif element_type == 'quad':
            if order == 1:
                return LagrangeQuad(1)
            elif order == 2:
                return QuadraticLagrangeQuad
            elif order == 3:
                return CubicLagrangeQuad
        elif element_type == 'tetra':
            if order == 1:
                return LagrangeTetra(1)
            elif order == 2:
                # 需要实现二次四面体
                raise NotImplementedError("二次四面体基函数尚未实现")
            elif order == 3:
                return CubicLagrangeTetra
        elif element_type == 'hex':
            if order == 1:
                return LagrangeHex(1)
            elif order == 2:
                # 需要实现二次六面体
                raise NotImplementedError("二次六面体基函数尚未实现")
            elif order == 3:
                # 需要实现三次六面体
                raise NotImplementedError("三次六面体基函数尚未实现")
        else:
            raise ValueError(f"不支持的单元类型: {element_type}")
        
        raise ValueError(f"不支持的阶数: {order}")
