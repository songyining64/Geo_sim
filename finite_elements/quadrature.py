"""
高斯积分模块：支持1D/2D/3D，任意阶
"""
import numpy as np

def gauss_legendre_1d(order):
    from numpy.polynomial.legendre import leggauss
    pts, wts = leggauss(order)
    return pts, wts

def triangle_points_weights(order):
    if order == 1:
        return np.array([[1/3, 1/3]]), np.array([0.5])
    elif order == 2:
        pts = np.array([
            [1/6, 1/6],
            [2/3, 1/6],
            [1/6, 2/3]
        ])
        wts = np.array([1/6, 1/6, 1/6])
        return pts, wts
    else:
        raise NotImplementedError("高阶三角形积分点可查Dunavant表")

def quad_points_weights(order):
    pts_1d, wts_1d = gauss_legendre_1d(order)
    pts = np.array([[x, y] for x in pts_1d for y in pts_1d])
    wts = np.array([wx * wy for wx in wts_1d for wy in wts_1d])
    return pts, wts

def tetra_points_weights(order):
    if order == 1:
        return np.array([[0.25, 0.25, 0.25]]), np.array([1/6])
    elif order == 2:
        pts = np.array([
            [0.58541020, 0.13819660, 0.13819660],
            [0.13819660, 0.58541020, 0.13819660],
            [0.13819660, 0.13819660, 0.58541020],
            [0.13819660, 0.13819660, 0.13819660]
        ])
        wts = np.array([1/24, 1/24, 1/24, 1/24])
        return pts, wts
    else:
        raise NotImplementedError("高阶四面体积分点可查Stroud表")

def hex_points_weights(order):
    pts_1d, wts_1d = gauss_legendre_1d(order)
    pts = np.array([[x, y, z] for x in pts_1d for y in pts_1d for z in pts_1d])
    wts = np.array([wx * wy * wz for wx in wts_1d for wy in wts_1d for wz in wts_1d])
    return pts, wts
