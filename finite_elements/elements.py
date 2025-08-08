"""
有限元单元类型注册与工厂
支持线性/二次/三次/高阶单元类型（1D/2D/3D）
"""
from .basis_functions import (
    Lagrange1D, LagrangeTriangle, LagrangeQuad, LagrangeTetra, LagrangeHex
)

# 单元类型注册表
ELEMENT_TYPE_MAP = {
    'line': Lagrange1D,         # 1D
    'triangle': LagrangeTriangle, # 2D
    'quad': LagrangeQuad,         # 2D
    'tetra': LagrangeTetra,       # 3D
    'hex': LagrangeHex,           # 3D
}

# 单元元信息（可扩展）
ELEMENT_META = {
    'line':    {'dim': 1, 'shape': 'line'},
    'triangle':{'dim': 2, 'shape': 'triangle'},
    'quad':    {'dim': 2, 'shape': 'quad'},
    'tetra':   {'dim': 3, 'shape': 'tetra'},
    'hex':     {'dim': 3, 'shape': 'hex'},
}

def create_element(element_type, order):
    """
    创建指定类型和阶次的单元基函数对象
    element_type: 'line', 'triangle', 'quad', 'tetra', 'hex'
    order: 1, 2, 3, ...
    """
    if element_type not in ELEMENT_TYPE_MAP:
        raise ValueError(f"未知单元类型: {element_type}")
    return ELEMENT_TYPE_MAP[element_type](order)

def get_element_meta(element_type):
    """
    获取单元类型的元信息（维数、形状等）
    """
    if element_type not in ELEMENT_META:
        raise ValueError(f"未知单元类型: {element_type}")
    return ELEMENT_META[element_type] 