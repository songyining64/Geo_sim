"""
有限元方法模块

提供完整的有限元方法实现，包括：
- 基函数（线性、二次、三次）
- 高斯积分
- 等参变换
- 矩阵组装
- 边界条件处理
- 高性能求解器
"""

from .basis_functions import *
from .quadrature import *
from .transformations import *
from .assembly import *
from .global_assembly import *
from .elements import *
from .boundary_conditions import *
from .dof_manager import *

# 导入求解器模块
try:
    from .solvers import *
    SOLVERS_AVAILABLE = True
except ImportError:
    SOLVERS_AVAILABLE = False

__version__ = "0.2.0"
__all__ = [
    # 基函数类
    "LinearBasisFunctions",
    "LagrangeTriangle",
    "LagrangeQuad", 
    "LagrangeTetra",
    "LagrangeHex",
    "QuadraticLagrange1D",
    "CubicLagrange1D",
    "QuadraticLagrangeTriangle",
    "QuadraticLagrangeQuad",
    
    # 单元类型常量
    "TRIANGLE",
    "QUADRILATERAL", 
    "TETRAHEDRON",
    "HEXAHEDRON",
    
    # 积分类
    "triangle_points_weights",
    "quad_points_weights",
    "tetra_points_weights",
    "hex_points_weights",
    
    # 变换类
    "jacobian_matrix",
    "jacobian_det",
    "dN_dx",
    
    # 组装类
    "ElementAssembly",
    "GlobalAssembly",
    "ParallelAssembly",
    "BoundaryAssembly",
    "triangle_stiffness",
    "quad_stiffness",
    "tetra_stiffness",
    "hex_stiffness",
    "elasticity_triangle_stiffness",
    
    # 边界条件类
    "DirichletBC",
    "NeumannBC", 
    "RobinBC",
    "AdaptiveBoundaryConditions",
    "ImmersedBoundaryMethod",
    "create_dirichlet_bc",
    "create_neumann_bc", 
    "create_robin_bc",
    
    # 求解器类（如果可用）
    "SolverConfig",
    "SolverFactory",
    "LinearSolver",
    "DirectSolver",
    "IterativeSolver",
    "PETScSolver",
    "MultigridSolver",
    "NonlinearSolver",
    "TimeStepper",
    "solve_linear_system",
    "solve_nonlinear_system",
] if SOLVERS_AVAILABLE else [
    # 基函数类
    "LinearBasisFunctions",
    "LagrangeTriangle",
    "LagrangeQuad", 
    "LagrangeTetra",
    "LagrangeHex",
    "QuadraticLagrange1D",
    "CubicLagrange1D",
    "QuadraticLagrangeTriangle",
    "QuadraticLagrangeQuad",
    
    # 单元类型常量
    "TRIANGLE",
    "QUADRILATERAL", 
    "TETRAHEDRON",
    "HEXAHEDRON",
    
    # 积分类
    "triangle_points_weights",
    "quad_points_weights",
    "tetra_points_weights",
    "hex_points_weights",
    
    # 变换类
    "jacobian_matrix",
    "jacobian_det",
    "dN_dx",
    
    # 组装类
    "ElementAssembly",
    "GlobalAssembly",
    "ParallelAssembly",
    "BoundaryAssembly",
    "triangle_stiffness",
    "quad_stiffness",
    "tetra_stiffness",
    "hex_stiffness",
    "elasticity_triangle_stiffness",
    
    # 边界条件类
    "DirichletBC",
    "NeumannBC", 
    "RobinBC",
    "AdaptiveBoundaryConditions",
    "ImmersedBoundaryMethod",
    "create_dirichlet_bc",
    "create_neumann_bc", 
    "create_robin_bc",
] 