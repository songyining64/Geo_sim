"""
自适应网格模块 - 提供网格细化、粗化和误差估计功能
"""

from .error_estimator import ErrorEstimator
from .mesh_refinement import MeshRefiner
from .adaptive_solver import AdaptiveSolver

__all__ = [
    'ErrorEstimator',
    'MeshRefiner', 
    'AdaptiveSolver'
] 