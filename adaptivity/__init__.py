"""
自适应网格模块 - 提供网格细化、粗化和误差估计功能
"""

from .error_estimator import ErrorEstimator
from .mesh_refinement import MeshRefinement
from .advanced_mesh import AdaptiveMeshRefiner

__all__ = [
    'ErrorEstimator',
    'MeshRefinement',
    'AdaptiveMeshRefiner'
] 