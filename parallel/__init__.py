"""
并行计算模块 - 提供域分解、并行求解器和负载均衡功能
"""

from .domain_decomposition import DomainDecomposer
from .parallel_solvers import ParallelSolver
from .advanced_parallel_solver_v2 import AdvancedParallelSolver

__all__ = [
    'DomainDecomposer',
    'ParallelSolver',
    'AdvancedParallelSolver'
] 