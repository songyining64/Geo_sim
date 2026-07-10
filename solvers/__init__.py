"""
求解器模块 - 提供多重网格、多物理场耦合等求解器
"""

from .multigrid_solver import MultigridSolver, MultigridConfig
from .multiphysics_coupling_solver import MultiphysicsCouplingSolver, CouplingConfig

__all__ = [
    'MultigridSolver',
    'MultigridConfig',
    'MultiphysicsCouplingSolver',
    'CouplingConfig'
]
