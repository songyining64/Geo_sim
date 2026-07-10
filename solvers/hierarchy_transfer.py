"""Safety checks for transferring an AMG first level between nearby matrices.

The learned model is allowed to propose a C/F split, but it is never allowed to
silently install an invalid hierarchy.  This module repairs local coverage
violations and measures the actual two-grid residual reduction before the
proposal is used by a solver.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .multigrid_solver import AlgebraicMultigridSolver, MultigridConfig


@dataclass
class HierarchyAssessment:
    """Result of checking a learned or warm-started first AMG level."""

    accepted: bool
    coarse: List[int]
    fine: List[int]
    two_grid_factor: float = float("inf")
    operator_complexity: float = float("inf")
    reason: str = ""
    repaired_points: int = 0


class HierarchyTransferValidator:
    """Repair and validate a candidate first-level C/F split.

    Validation uses deterministic random residual probes, rather than C/F label
    accuracy.  It therefore checks the quantity that matters operationally:
    whether one two-grid cycle reduces residuals on the current matrix.
    """

    def __init__(self, config: MultigridConfig, *, strong_threshold: float,
                 min_coarse_fraction: float = 0.10,
                 max_coarse_fraction: float = 0.60,
                 max_two_grid_factor: float = 0.85,
                 max_operator_complexity: float = 3.0,
                 probes: int = 2,
                 seed: int = 0):
        self.config = config
        self.strong_threshold = strong_threshold
        self.min_coarse_fraction = min_coarse_fraction
        self.max_coarse_fraction = max_coarse_fraction
        self.max_two_grid_factor = max_two_grid_factor
        self.max_operator_complexity = max_operator_complexity
        self.probes = probes
        self.seed = seed

    def repair(self, A: sp.spmatrix, coarse: Sequence[int],
               scores: Optional[Sequence[float]] = None) -> Tuple[List[int], List[int], int]:
        """Make a candidate split usable by interpolation.

        Every F point without a strong C neighbour is promoted to C.  This is a
        local, deterministic repair: it avoids a full greedy coarsening pass.
        """
        A = A.tocsr()
        n = A.shape[0]
        valid = {int(i) for i in coarse if 0 <= int(i) < n}
        if scores is not None:
            order = np.argsort(np.asarray(scores))[::-1]
        else:
            order = np.arange(n)

        min_c = max(1, int(np.ceil(n * self.min_coarse_fraction)))
        for point in order:
            if len(valid) >= min_c:
                break
            valid.add(int(point))

        repaired = 0
        # A newly-promoted C point may make another F point valid; iterate to a
        # fixed point so the result is independent of row ordering.
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if i in valid:
                    continue
                start, end = A.indptr[i], A.indptr[i + 1]
                cols = A.indices[start:end]
                vals = A.data[start:end]
                diag = abs(A[i, i])
                threshold = self.strong_threshold * max(diag, 1e-14)
                if not any(j in valid and abs(value) >= threshold
                           for j, value in zip(cols, vals) if j != i):
                    valid.add(i)
                    repaired += 1
                    changed = True

        coarse_out = sorted(valid)
        fine_out = [i for i in range(n) if i not in valid]
        return coarse_out, fine_out, repaired

    def assess(self, A: sp.spmatrix, coarse: Sequence[int],
               scores: Optional[Sequence[float]] = None) -> HierarchyAssessment:
        """Return an accepted hierarchy only if its measured quality is safe."""
        A = A.tocsr()
        n = A.shape[0]
        coarse, fine, repaired = self.repair(A, coarse, scores)
        fraction = len(coarse) / max(n, 1)
        if not fine:
            return HierarchyAssessment(False, coarse, fine, reason="no fine points",
                                       repaired_points=repaired)
        if fraction > self.max_coarse_fraction:
            return HierarchyAssessment(False, coarse, fine,
                                       reason="coarse fraction exceeds limit",
                                       repaired_points=repaired)

        try:
            solver = AlgebraicMultigridSolver(self.config)
            P = solver._build_advanced_interpolation_operator(A, coarse, fine)
            coarse_A = (P.T @ A @ P).tocsr()
            if P.shape != (n, len(coarse)) or P.nnz == 0:
                raise ValueError("invalid interpolation shape")
            if not np.isfinite(P.data).all() or not np.isfinite(coarse_A.data).all():
                raise ValueError("non-finite hierarchy operator")
            complexity = (A.nnz + coarse_A.nnz) / max(A.nnz, 1)
            if complexity > self.max_operator_complexity:
                return HierarchyAssessment(False, coarse, fine,
                                           operator_complexity=complexity,
                                           reason="operator complexity exceeds limit",
                                           repaired_points=repaired)

            solver.levels = [
                {'matrix': A, 'size': n, 'level': 0},
                {'matrix': coarse_A, 'size': coarse_A.shape[0], 'level': 1},
            ]
            solver.interpolation_operators = [P]
            solver.restriction_operators = [P.T]
            rng = np.random.default_rng(self.seed)
            factors = []
            for _ in range(max(1, self.probes)):
                b = rng.standard_normal(n)
                before = np.linalg.norm(b)
                x = solver.v_cycle(0, b, np.zeros(n, dtype=float))
                factors.append(np.linalg.norm(b - A @ x) / max(before, 1e-14))
            factor = float(max(factors))
            return HierarchyAssessment(
                accepted=np.isfinite(factor) and factor <= self.max_two_grid_factor,
                coarse=coarse, fine=fine, two_grid_factor=factor,
                operator_complexity=complexity,
                reason="" if factor <= self.max_two_grid_factor else "two-grid contraction too weak",
                repaired_points=repaired,
            )
        except Exception as exc:
            return HierarchyAssessment(False, coarse, fine,
                                       reason=f"hierarchy construction failed: {exc}",
                                       repaired_points=repaired)
