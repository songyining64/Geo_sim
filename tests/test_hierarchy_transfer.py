"""Tests for the quality-gated AMG hierarchy transfer path."""

import numpy as np
import scipy.sparse as sp
from solvers.hierarchy_transfer import HierarchyTransferValidator
from solvers.multigrid_solver import AdaptiveCoarsening, MultigridConfig


def poisson_matrix(n):
    side = int(np.sqrt(n))
    assert side * side == n
    main = 4.0 * np.ones(n)
    left_right = -np.ones(n - 1)
    left_right[np.arange(1, side) * side - 1] = 0.0
    up_down = -np.ones(n - side)
    return sp.diags([up_down, left_right, main, left_right, up_down],
                    [-side, -1, 0, 1, side], format="csr")


def test_algebraic_coarsening_returns_a_complete_partition():
    A = poisson_matrix(100)
    coarse, fine = AdaptiveCoarsening.algebraic_coarsening(A)
    assert coarse
    assert fine
    assert set(coarse).isdisjoint(fine)
    assert set(coarse) | set(fine) == set(range(A.shape[0]))


def test_validator_repairs_an_empty_prediction_and_measures_contraction():
    A = poisson_matrix(100)
    validator = HierarchyTransferValidator(
        MultigridConfig(max_coarse_size=20), strong_threshold=0.25,
        max_two_grid_factor=1.0,
    )
    result = validator.assess(A, [])
    assert result.repaired_points > 0
    assert result.coarse
    assert result.fine
    assert np.isfinite(result.two_grid_factor)


def test_validator_rejects_a_split_that_has_no_fine_level():
    A = poisson_matrix(64)
    validator = HierarchyTransferValidator(
        MultigridConfig(), strong_threshold=0.25,
    )
    result = validator.assess(A, list(range(A.shape[0])))
    assert not result.accepted
    assert result.reason == "no fine points"
