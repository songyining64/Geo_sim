import numpy as np
import scipy.sparse as sp

from experiments.run_experiments import (
    _relative_matrix_change,
    _relative_residual,
    _solution_is_acceptable,
    _traditional_step,
    benchmark_sequence_methods,
)
from gpu_acceleration.meta_amg import MetaAMGConfig, MetaAMGSolver
from solvers.multigrid_solver import AlgebraicMultigridSolver, MultigridConfig


def test_solution_quality_uses_relative_residual_and_rejects_nonfinite_values():
    A = sp.eye(3, format='csr')
    b = np.array([1.0, 2.0, 3.0])

    assert _solution_is_acceptable(A, b, b.copy())
    assert not _solution_is_acceptable(A, b, np.array([np.inf, 0.0, 0.0]))
    assert not _solution_is_acceptable(A, b, np.zeros(3))


def test_traditional_benchmark_reports_a_finite_acceptable_solution():
    A = sp.diags([2.0, 3.0, 4.0], format='csr')
    b = np.array([2.0, 6.0, 8.0])
    step = _traditional_step(MetaAMGSolver(MetaAMGConfig()), A, b)

    assert step['accepted']
    assert np.isfinite(step['residual_norm'])
    assert step['solver_backend'] in {'amg', 'direct', 'least_squares'}


def test_relative_residual_is_scale_invariant_and_handles_zero_rhs():
    A = sp.diags([2.0, 3.0, 4.0], format='csr')
    b = np.array([2.0, 6.0, 8.0])
    x = np.array([0.9, 1.9, 1.9])

    assert np.isclose(_relative_residual(A, b, x),
                      _relative_residual(100.0 * A, 100.0 * b, x))
    assert _relative_residual(A, np.zeros(3), np.zeros(3)) == 0.0


def test_sparse_relative_matrix_change():
    A = sp.eye(1000, format='csr')
    changed = 1.05 * A

    assert np.isclose(_relative_matrix_change(changed, A), 0.05)
    assert np.isinf(_relative_matrix_change(sp.eye(2), A))


class _FixedAdapter:
    @staticmethod
    def zero_shot_predict(A):
        coarse = list(range(0, A.shape[0], 2))
        return coarse, [i for i in range(A.shape[0]) if i not in coarse]

    @staticmethod
    def adapt(A_prev, labels, A_curr, adapt_steps=3):
        coarse = np.flatnonzero(labels > 0.5).tolist()
        return coarse, [i for i in range(A_curr.shape[0]) if i not in coarse]


class _FixedMeta:
    adapter = _FixedAdapter()


def test_periodic_and_change_aware_baselines_rebuild_at_expected_steps():
    matrices = [scale * sp.eye(6, format='csr') for scale in [1.0, 1.05, 1.10, 1.12]]
    rhs = [A @ np.ones(6) for A in matrices]
    results = benchmark_sequence_methods(
        MetaAMGConfig(), matrices, rhs, _FixedMeta(),
        periodic_rebuild_interval=2, matrix_change_threshold=0.08)

    periodic = results['periodic_rebuild']['steps']
    assert [step['rebuild'] for step in periodic] == [True, False, True, False]
    change_aware = results['change_aware_reuse']['steps']
    assert [step['rebuild'] for step in change_aware] == [True, False, True, False]
    assert change_aware[2]['decision_reason'] == 'matrix_change_rebuild'
    assert results['change_aware_reuse']['relative_residual_max'] <= 1e-6


def test_coarse_direct_factorization_is_cached(monkeypatch):
    import solvers.multigrid_solver as multigrid_module

    calls = {'count': 0}
    real_splu = multigrid_module.splu

    def counting_splu(A):
        calls['count'] += 1
        return real_splu(A)

    monkeypatch.setattr(multigrid_module, 'splu', counting_splu)
    solver = AlgebraicMultigridSolver(MultigridConfig(coarse_solver='direct'))
    A = sp.diags([2.0, 3.0, 4.0], format='csr')
    b = np.ones(3)

    first = solver._solve_coarse_system(A, b)
    second = solver._solve_coarse_system(A, b)

    assert calls['count'] == 1
    assert np.allclose(first, second)
