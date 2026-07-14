"""Optional PyAMG/PETSc velocity-block baselines for Stokes Picard matrices."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stokes_solver import PicardStokesSolver, StokesConfig


def _relative_residual(A, b, x):
    residual = np.linalg.norm(b - A @ x)
    return float(residual / max(np.linalg.norm(b), 1e-14))


def collect_velocity_case(nx: int, ny: int, seed: int):
    solver = PicardStokesSolver(StokesConfig(
        nx=nx, ny=ny, rayleigh=1e4, viscosity_contrast=1e2,
        max_picard_iterations=2, picard_tolerance=0.0,
        use_meta_amg=False,
    ))
    sequence = solver.collect_picard_sequence(max_iterations=2, seed=seed)
    record = sequence[-1]
    A = record['matrix'].tocsr()
    b = record['rhs'][record['velocity_indices']]
    return A, b


def benchmark_scipy_direct(A, b):
    from scipy.sparse.linalg import spsolve
    t0 = time.time()
    x = spsolve(A, b)
    return {
        'available': True,
        'setup_time': 0.0,
        'solve_time': float(time.time() - t0),
        'relative_residual': _relative_residual(A, b, x),
    }


def benchmark_pyamg(A, b):
    if importlib.util.find_spec('pyamg') is None:
        return {'available': False, 'reason': 'pyamg is not installed'}
    import pyamg

    attempts = {}
    for name, builder in [
        ('ruge_stuben', pyamg.ruge_stuben_solver),
        ('smoothed_aggregation', pyamg.smoothed_aggregation_solver),
    ]:
        try:
            t0 = time.time()
            ml = builder(A)
            setup_time = time.time() - t0
            t0 = time.time()
            x = ml.solve(b, tol=1e-8, maxiter=200)
            solve_time = time.time() - t0
            attempts[name] = {
                'available': True,
                'setup_time': float(setup_time),
                'solve_time': float(solve_time),
                'relative_residual': _relative_residual(A, b, x),
                'n_levels': len(ml.levels),
            }
        except Exception as exc:
            attempts[name] = {
                'available': True,
                'failed': True,
                'reason': str(exc),
            }
    successful = {k: v for k, v in attempts.items() if not v.get('failed')}
    return {
        'available': True,
        'best_method': min(successful, key=lambda k: successful[k]['relative_residual']) if successful else None,
        'attempts': attempts,
    }


def benchmark_petsc(A, b):
    if importlib.util.find_spec('petsc4py') is None:
        return {'available': False, 'reason': 'petsc4py is not installed'}
    from finite_elements.solvers import PETScSolver, SolverConfig

    attempts = {}
    for name, method, preconditioner in [
        ('cg_gamg', 'cg', 'gamg'),
        ('gmres_gamg', 'gmres', 'gamg'),
        ('gmres_ilu', 'gmres', 'ilu'),
        ('preonly_lu', 'preonly', 'lu'),
    ]:
        try:
            config = SolverConfig(
                solver_type='petsc', method=method, preconditioner=preconditioner,
                tolerance=1e-8, max_iterations=500)
            solver = PETScSolver(config)
            t0 = time.time()
            solver.setup(A)
            setup_time = time.time() - t0
            t0 = time.time()
            x = solver.solve(A, b)
            solve_time = time.time() - t0
            attempts[name] = {
                'available': True,
                'setup_time': float(setup_time),
                'solve_time': float(solve_time),
                'relative_residual': _relative_residual(A, b, x),
            }
        except Exception as exc:
            attempts[name] = {
                'available': True,
                'failed': True,
                'reason': str(exc),
            }
    successful = {k: v for k, v in attempts.items() if not v.get('failed')}
    return {
        'available': True,
        'best_method': min(successful, key=lambda k: successful[k]['relative_residual']) if successful else None,
        'attempts': attempts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, default=16)
    parser.add_argument('--ny', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='experiments/results_external_velocity_baselines')
    args = parser.parse_args()

    A, b = collect_velocity_case(args.nx, args.ny, args.seed)
    payload = {
        'config': {'nx': args.nx, 'ny': args.ny, 'seed': args.seed, 'matrix_size': A.shape[0], 'nnz': A.nnz},
        'scipy_direct': benchmark_scipy_direct(A, b),
        'pyamg': benchmark_pyamg(A, b),
        'petsc': benchmark_petsc(A, b),
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'external_velocity_baselines.json'
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
