"""Fair replay comparison on shared 3D variable-viscosity Stokes systems."""

import argparse
import json
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stokes_solver import BlockStokesSolver, PicardStokesSolver, StokesConfig


def _sparse_bytes(matrix):
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _mean_std(values):
    return {'mean': float(np.mean(values)), 'std': float(np.std(values))}


def _config(n, contrast, picard_iterations):
    return StokesConfig(
        nx=n, ny=n, nz=n, rayleigh=1e3, viscosity_contrast=contrast,
        max_picard_iterations=picard_iterations, picard_tolerance=0.0,
        use_meta_amg=False, pressure_solver='matrix_free_schur',
        schur_velocity_inverse='lu', meta_adapt_steps=1,
    )


def build_shared_trajectory(n, contrast, seed, picard_iterations):
    solver = PicardStokesSolver(_config(n, contrast, picard_iterations))
    start = time.perf_counter()
    records = solver.collect_picard_sequence(picard_iterations, seed=seed)
    generation_time = time.perf_counter() - start
    element_viscosities = np.concatenate([
        np.asarray([np.mean(record['viscosity'][element])
                    for element in solver.mesh.elements])
        for record in records
    ])
    nodal_viscosities = np.concatenate([record['viscosity'] for record in records])
    return solver.mesh, records, {
        'trajectory_generation_time_s': float(generation_time),
        'nodal_viscosity_min': float(np.min(nodal_viscosities)),
        'nodal_viscosity_max': float(np.max(nodal_viscosities)),
        'nodal_viscosity_contrast': float(
            np.max(nodal_viscosities) / max(np.min(nodal_viscosities), 1e-30)),
        'element_viscosity_min': float(np.min(element_viscosities)),
        'element_viscosity_max': float(np.max(element_viscosities)),
        'element_viscosity_contrast': float(
            np.max(element_viscosities) / max(np.min(element_viscosities), 1e-30)),
        'trajectory_sparse_bytes': int(sum(_sparse_bytes(record['full_matrix'])
                                           for record in records)),
    }


def _quality(solutions, records, references):
    residuals = [
        np.linalg.norm(record['full_matrix'] @ solution - record['rhs']) /
        max(np.linalg.norm(record['rhs']), 1e-14)
        for solution, record in zip(solutions, records)
    ]
    errors = [
        np.linalg.norm(solution - reference) / max(np.linalg.norm(reference), 1e-14)
        for solution, reference in zip(solutions, references)
    ] if references is not None else [0.0] * len(solutions)
    return float(max(residuals)), float(max(errors))


def replay_direct(records):
    setup_time = 0.0
    solve_time = 0.0
    solutions = []
    tracemalloc.start()
    total_start = time.perf_counter()
    for record in records:
        start = time.perf_counter()
        factor = splu(record['full_matrix'].tocsc())
        setup_time += time.perf_counter() - start
        start = time.perf_counter()
        solutions.append(factor.solve(record['rhs']))
        solve_time += time.perf_counter() - start
    total_time = time.perf_counter() - total_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    residual, error = _quality(solutions, records, None)
    return {
        'method': 'direct', 'setup_time_s': float(setup_time),
        'solve_time_s': float(solve_time), 'total_time_s': float(total_time),
        'linear_solver_calls': len(records), 'velocity_krylov_iterations': 0,
        'velocity_solve_calls': 0, 'max_linear_relative_residual': residual,
        'max_solution_relative_error_vs_direct': error,
        'python_peak_memory_bytes': int(peak), 'velocity_fallbacks': 0,
        'pressure_fallbacks': 0, 'full_fallbacks': 0,
    }, solutions


def replay_blocked(mesh, records, config, method, references, trained_meta=None):
    block = BlockStokesSolver(mesh, config)
    if method == 'meta_amg':
        block.set_meta_amg(trained_meta)
    solutions = []
    x0 = None
    tracemalloc.start()
    total_start = time.perf_counter()
    for record in records:
        x0 = block.solve(record['full_matrix'], record['rhs'], record['temperature'],
                         record['viscosity'], x0=x0, A_vel_prebuilt=record['matrix'])
        solutions.append(x0.copy())
    total_time = time.perf_counter() - total_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stats = block.get_stats()
    setup_time = (stats['traditional_hierarchy_setup_time'] +
                  stats['adapted_hierarchy_setup_time'] + stats['meta_adaptation_time'] +
                  stats.get('schur_velocity_lu_setup_time', 0.0) +
                  stats.get('schur_velocity_ilu_setup_time', 0.0))
    residual, error = _quality(solutions, records, references)
    return {
        'method': method, 'setup_time_s': float(setup_time),
        'solve_time_s': float(max(total_time - setup_time, 0.0)),
        'total_time_s': float(total_time), 'linear_solver_calls': len(records),
        'velocity_krylov_iterations': int(stats['velocity_krylov_iterations']),
        'velocity_solve_calls': int(stats['velocity_solve_calls']),
        'max_linear_relative_residual': residual,
        'max_solution_relative_error_vs_direct': error,
        'python_peak_memory_bytes': int(peak),
        'velocity_fallbacks': int(stats['velocity_direct_fallbacks']),
        'pressure_fallbacks': int(stats['pressure_solver_fallbacks']),
        'full_fallbacks': int(stats['full_direct_fallbacks']),
    }


def train_meta(n, seed, tasks, epochs):
    config = _config(n, 1.0, 2)
    config.use_meta_amg = True
    config.meta_training_sequences = tasks
    config.meta_training_epochs = epochs
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return PicardStokesSolver(config).meta_amg
    finally:
        np.random.set_state(state)


def run_case(n, contrast, seed, picard_iterations, trained_meta):
    mesh, records, trajectory = build_shared_trajectory(
        n, contrast, seed, picard_iterations)
    direct, references = replay_direct(records)
    config = _config(n, contrast, picard_iterations)
    methods = [
        direct,
        replay_blocked(mesh, records, config, 'traditional_amg', references),
        replay_blocked(mesh, records, config, 'meta_amg', references, trained_meta),
    ]
    for method in methods:
        method.update({'seed': seed, 'viscosity_contrast': contrast, **trajectory})
    return methods


def aggregate(records):
    groups = {}
    for record in records:
        groups.setdefault((record['method'], record['viscosity_contrast']), []).append(record)
    metrics = (
        'setup_time_s', 'solve_time_s', 'total_time_s', 'linear_solver_calls',
        'velocity_krylov_iterations', 'velocity_solve_calls',
        'max_linear_relative_residual', 'max_solution_relative_error_vs_direct',
        'python_peak_memory_bytes', 'velocity_fallbacks', 'pressure_fallbacks',
        'full_fallbacks', 'nodal_viscosity_contrast', 'element_viscosity_contrast',
        'trajectory_sparse_bytes',
    )
    output = []
    for (method, contrast), items in sorted(groups.items()):
        row = {'method': method, 'viscosity_contrast': contrast, 'n_seeds': len(items)}
        row.update({metric: _mean_std([item[metric] for item in items]) for metric in metrics})
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--contrasts', default='1,1e2,1e4,1e6')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--picard-iterations', type=int, default=3)
    parser.add_argument('--meta-tasks', type=int, default=2)
    parser.add_argument('--meta-epochs', type=int, default=1)
    parser.add_argument('--output', default='experiments/results_stokes_3d_methods')
    args = parser.parse_args()
    contrasts = [float(value) for value in args.contrasts.split(',') if value]
    seeds = [int(value) for value in args.seeds.split(',') if value]
    trained = {seed: train_meta(args.n, seed, args.meta_tasks, args.meta_epochs)
               for seed in seeds}
    records = []
    for contrast in contrasts:
        for seed in seeds:
            records.extend(run_case(args.n, contrast, seed, args.picard_iterations,
                                    trained[seed]))
    payload = {
        'benchmark': 'Shared-trajectory 3D variable-viscosity Stokes replay',
        'scope_note': 'All methods solve identical preassembled A_k x_k=b_k systems; trajectory generation is excluded from method timings.',
        'config': vars(args),
        'timing_note': 'Direct setup is sparse LU factorization; blocked setup is hierarchy/adaptation/Schur factorization; solve excludes those setup components.',
        'memory_note': 'Python peak excludes some native SciPy/SuperLU allocations; trajectory CSR bytes are reported separately.',
        'per_run': records, 'aggregate': aggregate(records),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / 'stokes_3d_methods.json', 'w') as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload['aggregate'], indent=2))


if __name__ == '__main__':
    main()
