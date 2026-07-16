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

from core.stokes_solver import BlockStokesSolver, PicardStokesSolver, StokesConfig, probe_velocity_amg_backend
from experiments.stokes_baseline_common import (
    BACKENDS, METHODS, NativeRSSMonitor, build_case_config, build_case_mesh,
)


def _sparse_bytes(matrix):
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _mean_std(values):
    return {'mean': float(np.mean(values)), 'std': float(np.std(values))}


def _config(n, contrast, picard_iterations, dimension=3,
            viscosity_mode='temperature_dependent'):
    return build_case_config(
        n, contrast, picard_iterations,
        dimension=dimension,
        viscosity_mode=viscosity_mode,
        backend='internal',
        policy='fresh',
        schur_velocity_inverse='lu',
    )


def build_shared_trajectory(n, contrast, seed, picard_iterations, dimension=3,
                            mesh_mode='structured', viscosity_mode='temperature_dependent',
                            unstructured_points=None):
    mesh, mesh_stats = build_case_mesh(
        n, dimension=dimension, mesh_mode=mesh_mode,
        seed=seed, unstructured_points=unstructured_points)
    solver = PicardStokesSolver(
        _config(n, contrast, picard_iterations, dimension, viscosity_mode),
        mesh=mesh,
    )
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
        **mesh_stats,
        'viscosity_mode': viscosity_mode,
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
                  stats.get('reused_hierarchy_update_time', 0.0) +
                  stats.get('schur_velocity_lu_setup_time', 0.0) +
                  stats.get('schur_velocity_ilu_setup_time', 0.0))
    residual, error = _quality(solutions, records, references)
    return {
        'method': method, 'setup_time_s': float(setup_time),
        'solve_time_s': float(max(total_time - setup_time, 0.0)),
        'total_time_s': float(total_time), 'linear_solver_calls': len(records),
        'velocity_krylov_iterations': int(stats['velocity_krylov_iterations']),
        'velocity_solve_calls': int(stats['velocity_solve_calls']),
        'pressure_krylov_iterations': int(stats['pressure_krylov_iterations']),
        'max_linear_relative_residual': residual,
        'max_solution_relative_error_vs_direct': error,
        'python_peak_memory_bytes': int(peak),
        'velocity_fallbacks': int(stats['velocity_direct_fallbacks']),
        'pressure_fallbacks': int(stats['pressure_solver_fallbacks']),
        'pressure_preconditioner_fallbacks': int(
            stats['pressure_preconditioner_fallbacks']),
        'full_fallbacks': int(stats['full_direct_fallbacks']),
        'hierarchy_setups': int(stats['traditional_hierarchy_setups'] +
                                stats['adapted_hierarchy_setups']),
        'hierarchy_updates': int(stats.get('reused_hierarchy_updates', 0)),
        'hierarchy_rebuilds': int(stats.get('velocity_hierarchy_rebuilds', 0)),
        'hierarchy_reuses': int(stats.get('velocity_hierarchy_reuses', 0)),
        'hierarchy_decisions': stats.get('velocity_hierarchy_decisions', []),
        'backend_setup_failures': int(stats.get('velocity_backend_setup_failures', 0)),
        'preconditioner_apply_calls': int(
            stats.get('velocity_preconditioner_apply_calls', 0)),
        'preconditioner_apply_time_s': float(
            stats.get('velocity_preconditioner_apply_time', 0.0)),
        'backend_errors': stats.get('velocity_backend_errors', []),
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


def run_backend_case(n, contrast, seed, picard_iterations, method,
                     dimension=3, mesh_mode='structured',
                     viscosity_mode='temperature_dependent',
                     unstructured_points=None,
                     schur_velocity_inverse='lu',
                     schur_velocity_vcycles=2,
                     velocity_solver_rtol=1e-6,
                     pressure_solver_rtol=1e-6):
    mesh, records, trajectory = build_shared_trajectory(
        n, contrast, seed, picard_iterations,
        dimension=dimension,
        mesh_mode=mesh_mode,
        viscosity_mode=viscosity_mode,
        unstructured_points=unstructured_points,
    )
    if method == 'direct':
        with NativeRSSMonitor() as memory:
            result, _ = replay_direct(records)
        result.update(memory.metrics())
        result['quality_accepted'] = True
    else:
        references = [np.asarray(record['solution']) for record in records]
        backend, policy = BACKENDS[method]
        available, reason = probe_velocity_amg_backend(backend)
        if not available:
            return {
                'status': 'unavailable',
                'n': n,
                'method': method,
                'backend': backend,
                'unavailable_reason': reason,
                'seed': seed,
                'viscosity_contrast': contrast,
                **trajectory,
            }
        config = build_case_config(
            n, contrast, picard_iterations,
            dimension=dimension,
            viscosity_mode=viscosity_mode,
            backend=backend,
            policy=policy,
            velocity_solver_max_iterations=500,
            velocity_solver_rtol=velocity_solver_rtol,
            pressure_solver_rtol=pressure_solver_rtol,
            schur_velocity_inverse=schur_velocity_inverse,
            schur_velocity_vcycles=schur_velocity_vcycles,
        )
        with NativeRSSMonitor() as memory:
            result = replay_blocked(mesh, records, config, method, references)
        result.update(memory.metrics())
        result['quality_accepted'] = bool(
            result['max_linear_relative_residual'] < 1e-6 and
            result['max_solution_relative_error_vs_direct'] < 1e-5 and
            result['velocity_fallbacks'] == 0 and
            result['pressure_fallbacks'] == 0 and
            result['full_fallbacks'] == 0 and
            result['backend_setup_failures'] == 0)
    result.update({
        'status': 'success',
        'n': n,
        'method': method,
        'schur_velocity_inverse': schur_velocity_inverse,
        'schur_velocity_vcycles': schur_velocity_vcycles,
        'velocity_solver_rtol': velocity_solver_rtol,
        'pressure_solver_rtol': pressure_solver_rtol,
        'seed': seed,
        'viscosity_contrast': contrast,
        **trajectory,
    })
    return result


def aggregate(records):
    groups = {}
    for record in records:
        if record.get('status', 'success') != 'success':
            continue
        groups.setdefault((
            record['dimension'], record['mesh_mode'], record['viscosity_mode'],
            record['method'], record['viscosity_contrast']), []).append(record)
    metrics = (
        'setup_time_s', 'solve_time_s', 'total_time_s', 'linear_solver_calls',
        'velocity_krylov_iterations', 'velocity_solve_calls',
        'max_linear_relative_residual', 'max_solution_relative_error_vs_direct',
        'python_peak_memory_bytes', 'velocity_fallbacks', 'pressure_fallbacks',
        'full_fallbacks', 'nodal_viscosity_contrast', 'element_viscosity_contrast',
        'trajectory_sparse_bytes',
    )
    output = []
    for (dimension, mesh_mode, viscosity_mode, method, contrast), items in sorted(groups.items()):
        row = {
            'dimension': dimension,
            'mesh_mode': mesh_mode,
            'viscosity_mode': viscosity_mode,
            'method': method,
            'viscosity_contrast': contrast,
            'n_seeds': len(items),
        }
        row.update({metric: _mean_std([item[metric] for item in items]) for metric in metrics})
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--dimension', type=int, choices=[2, 3], default=3)
    parser.add_argument('--mesh-mode', choices=['structured', 'unstructured'], default='structured')
    parser.add_argument('--viscosity-mode',
                        choices=['isoviscous', 'temperature_dependent', 'temperature_strain_rate'],
                        default='temperature_dependent')
    parser.add_argument('--unstructured-points', type=int, default=None)
    parser.add_argument('--contrasts', default='1,1e2,1e4,1e6')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--picard-iterations', type=int, default=3)
    parser.add_argument('--methods', default='direct,internal_fresh,pyamg_fresh,petsc_gamg_fresh,hypre_boomeramg_fresh')
    parser.add_argument('--output', default='experiments/results_stokes_3d_methods')
    args = parser.parse_args()
    contrasts = [float(value) for value in args.contrasts.split(',') if value]
    seeds = [int(value) for value in args.seeds.split(',') if value]
    methods = [value for value in args.methods.split(',') if value]
    records = []
    for contrast in contrasts:
        for seed in seeds:
            for method in methods:
                records.append(run_backend_case(
                    args.n, contrast, seed, args.picard_iterations, method,
                    dimension=args.dimension,
                    mesh_mode=args.mesh_mode,
                    viscosity_mode=args.viscosity_mode,
                    unstructured_points=args.unstructured_points,
                ))
    payload = {
        'benchmark': 'Shared-trajectory blocked Stokes baseline replay',
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
