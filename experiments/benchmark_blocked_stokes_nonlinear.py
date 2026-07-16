"""Full nonlinear blocked Stokes baseline benchmark with amortized metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.sparse.linalg import lsmr, spsolve

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stokes_solver import PicardStokesSolver, probe_velocity_amg_backend
from experiments.stokes_baseline_common import (
    BACKENDS,
    METHODS,
    NativeRSSMonitor,
    build_case_config,
    build_case_mesh,
    deterministic_method_order,
    environment_metadata,
    paired_seed_summary_stats,
    peak_rss_bytes,
    prepare_output_directory,
    probe_backend_in_subprocess,
    seeded_summary_stats,
)


def _rms_velocity(solver: PicardStokesSolver) -> float:
    ndpn = solver.mesh.n_dofs_per_node
    squared = np.zeros(solver.mesh.n_nodes)
    for component in range(solver.mesh.dim):
        squared += solver.velocity[component::ndpn] ** 2
    return float(np.sqrt(np.mean(squared)))


def _blocked_setup_time(block_stats):
    return float(
        block_stats.get('traditional_hierarchy_setup_time', 0.0) +
        block_stats.get('adapted_hierarchy_setup_time', 0.0) +
        block_stats.get('meta_adaptation_time', 0.0) +
        block_stats.get('reused_hierarchy_update_time', 0.0) +
        block_stats.get('schur_velocity_lu_setup_time', 0.0) +
        block_stats.get('schur_velocity_ilu_setup_time', 0.0)
    )


def run_simulation(n, contrast, seed, method, n_steps, picard_iterations,
                   dimension=3, mesh_mode='structured',
                   viscosity_mode='temperature_dependent',
                   unstructured_points=None, repeat=0,
                   schur_velocity_inverse='lu', schur_velocity_vcycles=2,
                   velocity_solver_rtol=1e-6, pressure_solver_rtol=1e-6):
    if method != 'direct':
        backend, _ = BACKENDS[method]
        available, reason = probe_velocity_amg_backend(backend)
        if not available:
            return {
                'status': 'unavailable',
                'method': method,
                'backend': backend,
                'unavailable_reason': reason,
                'dimension': dimension,
                'mesh_mode': mesh_mode,
                'viscosity_mode': viscosity_mode,
                'n': n,
                'seed': seed,
                'repeat': repeat,
                'n_steps': n_steps,
            }

    mesh, mesh_stats = build_case_mesh(
        n, dimension=dimension, mesh_mode=mesh_mode,
        seed=seed, unstructured_points=unstructured_points)
    if method == 'direct':
        backend = 'internal'
        policy = 'fresh'
    else:
        backend, policy = BACKENDS[method]
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
    solver = PicardStokesSolver(config, mesh=mesh)
    direct_fallbacks = {'count': 0}
    if method == 'direct':
        def direct_solve(A, b):
            try:
                x = spsolve(A, b)
                if np.isfinite(x).all():
                    return np.asarray(x)
            except Exception:
                pass
            direct_fallbacks['count'] += 1
            return np.asarray(lsmr(A, b, atol=1e-10, btol=1e-10)[0])
        solver._solve_with_amg = direct_solve
        solver._solve_with_meta_amg = direct_solve

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        solver.initialize_temperature()
    finally:
        np.random.set_state(state)

    steps = []
    cumulative_block_stats = {}
    list_stats = {'velocity_backend_errors', 'velocity_hierarchy_decisions'}
    memory = NativeRSSMonitor()
    wall_start = time.perf_counter()
    with memory:
        for step_index in range(n_steps):
            stats = solver.solve_timestep()
            step_block_stats = stats.get('block_stats', {})
            for name, value in step_block_stats.items():
                if name in list_stats:
                    cumulative_block_stats.setdefault(name, []).extend(value)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    cumulative_block_stats[name] = cumulative_block_stats.get(name, 0) + value
                elif name not in cumulative_block_stats:
                    cumulative_block_stats[name] = value
            steps.append({
                'step': step_index,
                'time': float(stats['time']),
                'picard_iterations': int(stats['n_iterations']),
                'assembly_time_s': float(stats['assembly_time_s']),
                'linear_solve_time_s': float(stats['linear_solve_time_s']),
                'nonlinear_update_time_s': float(stats['nonlinear_update_time_s']),
                'linear_relative_residual': float(stats['linear_relative_residual']),
                'nusselt': float(stats['nusselt']),
                'rms_velocity': _rms_velocity(solver),
            })
    wall_time = time.perf_counter() - wall_start

    block_stats = cumulative_block_stats
    total_picard = int(sum(step['picard_iterations'] for step in steps))
    total_assembly = float(sum(step['assembly_time_s'] for step in steps))
    total_linear = float(sum(step['linear_solve_time_s'] for step in steps))
    total_update = float(sum(step['nonlinear_update_time_s'] for step in steps))
    setup_time = _blocked_setup_time(block_stats)
    velocity_solves = int(block_stats.get('velocity_solve_calls', 0))
    max_residual = float(max(step['linear_relative_residual'] for step in steps)) if steps else 0.0
    quality_accepted = bool(
        np.isfinite(solver.velocity).all() and
        np.isfinite(solver.temperature).all() and
        max_residual < 1e-6 and
        block_stats.get('velocity_direct_fallbacks', 0) == 0 and
        block_stats.get('pressure_solver_fallbacks', 0) == 0 and
        block_stats.get('full_direct_fallbacks', 0) == 0 and
        block_stats.get('velocity_backend_setup_failures', 0) == 0
        and direct_fallbacks['count'] == 0
    )
    return {
        'status': 'success',
        'method': method,
        'backend': backend,
        'hierarchy_policy': policy,
        'schur_velocity_inverse': schur_velocity_inverse,
        'schur_velocity_vcycles': schur_velocity_vcycles,
        'velocity_solver_rtol': velocity_solver_rtol,
        'pressure_solver_rtol': pressure_solver_rtol,
        'dimension': dimension,
        'mesh_mode': mesh_mode,
        'viscosity_mode': viscosity_mode,
        'n': n,
        'seed': seed,
        'repeat': repeat,
        'n_steps': n_steps,
        'picard_iterations_per_step_limit': picard_iterations,
        'viscosity_contrast': float(config.viscosity_contrast),
        'peak_rss_bytes': peak_rss_bytes(),
        **memory.metrics(),
        'wall_time_s': float(wall_time),
        'total_assembly_time_s': total_assembly,
        'total_linear_solve_time_s': total_linear,
        'total_nonlinear_update_time_s': total_update,
        'total_block_setup_time_s': setup_time,
        'total_picard_iterations': total_picard,
        'velocity_solve_calls': velocity_solves,
        'avg_picard_iterations_per_step': float(total_picard / max(n_steps, 1)),
        'amortized_wall_time_per_step_s': float(wall_time / max(n_steps, 1)),
        'amortized_linear_solve_time_per_step_s': float(total_linear / max(n_steps, 1)),
        'amortized_setup_time_per_step_s': float(setup_time / max(n_steps, 1)),
        'amortized_setup_time_per_velocity_solve_s': float(setup_time / max(velocity_solves, 1)),
        'amortized_linear_solve_time_per_picard_s': float(total_linear / max(total_picard, 1)),
        'max_linear_relative_residual': max_residual,
        'final_nusselt': float(steps[-1]['nusselt']) if steps else 0.0,
        'final_rms_velocity': float(steps[-1]['rms_velocity']) if steps else 0.0,
        'final_temperature_min': float(np.min(solver.temperature)),
        'final_temperature_max': float(np.max(solver.temperature)),
        'quality_accepted': quality_accepted,
        'direct_reference_fallbacks': direct_fallbacks['count'],
        'block_stats': block_stats,
        'step_metrics': steps,
        'final_temperature_field': solver.temperature.tolist(),
        'final_solution_field': solver.velocity.tolist(),
        **mesh_stats,
    }


def _relative_field_error(test, reference):
    test = np.asarray(test, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return float(np.linalg.norm(test - reference) / max(np.linalg.norm(reference), 1e-14))


def add_physics_equivalence(records, nusselt_rtol=0.02, rms_velocity_rtol=0.05,
                            temperature_field_rtol=0.02,
                            velocity_field_rtol=0.10):
    references = {}
    for record in records:
        if record.get('status') != 'success' or record.get('method') != 'direct':
            continue
        key = (
            record['dimension'], record['mesh_mode'], record['viscosity_mode'],
            record['n'], record['seed'], record['repeat'], record['n_steps'],
            record['viscosity_contrast'], record['picard_iterations_per_step_limit'],
            record.get('unstructured_points'), record['schur_velocity_inverse'],
            record['schur_velocity_vcycles'], record['velocity_solver_rtol'],
            record['pressure_solver_rtol'])
        if record.get('quality_accepted', False):
            references[key] = record

    for record in records:
        if record.get('status') != 'success':
            continue
        key = (
            record['dimension'], record['mesh_mode'], record['viscosity_mode'],
            record['n'], record['seed'], record['repeat'], record['n_steps'],
            record['viscosity_contrast'], record['picard_iterations_per_step_limit'],
            record.get('unstructured_points'), record['schur_velocity_inverse'],
            record['schur_velocity_vcycles'], record['velocity_solver_rtol'],
            record['pressure_solver_rtol'])
        reference = references.get(key)
        if reference is None:
            record['physics_equivalent'] = False
            record['reference_status'] = 'missing'
            continue
        record['reference_status'] = 'available'
        record['nusselt_relative_error_vs_direct'] = float(
            abs(record['final_nusselt'] - reference['final_nusselt']) /
            max(abs(reference['final_nusselt']), 1e-14))
        record['rms_velocity_relative_error_vs_direct'] = float(
            abs(record['final_rms_velocity'] - reference['final_rms_velocity']) /
            max(abs(reference['final_rms_velocity']), 1e-14))
        record['temperature_field_relative_error_vs_direct'] = _relative_field_error(
            record['final_temperature_field'], reference['final_temperature_field'])
        ndpn = record['dimension'] + 2
        velocity_indices = [
            node * ndpn + component
            for node in range(record['n_nodes'])
            for component in range(record['dimension'])
        ]
        test_solution = np.asarray(record['final_solution_field'])
        reference_solution = np.asarray(reference['final_solution_field'])
        record['velocity_field_relative_error_vs_direct'] = _relative_field_error(
            test_solution[velocity_indices], reference_solution[velocity_indices])
        record['physics_equivalent'] = bool(
            record['quality_accepted'] and
            record['nusselt_relative_error_vs_direct'] <= nusselt_rtol and
            record['rms_velocity_relative_error_vs_direct'] <= rms_velocity_rtol and
            record['temperature_field_relative_error_vs_direct'] <= temperature_field_rtol and
            record['velocity_field_relative_error_vs_direct'] <= velocity_field_rtol)
    return records


def aggregate(records, failures=(), paired_baseline=None):
    groups = {}
    for record in records:
        if record.get('status') != 'success':
            continue
        groups.setdefault((
            record['dimension'], record['mesh_mode'], record['viscosity_mode'],
            record['n'], record['n_steps'], record['method']), []).append(record)
    metrics = (
        'wall_time_s', 'total_assembly_time_s', 'total_linear_solve_time_s',
        'total_nonlinear_update_time_s', 'total_block_setup_time_s',
        'total_picard_iterations', 'velocity_solve_calls',
        'avg_picard_iterations_per_step', 'amortized_wall_time_per_step_s',
        'amortized_linear_solve_time_per_step_s', 'amortized_setup_time_per_step_s',
        'amortized_setup_time_per_velocity_solve_s', 'amortized_linear_solve_time_per_picard_s',
        'peak_rss_bytes', 'max_linear_relative_residual', 'final_nusselt', 'final_rms_velocity',
        'method_rss_baseline_bytes', 'method_peak_rss_bytes',
        'method_peak_rss_delta_bytes',
        'nusselt_relative_error_vs_direct', 'rms_velocity_relative_error_vs_direct',
        'temperature_field_relative_error_vs_direct', 'velocity_field_relative_error_vs_direct',
    )
    rows = []
    for key, items in sorted(groups.items()):
        dimension, mesh_mode, viscosity_mode, n, n_steps, method = key
        accepted_items = [item for item in items if item.get('quality_accepted', False)]
        backend, policy = BACKENDS.get(method, ('direct', 'direct'))
        row = {
            'dimension': dimension,
            'mesh_mode': mesh_mode,
            'viscosity_mode': viscosity_mode,
            'n': n,
            'n_steps': n_steps,
            'method': method,
            'backend': backend,
            'hierarchy_policy': policy,
            'schur_velocity_inverse': items[0].get('schur_velocity_inverse'),
            'schur_velocity_vcycles': items[0].get('schur_velocity_vcycles'),
            'n_samples': len(items),
            'n_seeds': len({item['seed'] for item in items}),
            'n_repetitions': len({item['repeat'] for item in items}),
            'quality_accept_rate': float(np.mean([item['quality_accepted'] for item in items])),
            'physics_equivalence_rate': float(np.mean([
                item.get('physics_equivalent', False) for item in items])),
            'stokes_dofs': items[0]['stokes_dofs'],
            'n_elements': items[0]['n_elements'],
            'n_executed': len(items),
            'n_success': len(accepted_items),
        }
        row.update({
            metric: seeded_summary_stats(accepted_items, metric)
            for metric in metrics
        })
        if paired_baseline is not None:
            reference_items = [
                item for item in groups.get(
                    (dimension, mesh_mode, viscosity_mode, n, n_steps,
                     paired_baseline), [])
                if item.get('quality_accepted', False)
            ]
            row['paired_vs_method'] = paired_baseline
            row['paired_metrics'] = {
                metric: paired_seed_summary_stats(
                    accepted_items, reference_items, metric)
                for metric in (
                    'wall_time_s', 'amortized_wall_time_per_step_s',
                    'amortized_linear_solve_time_per_picard_s')
            }
        rows.append(row)
    failure_groups = {}
    for failure in failures:
        key = (
            failure['dimension'], failure['mesh_mode'], failure['viscosity_mode'],
            failure['n'], failure['n_steps'], failure['method'])
        failure_groups[key] = failure_groups.get(key, 0) + 1
    for row in rows:
        key = (row['dimension'], row['mesh_mode'], row['viscosity_mode'],
               row['n'], row['n_steps'], row['method'])
        n_failures = failure_groups.get(key, 0)
        n_unavailable = sum(
            record.get('status') == 'unavailable' and
            (record['dimension'], record['mesh_mode'], record['viscosity_mode'],
             record['n'], record['n_steps'], record['method']) == key
            for record in records)
        row['n_failures'] = n_failures
        row['n_unavailable'] = n_unavailable
        row['n_attempts'] = row['n_executed'] + n_failures + n_unavailable
        row['failure_rate'] = float(n_failures / max(row['n_attempts'], 1))
        row['unavailable_rate'] = float(n_unavailable / max(row['n_attempts'], 1))
        row['numerical_failure_rate'] = 1.0 - row['quality_accept_rate']
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', default='4')
    parser.add_argument('--dimensions', default='2,3')
    parser.add_argument('--mesh-modes', default='structured')
    parser.add_argument('--viscosity-modes', default='temperature_dependent')
    parser.add_argument('--contrast', type=float, default=1e4)
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--picard-iterations', type=int, default=3)
    parser.add_argument('--methods', default='internal_fresh,pyamg_fresh,petsc_gamg_fresh,hypre_boomeramg_fresh')
    parser.add_argument('--paired-baseline', choices=METHODS, default='direct')
    parser.add_argument('--method-order', choices=['fixed', 'shuffled'], default='shuffled')
    parser.add_argument('--fresh-output', action='store_true')
    parser.add_argument('--require-hypre', action='store_true')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--unstructured-points', type=int, default=None)
    parser.add_argument('--schur-velocity-inverse',
                        choices=['vcycle', 'krylov', 'ilu', 'lu'], default='lu')
    parser.add_argument('--schur-velocity-vcycles', type=int, default=2)
    parser.add_argument('--velocity-solver-rtol', type=float, default=1e-6)
    parser.add_argument('--pressure-solver-rtol', type=float, default=1e-6)
    parser.add_argument('--output', default='experiments/results_blocked_stokes_nonlinear')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--n', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--dimension', type=int, choices=[2, 3], default=3)
    parser.add_argument('--mesh-mode', choices=['structured', 'unstructured'], default='structured')
    parser.add_argument('--viscosity-mode',
                        choices=['isoviscous', 'temperature_dependent', 'temperature_strain_rate'],
                        default='temperature_dependent')
    parser.add_argument('--worker-output')
    args = parser.parse_args()

    if args.worker:
        result = run_simulation(
            args.n, args.contrast, args.seed, args.method, args.steps,
            args.picard_iterations, dimension=args.dimension,
            mesh_mode=args.mesh_mode, viscosity_mode=args.viscosity_mode,
            unstructured_points=args.unstructured_points,
            repeat=args.repeat,
            schur_velocity_inverse=args.schur_velocity_inverse,
            schur_velocity_vcycles=args.schur_velocity_vcycles,
            velocity_solver_rtol=args.velocity_solver_rtol,
            pressure_solver_rtol=args.pressure_solver_rtol)
        with open(args.worker_output, 'w') as handle:
            json.dump(result, handle, indent=2)
        return

    levels = [int(value) for value in args.levels.split(',') if value]
    dimensions = [int(value) for value in args.dimensions.split(',') if value]
    mesh_modes = [value for value in args.mesh_modes.split(',') if value]
    viscosity_modes = [value for value in args.viscosity_modes.split(',') if value]
    seeds = [int(value) for value in args.seeds.split(',') if value]
    methods = [value for value in args.methods.split(',') if value]
    if 'direct' not in methods:
        methods.insert(0, 'direct')
    if args.paired_baseline not in methods:
        parser.error('--paired-baseline must be included in --methods')
    if args.require_hypre:
        available, reason = probe_backend_in_subprocess('hypre_boomeramg')
        if not available:
            parser.error(f'HYPRE is required for this formal run: {reason}')
    output = prepare_output_directory(args.output, fresh=args.fresh_output)
    records, failures = [], []

    for dimension in dimensions:
        for mesh_mode in mesh_modes:
            for viscosity_mode in viscosity_modes:
                for n in levels:
                    for seed in seeds:
                        for repeat in range(args.repetitions):
                            ordered_methods = (
                                deterministic_method_order(
                                    methods, dimension, mesh_mode, viscosity_mode,
                                    n, seed, repeat, args.steps)
                                if args.method_order == 'shuffled' else methods
                            )
                            for method in ordered_methods:
                                run_id = (
                                    f'{dimension}d-{mesh_mode}-{viscosity_mode}-'
                                    f'n{n}-c{args.contrast:g}-steps{args.steps}-'
                                    f'p{args.picard_iterations}-u{args.unstructured_points or 0}-'
                                    f's{args.schur_velocity_inverse}{args.schur_velocity_vcycles}-'
                                    f'vr{args.velocity_solver_rtol:g}-'
                                    f'pr{args.pressure_solver_rtol:g}-'
                                    f'seed{seed}-r{repeat}-{method}')
                                result_path = output / f'{run_id}.json'
                                if result_path.exists():
                                    with open(result_path) as handle:
                                        existing = json.load(handle)
                                    if existing.get('status') in {'success', 'unavailable'}:
                                        records.append(existing)
                                        continue
                                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as handle:
                                    worker_output = handle.name
                                command = [
                                    sys.executable, str(Path(__file__).resolve()), '--worker',
                                    '--dimension', str(dimension), '--mesh-mode', mesh_mode,
                                    '--viscosity-mode', viscosity_mode,
                                    '--n', str(n), '--seed', str(seed), '--method', method,
                                    '--repeat', str(repeat),
                                    '--steps', str(args.steps),
                                    '--contrast', str(args.contrast),
                                    '--picard-iterations', str(args.picard_iterations),
                                    '--schur-velocity-inverse', args.schur_velocity_inverse,
                                    '--schur-velocity-vcycles', str(args.schur_velocity_vcycles),
                                    '--velocity-solver-rtol', str(args.velocity_solver_rtol),
                                    '--pressure-solver-rtol', str(args.pressure_solver_rtol),
                                    '--worker-output', worker_output,
                                ]
                                if args.unstructured_points is not None:
                                    command.extend(['--unstructured-points', str(args.unstructured_points)])
                                start = time.perf_counter()
                                try:
                                    child_env = {
                                        **os.environ,
                                        'OMP_NUM_THREADS': '1',
                                        'OPENBLAS_NUM_THREADS': '1',
                                        'MKL_NUM_THREADS': '1',
                                        'VECLIB_MAXIMUM_THREADS': '1',
                                    }
                                    subprocess.run(
                                        command, timeout=args.timeout, check=True,
                                        capture_output=True, text=True, env=child_env)
                                    with open(worker_output) as handle:
                                        result = json.load(handle)
                                    result['subprocess_wall_time_s'] = time.perf_counter() - start
                                    with open(result_path, 'w') as handle:
                                        json.dump(result, handle, indent=2)
                                    records.append(result)
                                except Exception as error:
                                    failures.append({
                                        'dimension': dimension,
                                        'mesh_mode': mesh_mode,
                                        'viscosity_mode': viscosity_mode,
                                        'n': n,
                                        'seed': seed,
                                        'repeat': repeat,
                                        'n_steps': args.steps,
                                        'method': method,
                                        'error': f'{type(error).__name__}: {error}',
                                    })
                                finally:
                                    Path(worker_output).unlink(missing_ok=True)

    add_physics_equivalence(records)
    payload = {
        'benchmark': 'Full nonlinear blocked Stokes baseline benchmark',
        'config': vars(args),
        'environment': environment_metadata(),
        'thread_policy': 'single-threaded BLAS/OpenMP workers',
        'protocol': {
            'method_order': args.method_order,
            'paired_baseline': args.paired_baseline,
            'fresh_output': args.fresh_output,
            'require_hypre': args.require_hypre,
        },
        'per_run': records,
        'aggregate': aggregate(records, failures, args.paired_baseline),
        'failures': failures,
    }
    with open(output / 'blocked_stokes_nonlinear.json', 'w') as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({'aggregate': payload['aggregate'], 'failures': failures}, indent=2))


if __name__ == '__main__':
    main()
