"""Independent-process 2D/3D scaling for external AMG block solves."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.benchmark_stokes_3d_methods import run_backend_case
from experiments.stokes_baseline_common import (
    BACKENDS,
    METHODS,
    deterministic_method_order,
    environment_metadata,
    paired_seed_summary_stats,
    peak_rss_bytes,
    prepare_output_directory,
    probe_backend_in_subprocess,
    seeded_summary_stats,
)


def run_worker(n, contrast, seed, picard_iterations, method, dimension=3,
               mesh_mode='structured', viscosity_mode='temperature_dependent',
               unstructured_points=None, repeat=0,
               schur_velocity_inverse='lu', schur_velocity_vcycles=2,
               velocity_solver_rtol=1e-6, pressure_solver_rtol=1e-6):
    result = run_backend_case(
        n, contrast, seed, picard_iterations, method,
        dimension=dimension,
        mesh_mode=mesh_mode,
        viscosity_mode=viscosity_mode,
        unstructured_points=unstructured_points,
        schur_velocity_inverse=schur_velocity_inverse,
        schur_velocity_vcycles=schur_velocity_vcycles,
        velocity_solver_rtol=velocity_solver_rtol,
        pressure_solver_rtol=pressure_solver_rtol,
    )
    result['peak_rss_bytes'] = peak_rss_bytes()
    result['repeat'] = repeat
    return result


def aggregate(records, failures=(), paired_baseline=None):
    groups = {}
    for record in records:
        if record.get('status') != 'success':
            continue
        groups.setdefault((
            record['dimension'], record['mesh_mode'], record['viscosity_mode'],
            record['n'], record['method']), []).append(record)
    metrics = ('setup_time_s', 'solve_time_s', 'total_time_s', 'peak_rss_bytes',
               'method_rss_baseline_bytes', 'method_peak_rss_bytes',
               'method_peak_rss_delta_bytes',
               'velocity_krylov_iterations', 'max_linear_relative_residual',
               'pressure_krylov_iterations',
               'max_solution_relative_error_vs_direct', 'velocity_fallbacks',
               'pressure_fallbacks', 'full_fallbacks', 'hierarchy_setups',
               'hierarchy_updates', 'hierarchy_rebuilds', 'hierarchy_reuses',
               'backend_setup_failures',
               'preconditioner_apply_calls', 'preconditioner_apply_time_s')
    rows = []
    for (dimension, mesh_mode, viscosity_mode, n, method), items in sorted(groups.items()):
        accepted_items = [item for item in items if item.get('quality_accepted', False)]
        backend, policy = BACKENDS.get(method, ('direct', 'direct'))
        row = {'dimension': dimension, 'mesh_mode': mesh_mode,
               'viscosity_mode': viscosity_mode, 'n': n, 'method': method,
               'backend': backend, 'hierarchy_policy': policy,
               'schur_velocity_inverse': items[0].get('schur_velocity_inverse'),
               'schur_velocity_vcycles': items[0].get('schur_velocity_vcycles'),
               'n_samples': len(items),
               'n_seeds': len({item['seed'] for item in items}),
               'n_repetitions': len({item.get('repeat', 0) for item in items}),
               'velocity_dofs': items[0]['velocity_dofs'],
               'stokes_dofs': items[0]['stokes_dofs']}
        row.update({metric: seeded_summary_stats(accepted_items, metric)
                    for metric in metrics})
        key = (dimension, mesh_mode, viscosity_mode, n, method)
        n_failures = sum(
            (failure['dimension'], failure['mesh_mode'], failure['viscosity_mode'],
             failure['n'], failure['method']) == key
            for failure in failures)
        n_unavailable = sum(
            record.get('status') == 'unavailable' and
            (record['dimension'], record['mesh_mode'], record['viscosity_mode'],
             record['n'], record['method']) == key
            for record in records)
        row['n_executed'] = len(items)
        row['n_success'] = len(accepted_items)
        row['n_failures'] = n_failures
        row['n_unavailable'] = n_unavailable
        row['n_attempts'] = len(items) + n_failures + n_unavailable
        row['failure_rate'] = float(n_failures / max(row['n_attempts'], 1))
        row['unavailable_rate'] = float(n_unavailable / max(row['n_attempts'], 1))
        row['quality_accept_rate'] = float(np.mean([
            item.get('quality_accepted', False) for item in items]))
        row['numerical_failure_rate'] = 1.0 - row['quality_accept_rate']
        if paired_baseline is not None:
            reference_items = [
                item for item in groups.get(
                    (dimension, mesh_mode, viscosity_mode, n, paired_baseline), [])
                if item.get('quality_accepted', False)
            ]
            row['paired_vs_method'] = paired_baseline
            row['paired_metrics'] = {
                metric: paired_seed_summary_stats(
                    accepted_items, reference_items, metric)
                for metric in ('setup_time_s', 'solve_time_s', 'total_time_s')
            }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', default='8,10,12,16')
    parser.add_argument('--dimensions', default='3')
    parser.add_argument('--contrast', type=float, default=1e4)
    parser.add_argument('--mesh-modes', default='structured')
    parser.add_argument('--viscosity-modes', default='temperature_dependent')
    parser.add_argument('--unstructured-points', type=int, default=None)
    parser.add_argument('--schur-velocity-inverse',
                        choices=['vcycle', 'krylov', 'ilu', 'lu'], default='lu')
    parser.add_argument('--schur-velocity-vcycles', type=int, default=2)
    parser.add_argument('--velocity-solver-rtol', type=float, default=1e-6)
    parser.add_argument('--pressure-solver-rtol', type=float, default=1e-6)
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--picard-iterations', type=int, default=3)
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--paired-baseline', choices=METHODS)
    parser.add_argument('--method-order', choices=['fixed', 'shuffled'], default='shuffled')
    parser.add_argument('--fresh-output', action='store_true')
    parser.add_argument('--require-hypre', action='store_true')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--output', default='experiments/results_stokes_3d_scalable')
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
        result = run_worker(args.n, args.contrast, args.seed,
                            args.picard_iterations, args.method, args.dimension,
                            args.mesh_mode, args.viscosity_mode,
                            args.unstructured_points, args.repeat,
                            args.schur_velocity_inverse, args.schur_velocity_vcycles,
                            args.velocity_solver_rtol, args.pressure_solver_rtol)
        with open(args.worker_output, 'w') as handle:
            json.dump(result, handle, indent=2)
        return

    levels = [int(value) for value in args.levels.split(',') if value]
    dimensions = [int(value) for value in args.dimensions.split(',') if value]
    mesh_modes = [value for value in args.mesh_modes.split(',') if value]
    viscosity_modes = [value for value in args.viscosity_modes.split(',') if value]
    seeds = [int(value) for value in args.seeds.split(',') if value]
    methods = [value for value in args.methods.split(',') if value]
    paired_baseline = args.paired_baseline or methods[0]
    if paired_baseline not in methods:
        parser.error('--paired-baseline must be included in --methods')
    if args.require_hypre:
        available, reason = probe_backend_in_subprocess('hypre_boomeramg')
        if not available:
            parser.error(f'HYPRE is required for this formal run: {reason}')
    records, failures = [], []
    output = prepare_output_directory(args.output, fresh=args.fresh_output)
    for dimension in dimensions:
        for mesh_mode in mesh_modes:
            for viscosity_mode in viscosity_modes:
                for n in levels:
                    for seed in seeds:
                        for repeat in range(args.repetitions):
                            ordered_methods = (
                                deterministic_method_order(
                                    methods, dimension, mesh_mode, viscosity_mode,
                                    n, seed, repeat)
                                if args.method_order == 'shuffled' else methods
                            )
                            for method in ordered_methods:
                                run_id = (
                                    f'{dimension}d-{mesh_mode}-{viscosity_mode}-n{n}-'
                                    f'c{args.contrast:g}-p{args.picard_iterations}-'
                                    f'u{args.unstructured_points or 0}-'
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
                                        'method': method,
                                        'error': f'{type(error).__name__}: {error}',
                                    })
                                finally:
                                    Path(worker_output).unlink(missing_ok=True)
    payload = {'benchmark': 'Independent-process scalable blocked Stokes replay',
               'config': vars(args), 'environment': environment_metadata(),
               'thread_policy': 'single-threaded BLAS/OpenMP workers', 'per_run': records,
               'protocol': {
                   'method_order': args.method_order,
                   'paired_baseline': paired_baseline,
                   'fresh_output': args.fresh_output,
                   'require_hypre': args.require_hypre,
               },
               'aggregate': aggregate(records, failures, paired_baseline),
               'failures': failures}
    with open(output / 'stokes_3d_scalable.json', 'w') as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({'aggregate': payload['aggregate'], 'failures': failures}, indent=2))


if __name__ == '__main__':
    main()
