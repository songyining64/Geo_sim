"""Independent-process scaling for direct and scalable PyAMG block solves."""

import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stokes_solver import StokesConfig
from experiments.benchmark_stokes_3d_methods import (
    build_shared_trajectory, replay_blocked, replay_direct,
)


METHODS = ('direct', 'pyamg_rebuild', 'pyamg_reuse')


def _peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == 'Darwin' else value * 1024)


def run_worker(n, contrast, seed, picard_iterations, method):
    mesh, records, trajectory = build_shared_trajectory(
        n, contrast, seed, picard_iterations)
    direct, references = replay_direct(records)
    if method == 'direct':
        result = direct
    else:
        config = StokesConfig(
            nx=n, ny=n, nz=n, rayleigh=1e3, viscosity_contrast=contrast,
            max_picard_iterations=picard_iterations, picard_tolerance=0.0,
            pressure_solver='matrix_free_schur', schur_velocity_inverse='krylov',
            velocity_amg_backend='pyamg',
            reuse_velocity_hierarchy=(method == 'pyamg_reuse'),
            velocity_solver_max_iterations=500,
        )
        result = replay_blocked(mesh, records, config, method, references)
        result['quality_accepted'] = bool(
            result['max_linear_relative_residual'] < 1e-6 and
            result['max_solution_relative_error_vs_direct'] < 1e-5 and
            result['full_fallbacks'] == 0)
    if method == 'direct':
        result['quality_accepted'] = True
    result.update({
        'n': n, 'seed': seed, 'method': method, 'stokes_dofs': 4 * (n + 1) ** 3,
        'n_tetrahedra': 6 * n ** 3, 'peak_rss_bytes': _peak_rss_bytes(), **trajectory,
    })
    return result


def _mean_std(values):
    return {'mean': float(np.mean(values)), 'std': float(np.std(values))}


def aggregate(records):
    groups = {}
    for record in records:
        groups.setdefault((record['n'], record['method']), []).append(record)
    metrics = ('setup_time_s', 'solve_time_s', 'total_time_s', 'peak_rss_bytes',
               'velocity_krylov_iterations', 'max_linear_relative_residual',
               'max_solution_relative_error_vs_direct', 'velocity_fallbacks',
               'pressure_fallbacks', 'full_fallbacks')
    rows = []
    for (n, method), items in sorted(groups.items()):
        row = {'n': n, 'method': method, 'n_seeds': len(items),
               'stokes_dofs': items[0]['stokes_dofs']}
        row.update({metric: _mean_std([item[metric] for item in items])
                    for metric in metrics})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', default='8,10,12,16')
    parser.add_argument('--contrast', type=float, default=1e4)
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--picard-iterations', type=int, default=3)
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--output', default='experiments/results_stokes_3d_scalable')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--n', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--worker-output')
    args = parser.parse_args()
    if args.worker:
        result = run_worker(args.n, args.contrast, args.seed,
                            args.picard_iterations, args.method)
        with open(args.worker_output, 'w') as handle:
            json.dump(result, handle, indent=2)
        return

    levels = [int(value) for value in args.levels.split(',') if value]
    seeds = [int(value) for value in args.seeds.split(',') if value]
    methods = [value for value in args.methods.split(',') if value]
    records, failures = [], []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for n in levels:
        for seed in seeds:
            for method in methods:
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as handle:
                    worker_output = handle.name
                command = [
                    sys.executable, str(Path(__file__).resolve()), '--worker',
                    '--n', str(n), '--seed', str(seed), '--method', method,
                    '--contrast', str(args.contrast),
                    '--picard-iterations', str(args.picard_iterations),
                    '--worker-output', worker_output,
                ]
                start = time.perf_counter()
                try:
                    completed = subprocess.run(
                        command, timeout=args.timeout, check=True,
                        capture_output=True, text=True)
                    with open(worker_output) as handle:
                        result = json.load(handle)
                    result['subprocess_wall_time_s'] = time.perf_counter() - start
                    records.append(result)
                except Exception as error:
                    failures.append({'n': n, 'seed': seed, 'method': method,
                                     'error': f'{type(error).__name__}: {error}'})
                finally:
                    Path(worker_output).unlink(missing_ok=True)
    payload = {'benchmark': 'Independent-process scalable 3D Stokes replay',
               'config': vars(args), 'per_run': records,
               'aggregate': aggregate(records), 'failures': failures}
    with open(output / 'stokes_3d_scalable.json', 'w') as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({'aggregate': payload['aggregate'], 'failures': failures}, indent=2))


if __name__ == '__main__':
    main()
