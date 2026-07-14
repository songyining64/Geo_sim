"""Scale the shared-trajectory 3D Stokes replay across mesh levels."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.benchmark_stokes_3d_methods import run_case, train_meta


def _mean_std(values):
    return {'mean': float(np.mean(values)), 'std': float(np.std(values))}


def aggregate(records):
    groups = {}
    for record in records:
        groups.setdefault((record['n'], record['method']), []).append(record)
    metrics = (
        'setup_time_s', 'solve_time_s', 'total_time_s',
        'velocity_krylov_iterations', 'python_peak_memory_bytes',
        'max_linear_relative_residual', 'max_solution_relative_error_vs_direct',
        'velocity_fallbacks', 'pressure_fallbacks', 'full_fallbacks',
        'trajectory_generation_time_s', 'trajectory_sparse_bytes',
        'element_viscosity_contrast',
    )
    output = []
    for (n, method), items in sorted(groups.items()):
        row = {
            'n': n, 'method': method, 'n_seeds': len(items),
            'n_nodes': items[0]['n_nodes'], 'n_tetrahedra': items[0]['n_tetrahedra'],
            'stokes_dofs': items[0]['stokes_dofs'],
        }
        row.update({metric: _mean_std([item[metric] for item in items])
                    for metric in metrics})
        output.append(row)
    return output


def write_tables(output, rows, failures):
    flat = []
    for row in rows:
        flat.append({
            'n': row['n'], 'method': row['method'], 'n_seeds': row['n_seeds'],
            'stokes_dofs': row['stokes_dofs'], 'n_tetrahedra': row['n_tetrahedra'],
            'setup_s_mean': row['setup_time_s']['mean'],
            'setup_s_std': row['setup_time_s']['std'],
            'solve_s_mean': row['solve_time_s']['mean'],
            'solve_s_std': row['solve_time_s']['std'],
            'total_s_mean': row['total_time_s']['mean'],
            'total_s_std': row['total_time_s']['std'],
            'krylov_mean': row['velocity_krylov_iterations']['mean'],
            'python_peak_bytes_mean': row['python_peak_memory_bytes']['mean'],
            'max_residual_mean': row['max_linear_relative_residual']['mean'],
            'max_error_vs_direct_mean': row['max_solution_relative_error_vs_direct']['mean'],
            'full_fallbacks_mean': row['full_fallbacks']['mean'],
        })
    with open(output / 'table_stokes_3d_scaling.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = [
        '# 3D Stokes scaling replay', '',
        '| n | seeds | DOFs | method | setup (s) | solve (s) | total (s) | Krylov | peak Python MB | max error vs direct | full fallback |',
        '|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in flat:
        lines.append(
            f"| {row['n']} | {row['n_seeds']} | {row['stokes_dofs']} | {row['method']} | "
            f"{row['setup_s_mean']:.3f} +/- {row['setup_s_std']:.3f} | "
            f"{row['solve_s_mean']:.3f} +/- {row['solve_s_std']:.3f} | "
            f"{row['total_s_mean']:.3f} +/- {row['total_s_std']:.3f} | "
            f"{row['krylov_mean']:.1f} | {row['python_peak_bytes_mean'] / 1e6:.2f} | "
            f"{row['max_error_vs_direct_mean']:.2e} | {row['full_fallbacks_mean']:.1f} |"
        )
    if failures:
        lines.extend(['', '## Failures', ''])
        lines.extend(f"- n={item['n']}, seed={item['seed']}: {item['error']}"
                     for item in failures)
    lines.extend(['', '> All method timings exclude shared trajectory generation and Meta training.',
                  '> Python peak memory excludes native SciPy/SuperLU allocations.'])
    with open(output / 'summary.md', 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', default='4,6,8')
    parser.add_argument('--contrast', type=float, default=1e4)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--picard-iterations', type=int, default=3)
    parser.add_argument('--meta-tasks', type=int, default=2)
    parser.add_argument('--meta-epochs', type=int, default=1)
    parser.add_argument('--output', default='experiments/results_stokes_3d_scaling')
    parser.add_argument('--append', action='store_true')
    args = parser.parse_args()
    levels = [int(value) for value in args.levels.split(',') if value]
    seeds = [int(value) for value in args.seeds.split(',') if value]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'stokes_3d_scaling.json'
    records, failures, training = [], [], []
    if args.append and result_path.exists():
        with open(result_path) as handle:
            previous = json.load(handle)
        records.extend(previous.get('per_run', []))
        failures.extend(previous.get('failures', []))
        training.extend(previous.get('meta_training', []))
    for n in levels:
        for seed in seeds:
            try:
                start = time.perf_counter()
                meta = train_meta(n, seed, args.meta_tasks, args.meta_epochs)
                training.append({'n': n, 'seed': seed,
                                 'training_time_s': time.perf_counter() - start})
                case = run_case(n, args.contrast, seed, args.picard_iterations, meta)
                n_nodes = (n + 1) ** 3
                for record in case:
                    record.update({
                        'n': n, 'n_nodes': n_nodes,
                        'n_tetrahedra': 6 * n ** 3, 'stokes_dofs': 4 * n_nodes,
                    })
                records.extend(case)
            except Exception as error:
                failures.append({'n': n, 'seed': seed,
                                 'error': f'{type(error).__name__}: {error}'})
    payload = {
        'benchmark': '3D Stokes shared-trajectory scaling replay',
        'config': vars(args), 'meta_training': training,
        'per_run': records, 'aggregate': aggregate(records), 'failures': failures,
    }
    payload['tested_levels'] = sorted(set(record['n'] for record in records))
    with open(result_path, 'w') as handle:
        json.dump(payload, handle, indent=2)
    if records:
        write_tables(output, payload['aggregate'], failures)
    print(json.dumps({'aggregate': payload['aggregate'], 'failures': failures}, indent=2))


if __name__ == '__main__':
    main()
