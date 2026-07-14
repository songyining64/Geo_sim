"""Compare fresh and Meta-AMG blocked Stokes solves across fixed seeds."""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_experiments import (
    ExperimentConfig,
    apply_preset,
    benchmark_blocked_stokes_methods,
    benchmark_blocked_stokes_with_trained_meta,
    train_stokes_meta_amg_for_blocked_benchmark,
)


def _mean_std(values: List[float]) -> Dict[str, float]:
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
    }


def aggregate(results: List[Dict]) -> Dict:
    traditional_stats = [item['traditional_blocked']['block_stats'] for item in results]
    meta_stats = [item['meta_blocked']['block_stats'] for item in results]

    traditional_setup = [s['traditional_hierarchy_setup_time'] for s in traditional_stats]
    meta_build = [s['traditional_hierarchy_setup_time'] + s['adapted_hierarchy_setup_time']
                  for s in meta_stats]
    meta_online_setup = [build + s['meta_adaptation_time']
                         for build, s in zip(meta_build, meta_stats)]
    traditional_schur_setup = [
        s.get('schur_velocity_lu_setup_time', 0.0) +
        s.get('schur_velocity_ilu_setup_time', 0.0)
        for s in traditional_stats]
    meta_schur_setup = [
        s.get('schur_velocity_lu_setup_time', 0.0) +
        s.get('schur_velocity_ilu_setup_time', 0.0)
        for s in meta_stats]
    traditional_total_setup = [hierarchy + schur for hierarchy, schur in
                               zip(traditional_setup, traditional_schur_setup)]
    meta_total_setup = [online + schur for online, schur in
                        zip(meta_online_setup, meta_schur_setup)]
    traditional_krylov = [s['velocity_krylov_iterations'] for s in traditional_stats]
    meta_krylov = [s['velocity_krylov_iterations'] for s in meta_stats]
    traditional_krylov_per_solve = [
        s['velocity_krylov_iterations'] / max(s['velocity_solve_calls'], 1)
        for s in traditional_stats]
    adapted_krylov_per_solve = [
        s['adapted_velocity_krylov_iterations'] / max(s['adapted_velocity_solve_calls'], 1)
        for s in meta_stats]

    return {
        'n_seeds': len(results),
        'traditional_setup_time': _mean_std(traditional_setup),
        'meta_hierarchy_build_time': _mean_std(meta_build),
        'meta_online_setup_time': _mean_std(meta_online_setup),
        'meta_adaptation_time': _mean_std([
            s['meta_adaptation_time'] for s in meta_stats]),
        'traditional_schur_velocity_setup_time': _mean_std(traditional_schur_setup),
        'meta_schur_velocity_setup_time': _mean_std(meta_schur_setup),
        'traditional_total_online_preconditioner_setup': _mean_std(traditional_total_setup),
        'meta_total_online_preconditioner_setup': _mean_std(meta_total_setup),
        'total_online_preconditioner_setup_speedup': _mean_std([
            traditional / max(meta, 1e-14)
            for traditional, meta in zip(traditional_total_setup, meta_total_setup)
        ]),
        'hierarchy_build_reduction_fraction': _mean_std([
            1.0 - meta / max(traditional, 1e-14)
            for traditional, meta in zip(traditional_setup, meta_build)
        ]),
        'online_setup_change_fraction': _mean_std([
            meta / max(traditional, 1e-14) - 1.0
            for traditional, meta in zip(traditional_setup, meta_online_setup)
        ]),
        'setup_speedup_traditional_over_meta_online': _mean_std([
            traditional / max(meta, 1e-14)
            for traditional, meta in zip(traditional_setup, meta_online_setup)
        ]),
        'traditional_velocity_krylov_iterations': _mean_std(traditional_krylov),
        'meta_velocity_krylov_iterations': _mean_std(meta_krylov),
        'krylov_ratio_meta_over_traditional': _mean_std([
            meta / max(traditional, 1)
            for meta, traditional in zip(meta_krylov, traditional_krylov)
        ]),
        'traditional_krylov_per_velocity_solve': _mean_std(traditional_krylov_per_solve),
        'adapted_krylov_per_velocity_solve': _mean_std(adapted_krylov_per_solve),
        'wall_time_reduction_fraction': _mean_std([
            1.0 - meta / max(traditional, 1e-14)
            for traditional, meta in zip(
                [item['traditional_blocked']['wall_time'] for item in results],
                [item['meta_blocked']['wall_time'] for item in results])
        ]),
        'traditional_wall_time': _mean_std([
            item['traditional_blocked']['wall_time'] for item in results]),
        'meta_wall_time': _mean_std([
            item['meta_blocked']['wall_time'] for item in results]),
        'velocity_field_relative_error': _mean_std([
            item['meta_blocked']['velocity_field_relative_error'] for item in results]),
        'initial_temperature_difference': _mean_std([
            item['initial_temperature_difference'] for item in results]),
        'traditional_full_direct_fallbacks': _mean_std([
            s['full_direct_fallbacks'] for s in traditional_stats]),
        'meta_full_direct_fallbacks': _mean_std([
            s['full_direct_fallbacks'] for s in meta_stats]),
        'traditional_velocity_direct_fallbacks': _mean_std([
            s['velocity_direct_fallbacks'] for s in traditional_stats]),
        'meta_velocity_direct_fallbacks': _mean_std([
            s['velocity_direct_fallbacks'] for s in meta_stats]),
        'traditional_pressure_krylov_iterations': _mean_std([
            s['pressure_krylov_iterations'] for s in traditional_stats]),
        'meta_pressure_krylov_iterations': _mean_std([
            s['pressure_krylov_iterations'] for s in meta_stats]),
        'traditional_pressure_solver_fallbacks': _mean_std([
            s['pressure_solver_fallbacks'] for s in traditional_stats]),
        'meta_pressure_solver_fallbacks': _mean_std([
            s['pressure_solver_fallbacks'] for s in meta_stats]),
        'traditional_schur_velocity_lu_fallbacks': _mean_std([
            s.get('schur_velocity_lu_fallbacks', 0) for s in traditional_stats]),
        'meta_schur_velocity_lu_fallbacks': _mean_std([
            s.get('schur_velocity_lu_fallbacks', 0) for s in meta_stats]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', default='paper_medium',
                        choices=['quick', 'paper_medium', 'paper_large'])
    parser.add_argument('--output', default='experiments/results_blocked_comparison')
    parser.add_argument('--adapt-steps', default=None,
                        help='Comma-separated adaptation steps. If omitted, uses config default.')
    parser.add_argument('--stokes-size', type=int, default=None,
                        help='Override nx=ny for Stokes full benchmark')
    parser.add_argument('--seeds', default=None,
                        help='Comma-separated seeds overriding preset seeds')
    parser.add_argument('--pressure-solver', default='matrix_free_schur',
                        choices=['matrix_free_schur', 'approx_schur'])
    parser.add_argument('--schur-velocity-inverse', default='krylov',
                        choices=['krylov', 'vcycle', 'ilu', 'lu'])
    parser.add_argument('--meta-training-max-matrix-size', type=int, default=None,
                        help='Cap training matrix size while testing larger target grids')
    parser.add_argument('--velocity-max-iterations', type=int, default=500)
    args = parser.parse_args()

    config = ExperimentConfig()
    apply_preset(config, args.preset)
    if args.stokes_size is not None:
        config.stokes_nx = args.stokes_size
        config.stokes_ny = args.stokes_size
    if args.seeds is not None:
        config.seeds = [int(item) for item in args.seeds.split(',') if item]
    config.pressure_solver = args.pressure_solver
    config.schur_velocity_inverse = args.schur_velocity_inverse
    config.meta_training_max_matrix_size = args.meta_training_max_matrix_size
    config.velocity_solver_max_iterations = args.velocity_max_iterations
    adapt_steps = ([int(item) for item in args.adapt_steps.split(',') if item]
                   if args.adapt_steps else [config.meta_adapt_steps])

    results_by_adapt_steps = {}
    for seed in config.seeds:
        print(f'Training Meta-AMG once for seed {seed}...')
        trained_meta = train_stokes_meta_amg_for_blocked_benchmark(config, seed=seed)
        for steps in adapt_steps:
            print(f'Running seed {seed}, adapt_steps={steps}...')
            run_config = ExperimentConfig()
            apply_preset(run_config, args.preset)
            run_config.stokes_nx = config.stokes_nx
            run_config.stokes_ny = config.stokes_ny
            run_config.stokes_picard_iterations = config.stokes_picard_iterations
            run_config.stokes_rayleigh = config.stokes_rayleigh
            run_config.stokes_viscosity_contrast = config.stokes_viscosity_contrast
            run_config.meta_tasks = config.meta_tasks
            run_config.meta_epochs = config.meta_epochs
            run_config.meta_adapt_steps = steps
            run_config.pressure_solver = config.pressure_solver
            run_config.schur_velocity_inverse = config.schur_velocity_inverse
            run_config.meta_training_max_matrix_size = config.meta_training_max_matrix_size
            run_config.velocity_solver_max_iterations = config.velocity_solver_max_iterations
            result = benchmark_blocked_stokes_with_trained_meta(
                run_config, trained_meta, seed=seed)
            results_by_adapt_steps.setdefault(str(steps), []).append(result)

    aggregate_by_adapt_steps = {
        steps: aggregate(results)
        for steps, results in results_by_adapt_steps.items()
    }
    primary_steps = str(adapt_steps[-1])

    payload = {
        'preset': args.preset,
        'config': {
            'seeds': config.seeds,
            'nx': config.stokes_nx,
            'ny': config.stokes_ny,
            'picard_iterations': config.stokes_picard_iterations,
            'rayleigh': config.stokes_rayleigh,
            'viscosity_contrast': config.stokes_viscosity_contrast,
            'meta_tasks': config.meta_tasks,
            'meta_epochs': config.meta_epochs,
            'adapt_steps': adapt_steps,
            'pressure_solver': config.pressure_solver,
            'schur_velocity_inverse': config.schur_velocity_inverse,
            'meta_training_max_matrix_size': config.meta_training_max_matrix_size,
            'velocity_solver_max_iterations': config.velocity_solver_max_iterations,
        },
        'per_seed_by_adapt_steps': results_by_adapt_steps,
        'aggregate_by_adapt_steps': aggregate_by_adapt_steps,
        'per_seed': results_by_adapt_steps[primary_steps],
        'aggregate': aggregate_by_adapt_steps[primary_steps],
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'blocked_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload['aggregate_by_adapt_steps'], indent=2))
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
