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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', default='paper_medium',
                        choices=['quick', 'paper_medium', 'paper_large'])
    parser.add_argument('--output', default='experiments/results_blocked_comparison')
    args = parser.parse_args()

    config = ExperimentConfig()
    apply_preset(config, args.preset)
    results = []
    for seed in config.seeds:
        print(f'Running seed {seed}...')
        results.append(benchmark_blocked_stokes_methods(config, seed=seed))

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
        },
        'per_seed': results,
        'aggregate': aggregate(results),
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'blocked_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload['aggregate'], indent=2))
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
