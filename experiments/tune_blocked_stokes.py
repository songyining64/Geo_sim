"""Parameter scan for stable blocked Stokes solves used in the paper."""

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stokes_solver import PicardStokesSolver, StokesConfig


def run_case(nx: int, ny: int, rayleigh: float, contrast: float,
             picard_iterations: int, seed: int,
             schur_scale: float, velocity_solver_rtol: float,
             velocity_acceptance_rtol: float,
             full_blocked_acceptance_rtol: float,
             use_meta_amg: bool = True,
             meta_training_sequences: int = 60,
             meta_training_epochs: int = 20,
             meta_adapt_steps: int = 3):
    cfg = StokesConfig(
        nx=nx,
        ny=ny,
        rayleigh=rayleigh,
        viscosity_contrast=contrast,
        max_picard_iterations=picard_iterations,
        picard_tolerance=0.0,
        use_meta_amg=use_meta_amg,
        meta_training_sequences=meta_training_sequences,
        meta_training_epochs=meta_training_epochs,
        meta_adapt_steps=meta_adapt_steps,
        schur_scale=schur_scale,
        velocity_solver_rtol=velocity_solver_rtol,
        velocity_acceptance_rtol=velocity_acceptance_rtol,
        full_blocked_acceptance_rtol=full_blocked_acceptance_rtol,
    )
    solver = PicardStokesSolver(cfg)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        solver.initialize_temperature()
    finally:
        np.random.set_state(state)

    t0 = time.time()
    stats = solver.solve_picard()
    wall = time.time() - t0
    ndpn = solver.mesh.n_dofs_per_node
    vrms = float(np.sqrt(np.mean(solver.velocity[0::ndpn] ** 2 + solver.velocity[1::ndpn] ** 2)))
    block_stats = solver._block_solver.get_stats() if hasattr(solver, '_block_solver') else {}
    meta_stats = solver.meta_amg.get_stats() if solver.meta_amg is not None else {}
    return {
        'seed': seed,
        'wall_time': float(wall),
        'nusselt': float(stats['nusselt']),
        'rms_velocity': vrms,
        'block_stats': block_stats,
        'meta_stats': meta_stats,
    }


def aggregate(results, params):
    return {
        'params': params,
        'n_seeds': len(results),
        'wall_time_mean': float(np.mean([r['wall_time'] for r in results])),
        'wall_time_std': float(np.std([r['wall_time'] for r in results])),
        'nusselt_mean': float(np.mean([r['nusselt'] for r in results])),
        'nusselt_std': float(np.std([r['nusselt'] for r in results])),
        'rms_velocity_mean': float(np.mean([r['rms_velocity'] for r in results])),
        'rms_velocity_std': float(np.std([r['rms_velocity'] for r in results])),
        'velocity_fallbacks_mean': float(np.mean([r['block_stats'].get('velocity_direct_fallbacks', 0) for r in results])),
        'velocity_solves_mean': float(np.mean([r['block_stats'].get('velocity_solve_calls', 0) for r in results])),
        'full_direct_fallbacks_mean': float(np.mean([r['block_stats'].get('full_direct_fallbacks', 0) for r in results])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Tune blocked Stokes solver parameters')
    parser.add_argument('--output', type=str, default='experiments/tuning_results', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2]
    param_grid = [
        {'schur_scale': 0.25, 'velocity_solver_rtol': 1e-8, 'velocity_acceptance_rtol': 1e-6, 'full_blocked_acceptance_rtol': 1e-4},
        {'schur_scale': 0.5, 'velocity_solver_rtol': 1e-8, 'velocity_acceptance_rtol': 1e-6, 'full_blocked_acceptance_rtol': 1e-4},
        {'schur_scale': 1.0, 'velocity_solver_rtol': 1e-8, 'velocity_acceptance_rtol': 1e-6, 'full_blocked_acceptance_rtol': 1e-4},
        {'schur_scale': 0.5, 'velocity_solver_rtol': 1e-6, 'velocity_acceptance_rtol': 1e-4, 'full_blocked_acceptance_rtol': 1e-3},
        {'schur_scale': 0.5, 'velocity_solver_rtol': 1e-10, 'velocity_acceptance_rtol': 1e-8, 'full_blocked_acceptance_rtol': 1e-5},
    ]

    all_results = []
    for params in param_grid:
        per_seed = []
        print(f"Testing params: {params}")
        for seed in seeds:
            result = run_case(
                nx=8, ny=8, rayleigh=1e4, contrast=1e2,
                picard_iterations=4, seed=seed,
                use_meta_amg=True,
                meta_training_sequences=60,
                meta_training_epochs=20,
                **params,
            )
            per_seed.append(result)
        aggregate_result = aggregate(per_seed, params)
        all_results.append(aggregate_result)
        print(f"  wall={aggregate_result['wall_time_mean']:.3f}s +- {aggregate_result['wall_time_std']:.3f}, "
              f"Nu={aggregate_result['nusselt_mean']:.4f}, Vrms={aggregate_result['rms_velocity_mean']:.2f}, "
              f"vel_fallbacks={aggregate_result['velocity_fallbacks_mean']:.1f}, full_fallbacks={aggregate_result['full_direct_fallbacks_mean']:.1f}")

    out_path = output_dir / 'blocked_stokes_tuning.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved tuning results to {out_path}")


if __name__ == '__main__':
    main()
