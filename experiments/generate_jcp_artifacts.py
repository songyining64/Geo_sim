"""Generate JCP-ready tables and scaling plots from validated JSON results."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
OUTPUT = ROOT / 'paper_artifacts'


def load(relative_path):
    with open(ROOT / relative_path, 'r') as f:
        return json.load(f)


def write_csv(name, rows):
    path = OUTPUT / name
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(entry, key):
    value = entry[key]
    return value['mean'] if isinstance(value, dict) else value


def setup_value(aggregate, method):
    key = f'{method}_total_online_preconditioner_setup'
    if key in aggregate:
        return mean(aggregate, key)
    return mean(aggregate, f'{method}_setup_time' if method == 'traditional' else 'meta_online_setup_time')


def wall_reduction(aggregate):
    if 'wall_time_reduction_fraction' in aggregate:
        return mean(aggregate, 'wall_time_reduction_fraction')
    traditional = mean(aggregate, 'traditional_wall_time')
    meta = mean(aggregate, 'meta_wall_time')
    return 1.0 - meta / max(traditional, 1e-14)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    pure = {
        8: load('results_blocked_comparison_medium/blocked_comparison.json')['aggregate'],
        16: load('results_blocked_size16_adapt1_combined/blocked_comparison.json')['aggregate'],
        32: load('results_blocked_size32_adapt1_seed0/blocked_comparison.json')['aggregate'],
    }
    optimized = {
        16: load('results_blocked_size16_schur_lu_seed0/blocked_comparison.json')['aggregate'],
        32: load('results_blocked_size32_schur_lu/blocked_comparison.json')['aggregate'],
        64: load('results_blocked_size64_schur_lu_seed0_k1000/blocked_comparison.json')['aggregate'],
    }

    schur_rows = []
    for variant, datasets in [('pure_iterative', pure), ('schur_optimized_lu', optimized)]:
        for size, entry in datasets.items():
            schur_rows.append({
                'variant': variant,
                'size': size,
                'n_seeds': entry['n_seeds'],
                'traditional_setup_s': setup_value(entry, 'traditional'),
                'meta_setup_s': setup_value(entry, 'meta'),
                'setup_speedup': setup_value(entry, 'traditional') / setup_value(entry, 'meta'),
                'traditional_wall_s': mean(entry, 'traditional_wall_time'),
                'meta_wall_s': mean(entry, 'meta_wall_time'),
                'wall_reduction_fraction': wall_reduction(entry),
                'krylov_ratio': mean(entry, 'krylov_ratio_meta_over_traditional'),
                'velocity_field_relative_error': mean(entry, 'velocity_field_relative_error'),
                'meta_full_fallbacks': mean(entry, 'meta_full_direct_fallbacks'),
                'traditional_velocity_fallbacks': mean(entry, 'traditional_velocity_direct_fallbacks')
                if 'traditional_velocity_direct_fallbacks' in entry else 0.0,
                'meta_velocity_fallbacks': mean(entry, 'meta_velocity_direct_fallbacks')
                if 'meta_velocity_direct_fallbacks' in entry else 0.0,
                'production_eligible': bool(
                    mean(entry, 'meta_full_direct_fallbacks') == 0 and
                    (mean(entry, 'traditional_velocity_direct_fallbacks')
                     if 'traditional_velocity_direct_fallbacks' in entry else 0.0) == 0 and
                    (mean(entry, 'meta_velocity_direct_fallbacks')
                     if 'meta_velocity_direct_fallbacks' in entry else 0.0) == 0),
            })
    write_csv('table_schur_variants.csv', schur_rows)

    ablation = load('results_blocked_adapt_ablation_medium/blocked_comparison.json')
    adaptation_rows = []
    for steps, entry in sorted(ablation['aggregate_by_adapt_steps'].items(), key=lambda item: int(item[0])):
        adaptation_rows.append({
            'adapt_steps': int(steps),
            'n_seeds': entry['n_seeds'],
            'meta_adaptation_time_s': mean(entry, 'meta_adaptation_time'),
            'online_setup_change_fraction': mean(entry, 'online_setup_change_fraction'),
            'setup_speedup': mean(entry, 'setup_speedup_traditional_over_meta_online'),
            'krylov_ratio': mean(entry, 'krylov_ratio_meta_over_traditional'),
            'wall_reduction_fraction': mean(entry, 'wall_time_reduction_fraction'),
            'velocity_field_relative_error': mean(entry, 'velocity_field_relative_error'),
        })
    write_csv('table_adaptation_ablation.csv', adaptation_rows)

    external = load('results_external_velocity_baselines/external_velocity_baselines.json')
    external_rows = [{
        'method': 'scipy_direct',
        'setup_time_s': external['scipy_direct']['setup_time'],
        'solve_time_s': external['scipy_direct']['solve_time'],
        'relative_residual': external['scipy_direct']['relative_residual'],
        'status': 'success',
    }]
    for backend in ['pyamg', 'petsc']:
        for method_name, result in external[backend].get('attempts', {}).items():
            external_rows.append({
                'method': f'{backend}_{method_name}',
                'setup_time_s': result.get('setup_time', ''),
                'solve_time_s': result.get('solve_time', ''),
                'relative_residual': result.get('relative_residual', ''),
                'status': 'failed' if result.get('failed') else 'success',
            })
    write_csv('table_external_baselines.csv', external_rows)

    three_d = load('results_unstructured_3d/unstructured_3d.json')
    write_csv('table_unstructured_3d_poisson.csv', three_d['poisson_per_seed'])
    write_csv('table_unstructured_3d_stokes.csv', three_d['stokes_per_seed'])

    colors = {'traditional': '#3b4a5a', 'meta': '#c84b31'}
    for metric, filename, ylabel in [
        ('setup', 'figure_setup_vs_size.png', 'Online preconditioner setup (s)'),
        ('wall', 'figure_wall_time_vs_size.png', 'Full solve wall time (s)'),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for variant, datasets, linestyle in [
            ('Pure iterative Schur', pure, '--'),
            ('Schur-optimized LU', optimized, '-'),
        ]:
            sizes = sorted(datasets)
            for method in ['traditional', 'meta']:
                if metric == 'setup':
                    values = [setup_value(datasets[size], method) for size in sizes]
                else:
                    values = [mean(datasets[size], f'{method}_wall_time') for size in sizes]
                ax.plot(sizes, values, marker='o', linestyle=linestyle,
                        color=colors[method], label=f'{variant}: {method.title()}')
        ax.set_xlabel('Stokes grid size (N x N)')
        ax.set_ylabel(ylabel)
        ax.set_yscale('log')
        ax.set_xticks([8, 16, 32, 64])
        if 64 in optimized:
            fallback_y = (setup_value(optimized[64], 'traditional') if metric == 'setup'
                          else mean(optimized[64], 'traditional_wall_time'))
            ax.scatter([64], [fallback_y], marker='x', s=90, linewidths=2,
                       color='#111111', zorder=5)
            ax.annotate('64 smoke: traditional\nvelocity fallback',
                        xy=(64, fallback_y), xytext=(-105, -8),
                        textcoords='offset points', fontsize=8,
                        arrowprops={'arrowstyle': '->', 'color': '#555555'})
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTPUT / filename, dpi=220)
        plt.close(fig)

    print(f'Paper artifacts written to {OUTPUT}')


if __name__ == '__main__':
    main()
