"""Create paper-ready CSV and Markdown summaries for the 3D experiments."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).parent


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def _write_csv(path, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    convergence_dir = ROOT / 'results_stokes_3d_convergence'
    methods_dir = ROOT / 'results_stokes_3d_methods'
    convergence = _load(convergence_dir / 'stokes_3d_convergence.json')
    methods = _load(methods_dir / 'stokes_3d_methods.json')
    convergence_rows = []
    for level in convergence['levels']:
        orders = level['observed_orders']
        convergence_rows.append({
            'n': level['n'], 'dofs': level['stokes_dofs'],
            'L2_u': level['velocity_l2_error'],
            'order_L2_u': orders['velocity_l2_error'],
            'H1_u': level['velocity_h1_seminorm_error'],
            'order_H1_u': orders['velocity_h1_seminorm_error'],
            'L2_p': level['pressure_l2_error'],
            'order_L2_p': orders['pressure_l2_error'],
            'relative_residual': level['linear_relative_residual'],
        })
    method_rows = []
    for row in methods['aggregate']:
        method_rows.append({
            'method': row['method'],
            'viscosity_contrast': row['viscosity_contrast'],
            'n_seeds': row['n_seeds'],
            'element_contrast_mean': row['element_viscosity_contrast']['mean'],
            'element_contrast_std': row['element_viscosity_contrast']['std'],
            'setup_s_mean': row['setup_time_s']['mean'],
            'setup_s_std': row['setup_time_s']['std'],
            'solve_s_mean': row['solve_time_s']['mean'],
            'solve_s_std': row['solve_time_s']['std'],
            'total_s_mean': row['total_time_s']['mean'],
            'total_s_std': row['total_time_s']['std'],
            'linear_solver_calls_mean': row['linear_solver_calls']['mean'],
            'velocity_krylov_iterations_mean': row['velocity_krylov_iterations']['mean'],
            'velocity_krylov_iterations_std': row['velocity_krylov_iterations']['std'],
            'python_peak_bytes_mean': row['python_peak_memory_bytes']['mean'],
            'max_relative_residual_mean': row['max_linear_relative_residual']['mean'],
            'max_error_vs_direct_mean': row['max_solution_relative_error_vs_direct']['mean'],
            'velocity_fallbacks_mean': row['velocity_fallbacks']['mean'],
            'pressure_fallbacks_mean': row['pressure_fallbacks']['mean'],
            'full_fallbacks_mean': row['full_fallbacks']['mean'],
        })
    _write_csv(convergence_dir / 'table_stokes_3d_convergence.csv', convergence_rows)
    _write_csv(methods_dir / 'table_stokes_3d_methods.csv', method_rows)
    lines = [
        '# 3D Stokes validation summary', '',
        '## Manufactured-solution convergence', '',
        '| n | DOFs | L2(u) | order | H1(u) | order | L2(p) | order |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in convergence_rows:
        fmt = lambda value: '-' if value is None else f'{value:.3f}'
        lines.append(f"| {row['n']} | {row['dofs']} | {row['L2_u']:.3e} | "
                     f"{fmt(row['order_L2_u'])} | {row['H1_u']:.3e} | "
                     f"{fmt(row['order_H1_u'])} | {row['L2_p']:.3e} | "
                     f"{fmt(row['order_L2_p'])} |")
    lines.extend(['', '## Variable-viscosity solver comparison', '',
                  'Values are mean +/- population standard deviation over three seeds.', '',
                  '| method | nodal contrast | element contrast | setup (s) | solve (s) | total (s) | velocity Krylov | full fallback |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for row in method_rows:
        lines.append(f"| {row['method']} | {row['viscosity_contrast']:.0e} | "
                     f"{row['element_contrast_mean']:.2e} | "
                     f"{row['setup_s_mean']:.3f} +/- {row['setup_s_std']:.3f} | "
                     f"{row['solve_s_mean']:.3f} +/- {row['solve_s_std']:.3f} | "
                     f"{row['total_s_mean']:.3f} +/- {row['total_s_std']:.3f} | "
                     f"{row['velocity_krylov_iterations_mean']:.1f} +/- {row['velocity_krylov_iterations_std']:.1f} | "
                     f"{row['full_fallbacks_mean']:.1f} |")
    lines.extend(['', '> Scope: these n=4 results establish small-grid feasibility only. '
                  'Direct is faster at this size; no 3D large-scale speedup is claimed.',
                  '> Memory: Python peak memory excludes some native SciPy/SuperLU allocations.'])
    with open(methods_dir / 'summary.md', 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
