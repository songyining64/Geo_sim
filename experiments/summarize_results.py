"""Summarize experiment JSON outputs into paper-ready tables."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _load_results(path: Path) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_float(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    value = float(value)
    if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4f}"


def _markdown_table(rows: List[Dict], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "-")) for col in columns) + " |")
    return "\n".join([header, divider] + body)


def _format_mean_std(entry: Dict, base_key: str) -> str:
    mean_key = f'{base_key}_mean'
    std_key = f'{base_key}_std'
    if mean_key not in entry:
        return _format_float(entry.get(base_key))
    return f"{_format_float(entry[mean_key])} +- {_format_float(entry[std_key])}"


def summarize(result_path: Path, output_dir: Path) -> None:
    results = _load_results(result_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = ["# Experiment Summary", ""]
    config = results.get('config', {})
    summary_lines.append("## Config")
    for key in [
        'preset', 'meta_epochs', 'meta_tasks', 'n_runs', 'sequence_length',
        'matrix_sizes', 'contrasts', 'stokes_nx', 'stokes_ny',
        'stokes_picard_iterations', 'stokes_rayleigh', 'stokes_viscosity_contrast',
        'periodic_rebuild_interval', 'matrix_change_threshold',
        'nusselt_relative_tolerance', 'rms_velocity_relative_tolerance',
        'velocity_field_relative_tolerance', 'linear_residual_tolerance',
    ]:
        if key in config:
            summary_lines.append(f"- `{key}`: {config[key]}")
    summary_lines.append("")

    if 'experiment_2' in results:
        rows = []
        for entry in results['experiment_2']:
            rows.append({
                'matrix_size': entry['matrix_size'],
                'trad_setup': _format_mean_std(entry['traditional'], 'setup_time_mean'),
                'reuse_setup': _format_mean_std(entry['reuse'], 'setup_time_mean'),
                'zero_shot_setup': _format_mean_std(entry['zero_shot'], 'setup_time_mean'),
                'adapt_setup': _format_mean_std(entry['adapted'], 'setup_time_mean'),
                'trad_total': _format_mean_std(entry['traditional'], 'total_time_mean'),
                'reuse_total': _format_mean_std(entry['reuse'], 'total_time_mean'),
                'adapt_total': _format_mean_std(entry['adapted'], 'total_time_mean'),
                'adapt_speedup': _format_float(entry['adapt_speedup_vs_traditional']),
                'adapt_fallback': _format_mean_std(entry['adapted'], 'fallback_rate'),
            })
        _write_csv(output_dir / 'table_e2_synthetic.csv', rows, list(rows[0].keys()))
        summary_lines.append("## Table E2 Synthetic")
        summary_lines.append(_markdown_table(rows, list(rows[0].keys())))
        summary_lines.append("")

    replay = results.get('experiment_2_stokes_picard_replay') or results.get('experiment_6_stokes_picard_replay')
    if replay:
        rows = []
        for method in ['traditional', 'reuse', 'periodic_rebuild',
                       'change_aware_reuse', 'zero_shot', 'adapted']:
            if method not in replay:
                continue
            entry = replay[method]
            rows.append({
                'method': method,
                'setup_time': _format_mean_std(entry, 'setup_time_mean'),
                'solve_time': _format_mean_std(entry, 'solve_time_mean'),
                'total_time': _format_mean_std(entry, 'total_time_mean'),
                'iterations': _format_mean_std(entry, 'iterations_mean'),
                'fallback_rate': _format_mean_std(entry, 'fallback_rate'),
                'accepted_rate': _format_mean_std(entry, 'accepted_rate'),
                'relative_residual_mean': _format_mean_std(entry, 'relative_residual_mean'),
                'relative_residual_max': _format_mean_std(entry, 'relative_residual_max'),
                'rebuild_rate': _format_mean_std(entry, 'rebuild_rate'),
            })
        _write_csv(output_dir / 'table_stokes_replay.csv', rows, list(rows[0].keys()))
        summary_lines.append("## Table Stokes Replay")
        summary_lines.append(_markdown_table(rows, list(rows[0].keys())))
        summary_lines.append("")

    full = results.get('experiment_2_stokes_picard_full') or results.get('experiment_6_stokes_picard_full')
    if full:
        rows = [
            {
                'method': 'reference_direct',
                'wall_time': _format_mean_std(full['reference_direct'], 'wall_time'),
                'picard_iterations': str(full['reference_direct'].get('picard_iterations', '-')),
                'nusselt': _format_mean_std(full['reference_direct'], 'nusselt'),
                'rms_velocity': _format_mean_std(full['reference_direct'], 'rms_velocity'),
                'valid_physics': 'yes',
                'velocity_fallbacks': '-',
                'velocity_solves': '-',
                'velocity_krylov_iters': '-',
                'linear_relative_residual': _format_mean_std(full['reference_direct'], 'linear_relative_residual'),
                'nusselt_relative_error': '0.0000',
                'rms_velocity_relative_error': '0.0000',
                'velocity_field_relative_error': '0.0000',
                'blocked_solver_valid': 'yes',
                'direct_fallback_time': '-',
                'pressure_iters': '-',
                'pressure_fallbacks': '-',
                'pressure_pc_fallbacks': '-',
            },
            {
                'method': 'meta_blocked',
                'wall_time': _format_mean_std(full['meta_blocked'], 'wall_time'),
                'picard_iterations': str(full['meta_blocked'].get('picard_iterations', '-')),
                'nusselt': _format_mean_std(full['meta_blocked'], 'nusselt'),
                'rms_velocity': _format_mean_std(full['meta_blocked'], 'rms_velocity'),
                'valid_physics': _format_float(full['meta_blocked'].get('valid_physics_rate', full['meta_blocked'].get('valid_physics'))),
                'velocity_fallbacks': _format_float(full['meta_blocked'].get('block_stats', {}).get('velocity_direct_fallbacks_mean', '-')),
                'velocity_solves': _format_float(full['meta_blocked'].get('block_stats', {}).get('velocity_solve_calls_mean', '-')),
                'velocity_krylov_iters': _format_float(full['meta_blocked'].get('block_stats', {}).get(
                    'velocity_krylov_iterations_mean',
                    full['meta_blocked'].get('block_stats', {}).get('velocity_cg_iterations_mean', '-'))),
                'linear_relative_residual': _format_mean_std(full['meta_blocked'], 'linear_relative_residual'),
                'nusselt_relative_error': _format_mean_std(full['meta_blocked'], 'nusselt_relative_error'),
                'rms_velocity_relative_error': _format_mean_std(full['meta_blocked'], 'rms_velocity_relative_error'),
                'velocity_field_relative_error': _format_mean_std(full['meta_blocked'], 'velocity_field_relative_error'),
                'blocked_solver_valid': _format_float(full['meta_blocked'].get('blocked_solver_valid_rate')),
                'direct_fallback_time': _format_float(
                    full['meta_blocked'].get('block_stats', {}).get('velocity_direct_fallback_time_mean', 0.0) +
                    full['meta_blocked'].get('block_stats', {}).get('full_direct_fallback_time_mean', 0.0)),
                'pressure_iters': _format_float(full['meta_blocked'].get('block_stats', {}).get('pressure_krylov_iterations_mean', '-')),
                'pressure_fallbacks': _format_float(full['meta_blocked'].get('block_stats', {}).get('pressure_solver_fallbacks_mean', '-')),
                'pressure_pc_fallbacks': _format_float(full['meta_blocked'].get('block_stats', {}).get('pressure_preconditioner_fallbacks_mean', '-')),
            },
        ]
        _write_csv(output_dir / 'table_stokes_full.csv', rows, list(rows[0].keys()))
        summary_lines.append("## Table Stokes Full")
        summary_lines.append(_markdown_table(rows, list(rows[0].keys())))
        summary_lines.append("")

    if 'experiment_6' in results:
        rows = []
        for method in ['traditional', 'reuse', 'periodic_rebuild',
                       'change_aware_reuse', 'zero_shot', 'adapted']:
            if method not in results['experiment_6']:
                continue
            entry = results['experiment_6'][method]
            rows.append({
                'method': method,
                'setup_time': _format_mean_std(entry, 'setup_time_mean'),
                'solve_time': _format_mean_std(entry, 'solve_time_mean'),
                'total_time': _format_mean_std(entry, 'total_time_mean'),
                'iterations': _format_mean_std(entry, 'iterations_mean'),
                'fallback_rate': _format_mean_std(entry, 'fallback_rate'),
                'relative_residual_max': _format_mean_std(entry, 'relative_residual_max'),
                'rebuild_rate': _format_mean_std(entry, 'rebuild_rate'),
            })
        if 'neural_amg_setup_only' in results['experiment_6']:
            rows.append({
                'method': 'neural_amg_setup_only',
                'setup_time': _format_mean_std(results['experiment_6']['neural_amg_setup_only'], 'setup_time_mean'),
                'solve_time': '-',
                'total_time': '-',
                'iterations': '-',
                'fallback_rate': '-',
                'relative_residual_max': '-',
                'rebuild_rate': '-',
            })
        _write_csv(output_dir / 'table_e6_methods.csv', rows, list(rows[0].keys()))
        summary_lines.append("## Table E6 Methods")
        summary_lines.append(_markdown_table(rows, list(rows[0].keys())))
        summary_lines.append("")

    with open(output_dir / 'summary.md', 'w') as f:
        f.write("\n".join(summary_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize Meta-AMG experiment results')
    parser.add_argument('results', type=str, help='Path to results.json')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for summary tables')
    args = parser.parse_args()

    result_path = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else result_path.parent / 'summary'
    summarize(result_path, output_dir)
    print(f"Summary written to {output_dir}")


if __name__ == '__main__':
    main()
