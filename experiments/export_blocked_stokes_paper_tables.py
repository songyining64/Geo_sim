"""Export paper-ready blocked Stokes performance and equivalence tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def _merge_payloads(paths, benchmark):
    payloads = [_load(path) for path in paths]
    merged = {
        'benchmark': benchmark,
        'sources': list(paths),
        'per_run': [],
        'aggregate': [],
        'failures': [],
    }
    for payload in payloads:
        merged['per_run'].extend(payload.get('per_run', []))
        merged['aggregate'].extend(payload.get('aggregate', []))
        merged['failures'].extend(payload.get('failures', []))
    return merged


def _stat(row, metric, field='mean'):
    value = row.get(metric, {})
    return value.get(field, '') if isinstance(value, dict) else value


def _scaled_stat(row, metric, field='mean', scale=1.0):
    value = _stat(row, metric, field)
    return value / scale if isinstance(value, (int, float)) else ''


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scaling_rows(payload):
    output = []
    for row in payload.get('aggregate', []):
        output.append({
            'dimension': row['dimension'],
            'mesh_mode': row['mesh_mode'],
            'viscosity_mode': row['viscosity_mode'],
            'n': row['n'],
            'stokes_dofs': row['stokes_dofs'],
            'method': row['method'],
            'n_samples': row.get('n_samples', row.get('n_seeds', 0)),
            'setup_s_mean': _stat(row, 'setup_time_s'),
            'setup_s_ci95_half': _stat(row, 'setup_time_s', 'ci_half_width'),
            'solve_s_mean': _stat(row, 'solve_time_s'),
            'solve_s_ci95_half': _stat(row, 'solve_time_s', 'ci_half_width'),
            'total_s_mean': _stat(row, 'total_time_s'),
            'total_s_ci95_half': _stat(row, 'total_time_s', 'ci_half_width'),
            'method_peak_rss_mib_mean': _scaled_stat(row, 'method_peak_rss_bytes', scale=2**20),
            'method_peak_rss_mib_ci95_half': _scaled_stat(
                row, 'method_peak_rss_bytes', 'ci_half_width', 2**20),
            'method_peak_rss_delta_mib_mean': _scaled_stat(
                row, 'method_peak_rss_delta_bytes', scale=2**20),
            'velocity_krylov_mean': _stat(row, 'velocity_krylov_iterations'),
            'max_residual_mean': _stat(row, 'max_linear_relative_residual'),
            'max_error_vs_direct_mean': _stat(row, 'max_solution_relative_error_vs_direct'),
            'failure_rate': row.get('failure_rate', 0.0),
            'unavailable_rate': row.get('unavailable_rate', 0.0),
            'numerical_failure_rate': row.get('numerical_failure_rate', 0.0),
        })
    return output


def nonlinear_performance_rows(payload):
    output = []
    for row in payload.get('aggregate', []):
        output.append({
            'dimension': row['dimension'],
            'mesh_mode': row['mesh_mode'],
            'viscosity_mode': row['viscosity_mode'],
            'n': row['n'],
            'stokes_dofs': row['stokes_dofs'],
            'n_steps': row['n_steps'],
            'method': row['method'],
            'n_samples': row.get('n_samples', row.get('n_seeds', 0)),
            'wall_s_mean': _stat(row, 'wall_time_s'),
            'wall_s_ci95_half': _stat(row, 'wall_time_s', 'ci_half_width'),
            'wall_per_step_s_mean': _stat(row, 'amortized_wall_time_per_step_s'),
            'setup_per_step_s_mean': _stat(row, 'amortized_setup_time_per_step_s'),
            'linear_per_picard_s_mean': _stat(row, 'amortized_linear_solve_time_per_picard_s'),
            'picard_per_step_mean': _stat(row, 'avg_picard_iterations_per_step'),
            'method_peak_rss_mib_mean': _scaled_stat(row, 'method_peak_rss_bytes', scale=2**20),
            'method_peak_rss_delta_mib_mean': _scaled_stat(
                row, 'method_peak_rss_delta_bytes', scale=2**20),
            'failure_rate': row.get('failure_rate', 0.0),
            'unavailable_rate': row.get('unavailable_rate', 0.0),
            'numerical_failure_rate': row.get('numerical_failure_rate', 0.0),
        })
    return output


def physics_rows(payload):
    output = []
    for row in payload.get('aggregate', []):
        output.append({
            'dimension': row['dimension'],
            'mesh_mode': row['mesh_mode'],
            'viscosity_mode': row['viscosity_mode'],
            'n': row['n'],
            'n_steps': row['n_steps'],
            'method': row['method'],
            'n_samples': row.get('n_samples', row.get('n_seeds', 0)),
            'physics_equivalence_rate': row.get('physics_equivalence_rate', 0.0),
            'quality_accept_rate': row.get('quality_accept_rate', 0.0),
            'nusselt_rel_error_mean': _stat(row, 'nusselt_relative_error_vs_direct'),
            'nusselt_rel_error_ci95_half': _stat(
                row, 'nusselt_relative_error_vs_direct', 'ci_half_width'),
            'rms_velocity_rel_error_mean': _stat(row, 'rms_velocity_relative_error_vs_direct'),
            'temperature_field_rel_error_mean': _stat(
                row, 'temperature_field_relative_error_vs_direct'),
            'velocity_field_rel_error_mean': _stat(
                row, 'velocity_field_relative_error_vs_direct'),
            'max_linear_residual_mean': _stat(row, 'max_linear_relative_residual'),
        })
    return output


def _paired_stat(row, metric, kind, field='mean'):
    return (
        row.get('paired_metrics', {})
        .get(metric, {})
        .get(kind, {})
        .get(field, '')
    )


def ablation_rows(scaling_payload, nonlinear_payload):
    output = []
    specifications = (
        ('scaling', scaling_payload, 'total_time_s'),
        ('nonlinear', nonlinear_payload, 'wall_time_s'),
    )
    for benchmark, payload, metric in specifications:
        for row in payload.get('aggregate', []):
            output.append({
                'benchmark': benchmark,
                'dimension': row['dimension'],
                'mesh_mode': row['mesh_mode'],
                'viscosity_mode': row['viscosity_mode'],
                'n': row['n'],
                'n_steps': row.get('n_steps', ''),
                'method': row['method'],
                'backend': row.get('backend', ''),
                'hierarchy_policy': row.get('hierarchy_policy', ''),
                'schur_velocity_inverse': row.get('schur_velocity_inverse', ''),
                'schur_velocity_vcycles': row.get('schur_velocity_vcycles', ''),
                'metric': metric,
                'metric_mean_s': _stat(row, metric),
                'metric_ci95_half_s': _stat(row, metric, 'ci_half_width'),
                'paired_vs_method': row.get('paired_vs_method', ''),
                'paired_difference_mean_s': _paired_stat(
                    row, metric, 'difference'),
                'paired_difference_ci95_half_s': _paired_stat(
                    row, metric, 'difference', 'ci_half_width'),
                'paired_ratio_mean': _paired_stat(row, metric, 'ratio'),
                'paired_ratio_ci95_half': _paired_stat(
                    row, metric, 'ratio', 'ci_half_width'),
                'paired_seed_count': (
                    row.get('paired_metrics', {}).get(metric, {})
                    .get('paired_seed_count', 0)),
                'quality_accept_rate': row.get('quality_accept_rate', ''),
                'physics_equivalence_rate': row.get('physics_equivalence_rate', ''),
                'failure_rate': row.get('failure_rate', 0.0),
            })
    return output


def coverage_rows(*payloads):
    groups = {}
    for payload in payloads:
        benchmark = payload.get('benchmark', 'unknown')
        for record in payload.get('per_run', []):
            key = (
                benchmark, record.get('dimension'), record.get('mesh_mode'),
                record.get('viscosity_mode'), record.get('n'), record.get('method'))
            group = groups.setdefault(key, {
                'success': 0, 'unavailable': 0, 'failed': 0, 'numerical_failed': 0})
            status = record.get('status', 'success')
            if status == 'success' and not record.get('quality_accepted', True):
                status = 'numerical_failed'
            group[status] = group.get(status, 0) + 1
        for failure in payload.get('failures', []):
            key = (
                benchmark, failure.get('dimension'), failure.get('mesh_mode'),
                failure.get('viscosity_mode'), failure.get('n'), failure.get('method'))
            groups.setdefault(key, {
                'success': 0, 'unavailable': 0, 'failed': 0, 'numerical_failed': 0
            })['failed'] += 1
    output = []
    for key, counts in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        benchmark, dimension, mesh_mode, viscosity_mode, n, method = key
        attempts = sum(counts.values())
        output.append({
            'benchmark': benchmark,
            'dimension': dimension,
            'mesh_mode': mesh_mode,
            'viscosity_mode': viscosity_mode,
            'n': n,
            'method': method,
            'n_attempts': attempts,
            'n_success': counts['success'],
            'n_unavailable': counts['unavailable'],
            'n_failed': counts['failed'],
            'n_numerical_failed': counts['numerical_failed'],
            'success_rate': counts['success'] / max(attempts, 1),
            'failure_rate': counts['failed'] / max(attempts, 1),
            'numerical_failure_rate': counts['numerical_failed'] / max(attempts, 1),
            'unavailable_rate': counts['unavailable'] / max(attempts, 1),
        })
    return output


def _markdown_table(title, rows, columns):
    lines = [f'## {title}', '', '| ' + ' | '.join(columns) + ' |',
             '|' + '|'.join(['---'] * len(columns)) + '|']
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, '')
            if value is None or value == '':
                values.append('n/a')
            else:
                values.append(f'{value:.4g}' if isinstance(value, float) else str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return lines


def export_tables(scaling_payload, nonlinear_payload, output):
    output.mkdir(parents=True, exist_ok=True)
    scaling = scaling_rows(scaling_payload)
    nonlinear = nonlinear_performance_rows(nonlinear_payload)
    physics = physics_rows(nonlinear_payload)
    ablation = ablation_rows(scaling_payload, nonlinear_payload)
    coverage = coverage_rows(scaling_payload, nonlinear_payload)
    _write_csv(output / 'table_scaling.csv', scaling)
    _write_csv(output / 'table_nonlinear_amortized.csv', nonlinear)
    _write_csv(output / 'table_physics_equivalence.csv', physics)
    _write_csv(output / 'table_ablation.csv', ablation)
    _write_csv(output / 'table_coverage_failures.csv', coverage)

    lines = ['# Blocked Stokes paper tables', '',
             'Values are independent-process means. 95% Student-t confidence intervals are '
             'reported only when at least two physical seeds are available.', '']
    lines.extend(_markdown_table('Scaling', scaling, [
        'dimension', 'mesh_mode', 'viscosity_mode', 'n', 'stokes_dofs', 'method',
        'total_s_mean', 'total_s_ci95_half', 'method_peak_rss_mib_mean',
        'failure_rate', 'numerical_failure_rate']))
    lines.extend([''])
    lines.extend(_markdown_table('Nonlinear amortized performance', nonlinear, [
        'dimension', 'mesh_mode', 'viscosity_mode', 'n_steps', 'method',
        'wall_per_step_s_mean', 'setup_per_step_s_mean', 'linear_per_picard_s_mean']))
    lines.extend([''])
    lines.extend(_markdown_table('Physics equivalence', physics, [
        'dimension', 'mesh_mode', 'viscosity_mode', 'n_steps', 'method',
        'physics_equivalence_rate', 'nusselt_rel_error_mean',
         'temperature_field_rel_error_mean', 'velocity_field_rel_error_mean']))
    lines.extend([''])
    lines.extend(_markdown_table('Ablation and paired comparisons', ablation, [
        'benchmark', 'dimension', 'n', 'method', 'backend', 'hierarchy_policy',
        'schur_velocity_inverse', 'metric_mean_s', 'paired_vs_method',
        'paired_difference_mean_s', 'paired_difference_ci95_half_s',
        'paired_ratio_mean', 'paired_seed_count']))
    lines.extend([''])
    lines.extend(_markdown_table('Coverage and failures', coverage, [
        'benchmark', 'dimension', 'mesh_mode', 'viscosity_mode', 'n', 'method',
        'n_attempts', 'success_rate', 'failure_rate', 'numerical_failure_rate',
        'unavailable_rate']))
    with open(output / 'paper_tables.md', 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaling', action='append', required=True)
    parser.add_argument('--nonlinear', action='append', required=True)
    parser.add_argument('--output', default='experiments/paper_artifacts/blocked_stokes')
    args = parser.parse_args()
    export_tables(
        _merge_payloads(args.scaling, 'Blocked Stokes scaling'),
        _merge_payloads(args.nonlinear, 'Blocked Stokes nonlinear'),
        Path(args.output))


if __name__ == '__main__':
    main()
