import csv

from experiments.export_blocked_stokes_paper_tables import export_tables


def test_paper_table_export_writes_performance_physics_and_coverage(tmp_path):
    stat = {
        'mean': 1.0, 'std': 0.1, 'count': 3, 'confidence': 0.95,
        'ci_low': 0.8, 'ci_high': 1.2, 'ci_half_width': 0.2,
    }
    common = {
        'dimension': 2, 'mesh_mode': 'structured',
        'viscosity_mode': 'temperature_dependent', 'n': 4,
        'method': 'hypre_boomeramg_fresh', 'n_samples': 3,
        'stokes_dofs': 75, 'failure_rate': 0.0, 'unavailable_rate': 0.0,
        'backend': 'hypre_boomeramg', 'hierarchy_policy': 'fresh',
        'schur_velocity_inverse': 'lu', 'schur_velocity_vcycles': 2,
        'paired_vs_method': 'direct',
        'paired_metrics': {
            'total_time_s': {
                'difference': stat, 'ratio': stat, 'paired_seed_count': 3},
            'wall_time_s': {
                'difference': stat, 'ratio': stat, 'paired_seed_count': 3},
        },
    }
    scaling_row = {
        **common,
        **{name: stat for name in (
            'setup_time_s', 'solve_time_s', 'total_time_s', 'peak_rss_bytes',
            'velocity_krylov_iterations', 'max_linear_relative_residual',
            'max_solution_relative_error_vs_direct')},
    }
    nonlinear_row = {
        **common, 'n_steps': 20, 'quality_accept_rate': 1.0,
        'physics_equivalence_rate': 1.0,
        **{name: stat for name in (
            'wall_time_s', 'amortized_wall_time_per_step_s',
            'amortized_setup_time_per_step_s',
            'amortized_linear_solve_time_per_picard_s',
            'avg_picard_iterations_per_step', 'peak_rss_bytes',
            'nusselt_relative_error_vs_direct',
            'rms_velocity_relative_error_vs_direct',
            'temperature_field_relative_error_vs_direct',
            'velocity_field_relative_error_vs_direct',
            'max_linear_relative_residual')},
    }
    scaling = {'benchmark': 'scaling', 'aggregate': [scaling_row], 'per_run': []}
    nonlinear = {'benchmark': 'nonlinear', 'aggregate': [nonlinear_row], 'per_run': []}
    export_tables(scaling, nonlinear, tmp_path)

    expected = {
        'table_scaling.csv', 'table_nonlinear_amortized.csv',
        'table_physics_equivalence.csv', 'table_ablation.csv', 'paper_tables.md',
    }
    assert expected.issubset(path.name for path in tmp_path.iterdir())
    with open(tmp_path / 'table_physics_equivalence.csv') as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]['method'] == 'hypre_boomeramg_fresh'
    with open(tmp_path / 'table_ablation.csv') as handle:
        ablation = list(csv.DictReader(handle))
    assert ablation[0]['paired_vs_method'] == 'direct'
    assert ablation[0]['schur_velocity_inverse'] == 'lu'
