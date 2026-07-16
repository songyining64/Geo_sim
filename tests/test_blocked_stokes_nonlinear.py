import copy

from experiments.benchmark_blocked_stokes_nonlinear import (
    add_physics_equivalence,
    aggregate,
    run_simulation,
)


def test_full_nonlinear_benchmark_reports_amortized_metrics_for_internal_solver():
    result = run_simulation(
        n=2,
        contrast=1e2,
        seed=0,
        method='internal_fresh',
        n_steps=2,
        picard_iterations=2,
        dimension=2,
        mesh_mode='structured',
        viscosity_mode='temperature_dependent',
    )

    assert result['status'] == 'success'
    assert result['n_steps'] == 2
    assert result['total_picard_iterations'] >= 2
    assert result['amortized_wall_time_per_step_s'] > 0.0
    assert result['amortized_linear_solve_time_per_picard_s'] >= 0.0
    assert result['peak_rss_bytes'] > 0
    assert len(result['step_metrics']) == 2
    summary = aggregate([result])[0]
    assert summary['n_seeds'] == 1
    assert summary['quality_accept_rate'] in {0.0, 1.0}


def test_physics_equivalence_compares_final_fields_against_direct_reference():
    reference = run_simulation(
        n=2, contrast=1.0, seed=0, method='direct', n_steps=1,
        picard_iterations=1, dimension=2, repeat=0)
    matching = copy.deepcopy(reference)
    matching['method'] = 'internal_fresh'
    add_physics_equivalence([reference, matching])

    assert matching['physics_equivalent']
    assert matching['velocity_field_relative_error_vs_direct'] == 0.0

    changed = copy.deepcopy(matching)
    changed['final_temperature_field'][0] += 10.0
    add_physics_equivalence([reference, changed])
    assert not changed['physics_equivalent']
