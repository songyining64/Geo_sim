from experiments.benchmark_stokes_3d_methods import aggregate, replay_direct, build_shared_trajectory


def test_3d_method_benchmark_reports_required_metrics():
    _, records, trajectory = build_shared_trajectory(
        n=2, contrast=100.0, seed=0, picard_iterations=2)
    result, _ = replay_direct(records)
    result.update({'seed': 0, 'viscosity_contrast': 100.0, **trajectory})

    assert result['max_linear_relative_residual'] < 1e-10
    assert result['nodal_viscosity_contrast'] > 50.0
    assert 1.0 < result['element_viscosity_contrast'] < result['nodal_viscosity_contrast']
    assert result['python_peak_memory_bytes'] > 0
    assert result['trajectory_sparse_bytes'] > 0
    assert result['full_fallbacks'] == 0
    summary = aggregate([result])[0]
    assert summary['n_seeds'] == 1
    assert summary['total_time_s']['mean'] > 0.0
