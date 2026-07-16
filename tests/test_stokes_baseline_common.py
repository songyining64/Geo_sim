import numpy as np

from experiments.stokes_baseline_common import (
    apply_viscosity_mode,
    build_case_config,
    build_case_mesh,
    deterministic_method_order,
    paired_seed_summary_stats,
    prepare_output_directory,
    summary_stats,
)


def test_viscosity_modes_map_to_expected_solver_flags():
    config = build_case_config(2, 1e4, 2)
    apply_viscosity_mode(config, 'isoviscous', 1e4)
    assert config.viscosity_contrast == 1.0
    assert not config.strain_rate_dependent_viscosity

    apply_viscosity_mode(config, 'temperature_dependent', 1e4)
    assert config.viscosity_contrast == 1e4
    assert not config.strain_rate_dependent_viscosity

    apply_viscosity_mode(config, 'temperature_strain_rate', 1e4)
    assert config.viscosity_contrast == 1e4
    assert config.strain_rate_dependent_viscosity


def test_build_case_mesh_supports_structured_2d_and_unstructured_3d():
    mesh_2d, stats_2d = build_case_mesh(2, dimension=2, mesh_mode='structured', seed=0)
    assert mesh_2d.dim == 2
    assert stats_2d['n_nodes'] == mesh_2d.n_nodes
    assert stats_2d['stokes_dofs'] == 3 * mesh_2d.n_nodes

    mesh_unstructured_2d, stats_unstructured_2d = build_case_mesh(
        2, dimension=2, mesh_mode='unstructured', seed=0,
        unstructured_points=8)
    assert mesh_unstructured_2d.dim == 2
    assert stats_unstructured_2d['mesh_mode'] == 'unstructured'
    assert min(
        np.linalg.det((mesh_unstructured_2d.nodes[element][1:] -
                       mesh_unstructured_2d.nodes[element][0]).T)
        for element in mesh_unstructured_2d.elements) > 0.0

    mesh_3d, stats_3d = build_case_mesh(2, dimension=3, mesh_mode='unstructured',
                                        seed=0, unstructured_points=8)
    assert mesh_3d.dim == 3
    assert stats_3d['mesh_mode'] == 'unstructured'
    assert stats_3d['n_elements'] == mesh_3d.n_elements


def test_summary_stats_reports_student_t_confidence_interval():
    stats = summary_stats([1.0, 2.0, 3.0])
    assert stats['count'] == 3
    assert stats['ci_low'] < stats['mean'] < stats['ci_high']
    assert stats['ci_half_width'] > 0.0


def test_method_order_is_deterministic_and_case_specific():
    methods = ['direct', 'pyamg_fresh', 'hypre_boomeramg_fresh']
    first = deterministic_method_order(methods, 2, 8, 0, 0)
    assert first == deterministic_method_order(methods, 2, 8, 0, 0)
    assert sorted(first) == sorted(methods)
    assert first != deterministic_method_order(methods, 2, 8, 1, 0)


def test_paired_stats_match_repetitions_then_aggregate_by_seed():
    target = [
        {'seed': 0, 'repeat': 0, 'time': 3.0},
        {'seed': 0, 'repeat': 1, 'time': 5.0},
        {'seed': 1, 'repeat': 0, 'time': 8.0},
    ]
    reference = [
        {'seed': 0, 'repeat': 0, 'time': 2.0},
        {'seed': 0, 'repeat': 1, 'time': 3.0},
        {'seed': 1, 'repeat': 0, 'time': 4.0},
    ]
    stats = paired_seed_summary_stats(target, reference, 'time')
    assert stats['paired_seed_count'] == 2
    assert stats['paired_raw_count'] == 3
    assert stats['difference']['mean'] == 2.75


def test_fresh_output_rejects_nonempty_directory(tmp_path):
    output = prepare_output_directory(tmp_path / 'new', fresh=True)
    (output / 'result.json').write_text('{}')
    try:
        prepare_output_directory(output, fresh=True)
    except FileExistsError:
        pass
    else:
        raise AssertionError('expected nonempty formal output to be rejected')
