"""Integration tests for real FEM Stokes/Picard Meta-AMG data."""

import numpy as np

from core.stokes_solver import PicardStokesSolver, StokesConfig
from gpu_acceleration.meta_amg import MetaAMGConfig, StokesPicardSequenceGenerator


def test_picard_collector_returns_velocity_amg_blocks_and_full_systems():
    solver = PicardStokesSolver(StokesConfig(
        nx=4, ny=4, rayleigh=1e4, viscosity_contrast=1e2,
        max_picard_iterations=3,
    ))
    sequence = solver.collect_picard_sequence(max_iterations=3, seed=3)

    assert len(sequence) >= 2
    for record in sequence:
        assert record['matrix'].shape == (2 * solver.mesh.n_nodes,) * 2
        assert record['full_matrix'].shape == (solver.mesh.n_dofs,) * 2
        assert np.isfinite(record['matrix'].data).all()
        assert len(record['viscosity']) == solver.mesh.n_nodes


def test_stokes_tasks_do_not_split_one_physical_trajectory_across_sets():
    config = MetaAMGConfig(
        training_data_source='stokes_picard',
        stokes_nx=4, stokes_ny=4, stokes_picard_iterations=3,
        num_training_sequences=2,
        stokes_rayleigh_range=(1e3, 1e4),
        stokes_viscosity_contrast_range=(1.0, 1e2),
    )
    train, validation = StokesPicardSequenceGenerator(config).generate_training_data(2)

    assert train and validation
    assert set(task['sequence_id'] for task in train).isdisjoint(
        task['sequence_id'] for task in validation)
    assert all('full_matrix' in task['query_physics'] for task in train + validation)


def test_picard_solver_can_use_the_stateful_meta_amg_velocity_path():
    solver = PicardStokesSolver(StokesConfig(
        nx=3, ny=3, rayleigh=1e3, viscosity_contrast=10,
        max_picard_iterations=2, picard_tolerance=0.0,
        use_meta_amg=True, meta_training_sequences=2, meta_training_epochs=1,
    ))
    solver.initialize_temperature()
    result = solver.solve_picard()

    assert result['n_iterations'] == 2
    stats = solver.meta_amg.get_stats()
    assert stats['n_traditional'] >= 1
    assert stats['n_adapted'] >= 1
