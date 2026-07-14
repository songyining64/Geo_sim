from experiments.benchmark_unstructured_3d import run_patch_test, run_stokes_test
import numpy as np

from core.stokes_solver import StokesMesh
from finite_elements.assembly import stokes_heat_element_matrix_3d


def test_unstructured_tetra_patch_reproduces_affine_solution():
    result = run_patch_test(n_interior=12, seed=0)

    assert result['n_tetrahedra'] > 0
    assert result['min_tetra_volume'] > 0.0
    assert result['symmetry_error'] < 1e-12
    assert result['solution_relative_error'] < 1e-10
    assert result['free_residual_relative_norm'] < 1e-10
    assert result['passed']


def test_unstructured_tetra_stokes_blocked_path_is_finite_and_accurate():
    result = run_stokes_test(n_interior=8, seed=0)

    assert result['finite_solution']
    assert result['linear_relative_residual'] < 1e-6
    assert result['velocity_direct_fallbacks'] == 0
    assert result['full_direct_fallbacks'] == 0
    assert result['passed']


def test_structured_3d_stokes_mesh_has_positive_tetrahedra():
    mesh = StokesMesh(2, 2, 2)
    determinants = [
        np.linalg.det((mesh.nodes[element][1:] - mesh.nodes[element][0]).T)
        for element in mesh.elements]

    assert len(determinants) == 48
    assert min(determinants) > 0.0


def test_3d_stokes_velocity_element_block_is_symmetric():
    coordinates = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ])
    matrix = stokes_heat_element_matrix_3d(coordinates, {
        'viscosity': 2.0, 'thermal_conductivity': 1.0,
        'thermal_expansivity': 0.0, 'gravity': np.array([0.0, 0.0, -1.0]),
    })
    velocity_indices = [node * 5 + component for node in range(4) for component in range(3)]
    velocity_block = matrix[np.ix_(velocity_indices, velocity_indices)]

    assert np.isfinite(matrix).all()
    assert np.allclose(velocity_block, velocity_block.T, atol=1e-12)
