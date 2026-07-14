"""Manufactured 3D Poisson patch test on an unstructured tetrahedral mesh."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).parent.parent))

from finite_elements.global_assembly import assemble_global_stiffness
from core.stokes_solver import PicardStokesSolver, StokesConfig, StokesMesh


def _orient_tetrahedra(points, tetrahedra):
    oriented = np.asarray(tetrahedra, dtype=int).copy()
    determinants = []
    for index, tetra in enumerate(oriented):
        coords = points[tetra]
        determinant = np.linalg.det((coords[1:] - coords[0]).T)
        if determinant < 0.0:
            oriented[index, [2, 3]] = oriented[index, [3, 2]]
            determinant = -determinant
        if determinant <= 1e-12:
            raise ValueError(f'degenerate tetrahedron {index}: det={determinant}')
        determinants.append(determinant)
    return oriented, np.asarray(determinants)


def _generate_mesh(n_interior, seed):
    rng = np.random.default_rng(seed)
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=float)
    interior = rng.uniform(0.05, 0.95, size=(n_interior, 3))
    points = np.vstack([corners, interior])
    triangulation = Delaunay(points)
    tetrahedra, determinants = _orient_tetrahedra(points, triangulation.simplices)
    return points, tetrahedra, determinants


def run_patch_test(n_interior=80, seed=0):
    points, tetrahedra, determinants = _generate_mesh(n_interior, seed)

    t0 = time.time()
    stiffness = assemble_global_stiffness(
        points, tetrahedra, element_type='tetra', order=1).tocsr()
    assembly_time = time.time() - t0

    boundary = np.unique(ConvexHull(points).simplices)
    all_nodes = np.arange(points.shape[0])
    free = np.setdiff1d(all_nodes, boundary)
    exact = points[:, 0] + 2.0 * points[:, 1] - 0.5 * points[:, 2] + 0.25
    solution = exact.copy()

    t0 = time.time()
    rhs_free = -(stiffness[free][:, boundary] @ exact[boundary])
    solution[free] = spsolve(stiffness[free][:, free], rhs_free)
    solve_time = time.time() - t0

    residual_free = (stiffness[free][:, free] @ solution[free] +
                     stiffness[free][:, boundary] @ solution[boundary])
    symmetry_error = np.linalg.norm((stiffness - stiffness.T).data) / max(
        np.linalg.norm(stiffness.data), 1e-14)
    relative_error = np.linalg.norm(solution - exact) / max(np.linalg.norm(exact), 1e-14)
    relative_residual = np.linalg.norm(residual_free) / max(np.linalg.norm(rhs_free), 1e-14)
    return {
        'seed': seed,
        'n_nodes': int(points.shape[0]),
        'n_tetrahedra': int(tetrahedra.shape[0]),
        'n_boundary_nodes': int(boundary.size),
        'n_free_nodes': int(free.size),
        'matrix_nnz': int(stiffness.nnz),
        'min_tetra_volume': float(np.min(determinants) / 6.0),
        'max_tetra_volume': float(np.max(determinants) / 6.0),
        'assembly_time': float(assembly_time),
        'solve_time': float(solve_time),
        'symmetry_error': float(symmetry_error),
        'solution_relative_error': float(relative_error),
        'free_residual_relative_norm': float(relative_residual),
        'passed': bool(
            symmetry_error < 1e-12 and relative_error < 1e-10 and
            relative_residual < 1e-10 and np.min(determinants) > 1e-12),
    }


def run_stokes_test(n_interior=20, seed=0):
    points, tetrahedra, determinants = _generate_mesh(n_interior, seed)
    mesh = StokesMesh.from_unstructured_tetrahedra(points, tetrahedra)
    config = StokesConfig(
        nx=1, ny=1, nz=1,
        rayleigh=1e2, viscosity_contrast=1.0,
        max_picard_iterations=2, picard_tolerance=0.0,
        use_meta_amg=False, schur_velocity_inverse='lu',
    )
    solver = PicardStokesSolver(config, mesh=mesh)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        solver.initialize_temperature()
    finally:
        np.random.set_state(state)
    result = solver.solve_picard()
    stats = solver._block_solver.get_stats()
    return {
        'seed': seed,
        'n_nodes': mesh.n_nodes,
        'n_tetrahedra': mesh.n_elements,
        'min_tetra_volume': float(np.min(determinants) / 6.0),
        'picard_iterations': result['n_iterations'],
        'linear_relative_residual': result['linear_relative_residual'],
        'nusselt': result['nusselt'],
        'velocity_direct_fallbacks': stats['velocity_direct_fallbacks'],
        'full_direct_fallbacks': stats['full_direct_fallbacks'],
        'finite_solution': bool(np.isfinite(solver.velocity).all()),
        'passed': bool(
            np.isfinite(solver.velocity).all() and
            result['linear_relative_residual'] < 1e-6 and
            stats['velocity_direct_fallbacks'] == 0 and
            stats['full_direct_fallbacks'] == 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interior-points', type=int, default=80)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--output', default='experiments/results_unstructured_3d')
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(',') if item]
    results = [run_patch_test(args.interior_points, seed) for seed in seeds]
    stokes_results = [run_stokes_test(min(args.interior_points, 24), seed) for seed in seeds]
    payload = {
        'benchmark': '3D unstructured linear-tetra FEM and blocked Stokes validation',
        'scope_note': 'Poisson uses an affine manufactured solution; Stokes validates the runnable blocked path on the same mesh family.',
        'config': {'interior_points': args.interior_points, 'seeds': seeds},
        'poisson_per_seed': results,
        'stokes_per_seed': stokes_results,
        'poisson_pass_rate': float(np.mean([item['passed'] for item in results])),
        'stokes_pass_rate': float(np.mean([item['passed'] for item in stokes_results])),
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'unstructured_3d.json'
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
