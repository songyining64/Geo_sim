"""Manufactured-solution convergence study for the 3D tetrahedral Stokes discretization."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stokes_solver import GlobalStokesAssembler, StokesMesh
from finite_elements.basis_functions import LagrangeTetra
from finite_elements.quadrature import tetra_points_weights
from finite_elements.transformations import jacobian_matrix, jacobian_det, dN_dx


def exact_velocity(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.column_stack([
        np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z),
        -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z),
        np.zeros_like(x),
    ])


def exact_velocity_gradient(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    gradient = np.zeros((len(points), 3, 3))
    gradient[:, 0, 0] = np.pi * np.cos(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    gradient[:, 0, 1] = -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    gradient[:, 0, 2] = -np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)
    gradient[:, 1, 0] = np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    gradient[:, 1, 1] = -np.pi * np.cos(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    gradient[:, 1, 2] = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    return gradient


def exact_pressure(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)


def exact_pressure_gradient(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.column_stack([
        np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z),
        np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z),
        np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z),
    ])


def body_force(points):
    return 3.0 * np.pi ** 2 * exact_velocity(points) + exact_pressure_gradient(points)


def assemble_velocity_forcing(mesh):
    velocity_rhs = np.zeros(mesh.n_nodes * mesh.dim)
    pressure_rhs = np.zeros(mesh.n_nodes)
    basis = LagrangeTetra(1)
    quad_points, quad_weights = tetra_points_weights(2)
    for element in mesh.elements:
        coordinates = mesh.nodes[element]
        for point, weight in zip(quad_points, quad_weights):
            N = basis.evaluate(point)[0]
            derivatives = basis.evaluate_derivatives(point)[0]
            jacobian = jacobian_matrix(coordinates, derivatives)
            determinant = jacobian_det(jacobian)
            gradients = dN_dx(derivatives, np.linalg.inv(jacobian))
            physical_point = N @ coordinates
            force = body_force(physical_point.reshape(1, 3))[0]
            edge_lengths = [
                np.linalg.norm(coordinates[j] - coordinates[i])
                for i in range(4) for j in range(i + 1, 4)]
            tau_pspg = 0.15 * max(edge_lengths) ** 2 / 2.0
            for local_node, global_node in enumerate(element):
                for component in range(3):
                    velocity_rhs[global_node * 3 + component] += (
                        N[local_node] * force[component] * determinant * weight)
                pressure_rhs[global_node] -= (
                    tau_pspg * np.dot(gradients[local_node], force) * determinant * weight)
    return velocity_rhs, pressure_rhs


def apply_dirichlet(matrix, rhs, dofs, values):
    matrix = matrix.tolil()
    for dof, value in zip(dofs, values):
        column = matrix[:, dof].toarray().ravel()
        rhs -= column * value
        matrix[:, dof] = 0.0
        matrix[dof, :] = 0.0
        matrix[dof, dof] = 1.0
        rhs[dof] = value
    return matrix.tocsr(), rhs


def compute_errors(mesh, velocity, pressure):
    basis = LagrangeTetra(1)
    quad_points, quad_weights = tetra_points_weights(2)
    accumulators = {key: 0.0 for key in ('u_l2_error', 'u_l2_norm', 'u_h1_error',
                                         'u_h1_norm', 'p_l2_error', 'p_l2_norm')}
    nodal_velocity = velocity.reshape(mesh.n_nodes, 3)
    for element in mesh.elements:
        coordinates = mesh.nodes[element]
        for point, weight in zip(quad_points, quad_weights):
            N = basis.evaluate(point)[0]
            derivatives = basis.evaluate_derivatives(point)[0]
            jacobian = jacobian_matrix(coordinates, derivatives)
            dV = jacobian_det(jacobian) * weight
            gradients = dN_dx(derivatives, np.linalg.inv(jacobian))
            physical_point = (N @ coordinates).reshape(1, 3)
            velocity_h = N @ nodal_velocity[element]
            velocity_exact = exact_velocity(physical_point)[0]
            gradient_h = nodal_velocity[element].T @ gradients
            gradient_exact = exact_velocity_gradient(physical_point)[0]
            pressure_h = float(N @ pressure[element])
            pressure_exact = float(exact_pressure(physical_point)[0])
            accumulators['u_l2_error'] += np.dot(velocity_h - velocity_exact,
                                                 velocity_h - velocity_exact) * dV
            accumulators['u_l2_norm'] += np.dot(velocity_exact, velocity_exact) * dV
            accumulators['u_h1_error'] += np.sum((gradient_h - gradient_exact) ** 2) * dV
            accumulators['u_h1_norm'] += np.sum(gradient_exact ** 2) * dV
            accumulators['p_l2_error'] += (pressure_h - pressure_exact) ** 2 * dV
            accumulators['p_l2_norm'] += pressure_exact ** 2 * dV
    return {
        'velocity_l2_error': float(np.sqrt(accumulators['u_l2_error'])),
        'velocity_relative_l2_error': float(np.sqrt(
            accumulators['u_l2_error'] / accumulators['u_l2_norm'])),
        'velocity_h1_seminorm_error': float(np.sqrt(accumulators['u_h1_error'])),
        'velocity_relative_h1_seminorm_error': float(np.sqrt(
            accumulators['u_h1_error'] / accumulators['u_h1_norm'])),
        'pressure_l2_error': float(np.sqrt(accumulators['p_l2_error'])),
        'pressure_relative_l2_error': float(np.sqrt(
            accumulators['p_l2_error'] / accumulators['p_l2_norm'])),
    }


def run_level(n):
    mesh = StokesMesh(n, n, n)
    assembler = GlobalStokesAssembler(mesh)
    viscosity = np.ones(mesh.n_nodes)
    temperature = np.zeros(mesh.n_nodes)
    t0 = time.time()
    full_matrix = assembler.assemble(viscosity, temperature)
    assembly_time = time.time() - t0

    velocity_indices = np.array([
        node * 5 + component for node in range(mesh.n_nodes) for component in range(3)])
    pressure_indices = np.array([node * 5 + 3 for node in range(mesh.n_nodes)])
    system_indices = np.concatenate([velocity_indices, pressure_indices])
    matrix = full_matrix[system_indices][:, system_indices].tocsr()
    rhs = np.zeros(matrix.shape[0])
    velocity_rhs, pressure_rhs = assemble_velocity_forcing(mesh)
    rhs[:velocity_indices.size] = velocity_rhs
    rhs[velocity_indices.size:] = pressure_rhs

    exact = exact_velocity(mesh.nodes).reshape(-1)
    boundary_velocity_dofs = []
    boundary_values = []
    for node in mesh.boundary_nodes:
        for component in range(3):
            boundary_velocity_dofs.append(node * 3 + component)
            boundary_values.append(exact[node * 3 + component])
    pressure_pin = velocity_indices.size
    boundary_velocity_dofs.append(pressure_pin)
    boundary_values.append(0.0)
    matrix, rhs = apply_dirichlet(
        matrix, rhs, boundary_velocity_dofs, boundary_values)

    t0 = time.time()
    solution = spsolve(matrix, rhs)
    solve_time = time.time() - t0
    velocity = solution[:velocity_indices.size]
    pressure = solution[velocity_indices.size:]
    errors = compute_errors(mesh, velocity, pressure)
    residual = np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), 1e-14)
    return {
        'n': n,
        'h': 1.0 / n,
        'n_nodes': mesh.n_nodes,
        'n_tetrahedra': mesh.n_elements,
        'stokes_dofs': int(matrix.shape[0]),
        'matrix_nnz': int(matrix.nnz),
        'assembly_time': float(assembly_time),
        'solve_time': float(solve_time),
        'linear_relative_residual': float(residual),
        **errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', default='2,3,4,6')
    parser.add_argument('--output', default='experiments/results_stokes_3d_convergence')
    args = parser.parse_args()
    levels = [int(item) for item in args.levels.split(',') if item]
    results = [run_level(level) for level in levels]
    metrics = ('velocity_l2_error', 'velocity_h1_seminorm_error', 'pressure_l2_error')
    results[0]['observed_orders'] = {metric: None for metric in metrics}
    for previous, current in zip(results, results[1:]):
        current['observed_orders'] = {
            metric: float(np.log(previous[metric] / current[metric]) /
                          np.log(previous['h'] / current['h']))
            for metric in metrics
        }
    payload = {
        'benchmark': '3D manufactured divergence-free Stokes convergence',
        'exact_velocity': '[sin(pi*x)cos(pi*y)cos(pi*z), -cos(pi*x)sin(pi*y)cos(pi*z), 0]',
        'exact_pressure': 'sin(pi*x)sin(pi*y)sin(pi*z)',
        'discretization': 'continuous P1 velocity / P1 pressure on linear tetrahedra with PSPG',
        'pspg_beta': 0.15,
        'boundary_conditions': 'exact velocity Dirichlet on all faces; pressure pinned at (0,0,0)',
        'levels': results,
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'stokes_3d_convergence.json'
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
