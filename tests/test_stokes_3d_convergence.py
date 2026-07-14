from experiments.benchmark_stokes_3d_convergence import run_level


def test_3d_manufactured_stokes_solution_converges():
    coarse = run_level(2)
    medium = run_level(3)
    fine = run_level(4)

    assert coarse['n_tetrahedra'] == 48
    assert (fine['velocity_l2_error'] < medium['velocity_l2_error'] <
            coarse['velocity_l2_error'])
    assert (fine['velocity_h1_seminorm_error'] <
            medium['velocity_h1_seminorm_error'] <
            coarse['velocity_h1_seminorm_error'])
    assert fine['velocity_relative_l2_error'] < 0.2
    assert fine['pressure_relative_l2_error'] < 1.5
    assert max(coarse['linear_relative_residual'],
               medium['linear_relative_residual'],
               fine['linear_relative_residual']) < 1e-10
