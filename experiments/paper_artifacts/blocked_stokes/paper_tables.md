# Blocked Stokes paper tables

Values are independent-process means. 95% Student-t confidence intervals are reported only when at least two physical seeds are available.

## Scaling

| dimension | mesh_mode | viscosity_mode | n | stokes_dofs | method | total_s_mean | total_s_ci95_half | method_peak_rss_mib_mean | failure_rate | numerical_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | structured | temperature_dependent | 16 | 867 | hypre_boomeramg_fresh | 2.119 | 0.1212 | 176 | 0 | 0 |
| 2 | structured | temperature_dependent | 16 | 867 | petsc_gamg_fresh | 2.176 | 0.156 | 176.1 | 0 | 0 |
| 2 | structured | temperature_dependent | 16 | 867 | pyamg_fresh | 2.205 | 0.1571 | 182.2 | 0 | 0 |
| 2 | structured | temperature_dependent | 32 | 3267 | hypre_boomeramg_fresh | 30.3 | 2.471 | 202.3 | 0 | 0 |
| 2 | structured | temperature_dependent | 32 | 3267 | petsc_gamg_fresh | 30.86 | 5.463 | 201.1 | 0 | 0 |
| 2 | structured | temperature_dependent | 32 | 3267 | pyamg_fresh | 31.68 | 6.155 | 204.7 | 0 | 0 |
| 3 | structured | temperature_dependent | 4 | 500 | hypre_boomeramg_fresh | 1.17 | 0.105 | 174.1 | 0 | 0 |
| 3 | structured | temperature_dependent | 4 | 500 | petsc_gamg_fresh | 1.169 | 0.04607 | 174.5 | 0 | 0 |
| 3 | structured | temperature_dependent | 4 | 500 | pyamg_fresh | 1.21 | 0.06574 | 179.9 | 0 | 0 |
| 3 | structured | temperature_dependent | 6 | 1372 | hypre_boomeramg_fresh | 7.951 | 1.142 | 191.7 | 0 | 0 |
| 3 | structured | temperature_dependent | 6 | 1372 | petsc_gamg_fresh | 8.072 | 1.387 | 191.4 | 0 | 0 |
| 3 | structured | temperature_dependent | 6 | 1372 | pyamg_fresh | 7.943 | 1.015 | 196.2 | 0 | 0 |
| 3 | structured | temperature_dependent | 8 | 2916 | hypre_boomeramg_fresh | 31.59 | 5.458 | 234 | 0 | 0 |
| 3 | structured | temperature_dependent | 8 | 2916 | petsc_gamg_fresh | 30.03 | 6.349 | 230.6 | 0 | 0 |
| 3 | structured | temperature_dependent | 8 | 2916 | pyamg_fresh | 29.33 | 8.046 | 232.8 | 0 | 0 |

## Nonlinear amortized performance

| dimension | mesh_mode | viscosity_mode | n_steps | method | wall_per_step_s_mean | setup_per_step_s_mean | linear_per_picard_s_mean |
|---|---|---|---|---|---|---|---|
| 2 | structured | temperature_dependent | 20 | direct | 0.4795 | 0 | 0.0009133 |
| 2 | structured | temperature_dependent | 20 | hypre_boomeramg_fresh | 0.7585 | 0.004585 | 0.02883 |
| 3 | structured | temperature_dependent | 20 | direct | 4.038 | 0 | 0.003634 |
| 3 | structured | temperature_dependent | 20 | hypre_boomeramg_fresh | 4.578 | 0.01631 | 0.07179 |

## Physics equivalence

| dimension | mesh_mode | viscosity_mode | n_steps | method | physics_equivalence_rate | nusselt_rel_error_mean | temperature_field_rel_error_mean | velocity_field_rel_error_mean |
|---|---|---|---|---|---|---|---|---|
| 2 | structured | temperature_dependent | 20 | direct | 1 | 0 | 0 | 0 |
| 2 | structured | temperature_dependent | 20 | hypre_boomeramg_fresh | 1 | 3.016e-14 | 3.557e-13 | 5.413e-07 |
| 3 | structured | temperature_dependent | 20 | direct | 1 | 0 | 0 | 0 |
| 3 | structured | temperature_dependent | 20 | hypre_boomeramg_fresh | 1 | 6.419e-15 | 5.362e-14 | 3.217e-07 |

## Coverage and failures

| benchmark | dimension | mesh_mode | viscosity_mode | n | method | n_attempts | success_rate | failure_rate | numerical_failure_rate | unavailable_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Blocked Stokes nonlinear (2D + 3D) | 2 | structured | temperature_dependent | 8 | direct | 10 | 1 | 0 | 0 | 0 |
| Blocked Stokes nonlinear (2D + 3D) | 2 | structured | temperature_dependent | 8 | hypre_boomeramg_fresh | 10 | 1 | 0 | 0 | 0 |
| Blocked Stokes nonlinear (2D + 3D) | 3 | structured | temperature_dependent | 4 | direct | 10 | 1 | 0 | 0 | 0 |
| Blocked Stokes nonlinear (2D + 3D) | 3 | structured | temperature_dependent | 4 | hypre_boomeramg_fresh | 10 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 2 | structured | temperature_dependent | 16 | hypre_boomeramg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 2 | structured | temperature_dependent | 16 | petsc_gamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 2 | structured | temperature_dependent | 16 | pyamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 2 | structured | temperature_dependent | 32 | hypre_boomeramg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 2 | structured | temperature_dependent | 32 | petsc_gamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 2 | structured | temperature_dependent | 32 | pyamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 3 | structured | temperature_dependent | 4 | hypre_boomeramg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 3 | structured | temperature_dependent | 4 | petsc_gamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 3 | structured | temperature_dependent | 4 | pyamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 3 | structured | temperature_dependent | 6 | hypre_boomeramg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 3 | structured | temperature_dependent | 6 | petsc_gamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 3 | structured | temperature_dependent | 6 | pyamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 3 | structured | temperature_dependent | 8 | hypre_boomeramg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 3 | structured | temperature_dependent | 8 | petsc_gamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling (2D + 3D) | 3 | structured | temperature_dependent | 8 | pyamg_fresh | 15 | 1 | 0 | 0 | 0 |
