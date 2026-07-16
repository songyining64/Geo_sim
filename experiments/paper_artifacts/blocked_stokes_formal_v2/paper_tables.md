# Blocked Stokes paper tables

Values are independent-process means. 95% Student-t confidence intervals are reported only when at least two physical seeds are available.

## Scaling

| dimension | mesh_mode | viscosity_mode | n | stokes_dofs | method | total_s_mean | total_s_ci95_half | method_peak_rss_mib_mean | failure_rate | numerical_failure_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | structured | temperature_dependent | 16 | 867 | hypre_boomeramg_fresh | 2.137 | 0.6614 | 175.9 | 0 | 0 |
| 2 | structured | temperature_dependent | 16 | 867 | petsc_gamg_fresh | 2.142 | 0.4474 | 176.2 | 0 | 0 |
| 2 | structured | temperature_dependent | 16 | 867 | pyamg_fresh | 2.186 | 0.4788 | 182.5 | 0 | 0 |
| 2 | structured | temperature_dependent | 32 | 3267 | hypre_boomeramg_fresh | 27 | 1.481 | 202.6 | 0 | 0 |
| 2 | structured | temperature_dependent | 32 | 3267 | petsc_gamg_fresh | 27.34 | 0.355 | 201.9 | 0 | 0 |
| 2 | structured | temperature_dependent | 32 | 3267 | pyamg_fresh | 27.31 | 0.3919 | 206.9 | 0 | 0 |
| 3 | structured | temperature_dependent | 4 | 500 | hypre_boomeramg_fresh | 1.276 | 0.2886 | 173 | 0 | 0 |
| 3 | structured | temperature_dependent | 4 | 500 | petsc_gamg_fresh | 1.283 | 0.3604 | 174.6 | 0 | 0 |
| 3 | structured | temperature_dependent | 4 | 500 | pyamg_fresh | 1.311 | 0.2813 | 180 | 0 | 0 |
| 3 | structured | temperature_dependent | 6 | 1372 | hypre_boomeramg_fresh | 7.366 | 0.4591 | 191.8 | 0 | 0 |
| 3 | structured | temperature_dependent | 6 | 1372 | petsc_gamg_fresh | 7.608 | 1.123 | 191.5 | 0 | 0 |
| 3 | structured | temperature_dependent | 6 | 1372 | pyamg_fresh | 7.506 | 1.062 | 196.4 | 0 | 0 |
| 3 | structured | temperature_dependent | 8 | 2916 | hypre_boomeramg_fresh | 28.91 | 1.17 | 236.2 | 0 | 0 |
| 3 | structured | temperature_dependent | 8 | 2916 | petsc_gamg_fresh | 29.48 | 4.48 | 230.5 | 0 | 0 |
| 3 | structured | temperature_dependent | 8 | 2916 | pyamg_fresh | 28.17 | 0.8377 | 234.2 | 0 | 0 |

## Nonlinear amortized performance

| dimension | mesh_mode | viscosity_mode | n_steps | method | wall_per_step_s_mean | setup_per_step_s_mean | linear_per_picard_s_mean |
|---|---|---|---|---|---|---|---|
| 2 | structured | temperature_dependent | 20 | direct | 0.4234 | 0 | 0.0008082 |
| 2 | structured | temperature_dependent | 20 | hypre_boomeramg_fresh | 0.5136 | 0.003114 | 0.01955 |
| 3 | structured | temperature_dependent | 20 | direct | 3.409 | 0 | 0.00299 |
| 3 | structured | temperature_dependent | 20 | hypre_boomeramg_fresh | 3.8 | 0.01325 | 0.06025 |

## Physics equivalence

| dimension | mesh_mode | viscosity_mode | n_steps | method | physics_equivalence_rate | nusselt_rel_error_mean | temperature_field_rel_error_mean | velocity_field_rel_error_mean |
|---|---|---|---|---|---|---|---|---|
| 2 | structured | temperature_dependent | 20 | direct | 1 | 0 | 0 | 0 |
| 2 | structured | temperature_dependent | 20 | hypre_boomeramg_fresh | 1 | 3.016e-14 | 3.557e-13 | 5.413e-07 |
| 3 | structured | temperature_dependent | 20 | direct | 1 | 0 | 0 | 0 |
| 3 | structured | temperature_dependent | 20 | hypre_boomeramg_fresh | 1 | 6.419e-15 | 5.362e-14 | 3.217e-07 |

## Ablation and paired comparisons

| benchmark | dimension | n | method | backend | hierarchy_policy | schur_velocity_inverse | metric_mean_s | paired_vs_method | paired_difference_mean_s | paired_difference_ci95_half_s | paired_ratio_mean | paired_seed_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scaling | 2 | 16 | hypre_boomeramg_fresh | hypre_boomeramg | fresh | lu | 2.137 | hypre_boomeramg_fresh | 0 | 0 | 1 | 3 |
| scaling | 2 | 16 | petsc_gamg_fresh | petsc_gamg | fresh | lu | 2.142 | hypre_boomeramg_fresh | 0.004514 | 0.214 | 1.014 | 3 |
| scaling | 2 | 16 | pyamg_fresh | pyamg | fresh | lu | 2.186 | hypre_boomeramg_fresh | 0.0484 | 0.1847 | 1.03 | 3 |
| scaling | 2 | 32 | hypre_boomeramg_fresh | hypre_boomeramg | fresh | lu | 27 | hypre_boomeramg_fresh | 0 | 0 | 1 | 3 |
| scaling | 2 | 32 | petsc_gamg_fresh | petsc_gamg | fresh | lu | 27.34 | hypre_boomeramg_fresh | 0.3481 | 1.802 | 1.014 | 3 |
| scaling | 2 | 32 | pyamg_fresh | pyamg | fresh | lu | 27.31 | hypre_boomeramg_fresh | 0.3183 | 1.199 | 1.013 | 3 |
| scaling | 3 | 4 | hypre_boomeramg_fresh | hypre_boomeramg | fresh | lu | 1.276 | hypre_boomeramg_fresh | 0 | 0 | 1 | 3 |
| scaling | 3 | 4 | petsc_gamg_fresh | petsc_gamg | fresh | lu | 1.283 | hypre_boomeramg_fresh | 0.00628 | 0.1096 | 1.007 | 3 |
| scaling | 3 | 4 | pyamg_fresh | pyamg | fresh | lu | 1.311 | hypre_boomeramg_fresh | 0.03442 | 0.08202 | 1.033 | 3 |
| scaling | 3 | 6 | hypre_boomeramg_fresh | hypre_boomeramg | fresh | lu | 7.366 | hypre_boomeramg_fresh | 0 | 0 | 1 | 3 |
| scaling | 3 | 6 | petsc_gamg_fresh | petsc_gamg | fresh | lu | 7.608 | hypre_boomeramg_fresh | 0.242 | 1.481 | 1.037 | 3 |
| scaling | 3 | 6 | pyamg_fresh | pyamg | fresh | lu | 7.506 | hypre_boomeramg_fresh | 0.1394 | 0.6105 | 1.02 | 3 |
| scaling | 3 | 8 | hypre_boomeramg_fresh | hypre_boomeramg | fresh | lu | 28.91 | hypre_boomeramg_fresh | 0 | 0 | 1 | 3 |
| scaling | 3 | 8 | petsc_gamg_fresh | petsc_gamg | fresh | lu | 29.48 | hypre_boomeramg_fresh | 0.5698 | 4.285 | 1.022 | 3 |
| scaling | 3 | 8 | pyamg_fresh | pyamg | fresh | lu | 28.17 | hypre_boomeramg_fresh | -0.7408 | 1.245 | 0.9802 | 3 |
| nonlinear | 2 | 8 | direct | direct | direct | lu | 8.467 | direct | 0 | 0 | 1 | 5 |
| nonlinear | 2 | 8 | hypre_boomeramg_fresh | hypre_boomeramg | fresh | lu | 10.27 | direct | 1.806 | 0.3687 | 1.214 | 5 |
| nonlinear | 3 | 4 | direct | direct | direct | lu | 68.18 | direct | 0 | 0 | 1 | 5 |
| nonlinear | 3 | 4 | hypre_boomeramg_fresh | hypre_boomeramg | fresh | lu | 75.99 | direct | 7.819 | 4.924 | 1.113 | 5 |

## Coverage and failures

| benchmark | dimension | mesh_mode | viscosity_mode | n | method | n_attempts | success_rate | failure_rate | numerical_failure_rate | unavailable_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Blocked Stokes nonlinear | 2 | structured | temperature_dependent | 8 | direct | 10 | 1 | 0 | 0 | 0 |
| Blocked Stokes nonlinear | 2 | structured | temperature_dependent | 8 | hypre_boomeramg_fresh | 10 | 1 | 0 | 0 | 0 |
| Blocked Stokes nonlinear | 3 | structured | temperature_dependent | 4 | direct | 10 | 1 | 0 | 0 | 0 |
| Blocked Stokes nonlinear | 3 | structured | temperature_dependent | 4 | hypre_boomeramg_fresh | 10 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 2 | structured | temperature_dependent | 16 | hypre_boomeramg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 2 | structured | temperature_dependent | 16 | petsc_gamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 2 | structured | temperature_dependent | 16 | pyamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 2 | structured | temperature_dependent | 32 | hypre_boomeramg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 2 | structured | temperature_dependent | 32 | petsc_gamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 2 | structured | temperature_dependent | 32 | pyamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 3 | structured | temperature_dependent | 4 | hypre_boomeramg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 3 | structured | temperature_dependent | 4 | petsc_gamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 3 | structured | temperature_dependent | 4 | pyamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 3 | structured | temperature_dependent | 6 | hypre_boomeramg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 3 | structured | temperature_dependent | 6 | petsc_gamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 3 | structured | temperature_dependent | 6 | pyamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 3 | structured | temperature_dependent | 8 | hypre_boomeramg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 3 | structured | temperature_dependent | 8 | petsc_gamg_fresh | 15 | 1 | 0 | 0 | 0 |
| Blocked Stokes scaling | 3 | structured | temperature_dependent | 8 | pyamg_fresh | 15 | 1 | 0 | 0 | 0 |
