# Experiment Summary

## Config
- `preset`: paper_medium
- `meta_epochs`: 20
- `meta_tasks`: 60
- `n_runs`: 3
- `sequence_length`: 8
- `matrix_sizes`: [225, 400, 625]
- `contrasts`: [100.0, 10000.0, 1000000.0]
- `stokes_nx`: 8
- `stokes_ny`: 8
- `stokes_picard_iterations`: 4
- `stokes_rayleigh`: 10000.0
- `stokes_viscosity_contrast`: 100.0
- `periodic_rebuild_interval`: 3
- `matrix_change_threshold`: 0.1
- `nusselt_relative_tolerance`: 0.02
- `rms_velocity_relative_tolerance`: 0.05
- `velocity_field_relative_tolerance`: 0.1
- `linear_residual_tolerance`: 1e-06

## Table Stokes Replay
| method | setup_time | solve_time | total_time | iterations | fallback_rate | accepted_rate | relative_residual_mean | relative_residual_max | rebuild_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| traditional | 0.0187 +- 1.026e-04 | 0.0658 +- 6.395e-04 | 0.0848 +- 7.347e-04 | 100.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 6.465e-16 +- 1.482e-17 | 7.460e-16 +- 9.877e-18 | 1.0000 +- 0.0000 |
| reuse | 0.0191 +- 3.119e-04 | 0.0659 +- 7.703e-04 | 0.0852 +- 0.0011 | 100.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.2500 +- 0.0000 | 6.465e-16 +- 1.482e-17 | 7.460e-16 +- 9.877e-18 | 0.2500 +- 0.0000 |
| periodic_rebuild | 0.0192 +- 3.275e-04 | 0.0662 +- 5.304e-04 | 0.0856 +- 7.469e-04 | 100.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.5000 +- 0.0000 | 6.465e-16 +- 1.482e-17 | 7.460e-16 +- 9.877e-18 | 0.5000 +- 0.0000 |
| change_aware_reuse | 0.0188 +- 1.918e-04 | 0.0657 +- 4.459e-04 | 0.0848 +- 6.372e-04 | 100.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 6.465e-16 +- 1.482e-17 | 7.460e-16 +- 9.877e-18 | 1.0000 +- 0.0000 |
| zero_shot | 0.0197 +- 9.908e-05 | 0.0653 +- 1.987e-04 | 0.0853 +- 2.624e-04 | 100.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.2500 +- 0.0000 | 6.465e-16 +- 1.482e-17 | 7.460e-16 +- 9.877e-18 | 0.2500 +- 0.0000 |
| adapted | 0.0197 +- 1.249e-04 | 0.0655 +- 3.283e-04 | 0.0854 +- 3.516e-04 | 100.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.2500 +- 0.0000 | 6.465e-16 +- 1.482e-17 | 7.460e-16 +- 9.877e-18 | 0.2500 +- 0.0000 |

## Table Stokes Full
| method | wall_time | picard_iterations | nusselt | rms_velocity | valid_physics | velocity_fallbacks | velocity_solves | velocity_krylov_iters | linear_relative_residual | nusselt_relative_error | rms_velocity_relative_error | velocity_field_relative_error | blocked_solver_valid | direct_fallback_time | pressure_iters | pressure_fallbacks | pressure_pc_fallbacks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_direct | 0.2726 +- 0.0027 | [4, 4, 4] | 0.9956 +- 0.0070 | 78.8706 +- 1.1802 | yes | - | - | - | 1.932e-13 +- 1.555e-14 | 0.0000 | 0.0000 | 0.0000 | yes | - | - | - | - |
| meta_blocked | 2.0850 +- 0.0060 | - | 0.9956 +- 0.0070 | 78.8706 +- 1.1802 | 1.0000 | 0.0000 | 78.3333 | 3.831e+03 | 2.701e-09 +- 6.634e-10 | 0.0000 +- 0.0000 | 2.277e-09 +- 2.907e-10 | 4.150e-09 +- 8.868e-10 | 1.0000 | 0.0000 | 60.0000 | 0.0000 | 0.0000 |

## Table E6 Methods
| method | setup_time | solve_time | total_time | iterations | fallback_rate | relative_residual_max | rebuild_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| traditional | 0.1113 +- 0.0020 | 0.1814 +- 0.0054 | 0.2929 +- 0.0074 | 100.0000 +- 0.0000 | 0.2333 +- 0.0943 | 6.404e-08 +- 2.531e-08 | 1.0000 +- 0.0000 |
| reuse | 0.0139 +- 1.474e-04 | 0.0225 +- 3.037e-04 | 0.0365 +- 4.504e-04 | 10.9000 +- 0.0000 | 0.1000 +- 1.388e-17 | 2.413e-16 +- 1.386e-17 | 0.1000 +- 1.388e-17 |
| periodic_rebuild | 0.0462 +- 8.297e-04 | 0.0750 +- 0.0023 | 0.1213 +- 0.0031 | 40.6000 +- 0.0000 | 0.1333 +- 0.0471 | 5.597e-08 +- 3.185e-08 | 0.4000 +- 5.551e-17 |
| change_aware_reuse | 0.1080 +- 0.0040 | 0.1766 +- 0.0070 | 0.2848 +- 0.0108 | 96.7000 +- 4.6669 | 0.2333 +- 0.0943 | 6.397e-08 +- 2.536e-08 | 0.9667 +- 0.0471 |
| zero_shot | 0.0770 +- 0.0083 | 0.1287 +- 0.0091 | 0.2058 +- 0.0174 | 86.3667 +- 1.3021 | 0.6000 +- 0.0816 | 9.765e-07 +- 1.547e-08 | 0.1000 +- 1.388e-17 |
| adapted | 0.0773 +- 0.0079 | 0.1318 +- 0.0089 | 0.2092 +- 0.0168 | 88.7000 +- 1.0198 | 0.6000 +- 0.0816 | 9.943e-07 +- 2.971e-09 | 0.1000 +- 1.388e-17 |
| neural_amg_setup_only | 0.0216 +- 4.946e-05 | - | - | - | - | - | - |
