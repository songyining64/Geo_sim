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

## Table E2 Synthetic
| matrix_size | trad_setup | reuse_setup | zero_shot_setup | adapt_setup | trad_total | reuse_total | adapt_total | adapt_speedup | adapt_fallback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 225 | 0.0337 +- 0.0017 | 0.0056 +- 7.061e-05 | 0.0240 +- 0.0016 | 0.0232 +- 0.0024 | 0.1196 +- 0.0077 | 0.0203 +- 9.771e-05 | 0.0817 +- 0.0078 | 1.4505 | 0.2500 +- 0.1021 |
| 400 | 0.1112 +- 0.0019 | 0.0172 +- 2.333e-04 | 0.0763 +- 0.0043 | 0.0757 +- 0.0035 | 0.2943 +- 0.0081 | 0.0453 +- 5.156e-04 | 0.2002 +- 0.0093 | 1.4688 | 0.5417 +- 0.0589 |
| 625 | 0.2631 +- 0.0039 | 0.0377 +- 8.984e-04 | 0.1909 +- 0.0362 | 0.1997 +- 0.0429 | 0.5833 +- 0.0016 | 0.0798 +- 0.0013 | 0.4397 +- 0.0965 | 1.3174 | 0.6667 +- 0.2125 |

## Table Stokes Replay
| method | setup_time | solve_time | total_time | iterations | fallback_rate | accepted_rate | relative_residual_mean | relative_residual_max | rebuild_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| traditional | 0.0191 +- 5.820e-04 | 0.0654 +- 8.141e-04 | 0.0847 +- 0.0014 | 100.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 6.465e-16 +- 1.482e-17 | 7.460e-16 +- 9.877e-18 | 1.0000 +- 0.0000 |
| reuse | 0.0194 +- 4.644e-04 | 0.0656 +- 0.0010 | 0.0853 +- 0.0015 | 100.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.2500 +- 0.0000 | 6.465e-16 +- 1.482e-17 | 7.460e-16 +- 9.877e-18 | 0.2500 +- 0.0000 |
| periodic_rebuild | 0.0189 +- 2.371e-04 | 0.0650 +- 6.505e-04 | 0.0841 +- 9.064e-04 | 100.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.5000 +- 0.0000 | 6.465e-16 +- 1.482e-17 | 7.460e-16 +- 9.877e-18 | 0.5000 +- 0.0000 |
| change_aware_reuse | 0.0186 +- 2.050e-04 | 0.0686 +- 0.0040 | 0.0875 +- 0.0041 | 100.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 6.465e-16 +- 1.482e-17 | 7.460e-16 +- 9.877e-18 | 1.0000 +- 0.0000 |
| zero_shot | 0.0204 +- 0.0010 | 0.0655 +- 7.175e-04 | 0.0862 +- 0.0011 | 100.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.2500 +- 0.0000 | 6.465e-16 +- 1.482e-17 | 7.460e-16 +- 9.877e-18 | 0.2500 +- 0.0000 |
| adapted | 0.0197 +- 7.052e-05 | 0.0651 +- 8.467e-04 | 0.0851 +- 9.279e-04 | 100.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.2500 +- 0.0000 | 6.465e-16 +- 1.482e-17 | 7.460e-16 +- 9.877e-18 | 0.2500 +- 0.0000 |

## Table Stokes Full
| method | wall_time | picard_iterations | nusselt | rms_velocity | valid_physics | velocity_fallbacks | velocity_solves | velocity_krylov_iters | linear_relative_residual | nusselt_relative_error | rms_velocity_relative_error | velocity_field_relative_error | blocked_solver_valid | direct_fallback_time | pressure_iters | pressure_fallbacks | pressure_pc_fallbacks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_direct | 0.2772 +- 0.0067 | [4, 4, 4] | 0.9956 +- 0.0070 | 78.8706 +- 1.1802 | yes | - | - | - | 1.932e-13 +- 1.555e-14 | 0.0000 | 0.0000 | 0.0000 | yes | - | - | - | - |
| meta_blocked | 2.1433 +- 0.0286 | - | 0.9956 +- 0.0070 | 78.8706 +- 1.1802 | 1.0000 | 0.0000 | 79.0000 | 3.929e+03 | 2.561e-09 +- 5.744e-10 | 0.0000 +- 0.0000 | 2.351e-09 +- 4.922e-10 | 3.743e-09 +- 9.411e-10 | 1.0000 | 0.0000 | 60.6667 | 0.0000 | 0.0000 |
