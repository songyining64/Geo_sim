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

## Table Stokes Replay
| method | setup_time | solve_time | total_time | iterations | fallback_rate | accepted_rate |
| --- | --- | --- | --- | --- | --- | --- |
| traditional | 0.0187 +- 1.847e-04 | 0.0696 +- 5.172e-04 | 0.0883 +- 3.755e-04 | 100.0000 +- 0.0000 | 0.0000 +- 0.0000 | 1.0000 +- 0.0000 |
| reuse | 0.0191 +- 3.393e-04 | 0.0693 +- 3.796e-04 | 0.0884 +- 6.386e-04 | 100.0000 +- 0.0000 | 0.7500 +- 0.0000 | 0.2500 +- 0.0000 |
| zero_shot | 0.0198 +- 1.998e-04 | 0.0697 +- 3.211e-04 | 0.0895 +- 3.471e-04 | 100.0000 +- 0.0000 | 0.7500 +- 0.0000 | 0.2500 +- 0.0000 |
| adapted | 0.0199 +- 3.463e-04 | 0.0698 +- 5.612e-04 | 0.0897 +- 8.359e-04 | 100.0000 +- 0.0000 | 0.7500 +- 0.0000 | 0.2500 +- 0.0000 |

## Table Stokes Full
| method | wall_time | picard_iterations | nusselt | rms_velocity | valid_physics | velocity_fallbacks | velocity_solves | cg_iters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_direct | 0.2711 +- 5.940e-04 | [4, 4, 4] | 0.9956 +- 0.0070 | 78.8706 +- 1.1802 | yes | - | - | - |
| meta_blocked | 2.1775 +- 0.0100 | - | 0.9956 +- 0.0070 | 78.8706 +- 1.1802 | 1.0000 | 2.0000 | 2.0000 | 0.0000 |

## Table E6 Methods
| method | setup_time | solve_time | total_time | iterations | fallback_rate |
| --- | --- | --- | --- | --- | --- |
| traditional | 0.1129 +- 0.0019 | 0.1885 +- 0.0044 | 0.3014 +- 0.0063 | 100.0000 +- 0.0000 | 0.0000 +- 0.0000 |
| reuse | 0.0141 +- 1.985e-04 | 0.0231 +- 2.822e-04 | 0.0373 +- 4.372e-04 | 10.9000 +- 0.0000 | 0.0000 +- 0.0000 |
| zero_shot | 0.0300 +- 7.719e-05 | 0.0856 +- 2.806e-04 | 0.1157 +- 3.297e-04 | 100.0000 +- 0.0000 | 0.0000 +- 0.0000 |
| adapted | 0.0301 +- 6.618e-05 | 0.0858 +- 5.999e-04 | 0.1159 +- 6.511e-04 | 100.0000 +- 0.0000 | 0.0000 +- 0.0000 |
| neural_amg_setup_only | 0.0219 +- 3.765e-05 | - | - | - | - |
