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

## Table E2 Synthetic
| matrix_size | trad_setup | reuse_setup | zero_shot_setup | adapt_setup | trad_total | reuse_total | adapt_total | adapt_speedup | adapt_fallback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 225 | 0.0342 +- 0.0019 | 0.0057 +- 1.489e-04 | 0.0137 +- 6.651e-05 | 0.0138 +- 1.308e-04 | 0.1262 +- 0.0080 | 0.0212 +- 4.105e-04 | 0.0884 +- 4.642e-04 | 2.4814 | 0.0000 +- 0.0000 |
| 400 | 0.1120 +- 0.0020 | 0.0172 +- 1.044e-04 | 0.0322 +- 4.209e-05 | 0.0322 +- 8.012e-05 | 0.3015 +- 0.0072 | 0.0461 +- 1.772e-04 | 0.1221 +- 2.415e-04 | 3.4828 | 0.0000 +- 0.0000 |
| 625 | 0.2719 +- 0.0020 | 0.0384 +- 2.069e-04 | 0.0634 +- 3.053e-04 | 0.0635 +- 3.635e-04 | 0.6065 +- 0.0082 | 0.0816 +- 4.210e-04 | 0.1705 +- 6.586e-04 | 4.2823 | 0.0000 +- 0.0000 |

## Table Stokes Replay
| method | setup_time | solve_time | total_time | iterations | fallback_rate | accepted_rate |
| --- | --- | --- | --- | --- | --- | --- |
| traditional | 0.0189 +- 4.579e-04 | 0.0722 +- 0.0024 | 0.0911 +- 0.0029 | 100.0000 +- 0.0000 | 0.0000 +- 0.0000 | 1.0000 +- 0.0000 |
| reuse | 0.0194 +- 6.931e-04 | 0.0719 +- 0.0022 | 0.0913 +- 0.0029 | 100.0000 +- 0.0000 | 0.7500 +- 0.0000 | 0.2500 +- 0.0000 |
| zero_shot | 0.0208 +- 0.0012 | 0.0720 +- 0.0031 | 0.0929 +- 0.0031 | 100.0000 +- 0.0000 | 0.7500 +- 0.0000 | 0.2500 +- 0.0000 |
| adapted | 0.0202 +- 3.590e-04 | 0.0718 +- 0.0022 | 0.0919 +- 0.0024 | 100.0000 +- 0.0000 | 0.7500 +- 0.0000 | 0.2500 +- 0.0000 |

## Table Stokes Full
| method | wall_time | picard_iterations | nusselt | rms_velocity | valid_physics | velocity_fallbacks | velocity_solves | cg_iters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_direct | 0.2813 +- 0.0127 | [4, 4, 4] | 0.9956 +- 0.0070 | 78.8706 +- 1.1802 | yes | - | - | - |
| meta_blocked | 0.9270 +- 0.0683 | - | 0.9956 +- 0.0070 | 78.8706 +- 1.1802 | 1.0000 | 0.0000 | 2.0000 | 153.0000 |
