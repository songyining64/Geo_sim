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
| 225 | 0.0337 | 0.0056 | 0.0139 | 0.0138 | 0.1239 | 0.0212 | 0.0881 | 2.4363 | 0.0000 |
| 400 | 0.1162 | 0.0174 | 0.0327 | 0.0329 | 0.3138 | 0.0461 | 0.1234 | 3.5349 | 0.0000 |
| 625 | 0.2914 | 0.0448 | 0.0706 | 0.0705 | 0.6386 | 0.0898 | 0.1816 | 4.1312 | 0.0000 |

## Table Stokes Replay
| method | setup_time | solve_time | total_time | iterations | fallback_rate | accepted_rate |
| --- | --- | --- | --- | --- | --- | --- |
| traditional | 0.0189 | 0.0703 | 0.0892 | 100.0000 | 0.0000 | 1.0000 |
| reuse | 0.0192 | 0.0701 | 0.0894 | 100.0000 | 0.7500 | 0.2500 |
| zero_shot | 0.0200 | 0.0702 | 0.0902 | 100.0000 | 0.7500 | 0.2500 |
| adapted | 0.0202 | 0.0700 | 0.0902 | 100.0000 | 0.7500 | 0.2500 |

## Table Stokes Full
| method | wall_time | picard_iterations | nusselt | rms_velocity | valid_physics |
| --- | --- | --- | --- | --- | --- |
| reference_direct | 0.2728 | 4 | 0.9894 | 78.6279 | yes |
| meta_blocked | 44.8565 | 4 | 1.0130 | inf | no |
