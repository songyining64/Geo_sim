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
| 225 | 0.0341 | 0.0057 | 0.0160 | 0.0161 | 0.1248 | 0.0214 | 0.0978 | 2.1237 | 0.0000 |
| 400 | 0.1127 | 0.0171 | 0.0369 | 0.0370 | 0.3034 | 0.0460 | 0.1335 | 3.0448 | 0.0000 |
| 625 | 0.2749 | 0.0388 | 0.0747 | 0.0744 | 0.6074 | 0.0816 | 0.1942 | 3.6946 | 0.0000 |

## Table Stokes Replay
| method | setup_time | solve_time | total_time | iterations | fallback_rate | accepted_rate |
| --- | --- | --- | --- | --- | --- | --- |
| traditional | 0.0209 | 0.0780 | 0.0989 | 100.0000 | 0.0000 | 1.0000 |
| reuse | 0.0207 | 0.0752 | 0.0959 | 100.0000 | 0.7500 | 0.2500 |
| zero_shot | 0.0212 | 0.0761 | 0.0973 | 100.0000 | 0.7500 | 0.2500 |
| adapted | 0.0207 | 0.0762 | 0.0969 | 100.0000 | 0.7500 | 0.2500 |

## Table Stokes Full
| method | wall_time | picard_iterations | nusselt | rms_velocity | valid_physics | velocity_fallbacks | velocity_solves | cg_iters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_direct | 0.3111 | 4 | 0.9894 | 78.6279 | yes | - | - | - |
| meta_blocked | 2.2419 | 4 | 1.0098 | 83.6187 | yes | 2 | 2 | 0 |
