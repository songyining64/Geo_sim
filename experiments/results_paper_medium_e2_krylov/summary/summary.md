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
| 225 | 0.0347 | 0.0058 | 0.0159 | 0.0160 | 0.1265 | 0.0215 | 0.0982 | 2.1674 | 0.0000 |
| 400 | 0.1168 | 0.0184 | 0.0379 | 0.0379 | 0.3140 | 0.0478 | 0.1355 | 3.0798 | 0.0000 |
| 625 | 0.2750 | 0.0390 | 0.0744 | 0.0745 | 0.6123 | 0.0826 | 0.1956 | 3.6930 | 0.0000 |

## Table Stokes Replay
| method | setup_time | solve_time | total_time | iterations | fallback_rate | accepted_rate |
| --- | --- | --- | --- | --- | --- | --- |
| traditional | 0.0193 | 0.0733 | 0.0926 | 100.0000 | 0.0000 | 1.0000 |
| reuse | 0.0195 | 0.0728 | 0.0923 | 100.0000 | 0.7500 | 0.2500 |
| zero_shot | 0.0200 | 0.0732 | 0.0932 | 100.0000 | 0.7500 | 0.2500 |
| adapted | 0.0200 | 0.0719 | 0.0919 | 100.0000 | 0.7500 | 0.2500 |

## Table Stokes Full
| method | wall_time | picard_iterations | nusselt | rms_velocity | valid_physics |
| --- | --- | --- | --- | --- | --- |
| reference_direct | 0.2810 | 4 | 0.9894 | 78.6279 | yes |
| meta_blocked | 76.9693 | 4 | 1.0072 | 76.3083 | yes |
