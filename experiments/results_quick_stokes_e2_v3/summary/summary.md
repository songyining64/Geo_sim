# Experiment Summary

## Config
- `meta_epochs`: 10
- `meta_tasks`: 30
- `matrix_sizes`: [100, 225, 400]
- `contrasts`: [1000.0, 100000.0]

## Table E2 Synthetic
| matrix_size | trad_setup | reuse_setup | zero_shot_setup | adapt_setup | trad_total | reuse_total | adapt_total | adapt_speedup | adapt_fallback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 1.962e-05 | 1.843e-04 | 2.189e-05 | 2.313e-05 | 1.162e-04 | 2.958e-04 | 1.752e-04 | 0.8485 | 0.0000 |
| 225 | 0.0356 | 0.0046 | 0.0128 | 0.0128 | 0.1314 | 0.0169 | 0.0854 | 2.7898 | 0.0000 |
| 400 | 0.1139 | 0.0141 | 0.0293 | 0.0292 | 0.3043 | 0.0373 | 0.1147 | 3.8953 | 0.0000 |

## Table Stokes Replay
| method | setup_time | solve_time | total_time | iterations | fallback_rate | accepted_rate |
| --- | --- | --- | --- | --- | --- | --- |
| traditional | 1.915e-05 | 7.192e-05 | 9.441e-05 | 1.0000 | 0.0000 | 1.0000 |
| reuse | 3.825e-04 | 6.747e-05 | 4.537e-04 | 1.0000 | 0.6667 | 0.3333 |
| zero_shot | 0.0019 | 8.663e-05 | 0.0020 | 1.0000 | 0.6667 | 0.3333 |
| adapted | 0.0025 | 7.367e-05 | 0.0026 | 1.0000 | 0.6667 | 0.3333 |

## Table Stokes Full
| method | wall_time | picard_iterations | nusselt | rms_velocity | valid_physics |
| --- | --- | --- | --- | --- | --- |
| reference_direct | 0.0558 | 3 | 1.0061 | 1.8257 | yes |
| meta_blocked | 0.0789 | 3 | 0.9994 | 1.6900 | yes |
