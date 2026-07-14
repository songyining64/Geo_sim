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
| 225 | 0.0369 | 0.0058 | 0.0171 | 0.0176 | 0.1358 | 0.0219 | 0.0936 | 2.0967 | 0.0000 |
| 400 | 0.1164 | 0.0173 | 0.0445 | 0.0492 | 0.3166 | 0.0468 | 0.1810 | 2.3639 | 0.0000 |
| 625 | 0.2788 | 0.0396 | 0.0964 | 0.0978 | 0.6162 | 0.0846 | 0.2550 | 2.8508 | 0.0000 |

## Table Stokes Replay
| method | setup_time | solve_time | total_time | iterations | fallback_rate | accepted_rate |
| --- | --- | --- | --- | --- | --- | --- |
| traditional | 0.0196 | 0.0729 | 0.0926 | 100.0000 | 0.0000 | 1.0000 |
| reuse | 0.0194 | 0.0723 | 0.0917 | 100.0000 | 0.7500 | 0.2500 |
| zero_shot | 0.0248 | 0.0706 | 0.0955 | 100.0000 | 0.7500 | 0.2500 |
| adapted | 0.0244 | 0.0724 | 0.0968 | 100.0000 | 0.7500 | 0.2500 |

## Table Stokes Full
| method | wall_time | picard_iterations | nusselt | rms_velocity | valid_physics |
| --- | --- | --- | --- | --- | --- |
| reference_direct | 0.2775 | 4 | 0.9894 | 78.6279 | yes |
| meta_blocked | 49.7597 | 4 | 1.0017 | 75.1343 | yes |
