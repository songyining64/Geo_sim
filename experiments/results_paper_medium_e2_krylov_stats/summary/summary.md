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
| 225 | 0.0314 | 0.0056 | 0.0278 | 0.0284 | 0.1153 | 0.0213 | 0.1507 | 1.1059 | 0.0000 |
| 400 | 0.1099 | 0.0172 | 0.0635 | 0.0637 | 0.2946 | 0.0461 | 0.2034 | 1.7267 | 0.0000 |
| 625 | 0.2658 | 0.0385 | 0.1345 | 0.1343 | 0.5886 | 0.0821 | 0.3370 | 1.9784 | 0.0000 |

## Table Stokes Replay
| method | setup_time | solve_time | total_time | iterations | fallback_rate | accepted_rate |
| --- | --- | --- | --- | --- | --- | --- |
| traditional | 0.0193 | 0.0714 | 0.0907 | 100.0000 | 0.0000 | 1.0000 |
| reuse | 0.0192 | 0.0732 | 0.0925 | 100.0000 | 0.7500 | 0.2500 |
| zero_shot | 0.0199 | 0.0715 | 0.0914 | 100.0000 | 0.7500 | 0.2500 |
| adapted | 0.0197 | 0.0715 | 0.0912 | 100.0000 | 0.7500 | 0.2500 |

## Table Stokes Full
| method | wall_time | picard_iterations | nusselt | rms_velocity | valid_physics | velocity_fallbacks | velocity_solves | cg_iters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_direct | 0.2732 | 4 | 0.9894 | 78.6279 | yes | - | - | - |
| meta_blocked | 74.3638 | 4 | 0.9877 | 81.6890 | yes | 83 | 83 | 0 |
