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
| traditional | 0.0321 | 0.0761 | 0.1082 | 100.0000 | 0.0000 | 1.0000 |
| reuse | 0.0200 | 0.0726 | 0.0926 | 100.0000 | 0.7500 | 0.2500 |
| zero_shot | 0.0205 | 0.0750 | 0.0955 | 100.0000 | 0.7500 | 0.2500 |
| adapted | 0.0204 | 0.0713 | 0.0917 | 100.0000 | 0.7500 | 0.2500 |

## Table Stokes Full
| method | wall_time | picard_iterations | nusselt | rms_velocity | valid_physics |
| --- | --- | --- | --- | --- | --- |
| reference_direct | 0.2714 | 4 | 0.9894 | 78.6279 | yes |
| meta_blocked | 45.1804 | 4 | 1.0005 | inf | no |

## Table E6 Methods
| method | setup_time | solve_time | total_time | iterations | fallback_rate |
| --- | --- | --- | --- | --- | --- |
| traditional | 0.1155 | 0.1979 | 0.3134 | 100.0000 | 0.0000 |
| reuse | 0.0142 | 0.0233 | 0.0375 | 10.9000 | 0.0000 |
| zero_shot | 0.0613 | 0.1374 | 0.1988 | 100.0000 | 0.0000 |
| adapted | 0.0615 | 0.1393 | 0.2009 | 99.8000 | 0.0000 |
| neural_amg_setup_only | 0.0218 | - | - | - | - |
