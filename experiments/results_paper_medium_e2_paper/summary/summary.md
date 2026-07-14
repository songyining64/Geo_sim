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
| 225 | 0.0672 +- 0.0062 | 0.0127 +- 0.0016 | 0.0322 +- 0.0025 | 0.0324 +- 0.0026 | 0.2489 +- 0.0245 | 0.0467 +- 0.0044 | 0.1917 +- 0.0181 | 2.0738 | 0.0000 +- 0.0000 |
| 400 | 0.1743 +- 0.0093 | 0.0242 +- 5.746e-04 | 0.0553 +- 0.0015 | 0.0549 +- 0.0022 | 0.4752 +- 0.0300 | 0.0659 +- 0.0028 | 0.2047 +- 0.0097 | 3.1755 | 0.0000 +- 0.0000 |
| 625 | 0.5166 +- 0.0421 | 0.0754 +- 0.0063 | 0.1472 +- 0.0154 | 0.1413 +- 0.0129 | 1.1630 +- 0.1146 | 0.1567 +- 0.0172 | 0.3697 +- 0.0393 | 3.6568 | 0.0000 +- 0.0000 |

## Table Stokes Replay
| method | setup_time | solve_time | total_time | iterations | fallback_rate | accepted_rate |
| --- | --- | --- | --- | --- | --- | --- |
| traditional | 0.0258 +- 0.0044 | 0.0970 +- 0.0164 | 0.1229 +- 0.0208 | 100.0000 +- 0.0000 | 0.0000 +- 0.0000 | 1.0000 +- 0.0000 |
| reuse | 0.0262 +- 0.0044 | 0.0972 +- 0.0167 | 0.1234 +- 0.0211 | 100.0000 +- 0.0000 | 0.7500 +- 0.0000 | 0.2500 +- 0.0000 |
| zero_shot | 0.0270 +- 0.0044 | 0.0950 +- 0.0156 | 0.1221 +- 0.0200 | 100.0000 +- 0.0000 | 0.7500 +- 0.0000 | 0.2500 +- 0.0000 |
| adapted | 0.0271 +- 0.0046 | 0.0966 +- 0.0166 | 0.1237 +- 0.0212 | 100.0000 +- 0.0000 | 0.7500 +- 0.0000 | 0.2500 +- 0.0000 |

## Table Stokes Full
| method | wall_time | picard_iterations | nusselt | rms_velocity | valid_physics | velocity_fallbacks | velocity_solves | cg_iters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_direct | 0.4353 +- 0.1080 | [4, 4, 4] | 0.9956 +- 0.0070 | 78.8706 +- 1.1802 | yes | - | - | - |
| meta_blocked | 2.9970 +- 0.4520 | - | 0.9956 +- 0.0070 | 78.8706 +- 1.1802 | 1.0000 | 2.0000 | 2.0000 | 0.0000 |
