# Size-32 Schur-LU Optimized Variant

Configuration: `32x32`, four Picard iterations, seed `0`, `adapt_steps=1`. The pressure Schur operator uses one cached sparse LU factorization of the velocity block per Picard step; final velocity solves still use AMG-preconditioned Krylov.

| Metric | Traditional blocked | Meta blocked | Change |
| --- | ---: | ---: | ---: |
| Hierarchy/adaptation setup | 5.8260 s | 4.9828 s | 1.17x speedup |
| Schur velocity LU setup | 0.0364 s | 0.0322 s | included separately |
| Total online preconditioner setup | 5.8625 s | 5.0151 s | 1.17x speedup |
| Velocity Krylov iterations | 1835 | 1625 | ratio 0.886 |
| Full solve wall time | 22.5476 s | 18.7429 s | -16.9% |
| Pressure Krylov iterations | 205 | 205 | unchanged, cheap LU applications |
| Full/velocity direct fallbacks | 0 | 0 | stable |
| Velocity field relative error | reference | 4.23e-8 | acceptable |

Compared with the pure iterative Schur path (`242.3 s` traditional, `244.3 s` Meta), cached Schur-LU removes repeated inner velocity Krylov solves and makes the `32x32` production run practical. This optimized path must be reported separately from the pure iterative method.
