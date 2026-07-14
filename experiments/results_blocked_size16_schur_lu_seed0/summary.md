# Size-16 Schur-LU Optimized Variant

Configuration: `16x16`, four Picard iterations, seed `0`, `adapt_steps=1`. The pressure Schur operator uses one cached sparse LU factorization of the velocity block per Picard step; final velocity solves still use AMG-preconditioned Krylov.

| Metric | Traditional blocked | Meta blocked | Change |
| --- | ---: | ---: | ---: |
| Hierarchy/adaptation setup | 0.7112 s | 0.5078 s | 1.40x speedup |
| Schur velocity LU setup | 0.0058 s | 0.0140 s | included separately |
| Total online preconditioner setup | 0.7170 s | 0.5218 s | 1.37x speedup |
| Velocity Krylov iterations | 1386 | 672 | ratio 0.485 |
| Full solve wall time | 5.0029 s | 2.9726 s | -40.6% |
| Full/velocity direct fallbacks | 0 | 0 | stable |
| Velocity field relative error | reference | 7.53e-9 | equivalent |

This is an optimized Schur variant, not the pure iterative Schur path. Sparse LU is used only inside Schur applications; the reported Meta-AMG hierarchy remains active in the final velocity solves.
