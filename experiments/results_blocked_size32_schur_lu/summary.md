# Size-32 Schur-LU Three-Seed Result

Configuration: `32x32`, four Picard iterations, `Ra=1e4`, viscosity contrast `1e2`, seeds `0/1/2`, `adapt_steps=1`. Schur applications use one cached sparse velocity-block LU per Picard step; final velocity solves remain AMG-preconditioned Krylov.

| Metric | Traditional blocked | Meta blocked | Change |
| --- | ---: | ---: | ---: |
| Hierarchy/adaptation setup | 5.9005 +- 0.1410 s | 3.6384 +- 0.9507 s | 1.72x speedup |
| Schur velocity setup | 0.0343 +- 0.0020 s | 0.0319 +- 0.0069 s | negligible |
| Total online preconditioner setup | 5.9348 +- 0.1392 s | 3.6704 +- 0.9509 s | 1.72x speedup |
| Velocity Krylov iterations | 1933.0 +- 69.9 | 1802.7 +- 153.3 | ratio 0.932 |
| Full solve wall time | 21.8445 +- 0.5059 s | 16.0630 +- 1.9261 s | -26.6% |
| Pressure Krylov iterations | 201.3 +- 5.9 | 201.3 +- 5.9 | unchanged |
| Full/velocity/Schur-LU fallbacks | 0 | 0 | stable for all seeds |
| Velocity field relative error | reference | 3.64e-8 +- 7.45e-9 | acceptable |

This is the current large-scale production result. It supports hierarchy-transfer setup and end-to-end wall-time improvements, but it must be labeled as the Schur-optimized variant because sparse LU is used inside pressure Schur applications.
