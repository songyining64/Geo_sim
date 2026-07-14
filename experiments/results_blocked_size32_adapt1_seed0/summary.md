# Size-32 Blocked Stokes Smoke Result

Configuration: `32x32`, four Picard iterations, `Ra=1e4`, viscosity contrast `1e2`, seed `0`, `adapt_steps=1`.

| Metric | Traditional blocked | Meta blocked | Change |
| --- | ---: | ---: | ---: |
| Online setup including adaptation | 5.8413 s | 5.1115 s | 1.14x speedup |
| Velocity Krylov iterations | 52009 | 45539 | ratio 0.876 |
| Full solve wall time | 242.28 s | 244.26 s | -0.8% slower |
| Full direct fallbacks | 0 | 0 | unchanged |
| Velocity field relative error | reference | 4.92e-8 | acceptable |

Interpretation: `32x32` is numerically stable and still shows setup and Krylov benefits, but total wall time is not yet faster because both methods spend several minutes in matrix-free Schur applications and velocity Krylov solves. This is not ready for a three-seed production run until the velocity/Schur application cost is reduced.
