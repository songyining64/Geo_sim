# Paper-Medium Blocked Stokes Comparison

Configuration: `8x8`, four Picard iterations, `Ra=1e4`, viscosity contrast `1e2`, seeds `0/1/2`.

| Metric | Traditional blocked | Meta blocked | Change |
| --- | ---: | ---: | ---: |
| Hierarchy build time | 0.07596 +- 0.00168 s | 0.04585 +- 0.00657 s | -39.6% |
| Online setup including adaptation | 0.07596 +- 0.00168 s | 0.11291 +- 0.00690 s | +48.6% |
| Velocity Krylov iterations | 4183.7 +- 97.4 | 3940.7 +- 118.8 | -5.8% |
| Full solve wall time | 2.2939 +- 0.0719 s | 2.1520 +- 0.0493 s | -6.2% |
| Full direct fallbacks | 0 | 0 | unchanged |
| Velocity field relative error | reference | 1.03e-9 +- 3.09e-10 | physically equivalent |

## Interpretation

The transferred hierarchy is cheaper to construct and does not increase Krylov work. However, three gradient adaptation steps cost about `0.0671 s` per four-step trajectory, which removes the hierarchy-build saving and makes total online setup slower than rebuilding. The current defensible claim is an end-to-end wall-time reduction with equivalent algebraic and physical solutions, not a total online setup speedup.

The next ablation should reduce or amortize adaptation cost (`1/2/3` steps, ANIL-style partial adaptation, or inference-only reuse) while preserving the observed Krylov reduction.
