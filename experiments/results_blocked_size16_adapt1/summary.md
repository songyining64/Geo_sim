# Size-16 Blocked Stokes Adaptation Check

Configuration: `16x16`, four Picard iterations, `Ra=1e4`, viscosity contrast `1e2`, `adapt_steps=1`, seeds `0/1/2`.

| Metric | Traditional blocked | Meta blocked | Change |
| --- | ---: | ---: | ---: |
| Online setup including adaptation | 0.7171 +- 0.0383 s | 0.5129 +- 0.0172 s | 1.40x speedup |
| Velocity Krylov iterations | 17034.7 +- 2256.2 | 12078.7 +- 995.5 | ratio 0.715 |
| Full solve wall time | 33.9502 +- 3.7269 s | 15.6115 +- 1.3747 s | -53.9% |
| Full direct fallbacks | 0 | 0 | unchanged |
| Velocity field relative error | reference | 4.99e-9 +- 3.14e-9 | physically equivalent |

Interpretation: `8x8` is too small for gradient adaptation to pay off, but at `16x16` the one-step adaptation cost is amortized. This is the first configuration where the defensible claim becomes online setup speedup plus lower Krylov work, not only hierarchy-build speedup.
