# Size-64 Schur-LU Scalability Smoke

Configuration: `64x64`, four Picard iterations, seed `0`, `adapt_steps=1`, training matrix size capped at `2048`, velocity Krylov limit `1000`.

| Metric | Traditional blocked | Meta blocked | Change |
| --- | ---: | ---: | ---: |
| Total online preconditioner setup | 78.0694 s | 34.4302 s | 2.27x speedup |
| Velocity Krylov iterations | 1685 | 4092 | not directly comparable after fallback |
| Full solve wall time | 201.82 s | 99.12 s | -50.9% |
| Full-system fallbacks | 0 | 0 | stable |
| Velocity direct fallbacks | 6 | 0 | traditional baseline failed quality gate |
| Velocity field relative error | reference | 3.42e-7 | acceptable |

This result proves that the Meta hierarchy can complete the `64x64` target without velocity or full-system direct fallback. It is a scalability smoke result, not a fair performance result, because the traditional hierarchy triggered six velocity direct fallbacks.
