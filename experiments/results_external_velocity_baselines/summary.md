# External Velocity-Block Baselines

Configuration: `16x16`, seed `0`, velocity block size `578`, nnz `4988`.

| Method | Setup time | Solve time | Relative residual | Status |
| --- | ---: | ---: | ---: | --- |
| SciPy direct | 0.0000 s | 0.0010 s | 3.11e-15 | reference |
| PyAMG smoothed aggregation | 0.0028 s | 0.0158 s | 9.66e-9 | usable AMG baseline |
| PyAMG Ruge-Stuben | - | - | - | failed on coarse solve NaN/Inf |
| PETSc CG+GAMG | 0.0011 s | 0.0083 s | 6.94e-4 | insufficient residual |
| PETSc GMRES+GAMG | 0.0007 s | 0.0010 s | 1.32e-7 | usable but looser |
| PETSc GMRES+ILU | 0.0001 s | 0.0001 s | 1.40e-8 | usable iterative baseline |
| PETSc preonly+LU | 0.0006 s | 0.0000 s | 3.88e-15 | direct reference |

Interpretation: external dependencies are installed and callable. PyAMG's smoothed aggregation is the clean AMG baseline for this velocity block; PETSc GAMG needs tuning before it can be used as a high-accuracy AMG baseline.
