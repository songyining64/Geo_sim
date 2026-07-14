# 3D Stokes validation summary

## Manufactured-solution convergence

| n | DOFs | L2(u) | order | H1(u) | order | L2(p) | order |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 108 | 2.589e-01 | - | 2.284e+00 | - | 3.944e-01 | - |
| 3 | 256 | 1.465e-01 | 1.403 | 1.684e+00 | 0.751 | 5.348e-01 | -0.751 |
| 4 | 500 | 9.125e-02 | 1.647 | 1.314e+00 | 0.864 | 5.146e-01 | 0.134 |
| 6 | 1372 | 4.375e-02 | 1.813 | 9.028e-01 | 0.925 | 3.711e-01 | 0.806 |
| 8 | 2916 | 2.523e-02 | 1.914 | 6.843e-01 | 0.963 | 2.589e-01 | 1.251 |

## Variable-viscosity solver comparison

Values are mean +/- population standard deviation over three seeds.

| method | nodal contrast | element contrast | setup (s) | solve (s) | total (s) | velocity Krylov | full fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 1e+00 | 1.00e+00 | 0.012 +/- 0.000 | 0.000 +/- 0.000 | 0.012 +/- 0.000 | 0.0 +/- 0.0 | 0.0 |
| direct | 1e+02 | 5.61e+01 | 0.015 +/- 0.000 | 0.000 +/- 0.000 | 0.015 +/- 0.000 | 0.0 +/- 0.0 | 0.0 |
| direct | 1e+04 | 2.66e+03 | 0.015 +/- 0.000 | 0.000 +/- 0.000 | 0.015 +/- 0.000 | 0.0 +/- 0.0 | 0.0 |
| direct | 1e+06 | 1.06e+05 | 0.015 +/- 0.000 | 0.000 +/- 0.000 | 0.015 +/- 0.000 | 0.0 +/- 0.0 | 0.0 |
| meta_amg | 1e+00 | 1.00e+00 | 1.194 +/- 0.179 | 0.787 +/- 0.082 | 1.981 +/- 0.258 | 117.0 +/- 7.8 | 0.0 |
| meta_amg | 1e+02 | 5.61e+01 | 1.198 +/- 0.150 | 0.863 +/- 0.097 | 2.061 +/- 0.234 | 141.0 +/- 14.4 | 0.0 |
| meta_amg | 1e+04 | 2.66e+03 | 1.108 +/- 0.174 | 0.854 +/- 0.084 | 1.962 +/- 0.256 | 154.7 +/- 11.1 | 0.0 |
| meta_amg | 1e+06 | 1.06e+05 | 1.164 +/- 0.141 | 0.905 +/- 0.081 | 2.069 +/- 0.222 | 162.3 +/- 7.6 | 0.0 |
| traditional_amg | 1e+00 | 1.00e+00 | 1.841 +/- 0.004 | 1.427 +/- 0.019 | 3.269 +/- 0.018 | 115.0 +/- 1.4 | 0.0 |
| traditional_amg | 1e+02 | 5.61e+01 | 1.952 +/- 0.015 | 1.556 +/- 0.007 | 3.508 +/- 0.010 | 131.0 +/- 1.4 | 0.0 |
| traditional_amg | 1e+04 | 2.66e+03 | 1.601 +/- 0.195 | 1.480 +/- 0.130 | 3.081 +/- 0.324 | 152.0 +/- 7.1 | 0.0 |
| traditional_amg | 1e+06 | 1.06e+05 | 1.842 +/- 0.031 | 1.633 +/- 0.022 | 3.475 +/- 0.030 | 146.0 +/- 3.7 | 0.0 |

> Scope: these n=4 results establish small-grid feasibility only. Direct is faster at this size; no 3D large-scale speedup is claimed.
> Memory: Python peak memory excludes some native SciPy/SuperLU allocations.
