# 3D Unstructured Tetrahedral Patch Test

Three independently generated Delaunay tetrahedral meshes were tested with 128 nodes and approximately 730 elements each. All tetrahedra were reoriented to positive volume before assembly.

| Metric | Result |
| --- | ---: |
| Pass rate | 3/3 |
| Symmetry error | 0.0 |
| Affine manufactured solution error | 6.53e-16 to 2.26e-15 |
| Free-node relative residual | 1.05e-15 to 1.67e-15 |
| Minimum tetrahedron volume | 9.84e-6 |

The same mesh family was also used for a blocked 3D Stokes/Picard validation with 32 nodes and approximately 140 tetrahedra. All three seeds completed two Picard iterations with finite solutions, zero velocity/full fallback, Nusselt numbers near one, and linear relative residuals between `4.85e-11` and `1.88e-10`.

Scope: this establishes a runnable small 3D unstructured linear-tetra Stokes path. It does not yet establish mesh-refinement convergence, benchmark-flow accuracy, or large 3D performance.
