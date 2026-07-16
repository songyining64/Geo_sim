# Reproducing the blocked Stokes experiments

## Environment

The paper baselines require PyAMG and a PETSc build containing both GAMG and
HYPRE BoomerAMG. A normal `pip install petsc` wheel may not contain HYPRE.

On Apple Silicon, create an isolated environment from the repository root:

```bash
bash experiments/build_petsc_hypre_env.sh .venv-petsc-hypre
```

The script builds PETSc 3.25.3 with `--download-hypre`. Homebrew's standalone
HYPRE 3.1.0 currently uses bigint/mixedint and is incompatible with PETSc's
default 32-bit indices, so `--with-hypre-dir=/opt/homebrew` must not be used for
this protocol.

Verify the actual PETSc preconditioners, not only package imports:

```bash
.venv-petsc-hypre/bin/python - <<'PY'
from core.stokes_solver import probe_velocity_amg_backend
for backend in ('pyamg', 'petsc_gamg', 'hypre_boomeramg'):
    print(backend, probe_velocity_amg_backend(backend))
PY
```

## Scaling protocol

Each method runs in a fresh process. Native peak RSS is obtained from
a native process RSS sampler around the measured method region; process-wide
`getrusage(RUSAGE_SELF).ru_maxrss` is retained as a diagnostic only. Seeds
control physical initial conditions; repetitions estimate timing variability
within each seed. Confidence intervals use per-seed means, not pooled repeats.
Use at least three seeds and five repetitions for final tables.
Run each dimension as a separate command, serially, using a new output
directory. Do not launch 2D and 3D timing batches concurrently.

```bash
.venv-petsc-hypre/bin/python experiments/benchmark_stokes_3d_scalable.py \
  --levels 8,12,16 \
  --dimensions 2 \
  --mesh-modes structured,unstructured \
  --viscosity-modes isoviscous,temperature_dependent,temperature_strain_rate \
  --contrast 1e4 \
  --picard-iterations 5 \
  --schur-velocity-inverse lu \
  --velocity-solver-rtol 1e-6 \
  --pressure-solver-rtol 1e-6 \
  --seeds 0,1,2 \
  --repetitions 5 \
  --methods hypre_boomeramg_fresh,petsc_gamg_fresh,pyamg_fresh \
  --paired-baseline hypre_boomeramg_fresh \
  --method-order shuffled \
  --fresh-output --require-hypre \
  --output experiments/results_blocked_stokes_scaling_2d
```

After the 2D command finishes, run the same command with `--dimensions 3` and
a new `results_blocked_stokes_scaling_3d` output directory.

Unstructured cases use seeded Delaunay triangle or tetrahedral meshes with the
same unit-domain boundary conditions as the structured cases.

## Full nonlinear protocol

The nonlinear benchmark automatically adds a direct solve for every physical
seed and repetition. It compares final Nusselt number, RMS velocity,
temperature field, and velocity field against that direct reference.

```bash
.venv-petsc-hypre/bin/python experiments/benchmark_blocked_stokes_nonlinear.py \
  --levels 8 \
  --dimensions 2 \
  --mesh-modes structured,unstructured \
  --viscosity-modes isoviscous,temperature_dependent,temperature_strain_rate \
  --contrast 1e4 \
  --steps 20 \
  --picard-iterations 5 \
  --schur-velocity-inverse lu \
  --velocity-solver-rtol 1e-6 \
  --pressure-solver-rtol 1e-6 \
  --seeds 0,1,2,3,4 \
  --repetitions 2 \
  --methods hypre_boomeramg_fresh \
  --paired-baseline direct \
  --method-order shuffled \
  --fresh-output --require-hypre \
  --output experiments/results_blocked_stokes_nonlinear_2d
```

After the 2D command finishes, run the same command with `--dimensions 3` and
a new `results_blocked_stokes_nonlinear_3d` output directory.

For the long-run check, repeat with `--steps 50`. Keep raw per-run JSON files;
the aggregate contains 95% Student-t confidence intervals, failure rates, and
backend-unavailable rates.

## Paper tables

```bash
.venv-petsc-hypre/bin/python experiments/export_blocked_stokes_paper_tables.py \
  --scaling experiments/results_blocked_stokes_scaling_2d/stokes_3d_scalable.json \
  --scaling experiments/results_blocked_stokes_scaling_3d/stokes_3d_scalable.json \
  --nonlinear experiments/results_blocked_stokes_nonlinear_2d/blocked_stokes_nonlinear.json \
  --nonlinear experiments/results_blocked_stokes_nonlinear_3d/blocked_stokes_nonlinear.json \
  --output experiments/paper_artifacts/blocked_stokes
```

This writes scaling, amortized nonlinear, physics-equivalence, paired ablation,
and coverage/failure CSV tables plus one Markdown summary.

## Schur configuration

The production default uses one sparse LU factorization of the velocity block
per Picard system for matrix-free Schur applications. This removes nested
velocity Krylov solves while preserving the common outer block algorithm for
all AMG backends. Fixed V-cycle Schur inverses remain available for ablation,
but did not pass the no-full-fallback quality gate at viscosity contrast
`1e4`. Velocity and pressure Krylov tolerances are `1e-6`, matching the final
linear quality threshold and avoiding unnecessary over-solving.

## Reporting rules

- Report all unavailable backends and failed subprocesses; do not silently drop them.
- Treat seed and repetition as separate axes.
- Do not claim confidence intervals from a single sample.
- Workers pin OpenMP, OpenBLAS, MKL, and vecLib to one thread.
- Exclude shared trajectory generation from replay timings, but include assembly,
  nonlinear updates, advection, and solve time in full nonlinear wall time.
- Accept a blocked run only when it has no velocity, pressure, full-system, or
  backend-setup fallback and satisfies the reported residual threshold.

## PETSc field-split decision

PETSc `PCFIELDSPLIT` is not part of the primary baseline matrix at this stage.
The controlled comparison intentionally holds the block-Schur algorithm fixed
and changes only the velocity AMG implementation. Adding field-split now would
change both the outer block algorithm and the AMG backend, so it would not
isolate the claimed effect. It should be added later as a separate external
end-to-end baseline if the paper claims competitiveness with native PETSc
Stokes configurations or distributed-memory production scaling.
