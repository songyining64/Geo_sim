#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-.venv-petsc-hypre}"
PYTHON="${PYTHON:-python3}"
PETSC_VERSION="${PETSC_VERSION:-3.25.3}"

brew install cmake open-mpi
"${PYTHON}" -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel numpy==1.26.4

MPICC=/opt/homebrew/bin/mpicc \
MPICXX=/opt/homebrew/bin/mpicxx \
MPIFORT=/opt/homebrew/bin/mpifort \
PETSC_CONFIGURE_OPTIONS='--download-hypre --with-debugging=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3' \
"${ENV_DIR}/bin/python" -m pip install --no-cache-dir --no-binary=:all: \
  "petsc==${PETSC_VERSION}"

PETSC_DIR=$("${ENV_DIR}/bin/python" -c 'import petsc; print(petsc.get_config()["PETSC_DIR"])') \
"${ENV_DIR}/bin/python" -m pip install --no-cache-dir --no-binary=petsc4py \
  "petsc4py==${PETSC_VERSION}" scipy==1.13.1 pyamg==5.3.0 \
  pyyaml==6.0.3 sympy==1.14.0 tqdm==4.67.1 pytest==8.4.2 pandas==2.3.3 \
  pyparsing==3.3.2 psutil==7.0.0

"${ENV_DIR}/bin/python" - <<'PY'
from core.stokes_solver import probe_velocity_amg_backend

for backend in ('pyamg', 'petsc_gamg', 'hypre_boomeramg'):
    available, reason = probe_velocity_amg_backend(backend)
    print(f'{backend}: {"available" if available else reason}')
    if not available:
        raise SystemExit(1)
PY
