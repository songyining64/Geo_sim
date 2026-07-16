"""Shared helpers for blocked Stokes baseline experiments."""

from __future__ import annotations

import hashlib
import platform
import resource
import subprocess
import sys
import threading
import time
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import t as student_t

from core.stokes_solver import StokesConfig, StokesMesh


METHODS = (
    'direct',
    'internal_fresh', 'internal_reuse', 'internal_periodic', 'internal_change_aware',
    'pyamg_fresh', 'pyamg_reuse', 'pyamg_periodic', 'pyamg_change_aware',
    'petsc_gamg_fresh',
    'hypre_boomeramg_fresh',
)

BACKENDS = {
    'internal_fresh': ('internal', 'fresh'),
    'internal_reuse': ('internal', 'reuse'),
    'internal_periodic': ('internal', 'periodic'),
    'internal_change_aware': ('internal', 'change_aware'),
    'pyamg_fresh': ('pyamg', 'fresh'),
    'pyamg_reuse': ('pyamg', 'reuse'),
    'pyamg_periodic': ('pyamg', 'periodic'),
    'pyamg_change_aware': ('pyamg', 'change_aware'),
    'petsc_gamg_fresh': ('petsc_gamg', 'fresh'),
    'hypre_boomeramg_fresh': ('hypre_boomeramg', 'fresh'),
}

VISCOSITY_MODES = (
    'isoviscous',
    'temperature_dependent',
    'temperature_strain_rate',
)

MESH_MODES = (
    'structured',
    'unstructured',
)


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == 'Darwin' else value * 1024)


def summary_stats(values, confidence: float = 0.95) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    count = int(values.size)
    if not count:
        return {
            'mean': None, 'std': None, 'count': 0, 'confidence': confidence,
            'ci_low': None, 'ci_high': None, 'ci_half_width': None,
        }
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if count > 1 else 0.0
    if count > 1:
        half_width = float(
            student_t.ppf(0.5 + confidence / 2.0, count - 1) * std / np.sqrt(count))
    else:
        half_width = None
    return {
        'mean': mean,
        'std': std,
        'count': count,
        'confidence': confidence,
        'ci_low': mean - half_width if half_width is not None else None,
        'ci_high': mean + half_width if half_width is not None else None,
        'ci_half_width': half_width,
    }


def seeded_summary_stats(items, metric: str, confidence: float = 0.95):
    by_seed = {}
    for item in items:
        if metric in item:
            by_seed.setdefault(item['seed'], []).append(float(item[metric]))
    seed_means = [np.mean(values) for values in by_seed.values()]
    stats = summary_stats(seed_means, confidence)
    stats['raw_count'] = int(sum(len(values) for values in by_seed.values()))
    stats['seed_count'] = len(by_seed)
    return stats


def deterministic_method_order(methods, *case_key):
    """Return a stable per-case method permutation across Python processes."""
    digest = hashlib.sha256(
        json_key(case_key).encode('utf-8')).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    order = list(methods)
    rng.shuffle(order)
    return order


def json_key(values):
    return '|'.join(str(value) for value in values)


def paired_seed_summary_stats(items, reference_items, metric: str,
                              confidence: float = 0.95):
    """Summarize paired target-reference differences after seed aggregation."""
    target = {
        (item['seed'], item.get('repeat', 0)): float(item[metric])
        for item in items if metric in item
    }
    reference = {
        (item['seed'], item.get('repeat', 0)): float(item[metric])
        for item in reference_items if metric in item
    }
    by_seed_difference = {}
    by_seed_ratio = {}
    for key in sorted(target.keys() & reference.keys()):
        seed = key[0]
        by_seed_difference.setdefault(seed, []).append(target[key] - reference[key])
        if reference[key] != 0.0:
            by_seed_ratio.setdefault(seed, []).append(target[key] / reference[key])

    differences = [np.mean(values) for values in by_seed_difference.values()]
    ratios = [np.mean(values) for values in by_seed_ratio.values()]
    result = {
        'difference': summary_stats(differences, confidence),
        'ratio': summary_stats(ratios, confidence),
        'paired_seed_count': len(by_seed_difference),
        'paired_raw_count': int(sum(len(values) for values in by_seed_difference.values())),
    }
    return result


def prepare_output_directory(path, fresh: bool = False):
    """Create an output directory, optionally requiring a clean formal run."""
    from pathlib import Path

    path = Path(path)
    if fresh and path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f'formal output directory is not empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def probe_backend_in_subprocess(backend):
    """Probe native solver availability without initializing PETSc in the runner."""
    code = (
        'from core.stokes_solver import probe_velocity_amg_backend; '
        f'available, reason = probe_velocity_amg_backend({backend!r}); '
        'print(reason); raise SystemExit(0 if available else 1)'
    )
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True)
    reason = result.stdout.strip() or result.stderr.strip()
    return result.returncode == 0, reason


class NativeRSSMonitor:
    """Sample process RSS around only the measured method region."""

    def __init__(self, interval_s: float = 0.001):
        import psutil
        self.process = psutil.Process()
        self.interval_s = interval_s
        self.baseline = int(self.process.memory_info().rss)
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        def sample():
            while not self._stop.wait(self.interval_s):
                self.peak = max(self.peak, int(self.process.memory_info().rss))
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, int(self.process.memory_info().rss))

    def metrics(self):
        return {
            'method_rss_baseline_bytes': self.baseline,
            'method_peak_rss_bytes': self.peak,
            'method_peak_rss_delta_bytes': max(self.peak - self.baseline, 0),
        }


def environment_metadata():
    metadata = {
        'python': sys.version,
        'platform': platform.platform(),
        'numpy': np.__version__,
    }
    try:
        import scipy
        metadata['scipy'] = scipy.__version__
    except ImportError:
        pass
    try:
        import pyamg
        metadata['pyamg'] = pyamg.__version__
    except ImportError:
        pass
    try:
        from petsc4py import PETSc
        metadata['petsc'] = PETSc.Sys.getVersionInfo()
        get_arch = getattr(PETSc.Sys, 'getArchType', None)
        if get_arch is not None:
            metadata['petsc_arch'] = get_arch()
    except ImportError:
        pass
    return metadata


def apply_viscosity_mode(config: StokesConfig, viscosity_mode: str, contrast: float) -> None:
    if viscosity_mode == 'isoviscous':
        config.viscosity_contrast = 1.0
        config.strain_rate_dependent_viscosity = False
        return
    if viscosity_mode == 'temperature_dependent':
        config.viscosity_contrast = contrast
        config.strain_rate_dependent_viscosity = False
        return
    if viscosity_mode == 'temperature_strain_rate':
        config.viscosity_contrast = contrast
        config.strain_rate_dependent_viscosity = True
        return
    raise ValueError(f'unknown viscosity mode: {viscosity_mode}')


def build_case_mesh(n: int, dimension: int = 3, mesh_mode: str = 'structured',
                    seed: int = 0, unstructured_points: int | None = None
                    ) -> Tuple[StokesMesh, Dict[str, int]]:
    if mesh_mode == 'structured':
        mesh = StokesMesh(n, n, n if dimension == 3 else None)
        return mesh, {
            'mesh_mode': mesh_mode,
            'dimension': dimension,
            'n_nodes': mesh.n_nodes,
            'n_elements': mesh.n_elements,
            'velocity_dofs': dimension * mesh.n_nodes,
            'stokes_dofs': (dimension + 1) * mesh.n_nodes,
            'thermo_stokes_dofs': (dimension + 2) * mesh.n_nodes,
        }
    if mesh_mode != 'unstructured':
        raise ValueError(f'unknown mesh mode: {mesh_mode}')
    n_points = int(unstructured_points if unstructured_points is not None else max(8, 4 * n))
    rng = np.random.default_rng(seed)
    if dimension == 2:
        corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        interior = rng.uniform(0.05, 0.95, size=(n_points, 2))
        points = np.vstack([corners, interior])
        mesh = StokesMesh.from_unstructured_triangles(
            points, Delaunay(points).simplices)
    elif dimension == 3:
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=float)
        interior = rng.uniform(0.05, 0.95, size=(n_points, 3))
        points = np.vstack([corners, interior])
        mesh = StokesMesh.from_unstructured_tetrahedra(
            points, Delaunay(points).simplices)
    else:
        raise ValueError(f'unsupported dimension: {dimension}')
    return mesh, {
        'mesh_mode': mesh_mode,
        'dimension': dimension,
        'n_nodes': mesh.n_nodes,
        'n_elements': mesh.n_elements,
        'velocity_dofs': dimension * mesh.n_nodes,
        'stokes_dofs': (dimension + 1) * mesh.n_nodes,
        'thermo_stokes_dofs': (dimension + 2) * mesh.n_nodes,
        'unstructured_points': n_points,
    }


def build_case_config(n: int, contrast: float, picard_iterations: int,
                      dimension: int = 3,
                      viscosity_mode: str = 'temperature_dependent',
                      backend: str = 'internal', policy: str = 'fresh',
                      velocity_solver_max_iterations: int = 500,
                      velocity_solver_rtol: float = 1e-6,
                      pressure_solver_rtol: float = 1e-6,
                      schur_velocity_inverse: str = 'lu',
                      schur_velocity_vcycles: int = 2) -> StokesConfig:
    config = StokesConfig(
        nx=n,
        ny=n,
        nz=n if dimension == 3 else None,
        rayleigh=1e3,
        viscosity_contrast=contrast,
        max_picard_iterations=picard_iterations,
        picard_tolerance=0.0,
        use_meta_amg=False,
        pressure_solver='matrix_free_schur',
        schur_velocity_inverse=schur_velocity_inverse,
        schur_velocity_vcycles=schur_velocity_vcycles,
        meta_adapt_steps=1,
        velocity_amg_backend=backend,
        velocity_hierarchy_policy=policy,
        velocity_solver_max_iterations=velocity_solver_max_iterations,
        velocity_solver_rtol=velocity_solver_rtol,
        pressure_solver_rtol=pressure_solver_rtol,
    )
    apply_viscosity_mode(config, viscosity_mode, contrast)
    return config
