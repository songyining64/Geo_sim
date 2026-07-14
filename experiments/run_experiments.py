"""
Meta-AMG 论文实验框架

运行所有论文所需实验并生成数据/图表。

实验设计:
  Experiment 1: 收敛性验证 — MAML适配C/F的AMG vs 传统AMG求解精度
  Experiment 2: Setup加速比 — 不同矩阵规模下的setup时间对比
  Experiment 3: 消融实验 — 零样本 vs 适配 vs 适配步数影响
  Experiment 4: 粘度对比鲁棒性 — 不同contrast下的适配准确率
  Experiment 5: 可扩展性 — 训练矩阵规模 vs 测试矩阵规模的泛化

Usage:
  python experiments/run_experiments.py --exp all
  python experiments/run_experiments.py --exp e1  # 只跑实验1
"""

import numpy as np
import time
import json
import argparse
import warnings
import copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load modules (bypass broken gpu_acceleration/__init__.py)
import importlib.util

_neural_path = str(Path(__file__).parent.parent / 'gpu_acceleration' / 'neural_amg.py')
_nm_spec = importlib.util.spec_from_file_location('neural_amg', _neural_path)
_nm_mod = importlib.util.module_from_spec(_nm_spec)
sys.modules['neural_amg'] = _nm_mod
_nm_spec.loader.exec_module(_nm_mod)

_meta_path = str(Path(__file__).parent.parent / 'gpu_acceleration' / 'meta_amg.py')
_mm_spec = importlib.util.spec_from_file_location('meta_amg', _meta_path)
_mm_mod = importlib.util.module_from_spec(_mm_spec)
sys.modules['meta_amg'] = _mm_mod
_mm_spec.loader.exec_module(_mm_mod)

from meta_amg import (
    MetaAMGConfig, MetaAMG, MatrixSequenceGenerator,
    MetaAMGSolver, MetaAMGAdapter,
)
from neural_amg import (
    NeuralAMGConfig, NeuralAMG, AMGDataGenerator,
)
from solvers.multigrid_solver import (
    AlgebraicMultigridSolver, MultigridConfig
)
from scipy.sparse.linalg import lsmr, spsolve
from core.stokes_solver import PicardStokesSolver, StokesConfig


# ============================================================================
# 配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """实验总配置"""
    output_dir: str = "experiments/results"
    n_runs: int = 3
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    # 矩阵规模
    matrix_sizes: List[int] = field(default_factory=lambda: [
        100, 225, 400, 625, 900, 1600
    ])
    # 粘度对比度
    contrasts: List[float] = field(default_factory=lambda: [
        1e2, 1e3, 1e4, 1e5, 1e6
    ])
    # 序列长度
    sequence_length: int = 10
    # 训练参数
    meta_epochs: int = 30
    meta_tasks: int = 100
    # 适配步数消融
    adapt_steps_list: List[int] = field(default_factory=lambda: [
        1, 2, 3, 5, 10
    ])
    stokes_nx: int = 8
    stokes_ny: int = 8
    stokes_picard_iterations: int = 6
    stokes_rayleigh: float = 1e4
    stokes_viscosity_contrast: float = 1e2
    meta_adapt_steps: int = 3
    periodic_rebuild_interval: int = 3
    matrix_change_threshold: float = 0.10
    nusselt_relative_tolerance: float = 0.02
    rms_velocity_relative_tolerance: float = 0.05
    velocity_field_relative_tolerance: float = 0.10
    linear_residual_tolerance: float = 1e-6
    pressure_solver: str = 'matrix_free_schur'
    schur_velocity_inverse: str = 'krylov'
    meta_training_max_matrix_size: Optional[int] = None
    velocity_solver_max_iterations: int = 500


def apply_preset(config: ExperimentConfig, preset: str) -> None:
    if preset == 'quick':
        config.meta_epochs = 10
        config.meta_tasks = 30
        config.matrix_sizes = [100, 225, 400]
        config.contrasts = [1e3, 1e5]
        config.n_runs = 1
        config.stokes_nx = 4
        config.stokes_ny = 4
        config.stokes_picard_iterations = 3
        config.stokes_rayleigh = 1e3
        config.stokes_viscosity_contrast = 1e1
        config.seeds = [0]
    elif preset == 'paper_medium':
        config.meta_epochs = 20
        config.meta_tasks = 60
        config.matrix_sizes = [225, 400, 625]
        config.contrasts = [1e2, 1e4, 1e6]
        config.n_runs = 3
        config.sequence_length = 8
        config.stokes_nx = 8
        config.stokes_ny = 8
        config.stokes_picard_iterations = 4
        config.stokes_rayleigh = 1e4
        config.stokes_viscosity_contrast = 1e2
        config.seeds = [0, 1, 2]
    elif preset == 'paper_large':
        config.meta_epochs = 30
        config.meta_tasks = 100
        config.matrix_sizes = [225, 400, 625, 900]
        config.contrasts = [1e2, 1e4, 1e6]
        config.n_runs = 5
        config.sequence_length = 10
        config.stokes_nx = 12
        config.stokes_ny = 12
        config.stokes_picard_iterations = 5
        config.stokes_rayleigh = 1e4
        config.stokes_viscosity_contrast = 1e2
        config.seeds = [0, 1, 2, 3, 4]


def _with_seed(seed: int, fn):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return fn()
    finally:
        np.random.set_state(state)


def _with_reproducible_seed(seed: int, fn):
    state = np.random.get_state()
    np.random.seed(seed)
    torch_state = None
    try:
        import torch
        torch_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
    except ImportError:
        torch = None
    try:
        return fn()
    finally:
        np.random.set_state(state)
        if torch_state is not None:
            torch.random.set_rng_state(torch_state)


# ============================================================================
# 统一序列评测辅助函数
# ============================================================================

def _rhs_list_from_exact_solution(matrices: List, x_exact: np.ndarray) -> List[np.ndarray]:
    return [A @ x_exact for A in matrices]


def _relative_residual(A, b, x) -> float:
    if x is None or not np.isfinite(x).all():
        return float('inf')
    residual = float(np.linalg.norm(b - A @ x))
    rhs_norm = float(np.linalg.norm(b))
    return residual / rhs_norm if rhs_norm > 1e-14 else residual


def _solution_is_acceptable(A, b, x, relative_tolerance: float = 1e-6) -> bool:
    if x is None or not np.isfinite(x).all():
        return False
    relative_residual = _relative_residual(A, b, x)
    return bool(np.isfinite(relative_residual) and relative_residual <= relative_tolerance)


def _relative_matrix_change(A, reference) -> float:
    if A.shape != reference.shape:
        return float('inf')
    delta = (A - reference).tocsr()
    delta_norm = float(np.sqrt(np.dot(delta.data, delta.data)))
    reference = reference.tocsr()
    reference_norm = float(np.sqrt(np.dot(reference.data, reference.data)))
    return delta_norm / max(reference_norm, 1e-14)


def _direct_solution(A, b):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x = spsolve(A, b)
    if _solution_is_acceptable(A, b, x):
        return np.asarray(x), 'direct'
    return np.asarray(lsmr(A, b, atol=1e-10, btol=1e-10)[0]), 'least_squares'


def _extract_cf_labels_from_solver(solver, A) -> np.ndarray:
    n = A.shape[0]
    labels = np.zeros(n, dtype=np.float32)
    if hasattr(solver, 'interpolation_operators') and solver.interpolation_operators:
        P = solver.interpolation_operators[0]
        for i in range(min(P.shape[0], n)):
            row = P[i].toarray().flatten()
            dominant = np.max(np.abs(row)) if row.size else 0.0
            if dominant > 0.9 and np.sum(np.abs(row) > 0.1) == 1:
                labels[i] = 1.0
        if np.any(labels > 0):
            return labels

    from solvers.multigrid_solver import AdaptiveCoarsening
    coarse, _ = AdaptiveCoarsening.algebraic_coarsening(A, 0.25)
    labels[coarse] = 1.0
    return labels


def _solve_with_cf_metrics(meta_solver: MetaAMGSolver, A, b, coarse, fine, x0=None):
    solver = AlgebraicMultigridSolver(meta_solver.amg_config)
    solver.levels = []
    solver.interpolation_operators = []
    solver.restriction_operators = []
    solver.coarse_matrices = []

    current_A = A.copy()
    current_coarse = list(coarse)
    current_fine = list(fine)
    current_level = 0

    setup_start = time.time()
    while (current_level < solver.config.max_levels and
           current_A.shape[0] > solver.config.max_coarse_size):
        if len(current_coarse) == 0 or len(current_fine) == 0:
            break

        solver.levels.append({
            'matrix': current_A, 'size': current_A.shape[0], 'level': current_level,
        })

        P = solver._build_advanced_interpolation_operator(current_A, current_coarse, current_fine)
        R = P.T
        coarse_A = R @ current_A @ P

        solver.interpolation_operators.append(P)
        solver.restriction_operators.append(R)
        solver.coarse_matrices.append(coarse_A)

        current_A = coarse_A
        current_level += 1

        from solvers.multigrid_solver import AdaptiveCoarsening
        try:
            current_coarse, current_fine = AdaptiveCoarsening.algebraic_coarsening(
                current_A, meta_solver.config.strong_threshold)
        except Exception:
            break

    solver.levels.append({
        'matrix': current_A, 'size': current_A.shape[0], 'level': current_level,
    })
    solver.is_setup = True
    setup_time = time.time() - setup_start

    x = np.zeros_like(b) if x0 is None else x0.copy()
    solve_start = time.time()
    iteration = 0
    residual_norm = np.linalg.norm(b - A @ x)
    for iteration in range(solver.config.max_iterations):
        x = solver.v_cycle(0, b, x)
        residual_norm = np.linalg.norm(b - A @ x)
        if _relative_residual(A, b, x) <= 1e-6:
            break
    solve_time = time.time() - solve_start
    total_iterations = iteration + 1

    solver.performance_stats['setup_time'] = setup_time
    solver.performance_stats['solve_time'] = solve_time
    solver.performance_stats['total_iterations'] = total_iterations
    solver.performance_stats['final_residual'] = residual_norm
    return x, solver


def _traditional_step(meta_solver: MetaAMGSolver, A, b, x0=None):
    solver = AlgebraicMultigridSolver(meta_solver.amg_config)
    total_start = time.time()
    x = solver.solve(A, b, x0)
    direct_fallback = not _solution_is_acceptable(A, b, x)
    backend = 'amg'
    if direct_fallback:
        x, backend = _direct_solution(A, b)
    total_time = time.time() - total_start
    cf_labels = _extract_cf_labels_from_solver(solver, A)
    return {
        'solution': x,
        'cf_labels': cf_labels,
        'accepted': _solution_is_acceptable(A, b, x),
        'fallback': direct_fallback,
        'solver_backend': backend,
        'setup_time': float(solver.performance_stats.get('setup_time', total_time)),
        'solve_time': float(solver.performance_stats.get('solve_time', 0.0)),
        'total_time': float(total_time),
        'iterations': int(solver.performance_stats.get('total_iterations', 0)),
        'residual_norm': float(np.linalg.norm(b - A @ x)),
        'relative_residual': _relative_residual(A, b, x),
        'two_grid_factor': None,
        'operator_complexity': None,
        'repaired_points': 0,
    }


def _candidate_step(meta_solver: MetaAMGSolver, A, b, proposed_coarse, x0=None):
    assess_start = time.time()
    assessment = meta_solver._validate_transfer(A, proposed_coarse)
    validate_time = time.time() - assess_start

    if assessment.accepted:
        total_start = time.time()
        x, solver = _solve_with_cf_metrics(
            meta_solver, A, b, assessment.coarse, assessment.fine, x0)
        total_time = time.time() - total_start + validate_time
        if _solution_is_acceptable(A, b, x):
            cf_labels = np.zeros(A.shape[0], dtype=np.float32)
            cf_labels[assessment.coarse] = 1.0
            return {
            'solution': x,
            'cf_labels': cf_labels,
            'accepted': True,
            'fallback': False,
            'solver_backend': 'transferred_amg',
            'setup_time': float(validate_time + solver.performance_stats.get('setup_time', 0.0)),
            'solve_time': float(solver.performance_stats.get('solve_time', 0.0)),
            'total_time': float(total_time),
            'iterations': int(solver.performance_stats.get('total_iterations', 0)),
            'residual_norm': float(np.linalg.norm(b - A @ x)),
            'relative_residual': _relative_residual(A, b, x),
            'two_grid_factor': float(assessment.two_grid_factor),
            'operator_complexity': float(assessment.operator_complexity),
            'repaired_points': int(assessment.repaired_points),
            }

    fallback = _traditional_step(meta_solver, A, b, x0)
    fallback['accepted'] = False
    fallback['fallback'] = True
    fallback['setup_time'] += validate_time
    fallback['total_time'] += validate_time
    fallback['two_grid_factor'] = float(assessment.two_grid_factor)
    fallback['operator_complexity'] = float(assessment.operator_complexity)
    fallback['repaired_points'] = int(assessment.repaired_points)
    return fallback


def _sequence_summary(name: str, steps: List[Dict], *, compact: bool = True) -> Dict:
    setup_times = [s['setup_time'] for s in steps]
    solve_times = [s['solve_time'] for s in steps]
    total_times = [s['total_time'] for s in steps]
    iterations = [s['iterations'] for s in steps]
    residuals = [s['residual_norm'] for s in steps]
    relative_residuals = [s['relative_residual'] for s in steps]
    fallback_flags = [1.0 if s['fallback'] else 0.0 for s in steps]
    accepted_flags = [1.0 if s['accepted'] else 0.0 for s in steps]
    compact_steps = [{
        'accepted': s['accepted'],
        'fallback': s['fallback'],
        'solver_backend': s.get('solver_backend', 'unknown'),
        'setup_time': s['setup_time'],
        'solve_time': s['solve_time'],
        'total_time': s['total_time'],
        'iterations': s['iterations'],
        'residual_norm': s['residual_norm'],
        'two_grid_factor': s['two_grid_factor'],
        'operator_complexity': s['operator_complexity'],
        'repaired_points': s['repaired_points'],
        'relative_residual': s['relative_residual'],
        'rebuild': s.get('rebuild', False),
        'decision_reason': s.get('decision_reason', ''),
        'matrix_change': s.get('matrix_change'),
    } for s in steps]
    return {
        'method': name,
        'steps': compact_steps if compact else steps,
        'setup_time_mean': float(np.mean(setup_times)),
        'solve_time_mean': float(np.mean(solve_times)),
        'total_time_mean': float(np.mean(total_times)),
        'iterations_mean': float(np.mean(iterations)),
        'residual_norm_mean': float(np.mean(residuals)),
        'relative_residual_mean': float(np.mean(relative_residuals)),
        'relative_residual_max': float(np.max(relative_residuals)),
        'fallback_rate': float(np.mean(fallback_flags)),
        'accepted_rate': float(np.mean(accepted_flags)),
        'rebuild_rate': float(np.mean([1.0 if s.get('rebuild', False) else 0.0 for s in steps])),
    }


def benchmark_sequence_methods(cfg: MetaAMGConfig,
                               matrices: List,
                               rhs_list: List[np.ndarray],
                               meta: MetaAMG,
                               adapt_steps: int = 3,
                               periodic_rebuild_interval: int = 3,
                               matrix_change_threshold: float = 0.10,
                               compact: bool = True) -> Dict:
    meta_solver = MetaAMGSolver(copy.deepcopy(cfg))
    meta_solver.set_adapter(meta.adapter)
    methods = {
        'traditional': [],
        'reuse': [],
        'periodic_rebuild': [],
        'change_aware_reuse': [],
        'zero_shot': [],
        'adapted': [],
    }

    trad_prev = None
    reuse_prev = None
    adapt_prev = None
    periodic_prev = None
    change_prev = None
    change_anchor = None

    for k, (A, b) in enumerate(zip(matrices, rhs_list)):
        trad_step = _traditional_step(meta_solver, A, b)
        trad_step.update(rebuild=True, decision_reason='fresh_setup', matrix_change=0.0 if k == 0 else None)
        methods['traditional'].append(trad_step)
        trad_prev = {'A': A, 'cf_labels': trad_step['cf_labels']}

        if k == 0:
            for name in ['reuse', 'periodic_rebuild', 'change_aware_reuse', 'zero_shot', 'adapted']:
                methods[name].append(dict(trad_step))
            reuse_prev = {'A': A, 'cf_labels': trad_step['cf_labels']}
            periodic_prev = {'A': A, 'cf_labels': trad_step['cf_labels']}
            change_prev = {'A': A, 'cf_labels': trad_step['cf_labels']}
            change_anchor = A
            adapt_prev = {'A': A, 'cf_labels': trad_step['cf_labels']}
            continue

        reuse_coarse = np.flatnonzero(reuse_prev['cf_labels'] > 0.5).tolist()
        reuse_step = _candidate_step(meta_solver, A, b, reuse_coarse)
        reuse_step.update(rebuild=False, decision_reason='reuse_cf',
                          matrix_change=_relative_matrix_change(A, reuse_prev['A']))
        methods['reuse'].append(reuse_step)
        reuse_prev = {'A': A, 'cf_labels': reuse_step['cf_labels']}

        if k % max(1, periodic_rebuild_interval) == 0:
            periodic_step = _traditional_step(meta_solver, A, b)
            periodic_step.update(rebuild=True, decision_reason='periodic_rebuild',
                                 matrix_change=_relative_matrix_change(A, periodic_prev['A']))
        else:
            periodic_coarse = np.flatnonzero(periodic_prev['cf_labels'] > 0.5).tolist()
            periodic_step = _candidate_step(meta_solver, A, b, periodic_coarse)
            periodic_step.update(rebuild=False, decision_reason='periodic_reuse',
                                 matrix_change=_relative_matrix_change(A, periodic_prev['A']))
        methods['periodic_rebuild'].append(periodic_step)
        periodic_prev = {'A': A, 'cf_labels': periodic_step['cf_labels']}

        anchor_change = _relative_matrix_change(A, change_anchor)
        if anchor_change > matrix_change_threshold:
            change_step = _traditional_step(meta_solver, A, b)
            change_step.update(rebuild=True, decision_reason='matrix_change_rebuild',
                               matrix_change=anchor_change)
            change_anchor = A
        else:
            change_coarse = np.flatnonzero(change_prev['cf_labels'] > 0.5).tolist()
            change_step = _candidate_step(meta_solver, A, b, change_coarse)
            change_step.update(rebuild=False, decision_reason='change_aware_reuse',
                               matrix_change=anchor_change)
        methods['change_aware_reuse'].append(change_step)
        change_prev = {'A': A, 'cf_labels': change_step['cf_labels']}

        zs_coarse, _ = meta.adapter.zero_shot_predict(A)
        zero_shot_step = _candidate_step(meta_solver, A, b, zs_coarse)
        zero_shot_step.update(rebuild=False, decision_reason='zero_shot', matrix_change=None)
        methods['zero_shot'].append(zero_shot_step)

        ad_coarse, _ = meta.adapter.adapt(
            adapt_prev['A'], adapt_prev['cf_labels'], A, adapt_steps=adapt_steps)
        adapted_step = _candidate_step(meta_solver, A, b, ad_coarse)
        adapted_step.update(rebuild=False, decision_reason='meta_adapt',
                            matrix_change=_relative_matrix_change(A, adapt_prev['A']))
        methods['adapted'].append(adapted_step)
        adapt_prev = {'A': A, 'cf_labels': adapted_step['cf_labels']}

    return {name: _sequence_summary(name, steps, compact=compact)
            for name, steps in methods.items()}


def _aggregate_metric_dict(entries: List[Dict], keys: List[str]) -> Dict:
    out = {}
    for key in keys:
        values = [float(entry[key]) for entry in entries]
        out[f'{key}_mean'] = float(np.mean(values))
        out[f'{key}_std'] = float(np.std(values))
    return out


def _aggregate_method_summaries(summaries: List[Dict]) -> Dict:
    methods = summaries[0].keys()
    aggregated = {}
    metric_keys = [
        'setup_time_mean', 'solve_time_mean', 'total_time_mean',
        'iterations_mean', 'residual_norm_mean', 'fallback_rate', 'accepted_rate',
        'relative_residual_mean', 'relative_residual_max', 'rebuild_rate',
    ]
    for method in methods:
        method_entries = [summary[method] for summary in summaries]
        aggregated[method] = {
            'method': method,
            'n_seeds': len(method_entries),
            **_aggregate_metric_dict(method_entries, metric_keys),
        }
    return aggregated


def _velocity_rhs_list_from_stokes_sequence(sequence: List[Dict]) -> List[np.ndarray]:
    return [record['rhs'][record['velocity_indices']] for record in sequence]


def _rms_velocity_from_solver(solver: PicardStokesSolver) -> float:
    ndpn = solver.mesh.n_dofs_per_node
    ux = solver.velocity[0::ndpn]
    uy = solver.velocity[1::ndpn]
    return float(np.sqrt(np.mean(ux ** 2 + uy ** 2)))


def _relative_error(value: float, reference: float) -> float:
    return float(abs(value - reference) / max(abs(reference), 1e-14))


def _velocity_field_relative_error(test_solver: PicardStokesSolver,
                                   reference_solver: PicardStokesSolver) -> float:
    ndpn = reference_solver.mesh.n_dofs_per_node
    velocity_offsets = list(range(reference_solver.mesh.dim))
    indices = np.concatenate([
        np.arange(offset, reference_solver.velocity.size, ndpn)
        for offset in velocity_offsets
    ])
    reference = reference_solver.velocity[indices]
    test = test_solver.velocity[indices]
    return float(np.linalg.norm(test - reference) / max(np.linalg.norm(reference), 1e-14))


def build_stokes_picard_benchmark(config: ExperimentConfig, *, use_meta_training: bool = True,
                                  seed: int = 0):
    stokes_cfg = StokesConfig(
        nx=config.stokes_nx,
        ny=config.stokes_ny,
        rayleigh=config.stokes_rayleigh,
        viscosity_contrast=config.stokes_viscosity_contrast,
        max_picard_iterations=config.stokes_picard_iterations,
        picard_tolerance=0.0,
        use_meta_amg=False,
    )
    collector = PicardStokesSolver(stokes_cfg)
    sequence = collector.collect_picard_sequence(
        max_iterations=config.stokes_picard_iterations, seed=seed)
    matrices = [record['matrix'] for record in sequence]
    rhs_list = _velocity_rhs_list_from_stokes_sequence(sequence)

    meta_cfg = MetaAMGConfig(
        training_data_source='stokes_picard' if use_meta_training else 'synthetic',
        stokes_nx=config.stokes_nx,
        stokes_ny=config.stokes_ny,
        stokes_picard_iterations=config.stokes_picard_iterations,
        stokes_rayleigh_range=(config.stokes_rayleigh, config.stokes_rayleigh),
        stokes_viscosity_contrast_range=(
            config.stokes_viscosity_contrast, config.stokes_viscosity_contrast),
        min_matrix_size=max(16, matrices[0].shape[0] // 4),
        max_matrix_size=max(64, matrices[0].shape[0]),
        num_training_sequences=config.meta_tasks,
        sequence_length=max(2, min(config.sequence_length, config.stokes_picard_iterations)),
        num_meta_epochs=config.meta_epochs,
        inner_steps=3,
        meta_batch_size=4,
        hidden_dim=32,
        num_layers=2,
    )
    meta = MetaAMG(meta_cfg)
    meta.train(num_sequences=config.meta_tasks)
    return meta_cfg, meta, sequence, matrices, rhs_list


def benchmark_full_stokes_picard(config: ExperimentConfig, *, seed: int = 0) -> Dict:
    reference_cfg = StokesConfig(
        nx=config.stokes_nx,
        ny=config.stokes_ny,
        rayleigh=config.stokes_rayleigh,
        viscosity_contrast=config.stokes_viscosity_contrast,
        max_picard_iterations=config.stokes_picard_iterations,
        picard_tolerance=0.0,
        use_meta_amg=False,
    )
    meta_cfg = StokesConfig(
        nx=config.stokes_nx,
        ny=config.stokes_ny,
        rayleigh=config.stokes_rayleigh,
        viscosity_contrast=config.stokes_viscosity_contrast,
        max_picard_iterations=config.stokes_picard_iterations,
        picard_tolerance=0.0,
        use_meta_amg=True,
        meta_training_sequences=config.meta_tasks,
        meta_training_epochs=config.meta_epochs,
    )

    reference_solver = PicardStokesSolver(reference_cfg)
    t0 = time.time()
    reference_sequence = reference_solver.collect_picard_sequence(
        max_iterations=config.stokes_picard_iterations, seed=seed)
    reference_wall = time.time() - t0
    reference_nusselt = float(reference_solver._compute_nusselt())
    reference_vrms = _rms_velocity_from_solver(reference_solver)

    meta_solver = PicardStokesSolver(meta_cfg)
    _with_seed(seed, meta_solver.initialize_temperature)
    t0 = time.time()
    meta_stats = meta_solver.solve_picard()
    meta_wall = time.time() - t0
    meta_vrms = _rms_velocity_from_solver(meta_solver)
    block_stats = meta_solver._block_solver.get_stats() if hasattr(meta_solver, '_block_solver') else {}
    nusselt_relative_error = _relative_error(meta_stats['nusselt'], reference_nusselt)
    rms_velocity_relative_error = _relative_error(meta_vrms, reference_vrms)
    velocity_field_relative_error = _velocity_field_relative_error(meta_solver, reference_solver)
    meta_valid = bool(
        np.isfinite(meta_vrms) and np.isfinite(meta_stats['nusselt']) and
        nusselt_relative_error <= config.nusselt_relative_tolerance and
        rms_velocity_relative_error <= config.rms_velocity_relative_tolerance and
        velocity_field_relative_error <= config.velocity_field_relative_tolerance and
        meta_stats['linear_relative_residual'] <= config.linear_residual_tolerance
    )
    blocked_solver_valid = bool(
        meta_valid and block_stats.get('velocity_direct_fallbacks', 0) == 0 and
        block_stats.get('full_direct_fallbacks', 0) == 0
    )

    return {
        'reference_direct': {
            'wall_time': float(reference_wall),
            'picard_iterations': len(reference_sequence),
            'nusselt': reference_nusselt,
            'rms_velocity': reference_vrms,
            'linear_relative_residual': float(
                np.linalg.norm(reference_sequence[-1]['rhs'] -
                               reference_sequence[-1]['full_matrix'] @ reference_solver.velocity) /
                max(np.linalg.norm(reference_sequence[-1]['rhs']), 1e-14)
            ),
        },
        'meta_blocked': {
            'wall_time': float(meta_wall),
            'picard_iterations': int(meta_stats['n_iterations']),
            'nusselt': float(meta_stats['nusselt']),
            'rms_velocity': meta_vrms,
            'valid_physics': meta_valid,
            'blocked_solver_valid': blocked_solver_valid,
            'nusselt_relative_error': nusselt_relative_error,
            'rms_velocity_relative_error': rms_velocity_relative_error,
            'velocity_field_relative_error': velocity_field_relative_error,
            'linear_relative_residual': float(meta_stats['linear_relative_residual']),
            'block_stats': block_stats,
            'solver_stats': meta_solver.meta_amg.get_stats() if meta_solver.meta_amg is not None else {},
        },
    }


def benchmark_blocked_stokes_methods(config: ExperimentConfig, *, seed: int = 0) -> Dict:
    common = dict(
        nx=config.stokes_nx,
        ny=config.stokes_ny,
        rayleigh=config.stokes_rayleigh,
        viscosity_contrast=config.stokes_viscosity_contrast,
        max_picard_iterations=config.stokes_picard_iterations,
        picard_tolerance=0.0,
    )

    def build_solver(use_meta_amg: bool, trained_meta=None):
        cfg = StokesConfig(
            **common,
            use_meta_amg=use_meta_amg and trained_meta is None,
            meta_training_sequences=config.meta_tasks,
            meta_training_epochs=config.meta_epochs,
            meta_adapt_steps=config.meta_adapt_steps,
            pressure_solver=config.pressure_solver,
            schur_velocity_inverse=config.schur_velocity_inverse,
            meta_training_max_matrix_size=config.meta_training_max_matrix_size,
            velocity_solver_max_iterations=config.velocity_solver_max_iterations,
        )
        solver = PicardStokesSolver(cfg)
        if trained_meta is not None:
            solver.meta_amg = trained_meta
        return solver

    traditional = _with_reproducible_seed(
        seed, lambda: build_solver(False))
    meta = _with_reproducible_seed(
        seed, lambda: build_solver(True))
    _with_seed(seed, traditional.initialize_temperature)
    _with_seed(seed, meta.initialize_temperature)
    initial_temperature_difference = float(
        np.linalg.norm(meta.temperature - traditional.temperature))

    traditional_start = time.time()
    traditional_result = traditional.solve_picard()
    traditional_wall = time.time() - traditional_start

    meta_start = time.time()
    meta_result = meta.solve_picard()
    meta_wall = time.time() - meta_start

    traditional_stats = traditional._block_solver.get_stats()
    meta_stats = meta._block_solver.get_stats()
    velocity_error = _velocity_field_relative_error(meta, traditional)
    return {
        'seed': seed,
        'initial_temperature_difference': initial_temperature_difference,
        'traditional_blocked': {
            'wall_time': float(traditional_wall),
            'linear_relative_residual': float(traditional_result['linear_relative_residual']),
            'nusselt': float(traditional_result['nusselt']),
            'rms_velocity': _rms_velocity_from_solver(traditional),
            'block_stats': traditional_stats,
        },
        'meta_blocked': {
            'wall_time': float(meta_wall),
            'linear_relative_residual': float(meta_result['linear_relative_residual']),
            'nusselt': float(meta_result['nusselt']),
            'rms_velocity': _rms_velocity_from_solver(meta),
            'velocity_field_relative_error': velocity_error,
            'block_stats': meta_stats,
        },
    }


def train_stokes_meta_amg_for_blocked_benchmark(config: ExperimentConfig, *, seed: int = 0):
    cfg = StokesConfig(
        nx=config.stokes_nx,
        ny=config.stokes_ny,
        rayleigh=config.stokes_rayleigh,
        viscosity_contrast=config.stokes_viscosity_contrast,
        max_picard_iterations=config.stokes_picard_iterations,
        picard_tolerance=0.0,
        use_meta_amg=True,
        meta_training_sequences=config.meta_tasks,
        meta_training_epochs=config.meta_epochs,
        meta_adapt_steps=config.meta_adapt_steps,
        pressure_solver=config.pressure_solver,
        schur_velocity_inverse=config.schur_velocity_inverse,
        meta_training_max_matrix_size=config.meta_training_max_matrix_size,
        velocity_solver_max_iterations=config.velocity_solver_max_iterations,
    )
    solver = _with_reproducible_seed(seed, lambda: PicardStokesSolver(cfg))
    return solver.meta_amg


def benchmark_blocked_stokes_with_trained_meta(config: ExperimentConfig, trained_meta,
                                               *, seed: int = 0) -> Dict:
    common = dict(
        nx=config.stokes_nx,
        ny=config.stokes_ny,
        rayleigh=config.stokes_rayleigh,
        viscosity_contrast=config.stokes_viscosity_contrast,
        max_picard_iterations=config.stokes_picard_iterations,
        picard_tolerance=0.0,
    )

    def build_solver(use_meta_amg: bool):
        cfg = StokesConfig(
            **common,
            use_meta_amg=False,
            meta_training_sequences=config.meta_tasks,
            meta_training_epochs=config.meta_epochs,
            meta_adapt_steps=config.meta_adapt_steps,
            pressure_solver=config.pressure_solver,
            schur_velocity_inverse=config.schur_velocity_inverse,
            meta_training_max_matrix_size=config.meta_training_max_matrix_size,
            velocity_solver_max_iterations=config.velocity_solver_max_iterations,
        )
        solver = PicardStokesSolver(cfg)
        if use_meta_amg:
            solver.meta_amg = trained_meta
        return solver

    traditional = _with_reproducible_seed(seed, lambda: build_solver(False))
    meta = _with_reproducible_seed(seed, lambda: build_solver(True))
    _with_seed(seed, traditional.initialize_temperature)
    _with_seed(seed, meta.initialize_temperature)
    initial_temperature_difference = float(np.linalg.norm(meta.temperature - traditional.temperature))

    traditional_start = time.time()
    traditional_result = traditional.solve_picard()
    traditional_wall = time.time() - traditional_start

    meta_start = time.time()
    meta_result = meta.solve_picard()
    meta_wall = time.time() - meta_start

    traditional_stats = traditional._block_solver.get_stats()
    meta_stats = meta._block_solver.get_stats()
    return {
        'seed': seed,
        'adapt_steps': config.meta_adapt_steps,
        'initial_temperature_difference': initial_temperature_difference,
        'traditional_blocked': {
            'wall_time': float(traditional_wall),
            'linear_relative_residual': float(traditional_result['linear_relative_residual']),
            'nusselt': float(traditional_result['nusselt']),
            'rms_velocity': _rms_velocity_from_solver(traditional),
            'block_stats': traditional_stats,
        },
        'meta_blocked': {
            'wall_time': float(meta_wall),
            'linear_relative_residual': float(meta_result['linear_relative_residual']),
            'nusselt': float(meta_result['nusselt']),
            'rms_velocity': _rms_velocity_from_solver(meta),
            'velocity_field_relative_error': _velocity_field_relative_error(meta, traditional),
            'block_stats': meta_stats,
        },
    }


# ============================================================================
# Experiment 1: 收敛性验证
# ============================================================================

def experiment_1_convergence(config: ExperimentConfig) -> Dict:
    """
    验证MAML适配的AMG是否能收敛到与传统AMG相同的解。

    Metric: 相对传统AMG的解误差、残差、迭代数与fallback率
    """
    print("=" * 60)
    print("Experiment 1: Convergence Verification")
    print("=" * 60)

    cfg = MetaAMGConfig(
        min_matrix_size=64, max_matrix_size=400,
        num_training_sequences=config.meta_tasks, sequence_length=8,
        num_meta_epochs=config.meta_epochs, inner_steps=3,
        meta_batch_size=4, hidden_dim=32, num_layers=2,
    )

    meta = MetaAMG(cfg)
    meta.train(num_sequences=config.meta_tasks)

    results = []

    for n in config.matrix_sizes[:4]:  # 前4个规模
        gen = MatrixSequenceGenerator(cfg)
        seq = gen.generate_sequence(n=n, pattern='slab',
                                    length=config.sequence_length, contrast=1e4)
        matrices = [s['matrix'] for s in seq]

        x_exact = np.random.randn(n)
        rhs_list = _rhs_list_from_exact_solution(matrices, x_exact)
        benchmark = benchmark_sequence_methods(
            cfg, matrices, rhs_list, meta,
            periodic_rebuild_interval=config.periodic_rebuild_interval,
            matrix_change_threshold=config.matrix_change_threshold,
            compact=False)

        method_errors = {}
        for method in ['reuse', 'zero_shot', 'adapted']:
            errors = []
            for ref_step, test_step in zip(benchmark['traditional']['steps'], benchmark[method]['steps']):
                err = np.linalg.norm(test_step['solution'] - ref_step['solution']) / (
                    np.linalg.norm(ref_step['solution']) + 1e-12)
                errors.append(float(err))
            method_errors[method] = {
                'errors': errors,
                'mean_error': float(np.mean(errors)),
                'max_error': float(np.max(errors)),
                'iterations_mean': benchmark[method]['iterations_mean'],
                'residual_norm_mean': benchmark[method]['residual_norm_mean'],
                'fallback_rate': benchmark[method]['fallback_rate'],
            }

        result = {
            'matrix_size': n,
            'traditional': {
                'iterations_mean': benchmark['traditional']['iterations_mean'],
                'residual_norm_mean': benchmark['traditional']['residual_norm_mean'],
                'total_time_mean': benchmark['traditional']['total_time_mean'],
            },
            'reuse': method_errors['reuse'],
            'zero_shot': method_errors['zero_shot'],
            'adapted': method_errors['adapted'],
        }
        results.append(result)
        print(f"  n={n:4d}: reuse_err={method_errors['reuse']['mean_error']:.2e}, "
              f"zs_err={method_errors['zero_shot']['mean_error']:.2e}, "
              f"adapt_err={method_errors['adapted']['mean_error']:.2e}, "
              f"adapt_fallback={method_errors['adapted']['fallback_rate']:.2f}")

    return {'experiment_1': results}


# ============================================================================
# Experiment 2: Setup 加速比
# ============================================================================

def experiment_2_setup_speedup(config: ExperimentConfig) -> Dict:
    """
    对比 MAML-AMG adapt vs 传统AMG setup 的 wall-clock 时间。

    Metric: 各方法的 setup/solve/total 时间与 fallback 率
    """
    print("=" * 60)
    print("Experiment 2: Setup Speedup")
    print("=" * 60)

    cfg = MetaAMGConfig(
        min_matrix_size=64, max_matrix_size=500,
        num_training_sequences=config.meta_tasks, sequence_length=8,
        num_meta_epochs=config.meta_epochs, inner_steps=3,
        meta_batch_size=4, hidden_dim=32, num_layers=2,
    )

    meta = MetaAMG(cfg)
    meta.train(num_sequences=config.meta_tasks)

    results = []

    for n in config.matrix_sizes:
        per_seed = []
        for seed in config.seeds:
            def build_case():
                gen = MatrixSequenceGenerator(cfg)
                seq = gen.generate_sequence(n=n, pattern='slab',
                                            length=config.sequence_length, contrast=1e4)
                matrices = [s['matrix'] for s in seq]
                x_exact = np.random.randn(n)
                rhs_list = _rhs_list_from_exact_solution(matrices, x_exact)
                return benchmark_sequence_methods(
                    cfg, matrices, rhs_list, meta,
                    periodic_rebuild_interval=config.periodic_rebuild_interval,
                    matrix_change_threshold=config.matrix_change_threshold)

            per_seed.append(_with_seed(seed, build_case))

        aggregated = _aggregate_method_summaries(per_seed)

        result = {
            'matrix_size': n,
            'traditional': aggregated['traditional'],
            'reuse': aggregated['reuse'],
            'zero_shot': aggregated['zero_shot'],
            'adapted': aggregated['adapted'],
            'adapt_speedup_vs_traditional': float(
                aggregated['traditional']['setup_time_mean_mean'] /
                max(aggregated['adapted']['setup_time_mean_mean'], 1e-10)
            ),
            'reuse_speedup_vs_traditional': float(
                aggregated['traditional']['setup_time_mean_mean'] /
                max(aggregated['reuse']['setup_time_mean_mean'], 1e-10)
            ),
        }
        results.append(result)
        print(f"  n={n:4d}: trad_setup={aggregated['traditional']['setup_time_mean_mean']:.4f}s, "
              f"reuse_setup={aggregated['reuse']['setup_time_mean_mean']:.4f}s, "
              f"adapt_setup={aggregated['adapted']['setup_time_mean_mean']:.4f}s, "
              f"adapt_speedup={result['adapt_speedup_vs_traditional']:.1f}x")

    stokes_replay_per_seed = []
    stokes_full_per_seed = []
    for seed in config.seeds:
        stokes_cfg, stokes_meta, stokes_sequence, stokes_matrices, stokes_rhs = \
            build_stokes_picard_benchmark(config, seed=seed)
        stokes_replay_per_seed.append(benchmark_sequence_methods(
            stokes_cfg, stokes_matrices, stokes_rhs, stokes_meta,
            periodic_rebuild_interval=config.periodic_rebuild_interval,
            matrix_change_threshold=config.matrix_change_threshold))
        stokes_full_per_seed.append(benchmark_full_stokes_picard(config, seed=seed))

    stokes_replay = _aggregate_method_summaries(stokes_replay_per_seed)
    stokes_full = {
        'reference_direct': {
            'n_seeds': len(stokes_full_per_seed),
            **_aggregate_metric_dict([entry['reference_direct'] for entry in stokes_full_per_seed],
                                     ['wall_time', 'nusselt', 'rms_velocity', 'linear_relative_residual']),
            'picard_iterations': [entry['reference_direct']['picard_iterations'] for entry in stokes_full_per_seed],
        },
        'meta_blocked': {
            'n_seeds': len(stokes_full_per_seed),
            **_aggregate_metric_dict([entry['meta_blocked'] for entry in stokes_full_per_seed],
                                     ['wall_time', 'nusselt', 'rms_velocity', 'linear_relative_residual',
                                      'nusselt_relative_error', 'rms_velocity_relative_error',
                                      'velocity_field_relative_error']),
            'valid_physics_rate': float(np.mean([1.0 if entry['meta_blocked']['valid_physics'] else 0.0
                                                 for entry in stokes_full_per_seed])),
            'blocked_solver_valid_rate': float(np.mean([
                1.0 if entry['meta_blocked']['blocked_solver_valid'] else 0.0
                for entry in stokes_full_per_seed])),
            'block_stats': {
                'velocity_solve_calls_mean': float(np.mean([entry['meta_blocked']['block_stats']['velocity_solve_calls'] for entry in stokes_full_per_seed])),
                'velocity_direct_fallbacks_mean': float(np.mean([entry['meta_blocked']['block_stats']['velocity_direct_fallbacks'] for entry in stokes_full_per_seed])),
                'velocity_cg_iterations_mean': float(np.mean([entry['meta_blocked']['block_stats']['velocity_cg_iterations'] for entry in stokes_full_per_seed])),
                'velocity_krylov_iterations_mean': float(np.mean([entry['meta_blocked']['block_stats']['velocity_krylov_iterations'] for entry in stokes_full_per_seed])),
                'full_direct_fallbacks_mean': float(np.mean([entry['meta_blocked']['block_stats']['full_direct_fallbacks'] for entry in stokes_full_per_seed])),
                'velocity_direct_fallback_time_mean': float(np.mean([entry['meta_blocked']['block_stats']['velocity_direct_fallback_time'] for entry in stokes_full_per_seed])),
                'full_direct_fallback_time_mean': float(np.mean([entry['meta_blocked']['block_stats']['full_direct_fallback_time'] for entry in stokes_full_per_seed])),
                'pressure_krylov_iterations_mean': float(np.mean([entry['meta_blocked']['block_stats']['pressure_krylov_iterations'] for entry in stokes_full_per_seed])),
                'pressure_solver_fallbacks_mean': float(np.mean([entry['meta_blocked']['block_stats']['pressure_solver_fallbacks'] for entry in stokes_full_per_seed])),
                'pressure_preconditioner_fallbacks_mean': float(np.mean([entry['meta_blocked']['block_stats']['pressure_preconditioner_fallbacks'] for entry in stokes_full_per_seed])),
            },
        },
    }

    print("\n  [stokes_picard] replay setup means:")
    print(f"    traditional={stokes_replay['traditional']['setup_time_mean_mean']:.4f}s | "
          f"reuse={stokes_replay['reuse']['setup_time_mean_mean']:.4f}s | "
          f"adapted={stokes_replay['adapted']['setup_time_mean_mean']:.4f}s")
    print("  [stokes_picard] full Picard:")
    print(f"    direct wall={stokes_full['reference_direct']['wall_time_mean']:.4f}s | "
          f"meta-block wall={stokes_full['meta_blocked']['wall_time_mean']:.4f}s | "
          f"meta_valid_rate={stokes_full['meta_blocked']['valid_physics_rate']:.2f}")

    return {
        'experiment_2': results,
        'experiment_2_stokes_picard_replay': stokes_replay,
        'experiment_2_stokes_picard_full': stokes_full,
    }


# ============================================================================
# Experiment 3: 消融实验
# ============================================================================

def experiment_3_ablation(config: ExperimentConfig) -> Dict:
    """
    消融实验:
      (a) 零样本 vs MAML适配
      (b) 不同适配步数的影响
      (c) First-order vs Second-order MAML
    """
    print("=" * 60)
    print("Experiment 3: Ablation Study")
    print("=" * 60)

    n = 400  # 固定矩阵规模

    # 3a: Zero-shot vs Adapted
    print("\n[3a] Zero-shot vs MAML Adaptation")
    cfg = MetaAMGConfig(
        min_matrix_size=64, max_matrix_size=400,
        num_training_sequences=config.meta_tasks, sequence_length=8,
        num_meta_epochs=config.meta_epochs, inner_steps=3,
        meta_batch_size=4, hidden_dim=32, num_layers=2,
    )
    meta = MetaAMG(cfg)
    meta.train(num_sequences=config.meta_tasks)

    gen = MatrixSequenceGenerator(cfg)
    seq = gen.generate_sequence(n=n, pattern='slab', length=8, contrast=1e4)

    zs_errors = []
    ad_errors = []
    trad_times = []
    zs_times = []
    ad_times = []

    for k in range(1, len(seq)):
        A_prev, A_curr = seq[k - 1]['matrix'], seq[k]['matrix']

        # Ground truth C/F for A_prev
        from solvers.multigrid_solver import AdaptiveCoarsening
        cp, fp = AdaptiveCoarsening.algebraic_coarsening(A_prev, 0.25)
        cf_prev = np.zeros(A_prev.shape[0]); cf_prev[cp] = 1.0

        # Traditional AMG on A_curr
        t0 = time.time()
        trad_cp, trad_fp = AdaptiveCoarsening.algebraic_coarsening(A_curr, 0.25)
        trad_times.append(time.time() - t0)

        # Zero-shot on A_curr
        t0 = time.time()
        zs_cp, zs_fp = meta.adapter.zero_shot_predict(A_curr)
        zs_times.append(time.time() - t0)
        zs_err = 1.0 - len(set(zs_cp) & set(trad_cp)) / max(len(trad_cp), 1)
        zs_errors.append(zs_err)

        # Adapted on A_curr
        t0 = time.time()
        ad_cp, ad_fp = meta.adapter.adapt(A_prev, cf_prev, A_curr,
                                          adapt_steps=3)
        ad_times.append(time.time() - t0)
        ad_err = 1.0 - len(set(ad_cp) & set(trad_cp)) / max(len(trad_cp), 1)
        ad_errors.append(ad_err)

    result_3a = {
        'zero_shot': {
            'error_mean': float(np.mean(zs_errors)),
            'error_std': float(np.std(zs_errors)),
            'time_mean': float(np.mean(zs_times)),
        },
        'adapted': {
            'error_mean': float(np.mean(ad_errors)),
            'error_std': float(np.std(ad_errors)),
            'time_mean': float(np.mean(ad_times)),
        },
        'traditional': {
            'time_mean': float(np.mean(trad_times)),
        },
    }
    print(f"  Zero-shot: err={np.mean(zs_errors):.4f}, time={np.mean(zs_times):.4f}s")
    print(f"  Adapted:   err={np.mean(ad_errors):.4f}, time={np.mean(ad_times):.4f}s")
    print(f"  Traditional:          time={np.mean(trad_times):.4f}s")

    # 3b: Adaptation steps
    print("\n[3b] Adaptation Steps Ablation")
    steps_results = []
    for steps in config.adapt_steps_list:
        step_errors = []
        step_times = []
        for k in range(1, min(5, len(seq))):
            A_prev, A_curr = seq[k - 1]['matrix'], seq[k]['matrix']
            cp, fp = AdaptiveCoarsening.algebraic_coarsening(A_prev, 0.25)
            cf_prev = np.zeros(A_prev.shape[0]); cf_prev[cp] = 1.0
            trad_cp, _ = AdaptiveCoarsening.algebraic_coarsening(A_curr, 0.25)

            t0 = time.time()
            ad_cp, _ = meta.adapter.adapt(A_prev, cf_prev, A_curr,
                                          adapt_steps=steps)
            step_times.append(time.time() - t0)
            err = 1.0 - len(set(ad_cp) & set(trad_cp)) / max(len(trad_cp), 1)
            step_errors.append(err)

        steps_results.append({
            'steps': steps,
            'error_mean': float(np.mean(step_errors)),
            'error_std': float(np.std(step_errors)),
            'time_mean': float(np.mean(step_times)),
        })
        print(f"  steps={steps:2d}: err={np.mean(step_errors):.4f}, "
              f"time={np.mean(step_times):.4f}s")

    return {
        'experiment_3a_zs_vs_adapted': result_3a,
        'experiment_3b_adapt_steps': steps_results,
    }


# ============================================================================
# Experiment 4: 粘度对比鲁棒性
# ============================================================================

def experiment_4_contrast_robustness(config: ExperimentConfig) -> Dict:
    """
    验证MAML适配在不同粘度对比度下的表现。
    这是地质模拟的关键——极端粘度对比(10^6:1+)是传统AMG的软肋。
    """
    print("=" * 60)
    print("Experiment 4: Contrast Robustness")
    print("=" * 60)

    n = 400
    cfg = MetaAMGConfig(
        min_matrix_size=64, max_matrix_size=400,
        num_training_sequences=config.meta_tasks, sequence_length=8,
        num_meta_epochs=config.meta_epochs, inner_steps=3,
        meta_batch_size=4, hidden_dim=32, num_layers=2,
    )
    meta = MetaAMG(cfg)
    meta.train(num_sequences=config.meta_tasks)

    results = []
    from solvers.multigrid_solver import AdaptiveCoarsening

    for contrast in config.contrasts:
        gen = MatrixSequenceGenerator(cfg)
        seq = gen.generate_sequence(n=n, pattern='slab', length=6,
                                    contrast=contrast)

        zs_errs, ad_errs, trad_iters, ad_iters = [], [], [], []

        for k in range(1, len(seq)):
            A_prev, A_curr = seq[k - 1]['matrix'], seq[k]['matrix']
            cp, fp = AdaptiveCoarsening.algebraic_coarsening(A_prev, 0.25)
            cf_prev = np.zeros(A_prev.shape[0]); cf_prev[cp] = 1.0
            trad_cp, trad_fp = AdaptiveCoarsening.algebraic_coarsening(A_curr, 0.25)

            zs_cp, _ = meta.adapter.zero_shot_predict(A_curr)
            zs_err = 1.0 - len(set(zs_cp) & set(trad_cp)) / max(len(trad_cp), 1)
            zs_errs.append(zs_err)

            ad_cp, ad_fp = meta.adapter.adapt(A_prev, cf_prev, A_curr,
                                              adapt_steps=3)
            ad_err = 1.0 - len(set(ad_cp) & set(trad_cp)) / max(len(trad_cp), 1)
            ad_errs.append(ad_err)

        results.append({
            'contrast': contrast,
            'zs_error_mean': float(np.mean(zs_errs)),
            'zs_error_std': float(np.std(zs_errs)),
            'ad_error_mean': float(np.mean(ad_errs)),
            'ad_error_std': float(np.std(ad_errs)),
        })
        print(f"  contrast={contrast:.0e}: zs_err={np.mean(zs_errs):.4f}, "
              f"ad_err={np.mean(ad_errs):.4f}")

    return {'experiment_4': results}


# ============================================================================
# Experiment 5: 可扩展性 (泛化测试)
# ============================================================================

def experiment_5_scalability(config: ExperimentConfig) -> Dict:
    """
    在小矩阵上训练，测试在大矩阵上的泛化能力。
    """
    print("=" * 60)
    print("Experiment 5: Scalability (Train Small → Test Large)")
    print("=" * 60)

    train_size = config.matrix_sizes[1]  # 225
    test_sizes = [400, 625, 900, 1600]

    cfg = MetaAMGConfig(
        min_matrix_size=64, max_matrix_size=train_size,
        num_training_sequences=config.meta_tasks, sequence_length=8,
        num_meta_epochs=config.meta_epochs, inner_steps=3,
        meta_batch_size=4, hidden_dim=32, num_layers=2,
    )
    meta = MetaAMG(cfg)
    meta.train(num_sequences=config.meta_tasks)

    results = []
    from solvers.multigrid_solver import AdaptiveCoarsening

    for n in test_sizes:
        gen = MatrixSequenceGenerator(cfg)
        seq = gen.generate_sequence(n=n, pattern='slab', length=6, contrast=1e4)

        zs_errs, ad_errs = [], []

        for k in range(1, len(seq)):
            A_prev, A_curr = seq[k - 1]['matrix'], seq[k]['matrix']
            cp, fp = AdaptiveCoarsening.algebraic_coarsening(A_prev, 0.25)
            cf_prev = np.zeros(A_prev.shape[0]); cf_prev[cp] = 1.0
            trad_cp, _ = AdaptiveCoarsening.algebraic_coarsening(A_curr, 0.25)

            zs_cp, _ = meta.adapter.zero_shot_predict(A_curr)
            zs_err = 1.0 - len(set(zs_cp) & set(trad_cp)) / max(len(trad_cp), 1)
            zs_errs.append(zs_err)

            ad_cp, _ = meta.adapter.adapt(A_prev, cf_prev, A_curr,
                                          adapt_steps=3)
            ad_err = 1.0 - len(set(ad_cp) & set(trad_cp)) / max(len(trad_cp), 1)
            ad_errs.append(ad_err)

        results.append({
            'test_size': n,
            'train_size': train_size,
            'zs_error_mean': float(np.mean(zs_errs)),
            'ad_error_mean': float(np.mean(ad_errs)),
            'generalization_gap': float(np.mean(ad_errs) - np.mean(zs_errs)),
        })
        print(f"  test_n={n:4d}: zs_err={np.mean(zs_errs):.4f}, "
              f"ad_err={np.mean(ad_errs):.4f}")

    return {'experiment_5': results}


# ============================================================================
# Experiment 6: 神经AMG vs MAML-AMG 对比
# ============================================================================

def experiment_6_compare_methods(config: ExperimentConfig) -> Dict:
    """
    四种方法的端到端对比:
      (a) 传统AMG: 每步重新setup
      (b) Reuse: 直接复用上一步C/F并验证
      (c) Zero-shot Meta-AMG: 每步独立预测C/F
      (d) MAML-AMG: 从第一步adapt
    """
    print("=" * 60)
    print("Experiment 6: Method Comparison")
    print("=" * 60)

    n = 400
    seq_len = 10

    # Train both models
    nn_cfg = NeuralAMGConfig(
        min_matrix_size=64, max_matrix_size=400,
        num_training_samples=max(60, config.meta_tasks * 2), num_epochs=config.meta_epochs,
        hidden_dim=32, num_layers=2,
    )
    meta_cfg = MetaAMGConfig(
        min_matrix_size=64, max_matrix_size=400,
        num_training_sequences=config.meta_tasks, sequence_length=8,
        num_meta_epochs=config.meta_epochs, inner_steps=3,
        meta_batch_size=4, hidden_dim=32, num_layers=2,
    )

    # Train Neural AMG
    print("\nTraining Neural AMG...")
    nn_gen = AMGDataGenerator(nn_cfg)
    nn_dataset = nn_gen.generate_dataset(num_samples=nn_cfg.num_training_samples)
    neural_amg = NeuralAMG(nn_cfg)
    neural_amg.train(dataset=nn_dataset)

    # Train Meta AMG
    print("\nTraining Meta AMG...")
    meta_amg = MetaAMG(meta_cfg)
    meta_amg.train(num_sequences=config.meta_tasks)

    per_seed = []
    neural_setup_per_seed = []
    for seed in config.seeds:
        def build_case():
            gen = MatrixSequenceGenerator(meta_cfg)
            seq = gen.generate_sequence(n=n, pattern='slab', length=seq_len, contrast=1e4)
            matrices = [s['matrix'] for s in seq]
            x_exact = np.random.randn(n)
            rhs_list = _rhs_list_from_exact_solution(matrices, x_exact)
            return matrices, rhs_list

        matrices, rhs_list = _with_seed(seed, build_case)
        per_seed.append(benchmark_sequence_methods(
            meta_cfg, matrices, rhs_list, meta_amg,
            periodic_rebuild_interval=config.periodic_rebuild_interval,
            matrix_change_threshold=config.matrix_change_threshold))

        neural_setup_times = []
        for A in matrices:
            t0 = time.time()
            neural_amg.solver.predict_cf_split(A)
            neural_setup_times.append(time.time() - t0)
        neural_setup_per_seed.append({
            'setup_time_mean': float(np.mean(neural_setup_times)),
            'setup_time_std': float(np.std(neural_setup_times)),
        })

    results = _aggregate_method_summaries(per_seed)

    results['neural_amg_setup_only'] = {
        'n_seeds': len(neural_setup_per_seed),
        'setup_time_mean_mean': float(np.mean([r['setup_time_mean'] for r in neural_setup_per_seed])),
        'setup_time_mean_std': float(np.std([r['setup_time_mean'] for r in neural_setup_per_seed])),
    }

    print(f"\n  Traditional total: {results['traditional']['total_time_mean_mean']:.4f}s")
    print(f"  Reuse total:       {results['reuse']['total_time_mean_mean']:.4f}s")
    print(f"  Zero-shot total:   {results['zero_shot']['total_time_mean_mean']:.4f}s")
    print(f"  Adapted total:     {results['adapted']['total_time_mean_mean']:.4f}s")
    print(f"  Neural setup-only: {results['neural_amg_setup_only']['setup_time_mean_mean']:.4f}s")

    stokes_replay_per_seed = []
    stokes_full_per_seed = []
    for seed in config.seeds:
        stokes_cfg, stokes_meta, stokes_sequence, stokes_matrices, stokes_rhs = \
            build_stokes_picard_benchmark(config, seed=seed)
        stokes_replay_per_seed.append(benchmark_sequence_methods(
            stokes_cfg, stokes_matrices, stokes_rhs, stokes_meta,
            periodic_rebuild_interval=config.periodic_rebuild_interval,
            matrix_change_threshold=config.matrix_change_threshold))
        stokes_full_per_seed.append(benchmark_full_stokes_picard(config, seed=seed))

    stokes_replay = _aggregate_method_summaries(stokes_replay_per_seed)
    stokes_full = {
        'reference_direct': {
            'n_seeds': len(stokes_full_per_seed),
            **_aggregate_metric_dict([entry['reference_direct'] for entry in stokes_full_per_seed],
                                     ['wall_time', 'nusselt', 'rms_velocity', 'linear_relative_residual']),
            'picard_iterations': [entry['reference_direct']['picard_iterations'] for entry in stokes_full_per_seed],
        },
        'meta_blocked': {
            'n_seeds': len(stokes_full_per_seed),
            **_aggregate_metric_dict([entry['meta_blocked'] for entry in stokes_full_per_seed],
                                     ['wall_time', 'nusselt', 'rms_velocity', 'linear_relative_residual',
                                      'nusselt_relative_error', 'rms_velocity_relative_error',
                                      'velocity_field_relative_error']),
            'valid_physics_rate': float(np.mean([1.0 if entry['meta_blocked']['valid_physics'] else 0.0
                                                 for entry in stokes_full_per_seed])),
            'blocked_solver_valid_rate': float(np.mean([
                1.0 if entry['meta_blocked']['blocked_solver_valid'] else 0.0
                for entry in stokes_full_per_seed])),
            'block_stats': {
                'velocity_solve_calls_mean': float(np.mean([entry['meta_blocked']['block_stats']['velocity_solve_calls'] for entry in stokes_full_per_seed])),
                'velocity_direct_fallbacks_mean': float(np.mean([entry['meta_blocked']['block_stats']['velocity_direct_fallbacks'] for entry in stokes_full_per_seed])),
                'velocity_cg_iterations_mean': float(np.mean([entry['meta_blocked']['block_stats']['velocity_cg_iterations'] for entry in stokes_full_per_seed])),
                'velocity_krylov_iterations_mean': float(np.mean([entry['meta_blocked']['block_stats']['velocity_krylov_iterations'] for entry in stokes_full_per_seed])),
                'full_direct_fallbacks_mean': float(np.mean([entry['meta_blocked']['block_stats']['full_direct_fallbacks'] for entry in stokes_full_per_seed])),
                'velocity_direct_fallback_time_mean': float(np.mean([entry['meta_blocked']['block_stats']['velocity_direct_fallback_time'] for entry in stokes_full_per_seed])),
                'full_direct_fallback_time_mean': float(np.mean([entry['meta_blocked']['block_stats']['full_direct_fallback_time'] for entry in stokes_full_per_seed])),
                'pressure_krylov_iterations_mean': float(np.mean([entry['meta_blocked']['block_stats']['pressure_krylov_iterations'] for entry in stokes_full_per_seed])),
                'pressure_solver_fallbacks_mean': float(np.mean([entry['meta_blocked']['block_stats']['pressure_solver_fallbacks'] for entry in stokes_full_per_seed])),
                'pressure_preconditioner_fallbacks_mean': float(np.mean([entry['meta_blocked']['block_stats']['pressure_preconditioner_fallbacks'] for entry in stokes_full_per_seed])),
            },
        },
    }

    print("\n  [stokes_picard replay] total times:")
    print(f"    traditional={stokes_replay['traditional']['total_time_mean_mean']:.4f}s | "
          f"reuse={stokes_replay['reuse']['total_time_mean_mean']:.4f}s | "
          f"zero-shot={stokes_replay['zero_shot']['total_time_mean_mean']:.4f}s | "
          f"adapted={stokes_replay['adapted']['total_time_mean_mean']:.4f}s")
    print("  [stokes_picard full] physical metrics:")
    print(f"    Nu direct={stokes_full['reference_direct']['nusselt_mean']:.3f} | "
          f"Nu meta={stokes_full['meta_blocked']['nusselt_mean']:.3f} | "
          f"vrms direct={stokes_full['reference_direct']['rms_velocity_mean']:.3e} | "
          f"vrms meta={stokes_full['meta_blocked']['rms_velocity_mean']:.3e} | "
          f"meta_valid={stokes_full['meta_blocked']['valid_physics_rate']:.2f}")

    return {
        'experiment_6': results,
        'experiment_6_stokes_picard_replay': stokes_replay,
        'experiment_6_stokes_picard_full': stokes_full,
    }


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Meta-AMG Paper Experiments')
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'],
                        help='Experiment to run')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (reduced parameters)')
    parser.add_argument('--preset', type=str, default=None,
                        choices=['quick', 'paper_medium', 'paper_large'],
                        help='Named experiment preset')
    parser.add_argument('--output', type=str, default='experiments/results',
                        help='Output directory')
    args = parser.parse_args()

    config = ExperimentConfig(output_dir=args.output)

    if args.preset is not None:
        apply_preset(config, args.preset)
    if args.quick:
        apply_preset(config, 'quick')

    Path(args.output).mkdir(parents=True, exist_ok=True)
    all_results = {'config': {
        'preset': args.preset or ('quick' if args.quick else 'custom'),
        'meta_epochs': config.meta_epochs,
        'meta_tasks': config.meta_tasks,
        'n_runs': config.n_runs,
        'sequence_length': config.sequence_length,
        'matrix_sizes': config.matrix_sizes,
        'contrasts': config.contrasts,
        'stokes_nx': config.stokes_nx,
        'stokes_ny': config.stokes_ny,
        'stokes_picard_iterations': config.stokes_picard_iterations,
        'stokes_rayleigh': config.stokes_rayleigh,
        'stokes_viscosity_contrast': config.stokes_viscosity_contrast,
        'periodic_rebuild_interval': config.periodic_rebuild_interval,
        'matrix_change_threshold': config.matrix_change_threshold,
        'nusselt_relative_tolerance': config.nusselt_relative_tolerance,
        'rms_velocity_relative_tolerance': config.rms_velocity_relative_tolerance,
        'velocity_field_relative_tolerance': config.velocity_field_relative_tolerance,
        'linear_residual_tolerance': config.linear_residual_tolerance,
    }}

    experiments = {
        'e1': experiment_1_convergence,
        'e2': experiment_2_setup_speedup,
        'e3': experiment_3_ablation,
        'e4': experiment_4_contrast_robustness,
        'e5': experiment_5_scalability,
        'e6': experiment_6_compare_methods,
    }

    if args.exp == 'all':
        for name, func in experiments.items():
            try:
                result = func(config)
                all_results.update(result)
            except Exception as e:
                print(f"  Experiment {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
    else:
        result = experiments[args.exp](config)
        all_results.update(result)

    # Save results
    output_path = Path(args.output) / 'results.json'
    with open(output_path, 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
