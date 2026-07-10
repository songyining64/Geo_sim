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


# ============================================================================
# 配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """实验总配置"""
    output_dir: str = "experiments/results"
    n_runs: int = 3
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


# ============================================================================
# Experiment 1: 收敛性验证
# ============================================================================

def experiment_1_convergence(config: ExperimentConfig) -> Dict:
    """
    验证MAML适配的AMG是否能收敛到与传统AMG相同的解。

    Metric: 相对误差 ||x_adapted - x_traditional|| / ||x_traditional||
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
        b = matrices[0] @ x_exact

        # 传统AMG求解每一步
        trad_mg = MultigridConfig(tolerance=1e-8, max_iterations=100)
        trad_sols = []
        for A in matrices:
            solver = AlgebraicMultigridSolver(trad_mg)
            x = solver.solve(A, A @ x_exact)
            trad_sols.append(x)

        # Meta-AMG求解
        meta_sols = meta.solve_sequence(matrices, b)

        errors = []
        for k, (xt, xm) in enumerate(zip(trad_sols, meta_sols)):
            if xm is not None:
                err = np.linalg.norm(xm - xt) / (np.linalg.norm(xt) + 1e-12)
                errors.append(err)

        result = {
            'matrix_size': n,
            'errors': errors,
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
        }
        results.append(result)
        print(f"  n={n:4d}: mean_err={np.mean(errors):.2e}, "
              f"max_err={np.max(errors):.2e}")

    return {'experiment_1': results}


# ============================================================================
# Experiment 2: Setup 加速比
# ============================================================================

def experiment_2_setup_speedup(config: ExperimentConfig) -> Dict:
    """
    对比 MAML-AMG adapt vs 传统AMG setup 的 wall-clock 时间。

    Metric: Speedup = T_traditional / T_adapted
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
        gen = MatrixSequenceGenerator(cfg)
        seq = gen.generate_sequence(n=n, pattern='slab',
                                    length=config.sequence_length, contrast=1e4)
        matrices = [s['matrix'] for s in seq]
        b = matrices[0] @ np.ones(n)

        # Run sequence
        _ = meta.solve_sequence(matrices, b)
        stats = meta.get_stats()

        # Benchmark individual steps
        trad_times = []
        adapt_times = []

        for run in range(config.n_runs):
            solver = MetaAMGSolver(cfg)
            solver.set_adapter(meta.adapter)
            solver.solve_sequence(matrices, b)
            s = solver.get_stats()

            if s['n_traditional'] > 0:
                trad_times.append(s['avg_traditional_time'])
            if s['n_adapted'] > 0:
                adapt_times.append(s['avg_adapt_time'])

        result = {
            'matrix_size': n,
            'trad_time_mean': float(np.mean(trad_times)),
            'trad_time_std': float(np.std(trad_times)),
            'adapt_time_mean': float(np.mean(adapt_times)),
            'adapt_time_std': float(np.std(adapt_times)),
            'speedup': float(np.mean(trad_times) / max(np.mean(adapt_times), 1e-10)),
        }
        results.append(result)
        print(f"  n={n:4d}: trad={np.mean(trad_times):.4f}s, "
              f"adapt={np.mean(adapt_times):.4f}s, "
              f"speedup={result['speedup']:.1f}x")

    return {'experiment_2': results}


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
    三种方法的端到端对比:
      (a) 传统AMG: 每步重新setup
      (b) 神经AMG: 每步独立预测C/F
      (c) MAML-AMG: 从第一步adapt
    """
    print("=" * 60)
    print("Experiment 6: Method Comparison")
    print("=" * 60)

    n = 400
    seq_len = 10

    # Train both models
    nn_cfg = NeuralAMGConfig(
        min_matrix_size=64, max_matrix_size=400,
        num_training_samples=200, num_epochs=30,
        hidden_dim=32, num_layers=2,
    )
    meta_cfg = MetaAMGConfig(
        min_matrix_size=64, max_matrix_size=400,
        num_training_sequences=100, sequence_length=8,
        num_meta_epochs=30, inner_steps=3,
        meta_batch_size=4, hidden_dim=32, num_layers=2,
    )

    # Train Neural AMG
    print("\nTraining Neural AMG...")
    nn_gen = AMGDataGenerator(nn_cfg)
    nn_dataset = nn_gen.generate_dataset(num_samples=200)
    neural_amg = NeuralAMG(nn_cfg)
    neural_amg.train(dataset=nn_dataset)

    # Train Meta AMG
    print("\nTraining Meta AMG...")
    meta_amg = MetaAMG(meta_cfg)
    meta_amg.train(num_sequences=100)

    # Generate test sequence
    gen = MatrixSequenceGenerator(meta_cfg)
    seq = gen.generate_sequence(n=n, pattern='slab', length=seq_len, contrast=1e4)
    matrices = [s['matrix'] for s in seq]
    b = matrices[0] @ np.ones(n)

    from solvers.multigrid_solver import AdaptiveCoarsening

    results = {
        'traditional': {'setup_times': [], 'solve_iters': []},
        'neural_amg': {'setup_times': [], 'solve_iters': []},
        'meta_amg': {'setup_times': [], 'solve_iters': []},
    }

    trad_mg = MultigridConfig(tolerance=1e-8, max_iterations=100)

    for k, A in enumerate(matrices):
        # (a) Traditional
        t0 = time.time()
        trad_solver = AlgebraicMultigridSolver(trad_mg)
        x_trad = trad_solver.solve(A, b)
        results['traditional']['setup_times'].append(time.time() - t0)
        results['traditional']['solve_iters'].append(
            len(getattr(trad_solver, 'performance_stats', {}).get('total_iterations', 0))
        )

        # (b) Neural AMG (zero-shot each step)
        t0 = time.time()
        nn_cp, nn_fp = neural_amg.solver.predict_cf_split(A)
        results['neural_amg']['setup_times'].append(time.time() - t0)

        # (c) Meta AMG (done via solve_sequence)
        # (runs separately)

    # Meta AMG as sequence
    meta_sols = meta_amg.solve_sequence(matrices, b)
    meta_stats = meta_amg.get_stats()

    print(f"\n  Traditional setup: {np.mean(results['traditional']['setup_times']):.4f}s")
    print(f"  Neural AMG setup:  {np.mean(results['neural_amg']['setup_times']):.4f}s")
    print(f"  Meta AMG adapt:    {meta_stats['avg_adapt_time']:.4f}s")

    return {'experiment_6': results}


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
    parser.add_argument('--output', type=str, default='experiments/results',
                        help='Output directory')
    args = parser.parse_args()

    config = ExperimentConfig(output_dir=args.output)

    if args.quick:
        config.meta_epochs = 10
        config.meta_tasks = 30
        config.matrix_sizes = [100, 225, 400]
        config.contrasts = [1e3, 1e5]
        config.n_runs = 1

    Path(args.output).mkdir(parents=True, exist_ok=True)
    all_results = {'config': {
        'meta_epochs': config.meta_epochs,
        'meta_tasks': config.meta_tasks,
        'matrix_sizes': config.matrix_sizes,
        'contrasts': config.contrasts,
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
