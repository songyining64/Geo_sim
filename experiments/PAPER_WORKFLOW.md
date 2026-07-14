# Meta-AMG 论文工作流

## 目标

把实验工作分成三层：

1. `synthetic sequence`：证明 setup / total time 行为；
2. `stokes_picard replay`：证明真实 FEM Picard 轨迹上的层级迁移行为；
3. `stokes_picard full`：证明完整物理量 (`Nu`, `RMS velocity`) 是否保持可信。

论文主张建议写成：

`Meta-AMG reduces repeated AMG setup cost on evolving Stokes/Picard matrix sequences while preserving acceptable solver quality and, on stable blocked solves, comparable physical diagnostics.`

## 运行命令

### 1. Smoke test

```bash
python3 experiments/run_experiments.py --exp e2 --quick --output "experiments/results_quick_stokes_e2_v3"
python3 experiments/summarize_results.py "experiments/results_quick_stokes_e2_v3/results.json"
```

用途：检查代码、结果落盘、表格输出。

### 2. 论文中等规模主结果

```bash
python3 experiments/run_experiments.py --exp e2 --preset paper_medium --output "experiments/results_paper_medium_e2"
python3 experiments/summarize_results.py "experiments/results_paper_medium_e2/results.json"
```

用途：生成论文主表初稿。

### 3. 方法对比表

```bash
python3 experiments/run_experiments.py --exp e6 --preset paper_medium --output "experiments/results_paper_medium_e6"
python3 experiments/summarize_results.py "experiments/results_paper_medium_e6/results.json"
```

用途：比较 `traditional / reuse / periodic_rebuild / change_aware_reuse / zero_shot / adapted / neural_amg_setup_only`。

### 4. Blocked Stokes 公平对照

```bash
python3 experiments/benchmark_blocked_comparison.py --preset paper_medium --output experiments/results_blocked_comparison_medium
```

用途：在相同 seed 和初始温度下比较 fresh traditional hierarchy 与 Meta transferred hierarchy，分开报告层级构建、adaptation、速度 Krylov 和完整 wall time。

### 5. Adaptation 步数与规模验证

```bash
python3 experiments/benchmark_blocked_comparison.py --preset paper_medium --adapt-steps 1,2,3 --output experiments/results_blocked_adapt_ablation_medium
python3 experiments/benchmark_blocked_comparison.py --preset paper_medium --adapt-steps 1 --stokes-size 16 --seeds 0,1,2 --output experiments/results_blocked_size16_adapt1
```

用途：证明小规模下 adaptation 固定开销会抵消收益，但 `16x16` 后 `adapt_steps=1` 可以产生 online setup speedup。

### 6. 外部速度块基线

```bash
python3 experiments/benchmark_external_velocity_baselines.py --nx 16 --ny 16 --seed 0 --output experiments/results_external_velocity_baselines
```

用途：生成外部求解器对照。当前本机已可运行 PyAMG 与 PETSc；PyAMG smoothed aggregation 是可用 AMG 基线，PETSc GAMG 仍需调参。

### 7. Schur-Optimized 大规模路径

```bash
python3 experiments/benchmark_blocked_comparison.py --preset paper_medium --adapt-steps 1 --stokes-size 32 --seeds 0,1,2 --pressure-solver matrix_free_schur --schur-velocity-inverse lu --output experiments/results_blocked_size32_schur_lu
```

用途：Schur matvec 每个 Picard 步缓存一次速度块稀疏 LU，最终速度回代仍使用 AMG/Krylov。该模式解决矩阵自由 Schur 重复内部 Krylov的性能瓶颈，但必须与 pure iterative Schur 分表报告。

当前 `32x32` 三 seed 正式结果位于 `experiments/results_blocked_size32_schur_lu/`：总 online preconditioner setup `1.72x` 加速，wall time 下降 `26.6%`，所有 fallback 为零。

## 输出文件

每个结果目录下会包含：

- `results.json`：完整实验结果。
- `summary/summary.md`：Markdown 主表。
- `summary/table_e2_synthetic.csv`：synthetic 序列表。
- `summary/table_stokes_replay.csv`：真实 Picard replay 表。
- `summary/table_stokes_full.csv`：完整物理量表。
- `summary/table_e6_methods.csv`：方法对比表（仅 `e6` 有）。

## 写论文时怎么用

### 正文主表

优先用：

- `table_e2_synthetic.csv`
- `table_stokes_replay.csv`
- `table_stokes_full.csv`

### 正文叙事建议

1. 先写 synthetic：
   setup time 和 total time 的趋势最清楚。

2. 再写 stokes replay：
   这里的重点不是物理量，而是 `fallback_rate` 和真实轨迹上的迁移可行性。

3. 最后写 stokes full：
   这里只能在 `valid_physics = yes` 的情况下声称物理保持。

## 当前结果应如何解读

注意：质量门加入前生成的结果若包含 `Infinity`、`NaN`，或方法跑满最大迭代仍被记为 `accepted`，均为无效旧结果，不能用于论文。必须用当前代码重跑。

### quick (`results_quick_stokes_e2_v3`)

- `synthetic`: `adapted` setup 对传统 AMG 有明显加速。
- `stokes replay`: `adapted` 仍有较高 fallback，说明真实轨迹上迁移不稳定。
- `stokes full`: 当前小 case 上 `meta_blocked` 物理量可信，但 wall time 还未赢直接参考。

### paper_medium (`results_paper_medium_e2`)

- `synthetic`: setup 加速随规模增长而增强。
- `stokes replay`: `reuse / zero_shot / adapted` fallback 都高。
- `stokes full`: `meta_blocked` 在 `8x8` case 上再次失稳 (`valid_physics = no`)；这不能进主结论，只能进局限性和失败分析。

## 论文里必须诚实写的局限

1. `reuse` 是强 baseline，不能省略。
2. `adapted` 并不总能优于 `reuse`。
3. setup speedup 不自动等于 full simulation speedup。
4. blocked Stokes 路径在更大 case 上仍可能失稳，需要更稳的 Schur / pressure treatment。
5. `valid_physics = yes` 若伴随 `full_direct_fallbacks > 0`，只证明兜底后的物理一致性，不证明 blocked 求解器本身稳定。
6. 正文必须同时报告 `valid_physics_rate` 与 `blocked_solver_valid_rate`；后者要求速度块和全系统都没有直接回退。
7. 序列实验必须报告相对残差最大值，不能只用平均残差掩盖单步发散。
8. full Stokes 表必须报告压力 Krylov 迭代、压力求解回退和预条件器回退；wall time 改善只有在这些回退均为零时才可归因于 blocked 路径。
9. setup 必须拆成 hierarchy build 与 gradient adaptation；当前 paper-medium 结果只支持 hierarchy-build speedup，不支持包含 adaptation 的 online-setup speedup。
10. `8x8` 不能作为 online setup speedup 的主证据；`16x16` 三 seed 才显示 `adapt_steps=1` 的 online setup 加速。
11. `schur_velocity_inverse=lu` 是工程优化变体，不得写成 AMG-only；setup 表必须把 Schur-LU factorization 时间计入 total online preconditioner setup。
12. 3D 离散为线性四面体 `P1-P1-P1 + PSPG`，`beta=0.15`；完整求解采用 free-slip，
    manufactured 解采用全边界解析速度 Dirichlet，二者都用单点压力规约。
13. 3D `n=4` 方法表只证明可行性。该规模 direct 更快，不能据此声称 Meta-AMG 具有3D总时间优势。
14. Python 峰值内存不覆盖全部 SciPy/SuperLU 原生分配；论文必须同时保留这一限制。
15. 高黏度表必须区分节点配置对比与实际单元平均黏度对比；性能比较只使用共享预装配
    Picard trajectory，轨迹生成时间不计入任一方法。

## 3D 可复现实验

```bash
python3 experiments/benchmark_stokes_3d_convergence.py \
  --levels 2,3,4,6,8 --output experiments/results_stokes_3d_convergence

python3 experiments/benchmark_stokes_3d_methods.py \
  --n 4 --contrasts 1,1e2,1e4,1e6 --seeds 0,1,2 \
  --picard-iterations 3 --meta-tasks 2 --meta-epochs 1 \
  --output experiments/results_stokes_3d_methods

python3 experiments/summarize_stokes_3d.py

python3 experiments/benchmark_stokes_3d_scaling.py \
  --levels 4,6,8 --contrast 1e4 --seeds 0,1,2 \
  --picard-iterations 3 --meta-tasks 2 --meta-epochs 1 \
  --output experiments/results_stokes_3d_scaling

python3 experiments/benchmark_stokes_3d_scaling.py \
  --levels 10 --contrast 1e4 --seeds 0 --picard-iterations 3 \
  --meta-tasks 2 --meta-epochs 1 \
  --output experiments/results_stokes_3d_scaling --append
```

输出包括原始 JSON、收敛 CSV、求解器均值/标准差 CSV 和 Markdown 摘要。Meta 训练不计入
online setup；训练配置保存在结果 JSON 中。
Scaling 表中 `n=10` 只有一个种子，必须标为扩展点；当前结果没有显示相对 direct 的交叉，
也不能用项目内近二次粗化基线替代成熟外部 AMG 的规模结论。

## 下一轮优先级

1. 稳定 `paper_medium` 的 `stokes full` 结果；
2. 跑 `e6` 的 `paper_medium`；
3. 补 3-5 seeds 汇总；
4. 加图表脚本（setup-vs-size, fallback-vs-method, Nu/RMS bar chart）。
