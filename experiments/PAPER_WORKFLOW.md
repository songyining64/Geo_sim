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

## 下一轮优先级

1. 稳定 `paper_medium` 的 `stokes full` 结果；
2. 跑 `e6` 的 `paper_medium`；
3. 补 3-5 seeds 汇总；
4. 加图表脚本（setup-vs-size, fallback-vs-method, Nu/RMS bar chart）。
