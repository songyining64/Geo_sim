# 面向 Underworld 级可用性的状态与缺口

本文档以真实可运行路径为准，不把演示代码、接口签名或历史 benchmark 当作完成标准。

## 本轮已闭环

- 真实 FEM Stokes/Picard 轨迹可作为 Meta-AMG 训练源；每个样本保留完整鞍点系统和用于 AMG 的 SPD 速度块。
- Meta-AMG 已接入块 Stokes 求解器的在线速度块路径，并跨新装配矩阵保留适配状态。
- 学习层级必须经过 C/F 覆盖修复、算子复杂度和两网格残差收缩验收；失败则回退传统 AMG。
- MAML 内循环已改为可微参数更新，查询损失能回传到元参数。
- Stokes 图已加入温度、粘度、应变率及矩阵/粘度变化特征；训练以随机残差 probe 收敛损失为主、BCE 为辅助。
- 论文实验已覆盖 synthetic、真实 Picard replay 和完整 Stokes 物理诊断，并统一记录 setup、总时间、迭代数、回退率、Nu 和 RMS velocity。
- 实验与 NeuralAMG 公共求解入口均有有限性和相对残差质量门，不再把发散结果统计为成功。
- 序列 benchmark 已加入定期重建和矩阵变化触发重建两类强基线，并报告相对残差最大值与重建率。
- 完整 Stokes benchmark 分开报告物理一致性与无直接回退的 blocked-solver 有效性，fallback 次数和耗时跨全部 Picard 步累计。
- blocked Stokes 压力路径已改为矩阵自由 Schur 补 `C - B A_u^{-1}G`，并补齐温度块消元；quick case 已实现零直接回退和约 `1e-10` 速度场相对误差。
- 对角速度逆构造的稀疏 Schur/LSC 型预条件器将 quick 压力 Krylov 迭代从约 73 次降至约 24 次；`8x8` case 保持零直接回退。
- 速度 AMG 改用实测更稳的阻尼 Jacobi (`omega=0.5`, 每侧一步)，并缓存粗层稀疏 LU；`8x8` 四步 Picard wall time 从约 6.36s 降至约 2.46s，保持零回退。
- paper-medium 三 seed 公平对照已完成：迁移层级构建时间下降约 39.6%，速度 Krylov 下降约 5.8%，wall time 下降约 6.2%，物理场相对差约 `1e-9`；但计入三步梯度 adaptation 后 online setup 增加约 48.6%。
- adaptation 步数消融已完成：`8x8` 上 `1` 步最优但 online setup 仍慢；`16x16` 三 seed、`adapt_steps=1` 上 online setup 达到 `1.40x` 加速，速度 Krylov 比例约 `0.715`，wall time 下降约 `53.9%`，零直接回退。
- 当前质量门下已重跑 paper-medium E2/E6；Stokes full 的 `valid_physics_rate=1.0`、`blocked_solver_valid_rate=1.0`、速度/全系统直接回退均为零。
- 已安装并验证 PyAMG/PETSc 外部速度块基线：PyAMG smoothed aggregation 残差约 `9.66e-9`，PETSc GMRES+ILU 残差约 `1.40e-8`，PETSc direct LU 残差约 `3.88e-15`；PETSc GAMG 当前需进一步调参。
- Schur-optimized 变体已验证：每个 Picard 步缓存一次速度块稀疏 LU，仅用于 Schur matvec；`32x32` seed0 保持零回退，Meta wall time `18.74s` 对传统 `22.55s`，总 online preconditioner setup（含 LU）约 `1.17x` 加速。
- `32x32` Schur-optimized 三 seed 已完成：总 online preconditioner setup 平均 `1.72x` 加速，速度 Krylov 比例 `0.932`，wall time 平均下降 `26.6%`，速度场误差约 `3.64e-8`，所有 fallback 为零。
- `64x64` seed0 scalability smoke 已完成：Meta 路径零速度/全系统回退，setup `2.27x` 加速；传统层级仍触发 6 次速度直接回退，因此该点只作稳定性与规模趋势证据，不进入公平性能主结论。
- 3D 非结构线性四面体路径已接通真实 blocked Stokes/Picard：三 seed 小网格均零速度/full fallback，线性残差 `5e-11` 到 `1.9e-10`；Poisson manufactured patch 同时保持 `1e-15` 解误差。
- 3D 采用 `P1(u)-P1(p)-P1(T)+PSPG`，`tau=0.15 h_e^2/(2 eta)`；free-slip 完整算例以单点
  `p=0` 处理压力零空间，blocked Schur 路径另有数值正则，二者已在论文说明中分开。
- 3D manufactured Stokes 已完成 `n=2,3,4,6,8` 收敛研究：最细两级观察阶约为
  `L2(u)=1.91`、`H1(u)=0.96`、`L2(p)=1.25`。
- `n=4` 三种子公平 replay 已完成：三种方法重放相同预装配系统，标称节点黏度比
  `1/10²/10⁴/10⁶` 的实际单元对比为约 `1/56/2.66e3/1.06e5`。全部 fallback 为零，
  最大相对 direct 解误差约 `1.9e-8`；direct 在该小规模仍快两个数量级。
- 3D scaling 已覆盖 `n=4/6/8` 三种子和 `n=10` 单种子扩展点（最高 `5324` Stokes DOF）。
  Meta 总时间从约 `2.00s` 增至 `74.40s`，传统 AMG 从 `3.14s` 增至 `183.62s`，direct
  从 `0.016s` 增至 `2.33s`；未出现 direct 交叉点。内部传统粗化的近二次 setup 是主要瓶颈。
- 单元测试覆盖真实轨迹、物理轨迹隔离和小型端到端 Picard 路径。

## 仍然是 P0 的缺口

| 缺口 | 为什么重要 | 下一项可交付 |
|---|---|---|
| 64x64 传统基线仍回退 | Meta 已零回退，但传统速度层级 6/8 次达到 Krylov 上限，无法公平比较 wall time | 引入 PyAMG/PETSc AMG 作为 64x64 传统 blocked 基线，或进一步调优传统层级 |
| 3D 尚缺大规模研究 | 已有解析解收敛阶和小网格三种子对比，但不能外推生产规模性能 | 增加更大网格、独立进程原生内存和外部 PETSc/HYPRE 对比 |
| 内部 AMG scaling 较差 | `n=10` 传统 setup 约 `134s`，不能作为成熟 AMG 的性能代表 | 用稀疏线性复杂度粗化或 PETSc/HYPRE 重做共享 replay |
| PETSc GAMG 还未调优 | PETSc 已可用，但当前 `GAMG` 残差比 PyAMG/ILU 差，不能作为强 AMG 结论 | 调 PETSc `GAMG` options 或改用 HYPRE/BoomerAMG 环境 |
| 直接解仍比小规模 blocked 快 | `8x8` full Stokes direct 约 0.27s，blocked 约 2.1s；这不是 AMG 目标规模 | 论文中把 direct 解作为精度参考，小规模性能主张只用 blocked-vs-blocked 和更大规模趋势 |

## P1 工程缺口

- 多数 GPU、并行、可视化与多物理模块仍是原型或未接入主求解路径；README 中的历史性能数字不能作为当前版本结论。
- 2D 和小型结构化 3D 已接入 Meta-AMG；非结构网格及自适应变拓扑尚未完成状态迁移验证。
- 需要锁定依赖版本、CI、性能回归阈值和可复现实验配置，才能称为跨平台研究软件。

## 推荐顺序

1. 将 GNN 自定义后端改为稀疏消息传递并跑 `64x64` smoke；
2. 调优 PETSc GAMG/HYPRE 并生成更强外部 AMG 基线表；
3. 补图表脚本和论文初稿；
4. GPU、MPI、3D 和交互式工作流。
