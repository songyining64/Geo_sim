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
- 单元测试覆盖真实轨迹、物理轨迹隔离和小型端到端 Picard 路径。

## 仍然是 P0 的缺口

| 缺口 | 为什么重要 | 下一项可交付 |
|---|---|---|
| 矩阵自由 Schur 的压力迭代成本仍高 | quick case 已稳定，但约需数十次压力 Krylov 迭代，规模增大后可能主导总时间 | 引入 PCD/LSC 或更强压力质量预条件，并验证 paper-medium 零直接回退 |
| adaptation 开销抵消层级构建收益 | 三步 adaptation 使 paper-medium online setup 比传统重建高约 48.6% | 做 `1/2/3` 步消融或 ANIL 式部分适配，保留 Krylov 质量并降低在线开销 |
| 没有外部 Underworld/PETSc 对照 | 自研装配器不能代替生态兼容性验证 | 增加 CSR/NPZ 矩阵导入器与 PETSc/HYPRE 可选后端适配器 |

## P1 工程缺口

- 多数 GPU、并行、可视化与多物理模块仍是原型或未接入主求解路径；README 中的历史性能数字不能作为当前版本结论。
- 2D 线性三角形 Stokes 已可用于数据与小型求解；3D、非结构网格、自适应网格变拓扑尚未纳入 Meta-AMG 状态迁移。
- 需要锁定依赖版本、CI、性能回归阈值和可复现实验配置，才能称为跨平台研究软件。

## 推荐顺序

1. 稳定 blocked Stokes 的 Schur/压力路径并减少直接回退；
2. 按质量门重跑 paper-medium E2/E6；
3. PETSc/HYPRE / Underworld 矩阵导入对照；
4. GPU、MPI、3D 和交互式工作流。
