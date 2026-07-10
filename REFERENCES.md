# Meta-AMG 参考文献

## 引用体系（按论文Section组织）

---

## 1. Algebraic Multigrid 基础

### [1] Classical AMG
> Ruge, J.W. and Stüben, K. (1987). "Algebraic Multigrid." In *Multigrid Methods*, SIAM.
> — 奠基性工作。定义C/F-splitting的经典算法：强连接检测 + 贪心最大独立集。

### [2] AMG综述
> Stüben, K. (2001). "A review of algebraic multigrid." *Journal of Computational and Applied Mathematics*, 128(1-2), 281-309.
> — AMG的标准参考文献。讨论C/F粗化策略、插值算子构造、收敛性理论。

---

## 2. 神经AMG (直接预测C/F)

### [3] 第一篇用GNN学习C/F粗化
> Greenfeld, D., Galun, M., Basri, R., Yavneh, I., and Kimmel, R. (2019). "Learning to Optimize Multigrid PDE Solvers." *ICML 2019*, PMLR 97.
> — 用GCN学习C/F标签。在均匀Poisson问题上验证。**局限：每步独立预测，不利用序列信息。**

### [4] RL-based AMG Coarsening  (55 citations)
> Taghibakhshi, A., MacLachlan, S., Olson, L., and West, M. (2021). "Optimization-based Algebraic Multigrid Coarsening Using Reinforcement Learning." *NeurIPS 2021*.
> — 用GNN + RL学习coarsening策略。reward = 收敛率。**与本文区别：RL需要大量交互训练，本文MAML更快且更稳定。**

### [5] 最新Neural AMG
> Luz, I., Galun, M., Maron, H., Basri, R., and Yavneh, I. (2024). "Learning Algebraic Multigrid." *ICLR 2024*.
> — 端到端学习整个AMG pipeline（包括P算子、smoother参数）。SoTA结果。**局限：仍在单个固定矩阵上测试，不处理序列。**

### [6] 图神经网络加速AMG压力求解器
> Chillón, E., Lidtke, A.K., Doan, N.A.K., and Font, B. (2026). "Acceleration of an algebraic multigrid pressure solver using graph neural networks." arXiv:2606.19251.
> — 在非结构网格Navier-Stokes压力求解器上用GNN加速AMG。验证GNN方法在真实CFD问题上的有效性。

### [7] 3D有限元AMG的AI增强
> Goik, D. and Banaś, K. (2026). "Artificial intelligence-enhanced algebraic multigrid for 3D finite element simulations." *Computer Methods in Materials Science*.
> — 用GNN预测3D有限元矩阵的C/F。**直接相关：证明了GNN在3D非均匀材料问题上的泛化能力。**

### [8] RAPNet: GNN学习AMG稀疏修正
> Fink, Y., Ben-Yair, I., Ruthotto, L., and Treister, E. (2026). "RAPNet: Accelerating Algebraic Multigrid with Learned Sparse Corrections." arXiv:2605.26854.
> — 用GNN学习AMG coarse-grid correction的稀疏模式。

### [9] COARSERL: Graph RL for AMG
> Yusuf, S., Zhang, Z., Thopalli, K., and Li, R.P. (2026). "COARSERL: A Graph Reinforcement Learning Method for Algebraic Multigrid Coarsening." *AI & PDE: ICLR 2026 Workshop*.
> — 最新的RL-based AMG工作。ICLR 2026 Workshop接收。证明该方向被顶会社区认可。

---

## 3. MAML (Model-Agnostic Meta-Learning)

### [10] MAML原论文 (11000+ citations)
> Finn, C., Abbeel, P., and Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." *ICML 2017*.
> — 提出MAML框架。证明了θ*可以通过内循环SGD + 外循环二阶梯度学习"如何快速适配"。**本文直接使用MAML作为元训练算法。**

### [11] MAML泛化理论
> Finn, C., Rajeswaran, A., Kakade, S., and Levine, S. (2019). "Online Meta-Learning." *ICML 2019*.
> — 证明MAML的泛化误差界：O(1/√K_inner + 1/√N_tasks)。**本文引用来保证Meta-AMG的适配质量。**

### [12] MAML在PDE中的应用
> Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. (2021). "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021*.
> — FNO的元学习视角：在参数化PDE族上训练，对新参数零样本推理。**与本文精神一致：PDE求解器可通过meta-learning加速。**

---

## 4. GNN表达能力 (理论基础)

### [13] GNN表达力
> Xu, K., Hu, W., Leskovec, J., and Jegelka, S. (2019). "How Powerful are Graph Neural Networks?" *ICLR 2019*.
> — GIN论文。证明GNN的表达能力上界是Weisfeiler-Lehman图同构测试。**本文引用：证明3层GCN的感受野覆盖AMG C/F决策所需的局部图结构。**

### [14] Graph U-Nets (图池化)
> Gao, H. and Ji, S. (2019). "Graph U-Nets." *ICML 2019*.
> — 提出图池化(gPool)和图上采样(gUnpool)。**本文引用来讨论GNN与AMG的层次化粗化的结构相似性。**

---

## 5. 地质动力学数值方法

### [15] Underworld2
> Moresi, L., et al. (2007). "Computational approaches to studying non-linear dynamics of the crust and mantle." *Physics of the Earth and Planetary Interiors*, 163(1-4), 69-82.
> — Underworld地质仿真框架。定义了有限元+AMG求解Stokes方程的标准范式。**本文的motivation直接来源。**

### [16] 地幔对流基准 (Blankenbach)
> Blankenbach, B., et al. (1989). "A benchmark comparison for mantle convection codes." *Geophysical Journal International*, 98(1), 23-38.
> — 地幔对流代码的标准benchmark。定义了不同Rayleigh数下的Nusselt数和RMS速度参考值。**实验部分的基准测试标准。**

### [17] 地幔流变学 (幂律蠕变)
> Hirth, G. and Kohlstedt, D.L. (2003). "Rheology of the upper mantle and the mantle wedge: A view from the experimentalists." *Geophysical Monograph*, 138, 83-105.
> — 地幔主要造岩矿物(橄榄石)的幂律蠕变参数：A=1e-16 s⁻¹, n=3.5, E=520 kJ/mol。定义了地幔对流中的粘度变化范围(10¹⁸-10²³ Pa·s)。**本文矩阵序列的粘度对比度来源。**

### [18] 板块构造中的极值粘度对比
> Moresi, L. and Solomatov, V. (1998). "Mantle convection with a brittle lithosphere: thoughts on the global tectonic styles of the Earth and Venus." *Geophysical Journal International*, 133(3), 669-682.
> — 粘塑性流变学模型。论证板块构造需要η_max/η_min > 10³的粘度对比。**本文实验设置粘度对比度10²-10⁶的理论依据。**

---

## 6. 迭代法收敛理论

### [19] AMG收敛理论
> Brandt, A. (1986). "Algebraic multigrid theory: The symmetric case." *Applied Mathematics and Computation*, 19(1-4), 23-56.
> — 证明AMG的两层收敛率依赖于插值算子P的精度。**本文引用：只要GNN预测的C/F能产生合理P，AMG就能收敛。**

### [20] Krylov子空间方法的收敛
> Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*. SIAM.
> — 标准教材。CG/GMRES的收敛率由矩阵条件数和Ritz值分布决定。**本文引用：讨论C/F质量对求解迭代次数的影响。**

---

## 7. 相关方向 (可选引)

### [21] PINN
> Raissi, M., Perdikaris, P., and Karniadakis, G.E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.
> — PINN奠基性工作。**本文在Related Work中列为替代加速方法，但指出PINN是"取代整个FEM"而我们是"加速已有FEM的求解器"。**

### [22] FNO
> Li, Z., et al. (2021). "Fourier Neural Operator for Parametric Partial Differential Equations." *ICLR 2021*.
> — FNO是另一种PDE加速方法。**本文在Related Work中讨论，指出FNO和Meta-AMG是互补的：FNO加速"仿真的前向计算"，Meta-AMG加速"求解器内部"。**

### [23] 多保真度建模
> Kennedy, M.C. and O'Hagan, A. (2000). "Predicting the output from a complex computer code when fast approximations are available." *Biometrika*, 87(1), 1-13.
> — 奠基性工作。**本文的meta-learning框架可视为一种动态多保真度策略。**

### [24] 图神经网络在稀疏矩阵中的应用
> Sappl, J., Daropoulos, V., Rauch, W., et al. (2026). "Convolutional neural network-driven preconditioners for conjugate gradients." *Machine Learning: Science and Technology*.
> — CNN/GNN用于预条件子。**验证了ML+线性求解器结合的有效性。**

---

## 引用策略总结

| 论文Section | 关键引用 | 作用 |
|-------------|---------|------|
| Introduction | [15] Underworld, [3] Greenfeld, [5] Luz | 问题陈述+相关工作 |
| Related Work | [3-9] 神经AMG系列, [10] MAML, [21,22] PINN/FNO | 定位本文在文献中的位置 |
| Background: AMG | [1] Ruge-Stüben, [2] Stüben | C/F算法和技术公式 |
| Background: MAML | [10] Finn et al., [11] Finn et al. | MAML算法块+泛化界 |
| Background: 地质 | [15,16,17,18] | 为什么矩阵演化是实际问题 |
| Method: GNN | [13] GIN, [14] Graph U-Nets | GNN架构选择的理论依据 |
| Method: MAML | [10,11] | 训练算法的正确性保证 |
| Method: Convergence | [19,20] | 论证C/F预测收敛性 |
| Experiments | [3,4,5,16] | Baseline对比+基准标准 |
| Discussion | [3,5,7,9] | 与最新工作的对比 |

---

## 投稿目标建议

按创新程度和实验完整度排序：

| 会议/期刊 | 接收率 | 适配度 | 需要追加 |
|-----------|--------|--------|---------|
| **ICLR 2027 Workshop (AI4Science)** | ~40% | ⭐⭐⭐⭐⭐ | 完成实验数据即可 |
| **NeurIPS 2026 Workshop** | ~35% | ⭐⭐⭐⭐ | 同上 |
| **Journal of Computational Physics (JCP)** | ~25% | ⭐⭐⭐⭐ | 需要Stokes基准 |
| **Geoscientific Model Development (GMD)** | ~30% | ⭐⭐⭐⭐⭐ | 需要Underworld对比 |
| **ICLR 2027 (正会)** | ~25% | ⭐⭐⭐ | 需要更大规模实验+理论证明 |
| **NeurIPS 2026 (正会)** | ~25% | ⭐⭐⭐ | 同上 |

**推荐策略：先投ICLR/NeurIPS Workshop (快速接收，获得社区反馈)，修改后投JCP/GMD (完整版)。**
