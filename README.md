# Geo_sim

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-127%20passed-brightgreen)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A51.9-red)](https://pytorch.org)

**Geo_sim** 是一个面向地质动力学仿真的可扩展数值模拟框架，集成现代机器学习方法加速传统有限元求解。

核心目标：解决 [Underworld2](https://www.underworldcode.org/) 等传统地质模拟软件在大规模非线性仿真中的计算瓶颈。

---

## 架构

```
Geo_sim/
├── core/                    # 统一API + GeodynamicSimulation主入口
│   ├── geodynamic_simulation.py   # 地幔对流/岩石圈变形仿真
│   ├── unified_api.py             # 抽象基类 + 工厂函数
│   └── debug_tools.py             # 实时诊断 + 性能监控
│
├── finite_elements/         # 标准FEM管线
│   ├── basis_functions.py   # Lagrange 线性/二次/三次
│   ├── assembly.py          # 单元+全局装配
│   ├── mesh_generation.py   # 自适应网格 h-refinement
│   └── boundary_conditions.py
│
├── materials/               # 地质材料物理模型
│   ├── advanced_material_models.py  # Material, 幂律蠕变, 复合材料
│   ├── plastic_models.py    # Von Mises, Drucker-Prager
│   ├── phase_change_models.py       # 固相线/液相线, 橄榄岩熔融
│   ├── damage_models.py     # 各向同性/各向异性损伤, 损伤-塑性耦合
│   └── fracture_models.py   # 主应力/应变判据, 裂纹扩展
│
├── solvers/                 # 线性/非线性求解器
│   ├── multigrid_solver.py  # 代数/几何多重网格 (AMG/GMG)
│   └── multiphysics_coupling_solver.py  # 多物理场耦合
│
├── gpu_acceleration/        # GPU + ML加速 (⭐ 核心创新)
│   ├── neural_amg.py        # 神经AMG: GNN预测C/F分裂
│   ├── meta_amg.py          # 元学习AMG适配器 (MAML)
│   ├── cuda_acceleration.py # CuPy CG求解器
│   ├── geological_ml_framework.py  # PINN, 代理模型, UNet
│   ├── geodynamics_gnn.py   # GNN: 地质拓扑+流变关系
│   └── advanced_ml.py       # RL求解器参数优化
│
├── coupling/                # 多物理场耦合
│   ├── thermal_mechanical.py       # 热-力(a)
│   ├── fluid_solid.py       # 流-固
│   ├── chemical_mechanical.py      # 化学-力
│   ├── electro_mechanical.py       # 电磁-力
│   └── multiphase_fluid.py  # 多相流
│
├── time_integration/        # 时间积分器 (RK/BDF/Crank-Nicolson)
├── adaptivity/              # 误差估计 + 自适应网格细化
├── parallel/                # MPI域分解 + 异构计算
├── ensemble/                # 多保真度集成学习
├── visualization/           # 2D/3D可视化
├── tests/                   # 127个测试 (pytest)
├── main.py                  # CLI入口
└── pyproject.toml           # 项目配置
```

---

## 快速开始

### 安装

```bash
git clone https://github.com/songyining64/Geo_sim.git
cd Geo_sim
pip install numpy scipy pyyaml h5py matplotlib  # 核心依赖
pip install torch torch-geometric                # ML加速 (可选)
```

### 运行测试

```bash
pytest tests/ -v              # 127 passed, 1 skipped
```

### CLI 命令

```bash
python main.py benchmark --nx 20 --ny 20 --time-steps 50
python main.py test
python main.py demo
python main.py info
```

---

## 核心创新: 神经AMG + 元学习适配器

### 问题

地质仿真中非线性Picard迭代（粘度随应变率更新）导致刚度矩阵每步都在变化。传统AMG每步重新setup（包括O(n²)的贪心C/F分裂），成为计算瓶颈。

### 方案1: 神经AMG (`neural_amg.py`)

**用GNN预测C/F分裂，替代传统贪心粗化。**

```
传统AMG setup:  矩阵 → O(n²)遍历+贪心 → C/F分裂 → 构建P → Galerkin粗化
神经AMG setup:  矩阵 → GNN单次推理 O(|E|) → C/F分裂 → 构建P → Galerkin粗化
```

| 指标 | 传统AMG | 神经AMG |
|------|---------|---------|
| Setup时间 (400 DOF) | 0.671s | 0.069s |
| 求解误差 | <10⁻⁶ | 2×10⁻⁶ |
| 加速比 | 1x | **9.7x** |

### 方案2: 元学习AMG适配器 (`meta_amg.py`) ⭐ 论文级创新

**用MAML训练GNN，使其能通过3步梯度从上一个矩阵的C/F快速适配到当前矩阵。**

```
Step 0: 传统AMG → C/F₀ (一次性开销, ~0.5s)
Step k: meta-θ → 在A_{k-1}上3步SGD → θ' → 预测A_k的C/F (~0.05s)
```

| 指标 | 数值 |
|------|------|
| 零样本准确率 | 67.1% |
| 适配后准确率 | 83.4% |
| 适配增益 | +16.3% |
| 每步Setup加速 | **10.8x** |

### 论文潜力

首次将MAML应用于离散化PDE矩阵序列的快速预条件子适配，针对Stokes系统的极端粘度对比(10⁶:1 ~ 10¹²:1)，在演化矩阵序列上实现setup加速。

```python
from gpu_acceleration.meta_amg import MetaAMG

# 训练
meta = MetaAMG(MetaAMGConfig(num_training_sequences=100))
meta.train()
meta.save("meta_amg.pt")

# 部署
matrices = [A0, A1, A2, ...]  # 非线性仿真中的演化序列
solutions = meta.solve_sequence(matrices, b)
stats = meta.get_stats()
# → setup_speedup: 10.8x
```

---

## 依赖

| 类别 | 包 | 必需 |
|------|----|------|
| 核心 | numpy, scipy, pyyaml, h5py | ✓ |
| 可视化 | matplotlib, pandas | 推荐 |
| GPU | torch, torch-geometric, cupy | 可选 |
| RL | stable-baselines3, gym | 可选 |
| MPI | mpi4py, petsc4py | 可选 |
| 测试 | pytest, pytest-cov | 开发 |

---

## 引用

```bibtex
@software{geo_sim2024,
  author = {GeoSim Team},
  title = {Geo\_sim: Geodynamic Simulation Framework with Neural Acceleration},
  year = {2024},
  url = {https://github.com/songyining64/Geo_sim}
}
```

## License

MIT
