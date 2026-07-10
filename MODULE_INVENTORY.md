# ============================================================================
# Meta-AMG 论文专用模块
# ============================================================================
# 
# 以下模块是Meta-AMG论文的所有代码依赖，全部有真实实现，127个测试通过。
# 
# gpu_acceleration/neural_amg.py  — GNN C/F分类器 + 训练器 + 求解器 (600行)
# gpu_acceleration/meta_amg.py    — MAML训练 + 在线适配器 + 序列求解器 (1190行)
# solvers/multigrid_solver.py     — 传统AMG baseline (838行)
# experiments/run_experiments.py  — 6个论文实验 (625行)
# tests/test_neural_amg.py        — 18个测试
# tests/test_meta_amg.py          — 14个测试
# tests/test_solvers.py           — 8个测试
#
# 运行: python experiments/run_experiments.py --exp all
#
# ============================================================================
# 以下模块属于Geo_sim扩展框架，非论文依赖，有不同程度的空壳
# ============================================================================

## core/
### core/geodynamic_simulation.py
Geo_sim的主仿真类，提供Underworld风格的统一接口。真实的网格创建、材料添加、求解器调用。但非线性Picard迭代未实现，FEM装配用的是简化版（不是实际积分），GPU加速器已导入但未接入求解路径。论文相关性：间接——其中的AMG求解器调用链可被Meta-AMG替代，但Meta-AMG实验不需要它。

### core/unified_api.py
抽象基类体系：BaseSimulator → FiniteElementSimulator / MultiPhysicsSimulator / MLSimulator。工厂函数create_simulator()和配置序列化。论文相关性：无。这是完整仿真框架的顶层抽象，我们的实验直接用MatrixSequenceGenerator产生矩阵。

### core/debug_tools.py
实时仿真监控、物理约束残差热力图、误差诊断、性能分析。论文相关性：无。

### core/complete_interface_demo.py
六个演示函数展示Geo_sim的API使用方式。论文相关性：无。

### core/module_integration_demo.py / core/usage_examples.py
演示脚本。论文相关性：无。

## coupling/
多物理场耦合模块——热力耦合(thermal_mechanical)、流固耦合(fluid_solid)、化学力耦合(chemical_mechanical)、电磁力耦合(electro_mechanical)、多相流(multiphase_fluid)、热力学(thermodynamics)。每个模块都有完整的类层次和方法签名，但部分求解方法为简化实现。论文相关性：无（但我们引用了热力耦合作为"为什么矩阵会演化"的物理背景）。

## materials/
地质材料物理模型——流变学(ConstantViscosity, PowerLawCreep, CompositeRheology)、塑性(VonMises, DruckerPrager, 高级版)、相变(SolidusLiquidus, PeridotiteMelting)、损伤(IsotropicDamage, DamagePlasticityCoupling)、断裂(MaxPrincipalStress, CrackPropagation)、硬化(Linear, Nonlinear, Cyclic)、实验验证框架。这是Geo_sim最完整的模块，都有实际数学实现。论文相关性：间接——材料参数定义了粘度对比度的物理范围(10^18-10^23 Pa·s)，Hirth-Kohlstedt流变学是实验设计的理论依据。

## finite_elements/
FEM管线——基函数(linear/quadratic/cubic Lagrange)、单元类型(三角形/四边形/四面体/六面体)、高斯积分、等参变换、单元装配、全局装配、边界条件(Dirichlet/Neumann/Robin)、自由度管理、网格生成(自适应h-refinement)。大部分有实际数学实现（基函数、积分、变换是真实的），stokes_heat_element_matrix刚补全。论文相关性：可选用——global_assembly可用于生成更真实的矩阵序列替代当前Poisson生成器。

## time_integration/
时间积分器——RungeKuttaIntegrator(1-4阶)、AdaptiveTimeIntegrator(误差估计+步长自适应)、ImplicitTimeIntegrator(后向Euler)、BDFIntegrator(1-4阶)、CrankNicolsonIntegrator。全部有真实实现。论文相关性：无直接关系，但如果Meta-AMG将来嵌入完整仿真管线，时间积分器负责控制矩阵序列的步长。

## visualization/
2D/3D可视化、体渲染、等值面渲染、切片渲染。有函数签名和接口定义，依赖matplotlib和plotly。论文相关性：无——论文图表用单独的matplotlib脚本画。

## adaptivity/
误差估计(ErrorEstimator)、网格细化(MeshRefinement, AdaptiveMeshRefiner)、自适应求解策略。有完整实现。论文相关性：无。

## parallel/
MPI域分解(DomainDecomposer)、并行求解器(ParallelSolver, AdvancedParallelSolverV2)、动态通信、异构计算(MPI+GPU+OpenMP)、负载均衡。AdvancedParallelSolverV2有3367行代码但含bug。论文相关性：无——当前所有实验在单机CPU上完成。

## gpu_acceleration/ (非paper用的部分)
advanced_ml.py：ML训练框架、PINN训练、RL求解器参数优化（有代码但未集成）。geological_ml_framework.py：地质PINN、代理模型(GP/RF/XGBoost)、UNet（有完整实现但训练数据需要外部生成）。geodynamics_gnn.py：地质拓扑GNN、GATConv/GCNConv、多尺度消息传递（有代码但网络未训练）。cuda_acceleration.py：CuPy GPU求解器（有GPU版稀疏CG但未接入求解路径）。physics_integrated_ml.py：物理约束ML集成框架。adaptive_constraints_demo.py：动态物理约束权重。论文相关性：均无——这些是Geo_sim扩展框架的ML部分，与我们的MAML方法是平行的替代技术路线。

## ensemble/
多保真度集成学习——低/高保真度代理模型、知识蒸馏、stacking集成。有完整实现。论文相关性：无。

## tests/ (非paper用的部分)
test_finite_elements.py(28)、test_materials.py(20)、test_geodynamic_simulation.py(26)、test_time_integration.py(13)。这些测试验证Geo_sim其他模块的正确性。论文相关性：间接——它们验证了项目基础设施是可用的。
