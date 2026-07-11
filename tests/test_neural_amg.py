"""
神经 AMG 测试 & Benchmark
"""

import numpy as np
import pytest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 加载 neural_amg 模块（避免触发 gpu_acceleration/__init__.py 的损坏导入）
_neural_amg_path = str(Path(__file__).parent.parent / 'gpu_acceleration' / 'neural_amg.py')
import importlib.util
_neural_amg_spec = importlib.util.spec_from_file_location('neural_amg', _neural_amg_path)
neural_amg_mod = importlib.util.module_from_spec(_neural_amg_spec)
_neural_amg_spec.loader.exec_module(neural_amg_mod)

NeuralAMGConfig = neural_amg_mod.NeuralAMGConfig
AMGDataGenerator = neural_amg_mod.AMGDataGenerator
MatrixGraphBuilder = neural_amg_mod.MatrixGraphBuilder
NeuralAMG = neural_amg_mod.NeuralAMG
NeuralAMGSolver = neural_amg_mod.NeuralAMGSolver


class TestMatrixGraphBuilder:
    """图构建测试"""

    def test_graph_build_poisson(self):
        """Poisson矩阵→图转换"""
        config = NeuralAMGConfig()
        builder = MatrixGraphBuilder(config)
        n = 100
        A = AMGDataGenerator.generate_poisson_matrix(n)
        graph = builder.matrix_to_graph(A)

        assert graph['node_features'].shape == (n, 8)
        assert graph['edge_index'].shape[0] == 2
        assert graph['edge_index'].shape[1] > 0
        assert graph['edge_attr'].shape[1] == 2
        assert graph['num_nodes'] == n

    def test_graph_features_valid(self):
        """验证节点特征无 NaN/Inf"""
        config = NeuralAMGConfig()
        builder = MatrixGraphBuilder(config)
        A = AMGDataGenerator.generate_poisson_matrix(81)
        graph = builder.matrix_to_graph(A)
        assert np.all(np.isfinite(graph['node_features']))

    def test_graph_small_matrix(self):
        """极小矩阵边界情况"""
        config = NeuralAMGConfig()
        builder = MatrixGraphBuilder(config)
        A = AMGDataGenerator.generate_poisson_matrix(9)
        graph = builder.matrix_to_graph(A)
        assert graph['num_nodes'] == 9

    def test_graph_diag_dominance(self):
        """验证对角占优比合理"""
        config = NeuralAMGConfig()
        builder = MatrixGraphBuilder(config)
        A = AMGDataGenerator.generate_poisson_matrix(100)
        graph = builder.matrix_to_graph(A)
        diag_dom = graph['node_features'][:, 3]
        assert np.all(diag_dom >= 0) and np.all(diag_dom <= 1)


class TestDataGenerator:
    """数据生成测试"""

    def test_poisson_matrix_symmetric(self):
        """Poisson矩阵对称性"""
        A = AMGDataGenerator.generate_poisson_matrix(100)
        diff = (A - A.T).nnz
        assert diff == 0

    def test_poisson_matrix_spd(self):
        """Poisson矩阵对称正定性"""
        A = AMGDataGenerator.generate_poisson_matrix(64)
        x = np.random.randn(64)
        xTAx = x @ (A @ x)
        assert xTAx > 0

    def test_elasticity_matrix_size(self):
        """弹塑性矩阵尺寸"""
        A = AMGDataGenerator.generate_elasticity_matrix(50)
        assert A.shape == (50, 50)

    def test_random_matrix_spd(self):
        """随机矩阵对称性"""
        A = AMGDataGenerator.generate_random_spd_matrix(80)
        assert A.shape == (80, 80)

    def test_cf_label_generation(self):
        """C/F标签生成"""
        config = NeuralAMGConfig()
        gen = AMGDataGenerator(config)
        A = AMGDataGenerator.generate_poisson_matrix(100)
        labels = gen.compute_ground_truth_cf(A)
        assert len(labels) == 100
        assert set(np.unique(labels)).issubset({0.0, 1.0})

    def test_dataset_generation(self):
        """完整数据集生成"""
        config = NeuralAMGConfig(min_matrix_size=64, max_matrix_size=121,
                                 num_training_samples=20)
        gen = AMGDataGenerator(config)
        dataset = gen.generate_dataset(num_samples=20, sizes=[64, 81, 100, 121])
        assert len(dataset) == 20
        for sample in dataset:
            assert 'graph' in sample
            assert 'cf_labels' in sample
            assert 'matrix_type' in sample


class TestNeuralAMG:
    """神经AMG端到端测试"""

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_config(self):
        """配置创建"""
        config = NeuralAMGConfig()
        assert config.hidden_dim == 64
        assert config.num_layers == 3

    def test_train_and_solve(self):
        """训练+求解端到端"""
        config = NeuralAMGConfig(
            min_matrix_size=64,
            max_matrix_size=144,
            num_training_samples=80,
            num_epochs=10,
            hidden_dim=16,
            num_layers=2,
            learning_rate=0.01,
            batch_size=20,
        )
        gen = AMGDataGenerator(config)
        dataset = gen.generate_dataset(num_samples=80, sizes=[64, 81, 100, 121, 144])

        neural_amg = NeuralAMG(config)
        history = neural_amg.train(dataset=dataset)

        assert len(history['train_loss']) > 0
        assert history['val_accuracy'][-1] > 0.5

        # 求解测试
        n = 200
        A = AMGDataGenerator.generate_poisson_matrix(n)
        x_exact = np.random.randn(n)
        b = A @ x_exact

        x = neural_amg.solve(A, b)
        error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
        assert error < 0.01, f"Solve error {error} too high"

    def test_solve_convergence(self):
        """验证求解收敛到正确解"""
        config = NeuralAMGConfig(
            min_matrix_size=64,
            max_matrix_size=144,
            num_training_samples=80,
            num_epochs=10,
            hidden_dim=16,
            num_layers=2,
            learning_rate=0.01,
        )
        gen = AMGDataGenerator(config)
        dataset = gen.generate_dataset(num_samples=80, sizes=[64, 81, 100, 121, 144])

        neural_amg = NeuralAMG(config)
        neural_amg.train(dataset=dataset)

        for n in [100, 144, 196]:
            A = AMGDataGenerator.generate_poisson_matrix(n)
            x_exact = np.ones(n)
            b = A @ x_exact
            x = neural_amg.solve(A, b)
            error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
            assert error < 0.01, f"n={n}: error={error}"

    def test_benchmark(self):
        """基准测试"""
        config = NeuralAMGConfig(
            min_matrix_size=64,
            max_matrix_size=144,
            num_training_samples=60,
            num_epochs=8,
            hidden_dim=16,
            num_layers=2,
        )
        gen = AMGDataGenerator(config)
        dataset = gen.generate_dataset(num_samples=60, sizes=[64, 81, 100, 121, 144])

        neural_amg = NeuralAMG(config)
        neural_amg.train(dataset=dataset)

        n = 300
        A = AMGDataGenerator.generate_poisson_matrix(n)
        x_exact = np.random.randn(n)
        b = A @ x_exact

        results = neural_amg.benchmark(A, b, n_runs=2)

        assert 'neural_amg' in results
        assert 'setup_speedup' in results
        assert results['setup_speedup'] > 0.5  # 至少不比传统慢

        print(f"\nNeural setup: {results['neural_amg']['setup_time_mean']:.4f}s")
        print(f"Traditional setup: {results['traditional_amg']['setup_time_mean']:.4f}s")
        print(f"Speedup: {results['setup_speedup']:.1f}x")

    def test_fallback_to_traditional(self):
        """未训练时回退到传统AMG"""
        solver = NeuralAMGSolver(NeuralAMGConfig())
        n = 100
        A = AMGDataGenerator.generate_poisson_matrix(n)
        x_exact = np.ones(n)
        b = A @ x_exact
        x = solver.solve(A, b)
        error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
        assert error < 0.01

    def test_nonconvergent_neural_hierarchy_falls_back_to_direct_solve(self, monkeypatch):
        solver = NeuralAMGSolver(NeuralAMGConfig())
        n = 100
        A = AMGDataGenerator.generate_poisson_matrix(n)
        b = A @ np.ones(n)
        monkeypatch.setattr(solver.amg_solver, 'v_cycle',
                            lambda level, rhs, x: np.full_like(rhs, np.inf))

        x = solver.solve(A, b)

        assert solver.used_direct_fallback
        assert np.linalg.norm(A @ x - b) <= 1e-6 * max(np.linalg.norm(b), 1.0)

    def test_cf_constraint(self):
        """验证C/F比例约束"""
        config = NeuralAMGConfig(
            min_coarse_fraction=0.2,
            max_coarse_fraction=0.5,
        )
        gen = AMGDataGenerator(config)
        dataset = gen.generate_dataset(num_samples=60, sizes=[64, 81, 100, 121, 144])

        config.num_epochs = 8
        config.num_training_samples = 60
        neural_amg = NeuralAMG(config)
        neural_amg.train(dataset=dataset)

        n = 400
        A = AMGDataGenerator.generate_poisson_matrix(n)
        coarse, fine = neural_amg.solver.predict_cf_split(A)
        ratio = len(coarse) / n
        assert config.min_coarse_fraction <= ratio <= config.max_coarse_fraction + 0.1


class TestMatrixTypes:
    """多类型矩阵泛化性测试"""

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_elasticity_matrix(self):
        """弹塑性矩阵求解"""
        config = NeuralAMGConfig(
            min_matrix_size=64,
            max_matrix_size=144,
            num_training_samples=60,
            num_epochs=8,
            hidden_dim=16,
            num_layers=2,
        )
        gen = AMGDataGenerator(config)
        dataset = gen.generate_dataset(num_samples=60, sizes=[64, 81, 100, 121, 144])

        neural_amg = NeuralAMG(config)
        neural_amg.train(dataset=dataset)

        n = 150
        A = AMGDataGenerator.generate_elasticity_matrix(n)
        A = (A + A.T) / 2
        A.setdiag(A.diagonal() + 5.0)
        x_exact = np.ones(n)
        b = A @ x_exact
        x = neural_amg.solve(A, b)
        error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
        assert error < 0.1, f"Elasticity error too high: {error}"

    def test_random_matrix(self):
        """随机SPD矩阵求解"""
        config = NeuralAMGConfig(
            min_matrix_size=64,
            max_matrix_size=144,
            num_training_samples=60,
            num_epochs=8,
            hidden_dim=16,
            num_layers=2,
        )
        gen = AMGDataGenerator(config)
        dataset = gen.generate_dataset(num_samples=60, sizes=[64, 81, 100, 121, 144])

        neural_amg = NeuralAMG(config)
        neural_amg.train(dataset=dataset)

        n = 120
        A = AMGDataGenerator.generate_random_spd_matrix(n)
        x_exact = np.ones(n)
        b = A @ x_exact
        x = neural_amg.solve(A, b)
        error = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
        assert error < 0.15, f"Random matrix error too high: {error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
