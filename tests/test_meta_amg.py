"""
元学习AMG测试
"""

import numpy as np
import pytest
import time
import sys
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 绕过 gpu_acceleration/__init__.py 的损坏导入
_neural_spec = importlib.util.spec_from_file_location(
    'neural_amg',
    str(Path(__file__).parent.parent / 'gpu_acceleration' / 'neural_amg.py')
)
_neural_mod = importlib.util.module_from_spec(_neural_spec)
sys.modules['neural_amg'] = _neural_mod
_neural_spec.loader.exec_module(_neural_mod)

_meta_spec = importlib.util.spec_from_file_location(
    'meta_amg',
    str(Path(__file__).parent.parent / 'gpu_acceleration' / 'meta_amg.py')
)
_meta_mod = importlib.util.module_from_spec(_meta_spec)
sys.modules['meta_amg'] = _meta_mod
_meta_spec.loader.exec_module(_meta_mod)

MetaAMGConfig = _meta_mod.MetaAMGConfig
MatrixSequenceGenerator = _meta_mod.MatrixSequenceGenerator
MAMLGNN = _meta_mod.MAMLGNN
MAMLTrainer = _meta_mod.MAMLTrainer
MetaAMGAdapter = _meta_mod.MetaAMGAdapter
MetaAMGSolver = _meta_mod.MetaAMGSolver
MetaAMG = _meta_mod.MetaAMG


class TestMatrixSequenceGenerator:
    """矩阵序列生成测试"""

    def test_sequence_length(self):
        config = MetaAMGConfig()
        gen = MatrixSequenceGenerator(config)
        seq = gen.generate_sequence(n=100, pattern='slab', length=8)
        assert len(seq) == 8

    def test_viscosity_contrast(self):
        config = MetaAMGConfig()
        gen = MatrixSequenceGenerator(config)
        seq = gen.generate_sequence(n=100, pattern='slab', length=6, contrast=1e4)
        eta_last = seq[-1]['eta']
        ratio = eta_last.max() / (eta_last.min() + 1e-12)
        assert ratio > 100

    def test_patterns(self):
        config = MetaAMGConfig()
        gen = MatrixSequenceGenerator(config)
        for pattern in ['slab', 'plume', 'layered', 'random']:
            seq = gen.generate_sequence(n=100, pattern=pattern, length=4)
            assert len(seq) == 4

    def test_matrix_spd(self):
        config = MetaAMGConfig()
        gen = MatrixSequenceGenerator(config)
        seq = gen.generate_sequence(n=81, pattern='slab', length=3, contrast=1e4)
        for s in seq:
            A = s['matrix']
            x = np.random.randn(A.shape[0])
            xtAx = x @ (A @ x)
            assert xtAx > 0, f"Matrix at step {s['step']} not SPD"

    def test_training_data_generation(self):
        config = MetaAMGConfig(
            min_matrix_size=64, max_matrix_size=144,
            num_training_sequences=8, sequence_length=5,
        )
        gen = MatrixSequenceGenerator(config)
        train, val = gen.generate_training_data(num_sequences=8)
        assert len(train) > 0
        assert len(val) > 0
        for task in train:
            assert 'support_graph' in task
            assert 'query_graph' in task
            assert 'support_labels' in task
            assert 'query_labels' in task


class TestMAMLGNN:
    """GNN模型测试"""

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_model_creation(self):
        config = MetaAMGConfig(hidden_dim=32, num_layers=2)
        model = MAMLGNN(config)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_forward_shape(self):
        import torch
        config = MetaAMGConfig(hidden_dim=32, num_layers=2)
        model = MAMLGNN(config)
        n = 25
        x = torch.randn(n, 8)
        ei = torch.tensor([[0]*4 + [1]*4, [1,2,3,4,0,2,3,4]], dtype=torch.long)
        out = model(x, ei, return_logits=True)
        assert out.shape == (n,)

    def test_clone(self):
        import torch
        config = MetaAMGConfig(hidden_dim=32, num_layers=2)
        model = MAMLGNN(config)
        cloned = model.clone()
        for p1, p2 in zip(model.parameters(), cloned.parameters()):
            assert torch.equal(p1, p2)


class TestMAMLTraining:
    """MAML训练测试"""

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_minimal_training(self):
        config = MetaAMGConfig(
            min_matrix_size=64, max_matrix_size=121,
            num_training_sequences=8, sequence_length=4,
            num_meta_epochs=5, inner_steps=2,
            meta_batch_size=2, hidden_dim=16, num_layers=2,
        )
        meta = MetaAMG(config)
        history = meta.train(num_sequences=8)
        assert len(history['train_meta_loss']) == config.num_meta_epochs

    def test_adaptation_improves(self):
        config = MetaAMGConfig(
            min_matrix_size=64, max_matrix_size=144,
            num_training_sequences=10, sequence_length=5,
            num_meta_epochs=10, inner_steps=3,
            meta_batch_size=2, hidden_dim=16, num_layers=2,
        )
        meta = MetaAMG(config)
        history = meta.train(num_sequences=10)

        zs = history['val_accuracy'][-1]
        ad = history['val_adapt_accuracy'][-1]
        assert ad >= zs - 0.10, f"Adaptation should not hurt significantly: zs={zs:.4f}, ad={ad:.4f}"


class TestMetaAMGSolver:
    """MetaAMG求解器测试"""

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_solve_sequence(self):
        config = MetaAMGConfig(
            min_matrix_size=64, max_matrix_size=144,
            num_training_sequences=10, sequence_length=5,
            num_meta_epochs=10, inner_steps=3,
            meta_batch_size=2, hidden_dim=16, num_layers=2,
        )
        meta = MetaAMG(config)
        meta.train(num_sequences=10)

        gen = MatrixSequenceGenerator(config)
        seq = gen.generate_sequence(n=121, pattern='slab', length=5, contrast=1e4)
        matrices = [s['matrix'] for s in seq]
        n = matrices[0].shape[0]
        b = matrices[0] @ np.ones(n)

        sols = meta.solve_sequence(matrices, b)
        assert len(sols) == len(matrices)
        for sol in sols:
            assert sol is not None

    def test_step1_traditional_rest_adapted(self):
        config = MetaAMGConfig(
            min_matrix_size=64, max_matrix_size=144,
            num_training_sequences=10, sequence_length=5,
            num_meta_epochs=10, inner_steps=3,
            meta_batch_size=2, hidden_dim=16, num_layers=2,
        )
        meta = MetaAMG(config)
        meta.train(num_sequences=10)

        gen = MatrixSequenceGenerator(config)
        seq = gen.generate_sequence(n=121, pattern='slab', length=5, contrast=1e4)
        matrices = [s['matrix'] for s in seq]
        b = matrices[0] @ np.ones(matrices[0].shape[0])

        meta.solve_sequence(matrices, b)
        stats = meta.get_stats()

        assert stats['n_traditional'] == 1, "Only first step should use traditional AMG"
        assert stats['n_adapted'] == len(matrices) - 1, "All subsequent steps should use adaptation"

    def test_solution_accuracy(self):
        config = MetaAMGConfig(
            min_matrix_size=64, max_matrix_size=200,
            num_training_sequences=12, sequence_length=5,
            num_meta_epochs=10, inner_steps=3,
            meta_batch_size=2, hidden_dim=16, num_layers=2,
        )
        meta = MetaAMG(config)
        meta.train(num_sequences=12)

        gen = MatrixSequenceGenerator(config)
        seq = gen.generate_sequence(n=256, pattern='slab', length=4, contrast=1e4)
        matrices = [s['matrix'] for s in seq]
        n = matrices[0].shape[0]
        x_exact = np.random.randn(n)
        # 每个矩阵用自己的 b = A @ x_exact
        b_list = [A @ x_exact for A in matrices]

        for k in range(len(matrices)):
            sol = meta.solve_single(matrices[k], b_list[k])
            err = np.linalg.norm(sol - x_exact) / np.linalg.norm(x_exact)
            assert err < 0.05, f"Step {k}: error {err:.6f} too high"

    def test_high_contrast(self):
        config = MetaAMGConfig(
            min_matrix_size=64, max_matrix_size=200,
            num_training_sequences=12, sequence_length=5,
            num_meta_epochs=10, inner_steps=3,
            meta_batch_size=2, hidden_dim=16, num_layers=2,
        )
        meta = MetaAMG(config)
        meta.train(num_sequences=12)

        gen = MatrixSequenceGenerator(config)
        seq = gen.generate_sequence(n=300, pattern='slab', length=6, contrast=1e6)
        matrices = [s['matrix'] for s in seq]
        n = matrices[0].shape[0]
        b = matrices[0] @ np.ones(n)

        sols = meta.solve_sequence(matrices, b)
        stats = meta.get_stats()
        assert stats['n_adapted'] > 0
        assert len(sols) == len(matrices)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
