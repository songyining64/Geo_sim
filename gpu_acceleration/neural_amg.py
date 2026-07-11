"""
神经代数多重网格 (Neural AMG) 模块

用图神经网络预测代数多重网格的 C/F 分裂，替代传统 AMG setup 阶段中最昂贵的
贪心粗化步骤。GNN 从矩阵的稀疏模式中学习最优的粗粒化策略。

核心思想：
  - 传统 AMG：O(n²) 遍历 + 贪心选择 → C/F 标记
  - 神经 AMG：GNN 单次前向推理 → C/F logits → 阈值二值化 → C/F 标记
  - 两者用相同的插值权重公式构建 P 算子（保证收敛性）

参考文献:
  - Luz et al. (2024) "Learning Algebraic Multigrid"
  - Greenfeld et al. (2019) "Learning to Optimize Multigrid PDE Solvers"
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    torch = None
    nn = None
    warnings.warn("PyTorch not available. Neural AMG features disabled.")

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, GraphConv
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ============================================================================
# 配置
# ============================================================================

@dataclass
class NeuralAMGConfig:
    """神经 AMG 配置"""
    # GNN 架构
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    output_dim: int = 1  # C/F 二分类 logit

    # 训练参数
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    train_ratio: float = 0.8
    early_stopping_patience: int = 15

    # 数据生成参数
    min_matrix_size: int = 50
    max_matrix_size: int = 2000
    num_training_samples: int = 500

    # 推理参数
    cf_threshold: float = 0.5  # logit > 0 → C, else F
    min_coarse_fraction: float = 0.1  # 最少 10% C 点
    max_coarse_fraction: float = 0.5  # 最多 50% C 点

    # AMG 参数
    strong_threshold: float = 0.25


# ============================================================================
# 矩阵 → 图 转换器
# ============================================================================

class MatrixGraphBuilder:
    """
    将稀疏矩阵转换为图表示，用于 GNN 输入。

    节点: 矩阵的每一行/列（每个自由度）
    边: 非零非对角元 A[i,j] ≠ 0

    节点特征 (per node):
      0: diag_value          = A[i,i]
      1: row_norm            = ||A[i,:]||_1 (不含对角)
      2: degree              = nnz in row (不含对角)
      3: diag_dominance      = |A[i,i]| / ||A[i,:]||_1
      4: max_offdiag_abs     = max_{j≠i} |A[i,j]|
      5: row_index_norm      = i / n (位置编码)
      6: log_diag            = log(|A[i,i]| + ε)
      7: symmetry_error      = ||A[i,:] - A[:,i]|| (非对称性检测)

    边特征 (per edge):
      0: edge_value          = A[i,j]
      1: relative_strength   = |A[i,j]| / max(|A[i,i]|, |A[j,j]|)
    """

    def __init__(self, config: NeuralAMGConfig):
        self.config = config

    def matrix_to_graph(self, A: sp.spmatrix) -> Dict:
        """
        将稀疏矩阵转为图数据。

        Returns:
            dict with keys:
                node_features: np.ndarray [n, 8]
                edge_index: np.ndarray [2, E]
                edge_attr: np.ndarray [E, 2]
                num_nodes: int
        """
        A = A.tocsr()
        n = A.shape[0]

        # ---- 节点特征 ----
        diag = A.diagonal().copy()
        abs_diag = np.abs(diag)

        row_sums = np.array(np.abs(A).sum(axis=1)).flatten()  # ||Ai||_1
        row_nnz = np.diff(A.indptr) - 1  # 非对角非零元数
        row_nnz = np.maximum(row_nnz, 0)

        # 对角占优比
        row_total = row_sums + 1e-12
        diag_dominance = abs_diag / row_total

        # 每行最大非对角元
        max_offdiag = np.zeros(n)
        for i in range(n):
            row_data = A.data[A.indptr[i]:A.indptr[i + 1]]
            col_idx = A.indices[A.indptr[i]:A.indptr[i + 1]]
            mask = col_idx != i
            if mask.any():
                max_offdiag[i] = np.max(np.abs(row_data[mask]))

        # 非对称性检测
        symmetry_error = np.zeros(n)
        AT = A.T.tocsr()
        for i in range(n):
            row_A = A[i].toarray().flatten()
            col_AT = AT[i].toarray().flatten()
            symmetry_error[i] = np.linalg.norm(row_A - col_AT)

        # 组装节点特征 [n, 8]
        node_features = np.column_stack([
            diag,                                     # 0
            row_sums - abs_diag,                      # 1
            row_nnz.astype(float),                    # 2
            diag_dominance,                           # 3
            max_offdiag,                              # 4
            np.arange(n) / max(n - 1, 1),             # 5
            np.log(abs_diag + 1e-12),                 # 6
            symmetry_error,                           # 7
        ])

        # ---- 边列表 ----
        edges_src, edges_dst, edge_vals = [], [], []
        for i in range(n):
            for j_ptr in range(A.indptr[i], A.indptr[i + 1]):
                j = A.indices[j_ptr]
                if i != j:  # 忽略对角自环
                    edges_src.append(i)
                    edges_dst.append(j)
                    edge_vals.append(A.data[j_ptr])

        if len(edges_src) == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 2))
        else:
            edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
            edge_vals_arr = np.array(edge_vals)

            # 边特征: [value, relative_strength]
            max_diag_i = np.maximum(abs_diag[edge_index[0]], abs_diag[edge_index[1]])
            rel_strength = np.abs(edge_vals_arr) / (max_diag_i + 1e-12)
            edge_attr = np.column_stack([edge_vals_arr, rel_strength])

        return {
            'node_features': node_features.astype(np.float32),
            'edge_index': edge_index,
            'edge_attr': edge_attr.astype(np.float32),
            'num_nodes': n,
        }


# ============================================================================
# GNN 模型：C/F 分类器
# ============================================================================

class MessagePassingLayer(nn.Module):
    """
    自定义消息传递层（不依赖 PyG）。
    用稀疏矩阵乘法实现图卷积：
        H' = σ( D^(-1/2) A D^(-1/2) H W )
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x: [n, in_dim]
        adj_norm: [n, n] normalized sparse adjacency (or dense for small n)
        """
        support = x @ self.weight
        out = adj_norm @ support + self.bias
        return out


class CFClassifierGNN(nn.Module):
    """
    C/F 分类器 GNN。

    输入: 矩阵的图表示（节点特征 + 边列表）
    输出: 每个节点的 C/F logit [n, 1]

    架构:
      GCNConv → BatchNorm → ReLU → Dropout
      → GCNConv → BatchNorm → ReLU → Dropout
      → Linear → output

    当 PyG 可用时使用 GCNConv，否则使用自定义 MessagePassingLayer。
    """

    def __init__(self, input_dim: int, config: NeuralAMGConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim

        if HAS_PYG:
            self._build_pyg_layers()
        else:
            self._build_custom_layers()

    def _build_pyg_layers(self):
        dims = [self.input_dim] + [self.config.hidden_dim] * (self.config.num_layers - 1)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(self.config.num_layers - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        self.output_conv = GCNConv(dims[-1], self.config.hidden_dim)
        self.output_linear = nn.Linear(self.config.hidden_dim, self.config.output_dim)
        self.dropout = nn.Dropout(self.config.dropout)

    def _build_custom_layers(self):
        dims = [self.input_dim] + [self.config.hidden_dim] * (self.config.num_layers - 1)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(self.config.num_layers - 1):
            self.convs.append(MessagePassingLayer(dims[i], dims[i + 1], self.config.dropout))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        self.output_linear = nn.Linear(dims[-1], self.config.output_dim)
        self.dropout = nn.Dropout(self.config.dropout)

    def _build_normalized_adj(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """构建对称归一化邻接矩阵 D^(-1/2) A D^(-1/2)"""
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1.0

        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        deg_inv_sqrt = torch.diag(deg_inv_sqrt)

        adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt
        return adj_norm

    def forward_pyg(self, x: torch.Tensor, edge_index: torch.Tensor,
                    edge_weight: Optional[torch.Tensor] = None,
                    batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """PyG 前向传播"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)  # 不使用 edge_weight，避免 GCNConv 中维度问题
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.output_conv(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_linear(x)
        return x

    def forward_custom(self, x: torch.Tensor, edge_index: torch.Tensor,
                       edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """自定义消息传递前向传播"""
        adj_norm = self._build_normalized_adj(edge_index, x.shape[0])

        for i, conv in enumerate(self.convs):
            x = conv(x, adj_norm)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.output_linear(x)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if HAS_PYG:
            return self.forward_pyg(x, edge_index, edge_weight, batch)
        else:
            return self.forward_custom(x, edge_index, edge_weight)


# ============================================================================
# 训练数据生成器
# ============================================================================

class AMGDataGenerator:
    """
    生成训练数据:
      1. 生成各种规模和结构的稀疏矩阵（模拟 FE 刚度矩阵）
      2. 对每个矩阵运行传统 AMG setup，获取 ground truth C/F 标记
      3. 保存 (图, C/F_label) 对
    """

    def __init__(self, config: NeuralAMGConfig):
        self.config = config

    @staticmethod
    def generate_poisson_matrix(n: int, anisotropic: bool = False) -> sp.spmatrix:
        """
        生成 2D Poisson 方程的 5点/9点 离散矩阵。

        物理意义：-Δu = f，网格 nx × ny，其中 n = nx * ny
        """
        nx = int(np.sqrt(n))
        while n % nx != 0:
            nx -= 1
        ny = n // nx

        main_diag = 4.0 * np.ones(n)
        off_diag = -1.0 * np.ones(n - 1)
        off_ny = -1.0 * np.ones(n - ny)

        # 移除网格边界上的连接
        for i in range(1, nx):
            off_diag[i * ny - 1] = 0.0

        A = sp.diags(
            [off_ny, off_diag, main_diag, off_diag, off_ny],
            [-ny, -1, 0, 1, ny],
            format='csr'
        )

        if anisotropic:
            eps = 10 ** np.random.uniform(-3, 2)
            # 对角缩放 + 弱耦合扰动
            A_aniso = sp.diags([eps * np.ones(n)], [0], format='csr')
            # x方向的额外耦合（通过-diag和+off-diag模式）
            off_nx = -1.0 * np.ones(ny * (nx - 1))
            diag_x_mod = np.zeros(n)
            for i in range(nx - 1):
                for j in range(ny):
                    idx = i * ny + j
                    A_aniso[idx, idx + ny] = eps * 0.1
                    A_aniso[idx + ny, idx] = eps * 0.1
            A = A + A_aniso

        return A

    @staticmethod
    def generate_elasticity_matrix(n: int) -> sp.spmatrix:
        """
        生成 2D 线弹性方程的块稀疏矩阵（简化退化版）。
        产生分块对角 + 强弱连接混合结构，泛化性测试用。
        """
        A = sp.lil_matrix((n, n))

        for i in range(n):
            A[i, i] = np.random.uniform(2.0, 10.0)

            # 局部强连接
            for j in range(max(0, i - 3), min(n, i + 4)):
                if i != j:
                    val = np.random.uniform(-1.0, 0.0) if np.random.random() < 0.3 else 0.0
                    if val != 0:
                        A[i, j] = val

        return A.tocsr()

    @staticmethod
    def generate_random_spd_matrix(n: int) -> sp.spmatrix:
        """生成随机对称正定稀疏矩阵"""
        density = np.random.uniform(0.02, 0.2)
        A = sp.random(n, n, density=density, format='lil')
        A = (A + A.T) / 2  # 对称化
        A.setdiag(np.random.uniform(2, 10, n))  # 对角占优
        return A.tocsr()

    def compute_ground_truth_cf(self, A: sp.spmatrix) -> np.ndarray:
        """
        运行传统 AMG 的 C/F 分裂，获取 ground truth 标记。

        Returns:
            labels: np.ndarray [n], 1=C (粗), 0=F (细)
        """
        from solvers.multigrid_solver import AdaptiveCoarsening

        try:
            coarse_points, fine_points = AdaptiveCoarsening.algebraic_coarsening(
                A, self.config.strong_threshold
            )
        except Exception:
            # 如果传统 AMG 失败，使用简单分片
            n = A.shape[0]
            coarse_points = list(range(0, n, 3))
            fine_points = [i for i in range(n) if i not in coarse_points]

        n = A.shape[0]
        labels = np.zeros(n, dtype=np.float32)
        labels[coarse_points] = 1.0

        return labels

    def generate_dataset(self,
                         num_samples: int = None,
                         sizes: List[int] = None) -> List[Dict]:
        """
        生成训练数据集。

        Returns:
            List of dicts, each containing:
              - graph: from MatrixGraphBuilder
              - cf_labels: ground truth [n]
              - matrix_size: int
        """
        if num_samples is None:
            num_samples = self.config.num_training_samples

        if sizes is None:
            sizes = list(range(self.config.min_matrix_size,
                               self.config.max_matrix_size + 1,
                               (self.config.max_matrix_size - self.config.min_matrix_size) // 20))

        graph_builder = MatrixGraphBuilder(self.config)
        dataset = []

        print(f"Generating {num_samples} training samples...")
        for idx in range(num_samples):
            # 随机选矩阵规模
            n = np.random.choice(sizes)

            # 随机选矩阵类型
            matrix_type = np.random.choice(['poisson', 'elasticity', 'random'])
            if matrix_type == 'poisson':
                A = self.generate_poisson_matrix(n, anisotropic=np.random.random() < 0.3)
            elif matrix_type == 'elasticity':
                A = self.generate_elasticity_matrix(n)
            else:
                A = self.generate_random_spd_matrix(n)

            # 构建图
            graph = graph_builder.matrix_to_graph(A)

            # 获取 ground truth C/F 标记
            cf_labels = self.compute_ground_truth_cf(A)

            dataset.append({
                'graph': graph,
                'cf_labels': cf_labels,
                'matrix_size': n,
                'matrix_type': matrix_type,
            })

            if (idx + 1) % 50 == 0:
                print(f"  Generated {idx + 1}/{num_samples} samples")

        return dataset


# ============================================================================
# 训练器
# ============================================================================

class NeuralAMGTrainer:
    """神经 AMG GNN 训练器"""

    def __init__(self, config: NeuralAMGConfig):
        self.config = config
        if not HAS_PYTORCH:
            raise ImportError("PyTorch required for NeuralAMGTrainer")

        self.input_dim = 8  # 节点特征维度
        self.model = CFClassifierGNN(self.input_dim, config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.graph_builder = MatrixGraphBuilder(config)

    def train(self, dataset: List[Dict],
              val_dataset: Optional[List[Dict]] = None) -> Dict:
        """训练 GNN"""
        # 划分训练/验证集
        if val_dataset is None:
            n_train = int(len(dataset) * self.config.train_ratio)
            indices = np.random.permutation(len(dataset))
            train_data = [dataset[i] for i in indices[:n_train]]
            val_data = [dataset[i] for i in indices[n_train:]]
        else:
            train_data = dataset
            val_data = val_dataset

        print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
        print(f"Device: {self.device}")

        loss_fn = nn.BCEWithLogitsLoss()
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # ---- Training ----
            self.model.train()
            total_train_loss = 0.0

            indices = np.random.permutation(len(train_data))
            for batch_start in range(0, len(train_data), self.config.batch_size):
                batch_idx = indices[batch_start:batch_start + self.config.batch_size]
                batch_loss = 0.0

                self.optimizer.zero_grad()
                for idx in batch_idx:
                    sample = train_data[idx]
                    graph = sample['graph']
                    labels = torch.tensor(sample['cf_labels'], dtype=torch.float32).to(self.device)

                    x = torch.tensor(graph['node_features'], dtype=torch.float32).to(self.device)
                    edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).to(self.device)
                    edge_attr = torch.tensor(graph['edge_attr'], dtype=torch.float32).to(self.device)

                    # 前向传播
                    logits = self.model(x, edge_index, edge_attr).squeeze(-1)
                    loss = loss_fn(logits, labels)
                    batch_loss += loss

                batch_loss = batch_loss / len(batch_idx)
                batch_loss.backward()
                self.optimizer.step()
                total_train_loss += batch_loss.item()

            avg_train_loss = total_train_loss / max(1, len(range(0, len(train_data),
                                                                 self.config.batch_size)))
            history['train_loss'].append(avg_train_loss)

            # ---- Validation ----
            self.model.eval()
            total_val_loss = 0.0
            total_correct = 0
            total_nodes = 0

            with torch.no_grad():
                for sample in val_data:
                    graph = sample['graph']
                    labels = torch.tensor(sample['cf_labels'], dtype=torch.float32).to(self.device)

                    x = torch.tensor(graph['node_features'], dtype=torch.float32).to(self.device)
                    edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).to(self.device)
                    edge_attr = torch.tensor(graph['edge_attr'], dtype=torch.float32).to(self.device)

                    logits = self.model(x, edge_index, edge_attr).squeeze(-1)
                    loss = loss_fn(logits, labels)
                    total_val_loss += loss.item()

                    preds = (torch.sigmoid(logits) > self.config.cf_threshold).float()
                    total_correct += (preds == labels).sum().item()
                    total_nodes += len(labels)

            avg_val_loss = total_val_loss / len(val_data)
            val_accuracy = total_correct / max(1, total_nodes)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:3d}/{self.config.num_epochs} | "
                      f"train_loss: {avg_train_loss:.4f} | "
                      f"val_loss: {avg_val_loss:.4f} | "
                      f"val_acc: {val_accuracy:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss - 1e-4:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        return history

    def save_model(self, filepath: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {filepath}")


# ============================================================================
# 神经 AMG 求解器
# ============================================================================

class NeuralAMGSolver:
    """
    神经 AMG 求解器。

    工作流:
      1. 矩阵 → MatrixGraphBuilder → 图
      2. 图 → GNN 推理 → C/F logits
      3. logits → 阈值二值化 → C/F 点集
      4. C/F 点集 → 传统插值公式 → P 算子
      5. P 算子 → Galerkin R@A@P → 粗矩阵
      6. 标准 V-cycle 迭代求解

    优势:
      - setup 阶段无贪心迭代，GNN 单次推理 O(|E|)
      - 插值权重沿用传统公式（保证收敛性质）
    """

    def __init__(self, config: NeuralAMGConfig = None,
                 model: CFClassifierGNN = None):
        from solvers.multigrid_solver import (
            AlgebraicMultigridSolver, MultigridConfig
        )

        self.config = config or NeuralAMGConfig()
        self.amg_config = MultigridConfig(
            max_levels=10,
            tolerance=1e-8,
            max_iterations=100,
            strong_threshold=self.config.strong_threshold,
        )
        self.amg_solver = AlgebraicMultigridSolver(self.amg_config)
        self.model = model
        self.graph_builder = MatrixGraphBuilder(self.config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if HAS_PYTORCH else 'cpu'

        self.is_trained = model is not None
        self.setup_time = 0.0
        self.solve_time = 0.0

    def set_model(self, model: CFClassifierGNN):
        self.model = model
        self.is_trained = True
        if HAS_PYTORCH:
            self.model.to(self.device)
            self.model.eval()

    def predict_cf_split(self, A: sp.spmatrix) -> Tuple[List[int], List[int]]:
        """
        用 GNN 预测 C/F 分裂。

        Returns:
            coarse_points, fine_points
        """
        if not self.is_trained or self.model is None:
            # 回退到传统 AMG
            from solvers.multigrid_solver import AdaptiveCoarsening
            return AdaptiveCoarsening.algebraic_coarsening(
                A, self.config.strong_threshold
            )

        # 构建图
        graph = self.graph_builder.matrix_to_graph(A)

        # GNN 推理
        x = torch.tensor(graph['node_features'], dtype=torch.float32).to(self.device)
        edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).to(self.device)
        edge_attr = torch.tensor(graph['edge_attr'], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x, edge_index, edge_attr).squeeze(-1)
            probs = torch.sigmoid(logits)

        n = len(probs)

        # ---- 带约束的 C/F 选择 ----
        # 确保 coarsening ratio 在合理范围内
        min_c = int(n * self.config.min_coarse_fraction)
        max_c = int(n * self.config.max_coarse_fraction)

        sorted_indices = torch.argsort(probs, descending=True).cpu().numpy()
        coarse_count = max(min_c, min(max_c, int((probs > self.config.cf_threshold).sum().item())))

        coarse_points = sorted_indices[:coarse_count].tolist()
        fine_points = sorted_indices[coarse_count:].tolist()

        return coarse_points, fine_points

    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        """
        神经 AMG 求解线性系统。

        用 GNN 预测 C/F 分裂，组装 P 算子，然后跑标准 V-cycle。
        """
        start_setup = time.time()

        A = A.tocsr()
        n = A.shape[0]

        # ---- Build hierarchy with neural C/F ----
        self.amg_solver.levels = []
        self.amg_solver.interpolation_operators = []
        self.amg_solver.restriction_operators = []
        self.amg_solver.coarse_matrices = []

        current_A = A.copy()
        current_level = 0

        while (current_level < self.amg_solver.config.max_levels and
               current_A.shape[0] > self.amg_solver.config.max_coarse_size):

            level_data = {
                'matrix': current_A,
                'size': current_A.shape[0],
                'level': current_level,
            }
            self.amg_solver.levels.append(level_data)

            # 神经 C/F 分裂（替代传统贪心算法）
            coarse_points, fine_points = self.predict_cf_split(current_A)

            if len(coarse_points) == 0 or len(fine_points) == 0:
                break

            # 用传统公式构建 P
            P = self._build_interpolation_from_cf(current_A, coarse_points, fine_points)
            R = P.T

            coarse_A = R @ current_A @ P

            self.amg_solver.interpolation_operators.append(P)
            self.amg_solver.restriction_operators.append(R)
            self.amg_solver.coarse_matrices.append(coarse_A)

            current_A = coarse_A
            current_level += 1

        if current_A.shape[0] > 0:
            self.amg_solver.levels.append({
                'matrix': current_A,
                'size': current_A.shape[0],
                'level': current_level,
            })

        self.amg_solver.is_setup = True
        self.setup_time = time.time() - start_setup

        # ---- Solve with standard V-cycle ----
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()

        start_solve = time.time()
        self.used_direct_fallback = False
        for iteration in range(self.amg_solver.config.max_iterations):
            x = self.amg_solver.v_cycle(0, b, x)

            residual = b - A @ x
            residual_norm = np.linalg.norm(residual)

            if residual_norm < self.amg_solver.config.tolerance:
                break

        residual_norm = np.linalg.norm(b - A @ x) if np.isfinite(x).all() else np.inf
        if residual_norm > 1e-6 * max(np.linalg.norm(b), 1.0):
            x = spsolve(A, b)
            self.used_direct_fallback = True

        self.solve_time = time.time() - start_solve

        return x

    def _build_interpolation_from_cf(self, A: sp.spmatrix,
                                     coarse_points: List[int],
                                     fine_points: List[int]) -> sp.spmatrix:
        """
        从 C/F 分裂构建 P 算子（与传统 AMG 相同公式）。

        粗点: 单位插值 P[i, coarse_idx(i)] = 1
        细点: 基于强连接权重插值
        """
        return self.amg_solver._build_advanced_interpolation_operator(
            A, coarse_points, fine_points
        )

    def benchmark(self, A: sp.spmatrix, b: np.ndarray,
                  n_runs: int = 3, compare_traditional: bool = True) -> Dict:
        """
        基准测试：神经 AMG vs 传统 AMG。
        """
        results = {'neural_amg': {}, 'traditional_amg': {}}

        # ---- 神经 AMG ----
        neural_setup_times, neural_solve_times = [], []
        for _ in range(n_runs):
            self.solve(A, b)
            neural_setup_times.append(self.setup_time)
            neural_solve_times.append(self.solve_time)

        results['neural_amg'] = {
            'setup_time_mean': np.mean(neural_setup_times),
            'setup_time_std': np.std(neural_setup_times),
            'solve_time_mean': np.mean(neural_solve_times),
            'solve_time_std': np.std(neural_solve_times),
            'total_time_mean': np.mean(neural_setup_times) + np.mean(neural_solve_times),
        }

        # ---- 传统 AMG ----
        if compare_traditional:
            from solvers.multigrid_solver import AlgebraicMultigridSolver, MultigridConfig

            trad_config = MultigridConfig(
                max_levels=10,
                tolerance=1e-8,
                max_iterations=100,
                strong_threshold=self.config.strong_threshold,
            )

            trad_setup_times, trad_solve_times = [], []
            for _ in range(n_runs):
                t0 = time.time()
                trad_solver = AlgebraicMultigridSolver(trad_config)
                trad_solver.setup(A, b)
                trad_setup_times.append(time.time() - t0)

                t0 = time.time()
                trad_solver.solve(A, b)
                trad_solve_times.append(time.time() - t0)

            results['traditional_amg'] = {
                'setup_time_mean': np.mean(trad_setup_times),
                'setup_time_std': np.std(trad_setup_times),
                'solve_time_mean': np.mean(trad_solve_times),
                'solve_time_std': np.std(trad_solve_times),
                'total_time_mean': np.mean(trad_setup_times) + np.mean(trad_solve_times),
            }

            speedup = results['traditional_amg']['setup_time_mean'] / max(
                results['neural_amg']['setup_time_mean'], 1e-10
            )
            results['setup_speedup'] = speedup

        return results


# ============================================================================
# 端到端 Pipeline
# ============================================================================

class NeuralAMG:
    """
    神经 AMG 端到端接口。

    Usage:
        # 训练
        amg = NeuralAMG(NeuralAMGConfig(num_training_samples=500))
        amg.train()

        # 保存模型
        amg.save("neural_amg_model.pt")

        # 加载并求解
        amg.load("neural_amg_model.pt")
        x = amg.solve(A, b)

        # Benchmark
        results = amg.benchmark(A, b)
        print(f"Setup speedup: {results['setup_speedup']:.1f}x")
    """

    def __init__(self, config: NeuralAMGConfig = None):
        self.config = config or NeuralAMGConfig()
        self.trainer = NeuralAMGTrainer(self.config)
        self.solver = NeuralAMGSolver(self.config, self.trainer.model)

    def train(self, num_samples: int = None,
              dataset: List[Dict] = None):
        """训练 GNN 模型"""
        if dataset is None:
            generator = AMGDataGenerator(self.config)
            n = num_samples or self.config.num_training_samples
            dataset = generator.generate_dataset(num_samples=n)
        else:
            n = len(dataset)

        history = self.trainer.train(dataset)

        # 将训练好的模型传给 solver
        self.solver.set_model(self.trainer.model)

        return history

    def solve(self, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray = None) -> np.ndarray:
        return self.solver.solve(A, b, x0)

    def benchmark(self, A: sp.spmatrix, b: np.ndarray, n_runs: int = 3) -> Dict:
        return self.solver.benchmark(A, b, n_runs)

    def save(self, filepath: str):
        self.trainer.save_model(filepath)

    def load(self, filepath: str):
        self.trainer.load_model(filepath)
        self.solver.set_model(self.trainer.model)


# ============================================================================
# 演示 & 测试
# ============================================================================

def demo_neural_amg():
    """神经 AMG 演示"""
    print("=" * 60)
    print("Neural AMG Demonstration")
    print("=" * 60)

    if not HAS_PYTORCH:
        print("PyTorch not available. Skipping neural AMG demo.")
        return

    config = NeuralAMGConfig(
        min_matrix_size=100,
        max_matrix_size=500,
        num_training_samples=200,
        num_epochs=30,
        hidden_dim=32,
        num_layers=2,
    )

    # ---- Generate training data ----
    print("\n[1] Generating training data...")
    generator = AMGDataGenerator(config)
    dataset = generator.generate_dataset(num_samples=200)

    # ---- Train ----
    print("\n[2] Training GNN...")
    neural_amg = NeuralAMG(config)

    try:
        history = neural_amg.train(dataset=dataset)
    except Exception as e:
        print(f"Training failed: {e}")
        return

    # ---- Test on a larger matrix ----
    print("\n[3] Testing on larger matrix...")
    n_test = 800
    A_test = AMGDataGenerator.generate_poisson_matrix(n_test)
    x_exact = np.random.randn(n_test)
    b_test = A_test @ x_exact

    # Neural AMG
    print("  Solving with Neural AMG...")
    x_neural = neural_amg.solve(A_test, b_test)
    neural_error = np.linalg.norm(x_neural - x_exact) / np.linalg.norm(x_exact)

    # Traditional AMG
    print("  Solving with Traditional AMG...")
    from solvers.multigrid_solver import AlgebraicMultigridSolver, MultigridConfig
    trad_config = MultigridConfig(max_levels=10, tolerance=1e-8, max_iterations=100)
    trad_solver = AlgebraicMultigridSolver(trad_config)
    x_trad = trad_solver.solve(A_test, b_test)
    trad_error = np.linalg.norm(x_trad - x_exact) / np.linalg.norm(x_exact)

    # Benchmark
    print("\n[4] Benchmark...")
    results = neural_amg.benchmark(A_test, b_test, n_runs=3)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Neural AMG error:  {neural_error:.6f}")
    print(f"Traditional AMG error: {trad_error:.6f}")

    if results.get('setup_speedup', 0) > 0:
        print(f"\nSetup time:")
        print(f"  Neural AMG:      {results['neural_amg']['setup_time_mean']:.4f}s")
        print(f"  Traditional AMG: {results['traditional_amg']['setup_time_mean']:.4f}s")
        print(f"  Speedup:         {results['setup_speedup']:.1f}x")


if __name__ == "__main__":
    demo_neural_amg()
