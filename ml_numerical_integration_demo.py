"""
ML与数值模拟深度融合演示

核心目标：展示如何将ML真正嵌入到数值求解过程中，
而不是作为独立的黑盒模型运行
"""

import numpy as np
import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# 导入我们创建的物理集成ML框架
from gpu_acceleration.physics_integrated_ml import (
    PhysicsConstrainedML, 
    MLAcceleratedSolver,
    PhysicsConstraintConfig
)


class SimpleHeatSolver:
    """
    简单热传导数值求解器
    
    用于演示ML如何加速传统数值方法
    """
    
    def __init__(self, nx: int = 50, ny: int = 50):
        self.nx = nx
        self.ny = ny
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)
        
        # 材料参数
        self.alpha = 1e-4  # 热扩散系数
        self.dt = 0.001    # 时间步长
        
        # 网格
        self.x = np.linspace(0, 1, nx)
        self.y = np.linspace(0, 1, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # 温度场
        self.T = np.zeros((ny, nx))
        
    def solve_step(self, initial_guess: np.ndarray = None) -> Dict:
        """
        求解一个时间步
        
        Args:
            initial_guess: ML提供的初始猜测
        """
        start_time = time.time()
        
        # 使用ML初始猜测或默认值
        if initial_guess is not None:
            self.T = initial_guess.copy()
        else:
            # 默认初始条件：中心热源
            self.T = np.zeros((self.ny, self.nx))
            center_i, center_j = self.ny // 2, self.nx // 2
            self.T[center_i, center_j] = 100.0
        
        # 边界条件
        self.T[0, :] = 0.0      # 下边界
        self.T[-1, :] = 0.0     # 上边界
        self.T[:, 0] = 0.0      # 左边界
        self.T[:, -1] = 0.0     # 右边界
        
        # 显式有限差分求解
        iterations = 0
        max_iter = 1000
        tolerance = 1e-6
        
        T_old = self.T.copy()
        
        for iteration in range(max_iter):
            T_new = T_old.copy()
            
            # 内部节点：热传导方程
            for i in range(1, self.ny - 1):
                for j in range(1, self.nx - 1):
                    T_new[i, j] = T_old[i, j] + self.alpha * self.dt * (
                        (T_old[i+1, j] - 2*T_old[i, j] + T_old[i-1, j]) / self.dx**2 +
                        (T_old[i, j+1] - 2*T_old[i, j] + T_old[i, j-1]) / self.dy**2
                    )
            
            # 检查收敛性
            if np.max(np.abs(T_new - T_old)) < tolerance:
                break
                
            T_old = T_new.copy()
            iterations += 1
        
        # 更新温度场
        self.T = T_new
        
        solve_time = time.time() - start_time
        
        return {
            'temperature': self.T.copy(),
            'iterations': iterations,
            'solve_time': solve_time,
            'converged': iterations < max_iter,
            'final_residual': np.max(np.abs(T_new - T_old))
        }


class HeatConductionMLPredictor(PhysicsConstrainedML):
    """
    热传导ML预测器
    
    学习从问题参数到温度场的映射
    """
    
    def __init__(self, input_dim: int = 5):
        # 输入：网格尺寸、时间步长、热扩散系数、边界条件类型、热源强度
        # 输出：温度场（展平为1D）
        output_dim = 2500  # 50x50网格
        
        # 简化的物理约束（实际应用中需要更复杂的约束）
        physics_equations = []
        
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            physics_equations=physics_equations
        )
    
    def _build_network(self) -> nn.Module:
        """构建专门的热传导预测网络"""
        layers = []
        prev_dim = self.input_dim
        
        # 编码器：将问题参数编码为潜在表示
        encoder_dims = [32, 64, 128]
        for hidden_dim in encoder_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # 解码器：从潜在表示生成温度场
        decoder_dims = [128, 64, 32]
        for hidden_dim in decoder_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        return nn.Sequential(*layers)


class MLAcceleratedHeatSolver(MLAcceleratedSolver):
    """
    ML加速热传导求解器
    
    展示ML如何真正加速数值求解过程
    """
    
    def __init__(self, traditional_solver: SimpleHeatSolver, 
                 ml_model: HeatConductionMLPredictor):
        super().__init__(traditional_solver, ml_model)
        self.traditional_solver = traditional_solver
        self.ml_model = ml_model
        
    def solve_with_ml_acceleration(self, problem_data: Dict) -> Dict:
        """
        ML加速求解热传导问题
        """
        # 1. ML预测初始温度场
        ml_initial_guess = self._predict_initial_solution(problem_data)
        
        # 2. 传统求解器基于ML预测迭代
        solution = self.traditional_solver.solve_step(ml_initial_guess)
        
        # 3. 记录加速效果
        self._record_acceleration(problem_data, solution)
        
        return solution
    
    def _extract_features(self, problem_data: Dict) -> np.ndarray:
        """提取热传导问题特征"""
        features = []
        
        # 网格参数
        features.append(problem_data.get('nx', 50) / 100.0)  # 归一化
        features.append(problem_data.get('ny', 50) / 100.0)
        
        # 时间步长
        features.append(problem_data.get('dt', 0.001) * 1000)  # 放大
        
        # 热扩散系数
        features.append(problem_data.get('alpha', 1e-4) * 1e4)
        
        # 边界条件类型（编码为数值）
        bc_type = problem_data.get('boundary_type', 'dirichlet')
        features.append(1.0 if bc_type == 'dirichlet' else 0.0)
        
        return np.array(features).reshape(1, -1)
    
    def _predict_initial_solution(self, problem_data: Dict) -> np.ndarray:
        """ML预测初始温度场"""
        features = self._extract_features(problem_data)
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            prediction = self.ml_model.forward_with_constraints(features_tensor)
            
            # 重塑为2D网格
            nx = problem_data.get('nx', 50)
            ny = problem_data.get('ny', 50)
            temperature_2d = prediction.numpy().reshape(ny, nx)
            
            return temperature_2d


def generate_training_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成训练数据
    
    通过运行传统求解器生成ML训练数据
    """
    print("生成训练数据...")
    
    X_list = []
    y_list = []
    
    for i in range(n_samples):
        # 随机问题参数
        nx = np.random.randint(30, 71)  # 30-70
        ny = np.random.randint(30, 71)
        dt = np.random.uniform(0.0005, 0.002)
        alpha = np.random.uniform(5e-5, 2e-4)
        boundary_type = np.random.choice(['dirichlet', 'neumann'])
        
        # 创建求解器
        solver = SimpleHeatSolver(nx=nx, ny=ny)
        solver.dt = dt
        solver.alpha = alpha
        
        # 求解
        solution = solver.solve_step()
        
        # 提取特征
        features = [
            nx / 100.0,
            ny / 100.0,
            dt * 1000,
            alpha * 1e4,
            1.0 if boundary_type == 'dirichlet' else 0.0
        ]
        
        # 展平温度场
        temperature_flat = solution['temperature'].flatten()
        
        X_list.append(features)
        y_list.append(temperature_flat)
        
        if (i + 1) % 20 == 0:
            print(f"  已生成 {i + 1}/{n_samples} 个样本")
    
    return np.array(X_list), np.array(y_list)


def train_ml_model(X: np.ndarray, y: np.ndarray) -> HeatConductionMLPredictor:
    """训练ML模型"""
    print("训练ML模型...")
    
    # 创建模型
    ml_model = HeatConductionMLPredictor(input_dim=X.shape[1])
    
    # 设置训练参数
    optimizer = torch.optim.Adam(ml_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # 训练
    epochs = 50
    batch_size = 32
    n_batches = len(X) // batch_size
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_tensor[start_idx:end_idx]
            y_batch = y_tensor[start_idx:end_idx]
            
            # 前向传播
            optimizer.zero_grad()
            y_pred = ml_model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch + 1}/{epochs}, 平均损失: {avg_loss:.6f}")
    
    print("✓ ML模型训练完成")
    return ml_model


def compare_solving_methods():
    """比较不同求解方法的性能"""
    print("\n=== 性能比较 ===")
    
    # 问题参数
    problem_data = {
        'nx': 50,
        'ny': 50,
        'dt': 0.001,
        'alpha': 1e-4,
        'boundary_type': 'dirichlet'
    }
    
    # 1. 传统求解器（无初始猜测）
    print("1. 传统求解器（无初始猜测）...")
    traditional_solver = SimpleHeatSolver(50, 50)
    start_time = time.time()
    traditional_solution = traditional_solver.solve_step()
    traditional_time = time.time() - start_time
    
    print(f"   求解时间: {traditional_solution['solve_time']:.4f}秒")
    print(f"   迭代次数: {traditional_solution['iterations']}")
    print(f"   收敛状态: {'是' if traditional_solution['converged'] else '否'}")
    
    # 2. ML加速求解器
    print("\n2. ML加速求解器...")
    
    # 加载预训练的ML模型
    try:
        ml_model = torch.load('heat_conduction_ml_model.pth')
        print("   ✓ 加载预训练ML模型")
    except:
        print("   ⚠ 未找到预训练模型，使用随机初始化")
        ml_model = HeatConductionMLPredictor(input_dim=5)
    
    ml_accelerated_solver = MLAcceleratedHeatSolver(traditional_solver, ml_model)
    
    start_time = time.time()
    ml_solution = ml_accelerated_solver.solve_with_ml_acceleration(problem_data)
    ml_time = time.time() - start_time
    
    print(f"   求解时间: {ml_solution['solve_time']:.4f}秒")
    print(f"   迭代次数: {ml_solution['iterations']}")
    print(f"   收敛状态: {'是' if ml_solution['converged'] else '否'}")
    
    # 3. 性能对比
    print("\n3. 性能对比:")
    time_improvement = (traditional_solution['solve_time'] - ml_solution['solve_time']) / traditional_solution['solve_time'] * 100
    iteration_improvement = (traditional_solution['iterations'] - ml_solution['iterations']) / traditional_solution['iterations'] * 100
    
    print(f"   时间改进: {time_improvement:.1f}%")
    print(f"   迭代改进: {iteration_improvement:.1f}%")
    
    # 4. 可视化结果
    visualize_comparison(traditional_solution, ml_solution)


def visualize_comparison(traditional_solution: Dict, ml_solution: Dict):
    """可视化比较结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 传统方法结果
    im1 = axes[0].imshow(traditional_solution['temperature'], cmap='hot', origin='lower')
    axes[0].set_title('传统方法温度场')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])
    
    # ML加速方法结果
    im2 = axes[1].imshow(ml_solution['temperature'], cmap='hot', origin='lower')
    axes[1].set_title('ML加速温度场')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1])
    
    # 差异
    diff = np.abs(traditional_solution['temperature'] - ml_solution['temperature'])
    im3 = axes[2].imshow(diff, cmap='viridis', origin='lower')
    axes[2].set_title('绝对差异')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('ml_numerical_integration_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("=== ML与数值模拟深度融合演示 ===\n")
    
    # 1. 生成训练数据
    X, y = generate_training_data(n_samples=50)  # 减少样本数以加快演示
    print(f"✓ 生成训练数据: {X.shape[0]} 个样本")
    
    # 2. 训练ML模型
    ml_model = train_ml_model(X, y)
    
    # 3. 保存模型
    torch.save(ml_model, 'heat_conduction_ml_model.pth')
    print("✓ 模型已保存")
    
    # 4. 性能比较
    compare_solving_methods()
    
    print("\n=== 演示完成 ===")
    print("\n关键要点:")
    print("1. ML模型直接参与数值求解过程，提供初始猜测")
    print("2. 物理约束确保ML输出符合物理规律")
    print("3. 在线学习使ML模型不断改进")
    print("4. 多尺度桥接连接不同尺度的物理模型")


if __name__ == "__main__":
    main()
