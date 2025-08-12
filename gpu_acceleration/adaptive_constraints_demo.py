"""
自适应物理约束与强化学习控制器演示
展示如何在现有的 geological_ml_framework.py 中添加动态权重调整功能
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 深度学习相关依赖
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.nn.parameter import Parameter
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    torch = None
    nn = None
    optim = None
    warnings.warn("PyTorch not available. Geological ML features will be limited.")

# 可选依赖检查
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_STABLE_BASELINES3 = True
except ImportError:
    HAS_STABLE_BASELINES3 = False
    warnings.warn("stable-baselines3 not available. RL features will be limited.")


class AdaptivePhysicalConstraint:
    """自适应物理约束类 - 动态调整权重"""
    
    def __init__(self, name: str, equation: Callable, initial_weight: float = 1.0,
                 min_weight: float = 0.01, max_weight: float = 10.0,
                 adaptation_rate: float = 0.1):
        self.name = name
        self.equation = equation
        self.current_weight = initial_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adaptation_rate = adaptation_rate
        
        # 历史记录
        self.residual_history = []
        self.weight_history = []
        self.adaptation_history = []
        
        # 自适应参数
        self.target_residual = 1e-6
        self.residual_window = 10  # 用于计算平均残差的窗口大小
        
    def compute_residual(self, *args, **kwargs) -> float:
        """计算约束残差"""
        try:
            residual = self.equation(*args, **kwargs)
            self.residual_history.append(residual)
            
            # 保持历史记录大小
            if len(self.residual_history) > 100:
                self.residual_history.pop(0)
                
            return float(residual)
        except Exception as e:
            warnings.warn(f"计算约束 {self.name} 残差失败: {str(e)}")
            return np.inf
    
    def adapt_weight(self, current_residual: float):
        """自适应调整权重"""
        if len(self.residual_history) < self.residual_window:
            return
        
        # 计算最近窗口内的平均残差
        recent_residuals = self.residual_history[-self.residual_window:]
        avg_residual = np.mean(recent_residuals)
        
        # 计算残差比率
        residual_ratio = avg_residual / (self.target_residual + 1e-12)
        
        # 自适应调整权重
        if residual_ratio > 1.0:  # 残差过大，增加权重
            weight_change = self.adaptation_rate * (residual_ratio - 1.0)
            new_weight = min(self.max_weight, self.current_weight * (1.0 + weight_change))
        else:  # 残差较小，适当减少权重
            weight_change = self.adaptation_rate * (1.0 - residual_ratio) * 0.5
            new_weight = max(self.min_weight, self.current_weight * (1.0 - weight_change))
        
        # 记录调整历史
        self.weight_history.append(self.current_weight)
        self.adaptation_history.append({
            'timestamp': time.time(),
            'old_weight': self.current_weight,
            'new_weight': new_weight,
            'residual_ratio': residual_ratio,
            'avg_residual': avg_residual
        })
        
        # 更新权重
        self.current_weight = new_weight
        
        # 保持历史记录大小
        if len(self.weight_history) > 100:
            self.weight_history.pop(0)
        if len(self.adaptation_history) > 100:
            self.adaptation_history.pop(0)
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """获取自适应调整摘要"""
        if not self.adaptation_history:
            return {}
        
        recent_adaptations = self.adaptation_history[-10:]  # 最近10次调整
        
        return {
            'name': self.name,
            'current_weight': self.current_weight,
            'initial_weight': self.weight_history[0] if self.weight_history else self.current_weight,
            'total_adaptations': len(self.adaptation_history),
            'recent_adaptations': recent_adaptations,
            'residual_trend': self._compute_residual_trend(),
            'weight_trend': self._compute_weight_trend()
        }
    
    def _compute_residual_trend(self) -> str:
        """计算残差趋势"""
        if len(self.residual_history) < 2:
            return "insufficient_data"
        
        recent = np.mean(self.residual_history[-10:])
        earlier = np.mean(self.residual_history[-20:-10]) if len(self.residual_history) >= 20 else self.residual_history[0]
        
        if recent < earlier * 0.8:
            return "decreasing"
        elif recent > earlier * 1.2:
            return "increasing"
        else:
            return "stable"
    
    def _compute_weight_trend(self) -> str:
        """计算权重趋势"""
        if len(self.weight_history) < 2:
            return "insufficient_data"
        
        recent = np.mean(self.weight_history[-10:])
        earlier = np.mean(self.weight_history[-20:-10]) if len(self.weight_history) >= 20 else self.weight_history[0]
        
        if recent > earlier * 1.1:
            return "increasing"
        elif recent < earlier * 0.9:
            return "decreasing"
        else:
            return "stable"


class RLConstraintController:
    """强化学习约束控制器 - 自动优化物理约束权重"""
    
    def __init__(self, constraints: List[AdaptivePhysicalConstraint],
                 state_dim: int = 10, action_dim: int = None,
                 learning_rate: float = 3e-4, gamma: float = 0.99):
        self.constraints = constraints
        self.state_dim = state_dim
        self.action_dim = action_dim or len(constraints)
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # 强化学习环境状态
        self.current_state = None
        self.action_history = []
        self.reward_history = []
        self.state_history = []
        
        # 初始化RL代理
        self.agent = self._create_rl_agent()
        
        # 控制参数
        self.control_frequency = 5  # 每5次迭代控制一次
        self.iteration_count = 0
        self.last_control_time = time.time()
        
    def _create_rl_agent(self):
        """创建强化学习代理"""
        if not HAS_STABLE_BASELINES3:
            warnings.warn("stable-baselines3 not available, using simple controller")
            return None
        
        try:
            # 创建简单的环境包装器
            class ConstraintEnvironment:
                def __init__(self, controller):
                    self.controller = controller
                    self.action_space = self.controller.action_dim
                    self.observation_space = self.controller.state_dim
                
                def reset(self):
                    return np.zeros(self.controller.state_dim)
                
                def step(self, action):
                    # 执行动作并返回奖励
                    reward = self.controller._execute_action(action)
                    next_state = self.controller._get_state()
                    done = False
                    info = {}
                    return next_state, reward, done, info
            
            env = DummyVecEnv([lambda: ConstraintEnvironment(self)])
            
            # 创建PPO代理
            agent = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                verbose=0
            )
            
            return agent
            
        except Exception as e:
            warnings.warn(f"创建RL代理失败: {str(e)}")
            return None
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态向量"""
        state = []
        
        # 约束残差
        for constraint in self.constraints:
            if constraint.residual_history:
                recent_residuals = constraint.residual_history[-5:]  # 最近5个残差
                avg_residual = np.mean(recent_residuals)
                state.append(avg_residual)
            else:
                state.append(0.0)
        
        # 权重信息
        for constraint in self.constraints:
            state.append(constraint.current_weight)
        
        # 自适应历史
        for constraint in self.constraints:
            if constraint.adaptation_history:
                recent_adaptations = constraint.adaptation_history[-3:]  # 最近3次调整
                avg_weight_change = np.mean([abs(a['new_weight'] - a['old_weight']) 
                                          for a in recent_adaptations])
                state.append(avg_weight_change)
            else:
                state.append(0.0)
        
        # 填充到指定维度
        while len(state) < self.state_dim:
            state.append(0.0)
        
        return np.array(state[:self.state_dim])
    
    def _execute_action(self, action: np.ndarray) -> float:
        """执行动作并计算奖励"""
        if len(action) != len(self.constraints):
            return 0.0
        
        # 应用权重调整
        old_weights = [c.current_weight for c in self.constraints]
        
        for i, constraint in enumerate(self.constraints):
            # 将动作映射到权重调整
            weight_adjustment = np.tanh(action[i]) * 0.2  # 限制调整幅度
            new_weight = constraint.current_weight * (1.0 + weight_adjustment)
            
            # 应用约束
            new_weight = np.clip(new_weight, constraint.min_weight, constraint.max_weight)
            constraint.current_weight = new_weight
        
        # 计算奖励：基于残差改善
        total_residual_improvement = 0.0
        for constraint in self.constraints:
            if len(constraint.residual_history) >= 2:
                old_residual = constraint.residual_history[-2]
                new_residual = constraint.compute_residual()  # 重新计算
                improvement = old_residual - new_residual
                total_residual_improvement += improvement
        
        # 归一化奖励
        reward = total_residual_improvement / (len(self.constraints) + 1e-6)
        
        return reward
    
    def control_constraints(self):
        """控制约束权重"""
        self.iteration_count += 1
        
        # 检查是否需要控制
        if self.iteration_count % self.control_frequency != 0:
            return
        
        current_time = time.time()
        if current_time - self.last_control_time < 1.0:  # 至少间隔1秒
            return
        
        try:
            # 获取当前状态
            current_state = self._get_state()
            self.state_history.append(current_state)
            
            if self.agent is not None:
                # 使用RL代理选择动作
                action, _ = self.agent.predict(current_state, deterministic=True)
                reward = self._execute_action(action)
                
                # 记录历史
                self.action_history.append(action)
                self.reward_history.append(reward)
                
                # 训练代理（如果数据足够）
                if len(self.state_history) >= 10:
                    self._train_agent()
            else:
                # 使用简单的启发式控制
                self._heuristic_control()
            
            self.last_control_time = current_time
            
        except Exception as e:
            warnings.warn(f"约束控制失败: {str(e)}")
    
    def _heuristic_control(self):
        """启发式控制策略"""
        for constraint in self.constraints:
            if constraint.residual_history:
                recent_residuals = constraint.residual_history[-5:]
                avg_residual = np.mean(recent_residuals)
                
                # 基于残差大小调整权重
                if avg_residual > constraint.target_residual * 2:
                    # 残差过大，增加权重
                    constraint.current_weight = min(
                        constraint.max_weight,
                        constraint.current_weight * 1.2
                    )
                elif avg_residual < constraint.target_residual * 0.5:
                    # 残差较小，适当减少权重
                    constraint.current_weight = max(
                        constraint.min_weight,
                        constraint.current_weight * 0.9
                    )
    
    def _train_agent(self):
        """训练RL代理"""
        if self.agent is None or len(self.state_history) < 10:
            return
        
        try:
            # 创建训练数据
            states = np.array(self.state_history[-10:])
            actions = np.array(self.action_history[-10:])
            rewards = np.array(self.reward_history[-10:])
            
            # 训练代理
            self.agent.learn(total_timesteps=100)
            
        except Exception as e:
            warnings.warn(f"训练RL代理失败: {str(e)}")
    
    def get_control_summary(self) -> Dict[str, Any]:
        """获取控制摘要"""
        return {
            'total_iterations': self.iteration_count,
            'control_frequency': self.control_frequency,
            'last_control_time': self.last_control_time,
            'total_actions': len(self.action_history),
            'total_rewards': len(self.reward_history),
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'rl_agent_active': self.agent is not None,
            'constraint_states': [c.get_adaptation_summary() for c in self.constraints]
        }


def demo_adaptive_constraints():
    """演示自适应物理约束功能"""
    print("=== 自适应物理约束演示 ===")
    
    # 1. 创建物理约束方程
    def darcy_equation(x, y_pred):
        """Darcy流动方程约束"""
        # 简化的Darcy方程残差计算
        # 在实际应用中，这里应该计算真实的PDE残差
        return np.random.normal(0, 1e-6)  # 模拟残差
    
    def heat_equation(x, y_pred):
        """热传导方程约束"""
        return np.random.normal(0, 1e-5)  # 模拟残差
    
    # 2. 创建自适应约束
    darcy_constraint = AdaptivePhysicalConstraint(
        name="Darcy方程",
        equation=darcy_equation,
        initial_weight=1.0,
        min_weight=0.01,
        max_weight=5.0,
        adaptation_rate=0.1
    )
    
    heat_constraint = AdaptivePhysicalConstraint(
        name="热传导方程",
        equation=heat_equation,
        initial_weight=0.5,
        min_weight=0.01,
        max_weight=3.0,
        adaptation_rate=0.08
    )
    
    print(f"✅ 创建物理约束: {darcy_constraint.name}, {heat_constraint.name}")
    
    # 3. 创建强化学习控制器
    controller = RLConstraintController(
        constraints=[darcy_constraint, heat_constraint],
        state_dim=6,  # 2个约束 × 3个状态维度
        action_dim=2, # 2个约束的权重调整
        learning_rate=1e-3,
        gamma=0.99
    )
    
    print(f"✅ 创建RL控制器，管理 {len(controller.constraints)} 个约束")
    
    # 4. 模拟训练过程
    print("\n🔄 开始模拟训练过程...")
    
    for step in range(20):
        # 模拟计算残差
        darcy_residual = darcy_constraint.compute_residual()
        heat_residual = heat_constraint.compute_residual()
        
        # 自适应调整权重
        darcy_constraint.adapt_weight(darcy_residual)
        heat_constraint.adapt_weight(heat_residual)
        
        # 控制器自动调整
        controller.control_constraints()
        
        # 每5步显示状态
        if step % 5 == 0:
            print(f"\n步骤 {step}:")
            print(f"  Darcy约束: 残差={darcy_residual:.2e}, 权重={darcy_constraint.current_weight:.4f}")
            print(f"  热传导约束: 残差={heat_residual:.2e}, 权重={heat_constraint.current_weight:.4f}")
            
            # 获取约束摘要
            darcy_summary = darcy_constraint.get_adaptation_summary()
            heat_summary = heat_constraint.get_adaptation_summary()
            
            print(f"  Darcy趋势: 残差={darcy_summary['residual_trend']}, 权重={darcy_summary['weight_trend']}")
            print(f"  热传导趋势: 残差={heat_summary['residual_trend']}, 权重={heat_summary['weight_trend']}")
    
    # 5. 获取最终摘要
    print("\n📊 训练完成，获取摘要信息:")
    
    controller_summary = controller.get_control_summary()
    print(f"控制器摘要:")
    print(f"  总迭代次数: {controller_summary['total_iterations']}")
    print(f"  总动作数: {controller_summary['total_actions']}")
    print(f"  平均奖励: {controller_summary['avg_reward']:.6f}")
    print(f"  RL代理状态: {'激活' if controller_summary['rl_agent_active'] else '未激活'}")
    
    # 约束摘要
    for constraint in [darcy_constraint, heat_constraint]:
        summary = constraint.get_adaptation_summary()
        print(f"\n{constraint.name} 约束摘要:")
        print(f"  当前权重: {summary['current_weight']:.4f}")
        print(f"  初始权重: {summary['initial_weight']:.4f}")
        print(f"  总调整次数: {summary['total_adaptations']}")
        print(f"  残差趋势: {summary['residual_trend']}")
        print(f"  权重趋势: {summary['weight_trend']}")
    
    print("\n🎉 自适应物理约束演示完成！")


def demo_integration_with_existing_pinn():
    """演示如何与现有PINN集成"""
    print("\n=== 与现有PINN集成演示 ===")
    
    if not HAS_PYTORCH:
        print("❌ PyTorch不可用，跳过PINN集成演示")
        return
    
    # 1. 创建物理约束
    def darcy_equation(x, y_pred):
        """Darcy流动方程约束"""
        return np.random.normal(0, 1e-6)
    
    darcy_constraint = AdaptivePhysicalConstraint(
        name="Darcy方程",
        equation=darcy_equation,
        initial_weight=1.0
    )
    
    # 2. 创建约束控制器
    controller = RLConstraintController(
        constraints=[darcy_constraint],
        state_dim=3,
        action_dim=1
    )
    
    print("✅ 创建约束系统")
    
    # 3. 模拟PINN训练过程
    print("🔄 模拟PINN训练过程...")
    
    for epoch in range(10):
        # 模拟计算残差
        residual = darcy_constraint.compute_residual()
        
        # 自适应调整权重
        darcy_constraint.adapt_weight(residual)
        
        # 控制器调整
        controller.control_constraints()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: 残差={residual:.2e}, 权重={darcy_constraint.current_weight:.4f}")
    
    print("✅ PINN集成演示完成")


if __name__ == "__main__":
    # 运行演示
    demo_adaptive_constraints()
    demo_integration_with_existing_pinn()
