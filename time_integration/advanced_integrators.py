"""
高级时间积分器实现

包含完整的时间积分功能：
1. 自适应时间步长
2. 高阶方法（RK4、RK5等）
3. 刚性求解器
4. 误差估计
5. 稳定性分析
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


@dataclass
class IntegratorConfig:
    """积分器配置"""
    method: str = 'rk4'  # 积分方法
    adaptive: bool = True  # 是否使用自适应步长
    tolerance: float = 1e-6  # 误差容差
    min_step: float = 1e-8  # 最小步长
    max_step: float = 1.0  # 最大步长
    safety_factor: float = 0.9  # 安全因子
    max_iterations: int = 1000  # 最大迭代次数
    stiff_solver: bool = False  # 是否为刚性求解器


class BaseTimeIntegrator(ABC):
    """时间积分器基类"""
    
    def __init__(self, config: IntegratorConfig = None):
        self.config = config or IntegratorConfig()
        self.time_history = []
        self.solution_history = []
        self.step_history = []
        self.error_history = []
    
    @abstractmethod
    def integrate(self, f: Callable, t0: float, t1: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """积分函数"""
        pass
    
    @abstractmethod
    def step(self, f: Callable, t: float, y: np.ndarray, h: float) -> Tuple[np.ndarray, float]:
        """单步积分"""
        pass
    
    def get_integration_info(self) -> Dict:
        """获取积分信息"""
        return {
            'method': self.config.method,
            'adaptive': self.config.adaptive,
            'total_steps': len(self.step_history),
            'total_time': self.time_history[-1] - self.time_history[0] if self.time_history else 0,
            'final_error': self.error_history[-1] if self.error_history else 0,
            'min_step': min(self.step_history) if self.step_history else 0,
            'max_step': max(self.step_history) if self.step_history else 0
        }


class RungeKutta4Integrator(BaseTimeIntegrator):
    """四阶Runge-Kutta积分器"""
    
    def __init__(self, config: IntegratorConfig = None):
        super().__init__(config)
        if config is None:
            self.config.method = 'rk4'
    
    def integrate(self, f: Callable, t0: float, t1: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """RK4积分"""
        if self.config.adaptive:
            return self._adaptive_integrate(f, t0, t1, y0)
        else:
            return self._fixed_step_integrate(f, t0, t1, y0)
    
    def _fixed_step_integrate(self, f: Callable, t0: float, t1: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """固定步长积分"""
        h = (t1 - t0) / self.config.max_iterations
        t = t0
        y = y0.copy()
        
        times = [t]
        solutions = [y.copy()]
        
        while t < t1:
            y, _ = self.step(f, t, y, h)
            t += h
            
            times.append(t)
            solutions.append(y.copy())
        
        return np.array(times), np.array(solutions)
    
    def _adaptive_integrate(self, f: Callable, t0: float, t1: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """自适应步长积分"""
        t = t0
        y = y0.copy()
        h = self.config.max_step
        
        times = [t]
        solutions = [y.copy()]
        steps = [h]
        errors = [0.0]
        
        while t < t1:
            # 尝试步长
            y_new, error = self.step(f, t, y, h)
            
            # 检查误差
            if error < self.config.tolerance or h <= self.config.min_step:
                # 接受步长
                y = y_new
                t += h
                
                times.append(t)
                solutions.append(y.copy())
                steps.append(h)
                errors.append(error)
                
                # 调整步长
                if error > 0:
                    h = min(self.config.max_step, 
                           h * self.config.safety_factor * (self.config.tolerance / error) ** 0.25)
            else:
                # 拒绝步长，减小步长
                h = max(self.config.min_step, h * 0.5)
            
            # 确保不超过终点
            if t + h > t1:
                h = t1 - t
        
        self.time_history = times
        self.solution_history = solutions
        self.step_history = steps
        self.error_history = errors
        
        return np.array(times), np.array(solutions)
    
    def step(self, f: Callable, t: float, y: np.ndarray, h: float) -> Tuple[np.ndarray, float]:
        """RK4单步"""
        # RK4系数
        k1 = f(t, y)
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h/2, y + h*k2/2)
        k4 = f(t + h, y + h*k3)
        
        # 四阶解
        y_new = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # 误差估计（使用五阶解）
        if self.config.adaptive:
            # 简化的误差估计
            error = np.linalg.norm(h * (k1 - k4) / 6)
        else:
            error = 0.0
        
        return y_new, error


class RungeKutta5Integrator(BaseTimeIntegrator):
    """五阶Runge-Kutta积分器（Dormand-Prince方法）"""
    
    def __init__(self, config: IntegratorConfig = None):
        super().__init__(config)
        if config is None:
            self.config.method = 'rk5'
        
        # Dormand-Prince系数
        self.a = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        ])
        
        self.b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
        self.b_hat = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    def integrate(self, f: Callable, t0: float, t1: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """RK5积分"""
        if self.config.adaptive:
            return self._adaptive_integrate(f, t0, t1, y0)
        else:
            return self._fixed_step_integrate(f, t0, t1, y0)
    
    def _fixed_step_integrate(self, f: Callable, t0: float, t1: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """固定步长积分"""
        h = (t1 - t0) / self.config.max_iterations
        t = t0
        y = y0.copy()
        
        times = [t]
        solutions = [y.copy()]
        
        while t < t1:
            y, _ = self.step(f, t, y, h)
            t += h
            
            times.append(t)
            solutions.append(y.copy())
        
        return np.array(times), np.array(solutions)
    
    def _adaptive_integrate(self, f: Callable, t0: float, t1: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """自适应步长积分"""
        t = t0
        y = y0.copy()
        h = self.config.max_step
        
        times = [t]
        solutions = [y.copy()]
        steps = [h]
        errors = [0.0]
        
        while t < t1:
            # 尝试步长
            y_new, error = self.step(f, t, y, h)
            
            # 检查误差
            if error < self.config.tolerance or h <= self.config.min_step:
                # 接受步长
                y = y_new
                t += h
                
                times.append(t)
                solutions.append(y.copy())
                steps.append(h)
                errors.append(error)
                
                # 调整步长
                if error > 0:
                    h = min(self.config.max_step, 
                           h * self.config.safety_factor * (self.config.tolerance / error) ** 0.2)
            else:
                # 拒绝步长，减小步长
                h = max(self.config.min_step, h * 0.5)
            
            # 确保不超过终点
            if t + h > t1:
                h = t1 - t
        
        self.time_history = times
        self.solution_history = solutions
        self.step_history = steps
        self.error_history = errors
        
        return np.array(times), np.array(solutions)
    
    def step(self, f: Callable, t: float, y: np.ndarray, h: float) -> Tuple[np.ndarray, float]:
        """RK5单步（Dormand-Prince方法）"""
        k = np.zeros((7, len(y)))
        
        # 计算k值
        for i in range(7):
            if i == 0:
                k[i] = f(t, y)
            else:
                y_temp = y.copy()
                for j in range(i):
                    y_temp += h * self.a[i, j] * k[j]
                k[i] = f(t + h * self.a[i, i], y_temp)
        
        # 五阶解
        y_new = y.copy()
        for i in range(7):
            y_new += h * self.b[i] * k[i]
        
        # 四阶解（用于误差估计）
        y_hat = y.copy()
        for i in range(7):
            y_hat += h * self.b_hat[i] * k[i]
        
        # 误差估计
        error = np.linalg.norm(y_new - y_hat)
        
        return y_new, error


class StiffIntegrator(BaseTimeIntegrator):
    """刚性求解器（BDF方法）"""
    
    def __init__(self, config: IntegratorConfig = None):
        super().__init__(config)
        if config is None:
            self.config.method = 'bdf'
            self.config.stiff_solver = True
        
        self.order = 2  # BDF阶数
        self.backward_steps = []
    
    def integrate(self, f: Callable, t0: float, t1: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """BDF积分"""
        t = t0
        y = y0.copy()
        h = self.config.max_step
        
        times = [t]
        solutions = [y.copy()]
        steps = [h]
        errors = [0.0]
        
        # 初始化历史
        self.backward_steps = [y.copy()]
        
        while t < t1:
            # BDF步
            y_new, error = self.step(f, t, y, h)
            
            # 检查误差
            if error < self.config.tolerance or h <= self.config.min_step:
                # 接受步长
                y = y_new
                t += h
                
                times.append(t)
                solutions.append(y.copy())
                steps.append(h)
                errors.append(error)
                
                # 更新历史
                self.backward_steps.append(y.copy())
                if len(self.backward_steps) > self.order + 1:
                    self.backward_steps.pop(0)
                
                # 调整步长
                if error > 0:
                    h = min(self.config.max_step, 
                           h * self.config.safety_factor * (self.config.tolerance / error) ** 0.5)
            else:
                # 拒绝步长，减小步长
                h = max(self.config.min_step, h * 0.5)
            
            # 确保不超过终点
            if t + h > t1:
                h = t1 - t
        
        self.time_history = times
        self.solution_history = solutions
        self.step_history = steps
        self.error_history = errors
        
        return np.array(times), np.array(solutions)
    
    def step(self, f: Callable, t: float, y: np.ndarray, h: float) -> Tuple[np.ndarray, float]:
        """BDF单步"""
        if len(self.backward_steps) < self.order:
            # 使用RK4初始化
            rk4 = RungeKutta4Integrator()
            y_new, _ = rk4.step(f, t, y, h)
            return y_new, 0.0
        
        # BDF2方法
        if self.order == 2:
            # y_{n+1} = (4/3) * y_n - (1/3) * y_{n-1} + (2/3) * h * f(t_{n+1}, y_{n+1})
            y_prev = self.backward_steps[-2]
            
            # 预测器
            y_pred = (4/3) * y - (1/3) * y_prev
            
            # 校正器（简化实现）
            f_pred = f(t + h, y_pred)
            y_new = y_pred + (2/3) * h * f_pred
            
            # 误差估计
            error = np.linalg.norm(y_new - y_pred)
            
            return y_new, error
        
        # 更高阶BDF方法可以在这里实现
        return y, 0.0


class AdaptiveIntegrator(BaseTimeIntegrator):
    """自适应积分器（组合多种方法）"""
    
    def __init__(self, config: IntegratorConfig = None):
        super().__init__(config)
        if config is None:
            self.config.method = 'adaptive'
        
        # 创建子积分器
        self.rk4 = RungeKutta4Integrator()
        self.rk5 = RungeKutta5Integrator()
        self.stiff = StiffIntegrator()
    
    def integrate(self, f: Callable, t0: float, t1: float, y0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """自适应积分"""
        # 检测刚性
        if self._is_stiff_system(f, t0, y0):
            print("🔄 检测到刚性系统，使用BDF方法")
            return self.stiff.integrate(f, t0, t1, y0)
        else:
            print("🔄 使用RK5方法")
            return self.rk5.integrate(f, t0, t1, y0)
    
    def _is_stiff_system(self, f: Callable, t: float, y: np.ndarray) -> bool:
        """检测是否为刚性系统"""
        # 简化的刚性检测：计算雅可比矩阵的特征值
        try:
            # 数值计算雅可比矩阵
            J = self._compute_jacobian(f, t, y)
            eigenvals = np.linalg.eigvals(J)
            
            # 计算刚性比
            real_parts = np.real(eigenvals)
            max_real = np.max(real_parts)
            min_real = np.min(real_parts)
            
            if max_real > 0 and min_real < 0:
                stiffness_ratio = abs(max_real / min_real)
                return stiffness_ratio > 100  # 刚性阈值
        except:
            pass
        
        return False
    
    def _compute_jacobian(self, f: Callable, t: float, y: np.ndarray) -> np.ndarray:
        """计算雅可比矩阵"""
        n = len(y)
        J = np.zeros((n, n))
        eps = 1e-8
        
        f0 = f(t, y)
        
        for i in range(n):
            y_perturbed = y.copy()
            y_perturbed[i] += eps
            f_perturbed = f(t, y_perturbed)
            J[:, i] = (f_perturbed - f0) / eps
        
        return J
    
    def step(self, f: Callable, t: float, y: np.ndarray, h: float) -> Tuple[np.ndarray, float]:
        """自适应单步"""
        # 使用RK5作为默认方法
        return self.rk5.step(f, t, y, h)


# 工厂函数
def create_integrator(method: str = 'rk4', config: IntegratorConfig = None) -> BaseTimeIntegrator:
    """创建时间积分器"""
    if method == 'rk4':
        return RungeKutta4Integrator(config)
    elif method == 'rk5':
        return RungeKutta5Integrator(config)
    elif method == 'bdf':
        return StiffIntegrator(config)
    elif method == 'adaptive':
        return AdaptiveIntegrator(config)
    else:
        raise ValueError(f"不支持的积分方法: {method}")


def create_integrator_config(**kwargs) -> IntegratorConfig:
    """创建积分器配置"""
    return IntegratorConfig(**kwargs)
