"""
高级时间积分器 - 实现Runge-Kutta方法和自适应时间步长
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import time


class TimeIntegrator:
    """时间积分器基类"""
    
    def __init__(self, order: int = 2):
        self.order = order
        self.dt = 0.01
        self.time = 0.0
        self.integration_time = 0.0
        self.steps_taken = 0
        
    def integrate(self, dt: float, system: Callable, initial_state: np.ndarray) -> np.ndarray:
        """积分系统"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_max_dt(self) -> float:
        """获取最大时间步长"""
        raise NotImplementedError("子类必须实现此方法")
    
    def get_integration_info(self) -> Dict:
        """获取积分信息"""
        return {
            'order': self.order,
            'dt': self.dt,
            'time': self.time,
            'integration_time': self.integration_time,
            'steps_taken': self.steps_taken
        }


class RungeKuttaIntegrator(TimeIntegrator):
    """Runge-Kutta时间积分器"""
    
    def __init__(self, order: int = 4):
        super().__init__(order=order)
        self.butcher_tableau = self._get_butcher_tableau(order)
        
    def integrate(self, dt: float, system: Callable, initial_state: np.ndarray) -> np.ndarray:
        """Runge-Kutta积分"""
        start_time = time.time()
        
        self.dt = dt
        n_stages = len(self.butcher_tableau['a'])
        
        # 初始化
        y = initial_state.copy()
        k = np.zeros((n_stages, len(y)))
        
        # 计算各阶段的斜率
        for i in range(n_stages):
            # 计算中间状态
            y_mid = y.copy()
            for j in range(i):
                y_mid += self.dt * self.butcher_tableau['a'][i][j] * k[j]
            
            # 计算斜率
            k[i] = system(self.time + self.dt * self.butcher_tableau['c'][i], y_mid)
        
        # 更新解
        for i in range(n_stages):
            y += self.dt * self.butcher_tableau['b'][i] * k[i]
        
        self.time += dt
        self.steps_taken += 1
        self.integration_time = time.time() - start_time
        
        return y
    
    def _get_butcher_tableau(self, order: int) -> Dict:
        """获取Butcher表"""
        if order == 1:
            # 前向Euler
            return {
                'a': [[0]],
                'b': [1],
                'c': [0]
            }
        elif order == 2:
            # 2阶Runge-Kutta
            return {
                'a': [[0, 0],
                      [0.5, 0]],
                'b': [0, 1],
                'c': [0, 0.5]
            }
        elif order == 4:
            # 4阶Runge-Kutta
            return {
                'a': [[0, 0, 0, 0],
                      [0.5, 0, 0, 0],
                      [0, 0.5, 0, 0],
                      [0, 0, 1, 0]],
                'b': [1/6, 1/3, 1/3, 1/6],
                'c': [0, 0.5, 0.5, 1]
            }
        else:
            raise ValueError(f"不支持的阶数: {order}")
    
    def get_max_dt(self) -> float:
        """获取最大时间步长"""
        # 简化的稳定性分析
        return 0.1


class AdaptiveTimeIntegrator(TimeIntegrator):
    """自适应时间积分器"""
    
    def __init__(self, tolerance: float = 1e-6, min_dt: float = 1e-8, max_dt: float = 0.1):
        super().__init__(order=4)
        self.tolerance = tolerance
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.error_history = []
        
    def integrate(self, dt: float, system: Callable, initial_state: np.ndarray) -> np.ndarray:
        """自适应积分"""
        start_time = time.time()
        
        # 使用两个不同阶数的方法估计误差
        integrator_high = RungeKuttaIntegrator(order=4)
        integrator_low = RungeKuttaIntegrator(order=2)
        
        # 计算高精度解
        y_high = integrator_high.integrate(dt, system, initial_state)
        
        # 计算低精度解
        y_low = integrator_low.integrate(dt, system, initial_state)
        
        # 估计误差
        error = np.linalg.norm(y_high - y_low)
        self.error_history.append(error)
        
        # 自适应调整时间步长
        if error > self.tolerance:
            # 误差太大，减小时间步长
            new_dt = dt * 0.5
            new_dt = max(new_dt, self.min_dt)
            
            if new_dt < dt:
                # 重新积分
                return self.integrate(new_dt, system, initial_state)
        else:
            # 误差可接受，可以增大时间步长
            if error < self.tolerance * 0.1:
                new_dt = min(dt * 1.5, self.max_dt)
                self.dt = new_dt
        
        self.time += dt
        self.steps_taken += 1
        self.integration_time = time.time() - start_time
        
        return y_high
    
    def get_max_dt(self) -> float:
        """获取最大时间步长"""
        return self.max_dt


class ImplicitTimeIntegrator(TimeIntegrator):
    """隐式时间积分器"""
    
    def __init__(self, order: int = 1, max_iterations: int = 100, tolerance: float = 1e-6):
        super().__init__(order=order)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def integrate(self, dt: float, system: Callable, initial_state: np.ndarray) -> np.ndarray:
        """隐式积分"""
        start_time = time.time()
        
        # 使用后向Euler方法
        y = initial_state.copy()
        
        for iteration in range(self.max_iterations):
            # 计算残差
            residual = y - initial_state - dt * system(self.time + dt, y)
            
            # 检查收敛性
            if np.linalg.norm(residual) < self.tolerance:
                break
            
            # 使用Newton法求解
            # 简化的实现：使用有限差分近似雅可比矩阵
            jacobian = self._compute_jacobian(system, self.time + dt, y, dt)
            
            # 求解线性系统
            delta_y = np.linalg.solve(jacobian, -residual)
            y += delta_y
        
        self.time += dt
        self.steps_taken += 1
        self.integration_time = time.time() - start_time
        
        return y
    
    def _compute_jacobian(self, system: Callable, t: float, y: np.ndarray, dt: float) -> np.ndarray:
        """计算雅可比矩阵"""
        n = len(y)
        jacobian = np.eye(n) - dt * self._finite_difference_jacobian(system, t, y)
        return jacobian
    
    def _finite_difference_jacobian(self, system: Callable, t: float, y: np.ndarray, 
                                  epsilon: float = 1e-8) -> np.ndarray:
        """使用有限差分计算雅可比矩阵"""
        n = len(y)
        jacobian = np.zeros((n, n))
        
        f0 = system(t, y)
        
        for i in range(n):
            y_perturbed = y.copy()
            y_perturbed[i] += epsilon
            f_perturbed = system(t, y_perturbed)
            jacobian[:, i] = (f_perturbed - f0) / epsilon
        
        return jacobian
    
    def get_max_dt(self) -> float:
        """获取最大时间步长"""
        return 0.1


def create_time_integrator(integrator_type: str = 'rk4', **kwargs) -> TimeIntegrator:
    """创建时间积分器"""
    if integrator_type == 'rk4':
        return RungeKuttaIntegrator(order=4, **kwargs)
    elif integrator_type == 'adaptive':
        return AdaptiveTimeIntegrator(**kwargs)
    elif integrator_type == 'implicit':
        return ImplicitTimeIntegrator(**kwargs)
    else:
        raise ValueError(f"不支持的积分器类型: {integrator_type}")


def demo_time_integration():
    """演示时间积分功能"""
    print("⏰ 时间积分演示")
    print("=" * 50)
    
    # 定义测试系统：简单的一阶常微分方程
    def test_system(t: float, y: np.ndarray) -> np.ndarray:
        """测试系统：dy/dt = -y"""
        return -y
    
    # 初始条件
    y0 = np.array([1.0])
    dt = 0.1
    t_final = 1.0
    
    # 测试不同的积分器
    integrators = {
        'rk4': RungeKuttaIntegrator(order=4),
        'adaptive': AdaptiveTimeIntegrator(tolerance=1e-6),
        'implicit': ImplicitTimeIntegrator()
    }
    
    for name, integrator in integrators.items():
        print(f"\n⏰ 测试 {name} 积分器...")
        
        # 重置积分器
        integrator.time = 0.0
        integrator.steps_taken = 0
        
        # 积分
        y = y0.copy()
        while integrator.time < t_final:
            y = integrator.integrate(dt, test_system, y)
        
        # 计算解析解
        analytical_solution = y0 * np.exp(-t_final)
        error = np.abs(y[0] - analytical_solution[0])
        
        info = integrator.get_integration_info()
        print(f"   积分时间: {info['integration_time']:.4f} 秒")
        print(f"   步数: {info['steps_taken']}")
        print(f"   最终时间: {info['time']:.4f}")
        print(f"   误差: {error:.2e}")
    
    print("\n✅ 时间积分演示完成!")


if __name__ == "__main__":
    demo_time_integration()
