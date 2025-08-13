"""
高级时间积分器 - 增强版

包含以下增强功能：
1. 隐式时间步进算法：BDF、Crank-Nicolson等
2. 多物理场耦合支持
3. 自适应时间步长控制
4. 高阶精度方法
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import time
from scipy.sparse import spmatrix
from scipy.sparse.linalg import spsolve, cg, gmres
from scipy.optimize import fsolve
import warnings


class AdvancedTimeIntegrator:
    """高级时间积分器基类"""
    
    def __init__(self, order: int = 2):
        self.order = order
        self.dt = 0.01
        self.time = 0.0
        self.integration_time = 0.0
        self.steps_taken = 0
        self.solution_history = []
        self.time_history = []
        
        # 性能统计
        self.performance_stats = {
            'total_steps': 0,
            'successful_steps': 0,
            'failed_steps': 0,
            'total_time': 0.0,
            'linear_solves': 0,
            'nonlinear_iterations': 0
        }
    
    def integrate(self, dt: float, system: Callable, initial_state: np.ndarray, 
                 **kwargs) -> np.ndarray:
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
            'integration_time': self.performance_stats['total_time'],
            'steps_taken': self.steps_taken,
            'performance_stats': self.performance_stats.copy()
        }
    
    def get_solution_history(self) -> Tuple[List[float], List[np.ndarray]]:
        """获取解的历史"""
        return self.time_history, self.solution_history


class BDFIntegrator(AdvancedTimeIntegrator):
    """后向差分公式(BDF)积分器"""
    
    def __init__(self, order: int = 2):
        super().__init__(order)
        self.bdf_coefficients = self._get_bdf_coefficients(order)
        self.past_solutions = []
    
    def _get_bdf_coefficients(self, order: int) -> List[float]:
        """获取BDF系数"""
        if order == 1:
            return [1.0, -1.0]  # y_n - y_{n-1} = dt * f_n
        elif order == 2:
            return [3.0/2, -2.0, 1.0/2]  # 3/2*y_n - 2*y_{n-1} + 1/2*y_{n-2} = dt * f_n
        elif order == 3:
            return [11.0/6, -3.0, 3.0/2, -1.0/3]
        elif order == 4:
            return [25.0/12, -4.0, 3.0, -4.0/3, 1.0/4]
        else:
            raise ValueError(f"不支持的BDF阶数: {order}")
    
    def integrate(self, dt: float, system: Callable, initial_state: np.ndarray, 
                 **kwargs) -> np.ndarray:
        """BDF积分"""
        start_time = time.time()
        
        self.dt = dt
        current_solution = initial_state.copy()
        
        # 初始化历史解
        if len(self.past_solutions) < self.order - 1:
            # 使用Runge-Kutta方法生成历史解
            self._initialize_history(dt, system, initial_state)
        
        # BDF积分
        for step in range(kwargs.get('max_steps', 1000)):
            # 保存当前解
            self.solution_history.append(current_solution.copy())
            self.time_history.append(self.time)
            
            # 计算BDF右端项
            bdf_rhs = self._compute_bdf_rhs(current_solution)
            
            # 求解非线性系统
            if kwargs.get('implicit', True):
                current_solution = self._solve_implicit_step(dt, system, bdf_rhs, current_solution)
            else:
                current_solution = self._solve_explicit_step(dt, system, bdf_rhs, current_solution)
            
            # 更新历史
            self._update_history(current_solution)
            
            # 更新时间
            self.time += dt
            self.steps_taken += 1
            self.performance_stats['total_steps'] += 1
            
            # 检查终止条件
            if kwargs.get('end_time') and self.time >= kwargs['end_time']:
                break
        
        self.performance_stats['total_time'] = time.time() - start_time
        return current_solution
    
    def _initialize_history(self, dt: float, system: Callable, initial_state: np.ndarray):
        """初始化历史解"""
        # 使用4阶Runge-Kutta方法生成历史解
        y = initial_state.copy()
        t = 0.0
        
        for _ in range(self.order - 1):
            # RK4步进
            k1 = system(t, y)
            k2 = system(t + dt/2, y + dt/2 * k1)
            k3 = system(t + dt/2, y + dt/2 * k2)
            k4 = system(t + dt, y + dt * k3)
            
            y = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += dt
            
            self.past_solutions.append(y.copy())
    
    def _compute_bdf_rhs(self, current_solution: np.ndarray) -> np.ndarray:
        """计算BDF右端项"""
        rhs = np.zeros_like(current_solution)
        
        # 应用BDF系数
        rhs += self.bdf_coefficients[0] * current_solution
        
        for i, coef in enumerate(self.bdf_coefficients[1:], 1):
            if i <= len(self.past_solutions):
                rhs += coef * self.past_solutions[-i]
        
        return rhs
    
    def _solve_implicit_step(self, dt: float, system: Callable, bdf_rhs: np.ndarray, 
                            current_guess: np.ndarray) -> np.ndarray:
        """求解隐式步"""
        # 使用牛顿法求解非线性系统
        
        def residual(y):
            """残差函数"""
            return y - bdf_rhs - dt * system(self.time, y)
        
        def jacobian(y):
            """雅可比矩阵（简化版本）"""
            # 这里应该计算真正的雅可比矩阵
            # 简化版本：单位矩阵
            return np.eye(len(y))
        
        # 牛顿迭代
        y = current_guess.copy()
        max_iter = 10
        tol = 1e-8
        
        for iter_count in range(max_iter):
            r = residual(y)
            if np.linalg.norm(r) < tol:
                break
            
            J = jacobian(y)
            dy = np.linalg.solve(J, -r)
            y += dy
            
            self.performance_stats['nonlinear_iterations'] += 1
        
        return y
    
    def _solve_explicit_step(self, dt: float, system: Callable, bdf_rhs: np.ndarray, 
                            current_solution: np.ndarray) -> np.ndarray:
        """求解显式步"""
        # 显式BDF（通常不稳定，仅用于测试）
        f_n = system(self.time, current_solution)
        return bdf_rhs + dt * f_n
    
    def _update_history(self, new_solution: np.ndarray):
        """更新历史解"""
        self.past_solutions.append(new_solution.copy())
        if len(self.past_solutions) > self.order - 1:
            self.past_solutions.pop(0)
    
    def get_max_dt(self) -> float:
        """获取最大时间步长"""
        # BDF的稳定性限制
        if self.order == 1:
            return 2.0  # 无条件稳定
        elif self.order == 2:
            return 1.0  # 稳定性限制
        else:
            return 0.5  # 高阶BDF的稳定性限制


class CrankNicolsonIntegrator(AdvancedTimeIntegrator):
    """Crank-Nicolson积分器"""
    
    def __init__(self, order: int = 2):
        super().__init__(order)
        self.theta = 0.5  # Crank-Nicolson参数
    
    def integrate(self, dt: float, system: Callable, initial_state: np.ndarray, 
                 **kwargs) -> np.ndarray:
        """Crank-Nicolson积分"""
        start_time = time.time()
        
        self.dt = dt
        current_solution = initial_state.copy()
        
        # Crank-Nicolson积分
        for step in range(kwargs.get('max_steps', 1000)):
            # 保存当前解
            self.solution_history.append(current_solution.copy())
            self.time_history.append(self.time)
            
            # 求解隐式步
            current_solution = self._solve_crank_nicolson_step(dt, system, current_solution)
            
            # 更新时间
            self.time += dt
            self.steps_taken += 1
            self.performance_stats['total_steps'] += 1
            
            # 检查终止条件
            if kwargs.get('end_time') and self.time >= kwargs['end_time']:
                break
        
        self.performance_stats['total_time'] = time.time() - start_time
        return current_solution
    
    def _solve_crank_nicolson_step(self, dt: float, system: Callable, 
                                  current_solution: np.ndarray) -> np.ndarray:
        """求解Crank-Nicolson步"""
        # Crank-Nicolson格式：y_{n+1} - y_n = dt/2 * (f_{n+1} + f_n)
        
        def residual(y_next):
            """残差函数"""
            f_n = system(self.time, current_solution)
            f_next = system(self.time + dt, y_next)
            return y_next - current_solution - dt/2 * (f_next + f_n)
        
        def jacobian(y_next):
            """雅可比矩阵"""
            # 这里应该计算真正的雅可比矩阵
            # 简化版本：单位矩阵
            return np.eye(len(y_next)) - dt/2 * np.eye(len(y_next))
        
        # 牛顿迭代求解
        y_next = current_solution.copy()
        max_iter = 10
        tol = 1e-8
        
        for iter_count in range(max_iter):
            r = residual(y_next)
            if np.linalg.norm(r) < tol:
                break
            
            J = jacobian(y_next)
            dy = np.linalg.solve(J, -r)
            y_next += dy
            
            self.performance_stats['nonlinear_iterations'] += 1
        
        return y_next
    
    def get_max_dt(self) -> float:
        """获取最大时间步长"""
        # Crank-Nicolson通常无条件稳定
        return 1.0


class AdaptiveTimeIntegrator(AdvancedTimeIntegrator):
    """自适应时间步长积分器"""
    
    def __init__(self, base_integrator: AdvancedTimeIntegrator, 
                 tolerance: float = 1e-6):
        super().__init__(base_integrator.order)
        self.base_integrator = base_integrator
        self.tolerance = tolerance
        self.min_dt = 1e-6
        self.max_dt = 1.0
        self.safety_factor = 0.9
        
    def integrate(self, dt: float, system: Callable, initial_state: np.ndarray, 
                 **kwargs) -> np.ndarray:
        """自适应积分"""
        start_time = time.time()
        
        current_dt = dt
        current_solution = initial_state.copy()
        
        # 自适应积分
        while self.time < kwargs.get('end_time', float('inf')):
            # 尝试时间步
            try:
                # 使用两个不同阶数的方法估计误差
                solution_high = self._step_with_order(current_dt, system, current_solution, 
                                                   self.base_integrator.order)
                solution_low = self._step_with_order(current_dt, system, current_solution, 
                                                  self.base_integrator.order - 1)
                
                # 估计误差
                error = np.linalg.norm(solution_high - solution_low)
                
                if error < self.tolerance:
                    # 步长成功，接受解
                    current_solution = solution_high
                    self.time += current_dt
                    self.steps_taken += 1
                    self.performance_stats['successful_steps'] += 1
                    
                    # 保存历史
                    self.solution_history.append(current_solution.copy())
                    self.time_history.append(self.time)
                    
                    # 增加时间步长
                    current_dt = min(self.max_dt, 
                                   current_dt * self.safety_factor * (self.tolerance / error) ** (1.0 / self.base_integrator.order))
                else:
                    # 步长失败，减小时间步长
                    current_dt = max(self.min_dt, 
                                   current_dt * self.safety_factor * (self.tolerance / error) ** (1.0 / self.base_integrator.order))
                    self.performance_stats['failed_steps'] += 1
                
            except Exception as e:
                # 步长失败，减小时间步长
                current_dt = max(self.min_dt, current_dt * 0.5)
                self.performance_stats['failed_steps'] += 1
                warnings.warn(f"时间步失败: {e}")
            
            self.performance_stats['total_steps'] += 1
            
            # 检查终止条件
            if kwargs.get('end_time') and self.time >= kwargs['end_time']:
                break
        
        self.performance_stats['total_time'] = time.time() - start_time
        return current_solution
    
    def _step_with_order(self, dt: float, system: Callable, current_solution: np.ndarray, 
                        order: int) -> np.ndarray:
        """使用指定阶数进行时间步进"""
        # 创建临时积分器
        if isinstance(self.base_integrator, BDFIntegrator):
            temp_integrator = BDFIntegrator(order)
        elif isinstance(self.base_integrator, CrankNicolsonIntegrator):
            temp_integrator = CrankNicolsonIntegrator(order)
        else:
            temp_integrator = self.base_integrator.__class__(order)
        
        # 执行单步
        return temp_integrator.integrate(dt, system, current_solution, max_steps=1)
    
    def get_max_dt(self) -> float:
        """获取最大时间步长"""
        return self.max_dt


class MultiphysicsTimeIntegrator(AdvancedTimeIntegrator):
    """多物理场时间积分器"""
    
    def __init__(self, integrators: Dict[str, AdvancedTimeIntegrator]):
        super().__init__(max(integrator.order for integrator in integrators.values()))
        self.integrators = integrators
        self.field_solutions = {}
        self.coupling_operators = {}
    
    def add_coupling_operator(self, field1: str, field2: str, operator: Callable):
        """添加耦合算子"""
        key = f"{field1}_to_{field2}"
        self.coupling_operators[key] = operator
    
    def integrate(self, dt: float, systems: Dict[str, Callable], 
                 initial_states: Dict[str, np.ndarray], **kwargs) -> Dict[str, np.ndarray]:
        """多物理场积分"""
        start_time = time.time()
        
        self.dt = dt
        self.field_solutions = {name: state.copy() for name, state in initial_states.items()}
        
        # 多物理场积分
        for step in range(kwargs.get('max_steps', 1000)):
            # 保存当前解
            for field_name, solution in self.field_solutions.items():
                if field_name not in self.solution_history:
                    self.solution_history.append({})
                self.solution_history[-1][field_name] = solution.copy()
            
            self.time_history.append(self.time)
            
            # 求解耦合系统
            self._solve_coupled_step(dt, systems)
            
            # 更新时间
            self.time += dt
            self.steps_taken += 1
            self.performance_stats['total_steps'] += 1
            
            # 检查终止条件
            if kwargs.get('end_time') and self.time >= kwargs['end_time']:
                break
        
        self.performance_stats['total_time'] = time.time() - start_time
        return self.field_solutions.copy()
    
    def _solve_coupled_step(self, dt: float, systems: Dict[str, Callable]):
        """求解耦合时间步"""
        # 使用分离式迭代求解耦合系统
        
        max_iter = kwargs.get('coupling_iterations', 5)
        tolerance = kwargs.get('coupling_tolerance', 1e-6)
        
        for iter_count in range(max_iter):
            prev_solutions = {name: sol.copy() for name, sol in self.field_solutions.items()}
            
            # 依次求解各物理场
            for field_name, integrator in self.integrators.items():
                if field_name in systems:
                    # 计算耦合项
                    coupling_rhs = self._compute_coupling_rhs(field_name)
                    
                    # 求解物理场
                    system_with_coupling = lambda t, y: systems[field_name](t, y) + coupling_rhs
                    self.field_solutions[field_name] = integrator.integrate(
                        dt, system_with_coupling, self.field_solutions[field_name], max_steps=1
                    )
            
            # 检查耦合收敛性
            max_change = 0.0
            for field_name in self.field_solutions:
                if field_name in prev_solutions:
                    change = np.linalg.norm(
                        self.field_solutions[field_name] - prev_solutions[field_name]
                    ) / (np.linalg.norm(self.field_solutions[field_name]) + 1e-10)
                    max_change = max(max_change, change)
            
            if max_change < tolerance:
                break
    
    def _compute_coupling_rhs(self, field_name: str) -> np.ndarray:
        """计算耦合右端项"""
        coupling_rhs = np.zeros_like(self.field_solutions[field_name])
        
        for coupling_key, operator in self.coupling_operators.items():
            if coupling_key.endswith(f"_to_{field_name}"):
                source_field_name = coupling_key.split("_to_")[0]
                if source_field_name in self.field_solutions:
                    source_solution = self.field_solutions[source_field_name]
                    coupling_effect = operator(source_solution)
                    coupling_rhs += coupling_effect
        
        return coupling_rhs
    
    def get_max_dt(self) -> float:
        """获取最大时间步长"""
        # 取所有积分器的最小值
        return min(integrator.get_max_dt() for integrator in self.integrators.values())


# 工厂函数
def create_time_integrator(integrator_type: str = 'bdf', **kwargs) -> AdvancedTimeIntegrator:
    """创建时间积分器"""
    if integrator_type == 'bdf':
        order = kwargs.get('order', 2)
        return BDFIntegrator(order)
    elif integrator_type == 'crank_nicolson':
        order = kwargs.get('order', 2)
        return CrankNicolsonIntegrator(order)
    elif integrator_type == 'adaptive':
        base_type = kwargs.get('base_type', 'bdf')
        base_integrator = create_time_integrator(base_type, **kwargs)
        tolerance = kwargs.get('tolerance', 1e-6)
        return AdaptiveTimeIntegrator(base_integrator, tolerance)
    else:
        raise ValueError(f"不支持的时间积分器类型: {integrator_type}")


def benchmark_time_integrators(system: Callable, initial_state: np.ndarray, 
                              time_span: Tuple[float, float], dt: float = 0.01) -> Dict:
    """时间积分器性能基准测试"""
    integrator_types = ['bdf', 'crank_nicolson', 'adaptive']
    results = {}
    
    for integrator_type in integrator_types:
        print(f"\n🧪 测试时间积分器: {integrator_type}")
        
        # 创建积分器
        integrator = create_time_integrator(integrator_type, order=2)
        
        # 积分
        start_time = time.time()
        final_solution = integrator.integrate(dt, system, initial_state, 
                                           end_time=time_span[1])
        solve_time = time.time() - start_time
        
        # 计算误差（如果有解析解）
        error = np.linalg.norm(final_solution - initial_state)  # 简化误差计算
        
        # 存储结果
        results[integrator_type] = {
            'solve_time': solve_time,
            'steps_taken': integrator.steps_taken,
            'final_error': error,
            'performance_stats': integrator.get_performance_stats()
        }
        
        print(f"   求解时间: {solve_time:.4f}s")
        print(f"   步数: {integrator.steps_taken}")
        print(f"   最终误差: {error:.2e}")
    
    return results
