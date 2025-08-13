"""
误差估计器实现
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import scipy.sparse as sp


@dataclass
class ErrorIndicator:
    """误差指示器"""
    element_id: int
    error_value: float
    error_type: str  # 'residual', 'recovery', 'gradient', 'hessian'
    refinement_flag: bool = False
    
    def __post_init__(self):
        if self.error_value < 0:
            raise ValueError("误差值不能为负数")


class BaseErrorEstimator(ABC):
    """误差估计器基类"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.error_indicators: List[ErrorIndicator] = []
    
    @abstractmethod
    def compute_error(self, solution: np.ndarray, mesh_data: Dict) -> List[ErrorIndicator]:
        """计算误差"""
        pass
    
    @abstractmethod
    def compute_global_error(self, error_indicators: List[ErrorIndicator]) -> float:
        """计算全局误差"""
        pass
    
    def estimate_convergence_rate(self, error_history: List[float]) -> float:
        """估计收敛率"""
        if len(error_history) < 2:
            return 0.0
        
        # 使用最小二乘法估计收敛率
        log_errors = np.log(error_history)
        n = len(log_errors)
        x = np.arange(n)
        
        # 线性拟合
        A = np.vstack([x, np.ones(n)]).T
        slope, _ = np.linalg.lstsq(A, log_errors, rcond=None)[0]
        
        return -slope  # 收敛率通常为负值


class ResidualErrorEstimator(BaseErrorEstimator):
    """残差误差估计器"""
    
    def __init__(self, tolerance: float = 1e-6):
        super().__init__(tolerance)
    
    def compute_residual(self, solution: np.ndarray, mesh_data: Dict) -> np.ndarray:
        """计算残差"""
        # 这里需要根据具体的PDE计算残差
        # 简化实现：假设残差为解的梯度
        if solution.ndim == 1:
            # 1D情况
            residual = np.gradient(solution)
        else:
            # 2D/3D情况
            residual = np.zeros_like(solution)
            for i in range(solution.ndim):
                residual += np.gradient(solution, axis=i) ** 2
            residual = np.sqrt(residual)
        
        return residual
    
    def compute_error(self, solution: np.ndarray, mesh_data: Dict) -> List[ErrorIndicator]:
        """计算残差误差"""
        residual = self.compute_residual(solution, mesh_data)
        
        # 计算每个单元的误差
        elements = mesh_data.get('elements', [])
        error_indicators = []
        
        for i, element in enumerate(elements):
            # 计算单元内的残差误差
            element_nodes = element['nodes']
            element_residual = residual[element_nodes]
            
            # 使用L2范数计算误差
            error_value = np.sqrt(np.mean(element_residual ** 2))
            
            # 创建误差指示器
            indicator = ErrorIndicator(
                element_id=i,
                error_value=error_value,
                error_type='residual',
                refinement_flag=error_value > self.tolerance
            )
            error_indicators.append(indicator)
        
        self.error_indicators = error_indicators
        return error_indicators
    
    def compute_global_error(self, error_indicators: List[ErrorIndicator]) -> float:
        """计算全局误差"""
        if not error_indicators:
            return 0.0
        
        error_values = [indicator.error_value for indicator in error_indicators]
        return np.sqrt(np.sum(np.array(error_values) ** 2))


class RecoveryErrorEstimator(BaseErrorEstimator):
    """恢复误差估计器（Zienkiewicz-Zhu方法）"""
    
    def __init__(self, tolerance: float = 1e-6):
        super().__init__(tolerance)
    
    def compute_recovered_gradient(self, solution: np.ndarray, mesh_data: Dict) -> np.ndarray:
        """计算恢复梯度"""
        # 使用Zienkiewicz-Zhu方法计算恢复梯度
        nodes = mesh_data.get('nodes', [])
        elements = mesh_data.get('elements', [])
        
        if not nodes or not elements:
            return np.zeros_like(solution)
        
        # 初始化恢复梯度
        recovered_gradient = np.zeros((len(nodes), solution.ndim))
        node_weights = np.zeros(len(nodes))
        
        # 对每个单元计算梯度并分配到节点
        for element in elements:
            element_nodes = element['nodes']
            element_coords = np.array([nodes[i] for i in element_nodes])
            element_solution = solution[element_nodes]
            
            # 计算单元梯度
            if len(element_nodes) == 3:  # 三角形
                gradient = self._compute_triangle_gradient(element_coords, element_solution)
            elif len(element_nodes) == 4:  # 四边形
                gradient = self._compute_quadrilateral_gradient(element_coords, element_solution)
            else:
                continue
            
            # 将梯度分配到节点
            for i, node_id in enumerate(element_nodes):
                recovered_gradient[node_id] += gradient
                node_weights[node_id] += 1
        
        # 平均化
        for i in range(len(nodes)):
            if node_weights[i] > 0:
                recovered_gradient[i] /= node_weights[i]
        
        return recovered_gradient
    
    def _compute_triangle_gradient(self, coords: np.ndarray, solution: np.ndarray) -> np.ndarray:
        """计算三角形单元的梯度"""
        # 使用有限差分计算梯度
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        
        # 计算面积
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        
        if area < 1e-12:
            return np.zeros(2)
        
        # 计算梯度
        dx = (solution[1] - solution[0]) * (y3 - y1) - (solution[2] - solution[0]) * (y2 - y1)
        dy = (solution[2] - solution[0]) * (x2 - x1) - (solution[1] - solution[0]) * (x3 - x1)
        
        gradient = np.array([dx, dy]) / (2 * area)
        return gradient
    
    def _compute_quadrilateral_gradient(self, coords: np.ndarray, solution: np.ndarray) -> np.ndarray:
        """计算四边形单元的梯度"""
        # 简化为两个三角形的组合
        # 这里可以实现更精确的四边形梯度计算
        return np.zeros(2)
    
    def compute_error(self, solution: np.ndarray, mesh_data: Dict) -> List[ErrorIndicator]:
        """计算恢复误差"""
        # 计算数值梯度
        numerical_gradient = self._compute_numerical_gradient(solution, mesh_data)
        
        # 计算恢复梯度
        recovered_gradient = self.compute_recovered_gradient(solution, mesh_data)
        
        # 计算误差
        elements = mesh_data.get('elements', [])
        error_indicators = []
        
        for i, element in enumerate(elements):
            element_nodes = element['nodes']
            
            # 计算单元内的梯度误差
            element_numerical = numerical_gradient[element_nodes]
            element_recovered = recovered_gradient[element_nodes]
            
            # 计算误差
            error = element_recovered - element_numerical
            error_value = np.sqrt(np.mean(error ** 2))
            
            indicator = ErrorIndicator(
                element_id=i,
                error_value=error_value,
                error_type='recovery',
                refinement_flag=error_value > self.tolerance
            )
            error_indicators.append(indicator)
        
        self.error_indicators = error_indicators
        return error_indicators
    
    def _compute_numerical_gradient(self, solution: np.ndarray, mesh_data: Dict) -> np.ndarray:
        """计算数值梯度"""
        # 简化实现：使用有限差分
        if solution.ndim == 1:
            return np.gradient(solution)
        else:
            gradient = np.zeros((len(solution), solution.ndim))
            for i in range(solution.ndim):
                gradient[:, i] = np.gradient(solution, axis=i)
            return gradient
    
    def compute_global_error(self, error_indicators: List[ErrorIndicator]) -> float:
        """计算全局误差"""
        if not error_indicators:
            return 0.0
        
        error_values = [indicator.error_value for indicator in error_indicators]
        return np.sqrt(np.sum(np.array(error_values) ** 2))


class StrainRateErrorEstimator(BaseErrorEstimator):
    """基于应变率的误差估计器 - 适用于地质力学问题"""
    
    def __init__(self, tolerance: float = 1e-6, strain_rate_threshold: float = 1e-6):
        super().__init__(tolerance)
        self.strain_rate_threshold = strain_rate_threshold
    
    def compute_strain_rate(self, displacement: np.ndarray, mesh_data: Dict) -> np.ndarray:
        """计算应变率"""
        # 这里需要根据具体的有限元实现来计算应变率
        # 简化实现：基于位移梯度
        
        if displacement.ndim == 1:
            # 1D情况
            strain_rate = np.gradient(displacement)
        else:
            # 2D/3D情况：计算位移梯度
            strain_rate = np.zeros_like(displacement)
            
            # 获取网格信息
            elements = mesh_data.get('elements', [])
            nodes = mesh_data.get('nodes', [])
            
            for element in elements:
                element_nodes = element.get('nodes', [])
                if len(element_nodes) >= 2:
                    # 计算单元内的应变率
                    element_displacement = displacement[element_nodes]
                    element_coords = nodes[element_nodes]
                    
                    # 简化的应变率计算
                    if len(element_nodes) == 2:  # 1D单元
                        strain_rate[element_nodes] = np.gradient(element_displacement)
                    else:  # 2D/3D单元
                        # 这里需要更复杂的应变率计算
                        # 简化：使用位移的梯度
                        for i in range(displacement.shape[1]):
                            strain_rate[element_nodes, i] = np.gradient(element_displacement[:, i])
        
        return strain_rate
    
    def compute_error(self, solution: np.ndarray, mesh_data: Dict) -> List[ErrorIndicator]:
        """计算基于应变率的误差"""
        # 假设solution包含位移场
        displacement = solution
        
        # 计算应变率
        strain_rate = self.compute_strain_rate(displacement, mesh_data)
        
        # 计算应变率幅值
        if strain_rate.ndim > 1:
            strain_rate_magnitude = np.sqrt(np.sum(strain_rate**2, axis=1))
        else:
            strain_rate_magnitude = np.abs(strain_rate)
        
        # 计算每个单元的误差
        elements = mesh_data.get('elements', [])
        error_indicators = []
        
        for i, element in enumerate(elements):
            element_nodes = element.get('nodes', [])
            if element_nodes:
                # 计算单元内的平均应变率
                element_strain_rate = strain_rate_magnitude[element_nodes]
                error_value = np.mean(element_strain_rate)
                
                # 创建误差指示器
                indicator = ErrorIndicator(
                    element_id=i,
                    error_value=error_value,
                    error_type="strain_rate",
                    refinement_flag=error_value > self.strain_rate_threshold
                )
                error_indicators.append(indicator)
        
        return error_indicators
    
    def compute_global_error(self, error_indicators: List[ErrorIndicator]) -> float:
        """计算全局误差"""
        if not error_indicators:
            return 0.0
        
        # 使用L2范数计算全局误差
        error_values = [indicator.error_value for indicator in error_indicators]
        return np.sqrt(np.mean(np.array(error_values) ** 2))
    
    def get_refinement_candidates(self, error_indicators: List[ErrorIndicator], 
                                refinement_ratio: float = 0.3) -> List[int]:
        """获取需要细化的单元候选"""
        if not error_indicators:
            return []
        
        # 按误差值排序
        sorted_indicators = sorted(error_indicators, key=lambda x: x.error_value, reverse=True)
        
        # 选择误差最大的单元进行细化
        n_refine = int(refinement_ratio * len(sorted_indicators))
        candidates = [indicator.element_id for indicator in sorted_indicators[:n_refine]]
        
        return candidates
    
    def get_coarsening_candidates(self, error_indicators: List[ErrorIndicator],
                                coarsening_ratio: float = 0.2) -> List[int]:
        """获取可以粗化的单元候选"""
        if not error_indicators:
            return []
        
        # 按误差值排序
        sorted_indicators = sorted(error_indicators, key=lambda x: x.error_value)
        
        # 选择误差最小的单元进行粗化
        n_coarsen = int(coarsening_ratio * len(sorted_indicators))
        candidates = [indicator.element_id for indicator in sorted_indicators[:n_coarsen]]
        
        return candidates


class GradientErrorEstimator(BaseErrorEstimator):
    """基于梯度的误差估计器"""
    
    def __init__(self, tolerance: float = 1e-6, gradient_threshold: float = 1e-3):
        super().__init__(tolerance)
        self.gradient_threshold = gradient_threshold
    
    def compute_gradient(self, solution: np.ndarray, mesh_data: Dict) -> np.ndarray:
        """计算解的梯度"""
        if solution.ndim == 1:
            # 1D情况
            gradient = np.gradient(solution)
        else:
            # 2D/3D情况
            gradient = np.zeros_like(solution)
            for i in range(solution.shape[1]):
                gradient[:, i] = np.gradient(solution[:, i])
        
        return gradient
    
    def compute_error(self, solution: np.ndarray, mesh_data: Dict) -> List[ErrorIndicator]:
        """计算基于梯度的误差"""
        gradient = self.compute_gradient(solution, mesh_data)
        
        # 计算梯度幅值
        if gradient.ndim > 1:
            gradient_magnitude = np.sqrt(np.sum(gradient**2, axis=1))
        else:
            gradient_magnitude = np.abs(gradient)
        
        # 计算每个单元的误差
        elements = mesh_data.get('elements', [])
        error_indicators = []
        
        for i, element in enumerate(elements):
            element_nodes = element.get('nodes', [])
            if element_nodes:
                # 计算单元内的平均梯度
                element_gradient = gradient_magnitude[element_nodes]
                error_value = np.mean(element_gradient)
                
                # 创建误差指示器
                indicator = ErrorIndicator(
                    element_id=i,
                    error_value=error_value,
                    error_type="gradient",
                    refinement_flag=error_value > self.gradient_threshold
                )
                error_indicators.append(indicator)
        
        return error_indicators
    
    def compute_global_error(self, error_indicators: List[ErrorIndicator]) -> float:
        """计算全局误差"""
        if not error_indicators:
            return 0.0
        
        # 使用L2范数计算全局误差
        error_values = [indicator.error_value for indicator in error_indicators]
        return np.sqrt(np.mean(np.array(error_values) ** 2))


class HessianErrorEstimator(BaseErrorEstimator):
    """基于Hessian的误差估计器 - 适用于高阶精度问题"""
    
    def __init__(self, tolerance: float = 1e-6, hessian_threshold: float = 1e-2):
        super().__init__(tolerance)
        self.hessian_threshold = hessian_threshold
    
    def compute_hessian(self, solution: np.ndarray, mesh_data: Dict) -> np.ndarray:
        """计算解的Hessian矩阵"""
        if solution.ndim == 1:
            # 1D情况：二阶导数
            hessian = np.gradient(np.gradient(solution))
        else:
            # 2D/3D情况：需要计算混合偏导数
            # 简化实现：只计算对角项
            hessian = np.zeros_like(solution)
            for i in range(solution.shape[1]):
                hessian[:, i] = np.gradient(np.gradient(solution[:, i]))
        
        return hessian
    
    def compute_error(self, solution: np.ndarray, mesh_data: Dict) -> List[ErrorIndicator]:
        """计算基于Hessian的误差"""
        hessian = self.compute_hessian(solution, mesh_data)
        
        # 计算Hessian的Frobenius范数
        if hessian.ndim > 1:
            hessian_norm = np.sqrt(np.sum(hessian**2, axis=1))
        else:
            hessian_norm = np.abs(hessian)
        
        # 计算每个单元的误差
        elements = mesh_data.get('elements', [])
        error_indicators = []
        
        for i, element in enumerate(elements):
            element_nodes = element.get('nodes', [])
            if element_nodes:
                # 计算单元内的平均Hessian范数
                element_hessian = hessian_norm[element_nodes]
                error_value = np.mean(element_hessian)
                
                # 创建误差指示器
                indicator = ErrorIndicator(
                    element_id=i,
                    error_value=error_value,
                    error_type="hessian",
                    refinement_flag=error_value > self.hessian_threshold
                )
                error_indicators.append(indicator)
        
        return error_indicators
    
    def compute_global_error(self, error_indicators: List[ErrorIndicator]) -> float:
        """计算全局误差"""
        if not error_indicators:
            return 0.0
        
        # 使用L2范数计算全局误差
        error_values = [indicator.error_value for indicator in error_indicators]
        return np.sqrt(np.mean(np.array(error_values) ** 2))


class AdaptiveErrorEstimator(BaseErrorEstimator):
    """自适应误差估计器 - 根据问题特性选择最佳估计方法"""
    
    def __init__(self, tolerance: float = 1e-6):
        super().__init__(tolerance)
        self.estimators = {
            'residual': ResidualErrorEstimator(tolerance),
            'strain_rate': StrainRateErrorEstimator(tolerance),
            'gradient': GradientErrorEstimator(tolerance),
            'hessian': HessianErrorEstimator(tolerance)
        }
        self.current_estimator = 'residual'
    
    def select_estimator(self, problem_features: Dict) -> str:
        """根据问题特性选择最佳误差估计器"""
        problem_type = problem_features.get('type', 'general')
        
        if problem_type == 'geomechanics':
            # 地质力学问题：优先使用应变率估计器
            return 'strain_rate'
        elif problem_type == 'fluid_dynamics':
            # 流体动力学：优先使用梯度估计器
            return 'gradient'
        elif problem_type == 'high_order':
            # 高阶精度问题：使用Hessian估计器
            return 'hessian'
        else:
            # 一般问题：使用残差估计器
            return 'residual'
    
    def compute_error(self, solution: np.ndarray, mesh_data: Dict, 
                     problem_features: Dict = None) -> List[ErrorIndicator]:
        """计算误差"""
        if problem_features:
            self.current_estimator = self.select_estimator(problem_features)
        
        estimator = self.estimators[self.current_estimator]
        return estimator.compute_error(solution, mesh_data)
    
    def compute_global_error(self, error_indicators: List[ErrorIndicator]) -> float:
        """计算全局误差"""
        estimator = self.estimators[self.current_estimator]
        return estimator.compute_global_error(error_indicators)
    
    def get_refinement_candidates(self, error_indicators: List[ErrorIndicator],
                                refinement_ratio: float = 0.3) -> List[int]:
        """获取需要细化的单元候选"""
        if self.current_estimator == 'strain_rate':
            estimator = self.estimators['strain_rate']
            return estimator.get_refinement_candidates(error_indicators, refinement_ratio)
        else:
            # 默认实现
            if not error_indicators:
                return []
            
            sorted_indicators = sorted(error_indicators, key=lambda x: x.error_value, reverse=True)
            n_refine = int(refinement_ratio * len(sorted_indicators))
            return [indicator.element_id for indicator in sorted_indicators[:n_refine]]


# 工厂函数
def create_error_estimator(estimator_type: str = 'adaptive', **kwargs) -> BaseErrorEstimator:
    """创建误差估计器"""
    if estimator_type == 'residual':
        return ResidualErrorEstimator(**kwargs)
    elif estimator_type == 'recovery':
        return RecoveryErrorEstimator(**kwargs)
    elif estimator_type == 'gradient':
        return GradientErrorEstimator(**kwargs)
    elif estimator_type == 'adaptive':
        return AdaptiveErrorEstimator(**kwargs)
    else:
        raise ValueError(f"不支持的误差估计器类型: {estimator_type}")


def demo_error_estimation():
    """演示误差估计功能 - 优化版本"""
    print("📊 误差估计演示")
    print("=" * 50)
    
    # 创建测试数据
    n_elements = 100
    solution = np.random.rand(n_elements)
    mesh_data = {
        'elements': [{'nodes': [i, i+1]} for i in range(n_elements-1)],
        'nodes': [[i, 0] for i in range(n_elements)],
        'element_types': ['line'] * (n_elements-1)
    }
    problem_data = {
        'material_properties': {'E': 1.0, 'nu': 0.3},
        'boundary_conditions': {'dirichlet': [0, n_elements-1]}
    }
    
    # 测试不同类型的误差估计器
    estimators = {
        'residual_energy': ResidualErrorEstimator(tolerance=1e-3),
        'residual_l2': ResidualErrorEstimator(tolerance=1e-3),
        'recovery_spr': RecoveryErrorEstimator(tolerance=1e-3),
        'adaptive': AdaptiveErrorEstimator(tolerance=1e-3, weights={'residual': 0.4, 'recovery': 0.4, 'gradient': 0.2}),
        'gradient': GradientErrorEstimator(tolerance=1e-3)
    }
    
    for name, estimator in estimators.items():
        print(f"\n🔍 测试 {name} 误差估计器...")
        
        try:
            error_indicators = estimator.compute_error(solution, mesh_data)
            global_error = estimator.compute_global_error(error_indicators)
            
            print(f"   计算时间: {estimator.computation_time:.4f} 秒")
            print(f"   平均误差: {estimator.error_history[-1]:.6f}")
            print(f"   全局误差: {global_error:.6f}")
            print(f"   需要细化的单元数: {np.sum([i.refinement_flag for i in error_indicators])}")
            
        except Exception as e:
            print(f"   ❌ 误差估计失败: {e}")
    
    print("\n✅ 误差估计演示完成!")


if __name__ == "__main__":
    demo_error_estimation() 