"""
网格细化器 - 提供网格细化、粗化和负载均衡功能
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class MeshRefiner:
    """网格细化器基类"""
    
    def __init__(self, refinement_type: str = 'h_refinement'):
        self.refinement_type = refinement_type
        self.refinement_history = []
        self.computation_time = 0.0
        
    def refine_mesh(self, mesh_data: Dict, refinement_indicator: np.ndarray, 
                   refinement_ratio: float = 0.3) -> Dict:
        """细化网格"""
        raise NotImplementedError("子类必须实现此方法")
    
    def coarsen_mesh(self, mesh_data: Dict, coarsening_indicator: np.ndarray, 
                    coarsening_ratio: float = 0.2) -> Dict:
        """粗化网格"""
        raise NotImplementedError("子类必须实现此方法")
    
    def maintain_mesh_quality(self, mesh_data: Dict) -> Dict:
        """保持网格质量"""
        raise NotImplementedError("子类必须实现此方法")


class HRefinement(MeshRefiner):
    """h-细化（网格细化）"""
    
    def __init__(self):
        super().__init__(refinement_type='h_refinement')
        
    def refine_mesh(self, mesh_data: Dict, refinement_indicator: np.ndarray, 
                   refinement_ratio: float = 0.3) -> Dict:
        """执行h-细化"""
        start_time = time.time()
        
        # 提取网格信息
        nodes = mesh_data.get('nodes', []).copy()
        elements = mesh_data.get('elements', []).copy()
        element_types = mesh_data.get('element_types', []).copy()
        
        # 确定需要细化的单元
        n_elements = len(elements)
        n_refine = int(refinement_ratio * n_elements)
        
        # 选择误差最大的单元进行细化
        sorted_indices = np.argsort(refinement_indicator)[::-1]
        refine_elements = sorted_indices[:n_refine]
        
        # 执行细化
        new_nodes, new_elements, new_element_types = self._perform_h_refinement(
            nodes, elements, element_types, refine_elements
        )
        
        # 更新网格数据
        refined_mesh = {
            'nodes': new_nodes,
            'elements': new_elements,
            'element_types': new_element_types,
            'refinement_level': mesh_data.get('refinement_level', 0) + 1
        }
        
        self.computation_time = time.time() - start_time
        self.refinement_history.append({
            'type': 'h_refinement',
            'elements_refined': len(refine_elements),
            'new_nodes': len(new_nodes) - len(nodes),
            'new_elements': len(new_elements) - len(elements)
        })
        
        return refined_mesh
    
    def _perform_h_refinement(self, nodes: List, elements: List, 
                             element_types: List, refine_elements: np.ndarray) -> Tuple:
        """执行具体的h-细化操作"""
        new_nodes = nodes.copy()
        new_elements = []
        new_element_types = []
        
        node_id_counter = len(nodes)
        
        for i, element in enumerate(elements):
            if i in refine_elements:
                # 细化单元
                refined_elements = self._refine_element(
                    element, new_nodes, node_id_counter
                )
                new_elements.extend(refined_elements)
                
                # 更新节点计数器
                node_id_counter += len(refined_elements) * 2  # 简化估计
                
                # 添加新的单元类型
                element_type = element_types[i]
                new_element_types.extend([element_type] * len(refined_elements))
            else:
                # 保持原单元
                new_elements.append(element)
                new_element_types.append(element_types[i])
        
        return new_nodes, new_elements, new_element_types
    
    def _refine_element(self, element: Dict, nodes: List, 
                       node_id_counter: int) -> List[Dict]:
        """细化单个单元"""
        element_nodes = element.get('nodes', [])
        element_type = element.get('type', 'quad')
        
        if element_type == 'line':
            # 1D线性单元细化
            return self._refine_line_element(element_nodes, node_id_counter)
        elif element_type == 'quad':
            # 2D四边形单元细化
            return self._refine_quad_element(element_nodes, nodes, node_id_counter)
        elif element_type == 'tri':
            # 2D三角形单元细化
            return self._refine_tri_element(element_nodes, nodes, node_id_counter)
        else:
            # 默认不细化
            return [element]
    
    def _refine_line_element(self, element_nodes: List, 
                           node_id_counter: int) -> List[Dict]:
        """细化1D线性单元"""
        if len(element_nodes) != 2:
            return [{'nodes': element_nodes}]
        
        # 创建中点
        mid_node = node_id_counter
        
        # 创建两个子单元
        sub_elements = [
            {'nodes': [element_nodes[0], mid_node]},
            {'nodes': [mid_node, element_nodes[1]]}
        ]
        
        return sub_elements
    
    def _refine_quad_element(self, element_nodes: List, nodes: List, 
                           node_id_counter: int) -> List[Dict]:
        """细化2D四边形单元"""
        if len(element_nodes) != 4:
            return [{'nodes': element_nodes}]
        
        # 创建边中点
        edge_midpoints = []
        for i in range(4):
            next_i = (i + 1) % 4
            mid_node = node_id_counter + i
            edge_midpoints.append(mid_node)
        
        # 创建中心点
        center_node = node_id_counter + 4
        
        # 创建四个子单元
        sub_elements = [
            {'nodes': [element_nodes[0], edge_midpoints[0], center_node, edge_midpoints[3]]},
            {'nodes': [edge_midpoints[0], element_nodes[1], edge_midpoints[1], center_node]},
            {'nodes': [center_node, edge_midpoints[1], element_nodes[2], edge_midpoints[2]]},
            {'nodes': [edge_midpoints[3], center_node, edge_midpoints[2], element_nodes[3]]}
        ]
        
        return sub_elements
    
    def _refine_tri_element(self, element_nodes: List, nodes: List, 
                          node_id_counter: int) -> List[Dict]:
        """细化2D三角形单元"""
        if len(element_nodes) != 3:
            return [{'nodes': element_nodes}]
        
        # 创建边中点
        edge_midpoints = []
        for i in range(3):
            next_i = (i + 1) % 3
            mid_node = node_id_counter + i
            edge_midpoints.append(mid_node)
        
        # 创建四个子三角形
        sub_elements = [
            {'nodes': [element_nodes[0], edge_midpoints[0], edge_midpoints[2]]},
            {'nodes': [edge_midpoints[0], element_nodes[1], edge_midpoints[1]]},
            {'nodes': [edge_midpoints[2], edge_midpoints[1], element_nodes[2]]},
            {'nodes': [edge_midpoints[0], edge_midpoints[1], edge_midpoints[2]]}
        ]
        
        return sub_elements
    
    def coarsen_mesh(self, mesh_data: Dict, coarsening_indicator: np.ndarray, 
                    coarsening_ratio: float = 0.2) -> Dict:
        """执行网格粗化"""
        start_time = time.time()
        
        # 简化的粗化实现
        nodes = mesh_data.get('nodes', []).copy()
        elements = mesh_data.get('elements', []).copy()
        element_types = mesh_data.get('element_types', []).copy()
        
        # 确定需要粗化的单元
        n_elements = len(elements)
        n_coarsen = int(coarsening_ratio * n_elements)
        
        # 选择误差最小的单元进行粗化
        sorted_indices = np.argsort(coarsening_indicator)
        coarsen_elements = sorted_indices[:n_coarsen]
        
        # 执行粗化（简化版本）
        coarsened_mesh = {
            'nodes': nodes,
            'elements': [elem for i, elem in enumerate(elements) if i not in coarsen_elements],
            'element_types': [elem_type for i, elem_type in enumerate(element_types) if i not in coarsen_elements],
            'refinement_level': max(0, mesh_data.get('refinement_level', 0) - 1)
        }
        
        self.computation_time = time.time() - start_time
        
        return coarsened_mesh
    
    def maintain_mesh_quality(self, mesh_data: Dict) -> Dict:
        """保持网格质量"""
        # 简化的网格质量检查
        nodes = mesh_data.get('nodes', [])
        elements = mesh_data.get('elements', [])
        
        # 检查单元质量
        quality_scores = []
        for element in elements:
            element_nodes = element.get('nodes', [])
            if len(element_nodes) >= 3:
                # 计算单元质量（简化版本）
                quality = self._compute_element_quality(element_nodes, nodes)
                quality_scores.append(quality)
        
        # 如果质量太差，进行优化
        if quality_scores and np.min(quality_scores) < 0.3:
            # 简化的网格优化
            optimized_mesh = self._optimize_mesh_quality(mesh_data)
            return optimized_mesh
        
        return mesh_data
    
    def _compute_element_quality(self, element_nodes: List, nodes: List) -> float:
        """计算单元质量"""
        if len(element_nodes) < 3:
            return 1.0
        
        # 简化的质量计算
        coords = [nodes[i] for i in element_nodes]
        coords = np.array(coords)
        
        if len(coords) == 3:  # 三角形
            # 计算面积
            area = 0.5 * abs(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
            # 计算边长
            edges = [
                np.linalg.norm(coords[1] - coords[0]),
                np.linalg.norm(coords[2] - coords[1]),
                np.linalg.norm(coords[0] - coords[2])
            ]
            # 质量指标（面积与周长的比值）
            perimeter = sum(edges)
            if perimeter > 0:
                quality = 4 * np.sqrt(3) * area / (perimeter ** 2)
                return max(0, min(1, quality))
        
        return 1.0
    
    def _optimize_mesh_quality(self, mesh_data: Dict) -> Dict:
        """优化网格质量"""
        # 简化的网格优化
        return mesh_data


class PRefinement(MeshRefiner):
    """p-细化（阶数提升）"""
    
    def __init__(self):
        super().__init__(refinement_type='p_refinement')
        
    def refine_mesh(self, mesh_data: Dict, refinement_indicator: np.ndarray, 
                   refinement_ratio: float = 0.3) -> Dict:
        """执行p-细化"""
        start_time = time.time()
        
        # 提取网格信息
        nodes = mesh_data.get('nodes', []).copy()
        elements = mesh_data.get('elements', []).copy()
        element_types = mesh_data.get('element_types', []).copy()
        polynomial_orders = mesh_data.get('polynomial_orders', [1] * len(elements)).copy()
        
        # 确定需要细化的单元
        n_elements = len(elements)
        n_refine = int(refinement_ratio * n_elements)
        
        # 选择误差最大的单元进行细化
        sorted_indices = np.argsort(refinement_indicator)[::-1]
        refine_elements = sorted_indices[:n_refine]
        
        # 提升多项式阶数
        for i in refine_elements:
            polynomial_orders[i] = min(polynomial_orders[i] + 1, 4)  # 最大4阶
        
        # 更新网格数据
        refined_mesh = {
            'nodes': nodes,
            'elements': elements,
            'element_types': element_types,
            'polynomial_orders': polynomial_orders,
            'refinement_level': mesh_data.get('refinement_level', 0) + 1
        }
        
        self.computation_time = time.time() - start_time
        self.refinement_history.append({
            'type': 'p_refinement',
            'elements_refined': len(refine_elements),
            'max_order': max(polynomial_orders)
        })
        
        return refined_mesh
    
    def coarsen_mesh(self, mesh_data: Dict, coarsening_indicator: np.ndarray, 
                    coarsening_ratio: float = 0.2) -> Dict:
        """执行p-粗化"""
        start_time = time.time()
        
        # 提取网格信息
        nodes = mesh_data.get('nodes', []).copy()
        elements = mesh_data.get('elements', []).copy()
        element_types = mesh_data.get('element_types', []).copy()
        polynomial_orders = mesh_data.get('polynomial_orders', [1] * len(elements)).copy()
        
        # 确定需要粗化的单元
        n_elements = len(elements)
        n_coarsen = int(coarsening_ratio * n_elements)
        
        # 选择误差最小的单元进行粗化
        sorted_indices = np.argsort(coarsening_indicator)
        coarsen_elements = sorted_indices[:n_coarsen]
        
        # 降低多项式阶数
        for i in coarsen_elements:
            polynomial_orders[i] = max(polynomial_orders[i] - 1, 1)  # 最小1阶
        
        # 更新网格数据
        coarsened_mesh = {
            'nodes': nodes,
            'elements': elements,
            'element_types': element_types,
            'polynomial_orders': polynomial_orders,
            'refinement_level': max(0, mesh_data.get('refinement_level', 0) - 1)
        }
        
        self.computation_time = time.time() - start_time
        
        return coarsened_mesh
    
    def maintain_mesh_quality(self, mesh_data: Dict) -> Dict:
        """保持网格质量（p-细化不需要网格质量检查）"""
        return mesh_data


class AdaptiveMeshRefiner(MeshRefiner):
    """自适应网格细化器"""
    
    def __init__(self, h_refiner: HRefinement = None, p_refiner: PRefinement = None):
        super().__init__(refinement_type='adaptive')
        
        self.h_refiner = h_refiner or HRefinement()
        self.p_refiner = p_refiner or PRefinement()
        
    def refine_mesh(self, mesh_data: Dict, refinement_indicator: np.ndarray, 
                   refinement_ratio: float = 0.3, method: str = 'h') -> Dict:
        """执行自适应细化"""
        if method == 'h':
            return self.h_refiner.refine_mesh(mesh_data, refinement_indicator, refinement_ratio)
        elif method == 'p':
            return self.p_refiner.refine_mesh(mesh_data, refinement_indicator, refinement_ratio)
        elif method == 'hp':
            # hp-自适应细化
            return self._hp_refinement(mesh_data, refinement_indicator, refinement_ratio)
        else:
            raise ValueError(f"不支持的细化方法: {method}")
    
    def _hp_refinement(self, mesh_data: Dict, refinement_indicator: np.ndarray, 
                      refinement_ratio: float) -> Dict:
        """hp-自适应细化"""
        # 简化的hp-自适应实现
        # 根据误差大小选择h-细化或p-细化
        
        # 选择误差最大的单元
        n_elements = len(mesh_data.get('elements', []))
        n_refine = int(refinement_ratio * n_elements)
        sorted_indices = np.argsort(refinement_indicator)[::-1]
        refine_elements = sorted_indices[:n_refine]
        
        # 根据误差大小选择细化方法
        h_refine_elements = []
        p_refine_elements = []
        
        for i in refine_elements:
            if refinement_indicator[i] > 0.7:  # 高误差用h-细化
                h_refine_elements.append(i)
            else:  # 低误差用p-细化
                p_refine_elements.append(i)
        
        # 执行细化
        refined_mesh = mesh_data.copy()
        
        if h_refine_elements:
            h_indicator = np.zeros_like(refinement_indicator)
            h_indicator[h_refine_elements] = refinement_indicator[h_refine_elements]
            refined_mesh = self.h_refiner.refine_mesh(refined_mesh, h_indicator, 1.0)
        
        if p_refine_elements:
            p_indicator = np.zeros_like(refinement_indicator)
            p_indicator[p_refine_elements] = refinement_indicator[p_refine_elements]
            refined_mesh = self.p_refiner.refine_mesh(refined_mesh, p_indicator, 1.0)
        
        return refined_mesh
    
    def coarsen_mesh(self, mesh_data: Dict, coarsening_indicator: np.ndarray, 
                    coarsening_ratio: float = 0.2) -> Dict:
        """执行自适应粗化"""
        # 简化的自适应粗化
        return self.h_refiner.coarsen_mesh(mesh_data, coarsening_indicator, coarsening_ratio)
    
    def maintain_mesh_quality(self, mesh_data: Dict) -> Dict:
        """保持网格质量"""
        return self.h_refiner.maintain_mesh_quality(mesh_data)


def create_mesh_refiner(refinement_type: str = 'h') -> MeshRefiner:
    """创建网格细化器"""
    if refinement_type == 'h':
        return HRefinement()
    elif refinement_type == 'p':
        return PRefinement()
    else:
        raise ValueError(f"不支持的细化类型: {refinement_type}")


def demo_mesh_refinement():
    """演示网格细化功能"""
    print("🔧 网格细化演示")
    print("=" * 50)
    
    # 创建测试网格
    mesh_data = {
        'nodes': [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]],
        'elements': [
            {'nodes': [0, 1, 4]},
            {'nodes': [1, 2, 4]},
            {'nodes': [2, 3, 4]},
            {'nodes': [3, 0, 4]}
        ],
        'element_types': ['tri', 'tri', 'tri', 'tri'],
        'refinement_level': 0
    }
    
    # 创建误差指标
    refinement_indicator = np.array([0.8, 0.3, 0.6, 0.2])
    
    # 测试不同类型的细化器
    refiners = {
        'h_refinement': HRefinement(),
        'p_refinement': PRefinement()
    }
    
    for name, refiner in refiners.items():
        print(f"\n🔧 测试 {name}...")
        
        refined_mesh = refiner.refine_mesh(mesh_data, refinement_indicator, refinement_ratio=0.5)
        
        print(f"   原始单元数: {len(mesh_data['elements'])}")
        print(f"   细化后单元数: {len(refined_mesh['elements'])}")
        print(f"   细化级别: {refined_mesh.get('refinement_level', 0)}")
        print(f"   计算时间: {refiner.computation_time:.4f} 秒")
        
        if name == 'p_refinement':
            orders = refined_mesh.get('polynomial_orders', [])
            print(f"   多项式阶数: {orders}")
    
    print("\n✅ 网格细化演示完成!")


if __name__ == "__main__":
    demo_mesh_refinement() 