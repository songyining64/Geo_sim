"""
动态通信调度演示

展示升级后的CommunicationOptimizer如何根据实时负载
动态调整通信调度，减少节点空闲时间。
"""

import numpy as np
import time
import threading
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# 导入升级后的通信优化器
from advanced_parallel_solver import CommunicationOptimizer, LoadMonitor

# 模拟MPI通信环境
class MockMPIComm:
    """模拟MPI通信环境"""
    
    def __init__(self, rank: int, size: int):
        self.rank = rank
        self.size = size
        self.loads = {i: 0.5 for i in range(size)}  # 初始负载
    
    def Get_rank(self) -> int:
        return self.rank
    
    def Get_size(self) -> int:
        return self.size
    
    def allgather(self, local_load: float) -> List[float]:
        """模拟allgather操作"""
        # 模拟负载变化
        for i in range(self.size):
            if i != self.rank:
                # 随机负载变化
                self.loads[i] = max(0.1, min(1.0, 
                    self.loads[i] + np.random.normal(0, 0.1)))
        
        return [self.loads[i] for i in range(self.size)]


def create_synthetic_partition_info(size: int, num_boundary_nodes: int = 20) -> Dict:
    """创建合成的分区信息"""
    partition_info = {
        'local_elements': {},
        'boundary_nodes': {},
        'ghost_nodes': {}
    }
    
    # 为每个进程创建边界节点
    for rank in range(size):
        # 创建一些边界节点
        boundary_nodes = set()
        for i in range(num_boundary_nodes):
            # 随机选择一些节点作为边界节点
            if np.random.random() < 0.3:
                boundary_nodes.add(f"node_{rank}_{i}")
        
        partition_info['boundary_nodes'][rank] = boundary_nodes
    
    return partition_info


def simulate_dynamic_communication_optimization():
    """模拟动态通信优化过程"""
    print("=== 动态通信调度优化演示 ===\n")
    
    # 创建模拟MPI环境
    size = 4
    comm = MockMPIComm(rank=0, size=size)
    
    # 创建通信优化器
    optimizer = CommunicationOptimizer(comm)
    
    # 创建分区信息
    partition_info = create_synthetic_partition_info(size)
    
    print("初始分区信息:")
    for rank in range(size):
        boundary_nodes = partition_info['boundary_nodes'].get(rank, set())
        print(f"  进程 {rank}: {len(boundary_nodes)} 个边界节点")
    
    print("\n开始动态通信优化...")
    
    # 模拟多次通信优化
    communication_results = []
    load_history = []
    
    for step in range(20):
        print(f"\n--- 通信步骤 {step + 1} ---")
        
        # 获取优化后的通信调度
        result = optimizer.optimize_communication_schedule(partition_info)
        
        # 记录结果
        communication_results.append(result)
        
        # 显示实时负载
        real_time_loads = result['real_time_loads']
        print(f"实时负载分布:")
        for rank, load in real_time_loads.items():
            print(f"  进程 {rank}: {load:.3f}")
        
        # 显示通信调度
        schedule = result['communication_schedule']
        print(f"通信调度:")
        print(f"  发送序列: {schedule['send_sequence']}")
        print(f"  接收序列: {schedule['receive_sequence']}")
        print(f"  负载均衡序列: {len(schedule['load_balanced_sequence'])} 个邻居")
        
        # 显示优先级队列
        if schedule['priority_queue']:
            print(f"  优先级队列 (前3个):")
            for i, (neighbor, priority) in enumerate(schedule['priority_queue'][:3]):
                print(f"    邻居 {neighbor}: 优先级 {priority:.3f}")
        
        # 记录负载历史
        load_history.append(real_time_loads.copy())
        
        # 模拟时间流逝
        time.sleep(0.1)
    
    # 显示统计信息
    print("\n=== 通信优化统计 ===")
    stats = optimizer.get_communication_statistics()
    print(f"总通信次数: {stats['total_communications']}")
    print(f"调度更新次数: {stats['schedule_updates']}")
    
    load_stats = stats['load_monitoring']
    if load_stats:
        print(f"负载监控统计:")
        print(f"  当前负载: {load_stats['current_load']:.3f}")
        print(f"  平均负载: {load_stats['average_load']:.3f}")
        print(f"  负载方差: {load_stats['load_variance']:.3f}")
        print(f"  CPU使用率: {load_stats['cpu_usage']:.1f}%")
        print(f"  内存使用率: {load_stats['memory_usage']:.1f}%")
        print(f"  GPU使用率: {load_stats['gpu_usage']:.1f}%")
    
    return communication_results, load_history


def analyze_communication_efficiency(communication_results: List[Dict]):
    """分析通信效率"""
    print("\n=== 通信效率分析 ===")
    
    # 统计调度更新频率
    schedule_updates = 0
    total_communications = len(communication_results)
    
    for i, result in enumerate(communication_results):
        if result.get('dynamic_optimization', False):
            schedule_updates += 1
    
    update_rate = schedule_updates / total_communications if total_communications > 0 else 0
    print(f"调度更新率: {update_rate:.2%}")
    
    # 分析负载分布变化
    load_variations = []
    for result in communication_results:
        loads = result.get('real_time_loads', {})
        if loads:
            load_std = np.std(list(loads.values()))
            load_variations.append(load_std)
    
    if load_variations:
        avg_load_variation = np.mean(load_variations)
        print(f"平均负载变化: {avg_load_variation:.3f}")
    
    # 分析通信模式
    communication_patterns = []
    for result in communication_results:
        pattern = result.get('communication_pattern', {})
        if pattern.get('neighbors'):
            num_neighbors = len(pattern['neighbors'])
            communication_patterns.append(num_neighbors)
    
    if communication_patterns:
        avg_neighbors = np.mean(communication_patterns)
        print(f"平均邻居数量: {avg_neighbors:.1f}")


def visualize_communication_optimization(load_history: List[Dict]):
    """可视化通信优化过程"""
    if not load_history:
        return
    
    print("\n生成可视化图表...")
    
    # 准备数据
    ranks = list(load_history[0].keys())
    steps = list(range(len(load_history)))
    
    # 创建负载变化图
    plt.figure(figsize=(12, 8))
    
    # 子图1: 负载变化
    plt.subplot(2, 2, 1)
    for rank in ranks:
        loads = [load_history[step].get(rank, 0) for step in steps]
        plt.plot(steps, loads, marker='o', label=f'进程 {rank}')
    
    plt.xlabel('通信步骤')
    plt.ylabel('负载')
    plt.title('实时负载变化')
    plt.legend()
    plt.grid(True)
    
    # 子图2: 负载分布箱线图
    plt.subplot(2, 2, 2)
    load_data = []
    labels = []
    for rank in ranks:
        loads = [load_history[step].get(rank, 0) for step in steps]
        load_data.append(loads)
        labels.append(f'进程 {rank}')
    
    plt.boxplot(load_data, labels=labels)
    plt.ylabel('负载分布')
    plt.title('负载分布统计')
    plt.grid(True)
    
    # 子图3: 负载不平衡度
    plt.subplot(2, 2, 3)
    imbalance_measures = []
    for step in steps:
        loads = list(load_history[step].values())
        if loads:
            max_load = max(loads)
            min_load = min(loads)
            avg_load = np.mean(loads)
            if avg_load > 0:
                imbalance = (max_load - min_load) / avg_load
                imbalance_measures.append(imbalance)
    
    if imbalance_measures:
        plt.plot(steps[:len(imbalance_measures)], imbalance_measures, marker='s', color='red')
        plt.xlabel('通信步骤')
        plt.ylabel('负载不平衡度')
        plt.title('负载不平衡度变化')
        plt.grid(True)
    
    # 子图4: 负载方差变化
    plt.subplot(2, 2, 4)
    variance_measures = []
    for step in steps:
        loads = list(load_history[step].values())
        if loads:
            variance = np.var(loads)
            variance_measures.append(variance)
    
    if variance_measures:
        plt.plot(steps[:len(variance_measures)], variance_measures, marker='^', color='green')
        plt.xlabel('通信步骤')
        plt.ylabel('负载方差')
        plt.title('负载方差变化')
        plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    try:
        plt.savefig('communication_optimization_analysis.png', dpi=300, bbox_inches='tight')
        print("图表已保存为: communication_optimization_analysis.png")
    except Exception as e:
        print(f"保存图表失败: {e}")
    
    plt.show()


def demo_load_monitor():
    """演示负载监控器功能"""
    print("\n=== 负载监控器演示 ===")
    
    # 创建负载监控器
    monitor = LoadMonitor()
    
    # 模拟负载变化
    print("模拟负载变化...")
    for i in range(10):
        load = monitor.get_current_load()
        stats = monitor.get_statistics()
        
        print(f"步骤 {i + 1}:")
        print(f"  当前负载: {load:.3f}")
        print(f"  CPU使用率: {stats.get('cpu_usage', 0):.1f}%")
        print(f"  内存使用率: {stats.get('memory_usage', 0):.1f}%")
        print(f"  GPU使用率: {stats.get('gpu_usage', 0):.1f}%")
        
        time.sleep(0.2)
    
    # 显示最终统计
    final_stats = monitor.get_statistics()
    print(f"\n最终统计:")
    print(f"  平均负载: {final_stats.get('average_load', 0):.3f}")
    print(f"  最大负载: {final_stats.get('max_load', 0):.3f}")
    print(f"  最小负载: {final_stats.get('min_load', 0):.3f}")
    print(f"  负载方差: {final_stats.get('load_variance', 0):.3f}")


def main():
    """主函数"""
    print("动态通信调度优化演示程序")
    print("=" * 50)
    
    try:
        # 演示负载监控器
        demo_load_monitor()
        
        # 模拟动态通信优化
        communication_results, load_history = simulate_dynamic_communication_optimization()
        
        # 分析通信效率
        analyze_communication_efficiency(communication_results)
        
        # 可视化结果
        visualize_communication_optimization(load_history)
        
        print("\n演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
