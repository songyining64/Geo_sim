"""
GPU加速模块 - 完整版

提供全面的GPU加速功能，包括：
1. CUDA基础加速
2. 机器学习集成
3. 地质模拟优化
4. 强化学习支持
5. 图神经网络
6. 元学习
7. 并行计算优化
"""

# 基础GPU加速
from .cuda_acceleration import (
    CUDAAccelerator,
    GPUMatrixOperations,
    HAS_CUPY,
    HAS_NUMBA
)

# 高级机器学习框架
from .advanced_ml import (
    AdvancedMLFramework,
    NeuralNetwork,
    TrainingManager,
    ModelOptimizer
)

# 地质机器学习框架
from .geological_ml_framework import (
    GeologicalMLFramework,
    GeologicalSurrogateModel,
    PhysicsInformedNN,
    MultiFidelityModel
)

# 物理集成机器学习
from .physics_integrated_ml import (
    PhysicsIntegratedML,
    PINNModel,
    PhysicsLoss,
    HybridSolver
)

# 地质示例（如果可用）
try:
    from .geological_examples import (
        GeologicalExamples,
        ReservoirSimulation,
        SeismicInversion,
        Geomechanics
    )
    HAS_GEOLOGICAL_EXAMPLES = True
except ImportError:
    HAS_GEOLOGICAL_EXAMPLES = False

# 机器学习优化
from .ml_optimization import (
    MLOptimizer,
    HyperparameterOptimizer,
    ArchitectureSearch,
    TrainingOptimizer
)

# 并行计算
from .parallel_computing import (
    GPUParallelComputing,
    DistributedTraining,
    MultiGPUManager
)

# 强化学习（如果可用）
try:
    from .rl_optimization_demo import (
        RLOptimizer,
        Environment,
        Agent,
        PolicyNetwork
    )
    HAS_RL_OPTIMIZATION = True
except ImportError:
    HAS_RL_OPTIMIZATION = False

# 强化学习+图神经网络（如果可用）
try:
    from .rl_gnn_demo import (
        RLGNNDemo,
        GNNEnvironment,
        GNNPolicy
    )
    HAS_RL_GNN = True
except ImportError:
    HAS_RL_GNN = False

# 图神经网络
from .geodynamics_gnn import (
    GeodynamicsGNN,
    GNNModel,
    GraphProcessor,
    PhysicsGNN
)

# 图神经网络+物理信息网络（如果可用）
try:
    from .gnn_pinn_integration_demo import (
        GNNPINNIntegration,
        HybridGNNPINN,
        PhysicsGNN as PGNN
    )
    HAS_GNN_PINN = True
except ImportError:
    HAS_GNN_PINN = False

# 元学习（如果可用）
try:
    from .meta_learning_demo import (
        MetaLearningDemo,
        MetaLearner,
        TaskGenerator,
        FewShotLearning
    )
    HAS_META_LEARNING = True
except ImportError:
    HAS_META_LEARNING = False

# 自适应约束
from .adaptive_constraints_demo import (
    AdaptiveConstraintsDemo,
    ConstraintOptimizer,
    DynamicConstraints
)

# 测试模块（如果可用）
try:
    from .test_geological_optimizations import (
        test_geological_optimizations,
        run_optimization_tests
    )
    HAS_GEO_OPT_TESTS = True
except ImportError:
    HAS_GEO_OPT_TESTS = False

try:
    from .test_surrogate_extensions import (
        test_surrogate_extensions,
        run_surrogate_tests
    )
    HAS_SURROGATE_TESTS = True
except ImportError:
    HAS_SURROGATE_TESTS = False

# 版本信息
__version__ = "2.0.0"
__author__ = "GeoSim Team"
__email__ = "geosim@example.com"

# 模块级配置
DEFAULT_GPU_MEMORY_FRACTION = 0.8
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001

# 可用性检查
def check_gpu_availability():
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0)
            }
        else:
            return {'available': False, 'reason': 'CUDA not available'}
    except ImportError:
        return {'available': False, 'reason': 'PyTorch not installed'}

def check_cuda_availability():
    """检查CUDA可用性"""
    try:
        import cupy as cp
        return {
            'available': True,
            'version': cp.cuda.runtime.runtimeGetVersion(),
            'device_count': cp.cuda.runtime.getDeviceCount()
        }
    except ImportError:
        return {'available': False, 'reason': 'CuPy not installed'}

def check_numba_availability():
    """检查Numba可用性"""
    try:
        import numba
        return {
            'available': True,
            'version': numba.__version__,
            'cuda_available': numba.cuda.is_available()
        }
    except ImportError:
        return {'available': False, 'reason': 'Numba not installed'}

# 快速创建函数
def create_gpu_accelerator(device_id: int = 0):
    """快速创建GPU加速器"""
    return CUDAAccelerator(device_id)

def create_ml_framework(gpu_enabled: bool = True):
    """快速创建机器学习框架"""
    return AdvancedMLFramework(gpu_enabled=gpu_enabled)

def create_geological_framework():
    """快速创建地质机器学习框架"""
    return GeologicalMLFramework()

def create_physics_integrated_ml():
    """快速创建物理集成机器学习"""
    return PhysicsIntegratedML()

# 性能监控
class GPUPerformanceMonitor:
    """GPU性能监控器"""
    
    def __init__(self):
        self.metrics = {}
    
    def get_gpu_utilization(self):
        """获取GPU利用率"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return [{
                'id': gpu.id,
                'load': gpu.load,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            } for gpu in gpus]
        except ImportError:
            return []
    
    def get_memory_info(self):
        """获取GPU内存信息"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'allocated': torch.cuda.memory_allocated(),
                    'cached': torch.cuda.memory_cached(),
                    'max_allocated': torch.cuda.max_memory_allocated()
                }
            else:
                return {}
        except ImportError:
            return {}

# 工具函数
def clear_gpu_memory():
    """清理GPU内存"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
        return False
    except ImportError:
        return False

def set_gpu_memory_fraction(fraction: float):
    """设置GPU内存使用比例"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction)
            return True
        return False
    except ImportError:
        return False

# 模块初始化时的检查
def _initialize_module():
    """模块初始化"""
    print("🚀 初始化GPU加速模块...")
    
    # 检查GPU可用性
    gpu_status = check_gpu_availability()
    if gpu_status['available']:
        print(f"✅ GPU可用: {gpu_status['device_name']}")
        print(f"   设备数量: {gpu_status['device_count']}")
    else:
        print(f"⚠️ GPU不可用: {gpu_status['reason']}")
    
    # 检查CUDA可用性
    cuda_status = check_cuda_availability()
    if cuda_status['available']:
        print(f"✅ CuPy可用: CUDA {cuda_status['version']}")
    else:
        print(f"⚠️ CuPy不可用: {cuda_status['reason']}")
    
    # 检查Numba可用性
    numba_status = check_numba_availability()
    if numba_status['available']:
        print(f"✅ Numba可用: v{numba_status['version']}")
        if numba_status['cuda_available']:
            print("   CUDA支持: 可用")
        else:
            print("   CUDA支持: 不可用")
    else:
        print(f"⚠️ Numba不可用: {numba_status['reason']}")
    
    print("🎯 GPU加速模块初始化完成！")

# 自动初始化
_initialize_module()

# 导出所有主要类和函数
__all__ = [
    # 基础GPU加速
    'CUDAAccelerator',
    'GPUMatrixOperations',
    'HAS_CUPY',
    'HAS_NUMBA',
    
    # 高级机器学习
    'AdvancedMLFramework',
    'NeuralNetwork',
    'TrainingManager',
    'ModelOptimizer',
    
    # 地质机器学习
    'GeologicalMLFramework',
    'GeologicalSurrogateModel',
    'PhysicsInformedNN',
    'MultiFidelityModel',
    
    # 物理集成机器学习
    'PhysicsIntegratedML',
    'PINNModel',
    'PhysicsLoss',
    'HybridSolver',
    
    # 地质示例
    'GeologicalExamples',
    'ReservoirSimulation',
    'SeismicInversion',
    'Geomechanics',
    
    # 机器学习优化
    'MLOptimizer',
    'HyperparameterOptimizer',
    'ArchitectureSearch',
    'TrainingOptimizer',
    
    # 并行计算
    'GPUParallelComputing',
    'DistributedTraining',
    'MultiGPUManager',
    
    # 强化学习
    'RLOptimizer',
    'Environment',
    'Agent',
    'PolicyNetwork',
    
    # 强化学习+图神经网络
    'RLGNNDemo',
    'GNNEnvironment',
    'GNNPolicy',
    
    # 图神经网络
    'GeodynamicsGNN',
    'GNNModel',
    'GraphProcessor',
    'PhysicsGNN',
    
    # 图神经网络+物理信息网络
    'GNNPINNIntegration',
    'HybridGNNPINN',
    
    # 元学习
    'MetaLearningDemo',
    'MetaLearner',
    'TaskGenerator',
    'FewShotLearning',
    
    # 自适应约束
    'AdaptiveConstraintsDemo',
    'ConstraintOptimizer',
    'DynamicConstraints',
    
    # 工具函数
    'check_gpu_availability',
    'check_cuda_availability',
    'check_numba_availability',
    'create_gpu_accelerator',
    'create_ml_framework',
    'create_geological_framework',
    'create_physics_integrated_ml',
    'clear_gpu_memory',
    'set_gpu_memory_fraction',
    
    # 性能监控
    'GPUPerformanceMonitor',
    
    # 配置常量
    'DEFAULT_GPU_MEMORY_FRACTION',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_LEARNING_RATE',
    
    # 版本信息
    '__version__',
    '__author__',
    '__email__'
]
