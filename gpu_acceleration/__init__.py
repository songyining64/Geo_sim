"""Optional GPU and learned-solver extensions.

This package must remain importable on CPU-only installations.  Individual
accelerators are deliberately optional: importing :mod:`gpu_acceleration`
never makes a simulation fail merely because CUDA, PyTorch Geometric, or a
research-only demo dependency is absent.
"""

__version__ = "2.0.0"
__author__ = "GeoSim Team"
__email__ = "geosim@example.com"

from .cuda_acceleration import CUDAAccelerator, GPUMatrixOperations, HAS_CUPY, HAS_NUMBA

HAS_NEURAL_AMG = False
HAS_META_AMG = False

try:
    from .neural_amg import NeuralAMG, NeuralAMGConfig, NeuralAMGSolver
    HAS_NEURAL_AMG = True
except (ImportError, AttributeError):
    # Neural AMG remains available through a direct module import once its
    # optional ML dependencies have been installed.
    pass

try:
    from .meta_amg import MetaAMG, MetaAMGConfig, MetaAMGSolver
    HAS_META_AMG = True
except (ImportError, AttributeError):
    pass


def check_gpu_availability():
    """Return CUDA availability without requiring PyTorch at install time."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0),
            }
        return {'available': False, 'reason': 'CUDA not available'}
    except ImportError:
        return {'available': False, 'reason': 'PyTorch not installed'}


def check_cuda_availability():
    """Backward-compatible alias for :func:`check_gpu_availability`."""
    return check_gpu_availability()
