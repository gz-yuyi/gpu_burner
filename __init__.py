"""
GPU Burner - GPU利用率控制程序

该包提供了完整的GPU利用率监控和控制功能。
"""

__version__ = "1.0.0"
__author__ = "GPU Burner Team"
__description__ = "GPU利用率控制程序，防止GPU资源因利用率过低被回收"

from .config_manager import ConfigManager
from .gpu_monitor import GPUMonitor
from .workload_generator import WorkloadGenerator, MultiGPUWorkloadManager
from .gpu_burner import GPUBurner

__all__ = [
    'ConfigManager',
    'GPUMonitor',
    'WorkloadGenerator',
    'MultiGPUWorkloadManager',
    'GPUBurner'
]