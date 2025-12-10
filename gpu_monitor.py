import logging
from typing import Dict, List, Optional

import pynvml


class GPUMonitor:
    """GPU监控类，负责获取GPU利用率和状态信息"""

    def __init__(self):
        """初始化GPU监控器"""
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            logging.info(f"已检测到 {self.device_count} 块GPU设备")
        except Exception as e:
            logging.error(f"初始化NVIDIA ML库失败: {e}")
            raise

    def get_gpu_utilization(self, gpu_id: int) -> Optional[float]:
        """
        获取指定GPU的利用率百分比

        Args:
            gpu_id: GPU设备ID

        Returns:
            GPU利用率百分比，如果获取失败返回None
        """
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except Exception as e:
            logging.error(f"获取GPU {gpu_id} 利用率失败: {e}")
            return None

    def get_gpu_info(self, gpu_id: int) -> Optional[Dict]:
        """
        获取指定GPU的详细信息

        Args:
            gpu_id: GPU设备ID

        Returns:
            包含GPU信息的字典，如果获取失败返回None
        """
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

            # 获取基本信息
            name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )

            # 获取内存使用情况
            memory_used_mb = memory_info.used / (1024 * 1024)
            memory_total_mb = memory_info.total / (1024 * 1024)
            memory_util_percent = (memory_info.used / memory_info.total) * 100

            return {
                "id": gpu_id,
                "name": name,
                "gpu_utilization": util.gpu,
                "memory_utilization": util.memory,
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb,
                "memory_util_percent": memory_util_percent,
                "temperature": temperature,
            }
        except Exception as e:
            logging.error(f"获取GPU {gpu_id} 详细信息失败: {e}")
            return None

    def get_all_gpus_utilization(self) -> Dict[int, float]:
        """
        获取所有GPU的利用率

        Returns:
            字典，键为GPU ID，值为利用率百分比
        """
        utilizations = {}
        for gpu_id in range(self.device_count):
            util = self.get_gpu_utilization(gpu_id)
            if util is not None:
                utilizations[gpu_id] = util
        return utilizations

    def is_gpu_available(self, gpu_id: int) -> bool:
        """
        检查指定的GPU是否可用

        Args:
            gpu_id: GPU设备ID

        Returns:
            GPU是否可用
        """
        if gpu_id >= self.device_count:
            return False

        try:
            pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            return True
        except Exception:
            return False

    def get_average_utilization(self, gpu_ids: List[int]) -> Optional[float]:
        """
        获取指定GPU列表的平均利用率

        Args:
            gpu_ids: GPU ID列表

        Returns:
            平均利用率百分比，如果无法获取任何数据返回None
        """
        valid_utils = []
        for gpu_id in gpu_ids:
            if self.is_gpu_available(gpu_id):
                util = self.get_gpu_utilization(gpu_id)
                if util is not None:
                    valid_utils.append(util)

        if not valid_utils:
            return None

        return sum(valid_utils) / len(valid_utils)

    def cleanup(self):
        """清理资源"""
        try:
            pynvml.nvmlShutdown()
        except Exception as e:
            logging.error(f"关闭NVIDIA ML库失败: {e}")

