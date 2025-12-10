import logging
import threading
from typing import Dict

import numpy as np


class WorkloadGenerator:
    """GPU工作负载生成器，用于在GPU上执行计算任务以提高利用率"""

    def __init__(self, gpu_id: int, config: Dict):
        """
        初始化工作负载生成器

        Args:
            gpu_id: GPU设备ID
            config: 配置字典
        """
        self.gpu_id = gpu_id
        self.config = config
        self.is_running = False
        self.current_intensity = 0.0
        self.workload_thread = None
        self.stop_event = threading.Event()

        logging.info(f"为GPU {gpu_id} 初始化工作负载生成器")

    def start_workload(self, intensity: float):
        """
        启动GPU工作负载

        Args:
            intensity: 工作负载强度 (0.0 - 1.0)
        """
        if self.is_running:
            self.stop_workload()

        if not 0.0 <= intensity <= 1.0:
            logging.warning(f"工作负载强度 {intensity} 超出范围 [0.0, 1.0]，将被限制")
            intensity = max(0.0, min(1.0, intensity))

        self.current_intensity = intensity
        self.is_running = True
        self.stop_event.clear()

        self.workload_thread = threading.Thread(
            target=self._workload_worker, args=(intensity,), daemon=True
        )
        self.workload_thread.start()

        logging.info(f"GPU {self.gpu_id} 工作负载已启动，强度: {intensity:.2f}")

    def stop_workload(self):
        """停止GPU工作负载"""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()

        if self.workload_thread and self.workload_thread.is_alive():
            self.workload_thread.join(timeout=5)

        logging.info(f"GPU {self.gpu_id} 工作负载已停止")

    def adjust_intensity(self, new_intensity: float):
        """
        调整工作负载强度

        Args:
            new_intensity: 新的工作负载强度
        """
        if self.is_running:
            self.current_intensity = new_intensity
            logging.info(f"GPU {self.gpu_id} 工作负载强度已调整为: {new_intensity:.2f}")
        else:
            self.start_workload(new_intensity)

    def _workload_worker(self, intensity: float):
        """
        工作负载工作线程

        Args:
            intensity: 工作负载强度
        """
        try:
            # 设置CUDA设备
            import torch

            if torch.cuda.is_available():
                device = torch.device(f"cuda:{self.gpu_id}")
                torch.cuda.set_device(device)
            else:
                logging.error(f"GPU {self.gpu_id} 不可用，PyTorch无法检测到CUDA设备")
                return

        except ImportError:
            logging.warning("PyTorch未安装，将使用NumPy进行CPU计算")
            device = None

        matrix_size = self.config.get("matrix_size", 2048)
        batch_size = self.config.get("batch_size", 10)

        while self.is_running and not self.stop_event.is_set():
            try:
                if device is not None:
                    # 使用GPU进行矩阵计算
                    self._gpu_matrix_operations(
                        device, matrix_size, batch_size, intensity
                    )
                else:
                    # 使用CPU进行计算（作为后备方案）
                    self._cpu_matrix_operations(matrix_size, batch_size, intensity)

                # 根据强度调整休眠时间
                sleep_time = (1.0 - intensity) * 0.1  # 强度越高，休眠时间越短
                if sleep_time > 0:
                    self.stop_event.wait(sleep_time)

            except Exception as e:
                logging.error(f"GPU {self.gpu_id} 工作负载执行出错: {e}")
                break

    def _gpu_matrix_operations(
        self, device, matrix_size: int, batch_size: int, intensity: float
    ):
        """
        在GPU上执行矩阵运算

        Args:
            device: CUDA设备
            matrix_size: 矩阵大小
            batch_size: 批次大小
            intensity: 工作负载强度
        """
        import torch

        # 根据强度调整批次大小
        adjusted_batch_size = max(1, int(batch_size * intensity))

        for _ in range(adjusted_batch_size):
            if self.stop_event.is_set():
                break

            try:
                # 创建随机矩阵并执行矩阵乘法
                a = torch.randn(
                    matrix_size, matrix_size, device=device, dtype=torch.float32
                )
                b = torch.randn(
                    matrix_size, matrix_size, device=device, dtype=torch.float32
                )

                # 执行矩阵乘法
                c = torch.matmul(a, b)

                # 执行一些额外的计算来增加负载
                d = torch.sum(c)
                e = torch.mean(c)
                f = torch.std(c)

                # 确保计算完成
                torch.cuda.synchronize()

            except Exception as e:
                logging.warning(f"GPU {self.gpu_id} 矩阵运算出错: {e}")

    def _cpu_matrix_operations(
        self, matrix_size: int, batch_size: int, intensity: float
    ):
        """
        在CPU上执行矩阵运算（后备方案）

        Args:
            matrix_size: 矩阵大小
            batch_size: 批次大小
            intensity: 工作负载强度
        """
        # 根据强度调整批次大小
        adjusted_batch_size = max(1, int(batch_size * intensity))

        for _ in range(adjusted_batch_size):
            if self.stop_event.is_set():
                break

            try:
                # 使用NumPy进行矩阵计算
                a = np.random.randn(matrix_size, matrix_size).astype(np.float32)
                b = np.random.randn(matrix_size, matrix_size).astype(np.float32)

                # 执行矩阵乘法
                c = np.dot(a, b)

                # 执行一些额外的计算
                d = np.sum(c)
                e = np.mean(c)
                f = np.std(c)

            except Exception as e:
                logging.warning(f"CPU矩阵运算出错: {e}")

    def get_status(self) -> Dict:
        """
        获取当前工作负载状态

        Returns:
            包含状态信息的字典
        """
        return {
            "gpu_id": self.gpu_id,
            "is_running": self.is_running,
            "current_intensity": self.current_intensity,
            "thread_alive": self.workload_thread.is_alive()
            if self.workload_thread
            else False,
        }

    def cleanup(self):
        """清理资源"""
        self.stop_workload()


class MultiGPUWorkloadManager:
    """多GPU工作负载管理器"""

    def __init__(self, config: Dict):
        """
        初始化多GPU工作负载管理器

        Args:
            config: 配置字典
        """
        self.config = config
        self.target_gpus = config.get("target_gpus", [0])
        self.workload_generators: Dict[int, WorkloadGenerator] = {}

        # 为每个目标GPU创建工作负载生成器
        for gpu_id in self.target_gpus:
            self.workload_generators[gpu_id] = WorkloadGenerator(gpu_id, config)

        logging.info(f"多GPU工作负载管理器已初始化，目标GPU: {self.target_gpus}")

    def start_workloads(self, intensity: float):
        """
        启动所有目标GPU的工作负载

        Args:
            intensity: 工作负载强度
        """
        for gpu_id, generator in self.workload_generators.items():
            try:
                generator.start_workload(intensity)
            except Exception as e:
                logging.error(f"启动GPU {gpu_id} 工作负载失败: {e}")

    def stop_workloads(self):
        """停止所有GPU的工作负载"""
        for gpu_id, generator in self.workload_generators.items():
            try:
                generator.stop_workload()
            except Exception as e:
                logging.error(f"停止GPU {gpu_id} 工作负载失败: {e}")

    def adjust_intensities(self, intensities: Dict[int, float]):
        """
        调整各个GPU的工作负载强度

        Args:
            intensities: 字典，键为GPU ID，值为新的强度
        """
        for gpu_id, intensity in intensities.items():
            if gpu_id in self.workload_generators:
                try:
                    self.workload_generators[gpu_id].adjust_intensity(intensity)
                except Exception as e:
                    logging.error(f"调整GPU {gpu_id} 工作负载强度失败: {e}")

    def get_all_status(self) -> Dict[int, Dict]:
        """
        获取所有GPU工作负载的状态

        Returns:
            字典，键为GPU ID，值为状态信息
        """
        status = {}
        for gpu_id, generator in self.workload_generators.items():
            status[gpu_id] = generator.get_status()
        return status

    def cleanup(self):
        """清理所有资源"""
        self.stop_workloads()
        for generator in self.workload_generators.values():
            generator.cleanup()

