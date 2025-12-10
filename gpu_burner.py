#!/usr/bin/env python3
"""
GPU Burner - GPU利用率控制程序

该程序实时监控GPU利用率，并在利用率低于设定阈值时自动增加计算负载，
确保GPU利用率始终保持在指定水平以上，防止因利用率过低导致的资源回收。
"""

import argparse
import logging
import os
import signal
import sys
import time
from typing import List

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager
from gpu_monitor import GPUMonitor
from workload_generator import MultiGPUWorkloadManager


class GPUBurner:
    """GPU利用率控制器主类"""

    def __init__(self, config_file: str = "config.yaml"):
        """
        初始化GPU利用率控制器

        Args:
            config_file: 配置文件路径
        """
        # 加载配置
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager

        # 设置日志
        self._setup_logging()

        # 初始化组件
        self.gpu_monitor = None
        self.workload_manager = None

        # 控制标志
        self.is_running = False
        self.shutdown_requested = False

        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logging.info("GPU利用率控制器已初始化")

    def _setup_logging(self):
        """设置日志配置"""
        logging_config = self.config.get_logging_config()

        # 配置日志格式
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        log_level = getattr(logging, logging_config["level"].upper())

        # 配置根日志记录器
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(logging_config["file"], encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logging.info(f"接收到信号 {signum}，准备关闭...")
        self.shutdown_requested = True

    def initialize_components(self) -> bool:
        """
        初始化各个组件

        Returns:
            初始化是否成功
        """
        try:
            # 初始化GPU监控器
            self.gpu_monitor = GPUMonitor()

            # 验证目标GPU是否可用
            target_gpus = self.config.get_target_gpus()
            available_gpus = []
            for gpu_id in target_gpus:
                if self.gpu_monitor.is_gpu_available(gpu_id):
                    available_gpus.append(gpu_id)
                else:
                    logging.warning(f"GPU {gpu_id} 不可用，将被跳过")

            if not available_gpus:
                logging.error("没有可用的GPU设备")
                return False

            # 更新目标GPU列表为可用的GPU
            self.config.update_config("target_gpus", available_gpus)

            # 初始化工作负载管理器
            self.workload_manager = MultiGPUWorkloadManager(
                self.config.get_workload_config()
            )

            # 设置工作负载管理器的目标GPU
            self.workload_manager.target_gpus = available_gpus

            # 打印GPU信息
            self._print_gpu_info(available_gpus)

            return True

        except Exception as e:
            logging.error(f"初始化组件失败: {e}")
            return False

    def _print_gpu_info(self, gpu_ids: List[int]):
        """打印GPU信息"""
        logging.info("=== GPU设备信息 ===")
        for gpu_id in gpu_ids:
            info = self.gpu_monitor.get_gpu_info(gpu_id)
            if info:
                logging.info(f"GPU {gpu_id}: {info['name']}")
                logging.info(f"  总内存: {info['memory_total_mb']:.0f} MB")
                logging.info(f"  当前利用率: {info['gpu_utilization']}%")
                logging.info(f"  当前温度: {info['temperature']}°C")
        logging.info("===================")

    def calculate_required_workload(
        self, current_utilization: float, threshold: float
    ) -> float:
        """
        计算需要的工作负载强度

        Args:
            current_utilization: 当前GPU利用率
            threshold: 目标阈值

        Returns:
            需要的工作负载强度 (0.0 - 1.0)
        """
        if current_utilization >= threshold:
            return 0.0  # 不需要额外负载

        # 计算需要补充的利用率
        utilization_gap = threshold - current_utilization

        # 计算工作负载强度 (使用线性映射，可以根据需要调整)
        base_intensity = self.config.get("workload.base_intensity", 0.5)
        max_intensity = self.config.get("workload.max_intensity", 0.9)

        # 将利用率差距映射到工作负载强度
        # 假设最大工作负载强度可以提供最多50%的额外利用率
        max_utilization_contribution = 50.0
        intensity = (utilization_gap / max_utilization_contribution) * max_intensity

        # 限制在有效范围内
        intensity = max(0.0, min(max_intensity, intensity))

        # 确保最小工作负载强度
        if intensity > 0 and intensity < base_intensity:
            intensity = base_intensity

        return intensity

    def run(self):
        """运行主循环"""
        if not self.initialize_components():
            logging.error("初始化失败，退出程序")
            return

        self.is_running = True
        logging.info("GPU利用率控制循环已启动")

        target_gpus = self.config.get_target_gpus()
        threshold = self.config.get_utilization_threshold()
        check_interval = self.config.get_check_interval()

        try:
            while self.is_running and not self.shutdown_requested:
                # 获取当前平均利用率
                avg_utilization = self.gpu_monitor.get_average_utilization(target_gpus)

                if avg_utilization is None:
                    logging.warning("无法获取GPU利用率信息，跳过本次检查")
                    time.sleep(check_interval)
                    continue

                logging.info(
                    f"当前平均GPU利用率: {avg_utilization:.1f}%, 目标阈值: {threshold}%"
                )

                # 计算需要的工作负载强度
                required_intensity = self.calculate_required_workload(
                    avg_utilization, threshold
                )

                # 调整工作负载
                if required_intensity > 0:
                    logging.info(f"启动GPU工作负载，强度: {required_intensity:.2f}")
                    self.workload_manager.start_workloads(required_intensity)
                else:
                    logging.info("GPU利用率已达标，停止工作负载")
                    self.workload_manager.stop_workloads()

                # 打印详细状态
                self._print_status()

                # 等待下次检查
                time.sleep(check_interval)

        except KeyboardInterrupt:
            logging.info("接收到中断信号")
        except Exception as e:
            logging.error(f"运行过程中发生错误: {e}")
        finally:
            self.cleanup()

    def _print_status(self):
        """打印详细状态信息"""
        target_gpus = self.config.get_target_gpus()

        # 打印各个GPU的详细信息
        for gpu_id in target_gpus:
            info = self.gpu_monitor.get_gpu_info(gpu_id)
            if info:
                logging.debug(
                    f"GPU {gpu_id} 状态: 利用率={info['gpu_utilization']}%, "
                    f"内存使用={info['memory_used_mb']:.0f}MB, "
                    f"温度={info['temperature']}°C"
                )

        # 打印工作负载状态
        workload_status = self.workload_manager.get_all_status()
        for gpu_id, status in workload_status.items():
            if status["is_running"]:
                logging.debug(
                    f"GPU {gpu_id} 工作负载: 运行中, 强度={status['current_intensity']:.2f}"
                )
            else:
                logging.debug(f"GPU {gpu_id} 工作负载: 已停止")

    def cleanup(self):
        """清理资源"""
        logging.info("正在清理资源...")

        self.is_running = False

        if self.workload_manager:
            self.workload_manager.cleanup()

        if self.gpu_monitor:
            self.gpu_monitor.cleanup()

        logging.info("资源清理完成，程序退出")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GPU利用率控制程序")
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="配置文件路径 (默认: config.yaml)"
    )
    parser.add_argument(
        "--print-config", action="store_true", help="打印当前配置并退出"
    )
    parser.add_argument(
        "--test-gpu", action="store_true", help="测试GPU连接并显示设备信息"
    )

    args = parser.parse_args()

    try:
        config_manager = ConfigManager(args.config)

        if args.print_config:
            config_manager.print_config()
            return

        if args.test_gpu:
            print("测试GPU连接...")
            monitor = GPUMonitor()
            for gpu_id in range(monitor.device_count):
                info = monitor.get_gpu_info(gpu_id)
                if info:
                    print(f"GPU {gpu_id}: {info['name']}")
                    print("  状态: 可用")
                    print(f"  总内存: {info['memory_total_mb']:.0f} MB")
                    print(f"  当前利用率: {info['gpu_utilization']}%")
                    print(f"  当前温度: {info['temperature']}°C")
                else:
                    print(f"GPU {gpu_id}: 不可用")
            monitor.cleanup()
            return

        # 创建并运行GPU利用率控制器
        burner = GPUBurner(args.config)
        burner.run()

    except Exception as e:
        print(f"程序运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
