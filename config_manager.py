import logging
import os
from typing import Any, Dict, List

import yaml


class ConfigManager:
    """配置管理器，负责加载和管理程序配置"""

    def __init__(self, config_file: str = "config.yaml"):
        """
        初始化配置管理器

        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_default_config()

        # 如果配置文件存在，则加载并合并配置
        if os.path.exists(config_file):
            self._load_config()
            logging.info(f"已加载配置文件: {config_file}")
        else:
            self._create_default_config()
            logging.info(f"已创建默认配置文件: {config_file}")

        self._validate_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            "target_gpus": [0],
            "utilization_threshold": 30,
            "check_interval": 5,
            "workload": {
                "base_intensity": 0.5,
                "max_intensity": 0.9,
                "matrix_size": 2048,
                "batch_size": 10,
            },
            "logging": {"level": "INFO", "file": "gpu_burner.log"},
        }

    def _load_config(self):
        """从文件加载配置"""
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._merge_config(file_config)
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
            logging.info("使用默认配置")

    def _merge_config(self, file_config: Dict[str, Any]):
        """合并文件配置到默认配置"""

        def merge_dict(default: Dict, override: Dict) -> Dict:
            result = default.copy()
            for key, value in override.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result

        self.config = merge_dict(self.config, file_config)

    def _create_default_config(self):
        """创建默认配置文件"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )
        except Exception as e:
            logging.error(f"创建默认配置文件失败: {e}")

    def _validate_config(self):
        """验证配置的有效性"""
        errors = []

        # 验证target_gpus
        if (
            not isinstance(self.config["target_gpus"], list)
            or not self.config["target_gpus"]
        ):
            errors.append("target_gpus必须是非空的列表")
        elif not all(
            isinstance(gpu_id, int) and gpu_id >= 0
            for gpu_id in self.config["target_gpus"]
        ):
            errors.append("target_gpus中的所有ID必须是非负整数")

        # 验证utilization_threshold
        threshold = self.config["utilization_threshold"]
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 100:
            errors.append("utilization_threshold必须是0-100之间的数值")

        # 验证check_interval
        interval = self.config["check_interval"]
        if not isinstance(interval, (int, float)) or interval <= 0:
            errors.append("check_interval必须是正数")

        # 验证workload配置
        workload = self.config["workload"]
        if (
            not isinstance(workload.get("base_intensity"), (int, float))
            or not 0 <= workload["base_intensity"] <= 1
        ):
            errors.append("workload.base_intensity必须是0-1之间的数值")

        if (
            not isinstance(workload.get("max_intensity"), (int, float))
            or not 0 <= workload["max_intensity"] <= 1
        ):
            errors.append("workload.max_intensity必须是0-1之间的数值")

        if workload["base_intensity"] > workload["max_intensity"]:
            errors.append("workload.base_intensity不能大于workload.max_intensity")

        if (
            not isinstance(workload.get("matrix_size"), int)
            or workload["matrix_size"] <= 0
        ):
            errors.append("workload.matrix_size必须是正整数")

        if (
            not isinstance(workload.get("batch_size"), int)
            or workload["batch_size"] <= 0
        ):
            errors.append("workload.batch_size必须是正整数")

        # 验证logging配置
        logging_config = self.config["logging"]
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if logging_config.get("level") not in valid_log_levels:
            errors.append(f"logging.level必须是以下值之一: {valid_log_levels}")

        if errors:
            error_msg = "配置验证失败:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ValueError(error_msg)

        logging.info("配置验证通过")

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点分隔的嵌套键

        Args:
            key: 配置键，支持 'a.b.c' 格式
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_target_gpus(self) -> List[int]:
        """获取目标GPU列表"""
        return self.config["target_gpus"]

    def get_utilization_threshold(self) -> float:
        """获取利用率阈值"""
        return float(self.config["utilization_threshold"])

    def get_check_interval(self) -> float:
        """获取检查间隔"""
        return float(self.config["check_interval"])

    def get_workload_config(self) -> Dict[str, Any]:
        """获取工作负载配置"""
        return self.config["workload"]

    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.config["logging"]

    def update_config(self, key: str, value: Any):
        """
        更新配置值

        Args:
            key: 配置键，支持 'a.b.c' 格式
            value: 新值
        """
        keys = key.split(".")
        config = self.config

        # 导航到目标位置
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # 设置值
        config[keys[-1]] = value
        logging.info(f"配置已更新: {key} = {value}")

    def save_config(self):
        """保存当前配置到文件"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )
            logging.info(f"配置已保存到: {self.config_file}")
        except Exception as e:
            logging.error(f"保存配置文件失败: {e}")

    def reload_config(self):
        """重新加载配置文件"""
        if os.path.exists(self.config_file):
            self._load_config()
            self._validate_config()
            logging.info("配置已重新加载")
        else:
            logging.warning("配置文件不存在，无法重新加载")

    def print_config(self):
        """打印当前配置"""
        print("当前配置:")
        print(
            yaml.dump(
                self.config, default_flow_style=False, allow_unicode=True, indent=2
            )
        )

