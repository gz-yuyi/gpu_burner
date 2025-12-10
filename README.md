# GPU Burner - GPU利用率控制程序

一个用于维持GPU利用率的Python程序，可以防止GPU服务器因利用率过低而被回收。

## 快速开始 - Docker部署

### 1. 拉取镜像

```bash
# 从阿里云容器镜像仓库拉取
docker pull crpi-lxfoqbwevmx9mc1q.cn-chengdu.personal.cr.aliyuncs.com/yuyi_tech/gpu_burner:latest
```

### 2. 运行容器

```bash
# 使用docker-compose运行
docker-compose up -d

# 或者直接使用docker命令运行
docker run -d \
  --name gpu-burner \
  --gpus all \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/logs:/app/logs \
  crpi-lxfoqbwevmx9mc1q.cn-chengdu.personal.cr.aliyuncs.com/yuyi_tech/gpu_burner:latest
```

### 3. Docker常用命令

```bash
# 查看容器日志
docker logs -f gpu-burner

# 进入容器内部
docker exec -it gpu-burner /bin/bash

# 停止并删除容器
docker stop gpu-burner && docker rm gpu-burner
```

## 功能特性

- **实时监控**: 实时监控指定GPU的利用率
- **动态调节**: 根据当前利用率动态调整计算负载
- **多GPU支持**: 支持同时控制多个GPU设备
- **灵活配置**: 通过配置文件设置各种参数
- **智能工作负载**: 使用矩阵运算产生真实的GPU负载

## 本地安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

编辑 `config.yaml` 文件来配置程序参数：

```yaml
# 目标GPU设备ID列表 (例如: [0, 1, 2] 表示前三块GPU)
target_gpus: [0]

# GPU利用率阈值 (百分比)
utilization_threshold: 30

# 检查间隔 (秒)
check_interval: 5

# 负载调整参数
workload:
  # 基础工作负载强度 (0.1-1.0)
  base_intensity: 0.5
  # 最大工作负载强度 (0.1-1.0)
  max_intensity: 0.9
  # 矩阵大小 (用于GPU计算)
  matrix_size: 2048
  # 每次计算的批次数量
  batch_size: 10

# 日志配置
logging:
  level: INFO
  file: gpu_burner.log
```

### 参数说明

- `target_gpus`: 要控制的GPU设备ID列表
- `utilization_threshold`: 目标GPU利用率阈值，程序会确保利用率不低于此值
- `check_interval`: 监控检查间隔时间（秒）
- `base_intensity`: 启动工作负载时的最小强度
- `max_intensity`: 最大工作负载强度
- `matrix_size`: GPU矩阵运算的矩阵大小，影响计算强度
- `batch_size`: 每次检查执行的批次数量

## 使用方法

### 基本使用

```bash
python gpu_burner.py
```

### 指定配置文件

```bash
python gpu_burner.py -c my_config.yaml
```

### 打印当前配置

```bash
python gpu_burner.py --print-config
```

### 测试GPU连接

```bash
python gpu_burner.py --test-gpu
```

## 工作原理

1. **监控阶段**: 程序定期检查指定GPU的当前利用率
2. **比较阶段**: 将当前利用率与设定的阈值进行比较
3. **调节阶段**:
   - 如果利用率 ≥ 阈值：停止或减少工作负载
   - 如果利用率 < 阈值：启动或增加工作负载
4. **负载生成**: 使用PyTorch在GPU上执行矩阵运算来产生真实的计算负载

## 日志输出

程序会同时输出到控制台和日志文件 `gpu_burner.log`，包含：
- GPU设备信息
- 当前利用率状态
- 工作负载调整决策
- 详细的运行状态

## 依赖要求

- Python 3.7+
- NVIDIA GPU 和 驱动
- NVIDIA ML库 (nvidia-ml-py3)
- PyTorch (可选，用于GPU计算)
- NumPy (CPU计算后备方案)
- Docker (可选，用于容器化部署)
- NVIDIA Container Toolkit (Docker GPU支持)

## Docker部署详细说明

### 1. 安装NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. 使用预构建镜像

```bash
# 直接拉取并运行
docker run -d \
  --name gpu-burner \
  --gpus all \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/logs:/app/logs \
  crpi-lxfoqbwevmx9mc1q.cn-chengdu.personal.cr.aliyuncs.com/yuyi_tech/gpu_burner:latest
```

### 3. 本地构建（可选）

```bash
# 克隆代码
git clone <your-repo-url>
cd gpu_burner

# 构建镜像
docker build -t gpu-burner:local .

# 运行本地构建的镜像
docker run -d \
  --name gpu-burner-local \
  --gpus all \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  -v $(pwd)/logs:/app/logs \
  gpu-burner:local
```

## 注意事项

1. 确保GPU驱动正确安装
2. 如果没有安装PyTorch，程序会使用NumPy进行CPU计算作为后备方案
3. 工作负载强度设置建议从低值开始，逐步调整到合适的水平
4. 监控GPU温度，确保不会过热

## 示例输出

```
2023-12-10 10:00:00 - INFO - 已检测到 2 块GPU设备
2023-12-10 10:00:00 - INFO - GPU 0: NVIDIA GeForce RTX 3080
2023-12-10 10:00:00 - INFO -   总内存: 10240 MB
2023-12-10 10:00:00 - INFO -   当前利用率: 15%
2023-12-10 10:00:00 - INFO -   当前温度: 45°C
2023-12-10 10:00:00 - INFO - GPU利用率控制循环已启动
2023-12-10 10:00:00 - INFO - 当前平均GPU利用率: 15.0%, 目标阈值: 30%
2023-12-10 10:00:00 - INFO - 启动GPU工作负载，强度: 0.60
```

## 故障排除

### 常见问题

1. **无法检测到GPU**: 检查NVIDIA驱动是否正确安装
2. **权限错误**: 确保有权限访问GPU设备
3. **内存不足**: 调小矩阵大小参数
4. **CPU占用过高**: 检查是否缺少PyTorch，使用了CPU后备方案

### 停止程序

使用 `Ctrl+C` 可以安全地停止程序，程序会自动清理所有工作负载。

```bash
# Docker环境停止
docker-compose down
# 或
docker stop gpu-burner && docker rm gpu-burner
```