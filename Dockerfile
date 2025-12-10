# GPU Burner Docker镜像
# 基于NVIDIA官方CUDA镜像构建，支持GPU计算

# 使用多阶段构建优化镜像大小
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS builder

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
ENV PYTHONDONTWRITEBYTECODE=1

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    # Python相关
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    # 系统工具
    curl \
    wget \
    git \
    vim \
    # 时区设置
    tzdata \
    # 清理缓存
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 创建软链接
RUN ln -s /usr/bin/python3 /usr/bin/python

# 升级pip
RUN python -m pip install --upgrade pip setuptools wheel

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装PyTorch (CUDA版本)
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 生产阶段
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

# 创建非root用户
RUN groupadd -r gpu_burner && useradd -r -g gpu_burner gpu_burner

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 创建软链接
RUN ln -s /usr/bin/python3 /usr/bin/python

# 创建工作目录
WORKDIR /app

# 从builder阶段复制Python包
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY . .

# 创建日志目录
RUN mkdir -p /app/logs && chown -R gpu_burner:gpu_burner /app

# 切换到非root用户
USER gpu_burner

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pynvml; pynvml.nvmlInit(); print('GPU OK')" || exit 1

# 暴露端口（如果有Web界面）
EXPOSE 8080

# 设置默认命令
CMD ["python", "gpu_burner.py"]