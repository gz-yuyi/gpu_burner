# GPU Burner - Optimized Docker Image
# Based on NVIDIA CUDA runtime image with minimal footprint

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install only essential system dependencies in a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        && \
    # Clean up apt caches
    apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create symlink for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Create app directory and user
RUN groupadd -r gpu_burner && \
    useradd -r -g gpu_burner -d /app -s /bin/bash gpu_burner && \
    mkdir -p /app && \
    chown gpu_burner:gpu_burner /app

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CUDA version)
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY --chown=gpu_burner:gpu_burner . .

# Switch to non-root user
USER gpu_burner

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pynvml; pynvml.nvmlInit(); print('GPU OK')" || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "gpu_burner.py"]