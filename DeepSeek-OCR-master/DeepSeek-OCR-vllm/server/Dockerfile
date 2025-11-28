FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements_server.txt /app/
COPY ../requirements.txt /app/requirements_base.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# Install vLLM from wheel or nightly
RUN pip3 install --no-cache-dir "vllm>=0.11.1" --extra-index-url https://wheels.vllm.ai/nightly

# Install base requirements
RUN pip3 install --no-cache-dir -r /app/requirements_base.txt

# Install server requirements
RUN pip3 install --no-cache-dir -r /app/requirements_server.txt

# Install flash-attention
RUN pip3 install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# Copy application code
COPY . /app/

# Create directories for logs and outputs
RUN mkdir -p /app/logs /app/outputs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python3", "deepseek_ocr_server.py", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--gpu-memory-utilization", "0.9", \
     "--max-model-len", "8192"]
