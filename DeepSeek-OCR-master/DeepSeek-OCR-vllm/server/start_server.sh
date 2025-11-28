#!/bin/bash
# DeepSeek-OCR Server Startup Script
# Optimal configuration: 0.9 GPU memory utilization

# Server configuration
MODEL_PATH="deepseek-ai/DeepSeek-OCR"
PORT=8010
GPU_DEVICES=0
GPU_MEMORY_UTILIZATION=0.9  # Optimal balance for OCRFlow training
MAX_MODEL_LEN=8192
SESSION_NAME="ocr_server_v1"
LOG_FILE="/tmp/ocr_server_v1.log"

# Kill existing session if it exists
tmux kill-session -t ${SESSION_NAME} 2>/dev/null && echo "✓ Stopped existing server session"

# Start server in tmux session
echo "Starting DeepSeek-OCR server with:"
echo "  - GPU: ${GPU_DEVICES}"
echo "  - Memory Utilization: ${GPU_MEMORY_UTILIZATION} (90%)"
echo "  - Port: ${PORT}"
echo "  - Log: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} tmux new-session -d -s ${SESSION_NAME} \
  "cd /share/project/xiyan/sources/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm && \
   python server/deepseek_ocr_server.py \
   --model-path ${MODEL_PATH} \
   --port ${PORT} \
   --gpu-devices ${GPU_DEVICES} \
   --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
   --max-model-len ${MAX_MODEL_LEN} \
   2>&1 | tee ${LOG_FILE}"

echo "✓ Server starting in tmux session: ${SESSION_NAME}"
echo ""
echo "Useful commands:"
echo "  - View logs:      tail -f ${LOG_FILE}"
echo "  - Attach to tmux: tmux attach -t ${SESSION_NAME}"
echo "  - Check health:   curl http://localhost:${PORT}/health"
echo "  - Stop server:    tmux kill-session -t ${SESSION_NAME}"
echo ""
echo "Waiting for server to be ready..."
sleep 60

# Check if server is healthy
if curl -s http://localhost:${PORT}/health | grep -q "healthy"; then
    echo "✅ Server is healthy and ready!"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | grep "^${GPU_DEVICES},"
else
    echo "⚠️  Server may still be initializing. Check logs: tail -f ${LOG_FILE}"
fi
