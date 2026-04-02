#!/bin/bash
# Qwen3-VL Thinking Mode Visualization Server Startup Script

# Default values
LORA_PATH="${LORA_PATH:-}"
MODEL_PATH="${MODEL_PATH:-Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking}"
PORT="${PORT:-8501}"
HOST="${HOST:-0.0.0.0}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lora-path)
            LORA_PATH="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --lora-path PATH      Path to LoRA adapter (default: none)"
            echo "  --model-path PATH     Path to base model (default: Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking)"
            echo "  --port PORT          Port to run server on (default: 8501)"
            echo "  --host HOST          Host to bind to (default: 0.0.0.0)"
            echo ""
            echo "Environment variables:"
            echo "  LORA_PATH            Same as --lora-path"
            echo "  MODEL_PATH           Same as --model-path"
            echo "  PORT                 Same as --port"
            echo "  HOST                 Same as --host"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate paths
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found: $MODEL_PATH"
    exit 1
fi

if [ -n "$LORA_PATH" ] && [ ! -d "$LORA_PATH" ]; then
    echo "Error: LoRA path not found: $LORA_PATH"
    exit 1
fi

echo "======================================"
echo "Qwen3-VL Thinking Mode Visualization"
echo "======================================"
echo "Model: $MODEL_PATH"
if [ -n "$LORA_PATH" ]; then
    echo "LoRA: $LORA_PATH"
fi
echo "Server: http://$HOST:$PORT"
echo "======================================"
echo ""

# Run the server
cd "$(dirname "$0")"
PYTHON_BIN="/share/project/xiyan/envs/ocrflow/bin/python"
if [ -n "$LORA_PATH" ]; then
    "$PYTHON_BIN" app.py \
        --model-path "$MODEL_PATH" \
        --lora-path "$LORA_PATH" \
        --port "$PORT" \
        --host "$HOST"
else
    "$PYTHON_BIN" app.py \
        --model-path "$MODEL_PATH" \
        --port "$PORT" \
        --host "$HOST"
fi
