#!/bin/bash
# Qwen3-VL Thinking Mode Visualization Server Startup Script

# Default values - use checkpoint from plan
CHECKPOINT="${CHECKPOINT:-Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_sota_1ep/checkpoint_latest}"
BASE_MODEL="${BASE_MODEL:-/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking}"
PORT="${PORT:-8501}"
HOST="${HOST:-0.0.0.0}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
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
            echo "  --checkpoint PATH    Path to LoRA checkpoint (default: Qwen/checkpoints/...)"
            echo "  --base-model PATH    Path to base model (default: /share/.../Qwen3-VL-2B-Thinking)"
            echo "  --port PORT          Port to run server on (default: 8501)"
            echo "  --host HOST          Host to bind to (default: 0.0.0.0)"
            echo ""
            echo "Environment variables:"
            echo "  CHECKPOINT           Same as --checkpoint"
            echo "  BASE_MODEL           Same as --base-model"
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
if [ ! -d "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -d "$BASE_MODEL" ]; then
    echo "Error: Base model not found: $BASE_MODEL"
    exit 1
fi

echo "======================================"
echo "Qwen3-VL Thinking Mode Visualization"
echo "======================================"
echo "Checkpoint: $CHECKPOINT"
echo "Base Model: $BASE_MODEL"
echo "Server: http://$HOST:$PORT"
echo "======================================"
echo ""

# Run the server
cd "$(dirname "$0")"
python app.py \
    --checkpoint "$CHECKPOINT" \
    --base-model "$BASE_MODEL" \
    --port "$PORT" \
    --host "$HOST"
