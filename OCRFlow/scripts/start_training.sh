#!/bin/bash
# Start OCRFlow training in tmux (protected from disconnection)
# Usage: ./start_training.sh [gpus] [max_steps]
#   Example: ./start_training.sh 0,1,2,3 50000

set -e

# Config
SESSION_NAME="ocrflow"
GPU_IDS="${1:-0,1,2,3,4,5,6,7}"
MAX_STEPS="${2:-100000}"

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DEEPSEEK_OCR_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
PROJECT_ROOT="${ROOT_DIR:-/share/project/xiyan}"
PYTHON="$PROJECT_ROOT/envs/ocrflow/bin/python"

# Check existing session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Session '$SESSION_NAME' exists. Attach with: tmux attach -t $SESSION_NAME"
    exit 1
fi

echo "Starting OCRFlow training..."
echo "  GPUs: $GPU_IDS"
echo "  Steps: $MAX_STEPS"
echo "  Session: $SESSION_NAME"
echo ""

# Create tmux session and start training
tmux new-session -d -s "$SESSION_NAME" -c "$DEEPSEEK_OCR_DIR"
tmux send-keys -t "$SESSION_NAME" "export CUDA_VISIBLE_DEVICES=$GPU_IDS" C-m
tmux send-keys -t "$SESSION_NAME" "export ROOT_DIR=$PROJECT_ROOT" C-m
tmux send-keys -t "$SESSION_NAME" "export PYTHONPATH=$DEEPSEEK_OCR_DIR:\$PYTHONPATH" C-m
tmux send-keys -t "$SESSION_NAME" "$PYTHON examples/train_rolling_cache.py --gpu-ids $GPU_IDS --max_steps $MAX_STEPS --save_interval 10000 --cache_size 5000 --encode_batch_size 24 --train_batch_size 128 --output_dir ./checkpoints/rolling_cache" C-m

echo "✓ Training started in tmux!"
echo ""
echo "Monitor:"
echo "  tmux attach -t $SESSION_NAME    # Attach to session (Ctrl+B D to detach)"
echo "  tmux kill-session -t $SESSION_NAME  # Stop training"
