#!/bin/bash
# Wrapper script to run alignment training with OCRVL patches

# Set PYTHONPATH to include sitecustomize.py
export PYTHONPATH="/share/project/xiyan/sources/DeepSeek-OCR/OCRVL/llamafactory:$PYTHONPATH"

# Navigate to project root
cd /share/project/xiyan/sources/DeepSeek-OCR

# Run training
llamafactory-cli train OCRVL/examples/llamafactory/qwen3vl_dpskocr_lora_alignment.yaml
