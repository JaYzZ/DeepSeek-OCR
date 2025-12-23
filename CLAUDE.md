# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Key Principals
1. DO NOT write random new .md to the repo, summarization should be in the chat
2. DO NOT change any other reference repo's content, edit is forbidden
3. DO NOT change the environment installation, the conda env is /share/project/xiyan/envs/ocrflow, any pip install should be passed for user validation

## Repository Overview

This is a research repository for **DeepSeek-OCR**, a vision-text compression model that investigates vision encoders from an LLM-centric viewpoint. The repository contains three main components:

1. **DeepSeek-OCR-vllm** - vLLM-based inference and server implementation
2. **OCRFlow** - High-performance training system with GPU-accelerated rendering
3. **OCRVL** - OCR-aware LLaVA wrappers for vision-language models

## Core Architecture

### Vision Token Encoding
- **Input**: Images or rendered text at 640×640 resolution
- **Output**: 111 visual tokens (100 grid tokens + 11 newline tokens), each 1280-dim
- **Encoder**: CLIP ViT (304M) + SAM ViT (89M) + Projector (2.6M) = ~400M params
- **Key Feature**: Encoder-only mode separates vision encoding from LLM inference

### Training Pipeline (OCRFlow)
```
Text → Vello Renderer (1565 img/s) → Vision Encoder (GPU 1-7) →
Rolling Cache (50K pairs) → Training (GPU 0) → Markovian Decoder
```

**Multi-GPU Architecture**:
- **7 Encoder GPUs** (1-7): Run vision encoder in parallel (~297 pairs/s each)
- **1 Training GPU** (0): Consumes from shared memory cache (~2,079 pairs/s total)
- **Zero-copy cache**: Shared memory tensors (no serialization overhead)
- **Expected throughput**: ~91% training GPU utilization

### Inference Architecture (vLLM)
- **Full mode**: Image → OCR text (requires vLLM with full model)
- **Encoder-only mode**: Text ↔ visual tokens (lightweight, no LLM)
- **Decoder**: DeepSeek 3B MoE for visual-tokens-to-text

## Common Commands

### OCRFlow Training

**Start training (8 GPUs, tmux):**
```bash
cd OCRFlow
./scripts/start_training.sh
```

**Manual training with custom config:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python OCRFlow/examples/train.py \
    --max_steps 50000 \
    --encode_batch_size 8 \
    --train_batch_size 16
```

**Monitor training:**
```bash
# Attach to tmux session
tmux attach -t ocrflow_training

# Watch GPU utilization
watch -n 1 nvidia-smi

# Check logs
tail -f checkpoints/dedicated_pool/training.log
```

**Kill training:**
```bash
tmux kill-session -t ocrflow_training
```

### Vello Renderer Setup

**Build GPU-accelerated renderer (one-time setup):**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Install dependencies
sudo apt-get update
sudo apt-get install -y libvulkan-dev vulkan-tools fonts-noto-cjk

# Build Vello (takes 3-5 min first time)
cd Renderer
pip install maturin
maturin develop --release

# Verify
python -c "from Renderer import VelloRenderer; print('OK')"
```

**Benchmark renderers:**
```bash
cd OCRFlow
python scripts/benchmark_renderers.py
# Expected: Vello 1565 img/s, Skia 778 img/s, PIL 130 img/s
```

### DeepSeek-OCR Server

**Start vLLM inference server:**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
./server/start_server.sh

# Or with custom config
python server/deepseek_ocr_server.py \
    --port 8010 \
    --gpu-devices 0 \
    --gpu-memory-utilization 0.9
```

**Start encoder-only server (lightweight):**
```bash
python server/deepseek_ocr_server.py \
    --encoder-only \
    --port 8011 \
    --gpu-devices 0
```

**Check server health:**
```bash
curl http://localhost:8010/health
```

**Kill stuck vLLM processes:**
```bash
# If GPU memory is not released after stopping server
pkill -9 -f "VLLM::EngineCore"
```

### vLLM Inference Scripts

**Single image OCR:**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_image.py
```

**PDF OCR (batch):**
```bash
python run_dpsk_ocr_pdf.py
```

**Batch evaluation:**
```bash
python run_dpsk_ocr_eval_batch.py
```

## Key File Locations

### Training Components
- `OCRFlow/examples/train.py` - Main training script with dedicated pool architecture
- `OCRFlow/examples/train_rolling_cache.py` - Training with rolling cache dataset
- `OCRFlow/training/rolling_cache_dataset.py` - Rolling cache implementation
- `OCRFlow/training/mar_diffusion.py` - MAR training with diffusion loss
- `OCRFlow/models/markovian_chunk_decoder.py` - Chunk-to-chunk decoder

### Vision Encoding
- `OCRInfer/encoder/dpsk_ocr_encoder.py` - Standalone vision encoder (~400M params)
- `Renderer/vello_renderer_wrapper.py` - Fast renderer wrapper

### Server & Inference
- `DeepSeek-OCR-master/DeepSeek-OCR-vllm/server/deepseek_ocr_server.py` - FastAPI server
- `DeepSeek-OCR-master/DeepSeek-OCR-vllm/server/standalone_vision_encoder.py` - V1 encoder
- `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py` - vLLM configuration

### OCRVL (Vision-Language)
- `OCRVL/builder.py` - OCR-LLaVA model loader
- `OCRVL/model/ocr_llava_arch.py` - OCR-aligned LLaVA architecture
- `OCRVL/model/language_model/ocr_llava_llama.py` - LLaMA integration

## Important Architecture Details

### Visual Token Format
- **Token count**: 111 tokens (100 grid + 11 newline separators)
- **Token dimension**: 1280
- **Encoding**: bfloat16 (2 bytes/value)
- **Binary size**: 111 × 1280 × 2 = ~284KB per image

### Rendering Pipeline
Three renderer backends (auto-selected in priority order):
1. **Vello** (Rust/Vulkan): 1565 img/s, GPU compute shaders, requires `maturin develop`
2. **Skia**: 778 img/s, CPU-based, fallback if Vello unavailable
3. **PIL**: 130 img/s, pure Python, always available

**CJK Support**: All renderers use Noto Sans CJK fonts for Chinese/Japanese/Korean

### Training Data Flow
```
FineWeb/OpenWebMath Dataset
    ↓
Variable Chunking (50-900 words)
    ↓
Vello Renderer → Images (640×640)
    ↓
Vision Encoder (parallel, GPU 1-7)
    ↓
Zero-Copy Shared Memory Cache (50K pairs)
    ↓
Training Worker (GPU 0) → Markovian Chunk Decoder
```

### vLLM Configuration
Required parameters for DeepSeek-OCR:
- `enable_prefix_caching=False` (vision tokens not cacheable)
- `mm_processor_cache_gb=0` (disable processor cache)
- `logits_processors=[NGramPerReqLogitsProcessor]` (prevent repetition)

**Known Issues**:
- vLLM v0.11.1rc7.dev234 produces repetitive output
- Use stable vLLM v0.11.1+ or newer nightly builds

## Development Workflow

### Making Changes to Training
1. Edit training code in `OCRFlow/examples/train.py`
2. If modifying encoder: edit `OCRInfer/encoder/dpsk_ocr_encoder.py`
3. Test with short run: `python OCRFlow/examples/train.py --max_steps 10`
4. Full training: `./OCRFlow/scripts/start_training.sh`

### Making Changes to Vello Renderer
1. Edit Rust code in `Renderer/src/lib.rs`
2. Rebuild: `cd Renderer && maturin develop --release`
3. Test: `python -c "from Renderer import VelloRenderer; print('OK')"`
4. Benchmark: `python OCRFlow/scripts/benchmark_renderers.py`

### Making Changes to Server
1. Edit `DeepSeek-OCR-master/DeepSeek-OCR-vllm/server/deepseek_ocr_server.py`
2. Kill existing server: `tmux kill-session -t ocr_server_v1`
3. Restart: `./DeepSeek-OCR-master/DeepSeek-OCR-vllm/server/start_server.sh`
4. Test: `curl http://localhost:8010/health`

### Debugging GPU Issues
```bash
# Check GPU memory
nvidia-smi

# Kill stuck processes
pkill -9 -f "VLLM::EngineCore"
pkill -9 python

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Check process using GPU
fuser -v /dev/nvidia*
lsof /dev/nvidia*
```

## Testing

### Unit Tests
```bash
# Test OCRVL OCR token runs
python OCRVL/tests/test_ocr_token_runs.py
```

### Integration Tests
```bash
# Test encoder-decoder roundtrip
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python test_encoder_decoder.py
python test_encoder_decoder_separation.py

# Test serverless inference
python test_serverless_roundtrip.py
python test_simple_ocr.py
```

### Server Validation
```bash
# End-to-end validation with server
cd OCRFlow
./scripts/start_server_and_validate.sh
```

## Environment Setup

### Python Environment
```bash
# Create conda environment
conda create -n deepseek-ocr python=3.12 -y
conda activate deepseek-ocr

# Install PyTorch (CUDA 11.8)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# Install vLLM (for inference)
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Install dependencies
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

### GPU Requirements
- **Minimum**: 1x GPU with 24GB VRAM
- **Training (full speed)**: 8x GPU (7 for encoding + 1 for training)
- **Inference**: 1x GPU with 16GB VRAM (encoder-only) or 24GB (full mode)

## Special Considerations

### Vision Encoder Performance

**Current Implementation (DPSKOCREncoder backend):**
- **Throughput**: ~117 img/s per GPU
- **Warmup**: ~0.14s (very fast)
- **Load time**: ~25s
- **Status**: Active, unified API with OCRInfer

**Pre-compiled Backup (deepencoder with torch.compile):**
- **Throughput**: ~127 img/s per GPU (~10% faster)
- **Warmup**: ~4.3s (after loading pre-compiled models)
- **Load time**: ~1.2s for pre-compiled + 25s base model
- **Location**: `checkpoints/compiled/`
  - `vision_encoder_clip_compiled.pt` (579 MB)
  - `vision_encoder_sam_compiled.pt` (183 MB)
  - `vision_encoder_old.py` (implementation)
  - `README.md` (usage instructions)
- **Status**: Backup only, for maximum throughput scenarios

**Note**: torch.compile benefits the old deepencoder implementation (3.3x speedup)
but makes the new DPSKOCREncoder 20x slower. Current version prioritizes API
consistency and fast warmup over peak throughput.

### Memory Management
- **Encoder GPUs**: ~800MB each (vision encoder only)
- **Training GPU**: ~20GB (full model + optimizer)
- **System RAM**: ~8GB (dataset + cache)
- **Disk**: ~500GB for full training datasets

### Working with tmux
Training and servers run in tmux for persistence:
```bash
# List sessions
tmux ls

# Attach to session
tmux attach -t ocrflow_training

# Detach from session (while inside)
Ctrl+B, then D

# Kill session
tmux kill-session -t session_name

# Send commands to session
tmux send-keys -t session_name "command" C-m
```

## Codebase Patterns

### Model Loading
Vision encoder loads weights directly from HuggingFace checkpoint without loading full LLM:
- Loads: `vision_model.*`, `mlp1.*`, `newline_emb`, `separator_emb`
- Skips: `language_model.*` (saves ~13GB VRAM)

### Visual Token Shape
All visual token tensors follow the same shape:
- `[batch_size, 111, 1280]` - Batch of encoded images
- `[111, 1280]` - Single encoded image
- Always bfloat16 dtype for consistency

### Error Handling
When debugging vLLM issues:
1. Check server logs: `tail -f /tmp/ocr_server_v1.log`
2. Verify vLLM version compatibility
3. Check GPU memory with `nvidia-smi`
4. Kill stuck processes with `pkill -9 -f "VLLM::EngineCore"`
