# OCRVL - OCR-aware Vision-Language Training & Inference

OCRVL provides OCR-aware wrappers for Qwen-VL models and a complete training pipeline for vision-language alignment and instruction tuning.

## Features

- **OCR-aware Model Wrappers**: `OCRQwen3VLForConditionalGeneration` and `OCRQwen25VLForConditionalGeneration`
- **Text & Image Adapters**: Unified interface for rendering text and encoding images via DeepSeek-OCR
- **Training Pipeline**: Stage-based training (alignment → instruction tuning) with LoRA support
- **vLLM Integration**: Fast inference with connector integration (no HF export needed)
- **Evaluation Suite**: Unified benchmarking on RealWorldQA, MMMU, MathVision, ODinW

## Quick Start

### Installation

```bash
# OCRInfer provides the DeepSeek-OCR encoder
pip install -e OCRInfer

# OCRVL training dependencies
pip install torch torchvision transformers peft datasets
pip install flash-attn==2.7.3 --no-build-isolation

# Optional: GPU-accelerated Vello renderer (recommended)
cd Renderer
pip install maturin
maturin develop --release
```

### Inference Example

```python
from OCRVL import (
    OCRQwen3VLForConditionalGeneration,
    Qwen3VLOCRTextAdapter,
)
from transformers import AutoTokenizer

qwen_path = "Qwen/Qwen3-VL-2B-Instruct"
ocr_path = "deepseek-ai/DeepSeek-OCR"

tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
adapter = Qwen3VLOCRTextAdapter(
    encoder_model_path=ocr_path,
    device="cuda",
    use_deepstack=True,  # Enable deepstack features
)

# Encode rendered text as vision tokens
instruction = "Read the document and summarize."
dense_text = open("long.txt").read()
input_ids, ocr_feats = adapter.prepare_qwen_inputs(
    instruction="",
    dense_text=dense_text,
    tokenizer=tokenizer,
    render_instruction=False,
)

# Or encode real images
from PIL import Image
images = [Image.open("doc.png")]
input_ids, ocr_feats = adapter.prepare_qwen_inputs_from_images(
    instruction="",
    images=images,
    tokenizer=tokenizer,
    return_deepstack=True,
)

# Generate
model = OCRQwen3VLForConditionalGeneration.from_pretrained(
    qwen_path,
    torch_dtype="bfloat16",
    device_map="cuda"
)
out = model.generate(
    input_ids=input_ids.to(model.device),
    ocr_image_features=ocr_feats,
    max_new_tokens=256,
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Training Pipeline

OCRVL provides two-stage training:

1. **Stage 1: Alignment** - Train connectors on BLIP3o captioning dataset
2. **Stage 2: Instruction Tuning** - Train on LLaVA-Instruct-150K for VQA

### Stage 1: Connector Alignment

Train connectors to align DeepSeek-OCR visual tokens with Qwen3-VL LLM:

```bash
# Train with LoRA + Connectors (default, recommended)
bash OCRVL/scripts/train_alignment.sh

# Configuration via environment variables:
# - LORA=1 (default): Enable LoRA on LLM (8.7M params) + connectors (13M)
# - LORA=0: Train only connectors (13M params)
# - DATASET_PCT=0.05: Use 5% of dataset (default)
# - BLIP3O_DATASET=long: Use long captions (default: long, choices: short/long/60k/mixed)
# - NUM_EPOCHS=1: Number of epochs
# - NUM_GPUS=8: Number of GPUs
# - LR=2e-4: Learning rate (auto-adjusted for LoRA)

# Example: Train on 10% dataset with 4 GPUs
DATASET_PCT=0.1 NUM_GPUS=4 bash OCRVL/scripts/train_alignment.sh

# Resume from checkpoint (weights only, restart from step 0)
RESUME_CHECKPOINT=OCRVL/checkpoints/alignment_long_20251227_142721/step_786 \
  bash OCRVL/scripts/train_alignment.sh

# Resume training state (weights + optimizer + step counter)
RESUME_CHECKPOINT=OCRVL/checkpoints/.../step_786 \
  RESUME_TRAINING_STATE=true \
  bash OCRVL/scripts/train_alignment.sh
```

**Expected Performance:**
- Throughput: ~90-130 samples/s (8 GPUs, batch_size=8, grad_accum=32)
- Training time: ~4-5 hours per epoch (5% of BLIP3o long dataset)
- Effective batch size: 2048 (8 GPUs × 8 batch × 32 accum)

**Output:** `OCRVL/checkpoints/alignment_long_YYYYMMDD_HHMMSS/step_N/`
- `connectors.pt` - Connector weights (~25 MB)
- `lora_adapters/` - LoRA adapters if enabled (~34 MB)
- `training_state.pt` - Optimizer/scheduler state (~117 MB)

### Stage 2: Instruction Tuning

Continue training on LLaVA-Instruct-150K for VQA capability:

```bash
# Continue from alignment checkpoint (recommended)
RESUME_CHECKPOINT=OCRVL/checkpoints/alignment_long_20251227_142721/step_786 \
  bash OCRVL/scripts/train_instruction.sh

# Configuration via environment variables:
# - RENDER=1 (default): Render questions as images (pure vision mode)
# - RENDER=0: Use text prompts (hybrid mode)
# - NUM_EPOCHS=1: Number of epochs (default)
# - BATCH_SIZE=4: Per-GPU batch size (default, lower than alignment)
# - SINGLE_TURN_RATIO=0.5: 50% single-turn, 50% multi-turn samples
# - MAX_TURNS=: Limit conversation turns (empty = all turns)

# Example: RENDER=0 (text prompts instead of rendered questions)
RENDER=0 \
  RESUME_CHECKPOINT=OCRVL/checkpoints/.../step_786 \
  bash OCRVL/scripts/train_instruction.sh
```

**RENDER Modes:**
- **RENDER=1** (default): Questions rendered as images, random ordering
  - 50%: `<Real Image> + <Rendered Question>` → Answer
  - 50%: `<Rendered Question> + <Real Image>` → Answer
- **RENDER=0**: Text prompts
  - `<Real Image> + Text Question` → Answer

**Expected Performance:**
- Throughput: ~30-50 samples/s (8 GPUs)
- Training time: ~2-3 hours per epoch (full LLaVA-150K)
- Effective batch size: 1024 (8 GPUs × 4 batch × 32 accum)

**Output:** `OCRVL/checkpoints/alignment_llava_YYYYMMDD_HHMMSS/step_N/`

### Direct Training (Low-level)

For custom training configurations:

```bash
# Stage 1: Alignment with custom args
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    OCRVL/train.py \
    --stage alignment \
    --dataset-type blip3o \
    --blip3o_dataset long \
    --blip3o_sample_percentage 0.05 \
    --num_epochs 1 \
    --batch_size 8 \
    --gradient_accumulation_steps 32 \
    --lr 2e-4 \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16

# Stage 2: Instruction tuning
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    OCRVL/train.py \
    --stage alignment \
    --dataset-type llava \
    --llava_json_path /path/to/llava_instruct_150k.json \
    --llava_image_dir /path/to/coco/train2014 \
    --llava_render_questions \
    --num_epochs 3 \
    --batch_size 4 \
    --load_checkpoint OCRVL/checkpoints/.../step_786 \
    --use_lora
```

## Evaluation

Evaluate checkpoints on vision-language benchmarks:

```bash
# Run all benchmarks (requires judge model for MMMU/MathVision)
bash OCRVL/evaluation/run_benchmarks.sh \
    -b realworldqa,mmmu,mathvision,odinw13 \
    OCRVL/checkpoints/alignment_llava_20251227_142721/step_2358

# Run single benchmark
bash OCRVL/evaluation/run_benchmarks.sh \
    -b realworldqa \
    OCRVL/checkpoints/.../step_786

# Custom configuration
EVAL_MODEL=gpt-4o \
  API_TYPE=dash \
  bash OCRVL/evaluation/run_benchmarks.sh \
    -b mmmu,mathvision \
    OCRVL/checkpoints/.../step_786
```

**Available Benchmarks:**
- `realworldqa` - Visual reasoning (fast, no judge needed)
- `mmmu` - Multimodal understanding (requires judge)
- `mathvision` - Mathematical reasoning (requires judge)
- `odinw13` - Object detection on 13 datasets

**Output:** `OCRVL/evaluation/results/YYYYMMDD_HHMMSS/`
- `results_summary.txt` - Overall accuracy metrics
- `{benchmark}/` - Detailed predictions and evaluations

## Model Architecture

### Trainable Components

The model has 3 separate modules:

1. **OCR Encoder** (DeepSeek-OCR, ~400M params)
   - CLIP ViT (304M) + SAM ViT (89M) + Projector (2.6M)
   - **Frozen during training** (pretrained weights)
   - Encodes both rendered text and real images to 111×1280 visual tokens

2. **Connectors** (~13M params)
   - `ocr_connector`: 1280→2048 (final features)
   - `_ocr_deepstack_connectors[1024]`: 1024→2048 (deepstack features)
   - **Always trainable**
   - Maps OCR visual tokens to Qwen3-VL LLM space

3. **Qwen3-VL LLM** (2B params)
   - **LoRA mode** (default): Only LoRA adapters trainable (~8.7M)
   - **Full mode**: All LLM params trainable (2B)
   - Controlled by `LORA=1/0` or `--use_lora / --no-use_lora`

**Total Trainable (default):** ~22M params (13M connectors + 8.7M LoRA)

### Checkpoint Structure

```
OCRVL/checkpoints/alignment_long_20251227_142721/
├── step_0/              # Initial checkpoint (validation)
├── step_500/            # Intermediate checkpoint
├── step_786/            # Final checkpoint
│   ├── connectors.pt           # Connector weights (~25 MB)
│   ├── lora_adapters/          # LoRA adapters if LORA=1 (~34 MB)
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── training_state.pt       # Optimizer state (~117 MB)
└── training.log         # Full training log
```

## Dataset Configuration

### BLIP3o (Alignment Stage)

```bash
# Dataset variants
--blip3o_dataset short      # Concise captions
--blip3o_dataset long       # Detailed captions (default)
--blip3o_dataset 60k        # Curated 60K subset
--blip3o_dataset mixed      # Mix of short/long

# Sampling
--blip3o_sample_percentage 0.05   # Use 5% (default)
--blip3o_sample_percentage 1.0    # Use 100% (full dataset)

# Task configuration
--enable_match_task       # Enable image-text matching (Task 3)
                          # Default: disabled (Task 1: caption, Task 2: OCR only)
```

**Dataset Locations:**
- BLIP3o: `/share/project/xiyan/huggingface/BLIP3o/`
- LLaVA: `/share/project/xiyan/data/llava_instruct/llava_instruct_150k.json`
- COCO: `/share/project/xiyan/data/coco/train2014/`

### LLaVA-Instruct-150K (Instruction Stage)

```bash
# Conversation configuration
--llava_single_turn_ratio 0.5    # 50% single-turn, 50% multi-turn
--llava_max_turns 3              # Limit to first 3 turns (None = all)

# Render mode
--llava_render_questions          # RENDER=1 (default, questions as images)
--no-llava_render_questions       # RENDER=0 (text prompts)
```

## vLLM Integration

OCRVL provides direct vLLM integration without HuggingFace export:

```python
from OCRVL.decoder import OCRVLProcessor
from OCRInfer.encoder import DPSKOCREncoder

# Initialize encoder
encoder = DPSKOCREncoder(device="cuda:0")

# Initialize vLLM processor with checkpoint
processor = OCRVLProcessor(
    checkpoint_path="OCRVL/checkpoints/alignment_llava_20251227_142721/step_2358",
    device="cuda:0",
    gpu_memory_utilization=0.85,
)

# Encode images
from PIL import Image
images = [Image.open("question.png")]
final_features, deepstack_features = encoder.encode_images_with_deepstack(images)

# Generate
predictions = processor.generate(
    visual_embeddings=[[final_features[0]]],
    deepstack_features=[[deepstack_features[0]]],
    prompts=["<|im_start|>assistant\n"],  # Pure vision mode
    max_tokens=256,
    temperature=0.0,
)
print(predictions[0])
```

See `OCRVL/evaluation/eval_realworldqa_vllm.py` for full example.

## Advanced Configuration

### LoRA Configuration

```bash
# LoRA rank and alpha
LORA_R=8 LORA_ALPHA=16 bash OCRVL/scripts/train_alignment.sh

# Disable LoRA (train full LLM, requires more memory)
LORA=0 bash OCRVL/scripts/train_alignment.sh
```

### Resume Training

```bash
# Load weights only (restart from step 0)
RESUME_CHECKPOINT=OCRVL/checkpoints/.../step_786 \
  bash OCRVL/scripts/train_alignment.sh

# Resume full training state (continue from step 786)
RESUME_CHECKPOINT=OCRVL/checkpoints/.../step_786 \
  RESUME_TRAINING_STATE=true \
  bash OCRVL/scripts/train_alignment.sh

# Load checkpoint but change LoRA mode
LORA=0 \
  RESUME_CHECKPOINT=OCRVL/checkpoints/.../step_786 \
  bash OCRVL/scripts/train_alignment.sh
```

**Checkpoint Loading Behavior:**
- Always loads available weights (connectors + LoRA if present)
- If checkpoint has LoRA but `LORA=0`: loads LoRA anyway (checkpoint takes priority)
- If checkpoint lacks LoRA but `LORA=1`: initializes new LoRA layers
- `RESUME_TRAINING_STATE=false` (default): only loads model weights
- `RESUME_TRAINING_STATE=true`: loads weights + optimizer + scheduler + step counter

### Multi-GPU Configuration

```bash
# Auto-detect GPUs (uses all available)
bash OCRVL/scripts/train_alignment.sh

# Specify number of GPUs
NUM_GPUS=4 bash OCRVL/scripts/train_alignment.sh

# Specify GPU IDs
GPU_IDS=0,2,4,6 NUM_GPUS=4 bash OCRVL/scripts/train_alignment.sh

# Change master port (if 29500 is busy)
MASTER_PORT=29501 bash OCRVL/scripts/train_instruction.sh
```

## Tips & Best Practices

### Training

- **Start with alignment**: Always train Stage 1 (BLIP3o) before Stage 2 (LLaVA)
- **Use LoRA by default**: 22M trainable params vs 2B (full LLM)
- **Validate at step 0**: Initial checkpoint saved to verify saving works before long training
- **Monitor GPU usage**: ~6-7 GB per GPU with default settings
- **Effective batch size**: Keep ≥1024 for stable training (adjust GPUs × batch × accum)

### Dataset

- **Alignment**: 5% of BLIP3o long is sufficient for connector alignment
- **Instruction**: Use full LLaVA-Instruct-150K (no sampling)
- **RENDER=1 recommended**: Pure vision mode matches training better than text prompts

### Evaluation

- **RealWorldQA**: Fast baseline, no judge needed
- **MMMU/MathVision**: Requires judge model (gpt-4o or local vLLM)
- **Data-parallel**: Benchmarks automatically shard across available GPUs

### Common Issues

**Empty checkpoints during training:**
- Fixed in current version with proper error handling
- Initial checkpoint saved at step 0 for validation

**GPU out of memory:**
- Reduce `BATCH_SIZE` (default: 8 for alignment, 4 for instruction)
- Increase `GRAD_ACCUM` to maintain effective batch size
- Use `LORA=1` (default) instead of full LLM training

**vLLM repetitive output:**
- Fixed with NGramPerReqLogitsProcessor in evaluation scripts
- Use stable vLLM v0.11.1+ (not rc7.dev234)

## File Structure

```
OCRVL/
├── README.md                    # This file
├── builder.py                   # Model loading utilities
├── train.py                     # Main training script
├── scripts/
│   ├── train_alignment.sh       # Stage 1: BLIP3o alignment
│   └── train_instruction.sh     # Stage 2: LLaVA instruction tuning
├── model/
│   ├── ocr_llava_arch.py       # OCR-aware model architecture
│   └── language_model/
│       └── ocr_llava_qwen.py   # Qwen3-VL integration
├── data/
│   ├── blip3o_dataset.py       # BLIP3o dataset loader
│   ├── blip3o_collate.py       # BLIP3o collate function
│   ├── blip3o_tasks.py         # Task formatting (caption/OCR/match)
│   ├── llava_instruct_dataset.py  # LLaVA dataset loader
│   └── ocr_text_adapter.py     # Text rendering + encoding
├── decoder/
│   ├── __init__.py             # vLLM processor exports
│   └── ocrvl_vllm_processor.py # Direct vLLM integration
├── evaluation/
│   ├── run_benchmarks.sh       # Unified evaluation script
│   ├── eval_realworldqa_vllm.py
│   ├── eval_mmmu_vllm.py
│   ├── eval_mathvision_vllm.py
│   └── eval_odinw_vllm.py
└── checkpoints/                # Training outputs
    └── alignment_*/step_*/
```

## See Also

- **DeepSeek-OCR**: Base vision encoder for text and image encoding
- **OCRInfer**: Encoder utilities and model paths
- **Renderer**: GPU-accelerated Vello renderer (optional but recommended)
- **Qwen3-VL**: Base vision-language model (Qwen/Qwen3-VL-2B-Instruct)
