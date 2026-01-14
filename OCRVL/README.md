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

OCRVL provides three-stage training:

1. **Stage 1: Alignment** - Train connectors on BLIP3o captioning dataset
2. **Stage 2: Instruction Tuning** - Train on LLaVA-Instruct-150K for VQA
3. **Stage 3: Thinking Training** - Train thinking projection for Chain-of-Thought reasoning

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

### Stage 3: Thinking Training

Train thinking projection MLP for Chain-of-Thought reasoning with latent tokens:

```bash
# Continue from instruction tuning checkpoint (Stage 2)
RESUME_CHECKPOINT=OCRVL/checkpoints/llava_20251231_010301/instruction/step_latest \
  OUTPUT_DIR=OCRVL/checkpoints/thinking_stage3 \
  NUM_GPUS=4 \
  GPU_IDS=0,1,2,3 \
  bash OCRVL/scripts/train_thinking.sh

# Configuration via environment variables:
# - RESUME_CHECKPOINT: Path to Phase 2 checkpoint (required)
# - OUTPUT_DIR: Output directory for checkpoints
# - NUM_GPUS: Number of GPUs (default: 4)
# - LORA: Use LoRA for LLM (default: 1)
# - LR: Learning rate (default: 1e-4 for thinking projection)
# - THINKING_LOSS_WEIGHT: Weight for thinking loss (default: 1.0)
# - MAX_SAMPLES: Max samples to load (default: 100000)
# - BATCH_SIZE: Per-GPU batch size (default: 4)
# - GRAD_ACCUM: Gradient accumulation steps (default: 4)

# Example: Fast testing (1000 samples)
MAX_SAMPLES=1000 \
  RESUME_CHECKPOINT=OCRVL/checkpoints/.../step_latest \
  bash OCRVL/scripts/train_thinking.sh

# Example: Full training (8 GPUs)
NUM_GPUS=8 \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  BATCH_SIZE=8 \

## LlamaFactory-Style Pipeline Config

If you prefer LlamaFactory's config-driven workflow (`.yaml` + `key=value` overrides), `OCRVL/scripts/train_llava.sh` supports a LlamaFactory-style YAML that drives the 3-stage OCRVL pipeline.

```bash
# Run the 2-stage LLaVA-style pipeline (alignment -> instruction), Phase 3 disabled by default
bash OCRVL/scripts/train_llava.sh OCRVL/examples/llamafactory/train_llava_pipeline.yaml

# Override fields like LlamaFactory (dot-path key=value)
bash OCRVL/scripts/train_llava.sh OCRVL/examples/llamafactory/train_llava_pipeline.yaml \
  system.cuda_visible_devices=0,1 phase2.train.num_train_epochs=1
```

Notes:
- GPU selection follows LlamaFactory conventions: `CUDA_VISIBLE_DEVICES` (or `system.cuda_visible_devices` in YAML) drives `NUM_GPUS/GPU_IDS`.
- Phase 3 is controlled by `phase3.enabled=true` (or `ENABLE_PHASE3=true`).
  RESUME_CHECKPOINT=OCRVL/checkpoints/.../step_latest \
  bash OCRVL/scripts/train_thinking.sh
```

**Training Details:**
- **Dataset**: LLaVA-CoT-100K with `<REASONING>` and `<CONCLUSION>` tags
- **Dual Loss**: MSE for thinking alignment + CE for answer tokens
- **New Component**: Thinking projection (hidden_dim 4096 → latent_dim 1280, ~10.5M params)
- **Architecture**: Maps model hidden states to OCR-encoded latent tokens
- **Default Setup**: 4 GPUs × 4 batch × 4 accum = 64 effective batch
- **Training Time**: ~2-3 hours (4 GPUs, 100K samples)

**Hyperparameter Guidelines:**

| Component | Recommended LR | Notes |
|-----------|---------------|-------|
| Thinking projection (new) | 1e-4 | Primary trainable component |
| LoRA adapters | 2e-5 (via alpha/r) | Set via `lora_alpha=128, lora_r=64` |

**Thinking Loss Weight:**
- `0.5`: Prioritize answer quality (if thinking loss dominates)
- `1.0`: Balanced (recommended, default)
- `2.0`: Prioritize thinking alignment (if thinking loss too low)

**Expected Losses:**
- Initial: `thinking_loss` 0.5-1.0, `answer_loss` 2.0-3.0, `total_loss` 2.5-4.0
- Converged: `thinking_loss` 0.05-0.15, `answer_loss` 1.0-1.5, `total_loss` 1.05-1.65

**Training Stages Comparison:**

| Stage | Focus | Trainable Params | LR | Data | Time |
|-------|-------|------------------|-----|------|------|
| 1. Alignment | OCR Connector | 22M (13M+9M LoRA) | 1e-3 | BLIP3o 5% | 30min |
| 2. Instruction | Connector + LoRA | 22M | 2e-4 | LLaVA-150K | 2h |
| 3. Thinking | Thinking Proj + LoRA | 60.5M (10.5M+50M) | 1e-4 | LLaVA-CoT-100K | 2-3h |

**Output:** `OCRVL/checkpoints/thinking_YYYYMMDD_HHMMSS/step_N/`
- `connectors.pt` - Includes thinking projection weights
- `lora_adapters/` - LoRA adapters (r=64, alpha=128)
- `training_state.pt` - Optimizer/scheduler state

**Dataset Format:**
```json
{
  "id": "...",
  "image": "sqa/train/20839/image.png",
  "conversations": [
    {"from": "human", "value": "Question text..."},
    {"from": "gpt", "value": "<SUMMARY>...</SUMMARY><REASONING>reasoning text</REASONING><CONCLUSION>answer</CONCLUSION>"}
  ]
}
```

The training extracts:
- **Reasoning** (`<REASONING>` tag) → Rendered + DPSK OCR encoded → Supervision latents
- **Conclusion** (`<CONCLUSION>` tag) → Answer tokens for CE loss

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

The model has 4 separate modules:

1. **OCR Encoder** (DeepSeek-OCR, ~400M params)
   - CLIP ViT (304M) + SAM ViT (89M) + Projector (2.6M)
   - **Frozen during training** (pretrained weights)
   - Encodes both rendered text and real images to 100×1280 visual tokens (10×10 grid, no separators)

2. **Connectors** (~13M params)
   - `ocr_connector`: 1280→2048 (final features)
   - `_ocr_deepstack_connectors[1024]`: 1024→2048 (deepstack features)
   - **Always trainable**
   - Maps OCR visual tokens to Qwen3-VL LLM space

3. **Thinking Projection** (~10.5M params, Stage 3 only)
   - Maps LLM hidden states (4096-dim) to OCR latent space (1280-dim)
   - **Trainable in Stage 3** for Chain-of-Thought reasoning
   - Enables generation of latent thinking tokens

4. **Qwen3-VL LLM** (2B params)
   - **LoRA mode** (default): Only LoRA adapters trainable (~8.7M Stage 1-2, ~50M Stage 3)
   - **Full mode**: All LLM params trainable (2B)
   - Controlled by `LORA=1/0` or `--use_lora / --no-use_lora`

**Total Trainable:**
- **Stage 1-2:** ~22M params (13M connectors + 9M LoRA)
- **Stage 3:** ~60.5M params (13M connectors + 10.5M thinking projection + 50M LoRA with higher rank)

### Sequence Format

OCRVL uses **Qwen3-VL official format**: all images concatenated in **ONE** user block.

**Training & Evaluation Format (consistent):**
```
<|im_start|>user
<|vision_start|>[IMAGE1_TOKENS]<|vision_end|><|vision_start|>[IMAGE2_TOKENS]<|vision_end|>
<|im_end|>
<|im_start|>assistant
[Response text]
<|im_end|>
```

**RENDER=1 Mode (default):**
- Real image + rendered question as two images in one user block
- Random ordering: 50% real-first, 50% rendered-first (for robustness)
- Example: `<|im_start|>user[PHOTO][RENDERED_Q&A]<|im_end|><|im_start|>assistant[Answer]<|im_end|>`

**RENDER=0 Mode:**
- Real image + text prompt (no rendering)
- Example: `<|im_start|>user[PHOTO]What is shown?<|im_end|><|im_start|>assistant[Answer]<|im_end|>`

**Critical:** Training and evaluation must use identical formats to avoid distribution mismatch.

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
- LLaVA-CoT: `/share/project/xiyan/huggingface/Xkev/LLaVA-CoT-100k/train.jsonl`

### LLaVA-Instruct-150K (Instruction Stage)

```bash
# Conversation configuration
--llava_single_turn_ratio 0.5    # 50% single-turn, 50% multi-turn
--llava_max_turns 3              # Limit to first 3 turns (None = all)

# Render mode
--llava_render_questions          # RENDER=1 (default, questions as images)
--no-llava_render_questions       # RENDER=0 (text prompts)
```

### LLaVA-CoT-100K (Thinking Stage)

```bash
# Dataset configuration
--thinking_jsonl_path /path/to/train.jsonl
--thinking_image_dir /path/to/images
--thinking_max_samples 100000        # Use all samples (default)
--thinking_loss_weight 1.0           # Balance thinking and answer losses

# Example: Custom dataset with lower sample count
--thinking_jsonl_path /path/to/custom_cot.jsonl \
--thinking_image_dir /path/to/images \
--thinking_max_samples 10000
```

**Format Requirements:**
- JSONL format with `<REASONING>` and `<CONCLUSION>` tags
- Each sample has `id`, `image`, and `conversations` fields
- Assistant response must contain reasoning in `<REASONING>` tags
- Final answer must be in `<CONCLUSION>` tags

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

- **Follow the stages**: Train Stage 1 (alignment) → Stage 2 (instruction) → Stage 3 (thinking) sequentially
- **Use LoRA by default**: Stage 1-2: 22M params, Stage 3: 60.5M params (vs 2B full LLM)
- **Validate at step 0**: Initial checkpoint saved to verify saving works before long training
- **Monitor GPU usage**: ~6-7 GB per GPU (Stage 1-2), ~18-20 GB per GPU (Stage 3)
- **Effective batch size**: Keep ≥64 for Stage 3, ≥1024 for Stage 1-2 (adjust GPUs × batch × accum)
- **Stage 3 requires Stage 2**: Must resume from instruction-tuned checkpoint

### Dataset

- **Alignment**: 5% of BLIP3o long is sufficient for connector alignment
- **Instruction**: Use full LLaVA-Instruct-150K (no sampling)
- **Thinking**: Full LLaVA-CoT-100K (100K samples, 1 epoch)
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
- Reduce `BATCH_SIZE` (default: 8 for alignment, 4 for instruction/thinking)
- Increase `GRAD_ACCUM` to maintain effective batch size
- Use `LORA=1` (default) instead of full LLM training
- Stage 3 uses more memory (~18-20GB per GPU) due to thinking projection

**Thinking loss not decreasing (Stage 3):**
- Verify `RESUME_CHECKPOINT` has instruction-tuned weights (Stage 2 checkpoint)
- Check dataset has `<REASONING>` and `<CONCLUSION>` tags
- Increase learning rate: `LR=2e-4` (default is 1e-4)
- Adjust `THINKING_LOSS_WEIGHT` (try 0.5 or 2.0 instead of 1.0)
- Monitor both `thinking_loss` and `answer_loss` - both should decrease

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
│   ├── train_instruction.sh     # Stage 2: LLaVA instruction tuning
│   ├── train_thinking.sh        # Stage 3: Thinking training
│   └── train_llava.sh          # Full 3-stage pipeline
├── model/
│   ├── ocr_llava_arch.py       # OCR-aware model architecture
│   └── language_model/
│       └── ocr_llava_qwen.py   # Qwen3-VL integration
├── data/
│   ├── blip3o_dataset.py       # BLIP3o dataset loader
│   ├── blip3o_collate.py       # BLIP3o collate function
│   ├── blip3o_tasks.py         # Task formatting (caption/OCR/match)
│   ├── llava_instruct_dataset.py  # LLaVA dataset loader
│   ├── thinking_dataset.py     # LLaVA-CoT dataset with reasoning extraction
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
    ├── alignment_*/step_*/
    ├── instruction_*/step_*/
    └── thinking_*/step_*/
```

## See Also

- **DeepSeek-OCR**: Base vision encoder for text and image encoding
- **OCRInfer**: Encoder utilities and model paths
- **Renderer**: GPU-accelerated Vello renderer (optional but recommended)
- **Qwen3-VL**: Base vision-language model (Qwen/Qwen3-VL-2B-Instruct)

## LlamaFactory Training (Recommended)

OCRVL can be trained using `llamafactory-cli` while still using the OCR-aware Qwen3-VL wrapper and the DeepSeek-OCR (DPSK) vision encoder.

Key idea:
- LlamaFactory uses an OCRVL-registered Qwen3-VL template (`ocrvl_qwen3_vl_nothink`) so tokenization does not eagerly load images.
- We patch the processor at runtime so LlamaFactory produces OCRVL-compatible `pixel_values` (DeepSeek-OCR preprocessed tensors).
- We register the OCR-aware model class so `AutoModelForCausalLM` instantiates it for Qwen3-VL configs.

### Quickstart

```bash
# 1) Prepare dataset links + dataset_info.json for LlamaFactory
bash OCRVL/scripts/prepare_llamafactory_datasets.sh

# 2) Run LoRA SFT on LLaVA Mix-665K (start with max_samples=1000 in the YAML)
bash OCRVL/scripts/train_llava_llamafactory.sh OCRVL/examples/llamafactory/qwen3vl_dpskocr_lora_llava665k.yaml
```

### Notes

- The patch is activated via `PYTHONPATH` and repo-root `sitecustomize.py` (the wrapper script sets this automatically).
- Train/save OCRVL connector modules via LlamaFactory `additional_target`:
  `additional_target: model.ocr_connector,model.ocr_deepstack_connector`
- DPSK encoder selection:
  - `DPSK_MODEL_PATH` default: `/share/project/xiyan/huggingface/deepseek-ai/DeepSeek-OCR`
  - `DPSK_DTYPE` default: `bf16`
- Qualitative checkpoint outputs:
  - Enabled by default in `OCRVL/scripts/train_llava_llamafactory.sh`
  - Saves to `${output_dir}/checkpoint-*/eval_results/transparent_eval.{json,txt}`
  - Uses samples from `OCRVL/llamafactory/transparent_eval_samples.json` (12 fixed images + 1 rendered-text OCR sample)
- Output directory:
  - Example configs write to `OCRVL/checkpoints/llamafactory/...`
  - If a config omits `output_dir`, `OCRVL/scripts/train_llava_llamafactory.sh` defaults to `OCRVL/checkpoints/llamafactory/<timestamp>`
- LlamaFactory dataset setup:
  - Dataset definitions live in `OCRVL/llamafactory/data/dataset_info.json`
  - `bash OCRVL/scripts/prepare_llamafactory_datasets.sh` creates symlinks for large upstream JSON/JSONL files
  - Available dataset names: `ocrvl_llava_mix665k`, `ocrvl_llava_cot_100k`, `ocrvl_alignment_llava_pretrain_doclaynet`

### Alignment Dataset (LLaVA-Pretrain + DocLayNet)

The alignment-stage data is provided as a LlamaFactory local file dataset: `ocrvl_alignment_llava_pretrain_doclaynet`.

Build it (writes a ShareGPT JSONL manifest that points at existing local media under `/share/project/xiyan/huggingface/liuhaotian/LLaVA-Pretrain` + DocLayNet):

```bash
bash OCRVL/scripts/prepare_llamafactory_alignment_dataset.sh \
  --doc_ratio 0.5
```

Example config: `OCRVL/examples/llamafactory/qwen3vl_dpskocr_lora_alignment.yaml`

Notes:
- Default build uses the full corpora (no truncation). Use `--max_samples N` only for quick debugging.
- LLaVA-Pretrain images must be extracted to `/share/project/xiyan/huggingface/liuhaotian/LLaVA-Pretrain/images/` so JSON `image` paths resolve.
