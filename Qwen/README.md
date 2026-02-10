# Qwen3VL Native Training

Native Qwen3-VL-2B-Thinking training with adaptive text rendering and Qwen3VL cross-attention latent injection.

## Quick Start

```bash
# 1. Create OCR subset from existing unified SFT data (one-time setup)
python Qwen/scripts/filter_ocr_subset.py

# 2. Train unified SFT (VQA + caption + OCR)
bash Qwen/scripts/train_qwen3vl.sh Qwen/configs/qwen3vl_native_unified_sft.yaml

# 3. Supplement with OCR-centric training (bbox_ocr, full_ocr, markdown, rendered_ocr)
bash Qwen/scripts/train_qwen3vl.sh Qwen/configs/qwen3vl_ocr_supplement.yaml
```

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Unified SFT (All Tasks)                                   │
│ - VQA: 2.0M samples (49%)                                        │
│ - bbox_ocr: 800K samples (20%)                                     │
│ - caption: 1.1M samples (27%)                                        │
│ - OCR: 160K samples (4%)                                            │
│                                                                  │
│ Output: Qwen/checkpoints/qwen3vl-2b/lora/native_unified_sft/       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: OCR Supplement (OCR-Centric)                             │
│ - bbox_ocr: 800K samples                                         │
│ - caption_render_ocr: 558K samples                               │
│ - full_document_ocr: 80K samples                                  │
│ - markdown_conversion: 80K samples                                │
│                                                                  │
│ Input: native_unified_sft checkpoint                             │
│ Output: Qwen/checkpoints/qwen3vl-2b/lora/ocr_supplement/          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌�─────────────────────────────────────────────────────────────────┐
│ Stage 3: R1-OneVision Thinking (with Latent Injection)            │
│ - 177K samples with CoT reasoning                                 │
│ - Adaptive rendering (256-1536)                                  │
│ - Qwen3VL cross-attention latent injection                       │
│                                                                  │
│ Input: ocr_supplement checkpoint                                  │
│ Output: Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/   │
└─────────────────────────────────────────────────────────────────┘
```

## Config Files

| Config | Purpose | Samples | Key Features |
|--------|---------|---------|--------------|
| `qwen3vl_native_unified_sft.yaml` | Stage 1: Unified SFT | 4.1M | All tasks, foundation |
| `qwen3vl_ocr_supplement.yaml` | Stage 2: OCR supplement | 1.5M | OCR-focused, bbox_ocr + full_ocr + markdown |
| `qwen3vl_native_r1onevision_thinking.yaml` | Stage 3: Thinking | 177K | CoT, latent injection |

## Key Features

### 1. Adaptive Text Rendering ✓
- **Variable sizes**: 64-1536px based on text length (not fixed 640×640)
- **Content-aware**: Detects 11 layout types (code, dialog, tables, diagrams)
- **Auto-validation**: Ensures no truncation (retries with 20% larger size)
- **Ready for thinking**: Variable-sized rendered images → cross-attention encoder

### 2. Pre-encoding with Qwen3VL Cross-Attention (Offline)
- Uses Qwen3VL's native ViT (not CLIP like DPSK)
- L-1 features (layer 22) from both question and thinking images
- Final layer (layer 23) as cross-attention
- **Variable input support**: Handles any image size (64-1536px)
- Output: [100, 2048] latent reference features → .latent.pt files
- **Offline only**: Pre-computed during dataset building, not used in forward pass

### 3. Latent Supervision (Training)
- Input: Question TEXT + Question IMAGE (both present)
  - Example: "What's in this image?" + `<vision_start><image_pad>*N<vision_end>`
- Output: `<think><|latent_step|></think> Answer text`
- Forward pass: Extract LLM hidden states at `<|latent_step|>` positions
- Loss: CE loss + Thinking loss (OT/MSE/REPA)
  - Compares LLM hidden states [B, seq, 2048] vs pre-encoded latent reference
  - **NO reverse projection** - direct supervision on LLM hidden states
- Special tokens: `<think>`, `</think>`, `<|latent_step|>`, `<|thinking_sep|>`
- **Semantic chunking**: 2,500 chars/chunk (empirical safe threshold for 1536px with layout preservation)
  - Splits on paragraph breaks > sentences > clauses (maintains coherence)
  - Keeps chunks balanced (avoids tiny last chunk)
  - Respects semantic boundaries (no mid-sentence splits)

### 4. OCR Supplement Training
- **Problem**: Unified SFT was 49% VQA (bbox coordinates), only 4% actual OCR
- **Solution**: Supplement with 1.5M pure OCR samples
- **Tasks**: bbox_ocr + full_document_ocr + markdown_conversion + caption_render_ocr
- **Training**: 2 epochs (vs 3 for main SFT)

## Data Format

### OCR Supplement Sample

```json
{
  "messages": [
    {"role": "user", "content": "Transcribe the text in [0.525, 0.5125, 0.9181, 0.6417]:<image>"},
    {"role": "assistant", "content": "South Spacific Climate Camp\nmeeting, idea's, organising.\nSat 1.15pm, main kitchen"}
  ],
  "images": ["/path/to/image.png"],
  "task": "bbox_ocr"
}
```

### R1-OneVision Thinking Sample

```json
{
  "messages": [
    {"role": "user", "content": "<image>\nQuestion about image?"},
    {"role": "assistant", "content": "Answer"}
  ],
  "images": ["/path/to/question.png"],
  "latent_supervision": ["/path/to/question__thinking_0.latent.pt"],
  "num_latent_steps": 1,
  "task": "r1_onevision_thinking"
}
```

## Usage

### One-Time Setup

```bash
# Create OCR supplement dataset from existing unified SFT
python Qwen/scripts/filter_ocr_subset.py

# Output: Qwen/data/ocrvl_ocr_supplement.jsonl (~1.5M samples)
```

### Training Commands

```bash
# Stage 1: Unified SFT (foundation)
bash Qwen/scripts/train_qwen3vl.sh Qwen/configs/qwen3vl_native_unified_sft.yaml

# Stage 2: OCR supplement (enhance OCR capabilities)
bash Qwen/scripts/train_qwen3vl.sh Qwen/configs/qwen3vl_ocr_supplement.yaml

# Stage 3: R1-OneVision thinking (CoT + latent injection)
# Build dataset with tmux (reuses question images from OCRVL/data/)
bash Qwen/scripts/build_r1_thinking_tmux.sh         # Full build
bash Qwen/scripts/build_r1_thinking_tmux.sh 1000    # Test with 1000 samples

# Monitor build progress
tmux attach -t qwen_r1_build    # Attach to session (Ctrl+B then D to detach)
tail -f Qwen/build_r1_thinking.log  # Watch logs

# After build completes, train
bash Qwen/scripts/train_qwen3vl.sh Qwen/configs/qwen3vl_native_r1onevision_thinking.yaml
```

### Verify Pipeline

```bash
# Test variable-size latent extraction (adaptive render → cross-attention → .latent.pt)
python Qwen/test_varied_size_latents.py
# Expected: ✓ Pipeline ready for R1-OneVision thinking training!
```

## Directory Structure

```
Qwen/
├── configs/
│   ├── qwen3vl_native_unified_sft.yaml          # Stage 1: All tasks
│   ├── qwen3vl_ocr_supplement.yaml                # Stage 2: OCR-focused
│   └── qwen3vl_native_r1onevision_thinking.yaml  # Stage 3: Thinking
├── data/
│   ├── dataset_info.json
│   ├── cache/                                    # Adaptive renders + latents
│   ├── ocrvl_ocr_supplement.jsonl              # Filtered OCR subset
│   └── qwen3vl_transparent_eval.jsonl
├── scripts/
│   ├── adaptive_text_renderer.py                # Adaptive 256-1536 rendering
│   ├── qwen3vl_cross_attention_encoder.py       # Qwen3VL cross-attention
│   ├── filter_ocr_subset.py                     # Create OCR subset
│   ├── build_qwen3vl_native_unified.py          # Build unified SFT
│   ├── build_r1_onevision_thinking.py           # Build thinking data
│   └── train_qwen3vl.sh                         # Training wrapper
└── checkpoints/
    └── qwen3vl-2b/
        └── lora/
            ├── native_unified_sft/               # Stage 1 output
            ├── ocr_supplement/                  # Stage 2 output
            └── r1_onevision_thinking/           # Stage 3 output
```

## Comparison with DPSK Version

| Aspect | DPSK OCR | Qwen3VL Native |
|--------|----------|----------------|
| Vision Encoder | CLIP ViT-L + SAM ViT | Qwen3VL native ViT ✓ |
| Pre-encoding | CLIP layer 23 cross-attention | Qwen3VL layer 23 cross-attention ✓ |
| Input (Training) | Question image only | **Question TEXT + IMAGE** ✓ |
| Reverse Projection | LatentVisualHead | **None** (direct supervision) ✓ |
| Image Size | Fixed 640×640 | **Adaptive 64-1536** ✓ |
| Chunk Size | 800 chars | **2,500 chars** (semantic) ✓ |
| Latent Shape | [100, 1280] | **[100, 2048]** ✓ |
| Layout Detection | None | **11 types** (code, dialog, tables) ✓ |
| Validation | Manual | **Auto-retry** (no truncation) ✓ |
| OCR Connectors | Yes (ocr_connector) | **No** (native vision) |
| Dataset Location | OCRVL/ | Qwen/ (separate cache) |

## Adaptive Renderer

Content-based adaptive renderer with **automatic layout detection** for Qwen3VL (supports **code, markdown, LaTeX, diagrams, tables**):

```python
from Qwen.adaptive_vello_renderer import AdaptiveVelloRenderer

renderer = AdaptiveVelloRenderer(
    min_vello_font=8.0,       # Vello's min font (compact sizing)
    max_vello_font=48.0,      # Vello's max font (readable)
    min_size=64,              # Qwen3VL minimum (can fit "Hello world!")
    padding=15,               # Small padding for efficiency
    safety_multiplier=1.5,    # 50% extra space (handles Vello auto-sizing)
    allow_asymmetric=True,    # Enable H×W asymmetry
    max_retries=3,            # Auto-retry if text is truncated (default: 3)
    edge_margin=5,            # Minimum margin from edges (default: 5px)
)

images = renderer.render_batch(texts, return_dimensions=True)
```

### **Automatic Validation & Retry**

The renderer automatically ensures all text fits within the image:

- **Truncation detection**: Checks for text within 5px of edges
- **Auto-retry**: If truncated, increases size by 20% and retries (up to 3 attempts)
- **Guaranteed fit**: All text is contained with proper margins
- **No manual tuning**: Works automatically for all content types

Example: "Hello world!" renders at 128×64, "Hi" at 64×64 (minimum size).

```python
# Validation happens automatically
images = renderer.render_batch([
    "Short text",           # → 96×64 (compact)
    "Very long text..." * 50,  # → Auto-sized, retries if needed
])
# All images guaranteed to contain full text with margins
```

### **Automatic Layout Detection**

Detects and preserves **11 layout types** including diagrams and scientific content:

| Layout | Detection | Behavior | Example |
|--------|-----------|----------|---------|
| **Plain** | Single line | Wraps naturally | "Hello world!" |
| **Paragraph** | Multi-line, flows | Wraps naturally | Essay text |
| **Dialog** | Q:/A:, User:/Assistant: | **Preserves turns** | "Q: ...?\nA: ..." |
| **List** | Short lines + newlines | **Preserves lines** | "Item 1\nItem 2" |
| **Structured** | Sections with empty lines | **Preserves lines** | "Sec 1\n\nSec 2" |
| **Code** | Indentation/braces/keywords | **Preserves + monospace** | Python, JSON |
| **Markdown** | Headers/lists/code blocks | **Preserves lines** | "# Title\n- Item" |
| **LaTeX** | Math formulas ($$, \frac) | **Preserves lines** | "$$\int x dx$$" |
| **Diagram** | Box chars (┌─┐│), arrows (→) | **Preserves + monospace** | Flowcharts, ASCII art |
| **Tree** | Tree chars (├──, └──, │) | **Preserves + monospace** | File trees |
| **Table** | Column separators (\|), borders | **Preserves + monospace** | Data tables |

### **Symbolic & Scientific Content**

**Diagrams**: Flowcharts, circuit diagrams, network topology, ASCII art
- Detects: Box-drawing chars (┌─┐│), arrows (→←↔), repeated special chars (+|-|*|#)
- Uses monospace for spatial alignment
- Preserves all newlines and spaces

**Trees**: File structures, hierarchies, org charts
- Detects: Tree chars (├──, └──, │)
- Uses monospace for indentation
- Preserves vertical structure

**Tables**: Data tables, comparison charts
- Detects: Column separators (|), border chars (-+)
- Uses monospace for column alignment
- Preserves rows and separator lines

**Chemical/Scientific**: Formulas, reactions, equations
- Detects: Chemical symbols, arrows (→←⇌), subscripts
- Uses monospace for proper spacing
- Examples: `H₂O + CO₂ → ...`, `E = mc²`

### **Why This Matters**

- ✅ **Dialog/QA turns preserved** (training data structure maintained)
- ✅ **Code preserves indentation** (not reflowed)
- ✅ **Diagrams maintain spatial layout** (flowcharts, trees, tables)
- ✅ **Math formulas don't wrap** (stay on single lines)
- ✅ **Fully automatic** (no manual configuration)
- ✅ **Validation**: 18/18 tests passed (100% success, no truncation)

### **Examples**

| Content | Detected | Canvas | Notes |
|---------|----------|--------|-------|
| "Hello world!" | plain | 128×64 | Minimum size ✓ |
| Q&A dialog | **dialog** | 288×96 | Turns preserved ✓ |
| Python function | **code** | 288×160 | Indentation preserved ✓ |
| File tree | **tree** | 224×256 | Structure preserved ✓ |
| Data table | **table** | 288×160 | Columns aligned ✓ |
| Flowchart with → | **diagram** | 224×256 | Spatial layout ✓ |
| LaTeX formulas | **latex** | 448×160 | No mid-formula wrapping ✓ |

**Test**: `python Qwen/demo_adaptive_renderer.py` or `python Qwen/test_final_validation.py`

## Future Optimization Options (Reference)

The following are documented for future reference if higher quality rendering is needed. Currently, the default configuration is sufficient.

### Option 1: Target Readable Font Size ⭐ **Recommended**

**Current**: Calculates space based on minimum font (8pt), not optimal readability.
**Proposed**: Use a target readable font size (14-16pt) instead of minimum.

```python
def calculate_adaptive_dimensions(
    text: str,
    target_font_size: float = 14.0,    # NEW: Target readable font
    min_font_size: float = 8.0,        # Fallback for very long text
    ...
):
    # Use target_font_size for calculation instead of min
    char_width = target_font_size * 0.58
    line_height = target_font_size * 1.5
```

**Expected Impact**: ~75% increase in resolution (14/8 = 1.75×), significantly more readable.

---

### Option 2: Layout-Aware Safety Multipliers

**Current**: All content types use the same 1.5× safety multiplier.
**Proposed**: Different safety multipliers for different layout types.

```python
LAYOUT_SAFETY_MULTIPLIERS = {
    'diagram': 2.5,      # Critical: need detail
    'table': 2.0,        # Column alignment matters
    'code': 2.0,         # Indentation readability
    'latex': 1.8,        # Formula clarity
    'markdown': 1.5,     # Current
    'plain': 1.3,        # Can be tighter
}

# In calculate_adaptive_dimensions:
safety_multiplier = LAYOUT_SAFETY_MULTIPLIERS.get(
    layout_info['layout_type'],
    default_safety_multiplier
)
```

**Expected Impact**: 30-70% higher resolution for structured content, more efficient for plain text.

---

### Option 3: Character Count-Based Scaling

**Current**: Short and long text use the same calculation approach.
**Proposed**: Adaptive scaling based on text length.

```python
def calculate_adaptive_dimensions(...):
    num_chars = len(text)

    # Short text gets higher quality (can afford more pixels)
    if num_chars < 100:
        effective_safety = 2.0  # High quality
    elif num_chars < 500:
        effective_safety = 1.5  # Balanced
    else:
        effective_safety = 1.2  # Efficiency for long text
```

**Expected Impact**: Better quality for short text, better efficiency for long text.

---

### Option 4: Minimum Token Density

**Current**: No consideration of Qwen3VL's token granularity (32×32 patches).
**Proposed**: Ensure sufficient pixels per token.

```python
MIN_PIXELS_PER_TOKEN = 6  # Target 6px per token side

def calculate_adaptive_dimensions(...):
    # After calculating dimensions
    width_tokens = width // 32
    height_tokens = height // 32

    # Estimate character density
    chars_per_token = num_chars / (width_tokens * height_tokens)

    # If too dense, increase resolution
    if chars_per_token > 2.0:  # More than 2 chars per token
        scale_factor = min(1.5, chars_per_token / 2.0)
        width = int(width * scale_factor)
        height = int(height * scale_factor)
```

**Expected Impact**: Prevents overcrowding, ensures vision model has sufficient detail.

---

### Option 5: Configurable Quality Presets

**Current**: Single configuration for all use cases.
**Proposed**: Add quality modes.

```python
class AdaptiveVelloRenderer:
    QUALITY_PRESETS = {
        'draft': {'target_font': 10.0, 'safety': 1.2},
        'balanced': {'target_font': 14.0, 'safety': 1.5},  # Default
        'high': {'target_font': 18.0, 'safety': 2.0},
        'ultra': {'target_font': 24.0, 'safety': 2.5},
    }

    def __init__(self, quality: str = 'balanced', ...):
        preset = self.QUALITY_PRESETS[quality]
        self.target_font_size = preset['target_font']
        self.safety_multiplier = preset['safety']
```

**Expected Impact**: Flexibility to trade quality vs. speed based on use case.

---

### Option 6: Combined Layout-Aware Target Fonts (Best Overall)

**Recommended Implementation**: Combine Option 1 + Option 2

```python
# Default to readable 14pt font
target_font_size = 14.0

# Adjust based on layout type
LAYOUT_TARGET_FONTS = {
    'diagram': 16.0,   # Need detail
    'code': 15.0,      # Readable syntax
    'table': 15.0,     # Column alignment
    'latex': 16.0,     # Formula clarity
    'plain': 14.0,     # Standard
    'paragraph': 13.0, # Can be smaller
}

target_font_size = LAYOUT_TARGET_FONTS.get(
    layout_info['layout_type'],
    target_font_size
)
```

**Expected Impact**: 40-100% higher resolution where it matters most, efficient for prose.

---

## Notes

1. **No dataset rebuilds needed** - Uses existing OCRVL data
2. **Images are reusable** - Source images not duplicated
3. **Only Qwen-specific renders cached** in Qwen/data/cache/
4. **Latent features pre-computed** for thinking training
5. **Three-stage progressive training**: Foundation → OCR supplement → Thinking
