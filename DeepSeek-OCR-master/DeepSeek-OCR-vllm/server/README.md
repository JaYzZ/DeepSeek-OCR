# DeepSeek-OCR Server

FastAPI server for DeepSeek-OCR with vLLM backend.

## ⚠️ IMPORTANT: Kill Stuck vLLM Processes

If GPU memory is not released after stopping the server, use this command:

```bash
pkill -9 -f "VLLM::EngineCore"
```

**When to use**: After stopping the OCR server, if `nvidia-smi` still shows GPU memory occupied.
**What it does**: Kills the vLLM engine core process that holds GPU memory.

## Directory Structure

```
server/
├── deepseek_ocr_server.py      # Main FastAPI server
├── standalone_vision_encoder.py # Vision encoder for V1 engine compatibility
├── text_renderer.py             # Text rendering utilities
├── start_server.sh              # Server startup script
├── deepseek-ocr.service         # Systemd service file
├── requirements_server.txt      # Python dependencies
├── Dockerfile                   # Docker container
├── docker-compose.yml           # Docker Compose config
├── examples/
│   ├── ocr_client.py            # Python client library
│   ├── example_pdf_ocr.py       # PDF OCR example
│   └── save_pdf_response.py     # PDF response utility
└── test_images/                 # Test images for validation
    ├── test_render_large_font.png           # Large font test (111 tokens)
    ├── test_render_large_font_visual_tokens.*
    ├── test_doc_font32.png                  # Medium density font 32 test
    ├── test_doc_font32_visual_tokens.*
    ├── rendered_dense_6375_chars.png        # Dense text ~6375 chars
    └── rendered_dense_6375_chars_visual_tokens.*
```

## Core Files

### Server Components
- **deepseek_ocr_server.py**: Main FastAPI server with endpoints for OCR, text-to-visual-tokens, and visual-tokens-to-text
- **standalone_vision_encoder.py**: Standalone vision encoder for vLLM V1 compatibility (separate process architecture)
- **text_renderer.py**: Text rendering to image utilities

### Configuration & Deployment
- **start_server.sh**: Quick start script with default parameters
- **deepseek-ocr.service**: Systemd service for production deployment
- **Dockerfile**: Container image for deployment
- **docker-compose.yml**: Multi-container orchestration
- **requirements_server.txt**: Server dependencies

## vLLM Configuration

According to [official vLLM documentation](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html), the following configuration is required for DeepSeek-OCR:

### Required Parameters
- `enable_prefix_caching=False` ✓
- `mm_processor_cache_gb=0` ✓
- `logits_processors=[NGramPerReqLogitsProcessor]` ✓

### Installation
For vLLM 0.11.1+, install from nightly:
```bash
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

### Known Issues
- **vLLM v0.11.1rc7.dev234**: OCR produces repetitive garbage output ("in the World? in the World?...")
- **Status**: Need to update to stable vLLM v0.11.1 release or newer nightly build
- **Workaround**: Server code is validated as correct; issue is in vLLM engine itself

## API Endpoints

### `/health` - Health Check
Server and model status

### `/ocr` - End-to-End OCR
Full OCR processing: image → text (requires full vLLM mode)

### `/text-to-vistok` - Text to Visual Tokens
Convert text to visual tokens (render → encode)
- Supports both `texts` (server-side rendering) and `images` (pre-rendered) inputs
- Output formats: `binary` (default, ~33% smaller) or `json` (base64-encoded)
- Available in **encoder-only mode** and full mode

### `/vistok-to-text` - Visual Tokens to Text
Decode visual tokens back to text using DeepSeek 3B MoE decoder
- Available in **encoder-only mode** (uses standalone decoder) and full mode
- Accepts binary tensor files (multipart/form-data)
- **Important:** Uses `prompt_prefix` parameter to provide instruction context
  - Default: `"<|grounding|>OCR the text in the image."`
  - This instruction tells the model what task to perform
  - Use `''` (empty) for no instruction (may reduce accuracy)

### Encoder-Only Mode

Start server in lightweight mode for text↔visual-token conversion only:
```bash
python deepseek_ocr_server.py --encoder-only --port 8011 --gpu-devices 0
```

**Available endpoints in encoder-only mode:**
- `/health` ✓
- `/text-to-vistok` ✓ (vision encoder)
- `/vistok-to-text` ✓ (uses standalone DeepSeek 3B MoE decoder)

**Unavailable in encoder-only mode:**
- `/ocr`, `/ocr/upload`, `/ocr/batch`, `/ocr/pdf` (require vLLM batching)

## Test Images

Three test images are provided in `test_images/`:

1. **test_render_large_font.png** (10KB)
   - Large font rendering test
   - 111 visual tokens at 640×640 native resolution

2. **test_doc_font32.png** (70KB)
   - Medium density font 32 test
   - 111 visual tokens, ~479 characters expected

3. **rendered_dense_6375_chars.png** (83KB)
   - Dense text rendering
   - 111 visual tokens, ~6375 characters expected

Each image includes `.bin` (binary visual tokens) and `.json` (metadata) files.

## Quick Start

```bash
# Start server on GPU 0, port 8011
./start_server.sh

# Or with custom parameters
python deepseek_ocr_server.py --port 8011 --gpu-devices 0
```

## References

- [vLLM DeepSeek-OCR Documentation](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- [DeepSeek-OCR on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
