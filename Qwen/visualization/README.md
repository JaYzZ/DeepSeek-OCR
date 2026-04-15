# Qwen3-VL Visualization

Interactive FastAPI app for tracing Qwen3-VL generations from the current Qwen training flow.

Current implementation state:

- backend inference is driven by [app.py](app.py) with Hugging Face generation and tracing
- default base model path is `$ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking`
- optional LoRA loading is supported with `--lora-path`
- the UI still exposes token views, t-SNE projections, and attention inspection

## Quick Start

Install the app dependencies:

```bash
pip install fastapi uvicorn[standard] python-multipart torch transformers scikit-learn plotly numpy pillow pydantic pyyaml safetensors
```

Start the server with the helper script:

```bash
cd $ROOT_DIR/sources/DeepSeek-OCR
Qwen/visualization/run_server.sh \
  --model-path $ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking \
  --port 8501
```

Start it directly with a LoRA checkpoint:

```bash
python -m Qwen.visualization.app \
  --model-path $ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking \
  --lora-path Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_XXXXXX/checkpoint-1000 \
  --port 8501
```

Then open `http://localhost:8501`.

## Main Files

- [app.py](app.py): FastAPI server, model loading, tracing, and API endpoints
- [run_server.sh](run_server.sh): startup wrapper with default model path
- [static/index.html](static/index.html): frontend shell
- [static/main.js](static/main.js): UI behavior and API calls
- [utils/visualization_utils.py](utils/visualization_utils.py): t-SNE and plotting helpers

## API Surface

- `GET /`: serve the UI
- `GET /api/health`: model and server status
- `GET /api/example`: default demo payload
- `POST /api/infer`: run traced generation
- `POST /api/tsne`: project hidden states
- `POST /api/attention`: build attention heatmap data

## Notes

- The visualization package now expects module execution from repo root, for example `python -m Qwen.visualization.app`.
- `--model-path` and `--lora-path` are the current CLI flags; older `--checkpoint` and `--base-model` examples are stale.
- Attention extraction is more expensive than plain generation and may need lower `max_tokens`.
- If a LoRA checkpoint contains `vae.safetensors`, the app will load it automatically.
