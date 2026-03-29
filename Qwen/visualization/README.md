# Qwen3-VL Thinking Mode Visualization

Interactive web-based visualization tool for exploring Qwen3-VL thinking mode checkpoints with continuous latent autoregressive generation.

## Features

- **Token Visualization**: Display generated tokens with clear distinction between continuous and discrete positions
- **t-SNE Projection**: Visualize hidden states in 2D with color-coding by token type
- **Attention Heatmap**: Interactive exploration of attention patterns (click any token to see its attention)
- **Multimodal Support**: Upload images and ask questions
- **Default Examples**: Pre-loaded deepvision math examples

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn[standard] python-multipart torch vllm transformers scikit-learn plotly numpy pillow pydantic pyyaml
```

### 2. Start the Server

```bash
# Using the startup script
cd Qwen/visualization
./run_server.sh

# Or directly with Python
python app.py \
    --checkpoint Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_sota_1ep/checkpoint_latest \
    --base-model /share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking \
    --port 8501
```

### 3. Open Browser

Navigate to `http://localhost:8501`

## Usage

### Load an Example

1. Click "Load Example" to load a pre-configured deepvision math example
2. The example includes an image and a question
3. Click "Generate" to run inference

### Custom Query

1. Upload an image using the file input (optional)
2. Type your question in the text area
3. Click "Generate" to run inference

### Visualization

Once generation completes:

1. **Token Sequence**: Shows all generated tokens
   - Green highlighted tokens are continuous latent positions
   - Click any token to see its attention pattern

2. **t-SNE Plot**: 2D projection of hidden states
   - Red: Image tokens
   - Blue: Question/prompt tokens
   - Green: Continuous (thinking) tokens
   - Orange: Answer tokens
   - Click points to select tokens

3. **Attention Heatmap**: Shows attention patterns
   - By default, shows full attention matrix
   - Click a token to see its attention to previous positions
   - Darker blue = higher attention weight

4. **Generated Answer**: The final text output

## Architecture

```
Qwen/visualization/
├── app.py                      # FastAPI server with vLLM integration
├── static/
│   ├── index.html              # Main UI
│   ├── styles.css              # Styling
│   └── main.js                 # Frontend logic + Plotly
├── utils/
│   ├── vllm_inference.py       # vLLM inference helpers
│   └── visualization_utils.py  # t-SNE, attention helpers
├── run_server.sh               # Startup script
└── README.md                   # This file
```

## Configuration

### Environment Variables

- `VLLM_MODEL_PATH`: Path to base model
- `VLLM_LORA_CHECKPOINT_PATH`: Path to LoRA checkpoint
- `VLLM_THINKING_MODE`: Enable thinking mode (set to "1")
- `VLLM_THINKING_DEBUG`: Enable debug logging (set to "1")

### Thinking Token IDs (Qwen3-VL)

- `QWEN3VL_THINKING_START_ID`: 151667 (Start of thinking)
- `QWEN3VL_THINKING_END_ID`: 151668 (End of thinking)
- `QWEN3VL_LATENT_TOKEN_ID`: 151669 (Latent token placeholder)
- `QWEN3VL_THINKING_SEP_ID`: 151670 (Thinking separator)

## API Endpoints

### GET `/`
Serve the main HTML interface

### GET `/api/health`
Check server health and model status

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "thinking_plugin_available": true,
  "checkpoint_path": "..."
}
```

### GET `/api/example`
Get default deepvision example

Response:
```json
{
  "image_base64": "...",
  "question": "..."
}
```

### POST `/api/infer`
Run inference with thinking mode

Request:
```json
{
  "text": "What is in this image?",
  "image_base64": null,
  "max_tokens": 2048,
  "temperature": 0.7
}
```

Response:
```json
{
  "answer": "The answer is...",
  "tokens": [...],
  "hidden_states": [...],
  "attention_weights": [...],
  "continuous_mask": [false, false, true, true, ...],
  "tsne_coordinates": [[x1, y1], [x2, y2], ...],
  "token_metadata": {...}
}
```

### POST `/api/tsne`
Compute t-SNE on provided hidden states

### POST `/api/attention`
Generate attention heatmap data

## Troubleshooting

### Model Not Loaded

- Check that the checkpoint path is correct
- Ensure VLLM can access the model files
- Check GPU memory availability

### No Hidden States

- Ensure thinking mode is enabled in the checkpoint
- Check that the vLLM thinking plugin is loaded
- Verify VLLM_THINKING_MODE=1 is set

### No Attention Weights

- Attention weights require model modification
- Not all checkpoints support attention extraction
- The visualization will work without attention

### Memory Issues

- Reduce `max_tokens` in the request
- Use a smaller model variant
- Check GPU utilization with `nvidia-smi`

## Development

### Modifying the UI

- `static/index.html`: HTML structure
- `static/styles.css`: Styling
- `static/main.js`: Frontend logic

### Modifying the Backend

- `app.py`: FastAPI server and inference logic
- `utils/vllm_inference.py`: vLLM integration
- `utils/visualization_utils.py`: Visualization helpers

### Adding New Visualizations

1. Add the visualization function to `visualization_utils.py`
2. Add an API endpoint in `app.py`
3. Add frontend code in `main.js` to call the API and render

## License

Same as the parent DeepSeek-OCR project.
