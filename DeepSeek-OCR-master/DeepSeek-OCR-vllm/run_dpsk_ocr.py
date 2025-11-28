from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image
import os
import torch

# Handle CUDA 12.8 specific configuration
if torch.version.cuda == '12.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-12.8/bin/ptxas"

# GPU device selection - can be overridden via environment variable
# Example: CUDA_VISIBLE_DEVICES=0 python run_dpsk_ocr.py
from gpu_manager import select_devices
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    # Auto-select first available GPU if not specified
    select_devices(None, auto_select=True, set_env=True)

# Create model instance
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor]
)

# Prepare batched input with your image file
image_1 = Image.open("/home/jianzhan/sources/DeepSeek-OCR/image_1.png").convert("RGB")
image_2 = Image.open("/home/jianzhan/sources/DeepSeek-OCR/image_2.png").convert("RGB")
prompt = "<image>\nFree OCR."

model_input = [
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_1}
    },
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_2}
    }
]

sampling_param = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            # ngram logit processor args
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
            ),
            skip_special_tokens=False,
        )
# Generate output
model_outputs = llm.generate(model_input, sampling_param)

# Print output
for output in model_outputs:
    print(output.outputs[0].text)