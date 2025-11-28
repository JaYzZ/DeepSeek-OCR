# ============================================================================
# GPU Device Configuration
# ============================================================================
# GPU device selection (auto-detected if None)
# Examples:
#   GPU_DEVICES = None          # Auto-select first available GPU
#   GPU_DEVICES = 0             # Use GPU 0
#   GPU_DEVICES = "0,1"         # Use GPUs 0 and 1
#   GPU_DEVICES = "all"         # Use all available GPUs
#   GPU_DEVICES = [0, 1, 2]     # Use GPUs 0, 1, and 2
GPU_DEVICES = None  # Auto-select first available GPU

# ============================================================================
# Resolution Modes
# ============================================================================
# Supported modes:
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False  ← ACTIVE
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True

# Current: Small mode (640×640 native, no padding, 111 visual tokens) - DEFAULT
BASE_SIZE = 640
IMAGE_SIZE = 640
CROP_MODE = False
MIN_CROPS= 2
MAX_CROPS= 6 # max:9; If your GPU memory is small, it is recommended to set it to 6.

# ============================================================================
# Processing Configuration
# ============================================================================
MAX_CONCURRENCY = 100 # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64 # image pre-process (resize/padding) workers
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR' # change to your model path

# ============================================================================
# Input/Output Paths
# ============================================================================
# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py;
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py;
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py

PDF_INPUT_PATH = '/home/jianzhan/sources/DeepSeek-OCR/assets/DeepSeek_OCR_paper.pdf'
PDF_OUTPUT_PATH = '/home/jianzhan/sources/DeepSeek-OCR/pdf/outputs'
IMG_INPUT_PATH = 'server/test_images/rendered_dense_6375_chars.png'  # Test image
IMG_OUTPUT_PATH = '/tmp/ocr_test_proper_3'
INPUT_PATH = '/path/to/omnidocbench/images'  # For batch evaluation
OUTPUT_PATH = '/path/to/output'

# ============================================================================
# Prompt Configuration
# ============================================================================
# PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
PROMPT = '<image>\nFree OCR.'  # Simple OCR test
# TODO commonly used prompts
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
# .......

# ============================================================================
# Initialize Components
# ============================================================================
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Initialize GPU device selection
import torch
import os
from gpu_manager import select_devices

# Handle CUDA 12.8 specific configuration
if torch.version.cuda == '12.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-12.8/bin/ptxas"

# Select GPU devices
try:
    SELECTED_GPU_DEVICES = select_devices(GPU_DEVICES, auto_select=True, set_env=True)
    print(f"[Config] Using GPU device(s): {SELECTED_GPU_DEVICES}")
except Exception as e:
    print(f"[Config] Warning: Failed to set GPU devices: {e}")
    SELECTED_GPU_DEVICES = []

