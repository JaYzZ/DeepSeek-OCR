# OCRInfer Configuration

from OCRInfer.utils.model_paths import resolve_model_path

# Model path (uses local mirror if present)
MODEL_ID = 'deepseek-ai/DeepSeek-OCR'
MODEL_PATH = resolve_model_path(MODEL_ID)

# Resolution configuration (640×640 for single-tile)
BASE_SIZE = 640
IMAGE_SIZE = 640

# Cropping configuration
CROP_MODE = False  # Disable cropping for single-tile rendering
MIN_CROPS = 2
MAX_CROPS = 6  # max:9; If your GPU memory is small, set to 6

# Visual token format
NUM_VISUAL_TOKENS = 111  # 100 grid + 10 newlines + 1 view_separator

# Device configuration
DEFAULT_DEVICE = 'cuda'
DEFAULT_DTYPE = 'bfloat16'

# Decoder configuration
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_GPU_MEMORY_UTILIZATION = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NGRAM_SIZE = 30
DEFAULT_WINDOW_SIZE = 90

# Prompt and tokenizer
PROMPT = '<image>\nFree OCR.'  # Default OCR prompt

# Initialize tokenizer (needed by image processor)
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
