#!/usr/bin/env python3
"""
DeepSeek-OCR vLLM Server
Production-ready HTTP API server for document OCR processing
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import socket
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from vllm import LLM, SamplingParams

from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from gpu_manager import select_devices
from text_renderer import render_text_to_image, validate_text_chunk, estimate_token_count, chunk_text_by_tokens
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log PDF support status
if not PDF_SUPPORT:
    logger.warning("PyMuPDF not installed. PDF processing disabled.")


# ============================================================================
# Utility Functions
# ============================================================================

def is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    """
    Check if a port is already in use

    Args:
        port: Port number to check
        host: Host address (default: 0.0.0.0)

    Returns:
        True if port is in use, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """
    Find an available port starting from start_port

    Args:
        start_port: Port to start searching from
        max_attempts: Maximum number of ports to try

    Returns:
        Available port number

    Raises:
        RuntimeError: If no available port found
    """
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port

    raise RuntimeError(
        f"Could not find available port in range {start_port}-{start_port + max_attempts - 1}"
    )


# ============================================================================
# Request/Response Models
# ============================================================================

class OCRRequest(BaseModel):
    """OCR request with base64 encoded image"""
    image_base64: str = Field(..., description="Base64 encoded image")
    prompt: str = Field(
        default="<image>\n<|grounding|>Convert the document to markdown.",
        description="OCR prompt template"
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=1, le=32768)
    ngram_size: int = Field(default=30, description="N-gram blocking size")
    window_size: int = Field(default=90, description="N-gram window size")
    output_format: str = Field(
        default="markdown",
        description="Output format: markdown, json, raw"
    )


class OCRBatchRequest(BaseModel):
    """Batch OCR request"""
    images_base64: List[str] = Field(..., description="List of base64 encoded images")
    prompt: str = Field(default="<image>\nFree OCR.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=1, le=32768)
    ngram_size: int = Field(default=30)
    window_size: int = Field(default=90)


class OCRResponse(BaseModel):
    """OCR response"""
    success: bool
    text: str
    metadata: Dict = Field(default_factory=dict)
    error: Optional[str] = None


class OCRBatchResponse(BaseModel):
    """Batch OCR response"""
    success: bool
    results: List[OCRResponse]
    total_processed: int
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    gpu_available: bool
    version: str = "1.0.0"


class PDFResponse(BaseModel):
    """PDF OCR response"""
    success: bool
    total_pages: int
    markdown_text: str = ""
    detection_text: str = ""
    pages: List[Dict] = Field(default_factory=list)
    output_files: Dict[str, Union[str, List[str]]] = Field(default_factory=dict)
    images: Dict[str, str] = Field(default_factory=dict, description="Base64 encoded images {filename: base64_data}")
    error: Optional[str] = None


class TextToVisualTokensRequest(BaseModel):
    """
    Request to convert texts OR images to visual tokens - supports batch processing

    Auto-detection:
    - If 'texts' is provided: Render text server-side to images, then encode
    - If 'images' is provided: Directly encode pre-rendered images
    """
    texts: Optional[List[str]] = Field(None, description="List of texts (each auto-chunked into 1000-token pieces)")
    images: Optional[List[str]] = Field(None, description="List of base64-encoded images to directly encode (skip rendering)")
    chunk_size: int = Field(default=1000, description="Token count per chunk (default: 1000, only used for texts)")
    render_width: int = Field(default=640, description="Rendered image width (only used for texts)")
    render_height: int = Field(default=640, description="Rendered image height (only used for texts)")
    font_size: int = Field(default=18, description="Font size for rendering (only used for texts)")
    include_rendered_images: bool = Field(default=True, description="Include rendered_image_base64 in response (set False to reduce payload)")
    skip_embeddings: bool = Field(default=False, description="Skip visual embeddings, only return rendered images (for vLLM decode path)")
    output_format: str = Field(default="binary", description="Output format: 'binary' (raw bytes, ~33% smaller, recommended) or 'json' (base64 encoded, more portable)")


class TextToVisualTokensResponse(BaseModel):
    """Response with visual tokens - supports batch processing"""
    success: bool
    results: List[Dict] = Field(default_factory=list, description="List of results, one per input text")
    total_texts: int = Field(default=0, description="Number of input texts processed")
    total_chunks: int = Field(default=0, description="Total chunks across all texts")
    total_text_tokens: int = Field(default=0, description="Total text tokens across all inputs")
    total_visual_tokens: int = Field(default=0, description="Total visual tokens generated")
    error: Optional[str] = None
    # Timing breakdown
    timing: Optional[Dict[str, float]] = Field(default=None, description="Timing breakdown: render_time, encode_time, postprocess_time, total_time")


class ChunkVisualTokens(BaseModel):
    """Visual tokens for a single chunk"""
    chunk_index: int
    text_token_count: int
    visual_tokens_base64: str
    embedding_shape: List[int]  # Should be [111, 1280] for 640x640
    rendered_image_base64: Optional[str] = None


class VisualTokensToTextRequest(BaseModel):
    """Request to convert visual tokens/images to text - supports batch processing

    Two modes:
    1. FAST (vLLM batching): Provide rendered_images_base64 - uses vLLM's native batching
    2. EMBEDDINGS: Provide batch_chunks with visual_tokens_base64 - manual autoregressive

    If both provided, rendered_images takes priority for better throughput.
    """
    batch_chunks: Optional[List[List[Dict]]] = Field(None, description="[SLOW] Visual embeddings path. Each chunk has visual_tokens_base64 and embedding_shape")
    rendered_images_base64: Optional[List[str]] = Field(None, description="[FAST] List of base64-encoded rendered images for vLLM batching")
    prompt: str = Field(default="<image>\nOCR the text in the image.", description="Prompt for vLLM generation (used with rendered_images)")
    prompt_prefix: str = Field(default="", description="Optional prompt prefix for embeddings generation")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)


class VisualTokensToTextResponse(BaseModel):
    """Response with generated text from visual tokens - supports batch processing"""
    success: bool
    results: List[Dict] = Field(default_factory=list, description="Decoded results for each text (list of chunks + combined_text)")
    total_texts: int = Field(default=0, description="Number of texts decoded")
    total_chunks: int = Field(default=0, description="Total chunks decoded")
    mode: str = Field(default="embeddings", description="Processing mode: 'vllm_batch' or 'embeddings'")
    error: Optional[str] = None


# ============================================================================
# Utility Functions for PDF Processing
# ============================================================================

def pdf_to_images(pdf_bytes: bytes, dpi: int = 144) -> List[Image.Image]:
    """
    Convert PDF bytes to list of PIL Images

    Args:
        pdf_bytes: PDF file content as bytes
        dpi: Resolution for rendering (default: 144)

    Returns:
        List of PIL Image objects
    """
    if not PDF_SUPPORT:
        raise RuntimeError("PyMuPDF not installed. Install with: pip install PyMuPDF")

    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pixmap.tobytes("png")))
        images.append(img.convert("RGB"))

    pdf_document.close()
    return images


def clean_formula(text: str) -> str:
    """Remove \\quad annotations from LaTeX formulas"""
    formula_pattern = r'\\[\[\(](.*?)\\[\]\)]'

    def process_formula(match):
        formula = match.group(1)
        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        return r'\[' + formula.strip() + r'\]'

    return re.sub(formula_pattern, process_formula, text)


def extract_grounding_tags(text: str) -> tuple:
    """Extract grounding tags from OCR output"""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    images = []
    others = []
    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            images.append(match[0])
        else:
            others.append(match[0])

    return matches, images, others


def extract_image_from_bbox(page_image: Image.Image, bbox_str: str) -> Optional[Image.Image]:
    """
    Extract image region from page based on bounding box coordinates

    Args:
        page_image: PIL Image of the PDF page
        bbox_str: Bounding box string like "[[x1,y1,x2,y2]]"

    Returns:
        Cropped PIL Image or None if invalid bbox
    """
    try:
        # Parse bbox coordinates
        coords = re.findall(r'\d+', bbox_str)
        if len(coords) < 4:
            return None

        x1, y1, x2, y2 = map(int, coords[:4])

        # Ensure coordinates are within image bounds
        width, height = page_image.size
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        # Crop and return image
        if x2 > x1 and y2 > y1:
            return page_image.crop((x1, y1, x2, y2))
        return None
    except Exception as e:
        logger.warning(f"Failed to extract image from bbox {bbox_str}: {e}")
        return None


def clean_ocr_output(
    text: str,
    page_idx: int,
    page_image: Optional[Image.Image] = None,
    output_dir: Optional[Path] = None
) -> tuple:
    """
    Clean OCR output by removing grounding tags and extracting images

    Args:
        text: OCR output text with grounding tags
        page_idx: Page index
        page_image: PIL Image of the page (for extracting images)
        output_dir: Directory to save extracted images

    Returns:
        Tuple of (cleaned_text, list of saved image paths, dict of {filename: base64_data})
    """
    # Clean formulas
    text = clean_formula(text)

    # Extract and process grounding tags
    matches, images, others = extract_grounding_tags(text)

    saved_images = []
    images_base64 = {}

    # Replace image tags with markdown links and extract images
    for idx, img_tag in enumerate(images):
        image_filename = f'images/{page_idx}_{idx}.jpg'
        text = text.replace(img_tag, f'![]({image_filename})\n')

        # Extract and save the image if we have the page image and output dir
        if page_image and output_dir:
            # Parse bbox from tag
            bbox_match = re.search(r'<\|det\|>(.*?)<\|/det\|>', img_tag)
            if bbox_match:
                bbox_str = bbox_match.group(1)
                cropped_img = extract_image_from_bbox(page_image, bbox_str)

                if cropped_img:
                    # Create images directory
                    images_dir = output_dir / "images"
                    images_dir.mkdir(exist_ok=True)

                    # Save image to disk
                    img_path = images_dir / f'{page_idx}_{idx}.jpg'
                    cropped_img.save(img_path, 'JPEG', quality=95)
                    saved_images.append(str(img_path))

                    # Also encode as base64 for transmission
                    img_buffer = io.BytesIO()
                    cropped_img.save(img_buffer, format='JPEG', quality=95)
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                    images_base64[f'{page_idx}_{idx}.jpg'] = img_base64

    # Remove other grounding tags
    for tag in others:
        text = text.replace(tag, '')

    # Clean up extra newlines
    text = text.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
    text = text.replace('<center>', '').replace('</center>', '')

    return text.strip(), saved_images, images_base64


# ============================================================================
# DeepSeek-OCR Server
# ============================================================================

class DeepSeekOCRServer:
    """DeepSeek-OCR vLLM Server"""

    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        gpu_devices: Optional[Union[int, str, List[int]]] = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        encoder_only: bool = False,
    ):
        """
        Initialize DeepSeek-OCR server

        Args:
            model_path: HuggingFace model path or local path
            gpu_devices: GPU device(s) to use (None=auto, int=single GPU,
                        str="0,1" or "all", list=[0,1])
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            max_model_len: Maximum model sequence length
            tensor_parallel_size: Number of GPUs for tensor parallelism
            trust_remote_code: Trust remote code in model
            encoder_only: If True, only load standalone encoder (text-to-visual only, no OCR)
        """
        self.model_path = model_path
        self.encoder_only = encoder_only
        self.llm = None
        self.model = None  # Direct reference to model instance for breakdown endpoints
        self.model_runner = None  # Reference to model runner for V1 compatibility
        self.standalone_encoder = None  # Standalone vision encoder for V1 compatibility

        # Setup CUDA 12.6 environment (matches driver 535.161.08)
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda-12.6')
        if not os.path.exists(cuda_home):
            logger.warning(f"CUDA_HOME path not found: {cuda_home}")
        else:
            os.environ['CUDA_HOME'] = cuda_home
            os.environ['PATH'] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
            os.environ['LD_LIBRARY_PATH'] = f"{cuda_home}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
            logger.info(f"Using CUDA 12.6 from: {cuda_home}")

            # Set Triton PTXAS path for CUDA 12.6
            ptxas_path = os.path.join(cuda_home, 'bin/ptxas')
            if os.path.exists(ptxas_path):
                os.environ['TRITON_PTXAS_PATH'] = ptxas_path
                logger.info(f"Set TRITON_PTXAS_PATH: {ptxas_path}")

        # Set vLLM environment variables for optimal performance
        os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'  # For H100/H200 GPUs

        # Select GPU devices
        logger.info(f"Selecting GPU device(s): {gpu_devices}")
        selected_gpus = select_devices(gpu_devices, auto_select=True, set_env=True)
        self.selected_gpus = selected_gpus  # Store for later use
        logger.info(f"Using GPU device(s): {selected_gpus}")

        self.app = FastAPI(
            title="DeepSeek-OCR API",
            description="Production-ready OCR API powered by DeepSeek-OCR and vLLM",
            version="1.0.0",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes()

        # Initialize model
        logger.info(f"Initializing DeepSeek-OCR model: {model_path}")
        self._initialize_model(
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
        )

    def _initialize_model(
        self,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int,
        trust_remote_code: bool,
    ):
        """Initialize vLLM model with V1 engine and model instance exposure"""
        try:
            # If encoder_only mode, skip vLLM loading and only load standalone encoder
            if self.encoder_only:
                logger.info("⚡ Encoder-only mode: Skipping vLLM model loading")
                logger.info("Initializing standalone vision encoder only...")

                try:
                    from server.standalone_vision_encoder import create_vision_encoder

                    # Use the first selected GPU device instead of defaulting to cuda:0
                    if torch.cuda.is_available() and self.selected_gpus:
                        device = f"cuda:{self.selected_gpus[0]}"
                    else:
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    dtype = torch.bfloat16

                    self.standalone_encoder = create_vision_encoder(
                        model_path=self.model_path,
                        device=device,
                        dtype=dtype
                    )
                    logger.info("✓ Standalone vision encoder initialized successfully")
                    logger.info("⚡ Encoder-only mode ready - text-to-visual endpoint available")
                    return
                except Exception as e:
                    logger.error(f"Failed to initialize standalone encoder in encoder-only mode: {e}")
                    raise

            # Full mode: Initialize vLLM + standalone encoder
            # Initialize with V1 engine (default in vLLM 0.11+)
            # Model instance will be accessed via engine_core path
            self.llm = LLM(
                model=self.model_path,
                enable_prefix_caching=False,
                mm_processor_cache_gb=0,
                logits_processors=[NGramPerReqLogitsProcessor],
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
            )
            logger.info("Model initialized successfully with V1 engine")

            # Verify we can access the model instance and store references
            model = self._get_model_instance()
            if model is not None:
                self.model = model  # Store direct reference
                logger.info("✓ Model instance successfully exposed for breakdown API endpoints")

                # Also store model runner reference for V1
                try:
                    engine = self.llm.llm_engine
                    if hasattr(engine, 'engine_core'):
                        core = engine.engine_core
                        if hasattr(core, 'model_executor') and hasattr(core.model_executor, 'driver_worker'):
                            driver_worker = core.model_executor.driver_worker
                            worker = driver_worker.worker if hasattr(driver_worker, 'worker') else driver_worker
                            if hasattr(worker, 'model_runner'):
                                self.model_runner = worker.model_runner
                                logger.info("✓ Model runner reference stored for V1 optimization")
                except Exception as e:
                    logger.debug(f"Could not store model runner reference: {e}")
            else:
                logger.warning("⚠ Model instance not accessible - breakdown endpoints may not work")
                logger.warning("⚠ Try setting VLLM_USE_V1=0 environment variable to force V0 engine")

            # Initialize standalone vision encoder ONLY if model access is not available
            # This avoids loading duplicate model copies
            if self.model is None:
                logger.info("Model not directly accessible (V1 engine), initializing standalone vision encoder...")
                try:
                    from server.standalone_vision_encoder import create_vision_encoder

                    # Get device from selected GPU
                    if torch.cuda.is_available() and self.selected_gpus:
                        device = f"cuda:{self.selected_gpus[0]}"
                    else:
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    dtype = torch.bfloat16

                    self.standalone_encoder = create_vision_encoder(
                        model_path=self.model_path,
                        device=str(device),
                        dtype=dtype
                    )
                    logger.info("✓ Standalone vision encoder initialized successfully")
                except Exception as e:
                    logger.warning(f"⚠ Could not initialize standalone encoder: {e}")
                    logger.warning("⚠ Visual encoding may not work")
            else:
                logger.info("✓ Using vLLM model's vision encoder directly (V0 engine) - no duplicate model load")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def _get_model_instance(self):
        """
        Get the underlying model instance from vLLM V1 engine

        V1 Engine Structure (vLLM 0.11+):
        llm.llm_engine.core_engine.model_executor.driver_worker.model_runner.model

        Returns:
            The DeepseekOCRForCausalLM model instance
        """
        try:
            if not hasattr(self.llm, 'llm_engine'):
                logger.error("LLM instance does not have 'llm_engine' attribute")
                return None

            engine = self.llm.llm_engine

            # V1 Engine Structure (vLLM 0.11+) - Try multiple possible paths
            # Path 1: engine → core_engine (singular)
            if hasattr(engine, 'core_engine'):
                logger.debug("Found V1 engine structure (core_engine)")
                core = engine.core_engine
                model = self._try_access_model_from_core(core, "V1 core_engine")
                if model:
                    return model

            # Path 2: engine → core_engines (plural, try first one)
            if hasattr(engine, 'core_engines') and len(engine.core_engines) > 0:
                logger.debug("Found V1 engine structure (core_engines)")
                core = engine.core_engines[0]
                model = self._try_access_model_from_core(core, "V1 core_engines[0]")
                if model:
                    return model

            # Fallback: V0 Engine Structure (vLLM < 0.11)
            # Path: engine → model_executor → driver_worker → model_runner → model
            if hasattr(engine, 'model_executor'):
                logger.debug("Found V0 engine structure (model_executor)")
                executor = engine.model_executor

                if hasattr(executor, 'driver_worker'):
                    worker = executor.driver_worker

                    if hasattr(worker, 'model_runner'):
                        runner = worker.model_runner

                        if hasattr(runner, 'model'):
                            model = runner.model
                            logger.info(f"✓ Successfully accessed model via V0 engine: {type(model).__name__}")
                            return model

            logger.error("Could not access model instance - all paths exhausted")
            logger.debug(f"Engine attributes: {dir(engine)}")
            return None

        except Exception as e:
            logger.error(f"Exception while accessing model instance: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _try_access_model_from_core(self, core, path_name: str):
        """Try to access model from an engine core object"""
        try:
            if hasattr(core, 'model_executor'):
                executor = core.model_executor
                logger.debug(f"[{path_name}] Model executor type: {type(executor).__name__}")

                # For single GPU or tensor parallel
                if hasattr(executor, 'driver_worker'):
                    driver_worker = executor.driver_worker
                    logger.debug(f"[{path_name}] Driver worker type: {type(driver_worker).__name__}")

                    # V1 may wrap worker in an extra layer
                    if hasattr(driver_worker, 'worker'):
                        worker = driver_worker.worker
                    else:
                        worker = driver_worker

                    if hasattr(worker, 'model_runner'):
                        runner = worker.model_runner
                        logger.debug(f"[{path_name}] Model runner type: {type(runner).__name__}")

                        if hasattr(runner, 'model'):
                            model = runner.model
                            logger.info(f"✓ Successfully accessed model via {path_name}: {type(model).__name__}")
                            return model
                        else:
                            logger.error(f"[{path_name}] Model runner does not have 'model' attribute")
                    else:
                        logger.error(f"[{path_name}] Worker does not have 'model_runner' attribute. Available: {dir(worker)}")
                else:
                    logger.error(f"[{path_name}] Model executor does not have 'driver_worker' attribute. Available: {dir(executor)}")
            else:
                logger.error(f"[{path_name}] Core does not have 'model_executor' attribute. Available: {dir(core)}")
        except Exception as e:
            logger.error(f"[{path_name}] Exception: {e}")
        return None

    def _execute_in_worker(self, func_name: str, *args, **kwargs):
        """
        Execute a function in the worker process (V1-compatible)

        This method allows executing custom operations in the worker's process space,
        which is necessary for V1 engine where the model is in a separate process.

        Args:
            func_name: Name of the function to execute in worker
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Result from the worker execution
        """
        try:
            engine = self.llm.llm_engine

            # V1 Engine - use engine_core
            if hasattr(engine, 'engine_core'):
                core = engine.engine_core

                # Try to execute through model executor
                if hasattr(core, 'model_executor'):
                    executor = core.model_executor

                    # For V1, we need to execute the operation in the worker process
                    # This is a workaround since V1 isolates the model in a separate process
                    if hasattr(executor, 'driver_worker'):
                        driver_worker = executor.driver_worker

                        # Check if worker has execute_method capability
                        if hasattr(driver_worker, 'execute_method'):
                            result = driver_worker.execute_method(func_name, *args, **kwargs)
                            return result

                        # Fallback: Try to access the actual worker
                        worker = driver_worker.worker if hasattr(driver_worker, 'worker') else driver_worker

                        if hasattr(worker, func_name):
                            method = getattr(worker, func_name)
                            result = method(*args, **kwargs)
                            return result

            logger.error(f"Could not execute {func_name} in worker process")
            return None

        except Exception as e:
            logger.error(f"Error executing {func_name} in worker: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_visual_embeddings(self, image: Image.Image, native_resolution: bool = True) -> Optional[torch.Tensor]:
        """
        Extract visual embeddings from an image using the vision encoders

        V1-Compatible: Uses stored model reference from initialization

        Args:
            image: PIL Image
            native_resolution: If True, process at native resolution without resizing (default: True)

        Returns:
            Visual embeddings tensor or None if failed
        """
        try:
            # Use stored model reference (V1-compatible)
            model = self.model
            if model is None:
                # Fallback: Try to get model instance
                model = self._get_model_instance()
                if model is None:
                    raise RuntimeError("Could not access model instance. Model not exposed in V1 engine.")

            # Process image through the standard pipeline to get visual inputs
            from process.image_process import DeepseekOCRProcessor

            processor = DeepseekOCRProcessor()

            # For 640x640 native resolution, disable cropping
            # Image will be processed at its native size
            processed = processor.tokenize_with_images(
                images=[image],
                bos=False,
                eos=False,
                cropping=False  # Always disable cropping for text rendering
            )

            # Extract the visual components
            pixel_values = processed.get('pixel_values')
            images_crop = processed.get('images_crop')
            images_spatial_crop = processed.get('images_spatial_crop')

            if pixel_values is None:
                raise RuntimeError("Failed to process image")

            # Convert to tensors and move to GPU
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype

            pixel_values = torch.tensor(pixel_values).to(device=device, dtype=dtype)
            images_spatial_crop = torch.tensor(images_spatial_crop).to(device=device, dtype=torch.long)

            # Process crops
            if images_crop and len(images_crop) > 0 and images_crop[0] is not None:
                images_crop_tensor = torch.tensor(images_crop).to(device=device, dtype=dtype)
            else:
                # No crops - create zero tensor
                images_crop_tensor = torch.zeros((1, 1, 1, 3, 64, 64)).to(device=device, dtype=dtype)

            # Extract visual embeddings using the model's vision encoders
            with torch.no_grad():
                vision_embeddings = model._pixel_values_to_embedding(
                    pixel_values=pixel_values,
                    images_crop=images_crop_tensor,
                    images_spatial_crop=images_spatial_crop
                )

            # vision_embeddings is a list of tensors, get the first one
            if isinstance(vision_embeddings, list) and len(vision_embeddings) > 0:
                embeddings = vision_embeddings[0]  # Shape: [seq_len, hidden_dim]
                return embeddings
            else:
                raise RuntimeError("No visual embeddings returned")

        except Exception as e:
            logger.error(f"Error extracting visual embeddings: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_visual_embeddings_batch(self, images: List[Image.Image], native_resolution: bool = True) -> Optional[List[torch.Tensor]]:
        """
        Extract visual embeddings from multiple images in batch (GPU parallel processing)

        V1-Compatible: Uses standalone encoder if available, otherwise falls back to direct model access

        Args:
            images: List of PIL Images
            native_resolution: If True, process at native resolution without resizing (default: True)

        Returns:
            List of visual embeddings tensors or None if failed
        """
        try:
            if not images:
                return []

            # Prefer standalone encoder (V1-compatible), fallback to direct model access
            use_standalone = self.standalone_encoder is not None

            if use_standalone:
                logger.debug(f"Using standalone encoder for {len(images)} images")
                # Use standalone vision encoder
                # For 640x640 rendered text images, return full spatial features [111, 1280]
                embeddings_list = self.standalone_encoder.encode_images(
                    images=images,
                    return_global=False,  # Don't return CLS token only
                    return_local=True     # Return full spatial patch features [111, 1280]
                )
                return embeddings_list
            else:
                logger.debug(f"Using direct model access for {len(images)} images")
                # Fallback: Use direct model access (original implementation)
                model = self.model
                if model is None:
                    # Try to get model instance
                    model = self._get_model_instance()
                    if model is None:
                        raise RuntimeError("Could not access model instance. Model not exposed in V1 engine.")

                # Process images through the standard pipeline to get visual inputs
                from process.image_process import DeepseekOCRProcessor

                processor = DeepseekOCRProcessor()

                # Process all images at once
                processed = processor.tokenize_with_images(
                    images=images,
                    bos=False,
                    eos=False,
                    cropping=False  # Always disable cropping for text rendering
                )

                # Extract the visual components
                pixel_values = processed.get('pixel_values')
                images_crop = processed.get('images_crop')
                images_spatial_crop = processed.get('images_spatial_crop')

                if pixel_values is None:
                    raise RuntimeError("Failed to process images")

                # Convert to tensors and move to GPU
                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype

                pixel_values = torch.tensor(pixel_values).to(device=device, dtype=dtype)
                images_spatial_crop = torch.tensor(images_spatial_crop).to(device=device, dtype=torch.long)

                # Process crops
                if images_crop and len(images_crop) > 0 and images_crop[0] is not None:
                    images_crop_tensor = torch.tensor(images_crop).to(device=device, dtype=dtype)
                else:
                    # No crops - create zero tensor with correct batch size
                    batch_size = len(images)
                    images_crop_tensor = torch.zeros((batch_size, 1, 1, 3, 64, 64)).to(device=device, dtype=dtype)

                # Extract visual embeddings using the model's vision encoders (BATCH PROCESSING!)
                with torch.no_grad():
                    vision_embeddings = model._pixel_values_to_embedding(
                        pixel_values=pixel_values,
                        images_crop=images_crop_tensor,
                        images_spatial_crop=images_spatial_crop
                    )

                # vision_embeddings is a list of tensors, one per image
                if isinstance(vision_embeddings, list) and len(vision_embeddings) > 0:
                    return vision_embeddings  # List of [seq_len, hidden_dim] tensors
                else:
                    raise RuntimeError("No visual embeddings returned")

        except Exception as e:
            logger.error(f"Error extracting visual embeddings in batch: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _decode_visual_embeddings(
        self,
        visual_embeddings: torch.Tensor,
        prompt_prefix: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        ngram_size: int = 30,
        window_size: int = 90,
    ) -> Optional[str]:
        """
        Decode visual embeddings to text using the language model

        V1-Compatible: Uses standalone encoder's model or stored model reference

        IMPORTANT: Uses same generation parameters as /ocr for consistency!

        Args:
            visual_embeddings: Visual embedding tensor [111, 1280] for 640x640 native
            prompt_prefix: Optional text prefix
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (~1000 expected per chunk)
            ngram_size: N-gram size for repetition blocking (default: 30, same as /ocr)
            window_size: Window size for n-gram blocking (default: 90, same as /ocr)

        Returns:
            Generated text or None if failed
        """
        try:
            # Use stored model reference or standalone encoder's model (V1-compatible)
            model = self.model
            if model is None:
                # Fallback: Try to get model instance
                model = self._get_model_instance()
                if model is None:
                    # Second fallback: Use standalone encoder's model (which is the full DeepSeek-OCR model)
                    if self.standalone_encoder is not None and hasattr(self.standalone_encoder, 'model'):
                        model = self.standalone_encoder.model
                        logger.info("Using standalone encoder's model for text generation")
                    else:
                        raise RuntimeError("Could not access model instance. Model not exposed in V1 engine.")

            # Get tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

            # Tokenize prompt prefix if provided
            if prompt_prefix:
                prompt_ids = tokenizer.encode(prompt_prefix, add_special_tokens=True)
            else:
                # Just use BOS token (note: bos_token_id can be 0, so check for None)
                prompt_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else [0]

            # Get device and dtype from MODEL (not embeddings, which may be on wrong device)
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype

            # Move visual embeddings to model's device
            visual_embeddings = visual_embeddings.to(device=model_device, dtype=model_dtype)

            # Get text embeddings for prompt
            input_ids = torch.tensor([prompt_ids]).to(device=model_device)

            with torch.no_grad():
                # Handle different model structures (vLLM vs standalone encoder)
                # DeepseekOCRForCausalLM structure: model.model.embed_tokens, model.model.layers, model.lm_head
                if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                    # Standalone encoder / HuggingFace model structure
                    embed_layer = model.model.embed_tokens
                elif hasattr(model, 'language_model'):
                    # vLLM model structure
                    lm = model.language_model
                    if hasattr(lm, 'get_input_embeddings'):
                        embed_layer = lm.get_input_embeddings()
                    elif hasattr(lm, 'embed_tokens'):
                        embed_layer = lm.embed_tokens
                    else:
                        raise RuntimeError(f"Cannot find embedding layer in language model: {type(lm)}")
                else:
                    raise RuntimeError(f"Cannot find embedding layer in model: {type(model)}")

                # Get input embeddings
                text_embeddings = embed_layer(input_ids)  # [1, prompt_len, hidden_dim]

            # visual_embeddings: [111, 1280] -> [1, 111, 1280]
            visual_embeddings_batched = visual_embeddings.unsqueeze(0)

            # Combine: [1, prompt_len + 111, 1280]
            combined_embeddings = torch.cat([text_embeddings, visual_embeddings_batched], dim=1)

            # Generate text from embeddings using autoregressive generation
            logger.info(f"Generating text from visual embeddings, shape: {combined_embeddings.shape}")
            generated_ids = self._generate_from_embeddings(
                model=model,
                input_embeddings=combined_embeddings,  # [1, seq_len, hidden_dim]
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                temperature=temperature,
                ngram_size=ngram_size,
                window_size=window_size,
                whitelist_token_ids={128821, 128822},  # <td>, </td> - same as /ocr
                device=model_device
            )
            logger.info(f"Generated {len(generated_ids)} tokens: {generated_ids[:10] if generated_ids else []}")

            # Decode generated tokens to text
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text

        except Exception as e:
            logger.error(f"Error decoding visual embeddings: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_from_embeddings(
        self,
        model,
        input_embeddings: torch.Tensor,
        tokenizer,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        ngram_size: int = 30,
        window_size: int = 90,
        whitelist_token_ids: set = None,
        device: str = "cuda"
    ) -> List[int]:
        """
        Generate text from input embeddings using manual autoregressive generation.

        Uses a compatibility patch for LlamaAttention to work with transformers 4.56+
        which changed the forward() signature to require position_embeddings.

        IMPORTANT: Uses same n-gram blocking as vLLM OCR path for consistency!

        Args:
            model: The DeepSeek-OCR model (DeepseekOCRForCausalLM)
            input_embeddings: Input embeddings [1, seq_len, hidden_dim]
            tokenizer: Tokenizer for decoding
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            ngram_size: N-gram size for repetition blocking (default: 30, same as /ocr)
            window_size: Window size for n-gram blocking (default: 90, same as /ocr)
            whitelist_token_ids: Set of token IDs to allow despite n-gram blocking (e.g., {128821, 128822} for <td>, </td>)
            device: Device to run on

        Returns:
            List of generated token IDs
        """
        import torch.nn.functional as F

        # Apply LlamaAttention compatibility patch for transformers 4.56+
        self._patch_llama_attention()

        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
        whitelist = whitelist_token_ids if whitelist_token_ids is not None else {128821, 128822}  # Default: <td>, </td>

        with torch.no_grad():
            embed_layer = model.model.embed_tokens
            current_embeds = input_embeddings.clone()
            generated_ids = []

            for step in range(max_tokens):
                seq_len = current_embeds.shape[1]
                dummy_ids = torch.zeros((1, seq_len), dtype=torch.long, device=current_embeds.device)
                dummy_imgs = [torch.zeros((2, 2), device=current_embeds.device)]

                try:
                    out = model(
                        input_ids=dummy_ids,
                        inputs_embeds=current_embeds,
                        images=dummy_imgs,
                        use_cache=False,
                        return_dict=True,
                    )

                    logits = out.logits[:, -1, :]  # [1, vocab_size]

                    # Apply n-gram blocking (same as vLLM /ocr path)
                    logits = self._apply_ngram_blocking(
                        logits=logits,
                        generated_ids=generated_ids,
                        ngram_size=ngram_size,
                        window_size=window_size,
                        whitelist_token_ids=whitelist
                    )

                    # Sample or greedy decode
                    if temperature <= 0.0:
                        next_token_id = torch.argmax(logits, dim=-1).item()
                    else:
                        probs = F.softmax(logits / temperature, dim=-1)
                        next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1).item()

                    generated_ids.append(next_token_id)

                    if next_token_id == eos_token_id:
                        logger.debug(f"Hit EOS at step {step}")
                        break

                    # Append next token embedding
                    next_embed = embed_layer(torch.tensor([[next_token_id]], device=current_embeds.device))
                    current_embeds = torch.cat([current_embeds, next_embed], dim=1)

                except Exception as e:
                    logger.error(f"Generation error at step {step}: {e}")
                    break

            logger.info(f"Generated {len(generated_ids)} tokens (with n-gram blocking: ngram_size={ngram_size}, window_size={window_size})")
            return generated_ids

    def _apply_ngram_blocking(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        ngram_size: int,
        window_size: int,
        whitelist_token_ids: set
    ) -> torch.Tensor:
        """
        Apply n-gram blocking to prevent repetitive output.

        Implements same logic as vLLM's NGramPerReqLogitsProcessor for consistency.

        Args:
            logits: Current logits [1, vocab_size]
            generated_ids: Previously generated token IDs
            ngram_size: Size of n-grams to block
            window_size: Window size for n-gram checking
            whitelist_token_ids: Token IDs to never block

        Returns:
            Modified logits with blocked n-grams
        """
        if len(generated_ids) < ngram_size - 1:
            return logits  # Not enough tokens yet

        # Get the last (ngram_size - 1) tokens
        context_window_start = max(0, len(generated_ids) - window_size)
        context = generated_ids[context_window_start:]

        if len(context) < ngram_size - 1:
            return logits

        # Find all n-grams in the context
        ngram_prefix = tuple(context[-(ngram_size - 1):])

        # Look for this prefix in the window
        blocked_tokens = set()
        for i in range(len(context) - ngram_size + 1):
            if tuple(context[i:i + ngram_size - 1]) == ngram_prefix:
                # Block the next token (completing this n-gram)
                next_token = context[i + ngram_size - 1]
                if next_token not in whitelist_token_ids:
                    blocked_tokens.add(next_token)

        # Apply blocking
        if blocked_tokens:
            logits = logits.clone()
            for token_id in blocked_tokens:
                logits[0, token_id] = float('-inf')

        return logits

    def _patch_llama_attention(self):
        """Apply compatibility patch for LlamaAttention in transformers 4.56+"""
        if hasattr(self, '_llama_attention_patched') and self._llama_attention_patched:
            return

        try:
            from transformers.models.llama.modeling_llama import LlamaAttention

            if hasattr(LlamaAttention, '_original_forward'):
                self._llama_attention_patched = True
                return

            _original_forward = LlamaAttention.forward

            def _patched_forward(attn_self, hidden_states, position_embeddings=None, attention_mask=None,
                                position_ids=None, past_key_values=None, cache_position=None,
                                output_attentions=False, use_cache=False, past_key_value=None, **kwargs):
                # Convert position_ids to position_embeddings if needed
                if position_embeddings is None and position_ids is not None:
                    if hasattr(attn_self, 'rotary_emb') and attn_self.rotary_emb is not None:
                        cos, sin = attn_self.rotary_emb(hidden_states, position_ids)
                        position_embeddings = (cos, sin)
                    else:
                        seq_len = hidden_states.shape[1]
                        head_dim = getattr(attn_self, 'head_dim', hidden_states.shape[-1] // getattr(attn_self, 'num_heads', 8))
                        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=hidden_states.device).float() / head_dim))
                        t = position_ids.squeeze(0).float()
                        freqs = torch.einsum('i,j->ij', t, inv_freq)
                        emb = torch.cat((freqs, freqs), dim=-1)
                        cos = emb.cos().unsqueeze(0).to(hidden_states.dtype)
                        sin = emb.sin().unsqueeze(0).to(hidden_states.dtype)
                        position_embeddings = (cos, sin)

                result = _original_forward(attn_self, hidden_states=hidden_states, position_embeddings=position_embeddings,
                                          attention_mask=attention_mask, past_key_values=past_key_values,
                                          cache_position=cache_position, **kwargs)

                # Convert new output format (2 values) to old format (3 values)
                if isinstance(result, tuple):
                    return (result[0], result[1] if len(result) > 1 else None, None)
                return (result, None, None)

            LlamaAttention.forward = _patched_forward
            LlamaAttention._original_forward = _original_forward
            self._llama_attention_patched = True
            logger.info("Applied LlamaAttention compatibility patch for transformers 4.56+")

        except Exception as e:
            logger.warning(f"Could not patch LlamaAttention: {e}")

    def _decode_visual_embeddings_batch(
        self,
        visual_embeddings_list: List[torch.Tensor],
        prompt_prefix: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Optional[List[str]]:
        """
        Decode multiple visual embeddings to text using the language model (BATCH PROCESSING)

        V1-Compatible: Uses stored model reference from initialization

        Args:
            visual_embeddings_list: List of visual embedding tensors, each [111, 1280] for 640x640 native
            prompt_prefix: Optional text prefix
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (~1000 expected per chunk)

        Returns:
            List of generated texts or None if failed
        """
        try:
            if not visual_embeddings_list:
                return []

            # Use stored model reference (V1-compatible)
            model = self.model
            if model is None:
                # Fallback: Try to get model instance
                model = self._get_model_instance()
                if model is None:
                    # Second fallback: Use standalone encoder's model (which is the full DeepSeek-OCR model)
                    if self.standalone_encoder is not None and hasattr(self.standalone_encoder, 'model'):
                        model = self.standalone_encoder.model
                        logger.info("Using standalone encoder's model for text generation")
                    else:
                        raise RuntimeError("Could not access model instance. Model not exposed in V1 engine.")

            # Get tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

            # Tokenize prompt prefix if provided
            if prompt_prefix:
                prompt_ids = tokenizer.encode(prompt_prefix, add_special_tokens=True)
            else:
                # Just use BOS token (note: bos_token_id can be 0, so check for None)
                prompt_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else [0]

            # Get device and dtype from MODEL (not embeddings, which may be on wrong device)
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            logger.info(f"Using model device: {model_device}, dtype: {model_dtype}")

            # Prepare batch of input embeddings
            batch_embeddings = []

            for visual_embeddings in visual_embeddings_list:
                # Move visual embeddings to model's device
                visual_embeddings = visual_embeddings.to(device=model_device, dtype=model_dtype)

                # Get text embeddings for prompt
                input_ids = torch.tensor([prompt_ids]).to(device=model_device)

                with torch.no_grad():
                    # Handle different model structures (vLLM vs standalone encoder)
                    # DeepseekOCRForCausalLM structure: model.model.embed_tokens, model.model.layers, model.lm_head
                    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                        # Standalone encoder / HuggingFace model structure
                        embed_layer = model.model.embed_tokens
                    elif hasattr(model, 'language_model'):
                        # vLLM model structure
                        lm = model.language_model
                        if hasattr(lm, 'get_input_embeddings'):
                            embed_layer = lm.get_input_embeddings()
                        elif hasattr(lm, 'embed_tokens'):
                            embed_layer = lm.embed_tokens
                        else:
                            raise RuntimeError(f"Cannot find embedding layer in language model: {type(lm)}")
                    else:
                        raise RuntimeError(f"Cannot find embedding layer in model: {type(model)}")

                    # Get input embeddings
                    text_embeddings = embed_layer(input_ids)  # [1, prompt_len, hidden_dim]

                # visual_embeddings: [111, 1280] -> [1, 111, 1280]
                visual_embeddings_batched = visual_embeddings.unsqueeze(0)

                # Combine: [1, prompt_len + 111, 1280]
                combined_embeddings = torch.cat([text_embeddings, visual_embeddings_batched], dim=1)
                batch_embeddings.append(combined_embeddings)

            # Stack into batch: [batch_size, seq_len, hidden_dim]
            batch_embeddings_tensor = torch.cat(batch_embeddings, dim=0)

            # Generate text from embeddings using autoregressive generation
            # This uses the model's language model directly for generation
            logger.info(f"Starting batch generation from {len(visual_embeddings_list)} visual embedding sets")

            results = []

            # Process each embedding set (can't easily batch due to variable output lengths)
            for idx, combined_emb in enumerate(batch_embeddings):
                try:
                    logger.info(f"Generating text for chunk {idx}, embeddings shape: {combined_emb.shape}")
                    generated_ids = self._generate_from_embeddings(
                        model=model,
                        input_embeddings=combined_emb,  # [1, seq_len, hidden_dim]
                        tokenizer=tokenizer,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        device=model_device
                    )
                    logger.info(f"Generated {len(generated_ids)} tokens: {generated_ids[:10] if generated_ids else []}")

                    # Decode generated tokens to text
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    results.append(generated_text)

                    if (idx + 1) % 10 == 0:
                        logger.info(f"Generated {idx + 1}/{len(visual_embeddings_list)} texts")

                except Exception as e:
                    logger.error(f"Error generating text for chunk {idx}: {e}")
                    results.append(f"[Generation error: {str(e)}]")

            logger.info(f"Batch generation complete: {len(results)} texts generated")
            return results

        except Exception as e:
            logger.error(f"Error decoding visual embeddings in batch: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _register_routes(self):
        """Register API routes"""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy",
                model_loaded=self.llm is not None or self.standalone_encoder is not None,
                gpu_available=torch.cuda.is_available(),
            )

        # Skip OCR endpoints if encoder_only mode
        if self.encoder_only:
            logger.info("⚡ Encoder-only mode: Registering /health, /text-to-vistok, and /vistok-to-text endpoints")
            logger.info("⚡ Standalone encoder includes DeepSeek 3B MoE decoder for full text↔visual-token conversion")
            # Register text-to-visual and visual-to-text endpoints
            # Skip only OCR endpoints (which require vLLM batching)

        # Full OCR endpoints (only in non-encoder-only mode)
        if not self.encoder_only:
            @self.app.post("/ocr", response_model=OCRResponse)
            async def ocr_endpoint(request: OCRRequest):
                """Single image OCR endpoint"""
                try:
                    # Decode base64 image
                    image = self._decode_base64_image(request.image_base64)

                    # Process OCR
                    result = self._process_single_ocr(
                        image=image,
                        prompt=request.prompt,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        ngram_size=request.ngram_size,
                        window_size=request.window_size,
                    )

                    # Format output
                    formatted_result = self._format_output(
                        result, request.output_format
                    )

                    return OCRResponse(
                        success=True,
                        text=formatted_result,
                        metadata={
                            "prompt": request.prompt,
                            "output_format": request.output_format,
                        }
                    )

                except Exception as e:
                    logger.error(f"OCR processing error: {e}")
                    return OCRResponse(
                        success=False,
                        text="",
                        error=str(e)
                    )

            @self.app.post("/ocr/upload", response_model=OCRResponse)
            async def ocr_upload_endpoint(
                file: UploadFile = File(...),
                prompt: str = Form("<image>\n<|grounding|>Convert the document to markdown."),
                temperature: float = Form(0.0),
                max_tokens: int = Form(8192),
                ngram_size: int = Form(30),
                window_size: int = Form(90),
                output_format: str = Form("markdown"),
            ):
                """Upload image file for OCR"""
                try:
                    # Read uploaded file
                    contents = await file.read()
                    image = Image.open(io.BytesIO(contents)).convert("RGB")

                    # Process OCR
                    result = self._process_single_ocr(
                        image=image,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        ngram_size=ngram_size,
                        window_size=window_size,
                    )

                    # Format output
                    formatted_result = self._format_output(result, output_format)

                    return OCRResponse(
                        success=True,
                        text=formatted_result,
                        metadata={
                            "filename": file.filename,
                            "prompt": prompt,
                            "output_format": output_format,
                        }
                    )

                except Exception as e:
                    logger.error(f"Upload OCR error: {e}")
                    return OCRResponse(
                        success=False,
                        text="",
                        error=str(e)
                    )

            @self.app.post("/ocr/batch", response_model=OCRBatchResponse)
            async def ocr_batch_endpoint(request: OCRBatchRequest):
                """Batch OCR endpoint"""
                try:
                    # Decode all images
                    images = [
                        self._decode_base64_image(img_b64)
                        for img_b64 in request.images_base64
                    ]

                    # Process batch
                    results = self._process_batch_ocr(
                        images=images,
                        prompt=request.prompt,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        ngram_size=request.ngram_size,
                        window_size=request.window_size,
                    )

                    # Format responses
                    ocr_responses = [
                        OCRResponse(
                            success=True,
                            text=result,
                            metadata={"batch_index": idx}
                        )
                        for idx, result in enumerate(results)
                    ]

                    return OCRBatchResponse(
                        success=True,
                        results=ocr_responses,
                        total_processed=len(results)
                    )

                except Exception as e:
                    logger.error(f"Batch OCR error: {e}")
                    return OCRBatchResponse(
                        success=False,
                        results=[],
                        total_processed=0,
                        error=str(e)
                    )

            @self.app.post("/ocr/pdf", response_model=PDFResponse)
            async def ocr_pdf_endpoint(
                file: UploadFile = File(...),
                prompt: str = Form("<image>\n<|grounding|>Convert the document to markdown."),
                temperature: float = Form(0.0),
                max_tokens: int = Form(8192),
                ngram_size: int = Form(30),
                window_size: int = Form(90),
                dpi: int = Form(144),
                output_dir: str = Form(None),
            ):
                """Process PDF file and extract text from all pages with full output structure"""
                if not PDF_SUPPORT:
                    return PDFResponse(
                        success=False,
                        total_pages=0,
                        error="PDF support not available. Install PyMuPDF: pip install PyMuPDF"
                    )

                try:
                    # Read PDF file
                    pdf_bytes = await file.read()
                    logger.info(f"Processing PDF: {file.filename}")

                    # Convert PDF to images
                    page_images = pdf_to_images(pdf_bytes, dpi=dpi)
                    logger.info(f"Converted {len(page_images)} pages to images")

                    # Process all pages in batch
                    results = self._process_batch_ocr(
                        images=page_images,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        ngram_size=ngram_size,
                        window_size=window_size,
                    )

                    # Determine output directory with timestamp
                    if output_dir:
                        out_dir = Path(output_dir)
                    else:
                        # Use server's working directory with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        base_name = Path(file.filename).stem
                        out_dir = Path.cwd() / "pdf" / f"{base_name}_{timestamp}_output"

                    out_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Output directory: {out_dir}")

                    # Process results and extract images
                    page_separator = '\n<--- Page Split --->\n'
                    markdown_pages = []
                    detection_pages = []
                    pages_info = []
                    all_extracted_images = []
                    all_images_base64 = {}

                    for idx, content in enumerate(results):
                        # Clean up ending tokens
                        content = content.replace('<｜end of sentence｜>', '').replace('</s>', '').rstrip('<|end|>')

                        # Store detection version (with grounding tags)
                        detection_pages.append(content)

                        # Clean version (without grounding tags) and extract images
                        clean_content, extracted_images, images_base64 = clean_ocr_output(
                            content,
                            idx,
                            page_image=page_images[idx],
                            output_dir=out_dir
                        )
                        markdown_pages.append(clean_content)
                        all_extracted_images.extend(extracted_images)
                        all_images_base64.update(images_base64)

                        # Page info
                        pages_info.append({
                            "page_number": idx + 1,
                            "has_grounding": '<|ref|>' in content,
                            "char_count": len(clean_content),
                            "extracted_images": len(images_base64)
                        })

                    # Combine all pages
                    markdown_text = page_separator.join(markdown_pages)
                    detection_text = page_separator.join(detection_pages)

                    # Save markdown and detection files
                    output_files = {}
                    base_filename = Path(file.filename).stem

                    # Save markdown file
                    md_path = out_dir / f"{base_filename}.mmd"
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(markdown_text)
                    output_files["markdown"] = str(md_path)

                    # Save detection file
                    det_path = out_dir / f"{base_filename}_det.mmd"
                    with open(det_path, "w", encoding="utf-8") as f:
                        f.write(detection_text)
                    output_files["detection"] = str(det_path)

                    # Add extracted images to output files
                    if all_extracted_images:
                        output_files["images"] = all_extracted_images
                        output_files["images_dir"] = str(out_dir / "images")

                    logger.info(f"Saved outputs to: {out_dir}")
                    logger.info(f"Extracted {len(all_extracted_images)} images")
                    logger.info(f"Encoded {len(all_images_base64)} images as base64 for transmission")

                    return PDFResponse(
                        success=True,
                        total_pages=len(page_images),
                        markdown_text=markdown_text,
                        detection_text=detection_text,
                        pages=pages_info,
                        output_files=output_files,
                        images=all_images_base64
                    )

                except Exception as e:
                    logger.error(f"PDF OCR error: {e}")
                    return PDFResponse(
                        success=False,
                        total_pages=0,
                        error=str(e)
                    )

        @self.app.post("/text-to-vistok")
        async def text_to_visual_tokens_endpoint(request: TextToVisualTokensRequest):
            """
            Convert texts OR images to visual tokens (BATCH PROCESSING with auto-detection)

            Auto-detection:
            - If 'texts' provided: Chunk texts → Render to images → Encode to visual tokens
            - If 'images' provided: Directly encode pre-rendered images to visual tokens

            For texts:
            - Automatic chunking: Each text split into ~1000-token chunks
            - Each chunk: 1000 text tokens → 640x640 image → 111 visual tokens

            For images:
            - Directly encode base64 images → 111 visual tokens per image
            - No rendering or chunking needed

            Output formats:
            - 'binary': Returns raw binary bytes (~33% smaller, recommended, metadata in headers)
            - 'json': Returns JSON with base64-encoded tensors (more portable, full metadata)

            Expected output: [111, 1280] visual tokens per chunk/image
            """
            import time
            import base64
            from io import BytesIO

            start_time = time.time()
            timing = {
                "render_time": 0.0,
                "encode_time": 0.0,
                "postprocess_time": 0.0,
                "total_time": 0.0
            }

            try:
                # Auto-detect input type
                if request.texts is not None and len(request.texts) > 0:
                    # TEXT MODE: Render texts to images, then encode
                    logger.info(f"TEXT MODE: Processing {len(request.texts)} texts")

                    # Get tokenizer for accurate token counting
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

                    # Phase 1: Chunk and render all texts
                    render_start = time.time()

                    all_images = []  # All rendered images across all texts
                    image_metadata = []  # Track which text/chunk each image belongs to

                    total_text_tokens = 0
                    total_chunks_count = 0

                    for text_idx, text in enumerate(request.texts):
                        # Count tokens for this text
                        text_token_count = estimate_token_count(text, tokenizer)
                        total_text_tokens += text_token_count

                        logger.info(f"Text {text_idx+1}/{len(request.texts)}: {text_token_count} tokens")

                        # Chunk text into ~1000 token pieces
                        text_chunks = chunk_text_by_tokens(
                            text,
                            chunk_size=request.chunk_size,
                            tokenizer=tokenizer
                        )

                        total_chunks_count += len(text_chunks)
                        logger.info(f"  Split into {len(text_chunks)} chunks")

                        # Render all chunks for this text
                        for chunk_idx, chunk_text in enumerate(text_chunks):
                            chunk_token_count = estimate_token_count(chunk_text, tokenizer)

                            # Render chunk as image
                            rendered_image = render_text_to_image(
                                text=chunk_text,
                                width=request.render_width,
                                height=request.render_height,
                                font_size=request.font_size
                            )

                            all_images.append(rendered_image)
                            image_metadata.append({
                                "text_idx": text_idx,
                                "chunk_idx": chunk_idx,
                                "chunk_text": chunk_text,
                                "chunk_token_count": chunk_token_count,
                                "rendered_image": rendered_image
                            })

                    render_end = time.time()
                    timing["render_time"] = render_end - render_start
                    logger.info(f"Rendering complete: {len(all_images)} total chunks in {timing['render_time']:.3f}s")

                elif request.images is not None and len(request.images) > 0:
                    # IMAGE MODE: Directly encode pre-rendered images
                    logger.info(f"IMAGE MODE: Processing {len(request.images)} pre-rendered images")

                    render_start = time.time()

                    all_images = []
                    image_metadata = []
                    total_text_tokens = 0  # Not applicable for image mode
                    total_chunks_count = len(request.images)

                    for img_idx, img_b64 in enumerate(request.images):
                        # Decode base64 image
                        img_bytes = base64.b64decode(img_b64)
                        img = Image.open(BytesIO(img_bytes))

                        all_images.append(img)
                        image_metadata.append({
                            "text_idx": 0,  # Single group for images
                            "chunk_idx": img_idx,
                            "chunk_text": None,
                            "chunk_token_count": 0,
                            "rendered_image": img
                        })

                    render_end = time.time()
                    timing["render_time"] = render_end - render_start
                    logger.info(f"Image loading complete: {len(all_images)} images in {timing['render_time']:.3f}s")

                else:
                    # Error: Neither texts nor images provided
                    return TextToVisualTokensResponse(
                        success=False,
                        error="Either 'texts' or 'images' must be provided",
                        results=[],
                        total_texts=0,
                        total_chunks=0,
                        total_text_tokens=0,
                        total_visual_tokens=0
                    )

                # Phase 2: Batch encode ALL images at once (GPU parallel processing!)
                # Skip if skip_embeddings=True (for vLLM decode path)
                encode_start = time.time()

                if request.skip_embeddings:
                    logger.info("Skipping visual embeddings (skip_embeddings=True)")
                    all_embeddings = None
                else:
                    all_embeddings = self._extract_visual_embeddings_batch(all_images, native_resolution=True)

                    if all_embeddings is None or len(all_embeddings) != len(all_images):
                        return TextToVisualTokensResponse(
                            success=False,
                            error=f"Failed to extract visual embeddings (expected {len(all_images)}, got {len(all_embeddings) if all_embeddings else 0})",
                            timing=timing
                        )

                encode_end = time.time()
                timing["encode_time"] = encode_end - encode_start

                if all_embeddings:
                    logger.info(f"Batch encoding complete: {len(all_embeddings)} embeddings in {timing['encode_time']:.3f}s")

                # BINARY OUTPUT MODE (default, recommended, ~33% smaller)
                if request.output_format == "binary":
                    if not all_embeddings:
                        return Response(
                            content=json.dumps({"error": "Binary output requires embeddings (skip_embeddings must be False)"}),
                            media_type="application/json",
                            status_code=400
                        )

                    # Concatenate all embeddings into single binary blob
                    # Each embedding: [111, 1280] float32 = 568,320 bytes
                    all_embeddings_np = [emb.cpu().to(torch.float32).numpy() for emb in all_embeddings]

                    # Stack into single array: [num_chunks, 111, 1280]
                    stacked_embeddings = np.stack(all_embeddings_np, axis=0)
                    binary_data = stacked_embeddings.tobytes()

                    total_end = time.time()
                    timing["total_time"] = total_end - start_time

                    num_processed = len(request.texts) if request.texts else len(request.images)
                    logger.info(f"Binary output: {len(binary_data):,} bytes, shape {stacked_embeddings.shape}")
                    logger.info(f"Batch processing complete: {num_processed} inputs, {len(all_embeddings)} chunks")
                    logger.info(f"Timing breakdown - Render: {timing['render_time']:.3f}s, Encode: {timing['encode_time']:.3f}s, Total: {timing['total_time']:.3f}s")

                    # Return binary response with metadata in headers
                    return Response(
                        content=binary_data,
                        media_type="application/octet-stream",
                        headers={
                            "X-Tensor-Shape": json.dumps(list(stacked_embeddings.shape)),
                            "X-Tensor-Dtype": "float32",
                            "X-Total-Texts": str(num_processed),
                            "X-Total-Chunks": str(len(all_embeddings)),
                            "X-Total-Text-Tokens": str(total_text_tokens),
                            "X-Total-Visual-Tokens": str(stacked_embeddings.shape[0] * stacked_embeddings.shape[1]),
                            "X-Render-Time": f"{timing['render_time']:.3f}",
                            "X-Encode-Time": f"{timing['encode_time']:.3f}",
                            "X-Total-Time": f"{timing['total_time']:.3f}",
                        }
                    )

                # JSON OUTPUT MODE (fallback, full metadata)
                # Phase 3: Map embeddings back to their respective texts/chunks
                postprocess_start = time.time()

                # Initialize results structure
                text_results = []
                num_texts = len(request.texts) if request.texts else 1  # 1 for image mode

                # Initialize results structure for each text
                for text_idx in range(num_texts):
                    text_results.append({
                        "text_index": text_idx,
                        "chunks": [],
                        "total_text_tokens": 0,
                        "total_visual_tokens": 0
                    })

                total_visual_tokens = 0

                for img_idx, metadata in enumerate(image_metadata):
                    text_idx = metadata["text_idx"]
                    chunk_idx = metadata["chunk_idx"]

                    chunk_result = {
                        "chunk_index": chunk_idx,
                        "text_token_count": metadata["chunk_token_count"],
                    }

                    # Only include embeddings if not skipped
                    if all_embeddings:
                        embeddings = all_embeddings[img_idx]
                        # Convert bfloat16 to float32 for NumPy compatibility
                        embeddings_np = embeddings.cpu().to(torch.float32).numpy()
                        embeddings_bytes = embeddings_np.tobytes()
                        embeddings_b64 = base64.b64encode(embeddings_bytes).decode('utf-8')
                        chunk_result["visual_tokens_base64"] = embeddings_b64
                        chunk_result["embedding_shape"] = list(embeddings_np.shape)
                        total_visual_tokens += embeddings_np.shape[0]

                    # Only include rendered image if requested
                    if request.include_rendered_images:
                        img_buffer = io.BytesIO()
                        metadata["rendered_image"].save(img_buffer, format='PNG')
                        rendered_image_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                        chunk_result["rendered_image_base64"] = rendered_image_b64

                    text_results[text_idx]["chunks"].append(chunk_result)
                    if all_embeddings:
                        text_results[text_idx]["total_visual_tokens"] += embeddings_np.shape[0]

                # Update text token counts
                if request.texts:
                    for text_idx, text in enumerate(request.texts):
                        text_results[text_idx]["total_text_tokens"] = estimate_token_count(text, tokenizer)

                postprocess_end = time.time()
                timing["postprocess_time"] = postprocess_end - postprocess_start

                total_end = time.time()
                timing["total_time"] = total_end - start_time

                num_processed = len(request.texts) if request.texts else len(request.images)
                logger.info(f"Batch processing complete: {num_processed} inputs, {total_chunks_count} chunks, {total_visual_tokens} visual tokens")
                logger.info(f"Timing breakdown - Render: {timing['render_time']:.3f}s, Encode: {timing['encode_time']:.3f}s, Postprocess: {timing['postprocess_time']:.3f}s, Total: {timing['total_time']:.3f}s")

                return TextToVisualTokensResponse(
                    success=True,
                    results=text_results,
                    total_texts=num_processed,
                    total_chunks=total_chunks_count,
                    total_text_tokens=total_text_tokens,
                    total_visual_tokens=total_visual_tokens,
                    timing=timing
                )

            except Exception as e:
                logger.error(f"Text-to-visual-tokens error: {e}")
                import traceback
                traceback.print_exc()
                return TextToVisualTokensResponse(
                    success=False,
                    error=str(e)
                )

        @self.app.post("/text-to-vistok/upload")
        async def text_to_visual_tokens_upload_endpoint(
            file: UploadFile = File(..., description="Image file to encode to visual tokens"),
            output_format: str = Form(default="binary", description="Output format: 'binary' (default) or 'json'")
        ):
            """
            Upload image file and encode to visual tokens (111 tokens for 640x640 image)

            Default output: Binary format (raw float32 bytes, ~33% smaller, recommended)
            Optional: JSON format (base64-encoded tensors, more portable)
            """
            import time
            import base64

            try:
                start_time = time.time()

                # Read uploaded file
                contents = await file.read()
                image = Image.open(io.BytesIO(contents)).convert("RGB")

                # Encode single image to visual tokens using standalone encoder
                logger.info(f"Encoding uploaded image: {file.filename}")
                encode_start = time.time()

                # Use standalone vision encoder
                embeddings_list = self.standalone_encoder.encode_images([image])  # Returns list
                embeddings = embeddings_list[0]  # Get first tensor [111, 1280]

                encode_time = time.time() - encode_start
                total_time = time.time() - start_time

                # BINARY OUTPUT MODE (default, recommended)
                if output_format.lower() == "binary":
                    # Convert to float32 for consistency
                    embeddings_np = embeddings.cpu().to(torch.float32).numpy()
                    embeddings_bytes = embeddings_np.tobytes()

                    logger.info(f"Binary response: {len(embeddings_bytes)} bytes, shape {embeddings_np.shape}")

                    return Response(
                        content=embeddings_bytes,
                        media_type="application/octet-stream",
                        headers={
                            "X-Tensor-Shape": str(list(embeddings_np.shape)),
                            "X-Tensor-Dtype": "float32",
                            "X-Encode-Time": f"{encode_time:.3f}",
                            "X-Total-Time": f"{total_time:.3f}",
                        }
                    )

                # JSON OUTPUT MODE (fallback)
                else:
                    embeddings_np = embeddings.cpu().to(torch.float32).numpy()
                    embeddings_bytes = embeddings_np.tobytes()
                    embeddings_b64 = base64.b64encode(embeddings_bytes).decode('utf-8')

                    return {
                        "success": True,
                        "visual_tokens_base64": embeddings_b64,
                        "shape": list(embeddings_np.shape),
                        "dtype": "float32",
                        "timing": {
                            "encode_time": encode_time,
                            "total_time": total_time
                        }
                    }

            except Exception as e:
                logger.error(f"Upload visual token encoding error: {e}")
                import traceback
                traceback.print_exc()
                return {"success": False, "error": str(e)}

        # Encoder-only mode: Enable vistok-to-text endpoint (standalone encoder has full model with 3B MoE decoder!)
        # This allows full bidirectional text↔visual-token conversion in lightweight mode
        if self.encoder_only:
            logger.info("⚡ Encoder-only mode: Now registering /vistok-to-text endpoint (standalone decoder available)")

        @self.app.post("/vistok-to-text")
        async def visual_tokens_to_text_endpoint(
            visual_tokens: UploadFile = File(..., description="Raw binary float32 tensor file"),
            shape: str = Form(..., description="Tensor shape as JSON array, e.g., '[1, 111, 1280]'"),
            dtype: str = Form(default="float32", description="Data type (float32 or float16)"),
            prompt_prefix: str = Form(default="Transcribe all text:", description="OCR instruction prompt (default: 'Transcribe all text:' - most stable, use '' for no instruction)"),
            temperature: float = Form(default=0.0, ge=0.0, le=2.0),
            max_tokens: int = Form(default=2048, ge=1, le=32768),
            ngram_size: int = Form(default=30, ge=1, le=100, description="N-gram blocking size (default: 30, same as /ocr)"),
            window_size: int = Form(default=90, ge=1, le=500, description="N-gram window size (default: 90, same as /ocr)")
        ):
            """Convert visual tokens to text using DeepSeek-OCR LLM

            Accepts raw binary float32 tensor via multipart/form-data.
            Expected shape: [num_chunks, seq_len, hidden_dim] e.g., [1, 111, 1280]

            Uses same generation parameters as /ocr for consistency:
            - N-gram blocking to prevent repetition
            - Same whitelist tokens (<td>, </td>)
            """
            try:
                # Parse shape
                shape_list = json.loads(shape)
                logger.info(f"Processing visual tokens: shape {shape_list}, dtype {dtype}")

                # Read binary data
                tensor_bytes = await visual_tokens.read()

                # Convert to numpy array
                if dtype == "float32":
                    tensor_np = np.frombuffer(tensor_bytes, dtype=np.float32)
                elif dtype == "float16":
                    tensor_np = np.frombuffer(tensor_bytes, dtype=np.float16).astype(np.float32)
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported dtype: {dtype}")

                # Reshape
                try:
                    tensor_np = tensor_np.reshape(shape_list)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"Cannot reshape {len(tensor_bytes)} bytes to {shape_list}: {e}")

                # Convert to torch tensor
                embeddings_tensor = torch.from_numpy(tensor_np).to(torch.device('cuda'))

                # Expected shape: [num_chunks, seq_len, hidden_dim]
                if len(embeddings_tensor.shape) != 3:
                    raise HTTPException(status_code=400, detail=f"Invalid shape: {embeddings_tensor.shape}. Expected [num_chunks, seq_len, hidden_dim]")

                num_chunks = embeddings_tensor.shape[0]
                logger.info(f"Decoding {num_chunks} chunks")

                # Decode each chunk
                decoded_chunks = []
                for chunk_idx in range(num_chunks):
                    chunk_emb = embeddings_tensor[chunk_idx]  # [seq_len, hidden_dim]
                    chunk_text = self._decode_visual_embeddings(
                        chunk_emb,
                        prompt_prefix=prompt_prefix,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        ngram_size=ngram_size,
                        window_size=window_size
                    )
                    decoded_chunks.append(chunk_text)
                    logger.info(f"  Chunk {chunk_idx + 1}/{num_chunks}: {len(chunk_text)} chars")

                combined_text = "\n".join(decoded_chunks)

                return JSONResponse({
                    "success": True,
                    "text": combined_text,
                    "chunks": [{"chunk_index": i, "text": t} for i, t in enumerate(decoded_chunks)],
                    "num_chunks": num_chunks
                })

            except Exception as e:
                logger.error(f"Visual-tokens-to-text error: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

    def _decode_base64_image(self, base64_str: str) -> Image.Image:
        """Decode base64 string to PIL Image"""
        try:
            # Remove data URI prefix if present
            if "base64," in base64_str:
                base64_str = base64_str.split("base64,")[1]

            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {e}")

    def _process_single_ocr(
        self,
        image: Image.Image,
        prompt: str,
        temperature: float,
        max_tokens: int,
        ngram_size: int,
        window_size: int,
    ) -> str:
        """Process single image OCR"""
        model_input = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            }
        ]

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            extra_args=dict(
                ngram_size=ngram_size,
                window_size=window_size,
                whitelist_token_ids={128821, 128822},  # <td>, </td>
            ),
            skip_special_tokens=False,
        )

        outputs = self.llm.generate(model_input, sampling_params)
        return outputs[0].outputs[0].text

    def _process_batch_ocr(
        self,
        images: List[Image.Image],
        prompt: str,
        temperature: float,
        max_tokens: int,
        ngram_size: int,
        window_size: int,
    ) -> List[str]:
        """Process batch of images"""
        model_inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": img}
            }
            for img in images
        ]

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            extra_args=dict(
                ngram_size=ngram_size,
                window_size=window_size,
                whitelist_token_ids={128821, 128822},
            ),
            skip_special_tokens=False,
        )

        outputs = self.llm.generate(model_inputs, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def _format_output(self, text: str, format_type: str) -> str:
        """Format OCR output based on requested format"""
        if format_type == "raw":
            return text
        elif format_type == "markdown":
            # Remove grounding tags for clean markdown
            import re
            cleaned = re.sub(
                r'<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>',
                '',
                text,
                flags=re.DOTALL
            )
            return cleaned.strip()
        elif format_type == "json":
            # Parse grounding tags into JSON
            import re
            elements = []
            pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
            matches = re.findall(pattern, text, re.DOTALL)
            for label, coords in matches:
                elements.append({
                    "type": label,
                    "bbox": coords,
                })
            return json.dumps({"elements": elements, "raw_text": text}, indent=2)
        else:
            return text

    def run(self, host: str = "0.0.0.0", port: int = 8001, **kwargs):
        """Run the server"""
        logger.info(f"Starting DeepSeek-OCR server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR vLLM Server"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="deepseek-ai/DeepSeek-OCR",
        help="Model path (HuggingFace or local)"
    )
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default=None,
        help="GPU device(s) to use: None (auto), '0' (single), '0,1' (multi), 'all' (all GPUs)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Server port (default: 8001)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model sequence length"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers"
    )
    parser.add_argument(
        "--encoder-only",
        action="store_true",
        help="Encoder-only mode: Only load standalone encoder for text-to-visual (no OCR, lightweight)"
    )

    args = parser.parse_args()

    # Check if port is available
    if is_port_in_use(args.port, args.host):
        logger.warning(f"Port {args.port} is already in use!")
        try:
            available_port = find_available_port(args.port + 1)
            logger.info(f"Found available port: {available_port}")
            logger.info(f"Use --port {available_port} to use this port, or stop the service on port {args.port}")
        except RuntimeError:
            pass
        logger.error(f"Cannot start server - port {args.port} is in use")
        sys.exit(1)

    logger.info(f"Port {args.port} is available")

    # Initialize and run server
    server = DeepSeekOCRServer(
        model_path=args.model_path,
        gpu_devices=args.gpu_devices,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        encoder_only=args.encoder_only,
    )

    server.run(
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
