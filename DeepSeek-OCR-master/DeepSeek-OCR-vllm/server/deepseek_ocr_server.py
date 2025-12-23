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

        # Setup CUDA 12.6 environment (cu126)
        cuda_home = '/usr/local/cuda'
        if os.path.exists(cuda_home):
            os.environ['CUDA_HOME'] = cuda_home
            os.environ['PATH'] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"
            os.environ['LD_LIBRARY_PATH'] = f"{cuda_home}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
            logger.info(f"Using CUDA from: {cuda_home}")

            # Set Triton PTXAS path for CUDA
            ptxas_path = os.path.join(cuda_home, 'bin/ptxas')
            if os.path.exists(ptxas_path):
                os.environ['TRITON_PTXAS_PATH'] = ptxas_path
                logger.info(f"Set TRITON_PTXAS_PATH: {ptxas_path}")
        else:
            logger.warning(f"CUDA_HOME path not found: {cuda_home}")

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
            logger.info("✓ Model initialized successfully - full OCR pipeline ready")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

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
            logger.info("⚡ Encoder-only mode: Only /health endpoint available")
            logger.info("⚡ Full OCR endpoints disabled in encoder-only mode")

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
