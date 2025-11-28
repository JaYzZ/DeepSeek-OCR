#!/usr/bin/env python3
"""
DeepSeek-OCR Client
Client library and CLI for interacting with DeepSeek-OCR server
"""

import argparse
import base64
import io
import json
import mimetypes
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
from PIL import Image
from tqdm import tqdm


class DeepSeekOCRClient:
    """Client for DeepSeek-OCR API server"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client

        Args:
            base_url: Base URL of the OCR server
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> Dict:
        """Check server health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def ocr_base64(
        self,
        image_base64: str,
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        ngram_size: int = 30,
        window_size: int = 90,
        output_format: str = "markdown",
    ) -> Dict:
        """
        OCR from base64 encoded image

        Args:
            image_base64: Base64 encoded image
            prompt: OCR prompt
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            ngram_size: N-gram blocking size
            window_size: N-gram window size
            output_format: Output format (markdown/json/raw)

        Returns:
            OCR response dict
        """
        payload = {
            "image_base64": image_base64,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "ngram_size": ngram_size,
            "window_size": window_size,
            "output_format": output_format,
        }

        response = self.session.post(
            f"{self.base_url}/ocr",
            json=payload,
            timeout=300,  # 5 minutes timeout
        )
        response.raise_for_status()
        return response.json()

    def ocr_file(
        self,
        file_path: Union[str, Path],
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        ngram_size: int = 30,
        window_size: int = 90,
        output_format: str = "markdown",
    ) -> Dict:
        """
        OCR from image file

        Args:
            file_path: Path to image file
            prompt: OCR prompt
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            ngram_size: N-gram blocking size
            window_size: N-gram window size
            output_format: Output format (markdown/json/raw)

        Returns:
            OCR response dict
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read and encode image
        with open(file_path, "rb") as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        return self.ocr_base64(
            image_base64=image_base64,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            ngram_size=ngram_size,
            window_size=window_size,
            output_format=output_format,
        )

    def ocr_upload(
        self,
        file_path: Union[str, Path],
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        ngram_size: int = 30,
        window_size: int = 90,
        output_format: str = "markdown",
    ) -> Dict:
        """
        OCR via file upload endpoint

        Args:
            file_path: Path to image file
            prompt: OCR prompt
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            ngram_size: N-gram blocking size
            window_size: N-gram window size
            output_format: Output format (markdown/json/raw)

        Returns:
            OCR response dict
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Prepare multipart form data
        files = {
            "file": (file_path.name, open(file_path, "rb"), mimetypes.guess_type(str(file_path))[0])
        }
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "ngram_size": ngram_size,
            "window_size": window_size,
            "output_format": output_format,
        }

        response = self.session.post(
            f"{self.base_url}/ocr/upload",
            files=files,
            data=data,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def ocr_batch(
        self,
        file_paths: List[Union[str, Path]],
        prompt: str = "<image>\nFree OCR.",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        ngram_size: int = 30,
        window_size: int = 90,
    ) -> Dict:
        """
        Batch OCR processing

        Args:
            file_paths: List of image file paths
            prompt: OCR prompt
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            ngram_size: N-gram blocking size
            window_size: N-gram window size

        Returns:
            Batch OCR response dict
        """
        # Encode all images
        images_base64 = []
        for file_path in file_paths:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue

            with open(file_path, "rb") as f:
                image_data = f.read()
            images_base64.append(base64.b64encode(image_data).decode("utf-8"))

        payload = {
            "images_base64": images_base64,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "ngram_size": ngram_size,
            "window_size": window_size,
        }

        response = self.session.post(
            f"{self.base_url}/ocr/batch",
            json=payload,
            timeout=600,  # 10 minutes for batch
        )
        response.raise_for_status()
        return response.json()

    def save_result(
        self,
        result: Dict,
        output_path: Union[str, Path],
        format: str = "auto",
    ):
        """
        Save OCR result to file

        Args:
            result: OCR response dict
            output_path: Output file path
            format: Output format (auto/markdown/json/txt)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not result.get("success", False):
            raise ValueError(f"OCR failed: {result.get('error', 'Unknown error')}")

        text = result.get("text", "")

        # Auto-detect format from extension
        if format == "auto":
            suffix = output_path.suffix.lower()
            if suffix == ".json":
                format = "json"
            elif suffix in [".md", ".markdown"]:
                format = "markdown"
            else:
                format = "txt"

        # Save based on format
        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

        print(f"Saved result to: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR Client CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image OCR
  python ocr_client.py --input image.jpg --output result.md

  # Batch processing
  python ocr_client.py --batch --input images/*.jpg --output-dir results/

  # Custom prompt
  python ocr_client.py --input doc.png --prompt "<image>\nFree OCR." --output doc.txt

  # JSON output with bounding boxes
  python ocr_client.py --input form.jpg --format json --output form.json

  # Health check
  python ocr_client.py --health
        """
    )

    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="OCR server URL"
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        help="Input image file(s)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (for single file mode)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (for batch mode)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>\n<|grounding|>Convert the document to markdown.",
        help="OCR prompt"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["auto", "markdown", "json", "raw"],
        default="auto",
        help="Output format"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch processing mode"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check server health"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum output tokens"
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=30,
        help="N-gram blocking size"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=90,
        help="N-gram window size"
    )

    args = parser.parse_args()

    # Initialize client
    client = DeepSeekOCRClient(base_url=args.server)

    # Health check
    if args.health:
        try:
            health = client.health_check()
            print("Server Health:")
            print(json.dumps(health, indent=2))
            sys.exit(0)
        except Exception as e:
            print(f"Health check failed: {e}")
            sys.exit(1)

    # Validate input
    if not args.input:
        parser.error("--input is required (unless using --health)")

    # Expand glob patterns
    from glob import glob
    input_files = []
    for pattern in args.input:
        input_files.extend(glob(pattern))

    if not input_files:
        print("Error: No input files found")
        sys.exit(1)

    # Batch mode
    if args.batch or len(input_files) > 1:
        print(f"Processing {len(input_files)} files in batch mode...")

        # Determine output directory
        output_dir = Path(args.output_dir) if args.output_dir else Path("ocr_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = client.ocr_batch(
                file_paths=input_files,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                ngram_size=args.ngram_size,
                window_size=args.window_size,
            )

            if result.get("success", False):
                # Save individual results
                for idx, ocr_result in enumerate(result.get("results", [])):
                    input_file = Path(input_files[idx])
                    output_file = output_dir / f"{input_file.stem}.md"

                    try:
                        client.save_result(
                            {"success": True, "text": ocr_result.get("text", "")},
                            output_file,
                            format=args.format,
                        )
                    except Exception as e:
                        print(f"Error saving {output_file}: {e}")

                print(f"\nBatch processing complete!")
                print(f"Processed: {result.get('total_processed', 0)} files")
                print(f"Results saved to: {output_dir}")
            else:
                print(f"Batch processing failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

        except Exception as e:
            print(f"Batch processing error: {e}")
            sys.exit(1)

    # Single file mode
    else:
        input_file = input_files[0]
        output_file = args.output if args.output else f"{Path(input_file).stem}_ocr.md"

        print(f"Processing: {input_file}")

        try:
            result = client.ocr_upload(
                file_path=input_file,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                ngram_size=args.ngram_size,
                window_size=args.window_size,
                output_format=args.format if args.format != "auto" else "markdown",
            )

            client.save_result(result, output_file, format=args.format)

            print("\nOCR complete!")
            print(f"Result saved to: {output_file}")

        except Exception as e:
            print(f"OCR error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
