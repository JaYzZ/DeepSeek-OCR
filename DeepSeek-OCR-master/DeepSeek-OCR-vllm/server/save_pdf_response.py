#!/usr/bin/env python3
"""
Save PDF OCR response to local folder structure
Complete package: markdown + images transmitted in single response
"""

import argparse
import base64
import json
import sys
from datetime import datetime
from pathlib import Path

import requests


def save_pdf_response(
    pdf_path: str,
    server_url: str = "http://localhost:8009",
    output_dir: str = None
):
    """
    Process PDF via API and save complete response (markdown + images) to local folder

    Args:
        pdf_path: Path to PDF file
        server_url: Server URL
        output_dir: Custom output directory (optional)
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    # Create output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("pdf") / f"{pdf_file.stem}_{timestamp}_output"

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"📄 Processing PDF: {pdf_path}")
    print(f"🌐 Server: {server_url}")
    print(f"📁 Output directory: {out_dir}")
    print("")

    # Make API request
    try:
        with open(pdf_file, "rb") as f:
            files = {"file": f}
            print("🚀 Sending request to server...")
            response = requests.post(
                f"{server_url}/ocr/pdf",
                files=files,
                timeout=600  # 10 minutes for large PDFs
            )

        if response.status_code != 200:
            print(f"❌ Error: Server returned status {response.status_code}")
            print(response.text)
            sys.exit(1)

        result = response.json()

    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Cannot connect to server at {server_url}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Check if successful
    if not result.get("success"):
        print(f"❌ Error: {result.get('error')}")
        sys.exit(1)

    print(f"✅ Received response from server\n")

    # Save markdown file
    markdown_path = out_dir / f"{pdf_file.stem}.mmd"
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(result["markdown_text"])
    print(f"✓ Saved markdown: {markdown_path} ({len(result['markdown_text']):,} chars)")

    # Save detection file
    detection_path = out_dir / f"{pdf_file.stem}_det.mmd"
    with open(detection_path, "w", encoding="utf-8") as f:
        f.write(result["detection_text"])
    print(f"✓ Saved detection: {detection_path} ({len(result['detection_text']):,} chars)")

    # Save images from base64
    images_dict = result.get("images", {})
    if images_dict:
        print(f"\n🖼️  Saving {len(images_dict)} images...")
        for filename, base64_data in images_dict.items():
            try:
                # Decode base64 and save
                img_data = base64.b64decode(base64_data)
                img_path = images_dir / filename
                with open(img_path, "wb") as f:
                    f.write(img_data)
                print(f"  ✓ {filename} ({len(img_data):,} bytes)")
            except Exception as e:
                print(f"  ✗ Failed to save {filename}: {e}")
    else:
        print("\n💡 No images extracted from this PDF")

    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"  Total pages: {result['total_pages']}")

    total_images = 0
    for page in result["pages"]:
        page_images = page.get("extracted_images", 0)
        total_images += page_images
        if page_images > 0:
            print(f"  Page {page['page_number']}: {page['char_count']:,} chars, {page_images} images")

    print(f"\n✅ Complete! Output structure:")
    print(f"   {out_dir}/")
    print(f"   ├── {pdf_file.stem}.mmd           ({len(result['markdown_text']):,} chars)")
    print(f"   ├── {pdf_file.stem}_det.mmd       ({len(result['detection_text']):,} chars)")
    if total_images > 0:
        print(f"   └── images/                     ({total_images} images)")
        for img_name in sorted(images_dict.keys()):
            print(f"       ├── {img_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Save PDF OCR response with complete package (markdown + images)"
    )
    parser.add_argument(
        "pdf_file",
        help="Path to PDF file"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8009",
        help="Server URL (default: http://localhost:8009)"
    )
    parser.add_argument(
        "--output-dir",
        help="Custom output directory (default: pdf/{filename}_{timestamp}_output)"
    )

    args = parser.parse_args()

    save_pdf_response(
        pdf_path=args.pdf_file,
        server_url=args.server,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
