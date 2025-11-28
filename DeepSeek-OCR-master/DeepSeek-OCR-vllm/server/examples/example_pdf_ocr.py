#!/usr/bin/env python3
"""
PDF OCR Example Script
Demonstrates PDF processing with output directory control
"""

import requests
import sys
from pathlib import Path

# Configuration
SERVER_URL = "http://localhost:8009"  # Change to http://192.168.10.118:8009 for remote

def process_pdf(
    pdf_path: str,
    output_dir: str = None,
    save_outputs: bool = True,
    server_url: str = SERVER_URL
):
    """
    Process PDF file with OCR

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save outputs (optional)
        save_outputs: Whether to save outputs to disk
        server_url: Server URL

    Returns:
        Response JSON
    """
    print(f"\n{'='*60}")
    print(f"Processing PDF: {pdf_path}")
    print(f"{'='*60}\n")

    try:
        with open(pdf_path, "rb") as f:
            files = {"file": f}
            data = {
                "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
                "dpi": "144",
                "save_outputs": str(save_outputs).lower(),
            }

            # Add output_dir if specified
            if output_dir:
                data["output_dir"] = output_dir
                print(f"📁 Output directory: {output_dir}")

            print(f"🚀 Sending request to {server_url}/ocr/pdf...")

            response = requests.post(
                f"{server_url}/ocr/pdf",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout for large PDFs
            )

            result = response.json()

            if result.get("success"):
                print(f"\n{'='*60}")
                print("✓ PDF Processing Successful!")
                print(f"{'='*60}")
                print(f"📄 Total pages: {result['total_pages']}")
                print(f"📝 Total characters: {len(result['markdown_text'])}")

                # Show page-by-page stats
                print(f"\n📊 Page Statistics:")
                for page in result.get("pages", []):
                    print(f"  Page {page['page_number']}: {page['char_count']} characters")

                # Show output files if saved
                if result.get("output_files"):
                    print(f"\n💾 Output Files:")
                    for file_type, file_path in result["output_files"].items():
                        print(f"  {file_type}: {file_path}")

                # Show preview of markdown text
                print(f"\n📖 Preview (first 500 characters):")
                print("-" * 60)
                print(result["markdown_text"][:500])
                if len(result["markdown_text"]) > 500:
                    print("...")
                print("-" * 60)

                return result

            else:
                print(f"\n✗ Error: {result.get('error')}")
                return None

    except FileNotFoundError:
        print(f"✗ PDF file not found: {pdf_path}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {server_url}")
        print("  Make sure the server is running")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def main():
    """Main function"""
    print("\n" + "="*60)
    print("DeepSeek-OCR PDF Processing Example")
    print("="*60)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print(f"  {sys.argv[0]} <pdf_file> [output_dir]")
        print("\nExamples:")
        print(f"  {sys.argv[0]} document.pdf")
        print(f"  {sys.argv[0]} document.pdf /home/user/ocr_results")
        print(f"  {sys.argv[0]} research_paper.pdf ./outputs")
        print("\nOptions:")
        print("  pdf_file    - Path to PDF file to process")
        print("  output_dir  - Directory to save outputs (optional)")
        print("\nNote:")
        print("  - If output_dir is not specified, files saved to /tmp/deepseek_ocr_output")
        print("  - Generates two files: document.mmd and document_det.mmd")
        print("  - .mmd = clean markdown, _det.mmd = with grounding tags")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # Process PDF
    result = process_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir,
        save_outputs=True,
        server_url=SERVER_URL
    )

    if result:
        print(f"\n{'='*60}")
        print("✓ Done!")
        print(f"{'='*60}\n")
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print("✗ Processing failed")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
