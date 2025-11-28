#!/usr/bin/env python3
"""
Data Preparation Script for Text-Only OCRFlow Training

Downloads and prepares text corpora from HuggingFace datasets.
Much easier than dealing with image datasets!
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm


def prepare_hf_dataset(
    dataset_name: str,
    output_path: str,
    text_field: str = "text",
    max_samples: int = 100_000,
    split: str = "train",
    min_length: int = 50,
    max_length: int = 2000,
):
    """
    Download and prepare text from HuggingFace dataset

    Args:
        dataset_name: HF dataset name (e.g., "wikipedia", "c4")
        output_path: Output JSONL file path
        text_field: Name of text field in dataset
        max_samples: Maximum number of samples
        split: Dataset split (train/validation/test)
        min_length: Minimum text length (characters)
        max_length: Maximum text length (truncate if longer)
    """
    from datasets import load_dataset

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset_name} from HuggingFace...")
    print(f"  Split: {split}")
    print(f"  Max samples: {max_samples:,}")

    try:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying with default config...")
        dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)

    count = 0
    skipped = 0

    with open(output_path, 'w', encoding='utf-8') as out:
        pbar = tqdm(total=max_samples, desc="Processing")

        for example in dataset:
            if count >= max_samples:
                break

            text = example.get(text_field, '')

            # Filter by length
            if len(text) < min_length:
                skipped += 1
                continue

            # Truncate long texts
            if len(text) > max_length:
                text = text[:max_length]

            # Save
            json.dump({
                "text": text,
                "source": dataset_name,
            }, out, ensure_ascii=False)
            out.write('\n')

            count += 1
            pbar.update(1)

        pbar.close()

    print(f"\n✓ Saved {count:,} examples to {output_path}")
    print(f"  Skipped {skipped:,} examples (too short)")


def prepare_wikipedia(output_path: str, max_samples: int = 100_000):
    """Prepare Wikipedia dataset"""
    print("\n" + "="*60)
    print("Preparing Wikipedia")
    print("="*60)

    prepare_hf_dataset(
        dataset_name="wikipedia",
        output_path=output_path,
        text_field="text",
        max_samples=max_samples,
        split="train",
        min_length=100,  # Skip very short articles
        max_length=2000,
    )


def prepare_c4(output_path: str, max_samples: int = 500_000):
    """Prepare C4 (Colossal Clean Crawled Corpus)"""
    print("\n" + "="*60)
    print("Preparing C4")
    print("="*60)

    prepare_hf_dataset(
        dataset_name="c4",
        output_path=output_path,
        text_field="text",
        max_samples=max_samples,
        split="train",
        min_length=200,
        max_length=2000,
    )


def prepare_bookcorpus(output_path: str, max_samples: int = 50_000):
    """Prepare BookCorpus"""
    print("\n" + "="*60)
    print("Preparing BookCorpus")
    print("="*60)

    prepare_hf_dataset(
        dataset_name="bookcorpus",
        output_path=output_path,
        text_field="text",
        max_samples=max_samples,
        split="train",
        min_length=100,
        max_length=2000,
    )


def prepare_openwebtext(output_path: str, max_samples: int = 200_000):
    """Prepare OpenWebText"""
    print("\n" + "="*60)
    print("Preparing OpenWebText")
    print("="*60)

    prepare_hf_dataset(
        dataset_name="openwebtext",
        output_path=output_path,
        text_field="text",
        max_samples=max_samples,
        split="train",
        min_length=150,
        max_length=2000,
    )


def prepare_arxiv_abstracts(output_path: str, max_samples: int = 100_000):
    """Prepare arXiv paper abstracts"""
    print("\n" + "="*60)
    print("Preparing arXiv Abstracts")
    print("="*60)

    from datasets import load_dataset

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading arXiv dataset...")
    dataset = load_dataset("scientific_papers", "arxiv", split="train", streaming=True)

    count = 0

    with open(output_path, 'w', encoding='utf-8') as out:
        pbar = tqdm(total=max_samples, desc="Processing")

        for example in dataset:
            if count >= max_samples:
                break

            # Extract abstract
            abstract = example.get("abstract", "")

            if len(abstract) < 100:
                continue

            # Format as markdown
            text = f"## Abstract\n\n{abstract}"

            # Save
            json.dump({
                "text": text,
                "source": "arxiv",
            }, out, ensure_ascii=False)
            out.write('\n')

            count += 1
            pbar.update(1)

        pbar.close()

    print(f"\n✓ Saved {count:,} arXiv abstracts to {output_path}")


def combine_datasets(input_paths: list, output_path: str, shuffle: bool = True):
    """
    Combine multiple JSONL files into one

    Args:
        input_paths: List of input JSONL files
        output_path: Output combined file
        shuffle: Shuffle the combined dataset
    """
    print("\n" + "="*60)
    print("Combining Datasets")
    print("="*60)

    all_data = []

    for input_path in input_paths:
        print(f"Loading {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_data.append(line)

    print(f"Total samples: {len(all_data):,}")

    if shuffle:
        import random
        random.shuffle(all_data)
        print("Shuffled dataset")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as out:
        for line in all_data:
            out.write(line)

    print(f"✓ Combined dataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare text-only datasets for OCRFlow")
    parser.add_argument("--output-dir", type=str, default="./data/text_only",
                       help="Output directory for prepared data")
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["wikipedia", "c4", "bookcorpus", "openwebtext", "arxiv", "all"],
                       help="Which dataset to prepare")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per dataset (uses defaults if not set)")
    parser.add_argument("--combine", action="store_true",
                       help="Combine all prepared datasets into one file")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("OCRFlow Text-Only Data Preparation")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print()

    prepared_files = []

    # Prepare datasets
    if args.dataset in ["wikipedia", "all"]:
        output_path = output_dir / "wikipedia.jsonl"
        max_samples = args.max_samples or 100_000
        prepare_wikipedia(str(output_path), max_samples)
        prepared_files.append(output_path)

    if args.dataset in ["c4", "all"]:
        output_path = output_dir / "c4.jsonl"
        max_samples = args.max_samples or 500_000
        prepare_c4(str(output_path), max_samples)
        prepared_files.append(output_path)

    if args.dataset in ["bookcorpus", "all"]:
        output_path = output_dir / "bookcorpus.jsonl"
        max_samples = args.max_samples or 50_000
        prepare_bookcorpus(str(output_path), max_samples)
        prepared_files.append(output_path)

    if args.dataset in ["openwebtext", "all"]:
        output_path = output_dir / "openwebtext.jsonl"
        max_samples = args.max_samples or 200_000
        prepare_openwebtext(str(output_path), max_samples)
        prepared_files.append(output_path)

    if args.dataset in ["arxiv", "all"]:
        output_path = output_dir / "arxiv.jsonl"
        max_samples = args.max_samples or 100_000
        prepare_arxiv_abstracts(str(output_path), max_samples)
        prepared_files.append(output_path)

    # Combine datasets
    if args.combine and len(prepared_files) > 1:
        combined_path = output_dir / "texts.jsonl"
        combine_datasets(
            [str(p) for p in prepared_files],
            str(combined_path),
            shuffle=True
        )

    print("\n" + "="*60)
    print("Data Preparation Complete!")
    print("="*60)
    print(f"\nPrepared files:")
    for f in prepared_files:
        if f.exists():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}: {size_mb:.1f} MB")

    if args.combine:
        combined_path = output_dir / "texts.jsonl"
        if combined_path.exists():
            size_mb = combined_path.stat().st_size / 1024 / 1024
            print(f"\n  Combined: texts.jsonl ({size_mb:.1f} MB)")

    print(f"\nYou can now train with:")
    print(f"  python examples/train_text_only.py --data-path {output_dir}")


if __name__ == "__main__":
    main()
