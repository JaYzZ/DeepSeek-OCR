#!/usr/bin/env python3
"""
Build Unified SFT Dataset combining 4 key components:

1. LLaVA Pretrain 558K (caption + rendered caption OCR)
2. DocLayNet Enhanced ALL (bbox-ocr + full-ocr + markdown)
3. HierText bbox-ocr
4. LLaVA Instruct 665K (VQA with rendered questions)

This is the ONE unified SFT dataset for training.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterator

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield items from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def count_samples(path: Path) -> int:
    """Count samples in a JSONL file."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def build_unified_dataset(
    doclaynet_jsonl: Path,
    hiertext_jsonl: Path,
    alignment_jsonl: Path,
    vqa_jsonl: Path,
    output_jsonl: Path,
    doclaynet_ratio: float = 1.0,
    hiertext_ratio: float = 1.0,
    alignment_ratio: float = 1.0,
    vqa_ratio: float = 1.0,
    max_samples: int = 0,
    seed: int = 42,
) -> None:
    """
    Build unified SFT dataset combining 4 components.

    Args:
        doclaynet_jsonl: DocLayNet enhanced ALL (bbox-ocr + full-ocr + markdown)
        hiertext_jsonl: HierText bbox-ocr
        alignment_jsonl: LLaVA pretrain 558K (caption + rendered OCR)
        vqa_jsonl: LLaVA instruct 665K (VQA with rendered questions)
        output_jsonl: Output path for unified dataset
        *_ratio: Sampling ratios for each component
        max_samples: Maximum total samples (0 = unlimited)
    """
    rng = random.Random(seed)

    print("\n" + "=" * 80)
    print("Building Unified SFT Dataset (4 Components)")
    print("=" * 80)
    print(f"1. DocLayNet Enhanced ALL: {doclaynet_jsonl}")
    print(f"2. HierText bbox-ocr: {hiertext_jsonl}")
    print(f"3. LLaVA Pretrain 558K: {alignment_jsonl}")
    print(f"4. LLaVA Instruct 665K: {vqa_jsonl}")
    print(f"Output: {output_jsonl}")
    print(f"Sampling ratios - DocLayNet: {doclaynet_ratio}, HierText: {hiertext_ratio}, "
          f"Alignment: {alignment_ratio}, VQA: {vqa_ratio}")
    print("=" * 80)

    # Verify input files exist
    for name, path in [
        ("DocLayNet", doclaynet_jsonl),
        ("HierText", hiertext_jsonl),
        ("Alignment", alignment_jsonl),
        ("VQA", vqa_jsonl),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} dataset not found: {path}")

    # Count samples
    print("\n[Step 1] Counting samples...")
    doclaynet_count = count_samples(doclaynet_jsonl)
    hiertext_count = count_samples(hiertext_jsonl)
    alignment_count = count_samples(alignment_jsonl)
    vqa_count = count_samples(vqa_jsonl)
    total_input = doclaynet_count + hiertext_count + alignment_count + vqa_count

    print(f"  DocLayNet ALL: {doclaynet_count:,} samples")
    print(f"  HierText: {hiertext_count:,} samples")
    print(f"  Alignment (558K): {alignment_count:,} samples")
    print(f"  VQA (665K): {vqa_count:,} samples")
    print(f"  Total input: {total_input:,} samples")

    # Calculate target counts based on ratios
    total_ratio = doclaynet_ratio + hiertext_ratio + alignment_ratio + vqa_ratio
    doclaynet_fraction = doclaynet_ratio / total_ratio
    hiertext_fraction = hiertext_ratio / total_ratio
    alignment_fraction = alignment_ratio / total_ratio
    vqa_fraction = vqa_ratio / total_ratio

    if max_samples > 0:
        doclaynet_target = int(max_samples * doclaynet_fraction)
        hiertext_target = int(max_samples * hiertext_fraction)
        alignment_target = int(max_samples * alignment_fraction)
        vqa_target = int(max_samples * vqa_fraction)
    else:
        # Use all samples
        doclaynet_target = doclaynet_count
        hiertext_target = hiertext_count
        alignment_target = alignment_count
        vqa_target = vqa_count

    print(f"\n[Step 2] Target samples:")
    print(f"  DocLayNet: {doclaynet_target:,} ({doclaynet_fraction:.1%})")
    print(f"  HierText: {hiertext_target:,} ({hiertext_fraction:.1%})")
    print(f"  Alignment: {alignment_target:,} ({alignment_fraction:.1%})")
    print(f"  VQA: {vqa_target:,} ({vqa_fraction:.1%})")
    if max_samples > 0:
        print(f"  Total target: {doclaynet_target + hiertext_target + alignment_target + vqa_target:,}")

    # Load samples
    print(f"\n[Step 3] Loading samples...")
    doclaynet_samples = list(iter_jsonl(doclaynet_jsonl))
    hiertext_samples = list(iter_jsonl(hiertext_jsonl))
    alignment_samples = list(iter_jsonl(alignment_jsonl))
    vqa_samples = list(iter_jsonl(vqa_jsonl))
    print(f"  Loaded {len(doclaynet_samples):,} DocLayNet samples")
    print(f"  Loaded {len(hiertext_samples):,} HierText samples")
    print(f"  Loaded {len(alignment_samples):,} Alignment samples")
    print(f"  Loaded {len(vqa_samples):,} VQA samples")

    # Sample with replacement if target > available
    def sample_with_replacement(samples: list, target: int, seed: int) -> list:
        if target <= len(samples):
            return rng.sample(samples, target)
        else:
            return [rng.choice(samples) for _ in range(target)]

    print(f"\n[Step 4] Sampling with target ratios...")
    sampled_doclaynet = sample_with_replacement(doclaynet_samples, doclaynet_target, seed)
    sampled_hiertext = sample_with_replacement(hiertext_samples, hiertext_target, seed + 1)
    sampled_alignment = sample_with_replacement(alignment_samples, alignment_target, seed + 2)
    sampled_vqa = sample_with_replacement(vqa_samples, vqa_target, seed + 3)
    print(f"  Sampled {len(sampled_doclaynet):,} DocLayNet samples")
    print(f"  Sampled {len(sampled_hiertext):,} HierText samples")
    print(f"  Sampled {len(sampled_alignment):,} Alignment samples")
    print(f"  Sampled {len(sampled_vqa):,} VQA samples")

    # Combine and shuffle
    print(f"\n[Step 5] Combining and shuffling...")
    unified_samples = sampled_doclaynet + sampled_hiertext + sampled_alignment + sampled_vqa
    rng.shuffle(unified_samples)
    print(f"  Total: {len(unified_samples):,} samples")

    # Write output
    print(f"\n[Step 6] Writing output...")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for i, sample in enumerate(unified_samples):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if (i + 1) % 50000 == 0:
                print(f"  Wrote {i + 1:,} samples...")

    print(f"\n{'=' * 80}")
    print(f"✓ Built unified SFT dataset: {output_jsonl}")
    print(f"  Total samples: {len(unified_samples):,}")
    print(f"  DocLayNet: {len(sampled_doclaynet):,} ({len(sampled_doclaynet)/len(unified_samples):.1%})")
    print(f"  HierText: {len(sampled_hiertext):,} ({len(sampled_hiertext)/len(unified_samples):.1%})")
    print(f"  Alignment: {len(sampled_alignment):,} ({len(sampled_alignment)/len(unified_samples):.1%})")
    print(f"  VQA: {len(sampled_vqa):,} ({len(sampled_vqa)/len(unified_samples):.1%})")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build unified SFT dataset combining 4 key components"
    )
    parser.add_argument("--doclaynet-jsonl",
                        default="OCRVL/llamafactory/data/doclaynet_all.jsonl",
                        help="DocLayNet enhanced ALL (bbox + full-ocr + markdown)")
    parser.add_argument("--hiertext-jsonl",
                        default="OCRVL/llamafactory/data/hiertext_unified.jsonl",
                        help="HierText bbox-ocr")
    parser.add_argument("--alignment-jsonl",
                        default="OCRVL/llamafactory/data/ocrvl_alignment_text_prompts.jsonl",
                        help="LLaVA pretrain 558K (caption + rendered OCR)")
    parser.add_argument("--vqa-jsonl",
                        default="OCRVL/llamafactory/data/ocrvl_llava_mix665k.jsonl",
                        help="LLaVA instruct 665K (VQA with rendered questions)")
    parser.add_argument("--output",
                        default="OCRVL/llamafactory/data/ocrvl_unified_sft.jsonl",
                        help="Output unified SFT JSONL file")
    parser.add_argument("--doclaynet-ratio", type=float, default=1.0)
    parser.add_argument("--hiertext-ratio", type=float, default=1.0)
    parser.add_argument("--alignment-ratio", type=float, default=1.0)
    parser.add_argument("--vqa-ratio", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    build_unified_dataset(
        doclaynet_jsonl=Path(args.doclaynet_jsonl),
        hiertext_jsonl=Path(args.hiertext_jsonl),
        alignment_jsonl=Path(args.alignment_jsonl),
        vqa_jsonl=Path(args.vqa_jsonl),
        output_jsonl=Path(args.output),
        doclaynet_ratio=args.doclaynet_ratio,
        hiertext_ratio=args.hiertext_ratio,
        alignment_ratio=args.alignment_ratio,
        vqa_ratio=args.vqa_ratio,
        max_samples=args.max_samples,
        seed=args.seed,
    )
