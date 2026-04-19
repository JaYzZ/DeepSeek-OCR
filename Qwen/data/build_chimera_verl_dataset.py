#!/usr/bin/env python3
"""Convert Chimera parquet files into VERL RL parquet format.

Chimera dataset structure:
- question: The problem statement (rendered as image during SFT)
- solution: The step-by-step reasoning (chunked for thinking images)
- answer: The final answer (used for reward computation in RL)
- original_solution: Full solution with thinking tags (for latent supervision in SFT)
- topic: Problem category (e.g., "math", "physics")

For RL training, we use:
1. the pre-rendered question image when available
2. a Chimera-specific prompt contract that matches multipart academic problems
3. the verified final answer as the reward target
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import datasets
import pandas as pd
from project_paths import deepseek_ocr_path, hf_path


DEFAULT_DATA_DIR = hf_path("TianHongZXY", "CHIMERA", "Qwen3.5-397B")
DEFAULT_OUTPUT_DIR = deepseek_ocr_path("Qwen", "data", "chimera_verl")
CHIMERA_SYSTEM_PROMPT = (
    "You are solving a challenging academic problem from a rendered question image. "
    "Reason carefully. In the final answer, provide only the final result. "
    "If the problem has labeled subparts, keep the part labels and provide only the final answer for each part. "
    "If the problem has a single final answer, put it in \\boxed{}. "
    "Do not restate the full question or include unnecessary explanation in the final answer."
)


def _to_python(value: Any) -> Any:
    """Convert PyArrow/other types to native Python types."""
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray, dict, list, tuple)):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_python(v) for v in value]
    if isinstance(value, list):
        return [_to_python(v) for v in value]
    return value


def _build_user_content(prompt_text: str, *, has_image: bool) -> str:
    if not has_image:
        return prompt_text
    if "<image>" in prompt_text:
        return prompt_text
    return f"<image>\n{prompt_text}".strip()


def _build_problem_context(subject: str, topic: str) -> str:
    parts: list[str] = []
    if subject:
        parts.append(subject)
    if topic and topic.lower() != subject.lower():
        parts.append(topic)
    if not parts:
        return ""
    if len(parts) == 1:
        return f"This is a {parts[0]} problem."
    return f"This is a {parts[0]} problem about {parts[1]}."


def _build_user_prompt_text(question: str, *, subject: str, topic: str, has_image: bool) -> str:
    context = _build_problem_context(subject, topic)
    if has_image:
        base = "The image contains the full problem statement. Solve it carefully."
        return f"{base} {context}".strip()

    base = "Solve the following problem carefully."
    if context:
        base = f"{base} {context}"
    return f"{base}\n\n{question}".strip()


def _normalize_prompt(
    row: dict[str, Any],
    sample_id: str,
    chimera_images_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, bytes | None]]]:
    """Build prompt messages from the question, using pre-rendered images."""
    question = str(row.get("question") or "").strip()
    subject = str(row.get("subject") or "").strip()
    topic = str(row.get("topic") or "").strip()

    # Try to use pre-rendered question image from SFT dataset
    question_image_path = chimera_images_dir / f"{sample_id}_question.png"

    images = []
    if question_image_path.exists():
        # Match the stable DeepVision VERL parquet schema: image bytes live in the
        # separate `images` column, while prompt text uses a plain `<image>` token.
        with open(question_image_path, "rb") as f:
            image_bytes = f.read()
        images = [{"bytes": image_bytes, "path": None}]
        user_content = _build_user_content(
            _build_user_prompt_text(question, subject=subject, topic=topic, has_image=True),
            has_image=True,
        )
    else:
        # Fallback to text if image not found (shouldn't happen if SFT dataset was built)
        import warnings
        warnings.warn(f"Question image not found: {question_image_path}, using text instead")
        user_content = _build_user_content(
            _build_user_prompt_text(question, subject=subject, topic=topic, has_image=False),
            has_image=False,
        )

    return [
        {"role": "system", "content": CHIMERA_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ], images


def _extract_equivalent_answers(answer: str, solution: str, original_solution: str) -> list[str]:
    """Extract equivalent answer forms from the solution text."""
    answers = [answer]

    # Try to extract boxed answers from solution
    import re
    boxed_re = re.compile(r"\\boxed\s*{([^}]+)}")
    for match in boxed_re.finditer(solution):
        boxed_answer = match.group(1).strip()
        if boxed_answer and boxed_answer not in answers:
            answers.append(boxed_answer)

    # Also check original_solution for boxed answers
    for match in boxed_re.finditer(original_solution):
        boxed_answer = match.group(1).strip()
        if boxed_answer and boxed_answer not in answers:
            answers.append(boxed_answer)

    # Extract numeric answers as fallback
    numeric_re = re.compile(r"(?<!\w)(-?\d+(?:\.\d+)?)(?!\w)")
    for match in numeric_re.finditer(answer):
        numeric = match.group(1).strip()
        if numeric and numeric not in answers:
            answers.append(numeric)

    return answers


def _build_record(row: dict[str, Any], source_name: str, row_idx: int, chimera_images_dir: Path) -> dict[str, Any] | None:
    """Build a VERL-format record from a Chimera row."""
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    solution = str(row.get("solution") or "").strip()
    original_solution = str(row.get("original_solution") or "").strip()
    subject = str(row.get("subject") or "general").strip()
    topic = str(row.get("topic") or "").strip()
    index = int(row.get("index", row_idx))
    correctness = row.get("correctness")

    # Skip samples without required fields. Keep verifier-failed traces because
    # RL supervision is based on the canonical answer, not the source solution.
    if not question or not answer:
        return None

    # Build sample_id for image lookup
    sample_id = f"chimera_{index:07d}"

    # Build prompt messages with images
    prompt, images = _normalize_prompt(row, sample_id, chimera_images_dir)

    # Build equivalent answers list
    equivalent_answers = _extract_equivalent_answers(answer, solution, original_solution)

    return {
        "data_source": f"chimera::{subject}",
        "prompt": prompt,
        "images": images,
        "ability": topic if topic else subject,
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
            "equivalent_answers": equivalent_answers,
        },
        "extra_info": {
            "split": source_name,
            "index": index,
            "question": question,
            "subject": subject,
            "topic": topic,
            "correctness": bool(correctness) if correctness is not None else None,
            "solution": solution,
            "original_solution": original_solution,
        },
    }


def _load_records(data_dir: Path, chimera_images_dir: Path, include_patterns: list[str]) -> list[dict[str, Any]]:
    """Load and convert Chimera parquet files to VERL format."""
    records: list[dict[str, Any]] = []

    # Find all matching parquet files
    parquet_files = []
    for pattern in include_patterns:
        matches = sorted(data_dir.glob(pattern))
        parquet_files.extend(matches)

    if not parquet_files:
        raise FileNotFoundError(f"No Chimera parquet files found matching: {include_patterns}")

    print(f"Found {len(parquet_files)} parquet files to process")

    for parquet_path in parquet_files:
        print(f"Processing: {parquet_path.name}")
        frame = pd.read_parquet(parquet_path)
        print(f"  Rows: {len(frame)}")

        for row_idx, (_, row) in enumerate(frame.iterrows()):
            record = _build_record(_to_python(row.to_dict()), source_name=parquet_path.stem, row_idx=row_idx, chimera_images_dir=chimera_images_dir)
            if record is not None:
                records.append(record)

            if (row_idx + 1) % 1000 == 0:
                print(f"  Processed {row_idx + 1}/{len(frame)} rows")

    print(f"Total valid records: {len(records)}")
    return records


def _split_records(
    records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
    target_val_samples: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into train and validation sets, stratified by data source (subject).

    Uses largest remainder method to ensure exact target_val_samples while maintaining ratios.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["data_source"], []).append(record)

    rng = random.Random(seed)

    # Calculate initial val counts using proportional allocation
    source_val_allocations: dict[str, int] = {}
    source_fractions: list[tuple[str, float]] = []  # For largest remainder method

    for source, source_records in sorted(grouped.items()):
        exact_val = len(source_records) * val_ratio
        base_val = int(exact_val)
        remainder = exact_val - base_val
        source_val_allocations[source] = base_val
        source_fractions.append((source, remainder))

    # Allocate remaining samples using largest remainder method
    current_total = sum(source_val_allocations.values())
    remaining = target_val_samples - current_total

    # Sort by remainder (descending) and allocate extras
    source_fractions.sort(key=lambda x: x[1], reverse=True)
    for source, _ in source_fractions[:remaining]:
        source_val_allocations[source] += 1

    # Now do the actual splitting
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []

    for source, source_records in sorted(grouped.items()):
        source_records = list(source_records)
        rng.shuffle(source_records)
        val_count = source_val_allocations[source]
        val_count = min(val_count, len(source_records) - 1) if len(source_records) > 1 else len(source_records)
        for record in source_records[:val_count]:
            val_records.append(record)
        for record in source_records[val_count:]:
            train_records.append(record)
        print(f"  {source}: train={len(source_records) - val_count}, val={val_count}")

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    if max_train_samples is not None:
        train_records = train_records[:max_train_samples]
    if max_val_samples is not None:
        val_records = val_records[:max_val_samples]

    return train_records, val_records


def _write_parquet(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write records to parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(records).to_parquet(str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to Chimera parquet files (default: %(default)s)",
    )
    parser.add_argument(
        "--chimera-images-dir",
        type=Path,
        default=Path("Qwen/data/chimera_images"),
        help="Path to Chimera rendered images from SFT dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for VERL parquet files (default: %(default)s)",
    )
    parser.add_argument("--train-file", default="train.parquet", help="Training output filename")
    parser.add_argument("--val-file", default="val.parquet", help="Validation output filename")
    parser.add_argument(
        "--patterns",
        default="train-*.parquet",
        help="Glob patterns for Chimera parquet files (comma-separated, default: %(default)s)",
    )
    parser.add_argument("--target-val-samples", type=int, default=100, help="Target number of validation samples (default: 100, stratified by subject)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Max training samples (default: all)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Max validation samples (default: all)")
    args = parser.parse_args()

    include_patterns = [p.strip() for p in args.patterns.split(",")]

    # Resolve chimera_images_dir relative to repo root if not absolute
    chimera_images_dir = args.chimera_images_dir
    if not chimera_images_dir.is_absolute():
        repo_root = Path(__file__).parent.parent.parent
        chimera_images_dir = repo_root / chimera_images_dir

    print("=" * 70)
    print("Chimera VERL Dataset Builder")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Chimera images directory: {chimera_images_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"File patterns: {include_patterns}")
    print(f"Target validation samples: {args.target_val_samples}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)

    records = _load_records(args.data_dir, chimera_images_dir, include_patterns=include_patterns)

    # Calculate val_ratio to achieve target_val_samples
    val_ratio = args.target_val_samples / len(records)
    print(f"\nCalculated validation ratio: {val_ratio:.4f} ({args.target_val_samples}/{len(records)})")

    print("\nSplitting into train/val sets...")
    train_records, val_records = _split_records(
        records=records,
        val_ratio=val_ratio,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        target_val_samples=args.target_val_samples,
    )

    train_path = args.output_dir / args.train_file
    val_path = args.output_dir / args.val_file

    print(f"\nWriting training data to: {train_path}")
    _write_parquet(train_records, train_path)

    print(f"Writing validation data to: {val_path}")
    _write_parquet(val_records, val_path)

    summary = {
        "data_dir": str(args.data_dir),
        "chimera_images_dir": str(chimera_images_dir),
        "patterns": include_patterns,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "train_file": str(train_path),
        "val_file": str(val_path),
        "total_samples": len(records),
        "stratified_by": "subject",
        "with_images": True,
    }

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
