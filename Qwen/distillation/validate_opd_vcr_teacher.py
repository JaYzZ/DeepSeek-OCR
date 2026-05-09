#!/usr/bin/env python3

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from accelerate import Accelerator

from Qwen.scripts.train_qwen3vl_opd_vcr import (
    _build_teacher_prompt_rows,
    _extract_thinking_and_answer,
    _generate_teacher_rollouts,
    _load_yaml,
    _prepare_distillation_env,
    _prepare_teacher_processor,
    _prepare_teacher_rollout_runtime,
    _resolve_config_paths,
    _teacher_force_discrete_think_prompt,
)
from Qwen.scripts.qwen3vl_opsd_common import (
    _configure_quiet_logging,
    _load_processor_for_model,
    _maybe_prepare_manifest,
    _set_seed,
)


DEFAULT_SAMPLE_ID = "deepvision_thinking_0047422"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OPD-VCR teacher rollout on one or many manifest rows.")
    parser.add_argument("--config", default="Qwen/configs/distillation/qwen3vl_opd_vcr.yaml")
    parser.add_argument("--sample-id", default=None, help="Validate one specific sample_id.")
    parser.add_argument("--dataset", default=None, help="Optional source_dataset filter.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of rows to validate.")
    parser.add_argument("--batch-size", type=int, default=8, help="Teacher rollout batch size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-jsonl",
        default="tmp_opd_vcr_teacher_validate.jsonl",
        help="Where to write per-sample validation results.",
    )
    return parser.parse_args()


def _load_rows(manifest: Path, *, sample_id: str | None, dataset: str | None) -> list[dict]:
    rows = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if sample_id is not None and rec.get("sample_id") != sample_id:
                continue
            if dataset is not None and rec.get("source_dataset") != dataset:
                continue
            rows.append(rec)
    return rows


def _summarize(results: list[dict]) -> None:
    finish_counter = Counter(str(item.get("finish_reason") or "unknown") for item in results)
    exact_matches = sum(1 for item in results if item.get("answer_match"))
    length_stops = sum(1 for item in results if item.get("finish_reason") == "length")
    close_tag = sum(1 for item in results if item.get("has_close_think_tag"))
    empty_answers = sum(1 for item in results if not str(item.get("answer_extracted") or "").strip())
    text_privileged = sum(1 for item in results if item.get("teacher_rationale_text_chars", 0) > 0)
    print("SUMMARY")
    print("samples", len(results))
    print("finish_reason", dict(finish_counter))
    print("answer_match", f"{exact_matches}/{len(results)}")
    print("length_stop", f"{length_stops}/{len(results)}")
    print("has_close_think_tag", f"{close_tag}/{len(results)}")
    print("empty_answer", f"{empty_answers}/{len(results)}")
    print("text_privileged", f"{text_privileged}/{len(results)}")


def main() -> int:
    args = _parse_args()

    cfg = _load_yaml(Path(args.config))
    cfg = _resolve_config_paths(cfg)
    cfg["training"]["output_dir"] = "/tmp/opd_vcr_teacher_validate"
    cfg["training"]["num_workers"] = 0
    cfg["training"]["per_device_train_batch_size"] = max(1, int(args.batch_size))
    _maybe_prepare_manifest(cfg)

    manifest = Path(cfg["data"]["opsd_manifest"])
    sample_id = args.sample_id
    if sample_id is None and int(args.num_samples) == 1:
        sample_id = DEFAULT_SAMPLE_ID

    rows = _load_rows(manifest, sample_id=sample_id, dataset=args.dataset)
    if not rows:
        raise RuntimeError("No matching rows found in manifest.")

    if sample_id is None:
        random.seed(int(args.seed))
        target_count = min(len(rows), int(args.num_samples))
        rows = random.sample(rows, target_count)
    else:
        rows = rows[:1]

    _configure_quiet_logging()
    acc = Accelerator(mixed_precision="bf16")
    _set_seed(int(args.seed))
    _prepare_distillation_env(cfg)
    processor = _load_processor_for_model(str(cfg["model"]["model_name_or_path"]), cfg)
    teacher_processor = _prepare_teacher_processor(cfg, processor)
    rollout_engine, sampling_params, rollout_processor, _ = _prepare_teacher_rollout_runtime(
        cfg, acc, teacher_processor
    )

    force_discrete_think_prompt = _teacher_force_discrete_think_prompt(cfg)
    output_path = Path(args.output_jsonl)
    results: list[dict] = []

    for start in range(0, len(rows), int(args.batch_size)):
        batch_rows = rows[start : start + int(args.batch_size)]
        prepared_rows = _build_teacher_prompt_rows(cfg, batch_rows)
        rollouts = _generate_teacher_rollouts(
            rollout_engine=rollout_engine,
            sampling_params=sampling_params,
            rollout_processor=rollout_processor,
            prepared_rows=prepared_rows,
            force_discrete_think_prompt=force_discrete_think_prompt,
        )
        for row, prepared, rollout in zip(batch_rows, prepared_rows, rollouts):
            decoded = rollout.get("decoded_text") or ""
            thinking, answer = _extract_thinking_and_answer(
                decoded,
                forced_think_prefix=force_discrete_think_prompt,
            )
            teacher_prompt_text = rollout_processor.apply_chat_template(
                prepared.get("teacher_messages") or [],
                tokenize=False,
                add_generation_prompt=False,
            )
            teacher_rationale_text = str(prepared.get("teacher_rationale_text") or "")
            result = {
                "sample_id": row.get("sample_id"),
                "source_dataset": row.get("source_dataset"),
                "question_text": row.get("student_user_text"),
                "target_answer": row.get("answer_text"),
                "teacher_rationale_text_chars": len(teacher_rationale_text),
                "teacher_rationale_text_preview": teacher_rationale_text[:500],
                "finish_reason": rollout.get("finish_reason"),
                "num_token_ids": len(rollout.get("token_ids") or []),
                "has_close_think_tag": "</think>" in decoded,
                "thinking_chars": len(thinking),
                "answer_extracted": answer,
                "answer_match": answer.strip() == str(row.get("answer_text") or "").strip(),
                "raw_prefix": decoded[:300],
                "raw_suffix": decoded[-300:],
                "teacher_messages": prepared.get("teacher_messages"),
                "teacher_prompt_text": teacher_prompt_text,
                "stop_token_ids": getattr(sampling_params, "stop_token_ids", None),
            }
            results.append(result)
            print(
                f"[{len(results)}/{len(rows)}] sample_id={result['sample_id']} "
                f"finish={result['finish_reason']} tokens={result['num_token_ids']} "
                f"close_think={result['has_close_think_tag']} "
                f"rationale_chars={result['teacher_rationale_text_chars']} "
                f"match={result['answer_match']}"
            )

    with output_path.open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("OUTPUT_JSONL", str(output_path.resolve()))
    print("STOP_IDS", getattr(sampling_params, "stop_token_ids", None))
    _summarize(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
