#!/usr/bin/env python3
"""OPSD dataset adapter for VERL RL training.

The source is the OPSD JSONL manifest built from the thinking SFT datasets.
Each row is converted to VERL's RL format while carrying an additional
privileged teacher prompt as token IDs for actor-side replay distillation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import datasets

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You solve visual reasoning problems. Think carefully, then provide the final answer."
)

DEFAULT_STUDENT_TEMPLATE = """Answer the question.

Question:
{question_text}"""

DEFAULT_TEACHER_TEMPLATE = """Problem:
{question_text}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{reference_solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand the reasoning behind each step. Do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above."""

DEFAULT_IMAGE_MIN_PIXELS = 32 * 32
DEFAULT_IMAGE_MAX_PIXELS = 262144


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _strip_image_placeholder(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("<image>"):
        text = text[len("<image>") :].lstrip()
    return text


def _build_student_prompt(
    *,
    system_prompt: str,
    user_text: str,
    has_image: bool,
) -> list[dict[str, str]]:
    if has_image and "<image>" not in user_text:
        user_text = f"<image>\n{user_text}".strip()
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return messages


def _build_teacher_messages(*, system_prompt: str, user_text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    return messages


def _apply_chat_template_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


class OPSDRLDataset(RLHFDataset):
    """Convert OPSD manifest rows into VERL RL samples with teacher replay IDs."""

    def _build_messages(self, example: dict):
        image_paths = example.pop("opsd_image_paths", None) or []
        images = []
        valid_paths = []
        for raw_path in image_paths:
            path = Path(str(raw_path))
            if not path.is_file():
                logger.warning("Skipping missing OPSD image path: %s", path)
                continue
            image_payload: dict[str, Any] = {"image": str(path)}
            if getattr(self, "opsd_image_min_pixels", None) is not None:
                image_payload["min_pixels"] = self.opsd_image_min_pixels
            if getattr(self, "opsd_image_max_pixels", None) is not None:
                image_payload["max_pixels"] = self.opsd_image_max_pixels
            images.append(image_payload)
            valid_paths.append(str(path))
        example[self.image_key] = images
        example["images"] = images
        messages = super()._build_messages(example)
        example[self.image_key] = images
        example["opsd_image_paths"] = valid_paths
        return messages

    def _read_files_and_tokenize(self):
        rows: list[dict[str, Any]] = []
        for data_file in self.data_files:
            if str(data_file).endswith(".jsonl"):
                rows.extend(_read_jsonl(data_file))
            elif str(data_file).endswith(".json"):
                rows.extend(_read_jsonl(data_file))
            else:
                loaded = datasets.load_dataset("parquet", data_files=data_file)["train"]
                rows.extend(dict(row) for row in loaded)

        opsd_cfg = self.config.get("opsd", {}) or {}
        system_prompt = str(opsd_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT) or "")
        student_template = str(opsd_cfg.get("student_template", DEFAULT_STUDENT_TEMPLATE))
        teacher_template = str(opsd_cfg.get("teacher_template", DEFAULT_TEACHER_TEMPLATE))
        self.opsd_system_prompt = system_prompt
        self.opsd_teacher_max_prompt_length = int(opsd_cfg.get("teacher_max_prompt_length", 4096))
        self.opsd_image_min_pixels = _optional_int(opsd_cfg.get("image_min_pixels", DEFAULT_IMAGE_MIN_PIXELS))
        self.opsd_image_max_pixels = _optional_int(opsd_cfg.get("image_max_pixels", DEFAULT_IMAGE_MAX_PIXELS))

        converted: list[dict[str, Any]] = []
        skipped = 0
        for row_idx, row in enumerate(rows):
            question_text = _strip_image_placeholder(row.get("student_user_text") or "")
            answer_text = str(row.get("answer_text") or "").strip()
            teacher_solution = str(row.get("teacher_solution_text") or "").strip()
            if not question_text or not answer_text or not teacher_solution:
                skipped += 1
                continue

            image_paths = [str(path) for path in (row.get("question_images") or []) if str(path)]
            student_user = student_template.format(question_text=question_text)
            student_prompt = _build_student_prompt(
                system_prompt=system_prompt,
                user_text=student_user,
                has_image=bool(image_paths),
            )

            teacher_user = teacher_template.format(
                question_text=question_text,
                reference_solution=teacher_solution,
            )

            sample_id = str(row.get("sample_id") or f"opsd_{row_idx:07d}")
            source_dataset = str(row.get("source_dataset") or "opsd")
            converted.append(
                {
                    "prompt": student_prompt,
                    "images": [],
                    "opsd_image_paths": image_paths,
                    "data_source": f"opsd::{source_dataset}",
                    "ability": str(row.get("task") or source_dataset or "reasoning"),
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer_text,
                        "equivalent_answers": [answer_text],
                    },
                    "extra_info": {
                        "index": row_idx,
                        "sample_id": sample_id,
                        "source_dataset": source_dataset,
                        "question": question_text,
                        "answer_text": answer_text,
                        "teacher_solution_text": teacher_solution,
                        "teacher_assistant_target": str(row.get("teacher_assistant_target") or ""),
                    },
                    "opsd_teacher_user_text": teacher_user,
                }
            )

        self.dataframe = datasets.Dataset.from_list(converted)
        if self.max_samples > 0 and self.max_samples < len(self.dataframe):
            self.dataframe = self.dataframe.select(range(self.max_samples))

        before_filter = len(self.dataframe)
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
        filtered = before_filter - len(self.dataframe)

        logger.warning(
            "Loaded OPSD VERL dataset: rows=%s skipped=%s filtered_overlong=%s files=%s",
            len(self.dataframe),
            skipped,
            filtered,
            self.data_files,
        )

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        row_dict.pop(self.image_key, None)
        if self.image_key != "images":
            row_dict.pop("images", None)
        teacher_user = str(row_dict.pop("opsd_teacher_user_text", "") or "")
        if not teacher_user:
            extra_info = row_dict.get("extra_info") or {}
            question_text = str(extra_info.get("question") or "")
            teacher_solution = str(extra_info.get("teacher_solution_text") or "")
            teacher_user = DEFAULT_TEACHER_TEMPLATE.format(
                question_text=question_text,
                reference_solution=teacher_solution,
            )

        teacher_messages = _build_teacher_messages(
            system_prompt=getattr(self, "opsd_system_prompt", ""),
            user_text=teacher_user,
        )
        teacher_prompt_ids = _apply_chat_template_ids(self.tokenizer, teacher_messages)
        max_prompt_length = int(getattr(self, "opsd_teacher_max_prompt_length", 4096))
        if max_prompt_length > 0 and len(teacher_prompt_ids) > max_prompt_length:
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]
        row_dict["opsd_teacher_prompt_ids"] = teacher_prompt_ids
        return row_dict
