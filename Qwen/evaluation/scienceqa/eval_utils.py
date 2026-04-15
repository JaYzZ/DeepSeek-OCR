from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from Qwen.evaluation.utils import strip_thinking_tokens


ALPHA_MAP = ["A", "B", "C", "D", "E", "F", "G"]


def extract_answer(text: Any, choices: list[str]) -> str:
    text = strip_thinking_tokens(text)
    if "[Answer]" in text:
        text = text.split("[Answer]")[-1].split("[Rationale]")[0].split("[Context]")[0]

    pattern = re.compile(r"\(([A-Za-z])\)")
    matches = pattern.findall(text)
    if matches:
        return matches[-1].upper()

    normalized = re.sub(r"[\n.,!?]", " ", str(text))
    tokens = normalized.split()
    for label in reversed(ALPHA_MAP[: len(choices)]):
        if label in tokens or label.lower() in tokens:
            return label

    lowered = str(text).lower()
    found = []
    for i, choice in enumerate(choices):
        if choice.lower() in lowered:
            found.append(ALPHA_MAP[i])
    return found[-1] if found else "FAILED"


def judge_answer(text: Any, choices: list[str], answer: str | int) -> bool:
    if isinstance(answer, int):
        answer = ALPHA_MAP[answer]
    return extract_answer(text, choices) == str(answer).upper()


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    by_topic = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for record in records:
        annotation = dict(record["annotation"])
        prediction = record["result"]["gen"]
        choices = list(annotation["choices"])
        answer = annotation["answer"]
        subject = annotation.get("subject", "unknown")
        topic = annotation.get("topic", "unknown")
        is_correct = judge_answer(prediction, choices, answer)
        correct += int(is_correct)
        by_subject[subject]["correct"] += int(is_correct)
        by_subject[subject]["total"] += 1
        by_topic[subject][topic]["correct"] += int(is_correct)
        by_topic[subject][topic]["total"] += 1

    total = len(records)
    summary_by_subject = {}
    for subject, stats in by_subject.items():
        summary_by_subject[subject] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": (stats["correct"] / stats["total"]) if stats["total"] else 0.0,
        }

    summary_by_topic = {}
    for subject, topics in by_topic.items():
        summary_by_topic[subject] = {}
        for topic, stats in topics.items():
            summary_by_topic[subject][topic] = {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": (stats["correct"] / stats["total"]) if stats["total"] else 0.0,
            }

    return {
        "overall_accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
        "by_subject": summary_by_subject,
        "by_topic": summary_by_topic,
    }


def save_eval_summary(path: str, summary: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
