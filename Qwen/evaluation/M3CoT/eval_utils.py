from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

try:
    from ..utils import strip_thinking_tokens
except ImportError:
    from Qwen.evaluation.utils import strip_thinking_tokens


ALPHA_MAP = ["A", "B", "C", "D", "E", "F", "G"]


def judge_answer(text: Any, choices: list[str], answer: str | int) -> bool:
    if isinstance(answer, int):
        answer = ALPHA_MAP[answer]

    text = strip_thinking_tokens(text)
    if "[Answer]" in text:
        text = text.split("[Answer]")[-1].split("[Rationale]")[0].split("[Context]")[0]

    pattern = re.compile(r"\(([A-Za-z])\)")
    res = pattern.findall(text)
    if len(res) >= 1:
        pred = res[-1].upper()
    else:
        matches: list[str] = []
        lowered = text.lower()
        for i, choice in enumerate(choices):
            if choice.lower() in lowered:
                matches.append(ALPHA_MAP[i])
        if matches:
            pred = matches[-1]
        else:
            normalized = re.sub(r"[\n.,!?]", " ", text)
            for i, _choice in enumerate(choices):
                if ALPHA_MAP[i] in normalized.split(" "):
                    matches.append(ALPHA_MAP[i])
            if matches:
                pred = matches[-1]
            else:
                for i, _choice in enumerate(choices):
                    if ALPHA_MAP[i].lower() in normalized.split(" "):
                        matches.append(ALPHA_MAP[i])
                pred = matches[-1] if matches else "FAILED"
    return pred == answer


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    by_domain = defaultdict(lambda: {"correct": 0, "total": 0})
    by_topic = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for record in records:
        annotation = dict(record["annotation"])
        prediction = record["result"]["gen"]
        choices = list(annotation["choices"])
        answer = annotation["answer"]
        domain = annotation.get("domain", "unknown")
        topic = annotation.get("topic", "unknown")
        is_correct = judge_answer(prediction, choices, answer)
        correct += int(is_correct)
        by_domain[domain]["correct"] += int(is_correct)
        by_domain[domain]["total"] += 1
        by_topic[domain][topic]["correct"] += int(is_correct)
        by_topic[domain][topic]["total"] += 1

    total = len(records)
    summary_by_domain = {}
    for domain, stats in by_domain.items():
        summary_by_domain[domain] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": (stats["correct"] / stats["total"]) if stats["total"] else 0.0,
        }

    summary_by_topic = {}
    for domain, topics in by_topic.items():
        summary_by_topic[domain] = {}
        for topic, stats in topics.items():
            summary_by_topic[domain][topic] = {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": (stats["correct"] / stats["total"]) if stats["total"] else 0.0,
            }

    return {
        "overall_accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
        "by_domain": summary_by_domain,
        "by_topic": summary_by_topic,
    }


def save_eval_summary(path: str, summary: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
