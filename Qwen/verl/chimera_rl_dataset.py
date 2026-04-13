#!/usr/bin/env python3
"""Chimera RL Dataset - reads Chimera parquet files directly for RL training.

This dataset class loads Chimera parquet files on-the-fly without requiring
a separate VERL dataset build step. It converts Chimera's format to the
VERL RL format automatically.

Usage:
    In your RL config, set:
        data:
          custom_cls:
            path: Qwen.verl.chimera_rl_dataset
            name: ChimeraRLDataset
"""

import copy
import logging
import re
from typing import Optional

import datasets
import numpy as np
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

logger = logging.getLogger(__name__)
CHIMERA_SYSTEM_PROMPT = (
    "You are solving a challenging academic problem from a rendered question image. "
    "Reason carefully. In the final answer, provide only the final result. "
    "If the problem has labeled subparts, keep the part labels and provide only the final answer for each part. "
    "If the problem has a single final answer, put it in \\boxed{}. "
    "Do not restate the full question or include unnecessary explanation in the final answer."
)


def _build_user_content(prompt_text: str, *, has_image: bool):
    if not has_image:
        return prompt_text
    return [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]


def _build_problem_context(subject: str, topic: str) -> str:
    parts = []
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


def _extract_equivalent_answers(answer: str, solution: str, original_solution: str) -> list:
    """Extract equivalent answer forms for robust reward computation."""
    answers = [answer]

    # Extract boxed answers from solution
    boxed_re = re.compile(r"\\boxed\s*{([^}]+)}")
    for match in boxed_re.finditer(solution or ""):
        boxed_answer = match.group(1).strip()
        if boxed_answer and boxed_answer not in answers:
            answers.append(boxed_answer)

    # Also check original_solution for boxed answers
    for match in boxed_re.finditer(original_solution or ""):
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


class ChimeraRLDataset(RLHFDataset):
    """
    Load Chimera parquet files directly for RL training.

    Converts Chimera format (question, solution, answer, original_solution, topic)
    to VERL format (prompt, images, reward_model, extra_info) on-the-fly.

    By default, uses pre-rendered question images from the SFT dataset building phase.
    Set --text-only flag to use text input instead.

    No pre-conversion step needed - just point to the Chimera parquet files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # OPTIMIZATION 6: Pre-index available images to avoid disk checks
        self._image_index = self._build_image_index()

    def _build_image_index(self) -> set:
        """Build index of available images for fast lookup."""
        import os
        from pathlib import Path

        chimera_images_dir = self.config.get("chimera_images_dir", "Qwen/data/chimera_images")
        text_only = self.config.get("text_only", False)

        if text_only:
            return set()

        # Resolve absolute path
        if not os.path.isabs(chimera_images_dir):
            repo_root = Path(__file__).parent.parent.parent
            chimera_images_dir = repo_root / chimera_images_dir

        # Build index of existing image files
        image_index = set()
        if os.path.exists(chimera_images_dir):
            for f in os.listdir(chimera_images_dir):
                if f.endswith("_question.png"):
                    image_index.add(f.replace("_question.png", ""))

        logger.info(f"Pre-indexed {len(image_index)} available question images")
        return image_index

    def _read_files_and_tokenize(self):
        """Read Chimera parquet files and convert to VERL format."""
        import os
        from pathlib import Path

        # Check if text-only mode is enabled
        text_only = self.config.get("text_only", False)
        chimera_images_dir = self.config.get("chimera_images_dir", "Qwen/data/chimera_images")

        dataframes = []
        for parquet_file in self.data_files:
            # OPTIMIZATION 1: Use memory mapping for faster loading
            chimera_df = datasets.load_dataset(
                "parquet",
                data_files=parquet_file,
                memory_mapping=True
            )["train"]

            # Convert Chimera format to VERL format
            verl_data = []
            for row in chimera_df:
                try:
                    # Extract fields from Chimera format
                    question = str(row.get("question", "")).strip()
                    answer = str(row.get("answer", "")).strip()
                    solution = str(row.get("solution", "")).strip()
                    original_solution = str(row.get("original_solution", "")).strip()
                    subject = str(row.get("subject", "general")).strip()
                    topic = str(row.get("topic", "general")).strip()
                    index = int(row.get("index", 0))
                    correctness = row.get("correctness")

                    # Skip samples without required fields. Keep verifier-failed
                    # traces because RL reward uses the canonical answer.
                    if not question or not answer:
                        continue

                    # Build prompt messages and images
                    # Try to use pre-rendered question image from SFT dataset
                    sample_id = f"chimera_{int(index):07d}"

                    images = []
                    if text_only:
                        # Text-only mode: no images, question in prompt
                        user_content = _build_user_prompt_text(
                            question,
                            subject=subject,
                            topic=topic,
                            has_image=False,
                        )
                    else:
                        # OPTIMIZATION 6: Use pre-indexed image lookup instead of disk check
                        if sample_id in self._image_index:
                            # Build full path for existing image
                            question_image_path = f"{chimera_images_dir}/{sample_id}_question.png"
                            if not os.path.isabs(question_image_path):
                                repo_root = Path(__file__).parent.parent.parent
                                question_image_path = str(repo_root / question_image_path)
                            images = [question_image_path]
                            user_content = _build_user_content(
                                _build_user_prompt_text(
                                    question,
                                    subject=subject,
                                    topic=topic,
                                    has_image=True,
                                ),
                                has_image=True,
                            )
                        else:
                            # Fallback to text if image not found
                            user_content = _build_user_content(
                                _build_user_prompt_text(
                                    question,
                                    subject=subject,
                                    topic=topic,
                                    has_image=False,
                                ),
                                has_image=False,
                            )

                    prompt = [
                        {"role": "system", "content": CHIMERA_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ]

                    # Extract equivalent answers
                    equivalent_answers = _extract_equivalent_answers(answer, solution, original_solution)

                    # Build VERL-format record
                    verl_row = {
                        "prompt": prompt,
                        "images": images,
                        "data_source": f"chimera::{subject or topic}",
                        "ability": topic if topic else subject or "reasoning",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": answer,
                            "equivalent_answers": equivalent_answers,
                        },
                        "extra_info": {
                            "split": parquet_file,
                            "index": index,
                            "question": question,
                            "subject": subject,
                            "topic": topic,
                            "correctness": bool(correctness) if correctness is not None else None,
                            "text_only": text_only,
                        },
                    }
                    verl_data.append(verl_row)

                except Exception as e:
                    logger.warning(f"Error processing Chimera row: {e}")
                    continue

            # Convert to HuggingFace Dataset
            if verl_data:
                verl_df = datasets.Dataset.from_list(verl_data)
                dataframes.append(verl_df)

        if not dataframes:
            raise ValueError("No valid Chimera samples found in parquet files")

        # Concatenate all dataframes
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"Chimera dataset len: {len(self.dataframe)}")

        # Apply max_samples limit if specified
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
