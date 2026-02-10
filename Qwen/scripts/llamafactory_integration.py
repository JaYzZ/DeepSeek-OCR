"""
LlamaFactory integration for Qwen3VL training with latent supervision.

This module patches the training pipeline to support:
1. Latent injection: Inject pre-encoded features at <|latent_step|> positions
2. Latent supervision: Load supervision targets for REPA loss computation
3. Thinking loss: Combine CE loss with REPA loss on latent predictions

To enable: Set environment variable QWEN3VL_LATENT_SUPERVISION=1
"""

import functools
import gc
import logging
import os
import traceback
from collections import defaultdict
from typing import Any, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# LlamaFactory imports (lazy loaded, may not be available in all contexts)
try:
    import datasets
    from llamafactory.data import SFTDataCollatorWith4DAttentionMask, loader as loader_module
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.mm_plugin import Qwen3VLPlugin
    from llamafactory.data.template import (
        FunctionFormatter,
        ReasoningTemplate,
        StringFormatter,
        ToolFormatter,
        register_template,
    )
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    from transformers import PreTrainedModel
    LLAMAFACTORY_AVAILABLE = True
except ImportError as e:
    LLAMAFACTORY_AVAILABLE = False
    _IMPORT_ERROR = str(e)

try:
    from OCRVL.llamafactory.transparent_eval_callback import TransparentEvalCallback
    TRANSPARENT_EVAL_AVAILABLE = True
except ImportError:
    TRANSPARENT_EVAL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Ensure logger outputs even if logging not yet configured
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Avoid duplicate logs


def _register_qwen3vl_latent_template(logger) -> None:
    """Register the qwen3vl_latent template for LlamaFactory."""
    if not LLAMAFACTORY_AVAILABLE:
        logger.warning(f"[Qwen3VL Latent] LlamaFactory not available: {_IMPORT_ERROR}")
        return

    try:
        # Create multimodal plugin instance with all required tokens
        mm_plugin = Qwen3VLPlugin(
            image_token="<|image_pad|>",
            video_token=None,  # Qwen3VL doesn't use video_token
            audio_token=None,  # Qwen3VL doesn't use audio_token
        )

        # Register the template
        register_template(
            name="qwen3vl_latent",
            format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
            format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
            format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
            format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
            format_observation=StringFormatter(
                slots=["<|im_start|>user\n\n{{content}}\n<|im_end|>\n<|im_start|>assistant\n"]
            ),
            format_tools=ToolFormatter(tool_format="qwen"),
            stop_words=["<|im_end|>"],
            replace_eos=True,
            mm_plugin=mm_plugin,
            template_class=ReasoningTemplate,
        )
        logger.debug("[Qwen3VL Latent] ✓ Registered qwen3vl_latent template with Qwen3VLPlugin")
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Template registration failed: {e}")
        raise


def _patch_qwen2vl_image_processor(logger) -> None:
    """No-op: grid_thw patch not needed, vision model handles pre-merge values correctly."""
    pass


def _patch_once() -> None:
    """Apply patches once per process."""
    # Check if latent supervision is enabled
    if os.environ.get("QWEN3VL_LATENT_SUPERVISION", "0") != "1":
        return

    if not LLAMAFACTORY_AVAILABLE:
        logger.warning(f"[Qwen3VL Latent] LlamaFactory not available: {_IMPORT_ERROR}")
        return

    # PID-based guard
    pid = str(os.getpid())
    if os.environ.get("QWEN3VL_LATENT_PATCHED_PID", "") == pid:
        return
    os.environ["QWEN3VL_LATENT_PATCHED_PID"] = pid

    logger.info("[Qwen3VL Latent] Starting latent supervision integration...")

    # Grid_thw handling: vision model manages pre-merge values correctly
    _patch_qwen2vl_image_processor(logger)

    # Register Qwen3VL latent template (before any other patches)
    try:
        _register_qwen3vl_latent_template(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to register template: {e}")

    # Patch 1: Add thinking_projection module to model
    try:
        _patch_model_for_thinking_projection(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch model for thinking_projection: {e}")

    # Patch 2: Patch dataset converter to preserve latent fields
    try:
        _patch_dataset_converter(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch dataset converter: {e}")

    # Patch 3: Patch dataset preprocessing to preserve latent columns
    try:
        _patch_dataset_preprocessing(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch dataset preprocessing: {e}")

    # Patch 3b: Disable dataset-level packing when pack-after-injection is enabled
    try:
        _patch_packed_dataset_processor(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch packed dataset processor: {e}")

    # Patch 4: Patch data collator to load latent supervision
    try:
        _patch_data_collator(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch data collator: {e}")

    # Patch 5: Patch forward pass to inject latents and compute thinking loss
    try:
        _patch_model_forward(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch model forward: {e}")

    # Patch 6: Register transparent eval callback
    try:
        _patch_trainer_callback(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch trainer callback: {e}")

    # Patch 7: Patch generate() to filter out latent supervision kwargs
    try:
        _patch_model_generate(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch model generate: {e}")

    logger.info("[Qwen3VL Latent] ✓ Latent supervision integration complete")


def _patch_model_for_thinking_projection(logger) -> None:
    """No-op: supervision on LLM hidden states directly, no projection layer needed."""
    logger.debug("[Qwen3VL Latent] Using direct LLM hidden state supervision (no projection needed)")


def _patch_dataset_converter(logger) -> None:
    """Patch SharegptDatasetConverter to preserve latent supervision fields.

    The standard converter only extracts: messages, images, videos, audios, tools, system
    We need to also preserve: latent_ground_truth, latent_supervision, num_latent_steps
    """
    # Store original __call__
    original_call = SharegptDatasetConverter.__call__

    @functools.wraps(original_call)
    def wrapped_call(self, example: dict[str, Any]) -> dict[str, Any]:
        """Wrapped converter that preserves latent supervision fields."""
        # Call original converter
        result = original_call(self, example)

        # Preserve latent supervision fields if present
        latent_fields = ['latent_ground_truth', 'latent_supervision', 'num_latent_steps']
        for field in latent_fields:
            if field in example:
                result[field] = example[field]

        return result

    # Apply patch
    SharegptDatasetConverter.__call__ = wrapped_call
    logger.debug("[Qwen3VL Latent] ✓ Patched SharegptDatasetConverter.__call__ to preserve latent fields")


def _patch_dataset_preprocessing(logger) -> None:
    """Patch dataset preprocessing to preserve latent supervision columns.

    LlamaFactory's _get_preprocessed_dataset uses remove_columns=column_names
    which strips ALL original columns including our latent fields.
    This patch preserves latent columns during preprocessing.
    """
    # Store original function
    original_get_preprocessed_dataset = loader_module._get_preprocessed_dataset

    @functools.wraps(original_get_preprocessed_dataset)
    def wrapped_get_preprocessed_dataset(
        dataset,
        data_args,
        training_args,
        stage,
        template,
        tokenizer,
        processor=None,
        is_eval=False,
    ):
        """Wrapped preprocessing that preserves latent columns."""
        # Get column names before preprocessing
        if dataset is not None:
            try:
                column_names = list(next(iter(dataset)).keys())
                latent_columns = [col for col in column_names if col in ['latent_ground_truth', 'latent_supervision', 'num_latent_steps']]

                if latent_columns:
                    logger.debug(f"[Qwen3VL Latent] Preserving latent columns during preprocessing: {latent_columns}")
            except StopIteration:
                latent_columns = []
        else:
            latent_columns = []

        # Call original preprocessing
        dataset = original_get_preprocessed_dataset(
            dataset=dataset,
            data_args=data_args,
            training_args=training_args,
            stage=stage,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            is_eval=is_eval,
        )

        # Note: We can't easily restore columns here because the original map() removes them.
        # Instead, we need to patch the actual dataset.map() call or use a different approach.
        # For now, this is a limitation - the columns are lost during map().
        # The real fix is to not use remove_columns or to use keep_columns instead.
        logger.warning(f"[Qwen3VL Latent] Latent columns {latent_columns} are removed by dataset.map() - need alternative approach")

        return dataset

    # Alternative: Patch dataset.map() to not remove latent columns
    # This is more invasive but actually works
    try:
        original_dataset_map = None

        def safe_map_with_latent_preservation(dataset, *args, **kwargs):
            """Wrapped map that preserves latent columns."""
            # Get original remove_columns
            remove_columns = kwargs.get('remove_columns', None)

            if remove_columns is not None and isinstance(remove_columns, list):
                # Check if we have latent columns to preserve
                try:
                    sample = next(iter(dataset))
                    column_names = list(sample.keys())
                    latent_columns = [col for col in column_names if col in ['latent_ground_truth', 'latent_supervision', 'num_latent_steps']]

                    if latent_columns:
                        # Remove non-latent columns only
                        keep_columns = latent_columns
                        new_remove_columns = [col for col in remove_columns if col not in keep_columns]
                        kwargs['remove_columns'] = new_remove_columns
                        logger.debug(f"[Qwen3VL Latent] Preserving {len(keep_columns)} latent columns during map()")
                except (StopIteration, AttributeError):
                    pass

            # Call original map with modified kwargs
            return original_dataset_map(dataset, *args, **kwargs)

        # Apply to the Dataset class
        original_dataset_map = datasets.Dataset.map
        datasets.Dataset.map = safe_map_with_latent_preservation
        logger.debug("[Qwen3VL Latent] ✓ Patched datasets.Dataset.map to preserve latent columns")
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch Dataset.map: {e}")

    # Apply the loader patch (as backup/alternative)
    # loader_module._get_preprocessed_dataset = wrapped_get_preprocessed_dataset
    logger.debug("[Qwen3VL Latent] ✓ Patched dataset preprocessing to preserve latent columns")


def _patch_packed_dataset_processor(logger) -> None:
    """Disable dataset-level packing so we can pack after injection."""
    try:
        from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor, SupervisedDatasetProcessor
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to import dataset processors: {e}")
        return

    original_preprocess = PackedSupervisedDatasetProcessor.preprocess_dataset

    @functools.wraps(original_preprocess)
    def wrapped_preprocess(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if not hasattr(wrapped_preprocess, "_logged_skip"):
            logger.debug("[Qwen3VL Latent] pack-after-injection: skipping dataset-level packing")
            wrapped_preprocess._logged_skip = True
        return SupervisedDatasetProcessor.preprocess_dataset(self, examples)

    PackedSupervisedDatasetProcessor.preprocess_dataset = wrapped_preprocess
    logger.debug("[Qwen3VL Latent] ✓ Patched PackedSupervisedDatasetProcessor for pack-after-injection")


def _get_cutoff_len_from_collator(collator, logger) -> int:
    env_cutoff = os.environ.get("QWEN3VL_CUTOFF_LEN", None)
    if env_cutoff is not None:
        try:
            cutoff = int(env_cutoff)
            if cutoff > 0:
                return cutoff
        except ValueError:
            pass

    cutoff = getattr(collator, "max_length", None)
    if cutoff is not None and cutoff > 0:
        return int(cutoff)

    cutoff = getattr(getattr(collator, "tokenizer", None), "model_max_length", None)
    if cutoff is None or cutoff <= 0 or cutoff > 100000:
        raise ValueError(
            "[Qwen3VL Latent] pack-after-injection requires a valid cutoff_len. "
            "Export QWEN3VL_CUTOFF_LEN (e.g., 2048)."
        )
    return int(cutoff)


def _latent_seq_len(feat: torch.Tensor) -> int:
    if feat.dim() == 1:
        return 1
    if feat.dim() == 2:
        return int(feat.shape[0])
    raise ValueError(f"Unexpected latent tensor dim: {feat.dim()}")


def _expand_sample_for_latent_injection(
    input_ids: list[int],
    labels: list[int],
    latent_token_id: int,
    thinking_start_id: int,
    thinking_end_id: int,
    latent_ground_truth: list[torch.Tensor],
    ignore_index: int,
) -> tuple[list[int], list[int]]:
    """Expand <|latent_step|> tokens into placeholder sequences to match latent lengths.

    OPTIMIZED: Uses torch tensors for O(1) lookups instead of O(n) scans.
    """
    import torch

    # Convert to tensor for O(1) indexing (faster than list.index() which is O(n))
    ids_tensor = torch.tensor(input_ids, dtype=torch.long)

    # Find thinking bounds using torch.where (O(1) overall)
    thinking_mask = (ids_tensor == thinking_start_id) | (ids_tensor == thinking_end_id)
    thinking_positions = torch.nonzero(thinking_mask).flatten()

    if len(thinking_positions) < 2:
        return input_ids, labels

    start_pos = thinking_positions[0].item()
    end_pos = thinking_positions[-1].item()

    if start_pos >= end_pos or ids_tensor[start_pos] != thinking_start_id:
        return input_ids, labels

    # Find latent tokens in thinking section (vectorized)
    thinking_section = ids_tensor[start_pos + 1:end_pos]
    latent_mask = (thinking_section == latent_token_id)
    latent_indices = torch.nonzero(latent_mask).flatten() + start_pos + 1

    if len(latent_indices) == 0 or not latent_ground_truth:
        return input_ids, labels

    latent_indices = latent_indices.tolist()

    if len(latent_indices) != len(latent_ground_truth):
        raise ValueError(
            "[Qwen3VL Latent] pack-after-injection mismatch: "
            f"found {len(latent_indices)} latent tokens but {len(latent_ground_truth)} "
            "latent_ground_truth tensors."
        )

    # Pre-compute expansion lengths
    expansion_lengths = [_latent_seq_len(feat) for feat in latent_ground_truth]

    # Build new sequences using torch.cat (faster than list.extend)
    # This is the key optimization - use tensor concatenation
    result_segments = []

    # Add tokens before first latent
    result_segments.append(ids_tensor[:latent_indices[0]])

    for i, (latent_idx, exp_len) in enumerate(zip(latent_indices, expansion_lengths)):
        # Add repeated latent tokens
        result_segments.append(torch.full((exp_len,), latent_token_id, dtype=torch.long))

        # Add tokens between latents
        if i < len(latent_indices) - 1:
            next_idx = latent_indices[i + 1]
            result_segments.append(ids_tensor[latent_idx + 1:next_idx])
        else:
            # Last latent - add remaining tokens
            result_segments.append(ids_tensor[latent_idx + 1:])

    # Concatenate all at once (much faster than list.extend in loop)
    new_input_ids_tensor = torch.cat(result_segments).tolist()

    # Build labels the same way
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Create ignore labels for latent positions
    num_tokens = result_segments[0].shape[0]  # Before first latent
    for i, (_, exp_len) in enumerate(zip(latent_indices, expansion_lengths)):
        # After previous segment, up to this latent
        if i < len(latent_indices) - 1:
            num_tokens += (latent_indices[i+1] - latent_indices[i] - 1)
        else:
            num_tokens += (latent_indices[i] - latent_indices[i] - 1)
        # Latent tokens (all ignore)
        num_tokens += exp_len

    # Before first latent
    new_labels_tensor = labels_tensor[:latent_indices[0]]

    # Add ignore labels for latent tokens and content between
    current_pos = latent_indices[0]
    for i, (latent_idx, exp_len) in enumerate(zip(latent_indices, expansion_lengths)):
        # Latent tokens (ignore)
        new_labels_tensor = torch.cat([
            new_labels_tensor,
            torch.full((exp_len,), ignore_index, dtype=torch.long)
        ])
        current_pos += 1  # Move past the latent_token we're replacing

        # Content between latents (keep original labels)
        if i < len(latent_indices) - 1:
            next_idx = latent_indices[i + 1]
            segment_len = next_idx - latent_idx - 1
            new_labels_tensor = torch.cat([new_labels_tensor, labels_tensor[latent_idx + 1:next_idx]])
            current_pos = next_idx
        else:
            # After last latent - keep remaining labels
            new_labels_tensor = torch.cat([new_labels_tensor, labels_tensor[latent_idx + 1:]])

    return new_input_ids_tensor, new_labels_tensor.tolist()


def _pack_features_after_injection(
    batch: List[dict],
    latent_fields_list: List[dict],
    collator,
    logger,
) -> tuple[List[dict], List[dict]]:
    """Pack sequences after expanding latent placeholders to match injected lengths."""
    cutoff_len = _get_cutoff_len_from_collator(collator, logger)
    latent_token_id = int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669"))
    thinking_start_id = int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667"))
    thinking_end_id = int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668"))
    ignore_index = getattr(collator, "label_pad_token_id", -100)
    pad_token_id = collator.tokenizer.pad_token_id
    block_diag_attn = getattr(collator, "block_diag_attn", False)

    if not hasattr(_pack_features_after_injection, "_logged"):
        logger.debug(
            f"[Qwen3VL Latent] pack-after-injection enabled: cutoff_len={cutoff_len}, "
            f"block_diag_attn={block_diag_attn}"
        )
        _pack_features_after_injection._logged = True

    expanded_features: List[dict] = []
    expanded_latent_fields: List[dict] = []
    lengths: List[int] = []
    # OPTIMIZED: No longer need length2indexes with first-fit decreasing algorithm
    dropped = 0
    max_seen_len = 0

    for idx, feature in enumerate(batch):
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        latent_fields = latent_fields_list[idx] if idx < len(latent_fields_list) else {}

        latent_gt = latent_fields.get("latent_ground_truth") or []
        if latent_gt:
            latent_gt = _load_latent_tensors(latent_gt)
            latent_fields["latent_ground_truth"] = latent_gt

        new_input_ids, new_labels = _expand_sample_for_latent_injection(
            input_ids=input_ids,
            labels=labels,
            latent_token_id=latent_token_id,
            thinking_start_id=thinking_start_id,
            thinking_end_id=thinking_end_id,
            latent_ground_truth=latent_gt,
            ignore_index=ignore_index,
        )

        length = len(new_input_ids)
        if length > max_seen_len:
            max_seen_len = length
        if length > cutoff_len:
            logger.warning(
                f"[Qwen3VL Latent] Dropped example: expanded length {length} > cutoff_len {cutoff_len}."
            )
            dropped += 1
            continue

        new_feature = dict(feature)
        new_feature["input_ids"] = new_input_ids
        new_feature["labels"] = new_labels
        new_feature["attention_mask"] = [1] * len(new_input_ids)

        # OPTIMIZED: No longer need length2indexes with first-fit decreasing
        lengths.append(length)
        expanded_features.append(new_feature)
        expanded_latent_fields.append(latent_fields)

    if not lengths:
        logger.warning("[Qwen3VL Latent] pack-after-injection: no valid samples after expansion.")
        return batch, latent_fields_list

    # Aggregate drop stats (log occasionally)
    stats = getattr(_pack_features_after_injection, "_stats", None)
    if stats is None:
        stats = {"total": 0, "dropped": 0, "last_log": 0, "max_len": 0}
    stats["total"] += len(batch)
    stats["dropped"] += dropped
    stats["max_len"] = max(stats["max_len"], max_seen_len)
    if stats["total"] - stats["last_log"] >= 1000:
        drop_rate = (stats["dropped"] / max(1, stats["total"])) * 100.0
        logger.info(
            "[Qwen3VL Latent] pack-after-injection drop stats: "
            f"dropped={stats['dropped']}/{stats['total']} ({drop_rate:.2f}%), "
            f"max_seen_len={stats['max_len']}, cutoff_len={cutoff_len}"
        )
        stats["last_log"] = stats["total"]
    _pack_features_after_injection._stats = stats

    # OPTIMIZED: First-fit decreasing bin packing (O(n log n) instead of O(n²))
    # Sort by length decreasing, then assign to bins using first-fit
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)

    knapsacks = []
    current_knapsack = []
    current_sum = 0

    for idx in sorted_indices:
        length = lengths[idx]
        if length > cutoff_len:
            # Drop this sample
            dropped += 1
            stats['dropped'] += 1
            continue

        if current_sum + length <= cutoff_len:
            # Add to current knapsack
            current_knapsack.append(idx)
            current_sum += length
        else:
            # Start new knapsack
            if current_knapsack:
                knapsacks.append(current_knapsack)
            current_knapsack = [idx]
            current_sum = length

    # Don't forget the last knapsack
    if current_knapsack:
        knapsacks.append(current_knapsack)
    packed_features: List[dict] = []
    packed_latent_fields: List[dict] = []

    for knapsack in knapsacks:
        # Pre-compute total lengths for pre-allocation
        # OPTIMIZED: knapsack contains indices, not lengths
        total_len = sum(lengths[idx] for idx in knapsack)

        # Use list comprehension for faster initialization
        packed_input_ids: list[int] = [0] * total_len
        packed_labels: list[int] = [0] * total_len
        packed_attention_mask: list[int] = [0] * total_len

        # Use lists for mutable accumulation (extend faster than +=)
        packed_images: list = []
        packed_videos: list = []
        packed_audios: list = []
        packed_latent_gt: list = []
        packed_latent_sup: list = []
        packed_num_latent_steps: list = []

        # Track current position for in-place assignment (faster than extend)
        current_pos = 0
        # OPTIMIZED: knapsack contains indices directly
        for seg_idx, index in enumerate(knapsack):
            feature = expanded_features[index]
            feat_input_ids = feature["input_ids"]
            feat_labels = feature["labels"]
            feat_len = len(feat_input_ids)

            # In-place assignment (faster than +=)
            packed_input_ids[current_pos:current_pos + feat_len] = feat_input_ids
            packed_labels[current_pos:current_pos + feat_len] = feat_labels

            # Vectorized attention mask creation
            attn_val = seg_idx + 1 if block_diag_attn else 1
            packed_attention_mask[current_pos:current_pos + feat_len] = [attn_val] * feat_len

            current_pos += feat_len

            # Extend lists (slightly faster than +=)
            images = feature.get("images")
            if images:
                packed_images.extend(images)
            videos = feature.get("videos")
            if videos:
                packed_videos.extend(videos)
            audios = feature.get("audios")
            if audios:
                packed_audios.extend(audios)

            latent_fields = expanded_latent_fields[index] if index < len(expanded_latent_fields) else {}
            latent_gt = latent_fields.get("latent_ground_truth")
            if latent_gt:
                packed_latent_gt.extend(latent_gt)
            latent_sup = latent_fields.get("latent_supervision")
            if latent_sup:
                packed_latent_sup.extend(latent_sup)
            num_steps = latent_fields.get("num_latent_steps")
            if num_steps is not None:
                if isinstance(num_steps, list):
                    packed_num_latent_steps.extend(num_steps)
                else:
                    packed_num_latent_steps.append(num_steps)

        # Trim to actual size
        packed_input_ids = packed_input_ids[:current_pos]
        packed_labels = packed_labels[:current_pos]
        packed_attention_mask = packed_attention_mask[:current_pos]

        # Padding (vectorized)
        if len(packed_input_ids) < cutoff_len + 1:
            pad_length = cutoff_len - len(packed_input_ids) + 1
            packed_input_ids.extend([pad_token_id] * pad_length)
            packed_labels.extend([ignore_index] * pad_length)
            packed_attention_mask.extend([0] * pad_length)

        if len(packed_input_ids) != cutoff_len + 1:
            raise ValueError(
                "[Qwen3VL Latent] pack-after-injection: packed length mismatch "
                f"{len(packed_input_ids)} != cutoff_len+1 ({cutoff_len + 1})."
            )

        packed_features.append(
            {
                "input_ids": packed_input_ids,
                "attention_mask": packed_attention_mask,
                "labels": packed_labels,
                "images": packed_images or None,
                "videos": packed_videos or None,
                "audios": packed_audios or None,
            }
        )
        packed_latent_fields.append(
            {
                "latent_ground_truth": packed_latent_gt,
                "latent_supervision": packed_latent_sup,
                "num_latent_steps": packed_num_latent_steps,
            }
        )

    return packed_features, packed_latent_fields


def _patch_data_collator(logger) -> None:
    """Patch data collator to load latent supervision from disk.

    This modifies the data collation to:
    1. Load latent_ground_truth tensors for injection (thinking features)
    2. Load latent_supervision tensors for OT loss (original image features)
    3. Compute latent_positions mask from input_ids

    Data format:
    - latent_ground_truth: [thinking_0.latent.pt, thinking_1.latent.pt, ...]
    - latent_supervision: [original_superv_0.latent.pt, original_superv_1.latent.pt, ...]
      Both lists have the same length (one per thinking chunk)
    """
    # Store original __call__ method
    original_call = SFTDataCollatorWith4DAttentionMask.__call__

    @functools.wraps(original_call)
    def wrapped_call(self, batch: List[dict]) -> dict:
        """Wrapped collator that loads latent supervision."""
        import time
        total_start = time.time()

        # Snapshot image paths per sample for debug (original collator mutates batch)
        batch_images_per_sample = []
        if batch and isinstance(batch[0], dict):
            for item in batch:
                images = item.get("images", None) or []
                batch_images_per_sample.append(list(images))

        # Extract latent fields BEFORE calling original collator
        # (tokenizer can't handle these fields)
        latent_fields_list = []
        has_latent = False

        if batch and isinstance(batch[0], dict):
            for item in batch:
                latent_item = {}
                for key in ['latent_ground_truth', 'latent_supervision', 'num_latent_steps']:
                    if key in item:
                        latent_item[key] = item.pop(key)  # Remove from batch item
                        has_latent = True
                latent_fields_list.append(latent_item)

            if has_latent and not hasattr(wrapped_call, '_logged_extract'):
                logger.debug(f"[Qwen3VL Latent] Extracted latent fields from batch (tokenizer won't see them)")
                wrapped_call._logged_extract = True

        # Pack-after-injection path (always enabled when latent supervision is active)
        pack_start = time.time()
        try:
            batch, latent_fields_list = _pack_features_after_injection(
                batch=batch,
                latent_fields_list=latent_fields_list,
                collator=self,
                logger=logger,
            )
            has_latent = any(bool(item) for item in latent_fields_list)
            batch_images_per_sample = [
                list(item.get("images", None) or []) for item in batch
            ]
        except Exception as e:
            logger.warning(f"[Qwen3VL Latent] pack-after-injection failed: {e}")
            raise
        pack_time = time.time() - pack_start

        # Call original collator (tokenizer only sees standard fields)
        collator_start = time.time()
        result = original_call(self, batch)
        collator_time = time.time() - collator_start

        # LOG: Show actual result shapes (debug level)
        if 'input_ids' in result:
            logger.debug(f"[Qwen3VL COLLATOR RESULT] input_ids={result['input_ids'].shape}, total_tokens={result['input_ids'].shape[0] * result['input_ids'].shape[1]}, images={len(result.get('pixel_values', [])) if isinstance(result.get('pixel_values'), list) else (result.get('pixel_values').shape[0] if torch.is_tensor(result.get('pixel_values')) else 'N/A')}")

        # Debug: validate pixel_values length matches image_grid_thw product
        try:
            pixel_values = result.get("pixel_values", None)
            image_grid_thw = result.get("image_grid_thw", None)
            if pixel_values is not None and image_grid_thw is not None:
                pv_len = pixel_values.shape[0] if torch.is_tensor(pixel_values) else len(pixel_values)
                if torch.is_tensor(image_grid_thw):
                    grid_prod = int((image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum().item())
                    num_grids = int(image_grid_thw.shape[0])
                    if grid_prod != pv_len:
                        # Build a detailed error to pinpoint the offending batch
                        total_images = sum(len(x) for x in batch_images_per_sample)
                        pv_shape = getattr(pixel_values, "shape", None)
                        pv_dtype = getattr(pixel_values, "dtype", None)
                        proc = getattr(self, "processor", None)
                        ip = getattr(proc, "image_processor", None) if proc is not None else None
                        msg_lines = [
                            "[Qwen3VL Latent] pixel_values/image_grid_thw mismatch in collator",
                            f"  pixel_values_len: {pv_len}",
                            f"  pixel_values_shape: {pv_shape}",
                            f"  pixel_values_dtype: {pv_dtype}",
                            f"  image_grid_thw_sum: {grid_prod}",
                            f"  image_grid_thw_count: {num_grids}",
                            f"  batch_images_total: {total_images}",
                            f"  image_grid_thw: {image_grid_thw.tolist()}",
                            f"  processor: {type(proc)}",
                            f"  image_processor: {type(ip)}",
                        ]
                        # Add per-sample image paths (truncate if too long)
                        for i, paths in enumerate(batch_images_per_sample):
                            if not paths:
                                msg_lines.append(f"  sample[{i}] images: []")
                                continue
                            preview = paths if len(paths) <= 2 else paths[:2] + ["..."]
                            msg_lines.append(f"  sample[{i}] images({len(paths)}): {preview}")
                        raise ValueError("\n".join(msg_lines))
        except Exception as e:
            # Re-raise to surface exact debug info
            raise

        # Add latent supervision handling for thinking datasets
        if has_latent:
            try:
                if not hasattr(wrapped_call, '_logged_process'):
                    logger.debug(f"[Qwen3VL Latent] Processing batch with latent supervision...")
                    wrapped_call._logged_process = True
                result = _add_latent_supervision_to_batch(result, latent_fields_list, logger)
            except Exception as e:
                logger.warning(f"[Qwen3VL Latent] Failed to add latent supervision: {e}")
                import traceback
                logger.warning(traceback.format_exc())

        total_time = time.time() - total_start
        # Log timing every 10 steps to avoid spam
        if not hasattr(wrapped_call, '_timing_count'):
            wrapped_call._timing_count = 0
        wrapped_call._timing_count += 1
        if wrapped_call._timing_count <= 10 or wrapped_call._timing_count % 100 == 0:
            logger.debug(f"[Qwen3VL Latent TIMING] Step {wrapped_call._timing_count}: pack={pack_time:.3f}s, collator={collator_time:.3f}s, total={total_time:.3f}s")

        return result

    # Apply patch
    SFTDataCollatorWith4DAttentionMask.__call__ = wrapped_call
    logger.debug("[Qwen3VL Latent] ✓ Patched SFTDataCollatorWith4DAttentionMask.__call__")


def _patch_model_forward(logger) -> None:
    """Patch model forward pass to inject latents and compute thinking loss.

    This patches the model's forward method to:
    1. Inject latent_ground_truth at <|latent_step|> positions (no projection needed)
    2. Compute thinking loss using latent_supervision directly on LLM hidden states
    3. Combine CE loss with thinking loss

    APPROACH (following RoT implementation, adapted for LlamaFactory PEFT):
    - Calls PEFT-wrapped forward with output_hidden_states=True
    - Extracts last_hidden_states from outputs.hidden_states[-1]
    - Preserves PEFT/LoRA optimization by using the full model forward
    """
    # Store original forward
    original_forward = Qwen3VLForConditionalGeneration.forward

    @functools.wraps(original_forward)
    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        image_grid_thw=None,
        latent_ground_truth=None,
        latent_supervision=None,
        latent_positions=None,
        thinking_loss_weight=None,
        **kwargs
    ):
        """Patched forward with latent injection and thinking loss."""
        # Inject latent_ground_truth features if provided
        if latent_ground_truth is not None and latent_positions is not None:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            inputs_embeds = _inject_latent_features_inplace(
                inputs_embeds=inputs_embeds,
                latent_supervision=latent_ground_truth,
                latent_positions=latent_positions,
            )
            if attention_mask is not None and attention_mask.shape[1] != inputs_embeds.shape[1]:
                raise ValueError("[Qwen3VL Latent] pack-after-injection: attention_mask length mismatch.")
            if labels is not None and labels.shape[1] != inputs_embeds.shape[1]:
                raise ValueError("[Qwen3VL Latent] pack-after-injection: labels length mismatch.")
            if position_ids is not None and position_ids.shape[-1] != inputs_embeds.shape[1]:
                raise ValueError("[Qwen3VL Latent] pack-after-injection: position_ids length mismatch.")

        # Pop output_hidden_states from kwargs to avoid duplicate keyword argument
        kwargs.pop('output_hidden_states', None)

        # Check if we need thinking loss
        has_latent_positions = False
        if latent_positions is not None:
            try:
                has_latent_positions = bool(latent_positions.any().item())
            except Exception:
                has_latent_positions = False
        need_hidden_states = latent_supervision is not None and has_latent_positions

        # Call PEFT-wrapped original forward (preserves LoRA optimization)
        # NOTE: output_hidden_states triggers the HF output recorder, which is very slow for long context.
        # We capture only the final hidden state via a forward hook to avoid per-layer recording.
        last_hidden_states = None
        hook_handle = None
        use_hook = bool(need_hidden_states) and os.environ.get("QWEN3VL_HIDDEN_STATES_HOOK", "1") != "0"
        if use_hook:
            try:
                text_model = self.model.language_model

                def _capture_last_hidden(_module, _inputs, output):
                    nonlocal last_hidden_states
                    # Qwen3VLTextModel returns BaseModelOutputWithPast or tuple
                    if hasattr(output, "last_hidden_state"):
                        last_hidden_states = output.last_hidden_state
                    elif isinstance(output, (tuple, list)) and len(output) > 0:
                        last_hidden_states = output[0]
                    else:
                        last_hidden_states = output

                hook_handle = text_model.register_forward_hook(_capture_last_hidden)
            except Exception:
                use_hook = False

        # Cast pixel_values to bfloat16 if present (fix for TransparentEvalCallback dtype mismatch)
        if pixel_values is not None and pixel_values.dtype != torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16)

        # Cast inputs_embeds to bfloat16 if they come from vision encoder in float32
        if inputs_embeds is not None and inputs_embeds.dtype != torch.bfloat16:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)

        outputs = original_forward(
            self,
            input_ids=None if inputs_embeds is not None else input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=need_hidden_states and not use_hook,
            **kwargs
        )

        if hook_handle is not None:
            hook_handle.remove()

        # Compute thinking loss using latent_supervision (direct on LLM hidden states)
        thinking_loss = None

        if need_hidden_states:
            # Extract last layer hidden states from hook or outputs.hidden_states[-1]
            if last_hidden_states is None:
                hidden_states_out = getattr(outputs, "hidden_states", None)
                if hidden_states_out is not None:
                    last_hidden_states = hidden_states_out[-1]
                else:
                    raise RuntimeError(
                        "[Qwen3VL Latent] Failed to capture last_hidden_states. "
                        "Set QWEN3VL_HIDDEN_STATES_HOOK=0 to fall back to output_hidden_states."
                    )

            # IMPORTANT: Delete intermediate layer outputs to free memory!
            if getattr(outputs, "hidden_states", None) is not None:
                del outputs.hidden_states
                outputs.hidden_states = None

            thinking_loss, ot_stats = _compute_thinking_loss(
                hidden_states=last_hidden_states,
                latent_supervision=latent_supervision,
                latent_positions=latent_positions,
            )

            if thinking_loss is not None:
                weight = float(thinking_loss_weight) if thinking_loss_weight is not None else 1.0
                thinking_loss = thinking_loss * weight

        # Combine losses (OT is default, CE is optional during eval)
        ce_loss = outputs.loss if hasattr(outputs, 'loss') else None
        loss = thinking_loss  # Default to OT loss
        if ce_loss is not None and thinking_loss is not None:
            loss = ce_loss + thinking_loss
            # Log loss breakdown with OT statistics (use INFO level to ensure it appears in training logs)
            try:
                rank_0 = not dist.is_initialized() or dist.get_rank() == 0
            except Exception:
                rank_0 = True
            if rank_0:
                # Main loss breakdown at INFO level
                logger.info(f"[Qwen3VL Latent] Loss breakdown: CE={ce_loss.item():.4f}, OT={thinking_loss.item():.4f}, Total={loss.item():.4f}")
                # Detailed OT statistics at DEBUG level (less spam)
                if ot_stats and "aggregated" in ot_stats:
                    agg = ot_stats["aggregated"]
                    logger.debug(f"  OT Stats: valid={ot_stats['num_valid_samples']}, pred_tokens={agg['avg_pred_tokens']:.1f}, target_tokens={agg['avg_target_tokens']:.1f}")
                    logger.debug(f"  Cost Matrix: min={agg['cost_min']:.4f}, max={agg['cost_max']:.4f}, mean={agg['cost_mean']:.4f}, std={agg['cost_std']:.4f}")
                    # Compute average cosine similarity (1 - cost)
                    cos_sim = 1.0 - agg['cost_mean']
                    logger.debug(f"  Cosine Similarity: {cos_sim:.4f}")

        # Update outputs
        if loss is not None:
            outputs.loss = loss

        return outputs

    # Apply patch
    Qwen3VLForConditionalGeneration.forward = patched_forward
    logger.debug("[Qwen3VL Latent] ✓ Patched Qwen3VLForConditionalGeneration.forward (PEFT-wrapped forward with latent injection)")


# Trainer callback patching is handled by sitecustomize.py, not here
# The TransparentEvalCallback is injected via sitecustomize's patched CustomSeq2SeqTrainer.__init__
def _patch_trainer_callback(logger) -> None:
    """No-op - callback injection handled by sitecustomize.py."""
    logger.debug("[Qwen3VL Latent] Trainer callback injection delegated to sitecustomize.py")


def _patch_model_generate(logger) -> None:
    """Patch model.generate() to filter out latent supervision kwargs and fix max_length.

    During evaluation with predict_with_generate=true, the data collator passes
    latent_positions, latent_ground_truth, and latent_supervision kwargs which
    are only valid for forward(), not generate(). This patch filters them out.

    Also fixes generation config to use max_new_tokens instead of max_length,
    preventing crashes when input length exceeds cutoff_len.
    """
    if not LLAMAFACTORY_AVAILABLE:
        return

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    # Store original generate method
    original_generate = Qwen3VLForConditionalGeneration.generate

    @functools.wraps(original_generate)
    def patched_generate(self, *args, **kwargs):
        """Filter out latent supervision kwargs before calling generate."""
        # Remove latent supervision kwargs (only valid for forward())
        latent_kwargs = {'latent_positions', 'latent_ground_truth', 'latent_supervision'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in latent_kwargs}

        # Fix generation config to use max_new_tokens instead of max_length
        # This prevents crash when input_length > cutoff_len due to special tokens
        generation_config = filtered_kwargs.get('generation_config')

        # Check if we should use max_new_tokens (from config or env var)
        max_new_tokens = filtered_kwargs.get('max_new_tokens')
        if max_new_tokens is None:
            # Try to get from environment (set in config)
            max_new_tokens_str = os.environ.get("QWEN3VL_MAX_NEW_TOKENS", "512")
            try:
                max_new_tokens = int(max_new_tokens_str)
            except ValueError:
                max_new_tokens = 512

        # If max_new_tokens is set, disable max_length to avoid conflicts
        if max_new_tokens and 'max_length' in filtered_kwargs:
            # Remove max_length to avoid "input exceeds max_length" error
            filtered_kwargs.pop('max_length', None)

        # Update generation_config if provided
        if generation_config is not None:
            if max_new_tokens:
                generation_config.max_new_tokens = max_new_tokens
                # Set max_length to a very high value or None to avoid conflicts
                generation_config.max_length = None

        return original_generate(self, *args, **filtered_kwargs)

    # Apply patch
    Qwen3VLForConditionalGeneration.generate = patched_generate
    logger.debug("[Qwen3VL Latent] ✓ Patched Qwen3VLForConditionalGeneration.generate to filter latent kwargs and fix max_length")


# ============================================================================
# Helper Functions
# ============================================================================

def _is_qwen3vl_model(model) -> bool:
    """Check if model is Qwen3VL."""
    try:
        config = model.config
        return hasattr(config, 'model_type') and config.model_type in ['qwen3_vl', 'qwen3_vl']
    except Exception:
        return False


def _expand_labels(
    labels: torch.Tensor,
    latent_positions: torch.BoolTensor,
    latent_supervision: List[List[torch.Tensor]],
    ignore_index: int = -100,
) -> torch.Tensor:
    """Expand labels to match injected latent sequences.

    When <|latent_step|> tokens are replaced with sequences, the labels need
    to be expanded. Injected latent tokens get ignore_index labels (no CE loss).

    Args:
        labels: Original labels [batch, seq_len]
        latent_positions: Boolean mask for <|latent_step|> positions [batch, seq_len]
        latent_supervision: List of latent feature tensors for each sample
        ignore_index: Value to use for injected tokens (default: -100)

    Returns:
        Expanded labels [batch, new_seq_len]
    """
    batch_size = labels.shape[0]
    device = labels.device
    dtype = labels.dtype

    new_labels_list = []

    for b in range(batch_size):
        sample_labels = labels[b]  # [seq_len]
        latent_mask = latent_positions[b]

        if not latent_mask.any():
            # No latent tokens, keep original labels
            new_labels_list.append(sample_labels)
            continue

        supervision = latent_supervision[b] if b < len(latent_supervision) else []
        if len(supervision) == 0:
            new_labels_list.append(sample_labels)
            continue

        # Get indices of <|latent_step|> tokens
        latent_indices = latent_mask.nonzero(as_tuple=False).squeeze(dim=-1)

        # Build new labels
        new_labels = []
        seq_idx = 0

        for latent_idx in latent_indices:
            # Add all labels before this <|latent_step|>
            new_labels.append(sample_labels[seq_idx:latent_idx])

            # Get the corresponding latent features
            sup_idx = len([idx for idx in latent_indices if idx < latent_idx])
            if sup_idx < len(supervision):
                feat = supervision[sup_idx]

                # Determine sequence length
                if isinstance(feat, torch.Tensor):
                    if feat.dim() == 1:
                        seq_len = 1
                    elif feat.dim() == 2:
                        seq_len = feat.shape[0]
                    else:
                        seq_len = 1
                else:
                    seq_len = 1

                # Add ignore_index labels for the latent sequence (no CE loss on injected tokens)
                new_labels.append(torch.full((seq_len,), ignore_index, device=device, dtype=dtype))

            # Move seq_idx past this <|latent_step|> token
            seq_idx = latent_idx + 1

        # Add remaining labels
        new_labels.append(sample_labels[seq_idx:])

        # Concatenate
        new_labels_concat = torch.cat(new_labels, dim=0)
        new_labels_list.append(new_labels_concat)

    # Pad to same length
    max_len = max(label.shape[0] for label in new_labels_list)

    padded_labels = []
    for label in new_labels_list:
        if label.shape[0] < max_len:
            # Pad with ignore_index
            padding = torch.full((max_len - label.shape[0],), ignore_index, device=device, dtype=dtype)
            label = torch.cat([label, padding], dim=0)
        padded_labels.append(label)

    # Stack into batch
    result = torch.stack(padded_labels, dim=0)  # [batch, max_seq_len]
    return result


def _expand_attention_mask(
    attention_mask: torch.Tensor,
    latent_positions: torch.BoolTensor,
    latent_supervision: List[List[torch.Tensor]],
) -> torch.Tensor:
    """Expand attention mask to match injected latent sequences.

    When <|latent_step|> tokens are replaced with sequences, the attention mask
    needs to be expanded to match the new sequence length.

    Args:
        attention_mask: Original attention mask [batch, seq_len]
        latent_positions: Boolean mask for <|latent_step|> positions [batch, seq_len]
        latent_supervision: List of latent feature tensors for each sample

    Returns:
        Expanded attention mask [batch, new_seq_len]
    """
    batch_size = attention_mask.shape[0]
    device = attention_mask.device
    dtype = attention_mask.dtype

    new_masks_list = []

    for b in range(batch_size):
        sample_mask = attention_mask[b]  # [seq_len]
        latent_mask = latent_positions[b]

        if not latent_mask.any():
            # No latent tokens, keep original mask
            new_masks_list.append(sample_mask)
            continue

        supervision = latent_supervision[b] if b < len(latent_supervision) else []
        if len(supervision) == 0:
            new_masks_list.append(sample_mask)
            continue

        # Get indices of <|latent_step|> tokens
        latent_indices = latent_mask.nonzero(as_tuple=False).squeeze(dim=-1)

        # Build new attention mask
        new_mask = []
        seq_idx = 0

        for latent_idx in latent_indices:
            # Add all mask values before this <|latent_step|>
            new_mask.append(sample_mask[seq_idx:latent_idx])

            # Get the corresponding latent features
            sup_idx = len([idx for idx in latent_indices if idx < latent_idx])
            if sup_idx < len(supervision):
                feat = supervision[sup_idx]

                # Determine sequence length
                if isinstance(feat, torch.Tensor):
                    if feat.dim() == 1:
                        seq_len = 1
                    elif feat.dim() == 2:
                        seq_len = feat.shape[0]
                    else:
                        seq_len = 1
                else:
                    seq_len = 1

                # Add attention mask for the latent sequence (all 1s = attend to all)
                new_mask.append(torch.ones(seq_len, device=device, dtype=dtype))

            # Move seq_idx past this <|latent_step|> token
            seq_idx = latent_idx + 1

        # Add remaining mask values
        new_mask.append(sample_mask[seq_idx:])

        # Concatenate
        new_mask_concat = torch.cat(new_mask, dim=0)
        new_masks_list.append(new_mask_concat)

    # Pad to same length
    max_len = max(mask.shape[0] for mask in new_masks_list)

    padded_masks = []
    for mask in new_masks_list:
        if mask.shape[0] < max_len:
            # Pad with 1s (attend to padding tokens)
            padding = torch.ones(max_len - mask.shape[0], device=device, dtype=dtype)
            mask = torch.cat([mask, padding], dim=0)
        padded_masks.append(mask)

    # Stack into batch
    result = torch.stack(padded_masks, dim=0)  # [batch, max_seq_len]
    return result


def _expand_latent_positions(
    latent_positions: torch.BoolTensor,
    latent_supervision: List[List[torch.Tensor]],
) -> torch.BoolTensor:
    """Expand latent_positions to match injected latent sequences.

    When <|latent_step|> tokens are replaced with sequences, the latent mask
    must expand so it aligns with the new hidden_states length.

    Args:
        latent_positions: Original mask [batch, seq_len]
        latent_supervision: List of latent feature tensors for each sample

    Returns:
        Expanded latent mask [batch, new_seq_len]
    """
    batch_size = latent_positions.shape[0]
    device = latent_positions.device

    new_masks_list = []

    for b in range(batch_size):
        latent_mask = latent_positions[b]

        if not latent_mask.any():
            new_masks_list.append(latent_mask)
            continue

        supervision = latent_supervision[b] if b < len(latent_supervision) else []
        if len(supervision) == 0:
            new_masks_list.append(latent_mask)
            continue

        latent_indices = latent_mask.nonzero(as_tuple=False).squeeze(dim=-1)

        new_mask = []
        seq_idx = 0

        for latent_idx in latent_indices:
            # Add mask values before this <|latent_step|>
            new_mask.append(latent_mask[seq_idx:latent_idx])

            # Determine sequence length for this latent injection
            sup_idx = len([idx for idx in latent_indices if idx < latent_idx])
            if sup_idx < len(supervision):
                feat = supervision[sup_idx]
                if isinstance(feat, torch.Tensor):
                    if feat.dim() == 1:
                        seq_len = 1
                    elif feat.dim() == 2:
                        seq_len = feat.shape[0]
                    else:
                        seq_len = 1
                else:
                    seq_len = 1

                # Mark injected latent tokens as True
                new_mask.append(torch.ones(seq_len, device=device, dtype=torch.bool))

            # Move seq_idx past this <|latent_step|> token
            seq_idx = latent_idx + 1

        # Add remaining mask values
        new_mask.append(latent_mask[seq_idx:])

        new_mask_concat = torch.cat(new_mask, dim=0)
        new_masks_list.append(new_mask_concat)

    max_len = max(mask.shape[0] for mask in new_masks_list)

    padded_masks = []
    for mask in new_masks_list:
        if mask.shape[0] < max_len:
            padding = torch.zeros(max_len - mask.shape[0], device=device, dtype=torch.bool)
            mask = torch.cat([mask, padding], dim=0)
        padded_masks.append(mask)

    result = torch.stack(padded_masks, dim=0)
    return result


def _expand_position_ids(
    position_ids: torch.Tensor,
    latent_positions: torch.BoolTensor,
    latent_supervision: List[List[torch.Tensor]],
) -> torch.Tensor:
    """Expand position_ids to match injected latent sequences.

    When <|latent_step|> tokens are replaced with sequences, the position_ids
    need to be expanded to match the new sequence length. This follows the
    same pattern as image token expansion in Qwen3VL.

    Args:
        position_ids: Original position IDs [batch, seq_len]
        latent_positions: Boolean mask for <|latent_step|> positions [batch, seq_len]
        latent_supervision: List of latent feature tensors for each sample

    Returns:
        Expanded position IDs [batch, new_seq_len]
    """
    batch_size = position_ids.shape[0]
    device = position_ids.device
    dtype = position_ids.dtype

    new_position_ids_list = []

    for b in range(batch_size):
        sample_pos_ids = position_ids[b]  # [seq_len]
        latent_mask = latent_positions[b]

        if not latent_mask.any():
            # No latent tokens, keep original position_ids
            new_position_ids_list.append(sample_pos_ids)
            continue

        supervision = latent_supervision[b] if b < len(latent_supervision) else []
        if len(supervision) == 0:
            new_position_ids_list.append(sample_pos_ids)
            continue

        # Get indices of <|latent_step|> tokens
        latent_indices = latent_mask.nonzero(as_tuple=False).squeeze(dim=-1)

        # Build new position_ids (following image style: continue incrementing)
        new_pos_ids = []
        seq_idx = 0
        current_pos = 0

        for latent_idx in latent_indices:
            # Add position_ids before this <|latent_step|>
            while seq_idx < latent_idx:
                new_pos_ids.append(sample_pos_ids[seq_idx] if seq_idx < len(sample_pos_ids) else current_pos)
                seq_idx += 1
                current_pos += 1

            # Get the corresponding latent features
            sup_idx = len([idx for idx in latent_indices if idx < latent_idx])
            if sup_idx < len(supervision):
                feat = supervision[sup_idx]

                # Determine sequence length
                if isinstance(feat, torch.Tensor):
                    if feat.dim() == 1:
                        seq_len = 1
                    elif feat.dim() == 2:
                        seq_len = feat.shape[0]
                    else:
                        seq_len = 1
                else:
                    seq_len = 1

                # Add position_ids for the latent sequence (incrementing positions)
                for _ in range(seq_len):
                    new_pos_ids.append(current_pos)
                    current_pos += 1

            # Move seq_idx past this <|latent_step|> token
            seq_idx = latent_idx + 1

        # Add remaining position_ids
        while seq_idx < len(sample_pos_ids):
            new_pos_ids.append(sample_pos_ids[seq_idx] if seq_idx < len(sample_pos_ids) else current_pos)
            seq_idx += 1
            current_pos += 1

        # Convert to tensor
        new_position_ids_list.append(torch.tensor(new_pos_ids, device=device, dtype=dtype))

    # Pad to same length
    max_len = max(pos.shape[0] for pos in new_position_ids_list)

    padded_pos_ids = []
    for pos in new_position_ids_list:
        if pos.shape[0] < max_len:
            # Pad with the last position (common practice)
            padding = torch.full((max_len - pos.shape[0],), pos[-1] if len(pos) > 0 else 0, device=device, dtype=dtype)
            pos = torch.cat([pos, padding], dim=0)
        padded_pos_ids.append(pos)

    # Stack into batch
    result = torch.stack(padded_pos_ids, dim=0)  # [batch, max_seq_len]
    return result


def _inject_latent_features(
    inputs_embeds: torch.Tensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
) -> torch.Tensor:
    """Inject latent_ground_truth features at marked positions.

    Replaces <|latent_step|> tokens with actual latent token sequences, similar to
    how <image> tokens are replaced with image patch tokens in VLMs.

    Args:
        inputs_embeds: Input embeddings [batch, seq_len, hidden_dim]
        latent_supervision: List of latent_ground_truth tensors for each sample
        latent_positions: Boolean mask indicating where <|latent_step|> tokens are

    Latent features are already at LLM hidden dimension, so no projection needed.

    Injection strategy:
    - Each <|latent_step|> token is replaced by its corresponding latent sequence
    - If latent is [seq_len, hidden_dim], it replaces 1 token with seq_len tokens
    - Input sequence is expanded accordingly
    """
    batch_size = inputs_embeds.shape[0]
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    hidden_dim = inputs_embeds.shape[-1]

    # Process each sample in the batch
    new_embeds_list = []
    new_masks_list = []  # Track which samples were modified

    for b in range(batch_size):
        sample_embeds = inputs_embeds[b]  # [seq_len, hidden_dim]
        latent_mask = latent_positions[b]

        if not latent_mask.any():
            # No latent tokens to inject, keep original
            new_embeds_list.append(sample_embeds)
            new_masks_list.append(False)
            continue

        supervision = latent_supervision[b] if b < len(latent_supervision) else []
        if len(supervision) == 0:
            new_embeds_list.append(sample_embeds)
            new_masks_list.append(False)
            continue

        # Get indices of <|latent_step|> tokens
        latent_indices = latent_mask.nonzero(as_tuple=False).squeeze(dim=-1)

        # Build new sequence by replacing <|latent_step|> with latent sequences
        new_tokens = []
        seq_idx = 0

        for latent_idx in latent_indices:
            # Add all tokens before this <|latent_step|>
            new_tokens.append(sample_embeds[seq_idx:latent_idx])

            # Get the corresponding latent features
            sup_idx = len([idx for idx in latent_indices if idx < latent_idx])
            if sup_idx < len(supervision):
                feat = supervision[sup_idx]

                # Move to correct device/dtype
                if isinstance(feat, torch.Tensor):
                    feat = feat.to(device=device, dtype=dtype)

                # Handle dimensions
                if feat.dim() == 1:
                    # Single token: [hidden_dim]
                    if feat.shape[0] != hidden_dim:
                        raise ValueError(
                            f"Latent feature dim {feat.shape[0]} != LLM hidden dim {hidden_dim}"
                        )
                    new_tokens.append(feat.unsqueeze(0))
                elif feat.dim() == 2:
                    # Sequence: [seq_len, hidden_dim] - replace with full sequence
                    if feat.shape[-1] != hidden_dim:
                        raise ValueError(
                            f"Latent feature dim {feat.shape[-1]} != LLM hidden dim {hidden_dim}"
                        )
                    new_tokens.append(feat)  # [seq_len, hidden_dim]
                else:
                    raise ValueError(f"Unexpected latent dim: {feat.dim()}")

            # Move seq_idx past this <|latent_step|> token
            seq_idx = latent_idx + 1

        # Add remaining tokens after the last <|latent_step|>
        new_tokens.append(sample_embeds[seq_idx:])

        # Concatenate all tokens
        new_embeds = torch.cat(new_tokens, dim=0)  # [new_seq_len, hidden_dim]
        new_embeds_list.append(new_embeds)
        new_masks_list.append(True)

    # Check if any samples were modified
    if not any(new_masks_list):
        return inputs_embeds

    # Pad sequences to the same length (max new sequence length)
    max_len = max(emb.shape[0] for emb in new_embeds_list)

    padded_embeds = []
    for emb in new_embeds_list:
        if emb.shape[0] < max_len:
            # Pad with zeros
            padding = torch.zeros(max_len - emb.shape[0], hidden_dim, device=device, dtype=dtype)
            emb = torch.cat([emb, padding], dim=0)
        padded_embeds.append(emb)

    # Stack into batch
    result = torch.stack(padded_embeds, dim=0)  # [batch, max_seq_len, hidden_dim]
    return result


def _inject_latent_features_inplace(
    inputs_embeds: torch.Tensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
) -> torch.Tensor:
    """Inject latent features in-place when sequences are pre-expanded.

    NOTE: With gradient checkpointing enabled, we must clone inputs_embeds first
    to avoid in-place modification errors in the autograd graph.
    """
    batch_size = inputs_embeds.shape[0]
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    hidden_dim = inputs_embeds.shape[-1]

    # Clone to avoid in-place modification issues with gradient checkpointing
    inputs_embeds = inputs_embeds.clone()

    for b in range(batch_size):
        latent_mask = latent_positions[b]
        if not latent_mask.any():
            continue

        supervision = latent_supervision[b] if b < len(latent_supervision) else []
        if len(supervision) == 0:
            continue

        flat_feats = []
        for feat in supervision:
            if isinstance(feat, torch.Tensor):
                feat = feat.to(device=device, dtype=dtype)
                if feat.dim() == 1:
                    if feat.shape[0] != hidden_dim:
                        raise ValueError(
                            f"Latent feature dim {feat.shape[0]} != LLM hidden dim {hidden_dim}"
                        )
                    flat_feats.append(feat.unsqueeze(0))
                elif feat.dim() == 2:
                    if feat.shape[-1] != hidden_dim:
                        raise ValueError(
                            f"Latent feature dim {feat.shape[-1]} != LLM hidden dim {hidden_dim}"
                        )
                    flat_feats.append(feat)
                else:
                    raise ValueError(f"Unexpected latent dim: {feat.dim()}")

        if not flat_feats:
            continue

        flat_tokens = torch.cat(flat_feats, dim=0)  # [total_latent_len, hidden_dim]
        latent_indices = latent_mask.nonzero(as_tuple=False).squeeze(dim=-1)
        if flat_tokens.shape[0] != latent_indices.shape[0]:
            raise ValueError(
                "[Qwen3VL Latent] pack-after-injection length mismatch: "
                f"latent_positions={latent_indices.shape[0]} vs latent_tokens={flat_tokens.shape[0]}"
            )

        inputs_embeds[b, latent_indices] = flat_tokens

    return inputs_embeds


def _match_sequence_length(
    pred: torch.Tensor,
    target: torch.Tensor,
    strategy: str = "truncate",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match sequence lengths between prediction and target tensors.

    Args:
        pred: Prediction tensor [T_pred, D]
        target: Target tensor [T_target, D]
        strategy: Matching strategy
            - 'truncate': Truncate both to min length (default, fastest)
            - 'repeat': Repeat shorter to match longer
            - 'interpolate': Linearly interpolate shorter to match longer

    Returns:
        Matched (pred, target) tensors with same sequence length
    """
    t_pred, t_target = pred.shape[0], target.shape[0]
    dim = pred.shape[-1]

    if t_pred == t_target:
        return pred, target

    if strategy == "truncate":
        min_len = min(t_pred, t_target)
        return pred[:min_len], target[:min_len]

    elif strategy == "repeat":
        max_len = max(t_pred, t_target)

        if t_pred < max_len:
            # Repeat prediction
            repeat_factor = (max_len + t_pred - 1) // t_pred
            pred = pred.repeat(repeat_factor, 1)[:max_len]

        if t_target < max_len:
            # Repeat target
            repeat_factor = (max_len + t_target - 1) // t_target
            target = target.repeat(repeat_factor, 1)[:max_len]

        return pred, target

    elif strategy == "interpolate":
        max_len = max(t_pred, t_target)

        if t_pred < max_len:
            # Interpolate prediction using nearest neighbor + repeat
            indices = torch.linspace(0, t_pred - 1, max_len, device=pred.device).long()
            pred = pred[indices]

        if t_target < max_len:
            # Interpolate target
            indices = torch.linspace(0, t_target - 1, max_len, device=target.device).long()
            target = target[indices]

        return pred, target

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _compute_thinking_loss(
    hidden_states: torch.Tensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
) -> Optional[torch.Tensor]:
    """Compute thinking loss on LLM hidden states at latent positions.

    Supports flexible loss combination via QWEN3VL_LOSS_TYPE env var:

    Syntax:
    - Single loss: "ot", "mse", "repa", "nce"
    - Equal weights: "ot+mse", "repa+nce+ot"
    - Custom weights: "ot:0.7+mse:0.3", "repa:0.5+nce:0.3+ot:0.2"

    Available loss types:
    - 'repa': Negative cosine similarity (for ground truth supervision)
    - 'nce': InfoNCE contrastive loss (aligned/negative pairs)
    - 'ot': EMO optimal transport (closed-form DEMD upper bound)
    - 'mse': Mean squared error (L2 distance)

    Examples:
        export QWEN3VL_LOSS_TYPE="ot"                    # Single loss
        export QWEN3VL_LOSS_TYPE="ot+mse"                # Equal weights (0.5, 0.5)
        export QWEN3VL_LOSS_TYPE="ot:0.7+mse:0.3"        # Custom weights
    """
    loss_spec = os.environ.get("QWEN3VL_LOSS_TYPE", "ot").lower()

    # Parse loss specification
    loss_configs = _parse_loss_spec(loss_spec)

    # Debug logging (first call only)
    if not hasattr(_compute_thinking_loss, '_logged'):
        try:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.debug(f"[Qwen3VL Latent] Computing thinking loss with spec: {loss_spec}, configs: {loss_configs}")
                logger.debug(f"[Qwen3VL Latent] hidden_states shape: {hidden_states.shape}")
                logger.debug(f"[Qwen3VL Latent] latent_supervision length: {len(latent_supervision)}")
                logger.debug(f"[Qwen3VL Latent] latent_positions shape: {latent_positions.shape}, any: {latent_positions.any().item()}")
        except Exception:
            pass
        _compute_thinking_loss._logged = True

    # Compute each loss and combine
    total_loss = 0.0
    total_weight = 0.0
    loss_values = {}
    ot_stats = None  # Store OT statistics for logging

    for loss_name, weight in loss_configs:
        if loss_name == "repa":
            loss = _compute_repa_loss(hidden_states, latent_supervision, latent_positions)
        elif loss_name == "nce":
            loss = _compute_contrastive_loss(hidden_states, latent_supervision, latent_positions)
        elif loss_name == "ot":
            loss, ot_stats = _compute_ot_loss(hidden_states, latent_supervision, latent_positions)
        elif loss_name == "mse":
            loss = _compute_mse_loss(hidden_states, latent_supervision, latent_positions)
        else:
            raise ValueError(f"Unknown loss type: {loss_name}. Must be 'repa', 'nce', 'ot', or 'mse'")

        # Debug logging (first call only)
        if not hasattr(_compute_thinking_loss, '_logged_loss'):
            try:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    logger.debug(f"[Qwen3VL Latent] {loss_name} loss result: {loss}, weight: {weight}")
            except Exception:
                pass
            _compute_thinking_loss._logged_loss = True

        if loss is not None:
            loss_values[loss_name] = loss.item() if isinstance(loss, torch.Tensor) else loss
            total_loss = total_loss + weight * loss
            total_weight = total_weight + weight

    if total_weight > 0:
        # Normalize by total weight
        total_loss = total_loss / total_weight
        return total_loss, ot_stats

    return None, None


def _parse_loss_spec(loss_spec: str) -> List[tuple]:
    """Parse loss specification string into list of (loss_name, weight) tuples.

    Syntax examples:
    - "ot" → [("ot", 1.0)]
    - "ot+mse" → [("ot", 0.5), ("mse", 0.5)]
    - "ot:0.7+mse:0.3" → [("ot", 0.7), ("mse", 0.3)]
    - "repa:0.5+nce:0.3+ot:0.2" → [("repa", 0.5), ("nce", 0.3), ("ot", 0.2)]

    Args:
        loss_spec: Loss specification string

    Returns:
        List of (loss_name, weight) tuples

    Raises:
        ValueError: If loss specification is invalid
    """
    if not loss_spec:
        raise ValueError("Loss specification cannot be empty")

    # Split by '+' to get individual loss specs
    loss_parts = loss_spec.split('+')

    loss_configs = []
    for part in loss_parts:
        part = part.strip()
        if not part:
            continue

        # Check if weight is specified
        if ':' in part:
            # Split by ':' to get loss name and weight
            loss_name, weight_str = part.split(':', 1)
            loss_name = loss_name.strip()
            weight_str = weight_str.strip()

            try:
                weight = float(weight_str)
            except ValueError:
                raise ValueError(f"Invalid weight '{weight_str}' for loss '{loss_name}'. Must be a number.")

            if weight < 0:
                raise ValueError(f"Weight for loss '{loss_name}' must be non-negative, got {weight}")
        else:
            # No weight specified, will use equal weighting
            loss_name = part.strip()
            weight = 1.0

        loss_configs.append((loss_name, weight))

    if not loss_configs:
        raise ValueError(f"No valid loss specifications found in '{loss_spec}'")

    # If all weights are 1.0, normalize to sum to 1
    all_weights = [w for _, w in loss_configs]
    if all(w == 1.0 for w in all_weights):
        num_losses = len(loss_configs)
        loss_configs = [(name, 1.0 / num_losses) for name, _ in loss_configs]

    return loss_configs


def _compute_repa_loss(
    hidden_states: torch.Tensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
) -> Optional[torch.Tensor]:
    """Compute REPA loss (direct negative cosine similarity) on LLM hidden states.

    Use this when you have ground truth supervision targets.

    Sequence matching: Uses QWEN3VL_MATCH_STRATEGY env var (default: truncate)
    - 'truncate': Truncate both to min length
    - 'repeat': Repeat shorter to match longer
    - 'interpolate': Interpolate shorter to match longer
    """
    batch_size = hidden_states.shape[0]
    thinking_losses = []

    for b in range(batch_size):
        latent_mask = latent_positions[b]
        if not latent_mask.any():
            continue

        # Extract hidden states at latent positions (already at LLM hidden dim)
        sample_hidden = hidden_states[b][latent_mask]  # [num_latents, hidden_dim]

        # Get supervision targets (also at LLM hidden dim)
        supervision = latent_supervision[b] if b < len(latent_supervision) else []
        if len(supervision) == 0:
            continue

        supervision_latents = []
        for sup_tensor in supervision:
            if isinstance(sup_tensor, torch.Tensor):
                # Mean-pool if spatial: [T, hidden_dim] → [hidden_dim]
                if sup_tensor.dim() == 2:
                    supervision_latents.append(sup_tensor.mean(dim=0))
                else:
                    supervision_latents.append(sup_tensor)

        if len(supervision_latents) == 0:
            continue

        supervision_tensor = torch.stack(supervision_latents, dim=0)
        supervision_tensor = supervision_tensor.to(sample_hidden.device, sample_hidden.dtype)

        # Verify dimensions match
        if sample_hidden.shape[-1] != supervision_tensor.shape[-1]:
            raise ValueError(
                f"Hidden dim {sample_hidden.shape[-1]} != supervision dim {supervision_tensor.shape[-1]}. "
                f"Both must be at LLM hidden dimension for direct supervision."
            )

        # Match sequence lengths
        strategy = os.environ.get("QWEN3VL_MATCH_STRATEGY", "truncate").lower()
        sample_hidden, supervision_tensor = _match_sequence_length(
            sample_hidden, supervision_tensor, strategy=strategy
        )

        # REPA loss: negative cosine similarity (direct on hidden states)
        pred_norm = F.normalize(sample_hidden, dim=-1)
        superv_norm = F.normalize(supervision_tensor, dim=-1)

        sample_loss = -torch.mean((pred_norm * superv_norm).sum(dim=-1))
        thinking_losses.append(sample_loss)

    if thinking_losses:
        return torch.stack(thinking_losses).mean()
    return None


def _compute_contrastive_loss(
    hidden_states: torch.Tensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
    temperature: float = 0.07,
) -> Optional[torch.Tensor]:
    """Compute InfoNCE-style contrastive loss on LLM hidden states.

    For varied sequence lengths, uses set-level pooling to create fixed-size
    representations per sample while preserving information.

    Pooling strategy: Mean + Max (captures both average and salient features)

    Args:
        hidden_states: LLM hidden states [batch, seq_len, hidden_dim]
        latent_supervision: Supervision targets for each sample
        latent_positions: Boolean mask indicating latent positions
        temperature: Temperature parameter for softmax (default: 0.07)

    Returns:
        Contrastive loss or None if no valid pairs
    """
    batch_size = hidden_states.shape[0]
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Collect all valid samples
    preds_list = []
    targets_list = []

    for b in range(batch_size):
        latent_mask = latent_positions[b]
        if not latent_mask.any():
            continue

        # Extract hidden states at latent positions
        sample_hidden = hidden_states[b][latent_mask]  # [num_latents, hidden_dim]

        # Get supervision targets
        supervision = latent_supervision[b] if b < len(latent_supervision) else []
        if len(supervision) == 0:
            continue

        supervision_latents = []
        for sup_tensor in supervision:
            if isinstance(sup_tensor, torch.Tensor):
                if sup_tensor.dim() == 2:
                    supervision_latents.append(sup_tensor.mean(dim=0))
                else:
                    supervision_latents.append(sup_tensor)

        if len(supervision_latents) == 0:
            continue

        supervision_tensor = torch.stack(supervision_latents, dim=0)  # [num_supervision, hidden_dim]

        # Set-level pooling: concatenate mean and max pooling
        # This captures both average features and salient features
        pred_mean = sample_hidden.mean(dim=0)  # [hidden_dim]
        pred_max = sample_hidden.max(dim=0)[0]  # [hidden_dim]

        target_mean = supervision_tensor.mean(dim=0)  # [hidden_dim]
        target_max = supervision_tensor.max(dim=0)[0]  # [hidden_dim]

        # Concatenate mean and max for richer representation
        pred_pooled = torch.cat([pred_mean, pred_max], dim=0)  # [2 * hidden_dim]
        target_pooled = torch.cat([target_mean, target_max], dim=0)  # [2 * hidden_dim]

        preds_list.append(pred_pooled)
        targets_list.append(target_pooled)

    if len(preds_list) < 2:
        return None  # Need at least 2 samples for contrastive loss

    # Stack into tensors
    preds = torch.stack(preds_list, dim=0)  # [N, 2 * hidden_dim]
    targets = torch.stack(targets_list, dim=0)  # [N, 2 * hidden_dim]

    # L2 normalize
    preds_norm = F.normalize(preds, dim=-1)
    targets_norm = F.normalize(targets, dim=-1)

    # Compute similarity matrix: N x N
    # sim[i, j] = cos(preds[i], targets[j])
    sim_matrix = torch.mm(preds_norm, targets_norm.t()) / temperature  # [N, N]

    # InfoNCE loss: for each row i, positive is at diagonal (i, i)
    # loss = -log(exp(sim[i,i]) / sum_j(exp(sim[i,j])))
    labels = torch.arange(len(preds_list), device=device)

    loss = F.cross_entropy(sim_matrix, labels)

    return loss


def _compute_ot_loss(
    hidden_states: torch.Tensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
    lm_head: nn.Module = None,
) -> Optional[torch.Tensor]:
    """Compute proper OT loss that NATIVELY handles varied sequence lengths.

    KEY INSIGHT: OT works with distributions of DIFFERENT sizes - no matching needed!

    For two distributions:
    - Q (predictions): [N, D]  - any size
    - P (targets): [M, D]     - any size (can be different!)

    Cost matrix: C[i,j] = 1 - cos(Q[i], P[j])  # [N, M] - rectangular!

    DEMD with uniform Q, P:
        DEMD = Q^T @ C @ P
             = (1/N) * Σ_i (1/M) * Σ_j C[i,j]
             = (1/(N*M)) * Σ_i Σ_j (1 - cos(Q[i], P[j]))

    This naturally handles varied lengths without truncation/padding!

    NOTE: No positional bias needed! ViT-encoded latents already contain spatial
    information in their feature representations. Pure semantic OT naturally respects
    the 2D structure through cosine similarity.

    Args:
        hidden_states: LLM hidden states [batch, seq_len, hidden_dim]
        latent_supervision: Supervision targets for each sample
        latent_positions: Boolean mask indicating latent positions
        lm_head: Optional language model head for vocabulary projection

    Returns:
        OT loss or None if no valid pairs
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    # Optional token sampling for OT approximation (default: 16)
    # Set QWEN3VL_OT_SAMPLE_K=0 or "none" to disable sampling (full OT).
    sample_k_raw = os.environ.get("QWEN3VL_OT_SAMPLE_K", "16")
    sample_k = None
    if isinstance(sample_k_raw, str):
        if sample_k_raw.strip().lower() in ("none", "null", "off", "disable", "disabled"):
            sample_k = None
        else:
            try:
                sample_k = int(sample_k_raw)
            except ValueError:
                sample_k = 16
    elif isinstance(sample_k_raw, int):
        sample_k = sample_k_raw

    # Debug logging (first call only)
    if not hasattr(_compute_ot_loss, '_logged'):
        try:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: hidden_states shape={hidden_states.shape}")
                logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: latent_supervision length={len(latent_supervision)}")
                logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: latent_positions shape={latent_positions.shape}, any={latent_positions.any()}")
        except Exception:
            pass
        _compute_ot_loss._logged = True

    # Compute OT loss per-sample and average
    ot_losses = []

    # Statistics collection for detailed logging
    stats = {
        "num_valid_samples": 0,
        "total_pred_tokens": 0,
        "total_target_tokens": 0,
        "cost_stats": [],  # (min, max, mean, std) per sample
    }

    for b in range(hidden_states.shape[0]):
        latent_mask = latent_positions[b]
        if not latent_mask.any():
            # Debug: log skipped samples
            if not hasattr(_compute_ot_loss, '_logged_skip'):
                try:
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: skipping batch {b}, no latent positions")
                except Exception:
                    pass
                _compute_ot_loss._logged_skip = True
            continue

        # Extract hidden states at latent positions (predictions)
        sample_hidden = hidden_states[b][latent_mask]  # [N, hidden_dim]

        # Get supervision targets
        supervision = latent_supervision[b] if b < len(latent_supervision) else []
        if len(supervision) == 0:
            # Debug: log missing supervision
            if not hasattr(_compute_ot_loss, '_logged_no_supervision'):
                try:
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: batch {b} has no supervision targets (len={len(supervision)})")
                except Exception:
                    pass
                _compute_ot_loss._logged_no_supervision = True
            continue

        supervision_latents = []
        for sup_tensor in supervision:
            if isinstance(sup_tensor, torch.Tensor):
                if sup_tensor.dim() == 2:
                    supervision_latents.append(sup_tensor)
                else:
                    supervision_latents.append(sup_tensor.unsqueeze(0))

        if len(supervision_latents) == 0:
            if not hasattr(_compute_ot_loss, '_logged_no_tensors'):
                try:
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: batch {b} has no valid supervision tensors")
                except Exception:
                    pass
                _compute_ot_loss._logged_no_tensors = True
            continue

        # Concatenate to allow variable-length supervision tensors
        superv_flat = torch.cat(supervision_latents, dim=0)  # [M, hidden_dim]

        # NO length matching for OT - let N and M be naturally different!
        N = sample_hidden.shape[0]  # Number of predicted tokens
        M = superv_flat.shape[0]    # Number of target tokens (can differ!)

        # Move to device
        pred_tokens = sample_hidden.to(device=device, dtype=dtype)          # [N, D]
        target_tokens = superv_flat.to(device=device, dtype=dtype)          # [M, D]

        # Optional sampling to reduce OT cost
        if sample_k is not None and sample_k > 0:
            if N > sample_k:
                idx = torch.randperm(N, device=device)[:sample_k]
                pred_tokens = pred_tokens[idx]
                N = pred_tokens.shape[0]
            if M > sample_k:
                idx = torch.randperm(M, device=device)[:sample_k]
                target_tokens = target_tokens[idx]
                M = target_tokens.shape[0]

        # NOTE: N and M can be DIFFERENT! Cost matrix will be [N, M] rectangular.

        # Compute semantic cost matrix (RECTANGULAR: [N, M])
        # C[i,j] = 1 - cos(pred[i], target[j])
        if lm_head is not None:
            # True EMO: Project to vocabulary space
            E = lm_head.weight.data  # [vocab_size, hidden_dim]
            E = E / torch.linalg.vector_norm(E, ord=2, dim=1, keepdim=True)
            E = E.to(device=device, dtype=dtype)

            pred_logits = pred_tokens @ E.t()      # [N, vocab_size]
            target_logits = target_tokens @ E.t()  # [M, vocab_size]

            Q_θ = F.softmax(pred_logits, dim=-1)      # [N, vocab_size]
            P = F.softmax(target_logits, dim=-1)      # [M, vocab_size]

            pred_repr = Q_θ @ E      # [N, hidden_dim]
            target_repr = P @ E      # [M, hidden_dim]

            # Semantic cost: [N, M] (rectangular!)
            pred_repr_norm = F.normalize(pred_repr, dim=-1)
            target_repr_norm = F.normalize(target_repr, dim=-1)
            semantic_sim = torch.mm(pred_repr_norm, target_repr_norm.t())  # [N, M]
            cost_matrix = 1.0 - semantic_sim
        else:
            # Direct hidden space (default, more efficient)
            pred_norm = F.normalize(pred_tokens, dim=-1)  # [N, D]
            target_norm = F.normalize(target_tokens, dim=-1)  # [M, D]
            semantic_sim = torch.mm(pred_norm, target_norm.t())  # [N, M]
            cost_matrix = 1.0 - semantic_sim

        # DEMD with uniform distributions over DIFFERENT sizes:
        # Q: uniform over N tokens → [1/N, ..., 1/N]  (size N)
        # P: uniform over M tokens → [1/M, ..., 1/M]  (size M)
        #
        # DEMD = Q^T @ C @ P
        #      = Σ_i (Q[i] * Σ_j (P[j] * C[i,j]))
        #      = (1/N) * Σ_i (1/M) * Σ_j C[i,j]
        #      = (1/(N*M)) * Σ_i Σ_j C[i,j]
        #      = mean(cost_matrix)

        sample_ot_loss = cost_matrix.mean()  # Average over N×M rectangular matrix
        ot_losses.append(sample_ot_loss)

        # Collect statistics
        stats["num_valid_samples"] += 1
        stats["total_pred_tokens"] += N
        stats["total_target_tokens"] += M
        stats["cost_stats"].append((
            cost_matrix.min().item(),
            cost_matrix.max().item(),
            cost_matrix.mean().item(),
            cost_matrix.std().item() if cost_matrix.numel() > 1 else 0.0,
        ))

    # Debug: log final result
    if not hasattr(_compute_ot_loss, '_logged_result'):
        try:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: ot_losses length={len(ot_losses)}, returning={ot_losses is not None and len(ot_losses) > 0}")
        except Exception:
            pass
        _compute_ot_loss._logged_result = True

    if ot_losses:
        # Compute aggregated statistics
        loss_value = torch.stack(ot_losses).mean()
        if stats["cost_stats"]:
            cost_mins = [s[0] for s in stats["cost_stats"]]
            cost_maxs = [s[1] for s in stats["cost_stats"]]
            cost_means = [s[2] for s in stats["cost_stats"]]
            cost_stds = [s[3] for s in stats["cost_stats"]]
            stats["aggregated"] = {
                "cost_min": min(cost_mins),
                "cost_max": max(cost_maxs),
                "cost_mean": sum(cost_means) / len(cost_means),
                "cost_std": sum(cost_stds) / len(cost_stds),
                "avg_pred_tokens": stats["total_pred_tokens"] / max(1, stats["num_valid_samples"]),
                "avg_target_tokens": stats["total_target_tokens"] / max(1, stats["num_valid_samples"]),
            }
        return loss_value, stats
    return None, None


def _compute_mse_loss(
    hidden_states: torch.Tensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
) -> Optional[torch.Tensor]:
    """Compute Mean Squared Error (MSE) loss on LLM hidden states at latent positions.

    Standard L2 loss that directly minimizes the squared distance between
    predicted hidden states and supervision targets.

    MSE Loss:
        MSE = (1/N) Σ_i ||pred_i - target_i||^2

    This is a simple, straightforward loss that works well when:
    - The supervision targets are reliable (ground truth or high-quality)
    - You want direct optimization of L2 distance
    - The hidden states are at similar scales

    Sequence matching: Uses QWEN3VL_MATCH_STRATEGY env var (default: truncate)
    - 'truncate': Truncate both to min length
    - 'repeat': Repeat shorter to match longer
    - 'interpolate': Interpolate shorter to match longer

    Args:
        hidden_states: LLM hidden states [batch, seq_len, hidden_dim]
        latent_supervision: Supervision targets for each sample
        latent_positions: Boolean mask indicating latent positions

    Returns:
        MSE loss or None if no valid pairs
    """
    batch_size = hidden_states.shape[0]
    device = hidden_states.device
    dtype = hidden_states.dtype

    mse_losses = []

    for b in range(batch_size):
        latent_mask = latent_positions[b]
        if not latent_mask.any():
            continue

        # Extract hidden states at latent positions
        sample_hidden = hidden_states[b][latent_mask]  # [num_latents, hidden_dim]

        # Get supervision targets
        supervision = latent_supervision[b] if b < len(latent_supervision) else []
        if len(supervision) == 0:
            continue

        supervision_latents = []
        for sup_tensor in supervision:
            if isinstance(sup_tensor, torch.Tensor):
                # Mean-pool if spatial: [T, hidden_dim] → [hidden_dim]
                if sup_tensor.dim() == 2:
                    supervision_latents.append(sup_tensor.mean(dim=0))
                else:
                    supervision_latents.append(sup_tensor)

        if len(supervision_latents) == 0:
            continue

        supervision_tensor = torch.stack(supervision_latents, dim=0)
        supervision_tensor = supervision_tensor.to(device=device, dtype=dtype)

        # Verify dimensions match
        if sample_hidden.shape[-1] != supervision_tensor.shape[-1]:
            raise ValueError(
                f"Hidden dim {sample_hidden.shape[-1]} != supervision dim {supervision_tensor.shape[-1]}. "
                f"Both must be at LLM hidden dimension for MSE loss."
            )

        # Match sequence lengths
        strategy = os.environ.get("QWEN3VL_MATCH_STRATEGY", "truncate").lower()
        sample_hidden, supervision_tensor = _match_sequence_length(
            sample_hidden, supervision_tensor, strategy=strategy
        )

        # Compute MSE loss for this sample
        # MSE = mean((pred - target)^2)
        sample_loss = F.mse_loss(
            sample_hidden,
            supervision_tensor,
            reduction='mean'
        )
        mse_losses.append(sample_loss)

    if mse_losses:
        return torch.stack(mse_losses).mean()
    return None


def _add_latent_supervision_to_batch(
    collated: dict,
    batch: List[dict],
    logger,
) -> dict:
    """Add latent supervision to collated batch.

    This loads latent tensors from file paths and computes latent_positions.

    Separates:
    - latent_ground_truth: Thinking features (for injection at <|latent_step|>)
    - latent_supervision: Original image features (for OT loss reference)

    Both lists have the same length (one entry per thinking chunk).
    """
    # Get special token IDs
    latent_token_id = int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669"))
    thinking_start_id = int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667"))
    thinking_end_id = int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668"))

    # Debug: Log token IDs on first call (rank 0 only to avoid spam in distributed)
    if not hasattr(_add_latent_supervision_to_batch, '_logged_ids'):
        try:
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.debug(f"[Qwen3VL Latent] Token IDs: latent={latent_token_id}, think_start={thinking_start_id}, think_end={thinking_end_id}")
        except Exception:
            logger.debug(f"[Qwen3VL Latent] Token IDs: latent={latent_token_id}, think_start={thinking_start_id}, think_end={thinking_end_id}")
        _add_latent_supervision_to_batch._logged_ids = True

    # Compute latent positions from input_ids first (to avoid loading latents when no positions)
    latent_positions = None
    if 'input_ids' in collated:
        latent_positions = _find_latent_positions(
            input_ids=collated['input_ids'],
            latent_token_id=latent_token_id,
            thinking_start_id=thinking_start_id,
            thinking_end_id=thinking_end_id,
        )
        collated['latent_positions'] = latent_positions

    # Handle latents (only load if this sample has latent positions)
    latent_ground_truth = []
    latent_supervision = []

    for idx, item in enumerate(batch):
        has_positions = False
        if latent_positions is not None and idx < latent_positions.shape[0]:
            try:
                has_positions = bool(latent_positions[idx].any().item())
            except Exception:
                has_positions = False

        if not has_positions:
            latent_ground_truth.append([])
            latent_supervision.append([])
            continue

        # Load latent_ground_truth for injection
        if 'latent_ground_truth' in item and item['latent_ground_truth']:
            tensors = _load_latent_tensors(item['latent_ground_truth'])
            latent_ground_truth.append(tensors)
        else:
            latent_ground_truth.append([])

        # Load latent_supervision for loss
        if 'latent_supervision' in item and item['latent_supervision']:
            tensors = _load_latent_tensors(item['latent_supervision'])
            latent_supervision.append(tensors)
        else:
            latent_supervision.append([])

    collated['latent_ground_truth'] = latent_ground_truth
    collated['latent_supervision'] = latent_supervision

    return collated


def _load_latent_tensors(paths: List[str]) -> List[torch.Tensor]:
    """Load latent tensors from file paths with LRU caching.

    Caches up to 10000 loaded tensors to avoid repeated disk I/O.

    Handles three formats:
    1. Dict with 'l_features' key (from feature cache): {'l_features': Tensor, 'grid_thw': Tensor}
    2. Dict with 'latent' key (from adaptive renderer cache): {'latent': Tensor, 'grid_thw': Tensor}
    3. Direct tensor (legacy format)
    """
    # Check cache first
    cache = _load_latent_tensors._cache
    cached_tensors = []
    remaining_paths = []

    for i, path in enumerate(paths):
        if isinstance(path, str) and path in cache:
            cached_tensors.append((i, cache[path]))
        else:
            remaining_paths.append((i, path))

    # Load uncached paths
    new_tensors = []
    for i, path in remaining_paths:
        if isinstance(path, str):
            latent = torch.load(path, map_location='cpu')
            # Extract latent features if it's a dict (cached format)
            if isinstance(latent, dict):
                if 'l_features' in latent:
                    tensor = latent['l_features']
                    cache[path] = tensor
                    new_tensors.append((i, tensor))
                elif 'latent' in latent:
                    tensor = latent['latent']
                    cache[path] = tensor
                    new_tensors.append((i, tensor))
                else:
                    continue
            elif isinstance(latent, torch.Tensor):
                cache[path] = latent
                new_tensors.append((i, latent))
            else:
                continue
        elif isinstance(path, torch.Tensor):
            new_tensors.append((i, path))

    # Merge cached and new tensors in original order
    all_tensors = sorted(cached_tensors + new_tensors, key=lambda x: x[0])
    return [t for _, t in all_tensors]

# Initialize cache
_load_latent_tensors._cache = {}


def _find_latent_positions(
    input_ids: torch.Tensor,
    latent_token_id: int,
    thinking_start_id: int,
    thinking_end_id: int,
) -> torch.BoolTensor:
    """Find positions of latent tokens in input_ids."""
    batch_size, seq_len = input_ids.shape
    latent_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for b in range(batch_size):
        in_thinking = False
        for i in range(seq_len):
            tok = input_ids[b, i].item()
            if tok == thinking_start_id:
                in_thinking = True
                continue
            if tok == thinking_end_id:
                in_thinking = False
                continue
            if in_thinking and tok == latent_token_id:
                latent_mask[b, i] = True

    # Debug: Log if any latent positions found (first call only, rank 0 only)
    if not hasattr(_find_latent_positions, '_logged_found') and latent_mask.any():
        try:
            should_log = not dist.is_initialized() or dist.get_rank() == 0
        except Exception:
            should_log = True

        if should_log:
            num_found = latent_mask.sum().item()
            logger.debug(f"[Qwen3VL Latent] Found {num_found} latent positions in batch")
        _find_latent_positions._logged_found = True

    return latent_mask


# Apply patches on import
_patch_once()
