#!/usr/bin/env python3
"""
Convert Qwen3-VL patch_embed Conv3d to Linear and save a linearized checkpoint.

This produces a fully loadable checkpoint with custom modeling code that swaps
the Conv3d patch embed with a Linear layer (same outputs, faster runtime).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq


class Qwen3VLVisionPatchEmbedLinear(nn.Module):
    def __init__(self, config, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        in_dim = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size

        self.proj = nn.Linear(in_dim, self.embed_dim, bias=True, device=weight.device, dtype=weight.dtype)
        self.proj.weight.data.copy_(weight.reshape(self.embed_dim, -1))
        if bias is not None:
            self.proj.bias.data.copy_(bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype))
        return hidden_states


def _verify_equivalence(
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    config,
    num_patches: int = 8,
) -> None:
    device = conv_weight.device
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size

    conv = nn.Conv3d(
        config.in_channels,
        config.hidden_size,
        kernel_size=(config.temporal_patch_size, config.patch_size, config.patch_size),
        stride=(config.temporal_patch_size, config.patch_size, config.patch_size),
        bias=True,
        device=device,
        dtype=torch.float32,
    )
    conv.weight.data.copy_(conv_weight.float())
    if conv_bias is not None:
        conv.bias.data.copy_(conv_bias.float())

    linear = Qwen3VLVisionPatchEmbedLinear(
        config, weight=conv_weight.float(), bias=conv_bias.float() if conv_bias is not None else None
    )

    x = torch.randn(num_patches, patch_dim, device=device, dtype=torch.float32)
    with torch.no_grad():
        y_conv = conv(x.view(-1, config.in_channels, config.temporal_patch_size, config.patch_size, config.patch_size))
        y_conv = y_conv.view(-1, config.hidden_size)
        y_lin = linear(x)

    torch.testing.assert_close(y_conv, y_lin, rtol=1e-5, atol=1e-5)


def _copy_support_files(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    skip = {"config.json", "model.safetensors", "pytorch_model.bin"}
    for name in os.listdir(src_dir):
        if name in skip:
            continue
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                continue
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        subdir = os.path.join(path, "transformers_modules")
        os.makedirs(subdir, exist_ok=True)
        test_path = os.path.join(subdir, ".write_test")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking",
        help="Source checkpoint directory",
    )
    parser.add_argument(
        "--dst",
        default="/share/project/xiyan/sources/DeepSeek-OCR/Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking",
        help="Destination checkpoint directory",
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.src}")
    model = AutoModelForVision2Seq.from_pretrained(
        args.src,
        trust_remote_code=True,
        device_map="cpu",
        dtype=torch.bfloat16,
    )

    patch_embed = model.visual.patch_embed
    proj = getattr(patch_embed, "proj", None)
    if not isinstance(proj, nn.Conv3d):
        raise RuntimeError("Expected patch_embed.proj to be nn.Conv3d")

    print("Verifying Conv3d -> Linear equivalence...")
    _verify_equivalence(proj.weight, proj.bias, model.visual.config)
    print("✓ Verification passed")

    print("Converting patch_embed to Linear...")
    model.visual.patch_embed = Qwen3VLVisionPatchEmbedLinear(
        model.visual.config, weight=proj.weight.detach(), bias=proj.bias.detach() if proj.bias is not None else None
    )

    # Mark config for linear patch embed and set auto_map for custom loader
    model.config.vision_config.patch_embed_type = "linear"
    model.config.auto_map = {
        "AutoModelForVision2Seq": "modeling_qwen3_vl_linear.Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText": "modeling_qwen3_vl_linear.Qwen3VLForConditionalGeneration",
        "AutoModelForCausalLM": "modeling_qwen3_vl_linear.Qwen3VLForConditionalGeneration",
        "AutoModel": "modeling_qwen3_vl_linear.Qwen3VLModel",
    }

    print(f"Saving linearized checkpoint to: {args.dst}")
    _copy_support_files(args.src, args.dst)
    model.save_pretrained(args.dst, safe_serialization=True)

    # Ensure custom modeling file exists in destination
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    local_modeling = os.path.join(
        repo_root,
        "Qwen",
        "checkpoints",
        "Qwen3-VL-Linear-2B-Thinking",
        "modeling_qwen3_vl_linear.py",
    )
    dst_modeling = os.path.join(args.dst, "modeling_qwen3_vl_linear.py")
    if os.path.exists(local_modeling) and os.path.abspath(local_modeling) != os.path.abspath(dst_modeling):
        shutil.copy2(local_modeling, dst_modeling)

    # Persist a minimal config hint for humans
    meta_path = os.path.join(args.dst, "linear_patch_embed.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "patch_embed_type": "linear",
                "source": args.src,
            },
            f,
            indent=2,
        )

    # Pre-populate transformers_modules cache so trust_remote_code resolves cleanly
    try:
        default_modules_cache = os.environ.get("HF_MODULES_CACHE")
        modules_cache = default_modules_cache
        if not modules_cache or not _is_writable_dir(modules_cache):
            modules_cache = os.path.join(args.dst, ".hf_modules")

        # Ensure transformers uses our chosen modules cache (override constants)
        import transformers.utils.hub as hf_hub_utils
        import transformers.dynamic_module_utils as dmu

        hf_hub_utils.HF_MODULES_CACHE = modules_cache
        dmu.HF_MODULES_CACHE = modules_cache

        get_class_from_dynamic_module = dmu.get_class_from_dynamic_module
        get_class_from_dynamic_module(
            "modeling_qwen3_vl_linear.Qwen3VLForConditionalGeneration",
            args.dst,
            local_files_only=True,
        )
        with open(os.path.join(args.dst, "hf_modules_cache.path"), "w") as f:
            f.write(modules_cache)
        print(f"✓ Cached linear modeling code in transformers_modules ({modules_cache})")
    except Exception as e:
        print(f"⚠️ Failed to populate transformers_modules cache: {e}")

    print("✓ Done")


if __name__ == "__main__":
    main()
