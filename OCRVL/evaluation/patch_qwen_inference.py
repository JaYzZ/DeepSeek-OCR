#!/usr/bin/env python3
"""
Patch Qwen3-VL inference scripts to support OCRVL model loading.

This script modifies the Qwen3-VL evaluation scripts to:
1. Import OCRVL model loader
2. Use load_ocrvl_model() when OCRVL_MODE=1
3. Fall back to standard loading otherwise

Usage:
    python OCRVL/evaluation/patch_qwen_inference.py [--dry-run]

This creates modified versions of:
- ../Qwen3-VL/evaluation/mmmu/run_mmmu.py
- ../Qwen3-VL/evaluation/MathVision/run_mathv.py
- ../Qwen3-VL/evaluation/RealWorldQA/run_realworldqa.py
- ../Qwen3-VL/evaluation/ODinW-13/run_odinw.py

The original files are backed up with .bak extension.
"""

import argparse
import os
import re
import shutil
from pathlib import Path


def find_repo_root():
    """Find repository root (DeepSeek-OCR directory)."""
    script_path = Path(__file__).resolve()
    # This script is at OCRVL/evaluation/patch_qwen_inference.py
    return script_path.parent.parent.parent


def find_model_loading(content: str) -> tuple[bool, str, str]:
    """
    Find model loading pattern in Python code.

    Returns:
        (found, import_section, loading_section)
    """
    # Pattern 1: AutoModelForCausalLM or AutoModel
    model_import_patterns = [
        r'from transformers import.*AutoModelForCausalLM',
        r'from transformers import.*AutoModel[^A-Za-z]',
    ]

    model_loading_patterns = [
        r'AutoModelForCausalLM\.from_pretrained\([^\)]+\)',
        r'AutoModel\.from_pretrained\([^\)]+\)',
    ]

    import_match = None
    for pattern in model_import_patterns:
        import_match = re.search(pattern, content)
        if import_match:
            break

    loading_match = None
    for pattern in model_loading_patterns:
        loading_match = re.search(pattern, content, re.DOTALL)
        if loading_match:
            break

    if import_match and loading_match:
        return True, import_match.group(0), loading_match.group(0)

    return False, "", ""


def patch_file(file_path: Path, dry_run: bool = False) -> bool:
    """
    Patch a single Python file to support OCRVL model loading.

    Returns:
        True if patched, False if skipped
    """
    if not file_path.exists():
        print(f"  ⊗ File not found: {file_path}")
        return False

    with open(file_path) as f:
        original_content = f.read()

    # Check if already patched
    if 'load_ocrvl_model' in original_content:
        print(f"  ✓ Already patched: {file_path.name}")
        return False

    # Find model loading code
    found, import_section, loading_section = find_model_loading(original_content)

    if not found:
        print(f"  ⊗ No model loading found: {file_path.name}")
        return False

    # Create patch
    ocrvl_import = """
# OCRVL model loading support
import sys
_OCRVL_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "DeepSeek-OCR"
if str(_OCRVL_ROOT) not in sys.path:
    sys.path.insert(0, str(_OCRVL_ROOT))
from OCRVL.evaluation.ocrvl_model_loader import load_ocrvl_model, is_ocrvl_mode
"""

    # Add import after transformers import
    patched_content = original_content.replace(
        import_section,
        import_section + ocrvl_import
    )

    # Replace model loading
    # Extract the variable name (e.g., "model = AutoModel...")
    var_match = re.search(r'(\w+)\s*=\s*' + re.escape(loading_section), patched_content)
    if not var_match:
        print(f"  ⊗ Cannot find variable assignment: {file_path.name}")
        return False

    var_name = var_match.group(1)

    ocrvl_loading = f"""# Load model (OCRVL-aware)
if is_ocrvl_mode():
    print("OCRVL mode enabled - loading with trained connectors")
    {var_name} = load_ocrvl_model(
        base_model_path=args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
else:
    {var_name} = {loading_section}"""

    patched_content = patched_content.replace(
        f"{var_name} = {loading_section}",
        ocrvl_loading
    )

    if dry_run:
        print(f"  ⊕ Would patch: {file_path.name}")
        return True

    # Backup original
    backup_path = file_path.with_suffix(file_path.suffix + '.bak')
    shutil.copy(file_path, backup_path)

    # Write patched version
    with open(file_path, 'w') as f:
        f.write(patched_content)

    print(f"  ✓ Patched: {file_path.name} (backup: {backup_path.name})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Patch Qwen3-VL inference scripts for OCRVL")
    parser.add_argument('--dry-run', action='store_true',
                       help="Show what would be patched without making changes")
    args = parser.parse_args()

    repo_root = find_repo_root()
    qwen_eval_root = repo_root.parent / "Qwen3-VL" / "evaluation"

    if not qwen_eval_root.exists():
        print(f"ERROR: Qwen3-VL evaluation directory not found: {qwen_eval_root}")
        print("Expected structure: ../Qwen3-VL/evaluation/")
        return 1

    print("=" * 80)
    print("OCRVL Inference Script Patcher")
    print("=" * 80)
    print(f"Repo root: {repo_root}")
    print(f"Qwen3-VL eval: {qwen_eval_root}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY PATCHES'}")
    print()

    # Files to patch
    files_to_patch = [
        qwen_eval_root / "mmmu" / "run_mmmu.py",
        qwen_eval_root / "MathVision" / "run_mathv.py",
        qwen_eval_root / "RealWorldQA" / "run_realworldqa.py",
        qwen_eval_root / "ODinW-13" / "run_odinw.py",
    ]

    patched_count = 0
    for file_path in files_to_patch:
        print(f"Checking: {file_path.relative_to(qwen_eval_root.parent)}")
        if patch_file(file_path, dry_run=args.dry_run):
            patched_count += 1
        print()

    print("=" * 80)
    print(f"Summary: {patched_count}/{len(files_to_patch)} files patched")
    if args.dry_run:
        print("Run without --dry-run to apply patches")
    else:
        print("Backups saved with .bak extension")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
