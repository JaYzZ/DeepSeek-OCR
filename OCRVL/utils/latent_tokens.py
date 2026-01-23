"""
Latent Token Utilities for Minimal-Change Thinking Training

This module provides utilities for adding special thinking tokens to the tokenizer
and managing their IDs. These tokens are used to mark positions where pre-encoded
features (from rendered thinking/CoT text) should be injected during forward pass.

Format (inspired by DPSK OCR's visual token layout):
    Question <|thinking_start|><|latent_step|><|sep|><|latent_step|><|sep|>...<|thinking_end|> Answer

This mirrors DPSK OCR's format:
    - 100 visual tokens (10×10 grid) + 10 newlines (row separators) + 1 view separator
    - k latent steps + (k-1) step separators + thinking boundary tokens

Special Tokens:
    <|thinking_start|>: Marks the beginning of the thinking section
    <|latent_step|>: Placeholder for a single thinking step (repeated k times, ADAPTIVE)
    <|thinking_sep|>: Separator between thinking steps (like newlines in OCR)
    <|thinking_end|>: Marks the end of the thinking section

Training Data Format:
    Question + <|thinking_start|> + k×<|latent_step|> + (k-1)×<|thinking_sep|> + <|thinking_end|> + Answer

    The number k is ADAPTIVE - determined by how many OCR chunks the thinking
    text produces. Short thinking = 1 step, long thinking = multiple steps.

Loss Mask:
    - Question: masked (-100)
    - Thinking tokens: masked (-100) during text generation
    - Answer: NOT masked (compute loss)
    - Latent supervision: MSE loss via reconstruction MLP

Architecture:
    Question (text) → [Thinking Section] → Answer
                        ↓
            Pre-encoded OCR features (100×1280 each, 10×10 grid)
                        ↓
            Existing OCR connector → Hidden states
                        ↓
            Reconstruction MLP → Predicted OCR features
                        ↓
            MSE loss against ground truth OCR features
"""

import logging
from typing import Tuple, Dict, Optional
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# Special token names
THINKING_START_TOKEN = "<|thinking_start|>"
LATENT_STEP_TOKEN = "<|latent_step|>"
THINKING_SEP_TOKEN = "<|thinking_sep|>"
THINKING_END_TOKEN = "<|thinking_end|>"

# All thinking tokens (in order)
ALL_THINKING_TOKENS = [
    THINKING_START_TOKEN,
    LATENT_STEP_TOKEN,
    THINKING_SEP_TOKEN,
    THINKING_END_TOKEN,
]

# Default token IDs and names for Qwen3-VL-2B-Thinking model
# These are pre-existing in the checkpoint
DEFAULT_THINKING_START_TOKEN = "<think>"
DEFAULT_THINKING_END_TOKEN = "</think>"
DEFAULT_THINKING_START_ID = 151667
DEFAULT_THINKING_END_ID = 151668


def get_thinking_token_ids(tokenizer: PreTrainedTokenizerBase) -> Dict[str, int]:
    """Get the token IDs for thinking special tokens.

    First tries to find named tokens. If not found, falls back to default
    IDs for Qwen3-VL-2B-Thinking (151667, 151668).

    Args:
        tokenizer: Tokenizer to query

    Returns:
        Dict mapping token names to their IDs

    Raises:
        ValueError: If tokens cannot be found or created
    """
    token_ids = {}

    # Try to find named tokens first
    found_all = True
    for token_name in ALL_THINKING_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token_name)

        if token_id == tokenizer.unk_token_id:
            found_all = False
            break

        token_ids[token_name] = token_id

    if found_all:
        return token_ids

    # Fall back to default IDs for Qwen3-VL-2B-Thinking
    # Use <|thinking_start|> = 151667, <|thinking_end|> = 151668
    # For <|latent_step|> and <|thinking_sep|>, add new tokens
    logger.info("Named thinking tokens not found, using default IDs for Qwen3-VL-2B-Thinking")

    token_ids[THINKING_START_TOKEN] = DEFAULT_THINKING_START_ID
    token_ids[THINKING_END_TOKEN] = DEFAULT_THINKING_END_ID

    # Check if we need to add <|latent_step|> and <|thinking_sep|>
    latent_step_id = tokenizer.convert_tokens_to_ids(LATENT_STEP_TOKEN)
    thinking_sep_id = tokenizer.convert_tokens_to_ids(THINKING_SEP_TOKEN)

    if latent_step_id == tokenizer.unk_token_id or thinking_sep_id == tokenizer.unk_token_id:
        # Add the missing tokens
        tokens_to_add = []
        if latent_step_id == tokenizer.unk_token_id:
            tokens_to_add.append(LATENT_STEP_TOKEN)
        if thinking_sep_id == tokenizer.unk_token_id:
            tokens_to_add.append(THINKING_SEP_TOKEN)

        if tokens_to_add:
            logger.info(f"Adding tokens: {tokens_to_add}")
            tokenizer.add_tokens(tokens_to_add)
            latent_step_id = tokenizer.convert_tokens_to_ids(LATENT_STEP_TOKEN)
            thinking_sep_id = tokenizer.convert_tokens_to_ids(THINKING_SEP_TOKEN)

    token_ids[LATENT_STEP_TOKEN] = latent_step_id
    token_ids[THINKING_SEP_TOKEN] = thinking_sep_id

    return token_ids


def add_thinking_tokens(
    tokenizer: PreTrainedTokenizerBase,
    save_path: Optional[str] = None,
) -> Tuple[PreTrainedTokenizerBase, Dict[str, int]]:
    """Add thinking special tokens to a tokenizer.

    This function adds the thinking tokens to the tokenizer's vocabulary
    and optionally saves the updated tokenizer.

    Args:
        tokenizer: Tokenizer to add tokens to
        save_path: Optional path to save the updated tokenizer

    Returns:
        Tuple of (updated_tokenizer, token_ids_dict)

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B")
        >>> tokenizer, token_ids = add_thinking_tokens(tokenizer, save_path="./tokenizer_updated")
        >>> print(token_ids)
        {'<|thinking_start|>': 151646, '<|latent_step|>': 151647,
         '<|thinking_sep|>': 151648, '<|thinking_end|>': 151649}
    """
    # Check which tokens need to be added
    tokens_to_add = []
    for token_name in ALL_THINKING_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token_name)
        if token_id == tokenizer.unk_token_id:
            tokens_to_add.append(token_name)
        else:
            logger.info(f"Token '{token_name}' already exists with ID {token_id}")

    if not tokens_to_add:
        logger.info("All thinking tokens already exist in tokenizer")

        # Get existing token IDs
        token_ids = get_thinking_token_ids(tokenizer)

        if save_path:
            logger.info(f"Saving tokenizer to {save_path}")
            tokenizer.save_pretrained(save_path)

        return tokenizer, token_ids

    # Add new tokens
    logger.info(f"Adding {len(tokens_to_add)} new tokens to tokenizer: {tokens_to_add}")
    num_added = tokenizer.add_tokens(tokens_to_add)
    logger.info(f"Added {num_added} tokens to tokenizer")

    # Get token IDs
    token_ids = get_thinking_token_ids(tokenizer)

    # Log token IDs
    logger.info("Thinking Token IDs:")
    for token_name, token_id in token_ids.items():
        logger.info(f"  {token_name}: {token_id}")

    # Save tokenizer if path provided
    if save_path:
        logger.info(f"Saving updated tokenizer to {save_path}")
        tokenizer.save_pretrained(save_path)

    return tokenizer, token_ids


def verify_thinking_tokens(
    tokenizer: PreTrainedTokenizerBase,
) -> bool:
    """Verify that all thinking tokens exist in the tokenizer.

    Args:
        tokenizer: Tokenizer to verify

    Returns:
        True if all tokens exist, False otherwise
    """
    try:
        token_ids = get_thinking_token_ids(tokenizer)
        logger.info("✓ All thinking tokens found:")
        for token_name, token_id in token_ids.items():
            logger.info(f"    {token_name}: {token_id}")
        return True
    except ValueError as e:
        logger.warning(f"✗ Thinking tokens not found: {e}")
        return False


def build_sequence_with_thinking(
    question_ids: list,
    answer_ids: list,
    num_steps: int,
    tokenizer: PreTrainedTokenizerBase,
    include_newline: bool = True,
) -> Tuple[list[int], list[int], int, int]:
    """Build a training sequence with thinking tokens (using separators).

    Format (inspired by DPSK OCR's visual token layout):
        Question <|thinking_start|><|latent_step|>[<|thinking_sep|><|latent_step|>]*<|thinking_end|> Answer

    Example with 3 thinking steps:
        "What is 2+2?" <|thinking_start|><|latent_step|><|thinking_sep|><|latent_step|><|thinking_sep|><|latent_step|><|thinking_end|> "4"

    This mirrors DPSK OCR's format:
    - Visual: 100 tokens + 10 newlines (row separators) + 1 view separator
    - Thinking: k steps + (k-1) separators + thinking boundaries

    Each thinking step represents one chunk of rendered thinking text.

    Args:
        question_ids: Tokenized question
        answer_ids: Tokenized answer
        num_steps: Number of thinking steps (each gets one <|latent_step|> token)
        tokenizer: Tokenizer (to get thinking token IDs)
        include_newline: Whether to add newline after thinking_end

    Returns:
        Tuple of (input_ids, labels, thinking_start_idx, thinking_end_idx)

        labels: -100 for question and thinking, actual IDs for answer
        thinking_start_idx: Index where thinking section starts
        thinking_end_idx: Index where thinking section ends
    """
    # Get token IDs
    token_ids = get_thinking_token_ids(tokenizer)

    start_id = token_ids[THINKING_START_TOKEN]
    step_id = token_ids[LATENT_STEP_TOKEN]
    sep_id = token_ids[THINKING_SEP_TOKEN]
    end_id = token_ids[THINKING_END_TOKEN]

    # Build thinking section: start + step + (sep + step)* + end
    thinking_block = [start_id]

    # Add steps with separators between them
    for i in range(num_steps):
        thinking_block.append(step_id)
        if i < num_steps - 1:  # Add separator after each step except the last
            thinking_block.append(sep_id)

    thinking_block.append(end_id)

    if include_newline:
        # Add newline token (you may need to adjust this for your tokenizer)
        newline_id = tokenizer.convert_tokens_to_ids("\n")
        if newline_id != tokenizer.unk_token_id:
            thinking_block.append(newline_id)

    # Combine: question + thinking + answer
    input_ids = question_ids + thinking_block + answer_ids

    # Create labels: only compute loss on answer
    labels = [-100] * (len(question_ids) + len(thinking_block)) + answer_ids

    # Track thinking positions
    thinking_start_idx = len(question_ids)
    thinking_end_idx = thinking_start_idx + len(thinking_block)

    return input_ids, labels, thinking_start_idx, thinking_end_idx


# Convenience function for command-line usage
def main():
    """CLI to add thinking tokens to a tokenizer checkpoint."""
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(
        description="Add thinking tokens to a Qwen3-VL tokenizer"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save updated tokenizer (defaults to input path)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify tokens exist, don't add new ones",
    )

    args = parser.parse_args()

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.verify_only:
        # Only verify
        success = verify_thinking_tokens(tokenizer)
        return 0 if success else 1
    else:
        # Add tokens
        output_path = args.output or args.tokenizer
        tokenizer, token_ids = add_thinking_tokens(tokenizer, save_path=output_path)

        logger.info("=" * 60)
        logger.info("Summary:")
        logger.info(f"  Updated tokenizer saved to: {output_path}")
        logger.info(f"  Token IDs:")
        for name, tid in token_ids.items():
            logger.info(f"    {name}: {tid}")
        logger.info("=" * 60)

        return 0



# Backward compatibility: aliases for old function names
def get_latent_token_ids(tokenizer: PreTrainedTokenizerBase) -> Dict[str, int]:
    """Deprecated: Use get_thinking_token_ids() instead."""
    return get_thinking_token_ids(tokenizer)


def add_latent_tokens(tokenizer: PreTrainedTokenizerBase, save_path: Optional[str] = None):
    """Deprecated: Use add_thinking_tokens() instead."""
    return add_thinking_tokens(tokenizer, save_path)


def verify_latent_tokens(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Deprecated: Use verify_thinking_tokens() instead."""
    return verify_thinking_tokens(tokenizer)


def build_sequence_with_latents(question_ids: list, answer_ids: list, num_latents: int, tokenizer, **kwargs):
    """Deprecated: Use build_sequence_with_thinking() instead."""
    return build_sequence_with_thinking(question_ids, answer_ids, num_latents, tokenizer, **kwargs)


if __name__ == "__main__":
    import sys
    sys.exit(main())

