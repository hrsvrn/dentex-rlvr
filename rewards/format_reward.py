"""Format reward: checks <think>/<answer> tag structure."""

import re

# Patterns for valid tag structure
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_FULL_FORMAT = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL
)

# Pattern to detect answer content leaking into think block
_ANSWER_CONTENT_PATTERN = re.compile(r"Q[1-4]T[1-8]:\w+")

FORMAT_SCORE = 0.10
LEAK_PENALTY = -0.20


def format_reward(model_output: str, ground_truth: dict) -> float:
    """Check that model output has valid <think>...</think><answer>...</answer> tags.

    Args:
        model_output: Raw text output from the model.
        ground_truth: Ground truth dict (unused, kept for consistent interface).

    Returns:
        Format reward score. 0.10 if valid, -0.20 penalty if answer leaks
        into think block, 0.0 otherwise.
    """
    if not model_output or not model_output.strip():
        return 0.0

    # Check overall format: <think>...</think><answer>...</answer>
    if not _FULL_FORMAT.match(model_output):
        return 0.0

    think_match = _THINK_PATTERN.search(model_output)
    answer_match = _ANSWER_PATTERN.search(model_output)

    if not think_match or not answer_match:
        return 0.0

    # Check for answer content leaking inside <think> block
    think_content = think_match.group(1)
    if _ANSWER_CONTENT_PATTERN.search(think_content):
        return LEAK_PENALTY

    # Check that answer block is non-empty
    answer_content = answer_match.group(1).strip()
    if not answer_content:
        return 0.0

    return FORMAT_SCORE
