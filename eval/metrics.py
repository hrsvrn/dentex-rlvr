"""Evaluation metrics: per-component accuracy, macro-F1, format compliance."""

from __future__ import annotations

import re
from collections import Counter

from rewards.fdi_reward import Finding, parse_findings

_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_FULL_FORMAT = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL
)

DIAGNOSES = ["caries", "deep_caries", "periapical", "impacted"]


def format_compliance(outputs: list[str]) -> float:
    """Fraction of outputs with valid <think>/<answer> format."""
    if not outputs:
        return 0.0
    valid = sum(1 for o in outputs if _FULL_FORMAT.match(o or ""))
    return valid / len(outputs)


def extract_predictions(model_output: str) -> list[Finding]:
    """Extract Finding tuples from model output's answer block."""
    match = _ANSWER_PATTERN.search(model_output or "")
    if not match:
        return []
    return parse_findings(match.group(1))


def quadrant_accuracy(
    predictions: list[list[Finding]],
    ground_truths: list[list[Finding]],
) -> float:
    """Per-finding quadrant exact match accuracy."""
    correct = 0
    total = 0

    for preds, gts in zip(predictions, ground_truths):
        for gt in gts:
            total += 1
            if any(p.quadrant == gt.quadrant for p in preds):
                correct += 1

    return correct / total if total > 0 else 0.0


def tooth_accuracy(
    predictions: list[list[Finding]],
    ground_truths: list[list[Finding]],
) -> float:
    """Per-finding tooth-level accuracy (quadrant must also match)."""
    correct = 0
    total = 0

    for preds, gts in zip(predictions, ground_truths):
        for gt in gts:
            total += 1
            if any(
                p.quadrant == gt.quadrant and p.tooth == gt.tooth
                for p in preds
            ):
                correct += 1

    return correct / total if total > 0 else 0.0


def diagnosis_f1_macro(
    predictions: list[list[Finding]],
    ground_truths: list[list[Finding]],
) -> float:
    """Macro-averaged F1 across pathology classes."""
    per_class: dict[str, dict[str, int]] = {
        d: {"tp": 0, "fp": 0, "fn": 0} for d in DIAGNOSES
    }

    for preds, gts in zip(predictions, ground_truths):
        gt_diags = Counter(g.diagnosis for g in gts)
        pred_diags = Counter(p.diagnosis for p in preds)

        for diag in DIAGNOSES:
            gt_count = gt_diags.get(diag, 0)
            pred_count = pred_diags.get(diag, 0)
            tp = min(gt_count, pred_count)
            per_class[diag]["tp"] += tp
            per_class[diag]["fp"] += max(0, pred_count - gt_count)
            per_class[diag]["fn"] += max(0, gt_count - pred_count)

    f1_scores = []
    for diag in DIAGNOSES:
        tp = per_class[diag]["tp"]
        fp = per_class[diag]["fp"]
        fn = per_class[diag]["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def compute_all_metrics(
    model_outputs: list[str],
    ground_truths: list[list[dict]],
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        model_outputs: List of raw model output strings.
        ground_truths: List of ground truth finding lists (each a list of dicts
            with quadrant/tooth/diagnosis keys).

    Returns:
        Dict of metric name -> value.
    """
    preds = [extract_predictions(o) for o in model_outputs]
    gts = [
        [Finding(f["quadrant"], f["tooth"], f["diagnosis"]) for f in gt_list]
        for gt_list in ground_truths
    ]

    return {
        "format_compliance": format_compliance(model_outputs),
        "quadrant_accuracy": quadrant_accuracy(preds, gts),
        "tooth_accuracy": tooth_accuracy(preds, gts),
        "diagnosis_f1_macro": diagnosis_f1_macro(preds, gts),
    }
