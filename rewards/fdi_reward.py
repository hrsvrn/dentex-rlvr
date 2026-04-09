"""FDI reward: hierarchical quadrant + tooth + diagnosis matching."""

import re
from typing import NamedTuple

_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# Matches Q{1-4}T{1-8}:{diagnosis}
_FINDING_PATTERN = re.compile(
    r"Q([1-4])T([1-8]):(caries|deep_caries|periapical|impacted)"
)

QUADRANT_SCORE = 0.30
TOOTH_SCORE = 0.30
DIAG_SCORE = 0.30

VALID_DIAGNOSES = {"caries", "deep_caries", "periapical", "impacted"}


class Finding(NamedTuple):
    quadrant: int
    tooth: int
    diagnosis: str


def parse_findings(text: str) -> list[Finding]:
    """Extract structured findings from answer text."""
    return [
        Finding(int(m.group(1)), int(m.group(2)), m.group(3))
        for m in _FINDING_PATTERN.finditer(text)
    ]


def fdi_reward(model_output: str, ground_truth: dict) -> float:
    """Compute hierarchical FDI match reward.

    Reward is computed per ground-truth finding and averaged:
    - +0.30 if quadrant matches
    - +0.30 if tooth number matches (only counted if quadrant correct)
    - +0.30 if diagnosis matches

    Args:
        model_output: Raw text output from the model.
        ground_truth: Dict with key "findings" containing list of dicts,
            each with keys "quadrant" (int), "tooth" (int), "diagnosis" (str).

    Returns:
        Average hierarchical reward across all ground-truth findings.
    """
    gt_findings_raw = ground_truth.get("findings", [])
    if not gt_findings_raw:
        return 0.0

    gt_findings = [
        Finding(f["quadrant"], f["tooth"], f["diagnosis"])
        for f in gt_findings_raw
    ]

    # Extract answer block
    answer_match = _ANSWER_PATTERN.search(model_output or "")
    if not answer_match:
        return 0.0

    pred_findings = parse_findings(answer_match.group(1))
    if not pred_findings:
        return 0.0

    total_reward = 0.0

    for gt in gt_findings:
        best_score = 0.0

        for pred in pred_findings:
            score = 0.0

            if pred.quadrant == gt.quadrant:
                score += QUADRANT_SCORE

                # Tooth reward only if quadrant is correct
                if pred.tooth == gt.tooth:
                    score += TOOTH_SCORE

            # Diagnosis reward is independent
            if pred.diagnosis == gt.diagnosis:
                score += DIAG_SCORE

            best_score = max(best_score, score)

        total_reward += best_score

    return total_reward / len(gt_findings)
