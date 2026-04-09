"""Convert LabelMe-format dental dataset to JSONL for GRPO training.

Input structure:
    disease/
    ├── input/   (test_0.png ... test_249.png)
    └── label/   (test_0.json ... test_249.json)

Label format (LabelMe JSON):
    shapes[*].label = "{class_id}-{disease_turkish}-{FDI_tooth_2digit}"
    e.g. "1-çürük-16" → caries at tooth 16 (Q1, T6)

Output: JSONL with messages + ground_truth for GRPOTrainer.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SYSTEM_PROMPT = "You are a dental radiologist. Analyze the panoramic X-ray."

USER_PROMPT = (
    "Identify all abnormal teeth. For each, output FDI quadrant (1–4), "
    "tooth number (1–8), and diagnosis "
    "(caries/deep_caries/periapical/impacted). "
    "Think step by step inside <think> tags. Output answer inside <answer> tags. "
    "Format: <answer>Q{q}T{t}:{diag}, Q{q}T{t}:{diag}, ...</answer>"
)

# Turkish disease name → English diagnosis
# Class 0 (sağlam/healthy) is skipped — no pathology.
DISEASE_MAP: dict[str, str] = {
    "çürük": "caries",         # 1 — caries
    "küretaj": "deep_caries",  # 2 — curettage needed (advanced decay)
    "kanal": "periapical",     # 3 — root canal (periapical lesion)
    "gömülü": "impacted",      # 6 — impacted tooth
    "çekim": "impacted",       # 5 — extraction needed (treated as impacted)
    "lezyon": "periapical",    # 7 — lesion (treated as periapical)
    "kırık": "caries",         # 8 — fracture (treated as caries for now)
}

# Labels with these prefixes are skipped (no pathology)
SKIP_CLASS_IDS = {"0"}

# Regex to parse LabelMe label: "{class_id}-{disease}-{FDI_tooth}"
_LABEL_PATTERN = re.compile(r"^(\d+)-(.+)-(\d{2})$")


def fdi_from_tooth_number(tooth_number: int) -> tuple[int, int]:
    """Convert absolute FDI tooth number (11-48) to (quadrant, tooth_in_quadrant).

    FDI notation: first digit = quadrant (1-4), second digit = tooth (1-8).
    """
    quadrant = tooth_number // 10
    tooth = tooth_number % 10
    return quadrant, tooth


def parse_labelme_label(label: str) -> dict | None:
    """Parse a LabelMe label string into a finding dict.

    Args:
        label: Label string like "1-çürük-16".

    Returns:
        Dict with quadrant, tooth, diagnosis — or None if invalid/skipped.
    """
    match = _LABEL_PATTERN.match(label)
    if not match:
        return None

    class_id = match.group(1)
    disease_turkish = match.group(2)
    fdi_number = int(match.group(3))

    # Skip healthy teeth
    if class_id in SKIP_CLASS_IDS:
        return None

    diagnosis = DISEASE_MAP.get(disease_turkish)
    if diagnosis is None:
        print(f"  WARNING: Unknown disease '{disease_turkish}' in label '{label}', skipping")
        return None

    quadrant, tooth = fdi_from_tooth_number(fdi_number)
    if not (1 <= quadrant <= 4 and 1 <= tooth <= 8):
        print(f"  WARNING: Invalid FDI number {fdi_number} in label '{label}', skipping")
        return None

    return {
        "quadrant": quadrant,
        "tooth": tooth,
        "diagnosis": diagnosis,
    }


def convert_labelme_sample(label_path: Path) -> dict | None:
    """Convert a single LabelMe JSON file to GRPO training format.

    Args:
        label_path: Path to the LabelMe .json file.

    Returns:
        Training sample dict, or None if no valid annotations.
    """
    with label_path.open() as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    if not shapes:
        return None

    findings = []
    for shape in shapes:
        label = shape.get("label", "")
        finding = parse_labelme_label(label)
        if finding is not None:
            findings.append(finding)

    if not findings:
        return None

    # Build expected answer string
    answer_parts = [
        f"Q{f['quadrant']}T{f['tooth']}:{f['diagnosis']}" for f in findings
    ]
    expected_answer = ", ".join(answer_parts)

    image_name = data.get("imagePath", label_path.stem + ".png")

    return {
        "image_id": label_path.stem,
        "image_path": image_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "ground_truth": {
            "findings": findings,
            "expected_answer": expected_answer,
        },
    }


def convert_directory(
    input_dir: Path,
    output_path: Path,
    split: str = "train",
) -> int:
    """Convert all LabelMe JSONs in a directory to JSONL.

    Args:
        input_dir: Path to 'disease/' directory containing input/ and label/ subdirs.
        output_path: Path to output JSONL file.
        split: Split name to attach to each sample.

    Returns:
        Number of valid samples written.
    """
    label_dir = input_dir / "label"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    label_files = sorted(label_dir.glob("*.json"))
    if not label_files:
        raise ValueError(f"No JSON files found in {label_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    with output_path.open("w") as f:
        for label_path in label_files:
            converted = convert_labelme_sample(label_path)
            if converted is not None:
                converted["split"] = split
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1
            else:
                skipped += 1

    return count, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe dental dataset to JSONL"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to disease/ directory (containing input/ and label/ subdirs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dentex_train.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name (train/test)",
    )
    args = parser.parse_args()

    count, skipped = convert_directory(args.input_dir, args.output, args.split)
    print(f"Wrote {count} samples to {args.output} ({skipped} skipped)")


if __name__ == "__main__":
    main()
