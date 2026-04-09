"""Convert DENTEX HuggingFace dataset to JSONL for GRPO training."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

SYSTEM_PROMPT = "You are a dental radiologist. Analyze the panoramic X-ray."

USER_PROMPT = (
    "Identify all abnormal teeth. For each, output FDI quadrant (1–4), "
    "tooth number (1–8), and diagnosis (caries/deep_caries/periapical/impacted). "
    "Think step by step inside <think> tags. Output answer inside <answer> tags. "
    "Format: <answer>Q{q}T{t}:{diag}, Q{q}T{t}:{diag}, ...</answer>"
)

# DENTEX pathology class mapping
PATHOLOGY_MAP = {
    0: "caries",
    1: "deep_caries",
    2: "periapical",
    3: "impacted",
}


def fdi_from_tooth_number(tooth_number: int) -> tuple[int, int]:
    """Convert absolute FDI tooth number (11-48) to (quadrant, tooth_in_quadrant).

    FDI notation: first digit = quadrant (1-4), second digit = tooth (1-8).
    """
    quadrant = tooth_number // 10
    tooth = tooth_number % 10
    return quadrant, tooth


def convert_sample(sample: dict) -> dict | None:
    """Convert a single DENTEX sample to GRPO training format.

    Returns None if sample has no valid annotations.
    """
    annotations = sample.get("annotations", [])
    if not annotations:
        return None

    findings = []
    for ann in annotations:
        category_id = ann.get("category_id") or ann.get("category")
        tooth_number = ann.get("tooth_number") or ann.get("fdi_number")

        if category_id is None or tooth_number is None:
            continue

        diagnosis = PATHOLOGY_MAP.get(int(category_id))
        if diagnosis is None:
            continue

        quadrant, tooth = fdi_from_tooth_number(int(tooth_number))
        if not (1 <= quadrant <= 4 and 1 <= tooth <= 8):
            continue

        findings.append({
            "quadrant": quadrant,
            "tooth": tooth,
            "diagnosis": diagnosis,
        })

    if not findings:
        return None

    # Build expected answer string
    answer_parts = [
        f"Q{f['quadrant']}T{f['tooth']}:{f['diagnosis']}" for f in findings
    ]
    expected_answer = ", ".join(answer_parts)

    return {
        "image_id": sample.get("image_id", sample.get("id", "")),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "ground_truth": {
            "findings": findings,
            "expected_answer": expected_answer,
        },
    }


def convert_split(
    dataset_name: str,
    split: str,
    output_path: Path,
) -> int:
    """Convert a DENTEX dataset split to JSONL.

    Returns the number of valid samples written.
    """
    ds = load_dataset(dataset_name, split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w") as f:
        for sample in ds:
            converted = convert_sample(sample)
            if converted is not None:
                converted["split"] = split
                f.write(json.dumps(converted) + "\n")
                count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DENTEX to JSONL")
    parser.add_argument(
        "--dataset",
        default="ibrahimhamamci/DENTEX",
        help="HuggingFace dataset name",
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
        help="Dataset split to convert",
    )
    args = parser.parse_args()

    count = convert_split(args.dataset, args.split, args.output)
    print(f"Wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
