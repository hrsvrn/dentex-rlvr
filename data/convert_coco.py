"""Convert DENTEX COCO-format training data to JSONL for GRPO training.

Input: quadrant-enumeration-disease/train_quadrant_enumeration_disease.json
       (COCO format with category_id_1=quadrant, category_id_2=tooth, category_id_3=disease)

Usage:
    python data/convert_coco.py \
      --json /path/to/train_quadrant_enumeration_disease.json \
      --output data/dentex_train.jsonl \
      --split train
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SYSTEM_PROMPT = "You are a dental radiologist. Analyze the panoramic X-ray."

USER_PROMPT = (
    "Identify all abnormal teeth. For each, output FDI quadrant (1–4), "
    "tooth number (1–8), and diagnosis "
    "(caries/deep_caries/periapical/impacted). "
    "Think step by step inside <think> tags. Output answer inside <answer> tags. "
    "Format: <answer>Q{q}T{t}:{diag}, Q{q}T{t}:{diag}, ...</answer>"
)

# DENTEX COCO category_id_3 → diagnosis string
PATHOLOGY_MAP = {
    0: "caries",
    1: "deep_caries",
    2: "periapical",
    3: "impacted",
}


def convert_coco(json_path: Path, output_path: Path, split: str) -> tuple[int, int]:
    """Convert DENTEX COCO JSON to JSONL.

    Returns:
        (num_written, num_skipped) tuple.
    """
    with json_path.open() as f:
        data = json.load(f)

    # Build image_id → image info lookup
    images = {img["id"]: img for img in data["images"]}

    # Group annotations by image_id
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with output_path.open("w") as f:
        for image_id, img_info in sorted(images.items()):
            anns = anns_by_image.get(image_id, [])
            if not anns:
                skipped += 1
                continue

            findings = []
            for ann in anns:
                quadrant = ann["category_id_1"]
                tooth = ann["category_id_2"]
                disease_id = ann["category_id_3"]

                diagnosis = PATHOLOGY_MAP.get(disease_id)
                if diagnosis is None:
                    continue

                if not (1 <= quadrant <= 4 and 1 <= tooth <= 8):
                    continue

                findings.append({
                    "quadrant": quadrant,
                    "tooth": tooth,
                    "diagnosis": diagnosis,
                })

            if not findings:
                skipped += 1
                continue

            answer_parts = [
                f"Q{f['quadrant']}T{f['tooth']}:{f['diagnosis']}"
                for f in findings
            ]

            sample = {
                "image_id": img_info["file_name"].replace(".png", ""),
                "image_path": img_info["file_name"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ],
                "ground_truth": {
                    "findings": findings,
                    "expected_answer": ", ".join(answer_parts),
                },
                "split": split,
            }

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DENTEX COCO annotations to JSONL"
    )
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to COCO JSON (e.g. train_quadrant_enumeration_disease.json)",
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
        help="Split name (train/test/val)",
    )
    args = parser.parse_args()

    written, skipped = convert_coco(args.json, args.output, args.split)
    print(f"Wrote {written} samples to {args.output} ({skipped} skipped)")


if __name__ == "__main__":
    main()
