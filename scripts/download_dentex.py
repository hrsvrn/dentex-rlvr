"""Download and extract the DENTEX dataset from HuggingFace.

This script downloads all ZIP archives from the ibrahimhamamci/DENTEX
repository and extracts them into a local directory, then runs the
LabelMe → JSONL converter to produce training-ready data.

Usage:
    python scripts/download_dentex.py
    python scripts/download_dentex.py --output-dir ./dataset --skip-convert

NOTE: The dataset is ~10.9 GB. Ensure sufficient disk space (~25 GB
recommended for download + extraction).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "ibrahimhamamci/DENTEX"
REPO_TYPE = "dataset"

# Files in the HuggingFace repo
ZIP_FILES = [
    "DENTEX/training_data.zip",
    "DENTEX/test_data.zip",
    "DENTEX/validation_data.zip",
]
EXTRA_FILES = [
    "DENTEX/validation_triple.json",
]


def download_file(filename: str, output_dir: Path) -> Path:
    """Download a single file from the HuggingFace dataset repo.

    Args:
        filename: Relative path within the repo (e.g. "DENTEX/training_data.zip").
        output_dir: Local directory to download into.

    Returns:
        Path to the downloaded file.
    """
    print(f"  Downloading {filename}...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local_path)


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a ZIP archive.

    Args:
        zip_path: Path to the ZIP file.
        extract_to: Directory to extract into.
    """
    print(f"  Extracting {zip_path.name} → {extract_to}/")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"    Extracted {len(zipfile.ZipFile(zip_path).namelist())} files")


def run_converter(
    input_dir: Path,
    output_path: Path,
    split: str,
) -> None:
    """Run the LabelMe → JSONL converter.

    Args:
        input_dir: Path to the disease/ directory.
        output_path: Path for output JSONL file.
        split: Split name (train/test/val).
    """
    script = Path(__file__).resolve().parent.parent / "data" / "convert_labelme.py"
    if not script.exists():
        print(f"  WARNING: Converter not found at {script}, skipping conversion")
        return

    cmd = [
        sys.executable,
        str(script),
        "--input-dir", str(input_dir),
        "--output", str(output_path),
        "--split", split,
    ]
    print(f"  Converting {split}: {input_dir} → {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"    {result.stdout.strip()}")
    else:
        print(f"    ERROR: {result.stderr.strip()}")


def find_disease_dirs(base_dir: Path) -> list[tuple[Path, str]]:
    """Find all disease/ directories within extracted data.

    Returns:
        List of (disease_dir_path, split_name) tuples.
    """
    found = []
    # Look for common patterns in DENTEX extracted structure
    for candidate in base_dir.rglob("disease"):
        if candidate.is_dir() and (candidate / "label").is_dir():
            # Infer split from parent directory name
            parent_name = candidate.parent.name.lower()
            if "train" in parent_name:
                split = "train"
            elif "test" in parent_name:
                split = "test"
            elif "val" in parent_name:
                split = "val"
            else:
                split = "unknown"
            found.append((candidate, split))

    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract the DENTEX dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Directory to download and extract into (default: ./dataset)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction (files already extracted)",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip JSONL conversion step",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download (files already downloaded)",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep ZIP files after extraction (default: delete to save space)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"DENTEX Dataset Setup")
    print(f"{'=' * 50}")
    print(f"Repository:  {REPO_ID}")
    print(f"Output dir:  {output_dir}")
    print(f"Disk needed: ~25 GB (10.9 GB download + extraction)")
    print()

    # ── Step 1: Download ─────────────────────────────────
    if not args.skip_download:
        print("Step 1/3: Downloading from HuggingFace...")
        all_files = ZIP_FILES + EXTRA_FILES
        for filename in all_files:
            local_path = download_file(filename, output_dir)
            print(f"    ✓ {local_path}")
        print()
    else:
        print("Step 1/3: Skipping download (--skip-download)")
        print()

    # ── Step 2: Extract ZIPs ─────────────────────────────
    if not args.skip_extract:
        print("Step 2/3: Extracting ZIP archives...")
        for zip_name in ZIP_FILES:
            zip_path = output_dir / zip_name
            if not zip_path.exists():
                print(f"  WARNING: {zip_path} not found, skipping")
                continue
            extract_to = zip_path.parent  # Extract alongside the ZIP
            extract_zip(zip_path, extract_to)

            if not args.keep_zips:
                zip_path.unlink()
                print(f"    Deleted {zip_path.name} to save space")
        print()
    else:
        print("Step 2/3: Skipping extraction (--skip-extract)")
        print()

    # ── Step 3: Convert to JSONL ─────────────────────────
    if not args.skip_convert:
        print("Step 3/3: Converting to training JSONL...")
        data_dir = Path(__file__).resolve().parent.parent / "data"

        # Find all disease/ directories in the extracted data
        disease_dirs = find_disease_dirs(output_dir)

        if disease_dirs:
            for disease_dir, split in disease_dirs:
                output_jsonl = data_dir / f"dentex_{split}.jsonl"
                run_converter(disease_dir, output_jsonl, split)
        else:
            # Fallback: let user know and try common paths
            print("  Could not auto-detect disease/ directories.")
            print("  You can manually run:")
            print("    python data/convert_labelme.py \\")
            print(f"      --input-dir {output_dir}/DENTEX/<split>/disease \\")
            print("      --output data/dentex_train.jsonl")
        print()
    else:
        print("Step 3/3: Skipping conversion (--skip-convert)")
        print()

    # ── Summary ──────────────────────────────────────────
    print("=" * 50)
    print("Done! Dataset structure:")
    # Show what we have
    for p in sorted(output_dir.rglob("*")):
        if p.is_dir() and any(p.iterdir()):
            depth = len(p.relative_to(output_dir).parts)
            if depth <= 3:  # Don't go too deep
                indent = "  " * depth
                print(f"  {indent}{p.name}/")

    print()
    print("Next steps:")
    print("  1. python train_dentex_grpo.py --config configs/grpo_config.yaml")
    print("  2. python eval/baseline_eval.py --data data/dentex_test.jsonl")


if __name__ == "__main__":
    main()
