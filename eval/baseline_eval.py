"""Zero-shot baseline evaluation on DENTEX test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from eval.metrics import compute_all_metrics

SYSTEM_PROMPT = "You are a dental radiologist. Analyze the panoramic X-ray."

USER_PROMPT = (
    "Identify all abnormal teeth. For each, output FDI quadrant (1–4), "
    "tooth number (1–8), and diagnosis (caries/deep_caries/periapical/impacted). "
    "Think step by step inside <think> tags. Output answer inside <answer> tags. "
    "Format: <answer>Q{q}T{t}:{diag}, Q{q}T{t}:{diag}, ...</answer>"
)


def load_test_data(data_path: Path) -> list[dict]:
    """Load JSONL test data."""
    samples = []
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def run_inference(
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    samples: list[dict],
    max_new_tokens: int = 512,
    batch_size: int = 1,
) -> list[str]:
    """Run zero-shot inference on all samples.

    Args:
        model: Loaded Qwen2VL model.
        processor: Corresponding processor/tokenizer.
        samples: List of sample dicts with "messages" key.
        max_new_tokens: Max tokens to generate.
        batch_size: Inference batch size.

    Returns:
        List of model output strings.
    """
    outputs = []
    device = next(model.parameters()).device

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]

        for sample in batch:
            messages = sample["messages"]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=text, return_tensors="pt").to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            # Decode only the generated tokens (skip input)
            input_len = inputs["input_ids"].shape[1]
            output_text = processor.decode(
                generated[0][input_len:], skip_special_tokens=True
            )
            outputs.append(output_text)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(samples))}/{len(samples)}")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot baseline eval")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-7B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/dentex_test.jsonl"),
        help="Test data JSONL path",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save results JSON",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"Loading test data: {args.data}")
    samples = load_test_data(args.data)
    print(f"Loaded {len(samples)} test samples")

    print("Running inference...")
    model_outputs = run_inference(
        model, processor, samples, max_new_tokens=args.max_new_tokens
    )

    ground_truths = [s["ground_truth"]["findings"] for s in samples]
    metrics = compute_all_metrics(model_outputs, ground_truths)

    print("\n=== Baseline Evaluation Results ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    if args.output:
        results = {
            "model": args.model,
            "num_samples": len(samples),
            "metrics": metrics,
            "predictions": model_outputs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
