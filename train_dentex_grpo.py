"""Main GRPO training entry point for DENTEX dental diagnosis."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from rewards.format_reward import format_reward
from rewards.fdi_reward import fdi_reward


@dataclass
class TrainConfig:
    """Training configuration loaded from YAML."""

    # Model
    model_name: str = "Qwen/Qwen3-VL-7B-Instruct"
    load_in_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

    # GRPO
    num_generations: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Training
    learning_rate: float = 5e-6
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    use_gradient_checkpointing: bool = True

    # Checkpointing
    save_steps: int = 50
    save_total_limit: int = 5
    output_dir: str = "checkpoints/grpo_run1"

    # Logging
    logging_steps: int = 10
    report_to: str = "wandb"
    wandb_project: str = "dentex-rlvr"
    wandb_run_name: str = "grpo_run1_baseline"

    # Data
    train_data_path: str = "data/dentex_train.jsonl"
    test_data_path: str = "data/dentex_test.jsonl"
    eval_steps: int = 100

    # Seed
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        with path.open() as f:
            data = yaml.safe_load(f)
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL dataset."""
    samples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_prompt_dataset(samples: list[dict]) -> Dataset:
    """Convert JSONL samples to HuggingFace Dataset for GRPOTrainer.

    GRPOTrainer expects a dataset with a "prompt" column containing
    the conversation messages, and we store ground_truth separately.
    """
    prompts = []
    ground_truths = []

    for sample in samples:
        prompts.append(sample["messages"])
        ground_truths.append(json.dumps(sample["ground_truth"]))

    return Dataset.from_dict({
        "prompt": prompts,
        "ground_truth": ground_truths,
    })


def make_reward_fn(
    tokenizer: Any,
) -> callable:
    """Create the combined reward function for GRPOTrainer.

    GRPOTrainer calls reward_fn(completions, prompts=...) where completions
    is a list of generated text strings.
    """

    def reward_fn(completions: list[str], **kwargs: Any) -> list[float]:
        ground_truths = kwargs.get("ground_truth", [])
        rewards = []

        for completion, gt_str in zip(completions, ground_truths):
            gt = json.loads(gt_str) if isinstance(gt_str, str) else gt_str

            r_format = format_reward(completion, gt)
            r_fdi = fdi_reward(completion, gt)
            total = r_format + r_fdi

            rewards.append(total)

        return rewards

    return reward_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="DENTEX GRPO Training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/grpo_config.yaml"),
        help="Path to GRPO config YAML",
    )
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)

    # Load model with Unsloth QLoRA
    print(f"Loading model: {cfg.model_name} (4-bit={cfg.load_in_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=2048,
        load_in_4bit=cfg.load_in_4bit,
        dtype=None,  # auto-detect
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
    )

    # Load training data
    print(f"Loading training data: {cfg.train_data_path}")
    train_samples = load_jsonl(Path(cfg.train_data_path))
    train_dataset = build_prompt_dataset(train_samples)
    print(f"  {len(train_dataset)} training samples")

    # Load eval data if available
    eval_dataset = None
    test_path = Path(cfg.test_data_path)
    if test_path.exists():
        test_samples = load_jsonl(test_path)
        eval_dataset = build_prompt_dataset(test_samples)
        print(f"  {len(eval_dataset)} eval samples")

    # Build reward function
    reward_fn = make_reward_fn(tokenizer)

    # Configure GRPO training
    grpo_config = GRPOConfig(
        output_dir=cfg.output_dir,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_new_tokens,
        temperature=cfg.temperature,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
        seed=cfg.seed,
        bf16=True,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=cfg.eval_steps if eval_dataset else None,
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    # Train
    print("Starting GRPO training...")
    trainer.train()

    # Save final model
    final_dir = Path(cfg.output_dir) / "final"
    print(f"Saving final model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("Training complete!")


if __name__ == "__main__":
    main()
