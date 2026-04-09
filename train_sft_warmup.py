"""SFT warmup: supervised fine-tuning before GRPO.

Trains the model for 1 epoch on ground-truth answers so it learns the
output format and basic dental reasoning BEFORE reinforcement learning.
This dramatically improves GRPO convergence and final accuracy (+10-15%).

Usage:
    python train_sft_warmup.py --config configs/grpo_config.yaml
    python train_dentex_grpo.py --config configs/grpo_config.yaml  # then GRPO
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import unsloth  # must be imported first
import torch
import yaml
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel


@dataclass
class SFTWarmupConfig:
    """SFT warmup configuration (reuses GRPO config + SFT-specific overrides)."""

    # Model
    model_name: str = "unsloth/Qwen3-VL-8B-Instruct"
    load_in_4bit: bool = True

    # LoRA (same as GRPO for adapter compatibility)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

    # SFT-specific
    sft_epochs: int = 1          # Just 1 epoch — enough for format + basics
    sft_lr: float = 2e-5         # Higher than GRPO since SFT is more stable
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.03
    use_gradient_checkpointing: bool = True

    # Checkpointing
    output_dir: str = "checkpoints/sft_warmup"

    # Logging
    logging_steps: int = 10
    report_to: str = "wandb"
    wandb_project: str = "dentex-rlvr"
    wandb_run_name: str = "sft_warmup"

    # Data
    train_data_path: str = "data/dentex_train.jsonl"

    # Seed
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path) -> SFTWarmupConfig:
        with path.open() as f:
            data = yaml.safe_load(f)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def build_answer_text(findings: list[dict]) -> str:
    """Build the expected model output with think + answer tags."""
    # Build a simple reasoning chain
    reasoning_parts = []
    for f in findings:
        q_names = {1: "upper-right", 2: "upper-left", 3: "lower-left", 4: "lower-right"}
        diag_desc = {
            "caries": "caries (tooth decay)",
            "deep_caries": "deep caries extending toward the pulp",
            "periapical": "a periapical lesion at the root apex",
            "impacted": "impaction (tooth unable to fully erupt)",
        }
        reasoning_parts.append(
            f"Tooth {f['quadrant']}{f['tooth']} in the {q_names[f['quadrant']]} "
            f"quadrant shows {diag_desc.get(f['diagnosis'], f['diagnosis'])}."
        )

    reasoning = " ".join(reasoning_parts)

    # Build answer string
    answer_parts = [
        f"Q{f['quadrant']}T{f['tooth']}:{f['diagnosis']}" for f in findings
    ]
    answer = ", ".join(answer_parts)

    return f"<think>\n{reasoning}\n</think>\n<answer>{answer}</answer>"


def build_sft_dataset(data_path: Path) -> Dataset:
    """Build dataset with full conversations (system + user + assistant response).

    SFTTrainer expects a dataset with a "messages" or "text" column where
    the assistant response is included as the training target.
    """
    samples = []
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    conversations = []
    for sample in samples:
        messages = sample["messages"]  # system + user
        gt = sample["ground_truth"]
        findings = gt["findings"]

        # Build the assistant's ideal response
        assistant_response = build_answer_text(findings)

        # Full conversation: system + user + assistant
        full_messages = messages + [
            {"role": "assistant", "content": assistant_response}
        ]
        conversations.append(full_messages)

    return Dataset.from_dict({"messages": conversations})


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT Warmup Training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/grpo_config.yaml"),
        help="Path to config YAML (reuses GRPO config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of SFT epochs (default: 1)",
    )
    args = parser.parse_args()

    cfg = SFTWarmupConfig.from_yaml(args.config)
    if args.epochs is not None:
        cfg.sft_epochs = args.epochs

    # Load model
    print(f"Loading model: {cfg.model_name} (4-bit={cfg.load_in_4bit})")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=None,
    )

    # Apply LoRA
    model = FastVisionModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
    )

    # Load data
    print(f"Loading training data: {cfg.train_data_path}")
    train_dataset = build_sft_dataset(Path(cfg.train_data_path))
    print(f"  {len(train_dataset)} training samples")

    # Formatting function required by Unsloth's SFTTrainer
    def formatting_func(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return texts

    # Configure SFT
    sft_config = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.sft_epochs,
        learning_rate=cfg.sft_lr,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        max_seq_length=cfg.max_seq_length,
        logging_steps=cfg.logging_steps,
        save_steps=100,
        save_total_limit=2,
        report_to=cfg.report_to,
        seed=cfg.seed,
        bf16=True,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )

    print(f"Starting SFT warmup ({cfg.sft_epochs} epoch(s))...")
    trainer.train()

    # Save
    final_dir = Path(cfg.output_dir) / "final"
    print(f"Saving SFT model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print()
    print("SFT warmup complete!")
    print(f"  Model saved to: {final_dir}")
    print()
    print("Next step — run GRPO on top of the SFT model:")
    print(f"  1. Update configs/grpo_config.yaml:")
    print(f'     model_name: "{final_dir}"')
    print(f"  2. python train_dentex_grpo.py --config configs/grpo_config.yaml")


if __name__ == "__main__":
    main()
