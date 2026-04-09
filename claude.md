# DENTEX RLVR — Project Instructions

## Project Overview

Fine-tune **Qwen3-VL-7B** on the **DENTEX** panoramic dental X-ray dataset using **RLVR (GRPO)** to perform hierarchical dental diagnosis: quadrant detection → tooth enumeration → pathology classification. Training uses **Unsloth QLoRA + TRL GRPOTrainer** on an H100 (48h budget). Final deliverable is a **Gradio demo** on HuggingFace Spaces.

This is a **toy research project** — scope is intentionally narrow. Do not over-engineer.

## Tech Stack

- **Base model:** `Qwen/Qwen3-VL-7B-Instruct` (4-bit QLoRA via Unsloth)
- **RL:** GRPO via `trl.GRPOTrainer` — no value network
- **Dataset:** `ibrahimhamamci/DENTEX` on HuggingFace (1,005 panoramic X-rays)
- **Env:** Custom `GymDentalEnv` (Gymnasium-compatible) wrapping DENTEX annotations
- **Tracking:** Weights & Biases
- **Inference/Export:** vLLM, GGUF/AWQ via Unsloth export
- **Demo:** Gradio 4.x on HF Spaces
- **Compute:** H100 SXM 80GB (RunPod spot preferred ~$2.50/hr)

## Repository Structure

```
dentex-rlvr/
├── claude.md                  # This file
├── README.md                  # Project writeup (Day 7)
├── train_dentex_grpo.py       # Main GRPO training entry point
├── env/
│   └── dental_env.py          # GymDentalEnv — Gymnasium wrapper over DENTEX
├── rewards/
│   ├── format_reward.py       # Checks <think>/<answer> tag structure
│   └── fdi_reward.py          # Hierarchical FDI match (quadrant + tooth + diag)
├── data/
│   ├── convert_dentex.py      # DENTEX HF → JSONL conversion
│   └── augment.py             # Horizontal flip + gamma jitter augmentations
├── eval/
│   ├── baseline_eval.py       # Zero-shot eval on test split
│   └── metrics.py             # Per-component accuracy, macro-F1, format compliance
├── demo/
│   └── app.py                 # Gradio app (single file, HF Spaces deployable)
├── configs/
│   └── grpo_config.yaml       # GRPOConfig hyperparameters
└── scripts/
    ├── setup_h100.sh          # H100 node setup (pip installs, env vars)
    └── export_model.sh        # LoRA → GGUF/AWQ export
```

## Key Domain Concepts

### FDI Notation
- **Quadrants:** Q1 (upper-right), Q2 (upper-left), Q3 (lower-left), Q4 (lower-right)
- **Tooth numbers:** 1–8 within each quadrant
- **Pathology classes:** `caries`, `deep_caries`, `periapical`, `impacted`

### DENTEX Splits
| Split | N | Labels | Reward Use |
|-------|---|--------|------------|
| Quadrant | 693 | Q1–Q4 | +0.30 for correct quadrant |
| Tooth Enum | 634 | Quadrant + tooth 1–8 | +0.30 for correct tooth ID |
| Full Diagnosis | 1,005 | 4 pathology classes | +0.30 for correct diagnosis |

### Reward Function (max = 1.0)
- `format_reward = 0.10` — valid `<think>...</think><answer>...</answer>` tags
- `quadrant_reward = 0.30` — FDI quadrant exact match
- `tooth_reward = 0.30` — tooth number match (only if quadrant correct)
- `diag_reward = 0.30` — pathology label match
- **Penalty:** `-0.20` if answer content leaks inside `<think>` block

### Prompt Template
```
SYSTEM: You are a dental radiologist. Analyze the panoramic X-ray.
USER: Identify all abnormal teeth. For each, output FDI quadrant (1–4),
      tooth number (1–8), and diagnosis (caries/deep_caries/periapical/impacted).
      Think step by step inside <think> tags. Output answer inside <answer> tags.
      Format: <answer>Q{q}T{t}:{diag}, Q{q}T{t}:{diag}, ...</answer>
```

### Expected Model Output
```
<think>
The X-ray shows a maxillary right quadrant with a large radiolucency at tooth 16.
The lesion extends into the pulp chamber, consistent with deep caries...
</think>
<answer>Q1T6:deep_caries, Q3T6:periapical</answer>
```

## Coding Conventions

- **Python 3.10+**, type hints on function signatures
- Use `pathlib.Path` over string paths
- Config via dataclasses or YAML — no magic constants in training scripts
- Reward functions must be **pure functions**: `(model_output: str, ground_truth: dict) -> float`
- All reward functions need unit tests with edge cases (empty output, malformed tags, partial matches)
- Checkpoint every 50–100 steps during GRPO training
- Log everything to W&B: reward curves, format compliance rate, KL divergence, per-class accuracy

## LoRA / Unsloth Config

```python
# Standard config — do not change without ablation justification
r = 16
lora_alpha = 32
target_modules = ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
load_in_4bit = True
use_gradient_checkpointing = True
```

## GRPO Training Config

```python
# Baseline config for Run 1
num_generations = 4       # rollouts per prompt
max_new_tokens = 512
# KL coefficient (beta) — tune on Day 4 if reward hacking detected
# Batch size tuned to fit ~55GB VRAM on H100
```

## Success Targets

| Metric | Target |
|--------|--------|
| Quadrant Accuracy | > 70% |
| Tooth-level Accuracy | > 50% |
| Pathology F1 (macro) | > 0.45 |
| Format Compliance | > 90% |
| Mean Episode Reward | > 0.55 (baseline ~0.20–0.25) |
| Reward Hacking Rate | < 15% |
| Demo Inference Latency | < 8s per image |

## Known Risks to Watch For

1. **Reward hacking** — model reveals answer inside `<think>`. Detect with regex, penalize with -0.20.
2. **Small dataset** (1,005 images) — augment with flips + gamma jitter; consider SFT warmup on unlabeled split.
3. **Tooth localization** — Qwen3-VL may struggle without bbox prompts. Fallback: text-only FDI output (no spatial overlay).
4. **Spot preemption** — checkpoint aggressively, use RunPod persistent volumes.

## Common Commands

```bash
# Setup H100 node
bash scripts/setup_h100.sh

# Convert DENTEX to JSONL
python data/convert_dentex.py --output data/dentex_train.jsonl

# Zero-shot baseline eval
python eval/baseline_eval.py --model Qwen/Qwen3-VL-7B-Instruct --split test

# GRPO training
python train_dentex_grpo.py --config configs/grpo_config.yaml

# Export trained model
bash scripts/export_model.sh --format gguf --adapter checkpoints/best

# Launch Gradio demo locally
python demo/app.py
```

## Do NOT

- Add unnecessary abstractions — this is a 7-day project, not a framework
- Use bfloat16 full fine-tuning — always 4-bit QLoRA via Unsloth
- Skip W&B logging — every training run must be tracked
- Hardcode file paths — use CLI args or config files
- Ignore format compliance — it's a core metric, not a nice-to-have
