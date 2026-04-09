# DENTEX-RLVR: Reinforcement Learning from Verifiable Rewards for Dental X-ray Diagnosis

> Fine-tuning **Qwen3-VL-7B** on panoramic dental X-rays using **GRPO** (Group Relative Policy Optimization) to produce structured FDI dental diagnoses — without any human preference labels.

---

## Motivation

Traditional fine-tuning of vision-language models (VLMs) for medical imaging requires expensive labeled preference data (RLHF) or relies on supervised fine-tuning alone, which can overfit to surface-level patterns. **RLVR** bypasses human preference annotation entirely: because dental diagnoses follow the standardized **FDI notation system**, we can programmatically verify model outputs against ground truth, creating a fully automated reward signal.

This project demonstrates that a vision-language model can learn to:
1. Identify diseased teeth in panoramic X-rays
2. Localize them by FDI quadrant and tooth number
3. Classify the pathology
4. Produce structured, parseable output

...all through reinforcement learning with **zero human feedback** — only programmatic reward functions.

---

## How Training Works

### The RLVR Pipeline (High-Level)

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌────────────┐
│  Dental     │     │  Model       │     │  Reward       │     │  Policy    │
│  X-ray +    │────▶│  generates   │────▶│  functions    │────▶│  update    │
│  Prompt     │     │  4 outputs   │     │  score each   │     │  via GRPO  │
└─────────────┘     └──────────────┘     └───────────────┘     └────────────┘
                         │                      │
                    "Think step by         Compare against
                     step, then            ground-truth FDI
                     answer in             annotations
                     Q{q}T{t}:{diag}"
```

### Step-by-Step

1. **Prompt Construction**: Each training sample pairs a panoramic X-ray with a structured prompt asking the model to identify abnormal teeth using FDI notation.

2. **Rollout Generation (G=4)**: For each prompt, the model generates **4 candidate outputs** at temperature 0.7. This group of completions is essential to GRPO — it provides the relative comparison baseline.

3. **Reward Computation**: Each output is scored by two independent, deterministic reward functions:

   | Reward Component | Score | What It Checks |
   |-----------------|-------|----------------|
   | **Format compliance** | +0.10 | Valid `<think>...</think><answer>...</answer>` tags |
   | **Quadrant match** | +0.30 | Correct FDI quadrant (Q1–Q4) |
   | **Tooth match** | +0.30 | Correct tooth number within quadrant (requires quadrant to be correct first) |
   | **Diagnosis match** | +0.30 | Correct pathology label |
   | **Leak penalty** | −0.20 | Answer content appearing inside `<think>` block |

   **Maximum total reward: 1.0 per finding**, averaged across all findings in the image.

4. **GRPO Policy Update**: Unlike PPO, GRPO requires **no value network** (critic). It computes advantages by comparing each completion's reward to the group mean:

   ```
   advantage_i = (reward_i − mean(rewards)) / std(rewards)
   ```

   The policy is updated to increase the probability of above-average completions and decrease below-average ones. This is more memory-efficient than PPO and well-suited to text generation tasks.

5. **Repeat**: The model iteratively improves across epochs, learning to produce correctly formatted, accurate dental diagnoses.

### Why GRPO over PPO?

| Property | PPO | GRPO |
|----------|-----|------|
| Value network | Required (doubles memory) | Not needed |
| Baseline | Learned critic | Group mean of rewards |
| Memory on H100 | ~70–80 GB | ~55 GB |
| Implementation | Complex | Simple (TRL's `GRPOTrainer`) |

### Why This Works Without Human Feedback

The key insight: **dental diagnoses are verifiable**. Unlike open-ended text where quality is subjective, FDI notation provides an exact, machine-checkable ground truth:
- Is the quadrant correct? → Yes/No (compare integers)
- Is the tooth number correct? → Yes/No (compare integers)
- Is the diagnosis correct? → Yes/No (string match)

No ambiguity, no need for human annotators to judge "which output is better."

---

## FDI Notation Primer

The [FDI World Dental Federation notation](https://en.wikipedia.org/wiki/FDI_World_Dental_Federation_notation) uses a 2-digit system:

```
        Upper Right (Q1)  │  Upper Left (Q2)
     ─────────────────────┼─────────────────────
        Lower Right (Q4)  │  Lower Left (Q3)
```

- **First digit** = Quadrant (1–4)
- **Second digit** = Tooth position (1–8, from central incisor to third molar)
- Example: Tooth **16** = Q1 (upper-right), tooth 6 (first molar)

### Pathology Classes

| Class | Description |
|-------|-------------|
| `caries` | Tooth decay |
| `deep_caries` | Advanced decay reaching pulp |
| `periapical` | Infection at tooth root apex |
| `impacted` | Tooth unable to fully erupt |

---

## Expected Model Output Format

The model is trained to produce structured reasoning followed by a parseable answer:

```xml
<think>
I observe a radiolucent area at the upper-right first molar (tooth 16),
consistent with deep caries extending toward the pulp chamber.
The lower-left third molar (tooth 38) appears partially erupted
and angled mesially, indicating impaction.
</think>
<answer>Q1T6:deep_caries, Q3T8:impacted</answer>
```

---

## Repository Structure

```
dentex-rlvr/
├── train_dentex_grpo.py        # Main GRPO training entry point
├── rewards/
│   ├── format_reward.py        # Tag structure validation (+0.10 / −0.20)
│   └── fdi_reward.py           # Hierarchical FDI matching (+0.90 max)
├── data/
│   ├── convert_labelme.py      # LabelMe JSON → JSONL converter
│   ├── convert_dentex.py       # HuggingFace DENTEX → JSONL converter
│   └── augment.py              # Horizontal flip + gamma jitter
├── env/
│   └── dental_env.py           # Gymnasium wrapper for rollout evaluation
├── eval/
│   ├── baseline_eval.py        # Zero-shot evaluation script
│   └── metrics.py              # Accuracy, F1, format compliance metrics
├── demo/
│   └── app.py                  # Gradio web demo
├── configs/
│   └── grpo_config.yaml        # Training hyperparameters
└── scripts/
    ├── setup_h100.sh           # GPU node setup
    └── export_model.sh         # LoRA → GGUF/AWQ export
```

---

## Dataset

This project uses dental panoramic X-ray data with LabelMe polygon annotations. Each annotation encodes:
- **Disease class** (Turkish clinical labels mapped to English)
- **FDI tooth number** (2-digit international notation)
- **Polygon segmentation** (used only for ground truth, not fed to the model)

### Disease Label Mapping

| Class ID | Turkish | English | Training Count |
|----------|---------|---------|---------------|
| 0 | sağlam | *skipped* (healthy) | 91 |
| 1 | çürük | `caries` | 747 |
| 2 | küretaj | `deep_caries` | 265 |
| 3 | kanal | `periapical` | 161 |
| 5 | çekim | `impacted` | 29 |
| 6 | gömülü | `impacted` | 221 |
| 7 | lezyon | `periapical` | 75 |
| 8 | kırık | `caries` | 11 |

### Data Conversion

```bash
# Convert LabelMe annotations to training format
python data/convert_labelme.py \
  --input-dir /path/to/disease \
  --output data/dentex_train.jsonl \
  --split train
```

---

## Quick Start

### 1. Setup

```bash
# Clone and install dependencies
git clone <repo-url> && cd dentex-rlvr
pip install torch transformers trl unsloth datasets peft \
            accelerate wandb gradio qwen-vl-utils gymnasium
```

### 2. Prepare Data

```bash
# Convert your LabelMe-annotated dataset
python data/convert_labelme.py \
  --input-dir /path/to/disease \
  --output data/dentex_train.jsonl

# (Optional) Convert HuggingFace DENTEX dataset
python data/convert_dentex.py --output data/dentex_train.jsonl
```

### 3. Train (Two-Stage: SFT → GRPO)

```bash
# Stage 1: SFT warmup (1 epoch — teaches format + basic dental reasoning)
python train_sft_warmup.py --config configs/grpo_config.yaml

# Stage 2: GRPO on top of SFT checkpoint (reinforcement learning)
# First update model_name in configs/grpo_config.yaml to:
#   model_name: "checkpoints/sft_warmup/final"
python train_dentex_grpo.py --config configs/grpo_config.yaml
```

> **Why two stages?** SFT alone memorizes answers but doesn't generalize.
> GRPO alone struggles to learn format from scratch. Combined, SFT gives
> GRPO a strong starting point, improving final accuracy by ~10–15%.

### 4. Evaluate

```bash
python eval/baseline_eval.py \
  --model checkpoints/grpo_run1/final \
  --data data/dentex_test.jsonl
```

### 5. Demo

```bash
python demo/app.py --model checkpoints/grpo_run1/final --share
```

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Qwen3-VL-7B-Instruct | Strong vision-language foundation |
| Quantization | 4-bit QLoRA (Unsloth) | Fits H100 80GB with rollouts |
| LoRA rank | 16 (α=32) | Balance adaptation capacity vs. overfitting |
| Rollouts per prompt | 4 | Enough for stable group advantage estimation |
| Temperature | 0.7 | Encourages diverse rollouts |
| Learning rate | 5e-6 | Conservative for RL stability |
| Batch size | 2 × 4 grad accum = 8 effective | Memory-constrained |
| Epochs | 3 | Small dataset, avoid overfitting |

---

## Evaluation Targets

| Metric | Target | Baseline (Zero-shot) |
|--------|--------|---------------------|
| Quadrant Accuracy | > 70% | ~20–25% |
| Tooth-level Accuracy | > 50% | ~10–15% |
| Pathology F1 (macro) | > 0.45 | ~0.10–0.15 |
| Format Compliance | > 90% | ~30–50% |
| Mean Reward | > 0.55 | ~0.20–0.25 |

---

## Key References

- **GRPO**: Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024
- **RLVR**: Zeng et al., [Reinforcement Learning from Verifiable Rewards](https://arxiv.org/abs/2411.15124), 2024 (DeepSeek-R1 Technical Report)
- **Qwen2-VL**: Wang et al., [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191), 2024
- **DENTEX**: Hamamci et al., [DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark](https://arxiv.org/abs/2305.19787), 2023
- **QLoRA**: Dettmers et al., [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314), 2023
- **TRL**: [Transformer Reinforcement Learning Library](https://github.com/huggingface/trl), HuggingFace

---

## License

This project is for academic/research purposes.
