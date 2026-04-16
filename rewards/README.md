# Reward Functions

This module implements the **verifiable reward functions** used during GRPO training of the DENTEX-RLVR model. The two reward signals here are the only feedback the policy receives — there is no human preference model, no learned critic, and no separate value network. Every reward is produced by a pure, deterministic Python function that compares the model's textual output against programmatically verifiable ground truth.

```
rewards/
├── __init__.py          # Public exports: format_reward, fdi_reward
├── format_reward.py     # Structural (tag-format) reward
└── fdi_reward.py        # Semantic (FDI dental finding) reward
```

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Public API](#public-api)
3. [Expected Input/Output Schema](#expected-inputoutput-schema)
4. [Format Reward (`format_reward`)](#format-reward-format_reward)
5. [FDI Reward (`fdi_reward`)](#fdi-reward-fdi_reward)
6. [Scoring Summary](#scoring-summary)
7. [Worked Examples](#worked-examples)
8. [Edge Cases & Return Values](#edge-cases--return-values)
9. [Integration with GRPO Trainer](#integration-with-grpo-trainer)
10. [Design Rationale](#design-rationale)

---

## Design Philosophy

The reward stack is split into **two orthogonal, additive components**:

| Component        | Purpose                                    | Max Reward | Min Reward |
|------------------|--------------------------------------------|------------|------------|
| `format_reward`  | Enforce the `<think>…</think><answer>…</answer>` scaffold | +0.10      | −0.20      |
| `fdi_reward`     | Score semantic correctness of dental findings | +0.90 (3 × 0.30) averaged per finding | 0.00 |

Each function has the same signature:

```python
def reward(model_output: str, ground_truth: dict) -> float
```

This uniform interface lets the GRPO trainer sum or weight rewards arbitrarily — in this project they are combined additively so a perfectly formatted, perfectly-correct answer yields approximately **+1.00**.

The functions are:
- **Deterministic** — the same input always produces the same score.
- **Stateless** — no model, no database, no network call.
- **Fast** — pure regex + Python; scales trivially to batches.
- **Interpretable** — each sub-component (quadrant / tooth / diagnosis / format) contributes a known fixed score.

---

## Public API

```python
from rewards import format_reward, fdi_reward
```

Both functions are registered in `rewards/__init__.py` and consumed by the GRPO training entry point (`train_dentex_grpo.py`). No other symbols are exported.

---

## Expected Input/Output Schema

### `model_output: str`

The raw text produced by the policy model for a single rollout. The reward functions expect the model to follow the training prompt instruction:

```
<think>
…chain-of-thought reasoning…
</think>
<answer>
Q{quadrant}T{tooth}:{diagnosis}
Q{quadrant}T{tooth}:{diagnosis}
…
</answer>
```

Where:
- `quadrant` ∈ `{1, 2, 3, 4}` (FDI quadrant)
- `tooth`    ∈ `{1, 2, 3, 4, 5, 6, 7, 8}` (FDI tooth position)
- `diagnosis` ∈ `{caries, deep_caries, periapical, impacted}`

### `ground_truth: dict`

A dictionary containing at least a `"findings"` key:

```python
{
    "findings": [
        {"quadrant": 1, "tooth": 6, "diagnosis": "caries"},
        {"quadrant": 3, "tooth": 8, "diagnosis": "impacted"},
    ]
}
```

Any additional keys are ignored, allowing the dataset loader to pass richer metadata without breaking the reward interface.

### Return Value

A single `float` scalar. The GRPO trainer uses these per-rollout scalars to compute group-normalized advantages:

```
advantage_i = (reward_i − mean(group_rewards)) / std(group_rewards)
```

---

## Format Reward (`format_reward`)

**File:** `rewards/format_reward.py`

### Purpose

Enforces the two-block scaffold the downstream parser relies on. Without a clean format signal, the model tends to emit free-form prose that is hard to parse and easy to game.

### Constants

```python
FORMAT_SCORE = 0.10    # Reward for correctly formatted output
LEAK_PENALTY = -0.20   # Penalty when answer content appears inside <think>
```

### Algorithm

1. If `model_output` is empty/whitespace → return `0.0`.
2. Check the **entire string** against `^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$` (DOTALL).
   - If the match fails → return `0.0`.
3. Re-extract the `<think>` and `<answer>` groups with their individual regexes. If either is missing → return `0.0`.
4. **Leak detection:** search the `<think>` block for the pattern `Q[1-4]T[1-8]:\w+`. If found, the model is "thinking in answers" — return `LEAK_PENALTY` (−0.20).
5. If the `<answer>` block is empty after stripping → return `0.0`.
6. Otherwise → return `FORMAT_SCORE` (+0.10).

### Why the Leak Penalty Exists

During early training, models often write their final answer tokens inside the `<think>` block, effectively skipping the reasoning stage. The negative reward actively discourages this collapse of the two-block structure.

---

## FDI Reward (`fdi_reward`)

**File:** `rewards/fdi_reward.py`

### Purpose

Measures semantic correctness against dental ground truth using the standardized [FDI World Dental Federation notation](https://en.wikipedia.org/wiki/FDI_World_Dental_Federation_notation). Because FDI is a fully discrete coordinate system (quadrant + tooth + diagnosis), exact matching is both meaningful and verifiable.

### Constants

```python
QUADRANT_SCORE = 0.30   # Reward for matching quadrant (Q1–Q4)
TOOTH_SCORE    = 0.30   # Reward for matching tooth number (ONLY if quadrant is correct)
DIAG_SCORE     = 0.30   # Reward for matching diagnosis (independent of location)

VALID_DIAGNOSES = {"caries", "deep_caries", "periapical", "impacted"}
```

### Patterns

```python
_ANSWER_PATTERN  = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_FINDING_PATTERN = re.compile(r"Q([1-4])T([1-8]):(caries|deep_caries|periapical|impacted)")
```

Anything inside `<answer>…</answer>` that does not match `_FINDING_PATTERN` is silently ignored — so the model is free to add commas, line breaks, or commentary between findings without being penalized on semantics.

### Algorithm

1. Gather ground-truth findings. If the list is empty → return `0.0`.
2. Extract the `<answer>` block. If absent → return `0.0`.
3. Parse all `Qq Tt :diag` findings from the answer. If none → return `0.0`.
4. For each ground-truth finding `gt`:
   - Iterate every predicted finding `pred`; compute a per-pair score:
     - `+0.30` if `pred.quadrant == gt.quadrant`
     - `+0.30` if **both** quadrants match **and** `pred.tooth == gt.tooth` (tooth score is gated by quadrant)
     - `+0.30` if `pred.diagnosis == gt.diagnosis` (independent of location)
   - Take the **best** score across all predictions as the match for this ground-truth finding.
5. Return the **average** of best-scores over all ground-truth findings.

### Hierarchical (Gated) Tooth Scoring

Note that tooth credit is only awarded when the **quadrant is already correct**. This mirrors the anatomical hierarchy: a "tooth 6" in the wrong quadrant is a different tooth entirely. Decoupling these would reward partial location guessing that has no clinical meaning.

### Greedy Per-GT Matching

For every ground-truth finding, the algorithm picks the **single best-scoring prediction** (without marking it as "used"). This means a model that outputs one perfect prediction and repeats it will get full credit for any ground-truth finding it happens to match — but because the final reward is **averaged over ground-truth findings**, the model still cannot exceed 0.90 (the per-finding max) without actually covering every finding.

---

## Scoring Summary

| Sub-score              | Where                     | Value    | Conditions                                                   |
|------------------------|---------------------------|----------|--------------------------------------------------------------|
| Format valid           | `format_reward`           | **+0.10** | Exact `<think>…</think><answer>…</answer>` shape, non-empty answer |
| Leak (thinking ≈ answering) | `format_reward`      | **−0.20** | `Q[1-4]T[1-8]:\w+` detected inside `<think>`                 |
| Quadrant correct       | `fdi_reward` per GT       | **+0.30** | `pred.quadrant == gt.quadrant`                               |
| Tooth correct          | `fdi_reward` per GT       | **+0.30** | Quadrant **and** tooth both match                            |
| Diagnosis correct      | `fdi_reward` per GT       | **+0.30** | `pred.diagnosis == gt.diagnosis`                             |

Maximum reward per rollout:
- Format: `+0.10`
- FDI (averaged over GT findings): up to `+0.90`
- **Total theoretical max ≈ +1.00**

---

## Worked Examples

### Example 1 — Perfect Answer

**Ground Truth**

```python
{"findings": [{"quadrant": 1, "tooth": 6, "diagnosis": "caries"}]}
```

**Model Output**

```
<think>
The upper-right molar area shows radiolucency consistent with caries on tooth 16.
</think>
<answer>
Q1T6:caries
</answer>
```

**Scoring**
- `format_reward`: scaffold valid, no leak → `+0.10`
- `fdi_reward`: quadrant ✓ (+0.30) + tooth ✓ (+0.30) + diagnosis ✓ (+0.30) = `+0.90`
- **Total: `+1.00`**

### Example 2 — Correct Diagnosis, Wrong Quadrant

**Ground Truth:** `Q1T6:caries`
**Model Output (answer):** `Q2T6:caries`

- Quadrant mismatch → 0
- Tooth score gated off (quadrant wrong) → 0
- Diagnosis match → +0.30
- `fdi_reward` = 0.30
- `format_reward` (assuming valid tags) = +0.10
- **Total: +0.40**

### Example 3 — Correct Location, Wrong Diagnosis

**Ground Truth:** `Q1T6:caries`
**Model Output:** `Q1T6:periapical`

- Quadrant ✓ (+0.30)
- Tooth ✓ (+0.30)
- Diagnosis ✗ (0)
- `fdi_reward` = 0.60
- `format_reward` = +0.10
- **Total: +0.70**

### Example 4 — Thinking Contains the Answer (Leak)

```
<think>
I see Q1T6:caries in the X-ray.
</think>
<answer>
Q1T6:caries
</answer>
```

- `format_reward` detects `Q1T6:caries` in think block → `−0.20`
- `fdi_reward` unaffected by leak logic → `+0.90`
- **Total: +0.70**

### Example 5 — Multiple Findings, Partial Coverage

**Ground Truth**

```python
{"findings": [
    {"quadrant": 1, "tooth": 6, "diagnosis": "caries"},
    {"quadrant": 3, "tooth": 8, "diagnosis": "impacted"},
]}
```

**Model Output (answer):** `Q1T6:caries`

- GT #1: best pred = `Q1T6:caries` → 0.90
- GT #2: best pred = `Q1T6:caries` → quadrant 1 ≠ 3 (0) + diagnosis caries ≠ impacted (0) = 0.0
- Average = (0.90 + 0.0) / 2 = **0.45**
- `format_reward` = +0.10
- **Total: +0.55**

---

## Edge Cases & Return Values

| Scenario                                     | `format_reward` | `fdi_reward` |
|----------------------------------------------|-----------------|--------------|
| Empty / whitespace-only output               | 0.00            | 0.00         |
| Missing `<think>` or `<answer>` tag          | 0.00            | 0.00         |
| Tags present but `<answer>` empty            | 0.00            | 0.00 (no findings parsed) |
| Valid tags, no `Q…T…:` tokens in answer      | +0.10           | 0.00         |
| Ground truth has no findings                 | (unaffected)    | 0.00         |
| `ground_truth` missing the `"findings"` key  | (unaffected)    | 0.00 (defaults to `[]`) |
| `model_output is None`                       | 0.00 (falsy check) | 0.00 (coerced via `model_output or ""`) |
| Multiple predictions, only some correct      | (unaffected)    | Best-match per GT, then averaged |

---

## Integration with GRPO Trainer

Both functions conform to the callable signature expected by `trl.GRPOTrainer`:

```python
def reward_fn(model_output: str, ground_truth: dict) -> float
```

In `train_dentex_grpo.py`, both reward functions are registered so that each rollout receives a summed score. GRPO then:

1. Samples `G=4` completions per prompt.
2. Calls each reward function on every completion.
3. Sums the rewards (format + FDI) into a single scalar per completion.
4. Normalizes within the group: `advantage_i = (r_i − mean) / std`.
5. Updates the policy toward above-average completions.

Because the rewards are bounded (`format_reward ∈ [−0.20, +0.10]`, `fdi_reward ∈ [0.00, +0.90]`), the normalized advantages remain numerically well-behaved throughout training.

---

## Design Rationale

### Why separate format and semantics?

Decoupling lets the model learn the scaffold quickly (format reward saturates in the first few hundred steps) and then spend the rest of training improving diagnostic accuracy. A combined reward would couple these learning signals and slow convergence.

### Why the leak penalty?

Without it, the chain-of-thought collapses: the model discovers it can satisfy the `<answer>` block by copying tokens it already produced in `<think>`, skipping real reasoning. The −0.20 penalty is larger than the +0.10 format reward, so any leak produces a net negative format score.

### Why hierarchical (gated) tooth scoring?

FDI tooth numbering is only meaningful within a quadrant — tooth 6 in Q1 is the upper-right first molar, while tooth 6 in Q3 is the lower-left first molar. Giving partial credit for matching the tooth number when the quadrant is wrong would reward the model for the wrong answer.

### Why best-match averaging instead of bipartite matching?

Greedy per-GT matching is:
- Simpler (no Hungarian algorithm / LP solver).
- Robust to duplicated predictions (a duplicate simply loses once averaging divides).
- Empirically sufficient — the training signal is relative (GRPO), not absolute.

### Why these four diagnoses?

`caries`, `deep_caries`, `periapical`, and `impacted` are the canonical label set in the [DENTEX challenge](https://github.com/ibrahimethemhamamci/DENTEX). Matching the dataset's label space keeps the reward function aligned with the ground-truth annotations.

---

## Quick Reference

```python
from rewards import format_reward, fdi_reward

gt = {"findings": [{"quadrant": 1, "tooth": 6, "diagnosis": "caries"}]}
out = "<think>upper right molar</think><answer>Q1T6:caries</answer>"

print(format_reward(out, gt))  # 0.10
print(fdi_reward(out, gt))     # 0.90
# Combined in GRPOTrainer: ≈ 1.00
```
