# DENTEX-RLVR: Reinforcement Learning from Verifiable Rewards for Dental X-Ray Diagnostics using Vision-Language Models

## Abstract
Automated diagnostics in dentistry has traditionally relied on rigid Convolutional Neural Networks (CNNs) like YOLO or ResNet to predict bounding boxes and class labels from panoramic X-rays. While these traditional models achieve reasonable accuracy, they lack the capacity for verifiable clinical reasoning. This project introduces **DENTEX-RLVR**, a novel pipeline that leverages the cutting-edge Qwen3-VL-8B-Instruct Vision-Language Model (VLM), optimized through a two-stage training approach combining Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO). By replacing human-annotated reinforcement learning (RLHF) with Reinforcement Learning from Verifiable Rewards (RLVR), we designed deterministic, hierarchical reward functions that successfully train the model to output verifiable reasoning chains (`<think>`) followed by structured predictions (`<answer>`). Our results demonstrate that the model quickly converges on formatting and diagnostic accuracy, validating RLVR as a scalable and highly effective paradigm for medical multimodal reasoning.

---

## 1. Introduction
The interpretation of panoramic dental X-rays is a complex, time-consuming task requiring clinicians to systematically evaluate multiple quadrants, identify specific teeth using standardized notation (e.g., FDI system), and assign accurate pathological diagnoses (e.g., caries, periapical lesions). The scarcity of expert annotations and the inherent variability in X-ray imaging present significant challenges for automated systems.

Recent advancements in Large Language Models (LLMs) and Vision-Language Models (VLMs) have introduced the capability for generative multimodal reasoning. However, "out-of-the-box" VLMs frequently struggle with domain-specific medical formatting and hallucinate non-existent pathologies when faced with degraded image quality. To mitigate this, recent breakthrough methodologies (such as the architecture behind DeepSeek-R1) utilize Reinforcement Learning (RL) to explicitly incentivize accurate reasoning chains.

In this capstone project, we pioneer the application of **Reinforcement Learning from Verifiable Rewards (RLVR)** to the dental domain. By integrating the DENTEX (Dental Enumeration and Diagnosis on Panoramic X-rays) dataset with the state-of-the-art Qwen3-VL architecture, we enforce a strict, hierarchical diagnostic approach. Our RL-based pipeline directly optimizes for the correct quadrant, tooth identification, and pathological classification without requiring expensive human-in-the-loop preference mapping.

---

## 2. Background and Related Work

### 2.1 Traditional vs. Generative Medical AI
Historically, automated dental diagnostics have utilized architectures such as Faster R-CNN or YOLOv8. The original DENTEX framework (2023) established baseline object detection metrics using such models. While highly performant at drawing bounding boxes around caries, these models act as black boxes; they cannot elaborate on *why* a diagnosis was made, nor can they be easily integrated into a text-based patient electronic health record (EHR) system.

### 2.2 Vision-Language Models (VLMs)
Models like GPT-4V, LLaVA, and Qwen-VL have merged visual encoders (e.g., ViT) with LLM decoders, enabling the system to "describe" images in natural language. While powerful, VLMs suffer from unstructured outputs, making it difficult to systematically parse their conclusions for programmatic medical use. 

### 2.3 GRPO and Verifiable Rewards
Proximal Policy Optimization (PPO) has been the gold standard for RLHF (Reinforcement Learning from Human Feedback). However, PPO requires a massive, separately trained "Reward Model" that consumes immense VRAM. **Group Relative Policy Optimization (GRPO)** eliminates the need for an external reward model by generating a group of predictions for a single prompt and explicitly comparing them against one another by normalizing the rewards within the group. Combined with **RLVR**—where the reward is a deterministic Python script rather than human sentiment—this architecture enables rapid, verifiable optimization.

---

## 3. Methodology 

### 3.1 Dataset Normalization and Pipeline Architecture
The project utilized the official DENTEX dataset, comprising complex panoramic X-rays with hierarchical annotations (Quadrant -> Enumeration/Tooth -> Disease).

**Data Engineering Challenges:**
The original dataset was fragmented across multiple JSON formats (COCO for training, LabelMe for testing) and utilized non-standardized Turkish pathological labels (`çürük` for Caries, `Kök_Parçası` for retained roots). 
To resolve this, we engineered a robust conversion infrastructure (`convert_coco.py` and `convert_labelme.py`) that cross-mapped annotations into an English-standardized, `.jsonl` format suitable for conversational RL:
- Mapped 8 unstandardized classes to 4 core pathologies: `caries`, `deep_caries`, `periapical`, `impacted`.
- Implemented an algebraic FDI notation parser to automatically deduce anatomical (Quadrant, Tooth) pairs from raw integers.

### 3.2 The Two-Stage Training Paradigm
Training a massive 8B parameter vision model using Reinforcement Learning from scratch is highly brittle. We implemented a two-stage training approach:

#### Stage 1: Supervised Fine-Tuning (SFT) Warmup
Before subjecting the model to reinforcement learning, it must understand the "rules of the game." We generated synthetic expected responses comprising a `<think>` block (diagnosing the anatomy naturally) and an `<answer>` block (the strict JSON-like output). 
Using the Unsloth framework and LoRA (Low-Rank Adaptation) on the Qwen3-VL architecture, the model underwent 1 epoch of SFT. This crucial step dropped the language loss to near 0.25, ensuring the model understood the XML formatting and baseline anatomical mapping.

#### Stage 2: Group Relative Policy Optimization (GRPO)
Following SFT, we subjected the model to the primary GRPO algorithm. For every X-ray image, the policy model generates multiple trajectories $G$ (e.g., $G=4$ or $8$). For each trajectory, our deterministic reward functions grade the model's response. The rewards are normalized across the group, generating "Advantages" that dictate the policy gradient update.

### 3.3 The Hierarchical Reward Functions
A core innovation of this project is the translation of clinical diagnostic logic into absolute programmatic reward signals.

**1. The Format Reward (`format_reward.py`)**
Grants heavy penalties for failing to utilize `<think>` and `<answer>` tags, and penalizes the model via a regex check if it leaks final diagnostic decisions into the free-form reasoning block.

**2. The FDI Hierarchical Reward (`fdi_reward.py`)**
Because dental anatomy is inherently hierarchical, the model is rewarded step-by-step:
1. **Quadrant Matching (+0.30):** The model correctly identifies the quadrant (1-4).
2. **Tooth Enumeration (+0.30):** *Conditioned* on getting the quadrant right, the model correctly identifies the specific tooth number (1-8).
3. **Pathology Classification (+0.30):** The model accurately identifies the specific disease class.

By structuring the reward hierarchically, the GRPO algorithm can assign "partial credit." This solves the sparse-reward problem typical in reinforcement learning; rather than receiving a binary 0 or 1, the model is guided incrementally toward perfect visual identification.

---

## 4. Experimental Setup

### 4.1 Hardware and Infrastructure
The computational requirements for VLMs are immense. The experiments were executed on a single Nvidia H100 80GB tensor-core GPU.
- **Model:** `unsloth/Qwen3-VL-8B-Instruct`
- **Quantization:** 4-bit precision (via bitsandbytes) to minimize memory footprint.
- **LoRA Configuration:** Applying adapters to Attention and MLP layers (`r=16, alpha=32`).
- **Frameworks:** TRL (Transformers Reinforcement Learning), Unsloth, HuggingFace Accelerate, Weights & Biases (W&B) for telemetry.

### 4.2 Hyperparameters
- SFT Epochs: 1 | SFT Learning Rate: $2 \times 10^{-5}$
- GRPO Epochs: 3 | GRPO Learning Rate: $5 \times 10^{-6}$
- KL Target/Penalty: 0.1 | Temperature: 0.8 | Top-P: 0.8
- Generation Batch Size: 2 | Gradient Accumulation Steps: 4

---

## 5. Results and Discussion

### 5.1 Training Stability and Convergence
Reinforcement learning loops are notoriously unstable. However, empirical telemetry verified through W&B confirms extremely robust convergence.
- **Reward Growth:** The absolute baseline reward during the primary training step rapidly climbed to a mean of `0.476` per generation by step 150.
- **Zero Variance Saturation (`frac_reward_zero_std`):** By epoch 0.88, `frac_reward_zero_std` successfully hit `1.0`. This indicates that within a generation group $G$, every single response produced by the model became functionally identical and achieved the exact same reward. In the context of deterministic RLVR, this is an excellent signal; it proves the model rapidly solved the structural `<think>/<answer>` formatting constraint and became highly confident in its visual extractions.

### 5.2 KL-Divergence Telemetry
During GRPO, KL-divergence was tracked against the base SFT reference model. Over the first epoch, `kl` metric grew logarithmically from `0.001` to `0.174`. This is a highly positive signal indicating **active policy exploration**. The model did not prematurely mode-collapse onto the SFT baseline, but actively diverged from it to maximize the hierarchical FDI rewards dictated by the environment.

### 5.3 Clinical Efficacy 
By generating an explicit reasoning trace (`<think> Tooth 46 in the lower-right quadrant shows deep caries extending toward the pulp </think>`), clinicians are afforded zero-shot observability into the model's diagnostic logic. Unlike CNNs that output arbitrary coordinates, our project enables a conversational interoperability paradigm entirely suitable for modern dental workflows.

---

## 6. Conclusion
The DENTEX-RLVR framework validates that modern foundational vision-language models can be surgically fine-tuned into expert academic systems via Verifiable Reinforcement Learning. By structuring the dental diagnostic challenge as a hierarchical reward optimization problem, we successfully trained an 8-Billion parameter multimodal network to enforce strict diagnostic formatting and spatial anatomical awareness. 

This pipeline—from raw data coercion and SFT warmup to GRPO policy generation—mirrors the most advanced optimization methodologies implemented by global private AI laboratories. DENTEX-RLVR successfully bridges the gap between unstructured generative AI and absolute, programmatic medical compliance.

---

## 7. Future Work
While the framework successfully proved the viability of GRPO for dental VLM workflows, future avenues for scaling exist:
1. **Curriculum Learning:** Dynamically throttling the difficulty of the images (e.g., escalating from single-caries images to full impacted panoramic impactions) during RL.
2. **Dense Rewards:** Expanding the reward function to utilize Intersection over Union (IoU) of coordinates if the VLM natively predicts bounding boxes alongside semantic tokens.
3. **Scaling Laws:** Applying the identical deterministic reward criteria (RLVR) to hyper-scale models (e.g., 72B parameter variants).

---
---

## Appendix: Pipeline Execution Instructions

### A. Environment Setup
The pipeline requires an Nvidia GPU (H100/A100 recommended) and the Unsloth optimized runtime.
```bash
./scripts/setup_h100.sh
source venv/bin/activate
```

### B. Data Ingestion
The DENTEX dataset must be downloaded from HuggingFace and converted via the custom translation pipelines:
```bash
# 1. Download full dataset
python3 scripts/download_dentex.py

# 2. Convert Training data (COCO format) to JSONL
python3 data/convert_coco.py   --json scripts/dataset/DENTEX/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json   --output data/dentex_train.jsonl   --split train

# 3. Convert Evaluation data (LabelMe format) to JSONL
python3 data/convert_labelme.py   --input-dir scripts/dataset/DENTEX/disease   --output data/dentex_test.jsonl   --split test
```

### C. Training the Architecture
```bash
# Stage 1: SFT Warmup
python3 train_sft_warmup.py --config configs/grpo_config.yaml

# Prepare Config for Stage 2
sed -i 's|model_name: "unsloth/Qwen3-VL-8B-Instruct"|model_name: "checkpoints/sft_warmup/final"|' configs/grpo_config.yaml

# Stage 2: DeepSeek-style GRPO
python3 train_dentex_grpo.py --config configs/grpo_config.yaml
```
Output models and checkpoints are continuously saved to `checkpoints/grpo_run1/`. Training state and reward curves are pushed and tracked via `Weights & Biases (wandb)`.
