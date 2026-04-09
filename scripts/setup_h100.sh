#!/bin/bash
# H100 node setup — install dependencies and configure environment
set -euo pipefail

echo "=== DENTEX RLVR — H100 Setup ==="

# System packages
echo "Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git wget > /dev/null

# Python packages
echo "Installing Python dependencies..."
pip install --upgrade pip > /dev/null

# Core ML stack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Unsloth (4-bit QLoRA)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# TRL + transformers
pip install trl>=0.12.0 transformers>=4.46.0 accelerate>=0.34.0

# Dataset and evaluation
pip install datasets peft bitsandbytes

# Qwen-VL dependencies
pip install qwen-vl-utils

# Training utilities
pip install wandb gymnasium pyyaml

# Inference and export
pip install vllm gradio>=4.0

# Image processing
pip install Pillow numpy

echo ""
echo "=== Verifying GPU ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. wandb login"
echo "  2. python data/convert_dentex.py --output data/dentex_train.jsonl"
echo "  3. python eval/baseline_eval.py --model Qwen/Qwen3-VL-7B-Instruct"
echo "  4. python train_dentex_grpo.py --config configs/grpo_config.yaml"
