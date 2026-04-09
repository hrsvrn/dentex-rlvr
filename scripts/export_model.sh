#!/bin/bash
# Export LoRA adapter to GGUF or AWQ format via Unsloth
set -euo pipefail

FORMAT="${1:?Usage: export_model.sh <format> <adapter_path> [output_dir]}"
ADAPTER_PATH="${2:?Usage: export_model.sh <format> <adapter_path> [output_dir]}"
OUTPUT_DIR="${3:-exports}"

echo "=== DENTEX RLVR — Model Export ==="
echo "Format: $FORMAT"
echo "Adapter: $ADAPTER_PATH"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

case "$FORMAT" in
    gguf)
        echo "Exporting to GGUF (Q4_K_M)..."
        python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='${ADAPTER_PATH}',
    max_seq_length=2048,
    load_in_4bit=True,
)
model.save_pretrained_gguf(
    '${OUTPUT_DIR}/dentex-rlvr-gguf',
    tokenizer,
    quantization_method='q4_k_m',
)
print('GGUF export complete.')
"
        ;;
    awq)
        echo "Exporting to AWQ..."
        python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='${ADAPTER_PATH}',
    max_seq_length=2048,
    load_in_4bit=True,
)
model.save_pretrained_merged(
    '${OUTPUT_DIR}/dentex-rlvr-awq',
    tokenizer,
    save_method='merged_16bit',
)
print('AWQ export complete.')
"
        ;;
    lora)
        echo "Saving LoRA adapter only..."
        cp -r "$ADAPTER_PATH" "$OUTPUT_DIR/dentex-rlvr-lora"
        echo "LoRA adapter saved."
        ;;
    *)
        echo "Unknown format: $FORMAT"
        echo "Supported: gguf, awq, lora"
        exit 1
        ;;
esac

echo ""
echo "=== Export Complete ==="
ls -lh "$OUTPUT_DIR/"
