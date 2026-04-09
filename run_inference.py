import os
import argparse
from pathlib import Path
from PIL import Image

import torch
import unsloth
from unsloth import FastVisionModel

def main():
    parser = argparse.ArgumentParser(description="Run DENTEX-RLVR inference on a single X-ray image")
    parser.add_argument("--image", required=True, help="Path to the panoramic X-ray image")
    parser.add_argument("--model", default="checkpoints/grpo_run1/final", help="Path to the trained LoRA adapter")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    args = parser.parse_args()

    # Load Model and Processor
    print(f"Loading model from {args.model} in 4-bit precision...")
    model, processor = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=True,
        use_gradient_checkpointing=False, # We are inferencing, not training
    )
    
    # Must switch model to inference mode
    FastVisionModel.for_inference(model)

    # Validate image
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found!")
        return
        
    image = Image.open(args.image).convert("RGB")
    print(f"Loaded image: {args.image} ({image.size[0]}x{image.size[1]})")

    # The exact system and user prompt used during training
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert dental diagnostic AI. Analyze the image and output findings exactly as requested. First think through the anatomy, then provide the final answer."}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Identify the quadrant, tooth, and pathology for all issues."}
        ]}
    ]

    # Process inputs for Qwen3-VL
    print("Processing inputs...")
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(
        text=[input_text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate output
    print("Generating diagnostic report (this may take a moment)...\n")
    print("-" * 50)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            use_cache=True,
            temperature=0.7, # Slight temperature is fine, though RL models are often evaluated at 0.0 or 0.7
            top_p=0.9,
        )

    # Decode the newly generated tokens (slice off the input prompt)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print(output_text)
    print("-" * 50)

if __name__ == "__main__":
    main()
