import os
import argparse
from PIL import Image
import torch
from unsloth import FastVisionModel

def main():
    parser = argparse.ArgumentParser(description="Run DENTEX-RLVR inference pulling directly from HuggingFace")
    parser.add_argument("--image", required=True, help="Path to the panoramic X-ray image")
    parser.add_argument("--model", default="hrsvrn/Qwen3-VL-8B-dentex-rlvr-grpo", 
                        help="HuggingFace Hub Repo ID (this pulls the ~168MB LoRA adapters)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    args = parser.parse_args()

    print(f"📡 Fetching adapters from HuggingFace Hub: {args.model}...")
    print("   (This will download ~168MB of LoRA weights and bind them to the base Qwen3-VL model in 4-bit precision)\n")
    
    model, processor = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=True,
        use_gradient_checkpointing=False, 
    )
    
    FastVisionModel.for_inference(model)

    if not os.path.exists(args.image):
        print(f"❌ Error: Image {args.image} not found!")
        return
        
    image = Image.open(args.image).convert("RGB")
    print(f"📸 Loaded image: {args.image} ({image.size[0]}x{image.size[1]})")

    # Must match the prompts used during training (data/convert_dentex.py)
    system_prompt = "You are a dental radiologist. Analyze the panoramic X-ray."
    user_prompt = (
        "Identify all abnormal teeth. For each, output FDI quadrant (1\u20134), "
        "tooth number (1\u20138), and diagnosis (caries/deep_caries/periapical/impacted). "
        "Think step by step inside <think> tags. Output answer inside <answer> tags. "
        "Format: <answer>Q{q}T{t}:{diag}, Q{q}T{t}:{diag}, ...</answer>"
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt},
        ]}
    ]

    print("⚙️ Processing multimodal inputs...")
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = processor(
        text=[input_text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    print("🧠 Generating diagnostic report (RLVR inference pending)...\n")
    print("=" * 60)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            use_cache=True,
            temperature=0.7, 
            top_p=0.9,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print(output_text)
    print("=" * 60)

if __name__ == "__main__":
    main()
