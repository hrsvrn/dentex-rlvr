import os
import torch
import gradio as gr
from PIL import Image
from unsloth import FastVisionModel

# Global model variables to avoid reloading
MODEL = None
PROCESSOR = None

def load_model():
    """Loads the model and processor globally on first launch."""
    global MODEL, PROCESSOR
    if MODEL is not None:
        return
        
    print("⏳ Loading Qwen3-VL base model and 168MB RLVR adapter from HuggingFace...")
    model_id = "hrsvrn/Qwen3-VL-8B-dentex-rlvr-grpo"
    
    MODEL, PROCESSOR = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=True,
        use_gradient_checkpointing=False,
    )
    FastVisionModel.for_inference(MODEL)
    print("✅ Model loaded successfully in 4-bit precision!")

def analyze_xray(image):
    """Gradio inference function."""
    if image is None:
        return "Please upload an X-ray image."
        
    try:
        load_model()
    except Exception as e:
        return f"Failed to load model from HuggingFace: {str(e)}"
        
    image = image.convert("RGB")
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert dental diagnostic AI. Analyze the image and output findings exactly as requested. First think through the anatomy, then provide the final answer."}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Identify the quadrant, tooth, and pathology for all issues."}
        ]}
    ]

    input_text = PROCESSOR.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = PROCESSOR(
        text=[input_text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(MODEL.device)

    with torch.no_grad():
        generated_ids = MODEL.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.3, # Low temperature for clinical determinism
            top_p=0.9,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = PROCESSOR.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text

# Gradio Interface Definition
with gr.Blocks(theme=gr.themes.Soft(), title="DENTEX-RLVR Diagnostic AI") as demo:
    gr.Markdown("# 🦷 DENTEX-RLVR: Automated Panoramic X-Ray Diagnostics")
    gr.Markdown("Upload a panoramic dental X-ray. The VLM (Qwen3-VL-8B) will automatically download its **RLVR GRPO LoRA weights (~168MB)** from HuggingFace, analyze the image, formulate a reasoning trace (`<think>`), and output standard FDI diagnostic JSON blocks (`<answer>`).")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Panoramic X-Ray")
            analyze_btn = gr.Button("🔍 Analyze Radiograph", variant="primary")
            
        with gr.Column():
            output_box = gr.Markdown(label="Diagnostic Report")
            
    analyze_btn.click(fn=analyze_xray, inputs=input_image, outputs=output_box)
    
    gr.Examples(
        examples=[
            # Optional: Add local paths to example images if testing UI
        ],
        inputs=input_image
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
