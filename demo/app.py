"""Gradio demo for DENTEX dental diagnosis using Unsloth + LoRA adapters."""

from __future__ import annotations

import re

import gradio as gr
import torch
from unsloth import FastVisionModel

DEFAULT_MODEL = "hrsvrn/Qwen3-VL-8B-dentex-rlvr-grpo"

# Must match the prompts used during training (data/convert_dentex.py)
SYSTEM_PROMPT = "You are a dental radiologist. Analyze the panoramic X-ray."

USER_PROMPT = (
    "Identify all abnormal teeth. For each, output FDI quadrant (1\u20134), "
    "tooth number (1\u20138), and diagnosis (caries/deep_caries/periapical/impacted). "
    "Think step by step inside <think> tags. Output answer inside <answer> tags. "
    "Format: <answer>Q{q}T{t}:{diag}, Q{q}T{t}:{diag}, ...</answer>"
)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_FINDING_RE = re.compile(r"Q([1-4])T([1-8]):(caries|deep_caries|periapical|impacted)")

QUADRANT_NAMES = {
    "1": "Upper Right",
    "2": "Upper Left",
    "3": "Lower Left",
    "4": "Lower Right",
}

model = None
processor = None


def load_model(model_id: str = DEFAULT_MODEL) -> None:
    global model, processor
    print(f"Loading LoRA adapter from: {model_id}")
    model, processor = FastVisionModel.from_pretrained(
        model_id,
        load_in_4bit=True,
        use_gradient_checkpointing=False,
    )
    FastVisionModel.for_inference(model)
    print("Model loaded.")


def parse_output(raw: str) -> tuple[str, str]:
    think = _THINK_RE.search(raw)
    answer = _ANSWER_RE.search(raw)
    reasoning = think.group(1).strip() if think else ""
    findings = answer.group(1).strip() if answer else raw.strip()
    return reasoning, findings


def findings_to_table(findings: str) -> str:
    matches = _FINDING_RE.findall(findings)
    if not matches:
        return "*No structured findings detected.*"
    header = "| Quadrant | Tooth | Diagnosis |\n|---|---|---|"
    rows = [
        f"| {QUADRANT_NAMES[q]} (Q{q}) | {t} | {d.replace('_', ' ').title()} |"
        for q, t, d in matches
    ]
    return header + "\n" + "\n".join(rows)


def predict(image, max_tokens: int, temperature: float):
    if model is None or processor is None:
        return "Model not loaded.", "", ""

    print(f"[predict] Received image: {image.size}, max_tokens={max_tokens}, temperature={temperature}")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    input_text = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    inputs = processor(
        text=[input_text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print(f"[predict] Input shape: {inputs['input_ids'].shape}")

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        use_cache=True,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(new_ids, skip_special_tokens=True)

    print(f"[predict] Generated {len(new_ids)} tokens")
    print(f"[predict] Raw output:\n{raw}")
    print("-" * 60)

    reasoning, findings = parse_output(raw)
    table = findings_to_table(findings)
    return table, reasoning, raw


def build_app() -> gr.Blocks:
    with gr.Blocks(title="DENTEX Dental Diagnosis") as app:
        gr.Markdown(
            "# DENTEX Dental Diagnosis\n"
            "Upload a panoramic dental X-ray for AI-powered diagnosis. "
            "The model identifies abnormal teeth using FDI notation.\n\n"
            "**Model:** `hrsvrn/Qwen3-VL-8B-dentex-rlvr-grpo` (Unsloth 4-bit)"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Panoramic X-ray")
                with gr.Row():
                    max_tokens = gr.Slider(
                        64, 1024, value=512, step=64, label="Max tokens"
                    )
                    temperature = gr.Slider(
                        0.0, 1.0, value=0.3, step=0.1, label="Temperature"
                    )
                submit_btn = gr.Button("Analyze", variant="primary")

            with gr.Column(scale=1):
                findings_output = gr.Markdown(label="Findings")
                reasoning_output = gr.Markdown(label="Reasoning")

        with gr.Accordion("Raw model output", open=False):
            raw_output = gr.Textbox(label="Raw", lines=8, interactive=False)

        submit_btn.click(
            fn=predict,
            inputs=[image_input, max_tokens, temperature],
            outputs=[findings_output, reasoning_output, raw_output],
        )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DENTEX Gradio Demo")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    load_model(args.model)
    app = build_app()
    app.launch(server_port=args.port, share=args.share)
