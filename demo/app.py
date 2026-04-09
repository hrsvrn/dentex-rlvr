"""Gradio demo for DENTEX dental diagnosis — HF Spaces deployable."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = "You are a dental radiologist. Analyze the panoramic X-ray."

USER_PROMPT = (
    "Identify all abnormal teeth. For each, output FDI quadrant (1–4), "
    "tooth number (1–8), and diagnosis (caries/deep_caries/periapical/impacted). "
    "Think step by step inside <think> tags. Output answer inside <answer> tags. "
    "Format: <answer>Q{q}T{t}:{diag}, Q{q}T{t}:{diag}, ...</answer>"
)

_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Globals set on startup
model = None
processor = None


def load_model(model_path: str) -> None:
    """Load model and processor into global state."""
    global model, processor
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()


def parse_output(raw: str) -> tuple[str, str]:
    """Parse model output into reasoning and findings."""
    think_match = _THINK_PATTERN.search(raw)
    answer_match = _ANSWER_PATTERN.search(raw)

    reasoning = think_match.group(1).strip() if think_match else ""
    findings = answer_match.group(1).strip() if answer_match else raw.strip()

    return reasoning, findings


def format_findings_table(findings_str: str) -> str:
    """Convert findings string to markdown table."""
    pattern = re.compile(r"Q([1-4])T([1-8]):(caries|deep_caries|periapical|impacted)")
    matches = pattern.findall(findings_str)

    if not matches:
        return "*No valid findings detected.*"

    quadrant_names = {
        "1": "Upper Right",
        "2": "Upper Left",
        "3": "Lower Left",
        "4": "Lower Right",
    }

    rows = []
    for q, t, diag in matches:
        rows.append(
            f"| {quadrant_names[q]} (Q{q}) | {t} | {diag.replace('_', ' ').title()} |"
        )

    header = "| Quadrant | Tooth # | Diagnosis |\n|---|---|---|"
    return header + "\n" + "\n".join(rows)


def predict(image) -> tuple[str, str, str]:
    """Run inference on uploaded X-ray image.

    Returns:
        (raw_output, reasoning, findings_table)
    """
    if model is None or processor is None:
        return "Model not loaded.", "", ""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    raw_output = processor.decode(
        generated[0][input_len:], skip_special_tokens=True
    )

    reasoning, findings = parse_output(raw_output)
    findings_table = format_findings_table(findings)

    return raw_output, reasoning, findings_table


def build_app() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="DENTEX Dental Diagnosis") as app:
        gr.Markdown(
            "# DENTEX Dental Diagnosis\n"
            "Upload a panoramic dental X-ray for AI-powered diagnosis. "
            "The model identifies abnormal teeth using FDI notation."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Panoramic X-ray")
                submit_btn = gr.Button("Analyze", variant="primary")

            with gr.Column():
                findings_output = gr.Markdown(label="Findings")
                reasoning_output = gr.Textbox(
                    label="Reasoning",
                    lines=8,
                    interactive=False,
                )
                raw_output = gr.Textbox(
                    label="Raw Model Output",
                    lines=4,
                    interactive=False,
                    visible=False,
                )

        with gr.Accordion("Show raw output", open=False):
            raw_visible = gr.Textbox(
                label="Raw Output", lines=6, interactive=False
            )

        submit_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[raw_visible, reasoning_output, findings_output],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="DENTEX Gradio Demo")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-7B-Instruct",
        help="Model name or path (e.g. checkpoints/grpo_run1/final)",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    load_model(args.model)
    app = build_app()
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
