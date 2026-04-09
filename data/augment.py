"""Augmentations for DENTEX panoramic X-rays: horizontal flip + gamma jitter."""

import random
from typing import Any

import numpy as np
from PIL import Image


def horizontal_flip(
    image: Image.Image,
    findings: list[dict[str, Any]],
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Flip image horizontally and mirror FDI quadrants.

    Q1 (upper-right) <-> Q2 (upper-left)
    Q3 (lower-left)  <-> Q4 (lower-right)
    """
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    quadrant_mirror = {1: 2, 2: 1, 3: 4, 4: 3}
    flipped_findings = []
    for f in findings:
        flipped_findings.append({
            "quadrant": quadrant_mirror[f["quadrant"]],
            "tooth": f["tooth"],
            "diagnosis": f["diagnosis"],
        })

    return flipped_image, flipped_findings


def gamma_jitter(
    image: Image.Image,
    gamma_range: tuple[float, float] = (0.7, 1.4),
) -> Image.Image:
    """Apply random gamma correction to simulate exposure variation."""
    gamma = random.uniform(*gamma_range)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.power(arr, gamma)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def augment_sample(
    image: Image.Image,
    findings: list[dict[str, Any]],
    flip_prob: float = 0.5,
    gamma_range: tuple[float, float] = (0.7, 1.4),
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Apply random augmentations to an image-findings pair.

    Args:
        image: Input PIL image.
        findings: List of finding dicts with quadrant/tooth/diagnosis.
        flip_prob: Probability of horizontal flip.
        gamma_range: Range for gamma jitter.

    Returns:
        Augmented (image, findings) tuple.
    """
    if random.random() < flip_prob:
        image, findings = horizontal_flip(image, findings)

    image = gamma_jitter(image, gamma_range)

    return image, findings
