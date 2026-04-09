"""GymDentalEnv — Gymnasium wrapper over DENTEX for GRPO rollouts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rewards.format_reward import format_reward
from rewards.fdi_reward import fdi_reward


class GymDentalEnv(gym.Env):
    """Gymnasium environment wrapping DENTEX annotations for GRPO training.

    Each episode presents a dental X-ray prompt and scores the model's
    text output against ground-truth FDI annotations.

    Observation: dict with "messages" (conversation so far) and "ground_truth".
    Action: model's text output (str).
    Reward: format_reward + fdi_reward (max 1.0).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_path: str | Path,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.data_path = Path(data_path)
        self.shuffle = shuffle
        self.samples = self._load_samples()
        self._index = 0

        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        if self.shuffle:
            self._rng.shuffle(self.samples)

        # Observation and action spaces (text-based, so we use generic spaces)
        self.observation_space = spaces.Dict({
            "messages": spaces.Sequence(spaces.Text(max_length=2048)),
            "image_id": spaces.Text(max_length=256),
        })
        self.action_space = spaces.Text(max_length=4096)

        self._current_sample: dict | None = None

    def _load_samples(self) -> list[dict]:
        """Load JSONL samples from data file."""
        samples = []
        with self.data_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        if not samples:
            raise ValueError(f"No samples found in {self.data_path}")
        return samples

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict]:
        """Reset environment to next sample.

        Returns:
            (observation, info) tuple.
        """
        super().reset(seed=seed)

        if self._index >= len(self.samples):
            self._index = 0
            if self.shuffle:
                self._rng.shuffle(self.samples)

        self._current_sample = self.samples[self._index]
        self._index += 1

        obs = {
            "messages": self._current_sample["messages"],
            "image_id": self._current_sample.get("image_id", ""),
        }
        info = {
            "ground_truth": self._current_sample["ground_truth"],
            "sample_index": self._index - 1,
        }

        return obs, info

    def step(self, action: str) -> tuple[dict, float, bool, bool, dict]:
        """Score the model's text output.

        Args:
            action: Model's generated text output.

        Returns:
            (observation, reward, terminated, truncated, info) tuple.
            Episode always terminates after one step.
        """
        if self._current_sample is None:
            raise RuntimeError("Must call reset() before step()")

        gt = self._current_sample["ground_truth"]

        # Compute reward components
        r_format = format_reward(action, gt)
        r_fdi = fdi_reward(action, gt)
        total_reward = r_format + r_fdi

        info = {
            "reward_format": r_format,
            "reward_fdi": r_fdi,
            "reward_total": total_reward,
            "ground_truth": gt,
            "model_output": action,
        }

        obs = {
            "messages": self._current_sample["messages"],
            "image_id": self._current_sample.get("image_id", ""),
        }

        # Single-step episode: always terminates
        return obs, total_reward, True, False, info

    def get_reward_breakdown(self, action: str, ground_truth: dict) -> dict[str, float]:
        """Get detailed reward breakdown for logging."""
        return {
            "format": format_reward(action, ground_truth),
            "fdi": fdi_reward(action, ground_truth),
        }
