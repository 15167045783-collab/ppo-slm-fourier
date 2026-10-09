from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PPOConfig:
    phase_size: int = 8
    image_size: int = 256
    action_scale: float = math.pi / 4.0
    total_steps: int = 20_000
    rollout_steps: int = 256
    episode_length: int = 256
    target_mse: float = 0.005
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 64
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    improvement_gain: float = 100.0
    mse_weight: float = 1.0
    action_penalty: float = 0.001
    seed: int = 7
    device: str = "auto"

    def __post_init__(self) -> None:
        positive_integers = {
            "phase_size": self.phase_size,
            "image_size": self.image_size,
            "total_steps": self.total_steps,
            "rollout_steps": self.rollout_steps,
            "episode_length": self.episode_length,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rollout_steps > self.total_steps:
            raise ValueError("rollout_steps cannot exceed total_steps")
        if self.minibatch_size > self.rollout_steps:
            raise ValueError("minibatch_size cannot exceed rollout_steps")
        if not 0.0 < self.target_mse < 1.0:
            raise ValueError("target_mse must be between 0 and 1")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must be in (0, 1)")
        finite_values = (
            self.action_scale,
            self.learning_rate,
            self.value_coefficient,
            self.entropy_coefficient,
            self.max_grad_norm,
            self.improvement_gain,
            self.mse_weight,
            self.action_penalty,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("all floating-point configuration values must be finite")
        if any(value < 0 for value in finite_values):
            raise ValueError("configuration weights and scales cannot be negative")
        for name, value in (
            ("action_scale", self.action_scale),
            ("learning_rate", self.learning_rate),
            ("max_grad_norm", self.max_grad_norm),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

    @property
    def action_size(self) -> int:
        return self.phase_size**2

    @classmethod
    def quick(cls) -> "PPOConfig":
        return replace(
            cls(),
            total_steps=16,
            rollout_steps=8,
            episode_length=8,
            update_epochs=1,
            minibatch_size=4,
        )


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def select_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
