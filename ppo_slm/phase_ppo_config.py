from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PhasePPOConfig:
    phase_size: int = 64
    image_size: int = 256
    rounds: int = 500
    population_size: int = 64
    sigma: float = 0.04
    clip_ratio: float = 0.02
    update_epochs: int = 4
    minibatch_size: int = 16
    learning_rate: float = 0.3
    max_grad_norm: float = 1.0
    foreground_weight: float = 8.0
    mse_coefficient: float = 0.60
    ncc_coefficient: float = 0.30
    outside_coefficient: float = 0.10
    target_threshold: float = 0.05
    seed: int = 7
    device: str = "auto"

    def __post_init__(self) -> None:
        positive_integers = {
            "phase_size": self.phase_size,
            "image_size": self.image_size,
            "rounds": self.rounds,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.minibatch_size > self.population_size:
            raise ValueError("minibatch_size cannot exceed population_size")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must be between 0 and 1")
        if not 0.0 < self.target_threshold < 1.0:
            raise ValueError("target_threshold must be between 0 and 1")

        positive_floats = {
            "sigma": self.sigma,
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
        }
        for name, value in positive_floats.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        objective_values = {
            "foreground_weight": self.foreground_weight,
            "mse_coefficient": self.mse_coefficient,
            "ncc_coefficient": self.ncc_coefficient,
            "outside_coefficient": self.outside_coefficient,
        }
        if not all(math.isfinite(value) for value in objective_values.values()):
            raise ValueError("objective values must be finite")
        if any(value < 0.0 for value in objective_values.values()):
            raise ValueError("objective values cannot be negative")
        if (
            self.mse_coefficient
            + self.ncc_coefficient
            + self.outside_coefficient
            <= 0.0
        ):
            raise ValueError("at least one objective coefficient must be positive")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")

    @property
    def action_size(self) -> int:
        return self.phase_size**2

    @classmethod
    def quick(cls) -> "PhasePPOConfig":
        return replace(
            cls(),
            rounds=2,
            population_size=8,
            update_epochs=1,
            minibatch_size=4,
        )
