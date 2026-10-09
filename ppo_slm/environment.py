from __future__ import annotations

import math

import torch

from .config import PPOConfig
from .optics import fourier_intensity, upsample_phase


class SLMEnvironment:
    def __init__(
        self,
        target: torch.Tensor,
        config: PPOConfig,
        device: torch.device,
    ) -> None:
        expected = (1, config.image_size, config.image_size)
        if tuple(target.shape) != expected:
            raise ValueError(f"target must have shape {expected}")
        if not target.is_floating_point():
            raise ValueError("target must be floating point")
        if not torch.isfinite(target).all():
            raise ValueError("target contains NaN or Inf")
        if float(target.min()) < 0.0 or float(target.max()) > 1.0:
            raise ValueError("target values must be in [0, 1]")
        self.target = target.detach().to(device=device, dtype=torch.float32).clone()
        self.config = config
        self.device = device
        self._phase = torch.empty(
            (config.phase_size, config.phase_size), device=device, dtype=torch.float32
        )
        self._camera = torch.empty_like(self.target)
        self._mse = float("nan")
        self._steps = 0
        self._reset_count = 0
        self._reached_target = False

    @property
    def phase(self) -> torch.Tensor:
        return self._phase.detach().clone()

    @property
    def camera(self) -> torch.Tensor:
        return self._camera.detach().clone()

    @property
    def mse(self) -> float:
        return self._mse

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def reached_target(self) -> bool:
        return self._reached_target

    def _capture(self) -> torch.Tensor:
        frame = fourier_intensity(self._phase, self.config.image_size)
        expected = (1, self.config.image_size, self.config.image_size)
        if tuple(frame.shape) != expected:
            raise ValueError(f"camera frame must have shape {expected}")
        if not frame.is_floating_point() or not torch.isfinite(frame).all():
            raise ValueError("camera frame must be finite and floating point")
        frame = frame.detach().to(device=self.device, dtype=torch.float32).clone()
        if float(frame.min()) < 0.0 or float(frame.max()) > 1.0:
            raise ValueError("camera frame values must be in [0, 1]")
        return frame

    def _weighted_mse(self, frame: torch.Tensor) -> float:
        weights = 1.0 + self.config.foreground_weight * self.target
        squared_error = (frame - self.target).square()
        weighted_error = (weights * squared_error).sum() / weights.sum()
        return float(weighted_error.item())

    def _observation(self) -> torch.Tensor:
        phase_channel = upsample_phase(
            self._phase / (2 * math.pi), self.config.image_size
        ).unsqueeze(0)
        observation = torch.cat((self.target, self._camera, phase_channel), dim=0)
        if not torch.isfinite(observation).all():
            raise ValueError("observation contains NaN or Inf")
        return observation.detach().to(dtype=torch.float32).clone()

    def reset(self, seed: int | None = None) -> torch.Tensor:
        actual_seed = self.config.seed + self._reset_count if seed is None else seed
        generator = torch.Generator(device="cpu").manual_seed(actual_seed)
        phase = torch.rand(
            (self.config.phase_size, self.config.phase_size),
            generator=generator,
            dtype=torch.float32,
        ) * (2 * math.pi)
        self._phase = phase.to(self.device)
        self._camera = self._capture()
        self._mse = self._weighted_mse(self._camera)
        if not math.isfinite(self._mse):
            raise ValueError("initial MSE is not finite")
        self._steps = 0
        self._reached_target = self._mse <= self.config.target_mse
        self._reset_count += 1
        return self._observation()

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, float, bool, dict[str, float | bool]]:
        if tuple(action.shape) != (self.config.action_size,):
            raise ValueError(f"action must have shape ({self.config.action_size},)")
        if not action.is_floating_point() or not torch.isfinite(action).all():
            raise ValueError("action must be finite and floating point")
        bounded = torch.tanh(action.detach().to(self.device, dtype=torch.float32))
        bounded = bounded.reshape(self.config.phase_size, self.config.phase_size)
        self._phase = torch.remainder(
            self._phase + self.config.action_scale * bounded,
            2 * math.pi,
        )
        previous_mse = self._mse
        self._camera = self._capture()
        current_mse = self._weighted_mse(self._camera)
        improvement = previous_mse - current_mse
        action_magnitude = float(bounded.square().mean().item())
        action_cost = self.config.action_penalty * action_magnitude
        reward = (
            self.config.improvement_gain * improvement
            - self.config.mse_weight * current_mse
            - action_cost
        )
        psnr = -10.0 * math.log10(max(current_mse, 1e-12))
        if not all(
            math.isfinite(value)
            for value in (current_mse, improvement, action_magnitude, reward, psnr)
        ):
            raise ValueError("environment produced a non-finite metric")
        self._mse = current_mse
        self._steps += 1
        self._reached_target = current_mse <= self.config.target_mse
        done = self._reached_target or self._steps >= self.config.episode_length
        info: dict[str, float | bool] = {
            "mse": current_mse,
            "previous_mse": previous_mse,
            "improvement": improvement,
            "action_magnitude": action_magnitude,
            "action_penalty": action_cost,
            "psnr": psnr,
            "reached_target": self._reached_target,
        }
        return self._observation(), reward, done, info
