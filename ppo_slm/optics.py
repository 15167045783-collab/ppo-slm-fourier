from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as functional


def _validate_phase(phase: torch.Tensor) -> None:
    if phase.ndim != 2:
        raise ValueError("phase must have shape [H, W]")
    if not phase.is_floating_point():
        raise ValueError("phase must be a floating-point tensor")
    if not torch.isfinite(phase).all():
        raise ValueError("phase contains NaN or Inf")


def upsample_phase(phase: torch.Tensor, output_size: int = 256) -> torch.Tensor:
    _validate_phase(phase)
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if output_size < max(phase.shape):
        raise ValueError("output_size cannot be smaller than the phase map")
    return functional.interpolate(
        phase[None, None],
        size=(output_size, output_size),
        mode="nearest",
    )[0, 0]


def fourier_intensity(phase: torch.Tensor, output_size: int = 256) -> torch.Tensor:
    _validate_phase(phase)
    with torch.no_grad():
        expanded = upsample_phase(phase, output_size)
        field = torch.polar(torch.ones_like(expanded), expanded)
        focal_field = torch.fft.fftshift(torch.fft.fft2(field, norm="ortho"))
        intensity = focal_field.abs().square()
        maximum = intensity.amax()
        if not torch.isfinite(maximum) or float(maximum) <= 0.0:
            raise ValueError("optical propagation produced invalid intensity")
        normalized = intensity / maximum
        if not torch.isfinite(normalized).all():
            raise ValueError("normalized intensity contains NaN or Inf")
        return normalized.unsqueeze(0).detach().clone()


def phase_to_uint8(phase: torch.Tensor) -> np.ndarray:
    _validate_phase(phase)
    wrapped = torch.remainder(phase.detach().cpu(), 2 * math.pi)
    levels = torch.round(wrapped / (2 * math.pi) * 255.0)
    return levels.to(torch.uint8).numpy()
