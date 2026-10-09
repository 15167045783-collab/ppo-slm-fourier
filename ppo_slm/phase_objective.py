from __future__ import annotations

from dataclasses import dataclass

import torch

from .phase_ppo_config import PhasePPOConfig


_EPSILON = 1e-8


@dataclass(frozen=True)
class PhaseObjectiveBatch:
    total_loss: torch.Tensor
    amplitude_mse: torch.Tensor
    ncc: torch.Tensor
    outside_energy_ratio: torch.Tensor
    intensity_mse: torch.Tensor
    psnr_db: torch.Tensor


def _validate_inputs(
    intensities: torch.Tensor,
    target: torch.Tensor,
    config: PhasePPOConfig,
) -> None:
    expected_image = (config.image_size, config.image_size)
    if intensities.ndim != 4 or tuple(intensities.shape[1:]) != (
        1,
        *expected_image,
    ):
        raise ValueError(
            f"intensities must have shape [B, 1, {config.image_size}, "
            f"{config.image_size}]"
        )
    if intensities.shape[0] == 0:
        raise ValueError("intensity batch cannot be empty")
    if not intensities.is_floating_point():
        raise ValueError("intensities must be floating point")
    if not torch.isfinite(intensities).all() or bool((intensities < 0).any()):
        raise ValueError("intensities must be finite and nonnegative")

    if tuple(target.shape) != (1, *expected_image):
        raise ValueError(
            f"target must have shape [1, {config.image_size}, {config.image_size}]"
        )
    if not target.is_floating_point() or not torch.isfinite(target).all():
        raise ValueError("target must be finite and floating point")
    if float(target.min()) < 0.0 or float(target.max()) > 1.0:
        raise ValueError("target values must be in [0, 1]")
    if not bool((target > config.target_threshold).any()):
        raise ValueError("target region of interest is empty")


def _rms_normalize(values: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, values.ndim))
    rms = values.square().mean(dim=dimensions, keepdim=True).sqrt()
    return values / rms.clamp_min(_EPSILON)


def evaluate_phase_objective(
    intensities: torch.Tensor,
    target: torch.Tensor,
    config: PhasePPOConfig,
) -> PhaseObjectiveBatch:
    _validate_inputs(intensities, target, config)
    target = target.to(device=intensities.device, dtype=intensities.dtype)
    target_batch = target.unsqueeze(0)

    amplitude = _rms_normalize(intensities.sqrt())
    target_amplitude = _rms_normalize(target_batch.sqrt())
    weights = 1.0 + config.foreground_weight * target_batch
    amplitude_mse = (
        weights * (amplitude - target_amplitude).square()
    ).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3))

    flat_amplitude = amplitude.flatten(start_dim=1)
    flat_target = target_amplitude.expand_as(amplitude).flatten(start_dim=1)
    centered_amplitude = flat_amplitude - flat_amplitude.mean(dim=1, keepdim=True)
    centered_target = flat_target - flat_target.mean(dim=1, keepdim=True)
    numerator = (centered_amplitude * centered_target).sum(dim=1)
    denominator = (
        centered_amplitude.square().sum(dim=1)
        * centered_target.square().sum(dim=1)
    ).sqrt()
    ncc = (numerator / denominator.clamp_min(_EPSILON)).clamp(-1.0, 1.0)

    region = target_batch > config.target_threshold
    outside = ~region
    total_energy = intensities.sum(dim=(1, 2, 3))
    outside_energy = (intensities * outside).sum(dim=(1, 2, 3))
    outside_energy_ratio = outside_energy / total_energy.clamp_min(_EPSILON)

    normalized_intensity = _rms_normalize(intensities)
    normalized_target = _rms_normalize(target_batch)
    intensity_mse = (normalized_intensity - normalized_target).square().mean(
        dim=(1, 2, 3)
    )
    psnr_db = -10.0 * torch.log10(intensity_mse.clamp_min(1e-12))

    total_loss = (
        config.mse_coefficient * amplitude_mse
        + config.ncc_coefficient * (1.0 - ncc)
        + config.outside_coefficient * outside_energy_ratio
    )
    values = (
        total_loss,
        amplitude_mse,
        ncc,
        outside_energy_ratio,
        intensity_mse,
        psnr_db,
    )
    if not all(torch.isfinite(value).all() for value in values):
        raise ValueError("phase objective produced NaN or Inf")
    return PhaseObjectiveBatch(*values)
