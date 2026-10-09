from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ppo_slm.optics import fourier_intensity, phase_to_uint8, upsample_phase


def test_upsample_phase_expands_64_by_64_with_constant_macro_pixels() -> None:
    phase = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)

    expanded = upsample_phase(phase, output_size=256)

    assert expanded.shape == (256, 256)
    assert torch.equal(expanded[:4, :4], torch.full((4, 4), phase[0, 0]))
    assert torch.equal(expanded[:4, 4:8], torch.full((4, 4), phase[0, 1]))
    assert torch.equal(expanded[252:, 252:], torch.full((4, 4), phase[-1, -1]))


def test_fourier_intensity_returns_detached_normalized_camera_frame() -> None:
    phase = torch.linspace(0.0, 2 * math.pi, 4096).reshape(64, 64).requires_grad_()

    frame = fourier_intensity(phase, output_size=256)

    assert frame.shape == (1, 256, 256)
    assert frame.dtype == torch.float32
    assert not frame.requires_grad
    assert torch.isfinite(frame).all()
    assert float(frame.min()) >= 0.0
    assert float(frame.max()) == pytest.approx(1.0)


def test_phase_to_uint8_wraps_phase_and_preserves_64_by_64_shape() -> None:
    phase = torch.zeros((64, 64), dtype=torch.float32)
    phase[0, :4] = torch.tensor([-math.pi, 0.0, math.pi, 2 * math.pi])

    result = phase_to_uint8(phase)

    assert result.shape == (64, 64)
    assert result.dtype == np.uint8
    assert result[0].tolist()[:4] == [128, 0, 128, 0]


@pytest.mark.parametrize(
    "phase",
    [
        torch.zeros(8),
        torch.zeros((1, 8, 8)),
        torch.zeros((64, 64), dtype=torch.int64),
        torch.full((64, 64), float("nan")),
        torch.full((64, 64), float("inf")),
    ],
)
def test_optical_functions_reject_invalid_phase(phase: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        fourier_intensity(phase)
    with pytest.raises(ValueError):
        phase_to_uint8(phase)


@pytest.mark.parametrize("output_size", [0, -1, 32])
def test_upsample_phase_rejects_invalid_output_size(output_size: int) -> None:
    with pytest.raises(ValueError, match="output_size"):
        upsample_phase(torch.zeros((64, 64)), output_size)
