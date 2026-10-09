from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ppo_slm.optics import fourier_intensity, phase_to_uint8, upsample_phase


def test_upsample_phase_expands_8_by_8_and_preserves_constant() -> None:
    phase = torch.full((8, 8), 1.25)

    expanded = upsample_phase(phase, output_size=256)

    assert expanded.shape == (256, 256)
    assert torch.allclose(expanded, torch.full_like(expanded, 1.25))


def test_fourier_intensity_returns_detached_normalized_camera_frame() -> None:
    phase = torch.linspace(0.0, 2 * math.pi, 64).reshape(8, 8).requires_grad_()

    frame = fourier_intensity(phase, output_size=256)

    assert frame.shape == (1, 256, 256)
    assert frame.dtype == torch.float32
    assert not frame.requires_grad
    assert torch.isfinite(frame).all()
    assert float(frame.min()) >= 0.0
    assert float(frame.max()) == pytest.approx(1.0)


def test_phase_to_uint8_wraps_phase_and_preserves_8_by_8_shape() -> None:
    phase = torch.zeros((8, 8), dtype=torch.float32)
    phase[0, :4] = torch.tensor([-math.pi, 0.0, math.pi, 2 * math.pi])

    result = phase_to_uint8(phase)

    assert result.shape == (8, 8)
    assert result.dtype == np.uint8
    assert result[0].tolist()[:4] == [128, 0, 128, 0]


@pytest.mark.parametrize(
    "phase",
    [
        torch.zeros(8),
        torch.zeros((1, 8, 8)),
        torch.zeros((8, 8), dtype=torch.int64),
        torch.full((8, 8), float("nan")),
        torch.full((8, 8), float("inf")),
    ],
)
def test_optical_functions_reject_invalid_phase(phase: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        fourier_intensity(phase)
    with pytest.raises(ValueError):
        phase_to_uint8(phase)


@pytest.mark.parametrize("output_size", [0, -1, 4])
def test_upsample_phase_rejects_invalid_output_size(output_size: int) -> None:
    with pytest.raises(ValueError, match="output_size"):
        upsample_phase(torch.zeros((8, 8)), output_size)
