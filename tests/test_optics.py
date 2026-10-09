from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ppo_slm.optics import (
    fourier_intensity,
    fourier_intensity_batch,
    phase_to_uint8,
    upsample_phase,
    upsample_phase_batch,
)


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


def test_upsample_phase_batch_preserves_nearest_neighbor_macro_pixels() -> None:
    first = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    phases = torch.stack((first, first.flip(0)))

    expanded = upsample_phase_batch(phases, output_size=256)

    assert expanded.shape == (2, 256, 256)
    assert torch.equal(expanded[0, :4, :4], first[0, 0].expand(4, 4))
    assert torch.equal(expanded[0, :4, 4:8], first[0, 1].expand(4, 4))
    assert torch.equal(expanded[1, :4, :4], first[-1, 0].expand(4, 4))


def test_fourier_intensity_batch_returns_raw_detached_intensity() -> None:
    phases = torch.stack(
        (
            torch.linspace(0.0, 2 * math.pi, 4096).reshape(64, 64),
            torch.linspace(2 * math.pi, 0.0, 4096).reshape(64, 64),
        )
    ).requires_grad_()

    intensities = fourier_intensity_batch(phases, output_size=256)

    assert intensities.shape == (2, 1, 256, 256)
    assert intensities.dtype == torch.float32
    assert not intensities.requires_grad
    assert torch.isfinite(intensities).all()
    assert float(intensities.min()) >= 0.0
    assert float(intensities.max()) > 1.0


def test_batched_fourier_matches_each_legacy_single_capture_after_normalizing() -> None:
    generator = torch.Generator().manual_seed(19)
    phases = torch.rand((3, 64, 64), generator=generator) * (2 * math.pi)

    raw = fourier_intensity_batch(phases, output_size=256)

    for index in range(len(phases)):
        normalized = raw[index] / raw[index].amax()
        legacy = fourier_intensity(phases[index], output_size=256)
        assert torch.allclose(normalized, legacy, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "phases",
    [
        torch.zeros((64, 64)),
        torch.zeros((0, 64, 64)),
        torch.zeros((2, 64, 64), dtype=torch.int64),
        torch.full((2, 64, 64), float("nan")),
        torch.full((2, 64, 64), float("inf")),
    ],
)
def test_batch_optics_rejects_invalid_phase_batches(phases: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        upsample_phase_batch(phases)
    with pytest.raises(ValueError):
        fourier_intensity_batch(phases)


@pytest.mark.parametrize("output_size", [0, -1, 32])
def test_batch_optics_rejects_invalid_output_size(output_size: int) -> None:
    phases = torch.zeros((2, 64, 64))
    with pytest.raises(ValueError, match="output_size"):
        upsample_phase_batch(phases, output_size)
    with pytest.raises(ValueError, match="output_size"):
        fourier_intensity_batch(phases, output_size)
