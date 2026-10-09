from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from ppo_slm.phase_objective import evaluate_phase_objective
from ppo_slm.phase_ppo_config import PhasePPOConfig


def _config(size: int = 2) -> PhasePPOConfig:
    return replace(
        PhasePPOConfig.quick(),
        phase_size=size,
        image_size=size,
    )


def test_objective_is_invariant_to_positive_intensity_scaling() -> None:
    target = torch.tensor([[[1.0, 0.25], [0.0, 0.5]]])
    first = torch.tensor([[[[0.8, 0.2], [0.1, 0.4]]]])
    intensities = torch.cat((first, first * 7.0), dim=0)

    metrics = evaluate_phase_objective(intensities, target, _config())

    for value in (
        metrics.total_loss,
        metrics.amplitude_mse,
        metrics.ncc,
        metrics.outside_energy_ratio,
        metrics.intensity_mse,
        metrics.psnr_db,
    ):
        assert value.shape == (2,)
        assert torch.isfinite(value).all()
        assert value[0].item() == pytest.approx(value[1].item(), abs=1e-6)


@pytest.mark.parametrize(
    ("intensities", "target"),
    [
        (torch.zeros((0, 1, 2, 2)), torch.ones((1, 2, 2))),
        (torch.zeros((1, 2, 2)), torch.ones((1, 2, 2))),
        (torch.zeros((1, 1, 2, 3)), torch.ones((1, 2, 2))),
        (torch.zeros((1, 1, 2, 2), dtype=torch.int64), torch.ones((1, 2, 2))),
        (torch.full((1, 1, 2, 2), float("nan")), torch.ones((1, 2, 2))),
        (-torch.ones((1, 1, 2, 2)), torch.ones((1, 2, 2))),
        (torch.ones((1, 1, 2, 2)), torch.ones((2, 2))),
        (torch.ones((1, 1, 2, 2)), torch.ones((1, 2, 2), dtype=torch.int64)),
        (torch.ones((1, 1, 2, 2)), torch.full((1, 2, 2), float("inf"))),
        (torch.ones((1, 1, 2, 2)), torch.full((1, 2, 2), 2.0)),
    ],
)
def test_objective_rejects_invalid_inputs(
    intensities: torch.Tensor, target: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        evaluate_phase_objective(intensities, target, _config())


def test_objective_rejects_target_with_empty_region_of_interest() -> None:
    target = torch.full((1, 2, 2), 0.01)
    with pytest.raises(ValueError, match="region of interest"):
        evaluate_phase_objective(torch.ones((1, 1, 2, 2)), target, _config())


def test_objective_metrics_match_hand_calculation() -> None:
    target = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    intensity = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])

    metrics = evaluate_phase_objective(intensity, target, _config())

    assert metrics.amplitude_mse.item() == pytest.approx(40.0 / 12.0)
    assert metrics.ncc.item() == pytest.approx(-1.0 / 3.0)
    assert metrics.outside_energy_ratio.item() == pytest.approx(1.0)
    assert metrics.intensity_mse.item() == pytest.approx(2.0)
    assert metrics.psnr_db.item() == pytest.approx(-3.0102999566)
    assert metrics.total_loss.item() == pytest.approx(2.5)


def test_perfect_reconstruction_beats_impulse_and_random_speckle() -> None:
    target = torch.zeros((1, 4, 4))
    target[:, 1:3, 1] = 1.0
    target[:, 2, 2] = 1.0
    impulse = torch.zeros((1, 1, 4, 4))
    impulse[:, :, 2, 2] = 1.0
    random = torch.rand((1, 1, 4, 4), generator=torch.Generator().manual_seed(3))
    candidates = torch.cat((target.unsqueeze(0), impulse, random), dim=0)

    metrics = evaluate_phase_objective(candidates, target, _config(4))

    assert metrics.total_loss[0].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics.ncc[0].item() == pytest.approx(1.0, abs=1e-6)
    assert metrics.total_loss[0] < metrics.total_loss[1]
    assert metrics.total_loss[0] < metrics.total_loss[2]
