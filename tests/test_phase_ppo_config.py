from __future__ import annotations

from dataclasses import replace

import pytest

from ppo_slm.phase_ppo_config import PhasePPOConfig


def test_default_phase_ppo_config_matches_paper_style_design() -> None:
    config = PhasePPOConfig()

    assert config.phase_size == 64
    assert config.image_size == 256
    assert config.action_size == 4096
    assert config.rounds == 500
    assert config.population_size == 64
    assert config.sigma == pytest.approx(0.04)
    assert config.clip_ratio == pytest.approx(0.02)
    assert config.update_epochs == 4
    assert config.minibatch_size == 16
    assert config.learning_rate == pytest.approx(0.3)
    assert config.max_grad_norm == pytest.approx(1.0)
    assert config.foreground_weight == pytest.approx(8.0)
    assert config.mse_coefficient == pytest.approx(0.60)
    assert config.ncc_coefficient == pytest.approx(0.30)
    assert config.outside_coefficient == pytest.approx(0.10)
    assert config.target_threshold == pytest.approx(0.05)
    assert config.seed == 7
    assert config.device == "auto"


def test_quick_phase_ppo_config_only_reduces_workload() -> None:
    normal = PhasePPOConfig()
    quick = PhasePPOConfig.quick()

    assert quick.rounds == 2
    assert quick.population_size == 8
    assert quick.update_epochs == 1
    assert quick.minibatch_size == 4
    assert quick.phase_size == normal.phase_size
    assert quick.image_size == normal.image_size
    assert quick.sigma == normal.sigma
    assert quick.seed == normal.seed


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phase_size", 0),
        ("image_size", 0),
        ("rounds", 0),
        ("population_size", 1),
        ("update_epochs", 0),
        ("minibatch_size", 0),
        ("sigma", 0.0),
        ("learning_rate", 0.0),
        ("max_grad_norm", 0.0),
    ],
)
def test_phase_ppo_config_rejects_nonpositive_values(
    field: str, value: int | float
) -> None:
    with pytest.raises(ValueError, match=field):
        replace(PhasePPOConfig(), **{field: value})


def test_phase_ppo_config_rejects_incoherent_batch_and_clip_values() -> None:
    with pytest.raises(ValueError, match="minibatch_size"):
        PhasePPOConfig(population_size=8, minibatch_size=9)
    for clip_ratio in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="clip_ratio"):
            PhasePPOConfig(clip_ratio=clip_ratio)


@pytest.mark.parametrize(
    "field",
    [
        "foreground_weight",
        "mse_coefficient",
        "ncc_coefficient",
        "outside_coefficient",
    ],
)
def test_phase_ppo_config_rejects_negative_objective_values(field: str) -> None:
    with pytest.raises(ValueError, match="negative"):
        replace(PhasePPOConfig(), **{field: -0.1})


def test_phase_ppo_config_requires_a_nonzero_objective() -> None:
    with pytest.raises(ValueError, match="objective"):
        PhasePPOConfig(
            mse_coefficient=0.0,
            ncc_coefficient=0.0,
            outside_coefficient=0.0,
        )


@pytest.mark.parametrize("threshold", [-0.1, 0.0, 1.0, 1.1])
def test_phase_ppo_config_rejects_invalid_target_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="target_threshold"):
        PhasePPOConfig(target_threshold=threshold)


def test_phase_ppo_config_rejects_invalid_seed_and_device() -> None:
    with pytest.raises(ValueError, match="seed"):
        PhasePPOConfig(seed=-1)
    with pytest.raises(ValueError, match="device"):
        PhasePPOConfig(device="tpu")
