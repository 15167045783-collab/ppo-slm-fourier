from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import ppo_slm.phase_ppo_training as training_module
from ppo_slm.phase_ppo_config import PhasePPOConfig
from ppo_slm.phase_ppo_training import train_phase_ppo


def _target(path: Path, size: int = 4) -> Path:
    values = np.zeros((size, size), dtype=np.uint8)
    values[1:-1, 1:-1] = 255
    Image.fromarray(values, mode="L").save(path)
    return path


def _config(seed: int = 7, rounds: int = 2) -> PhasePPOConfig:
    return replace(
        PhasePPOConfig.quick(),
        phase_size=2,
        image_size=4,
        rounds=rounds,
        population_size=4,
        minibatch_size=2,
        learning_rate=0.01,
        seed=seed,
        device="cpu",
    )


def test_training_counts_evaluations_tracks_best_and_returns_finite_histories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(rounds=3)
    target_path = _target(tmp_path / "target.bmp")
    original_capture = training_module.fourier_intensity_batch
    captured = 0

    def counted_capture(phases: torch.Tensor, output_size: int) -> torch.Tensor:
        nonlocal captured
        captured += phases.shape[0]
        return original_capture(phases, output_size)

    monkeypatch.setattr(training_module, "fourier_intensity_batch", counted_capture)

    result = train_phase_ppo(config, target_path)

    assert captured == 1 + config.rounds * config.population_size
    assert result.total_evaluations == captured
    assert result.best_loss == pytest.approx(
        min(result.initial_loss, *result.best_loss_history)
    )
    assert result.best_round in range(config.rounds + 1)
    assert result.initial_phase.shape == (2, 2)
    assert result.best_phase.shape == (2, 2)
    assert result.final_mean_phase.shape == (2, 2)
    assert result.final_mean_raw.shape == (2, 2)
    assert result.initial_camera.shape == (1, 4, 4)
    assert result.best_camera.shape == (1, 4, 4)
    assert float(result.best_phase.min()) >= 0.0
    assert float(result.best_phase.max()) < 2 * math.pi
    assert float(result.final_mean_phase.min()) >= 0.0
    assert float(result.final_mean_phase.max()) < 2 * math.pi
    assert torch.allclose(
        result.final_mean_phase,
        torch.remainder(result.final_mean_raw, 2 * math.pi),
    )

    histories = (
        result.best_loss_history,
        result.mean_loss_history,
        result.best_ncc_history,
        result.mean_outside_energy_history,
        result.policy_loss_history,
        result.approx_kl_history,
        result.clip_fraction_history,
    )
    assert all(len(history) == config.rounds for history in histories)
    assert all(math.isfinite(value) for history in histories for value in history)
    assert all(math.isfinite(value) for value in result.update_metrics.values())
    assert result.runtime_seconds >= 0.0
    assert result.target.device.type == "cpu"
    assert result.final_mean_raw.device.type == "cpu"


def test_repeated_cpu_training_is_reproducible(tmp_path: Path) -> None:
    target_path = _target(tmp_path / "target.bmp")
    config = _config(seed=31)

    first = train_phase_ppo(config, target_path)
    second = train_phase_ppo(config, target_path)

    assert torch.equal(first.initial_phase, second.initial_phase)
    assert torch.equal(first.best_phase, second.best_phase)
    assert torch.equal(first.final_mean_raw, second.final_mean_raw)
    assert first.best_loss_history == second.best_loss_history
    assert first.policy_loss_history == second.policy_loss_history


def test_different_seeds_create_different_initial_means(tmp_path: Path) -> None:
    target_path = _target(tmp_path / "target.bmp")

    first = train_phase_ppo(_config(seed=1, rounds=1), target_path)
    second = train_phase_ppo(_config(seed=2, rounds=1), target_path)

    assert not torch.equal(first.initial_phase, second.initial_phase)


def test_small_quick_workload_completes_two_rounds(tmp_path: Path) -> None:
    target_path = _target(tmp_path / "target.bmp")
    config = _config()

    result = train_phase_ppo(config, target_path)

    assert config.rounds == 2
    assert len(result.best_loss_history) == 2
    assert result.total_evaluations == 9


def test_cuda_oom_recommends_reducing_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path = _target(tmp_path / "target.bmp")

    def out_of_memory(*args: object, **kwargs: object) -> torch.Tensor:
        raise torch.cuda.OutOfMemoryError("synthetic OOM")

    monkeypatch.setattr(training_module, "fourier_intensity_batch", out_of_memory)
    with pytest.raises(RuntimeError, match="population-size"):
        train_phase_ppo(_config(), target_path)


def test_legacy_training_entry_point_remains_available() -> None:
    from ppo_slm.training import train

    assert callable(train)
