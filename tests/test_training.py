from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ppo_slm.config import PPOConfig
from ppo_slm.optics import fourier_intensity
from ppo_slm.training import TrainingResult, train


def _write_target(path: Path) -> None:
    pixels = np.zeros((256, 256), dtype=np.uint8)
    pixels[72:184, 88:168] = 255
    Image.fromarray(pixels).save(path)


def test_quick_training_tracks_histories_and_global_best(tmp_path: Path) -> None:
    target_path = tmp_path / "target.bmp"
    _write_target(target_path)
    config = replace(
        PPOConfig.quick(), total_steps=5, rollout_steps=4, episode_length=3,
        minibatch_size=4, target_mse=1e-9,
    )

    result = train(config, target_path)

    assert isinstance(result, TrainingResult)
    assert result.steps == 5
    assert result.stop_reason == "total_steps"
    assert len(result.mse_history) == 5
    assert len(result.reward_history) == 5
    assert len(result.psnr_history) == 5
    assert all(np.isfinite(result.mse_history))
    assert all(np.isfinite(result.reward_history))
    assert all(np.isfinite(result.psnr_history))
    assert result.initial_phase.shape == (8, 8)
    assert result.initial_camera.shape == (1, 256, 256)
    assert result.best_phase.shape == (8, 8)
    assert result.best_camera.shape == (1, 256, 256)
    assert result.best_mse == pytest.approx(
        min([result.initial_mse, *result.mse_history])
    )
    assert 0 <= result.best_step <= result.steps
    recomputed = fourier_intensity(result.best_phase, 256)
    assert torch.allclose(recomputed.cpu(), result.best_camera, atol=1e-6)


def test_training_is_repeatable_for_the_same_seed(tmp_path: Path) -> None:
    target_path = tmp_path / "target.bmp"
    _write_target(target_path)
    config = replace(
        PPOConfig.quick(), total_steps=4, rollout_steps=4, episode_length=4,
        target_mse=1e-9,
    )

    first = train(config, target_path)
    second = train(config, target_path)

    assert first.mse_history == pytest.approx(second.mse_history)
    assert first.reward_history == pytest.approx(second.reward_history)
    assert torch.equal(first.best_phase, second.best_phase)


def test_training_stops_before_acting_when_initial_state_meets_threshold(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "target.bmp"
    _write_target(target_path)
    config = replace(PPOConfig.quick(), target_mse=0.99)

    result = train(config, target_path)

    assert result.stop_reason == "target_mse"
    assert result.steps == 0
    assert result.best_step == 0
    assert result.mse_history == []
