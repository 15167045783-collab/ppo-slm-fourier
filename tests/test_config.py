from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from ppo_slm.config import PPOConfig, project_path, seed_everything, select_device


def test_default_config_has_expected_geometry_and_valid_training_limits() -> None:
    config = PPOConfig()

    assert config.phase_size == 64
    assert config.image_size == 256
    assert config.action_size == 4096
    assert config.foreground_weight == pytest.approx(8.0)
    assert config.total_steps > 0
    assert 0 < config.rollout_steps <= config.total_steps
    assert 0 < config.minibatch_size <= config.rollout_steps
    assert config.rollout_steps % config.minibatch_size == 0
    assert 0.0 < config.target_mse < 1.0


def test_quick_config_reduces_the_workload_without_changing_geometry() -> None:
    normal = PPOConfig()
    quick = PPOConfig.quick()

    assert quick.phase_size == normal.phase_size
    assert quick.image_size == normal.image_size
    assert quick.total_steps == 16
    assert quick.rollout_steps == 8
    assert quick.episode_length == 8
    assert quick.update_epochs == 1
    assert quick.minibatch_size == 4


def test_project_path_resolves_relative_to_ppo_root() -> None:
    assert project_path("data/1.bmp") == Path(__file__).resolve().parents[1] / "data" / "1.bmp"


def test_seed_everything_repeats_python_numpy_and_torch_streams() -> None:
    seed_everything(13)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(13)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second


def test_select_device_uses_cpu_when_requested() -> None:
    assert select_device("cpu") == torch.device("cpu")


def test_select_device_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="device"):
        select_device("tpu")


def test_explicit_cuda_fails_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        select_device("cuda")


def test_config_rejects_negative_seed_and_zero_optimizer_scales() -> None:
    with pytest.raises(ValueError, match="seed"):
        PPOConfig(seed=-1)
    with pytest.raises(ValueError, match="learning_rate"):
        PPOConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="action_scale"):
        PPOConfig(action_scale=0.0)
    with pytest.raises(ValueError, match="max_grad_norm"):
        PPOConfig(max_grad_norm=0.0)
    with pytest.raises(ValueError, match="weights and scales"):
        PPOConfig(foreground_weight=-1.0)
