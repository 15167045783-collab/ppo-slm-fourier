from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from ppo_slm.config import PPOConfig
from ppo_slm.environment import SLMEnvironment


def _target(size: int = 256) -> torch.Tensor:
    target = torch.zeros((1, size, size), dtype=torch.float32)
    target[:, size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = 1.0
    return target


def test_reset_is_reproducible_and_returns_three_channel_observation() -> None:
    config = PPOConfig.quick()
    env = SLMEnvironment(_target(), config, torch.device("cpu"))

    observation_a = env.reset(seed=23)
    phase_a = env.phase
    observation_b = env.reset(seed=23)

    assert torch.equal(phase_a, env.phase)
    assert torch.equal(observation_a, observation_b)
    assert observation_a.shape == (3, 256, 256)
    assert observation_a.dtype == torch.float32
    assert not observation_a.requires_grad
    assert torch.isfinite(observation_a).all()
    assert float(observation_a.min()) >= 0.0
    assert float(observation_a.max()) <= 1.0
    assert float(env.phase.min()) >= 0.0
    assert float(env.phase.max()) < 2 * math.pi
    weights = 1.0 + config.foreground_weight * env.target
    expected_weighted_mse = (
        weights * (env.camera - env.target).square()
    ).sum() / weights.sum()
    assert env.mse == pytest.approx(float(expected_weighted_mse.item()))


def test_unseeded_resets_advance_the_config_seed() -> None:
    env = SLMEnvironment(_target(), PPOConfig.quick(), torch.device("cpu"))
    env.reset()
    first = env.phase
    env.reset()

    assert not torch.equal(first, env.phase)


@pytest.mark.parametrize(
    "target",
    [
        torch.zeros((256, 256)),
        torch.zeros((1, 255, 256)),
        torch.zeros((1, 256, 256), dtype=torch.int64),
        torch.full((1, 256, 256), float("nan")),
        torch.full((1, 256, 256), 2.0),
    ],
)
def test_environment_rejects_invalid_targets(target: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        SLMEnvironment(target, PPOConfig.quick(), torch.device("cpu"))


def test_step_updates_wrapped_phase_and_uses_exact_reward_formula() -> None:
    config = replace(PPOConfig.quick(), target_mse=1e-9)
    env = SLMEnvironment(_target(), config, torch.device("cpu"))
    env.reset(seed=2)
    previous_phase = env.phase
    action = torch.full((config.action_size,), 10.0)

    _, reward, done, info = env.step(action)

    expected_phase = torch.remainder(
        previous_phase
        + config.action_scale
        * torch.tanh(action).reshape(config.phase_size, config.phase_size),
        2 * math.pi,
    )
    expected_reward = (
        config.improvement_gain * float(info["improvement"])
        - config.mse_weight * float(info["mse"])
        - config.action_penalty * float(info["action_magnitude"])
    )
    assert torch.allclose(env.phase, expected_phase)
    assert 0.0 <= float(env.phase.min())
    assert float(env.phase.max()) < 2 * math.pi
    assert reward == pytest.approx(expected_reward)
    assert done is False


def test_zero_action_preserves_phase() -> None:
    env = SLMEnvironment(
        _target(), replace(PPOConfig.quick(), target_mse=1e-9), torch.device("cpu")
    )
    env.reset(seed=3)
    phase = env.phase

    env.step(torch.zeros(env.config.action_size))

    assert torch.equal(phase, env.phase)


def test_environment_terminates_when_target_mse_is_reached() -> None:
    config = PPOConfig.quick()
    probe = SLMEnvironment(_target(), config, torch.device("cpu"))
    probe.reset(seed=31)
    exact_target = probe.camera
    env = SLMEnvironment(exact_target, config, torch.device("cpu"))
    env.reset(seed=31)

    _, _, done, info = env.step(torch.zeros(env.config.action_size))

    assert done is True
    assert info["reached_target"] is True
    assert env.reached_target is True


def test_environment_terminates_at_episode_length() -> None:
    config = replace(PPOConfig.quick(), episode_length=2, target_mse=1e-9)
    env = SLMEnvironment(_target(), config, torch.device("cpu"))
    env.reset(seed=7)

    assert env.step(torch.zeros(env.config.action_size))[2] is False
    assert env.step(torch.zeros(env.config.action_size))[2] is True
    assert env.steps == 2


@pytest.mark.parametrize(
    "action",
    [
        torch.zeros(63),
        torch.zeros((64, 64)),
        torch.zeros(4096, dtype=torch.int64),
        torch.full((4096,), float("nan")),
    ],
)
def test_step_rejects_invalid_actions(action: torch.Tensor) -> None:
    env = SLMEnvironment(_target(), PPOConfig.quick(), torch.device("cpu"))
    env.reset(seed=7)

    with pytest.raises(ValueError):
        env.step(action)
