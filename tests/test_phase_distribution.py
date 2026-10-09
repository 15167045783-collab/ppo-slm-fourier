from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from ppo_slm.phase_distribution import (
    PhaseDistributionPolicy,
    PhaseSamples,
    standardize_advantages,
    update_phase_policy,
)
from ppo_slm.phase_ppo_config import PhasePPOConfig


def _config(**overrides: int | float | str) -> PhasePPOConfig:
    values: dict[str, int | float | str] = {
        "phase_size": 2,
        "image_size": 4,
        "population_size": 4,
        "minibatch_size": 2,
    }
    values.update(overrides)
    return replace(PhasePPOConfig.quick(), **values)


def test_policy_initializes_one_seeded_trainable_mean_and_fixed_sigma() -> None:
    config = _config()
    first = PhaseDistributionPolicy(config, torch.device("cpu"))
    second = PhaseDistributionPolicy(config, torch.device("cpu"))

    assert first.mean.shape == (2, 2)
    assert float(first.mean.min()) >= 0.0
    assert float(first.mean.max()) < 2 * math.pi
    assert torch.equal(first.mean, second.mean)
    assert first.sigma.item() == pytest.approx(config.sigma)
    assert first.sigma.requires_grad is False
    assert [name for name, _ in first.named_parameters()] == ["mean"]


def test_policy_samples_are_reproducible_detached_and_wrapped_for_display() -> None:
    config = _config()
    first = PhaseDistributionPolicy(config, torch.device("cpu"))
    second = PhaseDistributionPolicy(config, torch.device("cpu"))
    with torch.no_grad():
        first.mean.fill_(2 * math.pi + 1.0)
        second.mean.copy_(first.mean)

    samples_a = first.sample(4)
    samples_b = second.sample(4)

    assert samples_a.unwrapped.shape == (4, 2, 2)
    assert samples_a.display.shape == (4, 2, 2)
    assert samples_a.old_log_probs.shape == (4, 2, 2)
    assert torch.equal(samples_a.unwrapped, samples_b.unwrapped)
    assert torch.equal(samples_a.display, samples_b.display)
    assert float(samples_a.unwrapped.min()) > 2 * math.pi
    assert float(samples_a.display.min()) >= 0.0
    assert float(samples_a.display.max()) < 2 * math.pi
    assert torch.allclose(
        samples_a.old_log_probs,
        first.log_prob(samples_a.unwrapped),
    )
    assert not samples_a.unwrapped.requires_grad
    assert not samples_a.display.requires_grad
    assert not samples_a.old_log_probs.requires_grad


def test_standardize_advantages_is_finite_and_handles_constant_rewards() -> None:
    advantages = standardize_advantages(torch.tensor([1.0, 2.0, 4.0, 5.0]))
    constant = standardize_advantages(torch.ones(4))

    assert advantages.mean().item() == pytest.approx(0.0, abs=1e-7)
    assert advantages.std(unbiased=False).item() == pytest.approx(1.0)
    assert torch.equal(constant, torch.zeros(4))
    assert torch.isfinite(advantages).all()
    assert torch.isfinite(constant).all()


def test_pixelwise_clipped_update_matches_hand_calculation() -> None:
    config = _config(
        phase_size=1,
        image_size=1,
        population_size=2,
        minibatch_size=2,
        update_epochs=1,
        sigma=1.0,
        clip_ratio=0.2,
    )
    policy = PhaseDistributionPolicy(config, torch.device("cpu"))
    with torch.no_grad():
        policy.mean.zero_()
    unwrapped = torch.zeros((2, 1, 1))
    new_log_probs = policy.log_prob(unwrapped).detach()
    desired_ratios = torch.tensor([1.5, 0.5]).reshape(2, 1, 1)
    samples = PhaseSamples(
        unwrapped=unwrapped,
        display=unwrapped,
        old_log_probs=new_log_probs - desired_ratios.log(),
    )
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)

    metrics = update_phase_policy(
        policy,
        optimizer,
        samples,
        torch.tensor([1.0, -1.0]),
        config,
    )

    assert metrics["policy_loss"] == pytest.approx(-0.2)
    assert metrics["clip_fraction"] == pytest.approx(1.0)
    assert all(math.isfinite(value) for value in metrics.values())


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters: object, learning_rate: float) -> None:
        super().__init__(parameters, lr=learning_rate)
        self.steps = 0

    def step(self, closure: object = None) -> object:
        self.steps += 1
        return super().step(closure)


def test_update_uses_partial_final_minibatch_and_changes_mean() -> None:
    config = _config(population_size=5, minibatch_size=2, update_epochs=2)
    policy = PhaseDistributionPolicy(config, torch.device("cpu"))
    samples = policy.sample(5)
    optimizer = _CountingSGD(policy.parameters(), 0.01)
    before = policy.mean.detach().clone()
    advantages = standardize_advantages(torch.tensor([3.0, 1.0, -1.0, 2.0, -2.0]))

    metrics = update_phase_policy(policy, optimizer, samples, advantages, config)

    assert optimizer.steps == 6
    assert not torch.equal(before, policy.mean)
    assert all(math.isfinite(value) for value in metrics.values())


def test_zero_advantages_leave_fresh_policy_unchanged() -> None:
    config = _config()
    policy = PhaseDistributionPolicy(config, torch.device("cpu"))
    samples = policy.sample(config.population_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    before = policy.mean.detach().clone()

    update_phase_policy(
        policy,
        optimizer,
        samples,
        torch.zeros(config.population_size),
        config,
    )

    assert torch.equal(before, policy.mean)


class _PoisonSGD(torch.optim.SGD):
    def step(self, closure: object = None) -> object:
        result = super().step(closure)
        with torch.no_grad():
            self.param_groups[0]["params"][0].fill_(float("nan"))
        return result


def test_update_rejects_nonfinite_policy_after_optimizer_step() -> None:
    config = _config(update_epochs=1)
    policy = PhaseDistributionPolicy(config, torch.device("cpu"))
    samples = policy.sample(config.population_size)
    optimizer = _PoisonSGD(policy.parameters(), lr=0.01)
    advantages = standardize_advantages(torch.arange(4, dtype=torch.float32))

    with pytest.raises(FloatingPointError, match="mean"):
        update_phase_policy(policy, optimizer, samples, advantages, config)


@pytest.mark.parametrize(
    ("rewards", "message"),
    [
        (torch.ones((2, 2)), "vector"),
        (torch.tensor([1.0, float("nan")]), "finite"),
        (torch.tensor([]), "empty"),
    ],
)
def test_standardize_advantages_rejects_invalid_rewards(
    rewards: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        standardize_advantages(rewards)


def test_update_rejects_malformed_or_nonfinite_inputs() -> None:
    config = _config()
    policy = PhaseDistributionPolicy(config, torch.device("cpu"))
    samples = policy.sample(config.population_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    with pytest.raises(ValueError, match="advantages"):
        update_phase_policy(
            policy,
            optimizer,
            samples,
            torch.zeros(config.population_size - 1),
            config,
        )
    broken = PhaseSamples(
        unwrapped=torch.full_like(samples.unwrapped, float("nan")),
        display=samples.display,
        old_log_probs=samples.old_log_probs,
    )
    with pytest.raises(ValueError, match="finite"):
        update_phase_policy(
            policy,
            optimizer,
            broken,
            torch.zeros(config.population_size),
            config,
        )
