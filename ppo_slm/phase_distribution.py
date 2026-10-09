from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal

from .phase_ppo_config import PhasePPOConfig


@dataclass(frozen=True)
class PhaseSamples:
    """A frozen rollout from the phase distribution.

    ``unwrapped`` is used for probability ratios. ``display`` is the same
    physical phase wrapped to the interval accepted by the SLM/simulator.
    """

    unwrapped: torch.Tensor
    display: torch.Tensor
    old_log_probs: torch.Tensor


class PhaseDistributionPolicy(nn.Module):
    """A factorized Gaussian distribution over a trainable phase mean."""

    def __init__(self, config: PhasePPOConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.sample_generator = torch.Generator(device=device)
        self.sample_generator.manual_seed(config.seed)
        self.update_generator = torch.Generator(device=device)
        self.update_generator.manual_seed(config.seed + 1)

        initial_mean = torch.rand(
            (config.phase_size, config.phase_size),
            generator=self.sample_generator,
            device=device,
            dtype=torch.float32,
        ) * (2.0 * math.pi)
        self.mean = nn.Parameter(initial_mean)
        self.register_buffer(
            "sigma",
            torch.tensor(config.sigma, device=device, dtype=torch.float32),
        )

    def log_prob(self, unwrapped_phase: torch.Tensor) -> torch.Tensor:
        if not unwrapped_phase.is_floating_point():
            raise ValueError("unwrapped phase must be floating point")
        expected = (self.config.phase_size, self.config.phase_size)
        if unwrapped_phase.shape[-2:] != expected:
            raise ValueError(f"phase must end with shape {expected}")
        return Normal(self.mean, self.sigma).log_prob(unwrapped_phase)

    @torch.no_grad()
    def sample(self, count: int) -> PhaseSamples:
        if count <= 0:
            raise ValueError("sample count must be positive")
        noise = torch.randn(
            (count, self.config.phase_size, self.config.phase_size),
            generator=self.sample_generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        unwrapped = self.mean.detach() + self.sigma * noise
        display = torch.remainder(unwrapped, 2.0 * math.pi)
        old_log_probs = self.log_prob(unwrapped)
        return PhaseSamples(
            unwrapped=unwrapped.detach(),
            display=display.detach(),
            old_log_probs=old_log_probs.detach(),
        )


def standardize_advantages(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.ndim != 1:
        raise ValueError("rewards must be a vector")
    if rewards.numel() == 0:
        raise ValueError("rewards cannot be empty")
    if not rewards.is_floating_point():
        rewards = rewards.float()
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite")
    centered = rewards - rewards.mean()
    scale = centered.std(unbiased=False)
    if scale <= torch.finfo(rewards.dtype).eps:
        return torch.zeros_like(rewards)
    return centered / scale


def _validate_update_inputs(
    policy: PhaseDistributionPolicy,
    samples: PhaseSamples,
    advantages: torch.Tensor,
) -> int:
    tensors = (samples.unwrapped, samples.display, samples.old_log_probs)
    if any(tensor.ndim != 3 for tensor in tensors):
        raise ValueError("sample tensors must have shape [population, height, width]")
    if not all(tensor.shape == tensors[0].shape for tensor in tensors[1:]):
        raise ValueError("sample tensors must have matching shapes")
    expected_phase = (policy.config.phase_size, policy.config.phase_size)
    if tensors[0].shape[-2:] != expected_phase:
        raise ValueError(f"sample phase must have shape {expected_phase}")
    count = tensors[0].shape[0]
    if advantages.ndim != 1 or advantages.shape[0] != count:
        raise ValueError("advantages must match the sampled population")
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise ValueError("sample tensors must be floating point")
    if not advantages.is_floating_point():
        raise ValueError("advantages must be floating point")
    if not all(torch.isfinite(tensor).all() for tensor in (*tensors, advantages)):
        raise ValueError("sample tensors and advantages must be finite")
    return count


def update_phase_policy(
    policy: PhaseDistributionPolicy,
    optimizer: torch.optim.Optimizer,
    samples: PhaseSamples,
    advantages: torch.Tensor,
    config: PhasePPOConfig,
) -> dict[str, float]:
    """Run PPO epochs using per-pixel likelihood ratios.

    Keeping log probabilities per pixel prevents the product of thousands of
    probabilities from collapsing for a 64x64 phase map.
    """

    count = _validate_update_inputs(policy, samples, advantages)
    device = policy.mean.device
    unwrapped = samples.unwrapped.to(device)
    old_log_probs = samples.old_log_probs.to(device)
    advantages = advantages.to(device)

    metric_totals = {
        "policy_loss": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "gradient_norm": 0.0,
    }
    updates = 0

    for _ in range(config.update_epochs):
        order = torch.randperm(
            count,
            generator=policy.update_generator,
            device=device,
        )
        for start in range(0, count, config.minibatch_size):
            indices = order[start : start + config.minibatch_size]
            new_log_probs = policy.log_prob(unwrapped[indices])
            log_ratio = new_log_probs - old_log_probs[indices]
            safe_log_ratio = log_ratio.clamp(-20.0, 20.0)
            ratio = safe_log_ratio.exp()
            minibatch_advantages = advantages[indices, None, None]
            unclipped = ratio * minibatch_advantages
            clipped = ratio.clamp(
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            ) * minibatch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            if policy.mean.grad is None or not torch.isfinite(policy.mean.grad).all():
                raise FloatingPointError("phase policy produced non-finite gradients")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), config.max_grad_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("phase policy gradient norm is not finite")
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - safe_log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > config.clip_ratio
                ).float().mean()
                values = {
                    "policy_loss": policy_loss,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "gradient_norm": gradient_norm,
                }
                for name, value in values.items():
                    scalar = float(value.detach().cpu())
                    if not math.isfinite(scalar):
                        raise FloatingPointError(f"{name} is not finite")
                    metric_totals[name] += scalar
            updates += 1

    return {name: total / updates for name, total in metric_totals.items()}
