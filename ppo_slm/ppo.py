from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .config import PPOConfig
from .networks import ActorCritic


@dataclass(frozen=True)
class PolicyStep:
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor


class RolloutBuffer:
    def __init__(self) -> None:
        self.observations: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        tensors = (observation, action, log_prob, value)
        if not all(torch.isfinite(item).all() for item in tensors) or not math.isfinite(
            reward
        ):
            raise ValueError("rollout transition must be finite")
        self.observations.append(observation.detach().cpu().clone())
        self.actions.append(action.detach().cpu().clone())
        self.log_probs.append(log_prob.detach().cpu().reshape(()).clone())
        self.values.append(value.detach().cpu().reshape(()).clone())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def finish(self, last_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        if not self.rewards:
            raise ValueError("cannot finish an empty rollout")
        self.advantages, self.returns = compute_gae(
            rewards=torch.tensor(self.rewards, dtype=torch.float32),
            values=torch.stack(self.values).float(),
            dones=torch.tensor(self.dones, dtype=torch.float32),
            last_value=last_value.detach().cpu().float().reshape(()),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    def clear(self) -> None:
        self.__init__()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.ndim != 1 or values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("rewards, values, and dones must be equal-length vectors")
    if rewards.numel() == 0:
        raise ValueError("GAE inputs cannot be empty")
    if not all(
        torch.isfinite(item).all() for item in (rewards, values, dones, last_value)
    ):
        raise ValueError("GAE inputs must be finite")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, dtype=rewards.dtype)
    next_value = last_value.to(dtype=rewards.dtype)
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
        next_value = values[index]
    returns = advantages + values
    return advantages, returns


class PPOAgent:
    def __init__(
        self, model: ActorCritic, config: PPOConfig, device: torch.device
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )

    def act(self, observation: torch.Tensor, deterministic: bool = False) -> PolicyStep:
        if observation.ndim != 3:
            raise ValueError("observation must have shape [C, H, W]")
        batch = observation.unsqueeze(0).to(self.device)
        with torch.no_grad():
            distribution, value = self.model.distribution_and_value(batch)
            action = distribution.mean if deterministic else distribution.sample()
            log_probability = distribution.log_prob(action).sum(dim=-1)
        return PolicyStep(
            action=action[0].cpu(),
            log_prob=log_probability[0].cpu(),
            value=value[0].cpu(),
        )

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        if buffer.advantages is None or buffer.returns is None:
            raise ValueError("rollout must be finished before PPO update")
        observations = torch.stack(buffer.observations).to(self.device)
        actions = torch.stack(buffer.actions).to(self.device)
        old_log_probs = torch.stack(buffer.log_probs).to(self.device)
        returns = buffer.returns.to(self.device)
        advantages = buffer.advantages.to(self.device)
        advantage_std = advantages.std(unbiased=False)
        advantages = (advantages - advantages.mean()) / (advantage_std + 1e-8)
        if not all(
            torch.isfinite(item).all()
            for item in (observations, actions, old_log_probs, returns, advantages)
        ):
            raise ValueError("PPO batch contains NaN or Inf")

        totals = {
            name: 0.0
            for name in ("policy_loss", "value_loss", "entropy", "total_loss")
        }
        updates = 0
        count = len(buffer)
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(count, device=self.device)
            for indices in permutation.split(self.config.minibatch_size):
                new_log_probs, entropy, values = self.model.evaluate_actions(
                    observations[indices], actions[indices]
                )
                ratios = (new_log_probs - old_log_probs[indices]).exp()
                unclipped = ratios * advantages[indices]
                clipped = ratios.clamp(
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = functional.mse_loss(values, returns[indices])
                entropy_mean = entropy.mean()
                total_loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy_mean
                )
                if not torch.isfinite(total_loss):
                    raise ValueError("PPO loss is NaN or Inf")
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                for name, value in zip(
                    totals,
                    (policy_loss, value_loss, entropy_mean, total_loss),
                    strict=True,
                ):
                    totals[name] += float(value.detach().cpu().item())
                updates += 1
        if updates == 0:
            raise ValueError("PPO update received no minibatches")
        return {name: total / updates for name, total in totals.items()}
