from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, observation_channels: int, action_size: int) -> None:
        super().__init__()
        if observation_channels <= 0 or action_size <= 0:
            raise ValueError("observation_channels and action_size must be positive")
        self.encoder = nn.Sequential(
            nn.Conv2d(observation_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 4:
            raise ValueError("observations must have shape [B, C, H, W]")
        features = self.encoder(observations)
        means = self.actor_mean(features)
        values = self.critic(features).squeeze(-1)
        return means, values

    def distribution(self, observations: torch.Tensor) -> Normal:
        means, _ = self(observations)
        standard_deviation = self.log_std.exp().expand_as(means)
        return Normal(means, standard_deviation)

    def distribution_and_value(
        self, observations: torch.Tensor
    ) -> tuple[Normal, torch.Tensor]:
        means, values = self(observations)
        standard_deviation = self.log_std.exp().expand_as(means)
        return Normal(means, standard_deviation), values

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self.distribution_and_value(observations)
        log_probabilities = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probabilities, entropy, values
