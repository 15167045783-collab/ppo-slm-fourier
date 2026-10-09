from __future__ import annotations

import pytest
import torch

from ppo_slm.config import PPOConfig, seed_everything
from ppo_slm.networks import ActorCritic
from ppo_slm.ppo import PPOAgent, RolloutBuffer, compute_gae


def test_network_outputs_actor_and_critic_shapes() -> None:
    model = ActorCritic(observation_channels=3, action_size=4096)
    observations = torch.rand((2, 3, 256, 256))

    means, values = model(observations)
    distribution = model.distribution(observations)

    assert means.shape == (2, 4096)
    assert values.shape == (2,)
    assert distribution.mean.shape == (2, 4096)
    assert model.log_std.shape == (4096,)
    assert torch.isfinite(means).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(model.log_std).all()


def test_agent_act_returns_one_action_log_probability_and_value() -> None:
    agent = PPOAgent(
        ActorCritic(3, 4096), PPOConfig.quick(), torch.device("cpu")
    )
    observation = torch.rand((3, 256, 256))

    sampled = agent.act(observation)
    deterministic = agent.act(observation, deterministic=True)

    assert sampled.action.shape == (4096,)
    assert sampled.log_prob.shape == ()
    assert sampled.value.shape == ()
    assert deterministic.action.shape == (4096,)
    assert torch.isfinite(sampled.action).all()


def test_agent_act_rejects_malformed_observation() -> None:
    agent = PPOAgent(
        ActorCritic(3, 4096), PPOConfig.quick(), torch.device("cpu")
    )
    with pytest.raises(ValueError, match="observation"):
        agent.act(torch.rand((1, 3, 256, 256)))


def test_high_dimensional_policy_normalizes_log_probability_and_entropy() -> None:
    model = ActorCritic(3, 4096)
    observation = torch.rand((1, 3, 256, 256))
    actions = torch.zeros((1, 4096))

    log_probabilities, entropy, _ = model.evaluate_actions(observation, actions)
    distribution = model.distribution(observation)

    assert torch.allclose(
        log_probabilities, distribution.log_prob(actions).mean(dim=-1)
    )
    assert torch.allclose(entropy, distribution.entropy().mean(dim=-1))


def test_compute_gae_matches_hand_calculated_terminal_example() -> None:
    advantages, returns = compute_gae(
        rewards=torch.tensor([1.0, 1.0]),
        values=torch.tensor([0.5, 0.25]),
        dones=torch.tensor([0.0, 1.0]),
        last_value=torch.tensor(0.75),
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.allclose(advantages, torch.tensor([1.5, 0.75]))
    assert torch.allclose(returns, torch.tensor([2.0, 1.0]))


def test_rollout_rejects_nonfinite_transition_and_empty_finish() -> None:
    buffer = RolloutBuffer()
    with pytest.raises(ValueError, match="empty"):
        buffer.finish(torch.tensor(0.0), 0.99, 0.95)
    with pytest.raises(ValueError, match="finite"):
        buffer.add(
            torch.full((3, 256, 256), float("nan")),
            torch.zeros(4096),
            torch.tensor(0.0),
            torch.tensor(0.0),
            0.0,
            False,
        )


def test_ppo_update_changes_model_parameters() -> None:
    seed_everything(5)
    config = PPOConfig.quick()
    model = ActorCritic(3, config.action_size)
    agent = PPOAgent(model, config, torch.device("cpu"))
    buffer = RolloutBuffer()
    for index in range(4):
        observation = torch.rand((3, 256, 256))
        step = agent.act(observation)
        buffer.add(
            observation,
            step.action,
            step.log_prob,
            step.value,
            reward=float(index + 1),
            done=index == 3,
        )
    buffer.finish(torch.tensor(0.0), config.gamma, config.gae_lambda)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = agent.update(buffer)

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )
    assert set(metrics) == {"policy_loss", "value_loss", "entropy", "total_loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_ppo_update_rejects_unfinished_rollout() -> None:
    config = PPOConfig.quick()
    agent = PPOAgent(ActorCritic(3, 4096), config, torch.device("cpu"))
    with pytest.raises(ValueError, match="finished"):
        agent.update(RolloutBuffer())
