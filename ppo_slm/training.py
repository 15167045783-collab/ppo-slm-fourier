from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import PPOConfig, seed_everything, select_device
from .data import load_target
from .environment import SLMEnvironment
from .networks import ActorCritic
from .ppo import PPOAgent, RolloutBuffer


@dataclass(frozen=True)
class TrainingResult:
    config: PPOConfig
    target_path: Path
    target: torch.Tensor
    initial_phase: torch.Tensor
    initial_camera: torch.Tensor
    initial_mse: float
    best_phase: torch.Tensor
    best_camera: torch.Tensor
    best_mse: float
    best_psnr: float
    best_step: int
    steps: int
    stop_reason: str
    mse_history: list[float]
    reward_history: list[float]
    psnr_history: list[float]
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]


def _bootstrap_value(agent: PPOAgent, observation: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        batch = observation.unsqueeze(0).to(agent.device)
        return agent.model(batch)[1][0].detach().cpu()


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def train(config: PPOConfig, target_path: Path) -> TrainingResult:
    seed_everything(config.seed)
    device = select_device(config.device)
    target_path = Path(target_path).resolve()
    target = load_target(target_path, config.image_size)
    environment = SLMEnvironment(target, config, device)
    model = ActorCritic(observation_channels=3, action_size=config.action_size)
    agent = PPOAgent(model, config, device)
    buffer = RolloutBuffer()

    observation = environment.reset(seed=config.seed)
    initial_phase = environment.phase.cpu()
    initial_camera = environment.camera.cpu()
    initial_mse = environment.mse
    best_phase = initial_phase.clone()
    best_camera = initial_camera.clone()
    best_mse = initial_mse
    best_psnr = -10.0 * math.log10(max(initial_mse, 1e-12))
    best_step = 0
    mse_history: list[float] = []
    reward_history: list[float] = []
    psnr_history: list[float] = []
    stop_reason = "target_mse" if environment.reached_target else "total_steps"
    completed_steps = 0

    if not environment.reached_target:
        for step_number in range(1, config.total_steps + 1):
            policy_step = agent.act(observation)
            next_observation, reward, done, info = environment.step(policy_step.action)
            buffer.add(
                observation=observation,
                action=policy_step.action,
                log_prob=policy_step.log_prob,
                value=policy_step.value,
                reward=reward,
                done=done,
            )
            current_mse = float(info["mse"])
            current_psnr = float(info["psnr"])
            mse_history.append(current_mse)
            reward_history.append(float(reward))
            psnr_history.append(current_psnr)
            completed_steps = step_number

            if current_mse < best_mse:
                best_mse = current_mse
                best_psnr = current_psnr
                best_step = step_number
                best_phase = environment.phase.cpu()
                best_camera = environment.camera.cpu()

            reached_target = bool(info["reached_target"])
            rollout_ends = (
                len(buffer) >= config.rollout_steps
                or step_number == config.total_steps
                or reached_target
            )
            if rollout_ends:
                last_value = (
                    torch.tensor(0.0)
                    if done
                    else _bootstrap_value(agent, next_observation)
                )
                buffer.finish(last_value, config.gamma, config.gae_lambda)
                agent.update(buffer)
                buffer.clear()

            observation = next_observation
            if reached_target:
                stop_reason = "target_mse"
                break
            if done:
                observation = environment.reset()

    model_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in agent.model.state_dict().items()
    }
    return TrainingResult(
        config=config,
        target_path=target_path,
        target=target.detach().cpu().clone(),
        initial_phase=initial_phase,
        initial_camera=initial_camera,
        initial_mse=initial_mse,
        best_phase=best_phase,
        best_camera=best_camera,
        best_mse=best_mse,
        best_psnr=best_psnr,
        best_step=best_step,
        steps=completed_steps,
        stop_reason=stop_reason,
        mse_history=mse_history,
        reward_history=reward_history,
        psnr_history=psnr_history,
        model_state=model_state,
        optimizer_state=_to_cpu(agent.optimizer.state_dict()),
    )
