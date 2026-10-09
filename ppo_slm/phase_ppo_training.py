from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import seed_everything, select_device
from .data import load_target
from .optics import fourier_intensity_batch
from .phase_distribution import (
    PhaseDistributionPolicy,
    standardize_advantages,
    update_phase_policy,
)
from .phase_objective import PhaseObjectiveBatch, evaluate_phase_objective
from .phase_ppo_config import PhasePPOConfig


@dataclass(frozen=True)
class PhasePPOResult:
    config: PhasePPOConfig
    target_path: Path
    target: torch.Tensor
    initial_phase: torch.Tensor
    initial_camera: torch.Tensor
    initial_loss: float
    initial_amplitude_mse: float
    initial_ncc: float
    initial_outside_energy_ratio: float
    initial_intensity_mse: float
    initial_psnr_db: float
    best_phase: torch.Tensor
    best_camera: torch.Tensor
    best_loss: float
    best_amplitude_mse: float
    best_ncc: float
    best_outside_energy_ratio: float
    best_intensity_mse: float
    best_psnr_db: float
    best_round: int
    final_mean_raw: torch.Tensor
    final_mean_phase: torch.Tensor
    total_evaluations: int
    runtime_seconds: float
    best_loss_history: list[float]
    mean_loss_history: list[float]
    best_ncc_history: list[float]
    mean_outside_energy_history: list[float]
    policy_loss_history: list[float]
    approx_kl_history: list[float]
    clip_fraction_history: list[float]
    update_metrics: dict[str, float]
    optimizer_state: dict[str, Any]


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _cpu_clone(value)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _metric(batch: PhaseObjectiveBatch, name: str, index: int = 0) -> float:
    value = float(getattr(batch, name)[index].detach().cpu())
    if not math.isfinite(value):
        raise FloatingPointError(f"{name} is not finite")
    return value


def _capture(
    phases: torch.Tensor,
    target: torch.Tensor,
    config: PhasePPOConfig,
) -> tuple[torch.Tensor, PhaseObjectiveBatch]:
    try:
        cameras = fourier_intensity_batch(phases, config.image_size)
        objective = evaluate_phase_objective(cameras, target, config)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA ran out of memory during batched optical propagation; "
            "retry with a lower --population-size"
        ) from exc
    return cameras, objective


def train_phase_ppo(
    config: PhasePPOConfig,
    target_path: Path,
) -> PhasePPOResult:
    started_at = time.perf_counter()
    seed_everything(config.seed)
    device = select_device(config.device)
    target_path = Path(target_path).resolve()
    target_cpu = load_target(target_path, config.image_size)
    target = target_cpu.to(device)
    policy = PhaseDistributionPolicy(config, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    with torch.no_grad():
        initial_phase_device = torch.remainder(policy.mean, 2.0 * math.pi)
    initial_cameras, initial_objective = _capture(
        initial_phase_device.unsqueeze(0), target, config
    )
    initial_loss = _metric(initial_objective, "total_loss")
    initial_amplitude_mse = _metric(initial_objective, "amplitude_mse")
    initial_ncc = _metric(initial_objective, "ncc")
    initial_outside = _metric(initial_objective, "outside_energy_ratio")
    initial_intensity_mse = _metric(initial_objective, "intensity_mse")
    initial_psnr = _metric(initial_objective, "psnr_db")

    best_phase = _cpu_clone(initial_phase_device)
    best_camera = _cpu_clone(initial_cameras[0])
    best_loss = initial_loss
    best_amplitude_mse = initial_amplitude_mse
    best_ncc = initial_ncc
    best_outside = initial_outside
    best_intensity_mse = initial_intensity_mse
    best_psnr = initial_psnr
    best_round = 0

    best_loss_history: list[float] = []
    mean_loss_history: list[float] = []
    best_ncc_history: list[float] = []
    mean_outside_history: list[float] = []
    policy_loss_history: list[float] = []
    approx_kl_history: list[float] = []
    clip_fraction_history: list[float] = []
    update_metrics: dict[str, float] = {}

    for round_number in range(1, config.rounds + 1):
        samples = policy.sample(config.population_size)
        cameras, objective = _capture(samples.display, target, config)
        round_best_index = int(objective.total_loss.argmin().item())
        round_best_loss = _metric(objective, "total_loss", round_best_index)

        best_loss_history.append(round_best_loss)
        mean_loss_history.append(float(objective.total_loss.mean().cpu()))
        best_ncc_history.append(float(objective.ncc.max().cpu()))
        mean_outside_history.append(float(objective.outside_energy_ratio.mean().cpu()))

        if round_best_loss < best_loss:
            best_loss = round_best_loss
            best_phase = _cpu_clone(samples.display[round_best_index])
            best_camera = _cpu_clone(cameras[round_best_index])
            best_amplitude_mse = _metric(
                objective, "amplitude_mse", round_best_index
            )
            best_ncc = _metric(objective, "ncc", round_best_index)
            best_outside = _metric(
                objective, "outside_energy_ratio", round_best_index
            )
            best_intensity_mse = _metric(
                objective, "intensity_mse", round_best_index
            )
            best_psnr = _metric(objective, "psnr_db", round_best_index)
            best_round = round_number

        rewards = -objective.total_loss.detach()
        advantages = standardize_advantages(rewards)
        update_metrics = update_phase_policy(
            policy, optimizer, samples, advantages, config
        )
        if not torch.isfinite(policy.mean).all():
            raise FloatingPointError("phase policy mean became NaN or Inf")
        policy_loss_history.append(update_metrics["policy_loss"])
        approx_kl_history.append(update_metrics["approx_kl"])
        clip_fraction_history.append(update_metrics["clip_fraction"])

    final_mean_raw = _cpu_clone(policy.mean)
    final_mean_phase = torch.remainder(final_mean_raw, 2.0 * math.pi)
    runtime_seconds = time.perf_counter() - started_at
    return PhasePPOResult(
        config=config,
        target_path=target_path,
        target=_cpu_clone(target_cpu),
        initial_phase=_cpu_clone(initial_phase_device),
        initial_camera=_cpu_clone(initial_cameras[0]),
        initial_loss=initial_loss,
        initial_amplitude_mse=initial_amplitude_mse,
        initial_ncc=initial_ncc,
        initial_outside_energy_ratio=initial_outside,
        initial_intensity_mse=initial_intensity_mse,
        initial_psnr_db=initial_psnr,
        best_phase=best_phase,
        best_camera=best_camera,
        best_loss=best_loss,
        best_amplitude_mse=best_amplitude_mse,
        best_ncc=best_ncc,
        best_outside_energy_ratio=best_outside,
        best_intensity_mse=best_intensity_mse,
        best_psnr_db=best_psnr,
        best_round=best_round,
        final_mean_raw=final_mean_raw,
        final_mean_phase=final_mean_phase,
        total_evaluations=1 + config.rounds * config.population_size,
        runtime_seconds=runtime_seconds,
        best_loss_history=best_loss_history,
        mean_loss_history=mean_loss_history,
        best_ncc_history=best_ncc_history,
        mean_outside_energy_history=mean_outside_history,
        policy_loss_history=policy_loss_history,
        approx_kl_history=approx_kl_history,
        clip_fraction_history=clip_fraction_history,
        update_metrics=copy.deepcopy(update_metrics),
        optimizer_state=_to_cpu(optimizer.state_dict()),
    )
