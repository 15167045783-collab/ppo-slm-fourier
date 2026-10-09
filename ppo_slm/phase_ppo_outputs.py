from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .optics import phase_to_uint8
from .phase_ppo_training import PhasePPOResult


def _display_array(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().squeeze().numpy().astype(np.float32)
    maximum = float(array.max())
    if maximum > 0.0:
        array = array / maximum
    return np.clip(array, 0.0, 1.0)


def _grayscale_image(tensor: torch.Tensor) -> Image.Image:
    pixels = np.rint(_display_array(tensor) * 255.0).astype(np.uint8)
    return Image.fromarray(pixels)


def _save_comparison(result: PhasePPOResult, path: Path) -> None:
    target = _display_array(result.target)
    initial = _display_array(result.initial_camera)
    best = _display_array(result.best_camera)
    error = np.abs(target - best)
    phase = result.best_phase.numpy()
    figure, axes = plt.subplots(1, 5, figsize=(20, 4))
    panels = (
        (target, "Target", "gray"),
        (initial, f"Initial camera\nLoss={result.initial_loss:.6f}", "gray"),
        (best, f"Best camera\nLoss={result.best_loss:.6f}", "gray"),
        (error, "Absolute error", "magma"),
        (
            phase,
            f"Best {result.config.phase_size}×{result.config.phase_size} phase",
            "hsv",
        ),
    )
    for axis, (image, title, color_map) in zip(axes, panels, strict=True):
        axis.imshow(image, cmap=color_map)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_curves(result: PhasePPOResult, path: Path) -> None:
    rounds = np.arange(1, result.config.rounds + 1)
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    axes[0, 0].plot(rounds, result.best_loss_history, label="batch best")
    axes[0, 0].plot(rounds, result.mean_loss_history, label="batch mean")
    axes[0, 0].set_ylabel("Composite loss")
    axes[0, 0].legend()
    axes[0, 1].plot(rounds, result.best_ncc_history, color="tab:green")
    axes[0, 1].set_ylabel("Best NCC")
    axes[1, 0].plot(
        rounds, result.mean_outside_energy_history, color="tab:orange"
    )
    axes[1, 0].set_ylabel("Mean outside energy")
    axes[1, 1].plot(rounds, result.policy_loss_history, color="tab:red")
    axes[1, 1].set_ylabel("Policy loss")
    axes[2, 0].plot(rounds, result.approx_kl_history, color="tab:purple")
    axes[2, 0].set_ylabel("Approximate KL")
    axes[2, 1].plot(rounds, result.clip_fraction_history, color="tab:brown")
    axes[2, 1].set_ylabel("Clip fraction")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel("Round")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_phase_ppo_result(result: PhasePPOResult, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        run_dir / "best_phase.npy",
        result.best_phase.numpy().astype(np.float32),
    )
    np.save(
        run_dir / "mean_phase.npy",
        result.final_mean_phase.numpy().astype(np.float32),
    )
    Image.fromarray(phase_to_uint8(result.best_phase)).save(
        run_dir / "slm_8bit.png"
    )
    _grayscale_image(result.best_camera).save(run_dir / "simulated_camera.png")
    _save_comparison(result, run_dir / "comparison.png")
    _save_curves(result, run_dir / "training_curves.png")

    metrics = {
        "target_path": str(result.target_path),
        "config": asdict(result.config),
        "initial": {
            "total_loss": result.initial_loss,
            "amplitude_mse": result.initial_amplitude_mse,
            "ncc": result.initial_ncc,
            "outside_energy_ratio": result.initial_outside_energy_ratio,
            "intensity_mse": result.initial_intensity_mse,
            "psnr_db": result.initial_psnr_db,
        },
        "best": {
            "total_loss": result.best_loss,
            "amplitude_mse": result.best_amplitude_mse,
            "ncc": result.best_ncc,
            "outside_energy_ratio": result.best_outside_energy_ratio,
            "intensity_mse": result.best_intensity_mse,
            "psnr_db": result.best_psnr_db,
        },
        "best_round": result.best_round,
        "total_evaluations": result.total_evaluations,
        "runtime_seconds": result.runtime_seconds,
        "improvement": result.initial_loss - result.best_loss,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    histories = {
        "best_loss": result.best_loss_history,
        "mean_loss": result.mean_loss_history,
        "best_ncc": result.best_ncc_history,
        "mean_outside_energy": result.mean_outside_energy_history,
        "policy_loss": result.policy_loss_history,
        "approx_kl": result.approx_kl_history,
        "clip_fraction": result.clip_fraction_history,
    }
    torch.save(
        {
            "phase_mean_raw": result.final_mean_raw,
            "phase_mean_wrapped": result.final_mean_phase,
            "optimizer_state": result.optimizer_state,
            "config": asdict(result.config),
            "histories": histories,
            "best_phase": result.best_phase,
            "best_camera": result.best_camera,
            "best_loss": result.best_loss,
            "best_round": result.best_round,
            "target_path": str(result.target_path),
        },
        run_dir / "checkpoint_best.pt",
    )
