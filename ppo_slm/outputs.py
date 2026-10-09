from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .optics import phase_to_uint8
from .training import TrainingResult


def create_run_directory(root: Path, name: str | None) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    base_name = name or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    if not base_name or Path(base_name).name != base_name or base_name in {".", ".."}:
        raise ValueError("run name must be a single non-empty directory name")
    candidate = root / base_name
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base_name}-{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _grayscale_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().squeeze().numpy()
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(pixels)


def _save_comparison(result: TrainingResult, path: Path) -> None:
    target = result.target.squeeze().numpy()
    initial = result.initial_camera.squeeze().numpy()
    best = result.best_camera.squeeze().numpy()
    error = np.abs(target - best)
    phase = result.best_phase.numpy()
    figure, axes = plt.subplots(1, 5, figsize=(20, 4))
    panels = (
        (target, "Target", "gray"),
        (initial, f"Initial camera\nMSE={result.initial_mse:.6f}", "gray"),
        (best, f"Best camera\nMSE={result.best_mse:.6f}", "gray"),
        (error, "Absolute error", "magma"),
        (phase, "Best 8×8 phase", "hsv"),
    )
    for axis, (image, title, color_map) in zip(axes, panels, strict=True):
        axis.imshow(image, cmap=color_map)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_curves(result: TrainingResult, path: Path) -> None:
    steps = np.arange(1, result.steps + 1)
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(steps, result.mse_history, color="tab:blue")
    axes[0].axhline(
        result.config.target_mse, color="tab:red", linestyle="--", label="target MSE"
    )
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(steps, result.reward_history, color="tab:orange")
    axes[1].set_ylabel("Reward")
    axes[1].grid(alpha=0.25)
    axes[2].plot(steps, result.psnr_history, color="tab:green")
    axes[2].set_ylabel("PSNR (dB)")
    axes[2].set_xlabel("Environment step")
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_training_result(result: TrainingResult, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "best_phase.npy", result.best_phase.numpy().astype(np.float32))
    Image.fromarray(phase_to_uint8(result.best_phase)).save(
        run_dir / "slm_8bit.png"
    )
    _grayscale_image(result.best_camera).save(run_dir / "simulated_camera.png")
    _save_comparison(result, run_dir / "comparison.png")
    _save_curves(result, run_dir / "training_curves.png")

    improvement = result.initial_mse - result.best_mse
    improvement_percent = 100.0 * improvement / max(result.initial_mse, 1e-12)
    metrics = {
        "target_path": str(result.target_path),
        "seed": result.config.seed,
        "steps": result.steps,
        "stop_reason": result.stop_reason,
        "initial_mse": result.initial_mse,
        "best_mse": result.best_mse,
        "best_psnr_db": result.best_psnr,
        "best_step": result.best_step,
        "improvement": improvement,
        "improvement_percent": improvement_percent,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "model_state": result.model_state,
            "optimizer_state": result.optimizer_state,
            "config": asdict(result.config),
            "steps": result.steps,
            "stop_reason": result.stop_reason,
            "best_phase": result.best_phase,
            "best_camera": result.best_camera,
            "best_mse": result.best_mse,
            "best_step": result.best_step,
            "target_path": str(result.target_path),
        },
        run_dir / "checkpoint_best.pt",
    )
