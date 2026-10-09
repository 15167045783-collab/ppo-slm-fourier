from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ppo_slm.config import PPOConfig
from ppo_slm.outputs import create_run_directory, save_training_result
from ppo_slm.training import TrainingResult
from train import _config_from_args, build_parser, main


def _result(target_path: Path) -> TrainingResult:
    config = PPOConfig.quick()
    target = torch.zeros((1, 256, 256))
    target[:, 100:156, 100:156] = 1.0
    phase = torch.linspace(0.0, 6.0, 64).reshape(8, 8)
    camera = target * 0.9
    return TrainingResult(
        config=config,
        target_path=target_path,
        target=target,
        initial_phase=torch.zeros((8, 8)),
        initial_camera=target * 0.5,
        initial_mse=0.05,
        best_phase=phase,
        best_camera=camera,
        best_mse=0.01,
        best_psnr=20.0,
        best_step=2,
        steps=3,
        stop_reason="total_steps",
        mse_history=[0.05, 0.02, 0.01],
        reward_history=[-0.1, 2.9, 0.9],
        psnr_history=[13.01, 16.99, 20.0],
        model_state={},
        optimizer_state={},
    )


def test_create_run_directory_never_overwrites_existing_run(tmp_path: Path) -> None:
    first = create_run_directory(tmp_path, "experiment")
    second = create_run_directory(tmp_path, "experiment")

    assert first.name == "experiment"
    assert second.name == "experiment-2"
    assert first.is_dir()
    assert second.is_dir()


def test_create_run_directory_rejects_path_like_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run name"):
        create_run_directory(tmp_path, "../outside")


def test_save_training_result_writes_all_required_artifacts(tmp_path: Path) -> None:
    target_path = tmp_path / "target.bmp"
    target_path.touch()
    run_dir = create_run_directory(tmp_path / "results", "save-test")

    save_training_result(_result(target_path), run_dir)

    expected = {
        "best_phase.npy",
        "slm_8bit.png",
        "simulated_camera.png",
        "comparison.png",
        "training_curves.png",
        "checkpoint_best.pt",
        "metrics.json",
    }
    assert {path.name for path in run_dir.iterdir()} == expected
    assert np.load(run_dir / "best_phase.npy").shape == (8, 8)
    assert Image.open(run_dir / "slm_8bit.png").size == (8, 8)
    assert Image.open(run_dir / "simulated_camera.png").size == (256, 256)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["stop_reason"] == "total_steps"
    assert metrics["best_step"] == 2
    assert metrics["improvement_percent"] == pytest.approx(80.0)
    checkpoint = torch.load(
        run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["config"]["phase_size"] == 8
    assert checkpoint["best_phase"].shape == (8, 8)


def test_parser_and_quick_config_apply_overrides() -> None:
    args = build_parser().parse_args(
        [
            "--target", "data/2.bmp", "--quick", "--seed", "19",
            "--target-mse", "0.02", "--episode-length", "4",
        ]
    )

    config = _config_from_args(args)

    assert args.target == "data/2.bmp"
    assert config.total_steps == 16
    assert config.seed == 19
    assert config.target_mse == pytest.approx(0.02)
    assert config.episode_length == 4


def test_quick_cli_runs_real_target_and_reports_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "--target", "data/1.bmp", "--quick", "--device", "cpu",
            "--output", str(tmp_path), "--run-name", "cli-test",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "Initial MSE:" in captured
    assert "Best MSE:" in captured
    assert "Output:" in captured
    assert (tmp_path / "cli-test" / "metrics.json").is_file()
