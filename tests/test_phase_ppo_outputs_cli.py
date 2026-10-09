from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ppo_slm.outputs import create_run_directory
from ppo_slm.phase_ppo_config import PhasePPOConfig
from ppo_slm.phase_ppo_outputs import save_phase_ppo_result
from ppo_slm.phase_ppo_training import PhasePPOResult
from train_phase_ppo import _config_from_args, build_parser, main


def _result(target_path: Path) -> PhasePPOResult:
    config = replace(
        PhasePPOConfig.quick(),
        phase_size=2,
        image_size=4,
        population_size=4,
        minibatch_size=2,
    )
    target = torch.zeros((1, 4, 4))
    target[:, 1:3, 1:3] = 1.0
    best_phase = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    final_raw = torch.tensor([[7.0, -1.0], [2.5, 9.0]])
    return PhasePPOResult(
        config=config,
        target_path=target_path,
        target=target,
        initial_phase=torch.zeros((2, 2)),
        initial_camera=target * 2.0,
        initial_loss=1.0,
        initial_amplitude_mse=1.1,
        initial_ncc=0.1,
        initial_outside_energy_ratio=0.8,
        initial_intensity_mse=0.5,
        initial_psnr_db=3.01,
        best_phase=best_phase,
        best_camera=target * 4.0,
        best_loss=0.25,
        best_amplitude_mse=0.3,
        best_ncc=0.75,
        best_outside_energy_ratio=0.1,
        best_intensity_mse=0.02,
        best_psnr_db=16.99,
        best_round=2,
        final_mean_raw=final_raw,
        final_mean_phase=torch.remainder(final_raw, 2 * torch.pi),
        total_evaluations=9,
        runtime_seconds=1.25,
        best_loss_history=[0.5, 0.25],
        mean_loss_history=[0.8, 0.6],
        best_ncc_history=[0.5, 0.75],
        mean_outside_energy_history=[0.4, 0.2],
        policy_loss_history=[-0.1, -0.2],
        approx_kl_history=[0.01, 0.02],
        clip_fraction_history=[0.2, 0.3],
        update_metrics={
            "policy_loss": -0.2,
            "approx_kl": 0.02,
            "clip_fraction": 0.3,
            "gradient_norm": 0.4,
        },
        optimizer_state={"state": {}, "param_groups": []},
    )


def test_save_phase_result_writes_exact_artifacts_and_plain_metadata(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "target.bmp"
    target_path.touch()
    run_dir = create_run_directory(tmp_path / "results", "phase")
    result = _result(target_path)

    save_phase_ppo_result(result, run_dir)

    assert {path.name for path in run_dir.iterdir()} == {
        "best_phase.npy",
        "mean_phase.npy",
        "slm_8bit.png",
        "simulated_camera.png",
        "comparison.png",
        "training_curves.png",
        "metrics.json",
        "checkpoint_best.pt",
    }
    assert np.load(run_dir / "best_phase.npy").shape == (2, 2)
    assert np.load(run_dir / "mean_phase.npy").shape == (2, 2)
    assert Image.open(run_dir / "slm_8bit.png").mode == "L"
    assert Image.open(run_dir / "slm_8bit.png").size == (2, 2)
    assert Image.open(run_dir / "simulated_camera.png").mode == "L"
    assert Image.open(run_dir / "simulated_camera.png").size == (4, 4)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["best_round"] == 2
    assert metrics["total_evaluations"] == 9
    assert metrics["runtime_seconds"] == pytest.approx(1.25)
    assert metrics["best"]["total_loss"] == pytest.approx(0.25)
    assert metrics["best"]["ncc"] == pytest.approx(0.75)
    assert metrics["config"]["phase_size"] == 2

    checkpoint = torch.load(
        run_dir / "checkpoint_best.pt", map_location="cpu", weights_only=False
    )
    assert torch.equal(checkpoint["phase_mean_raw"], result.final_mean_raw)
    assert torch.equal(checkpoint["phase_mean_wrapped"], result.final_mean_phase)
    assert not torch.equal(
        checkpoint["phase_mean_raw"], checkpoint["phase_mean_wrapped"]
    )
    assert isinstance(checkpoint["config"], dict)


def test_parser_quick_mode_and_overrides() -> None:
    args = build_parser().parse_args(
        [
            "--target", "data/2.bmp", "--quick", "--device", "cpu",
            "--seed", "19", "--rounds", "3", "--population-size", "2",
            "--sigma", "0.1", "--learning-rate", "0.02",
            "--update-epochs", "2",
        ]
    )

    config = _config_from_args(args)

    assert args.target == "data/2.bmp"
    assert config.device == "cpu"
    assert config.seed == 19
    assert config.rounds == 3
    assert config.population_size == 2
    assert config.minibatch_size == 2
    assert config.sigma == pytest.approx(0.1)
    assert config.learning_rate == pytest.approx(0.02)
    assert config.update_epochs == 2


def test_explicit_invalid_minibatch_override_is_rejected() -> None:
    args = build_parser().parse_args(
        ["--quick", "--population-size", "2", "--minibatch-size", "3"]
    )
    with pytest.raises(ValueError, match="minibatch"):
        _config_from_args(args)


def test_quick_cli_runs_real_target_and_reports_phase_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "--target", "data/1.bmp", "--quick", "--device", "cpu",
            "--output", str(tmp_path), "--run-name", "phase-cli",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Rounds: 2" in output
    assert "Optical evaluations: 17" in output
    assert "Best composite loss:" in output
    assert "Best NCC:" in output
    assert "Output:" in output
    assert (tmp_path / "phase-cli" / "metrics.json").is_file()
