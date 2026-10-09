from __future__ import annotations

import argparse
from dataclasses import replace

from ppo_slm.config import project_path, select_device
from ppo_slm.outputs import create_run_directory
from ppo_slm.phase_ppo_config import PhasePPOConfig
from ppo_slm.phase_ppo_outputs import save_phase_ppo_result
from ppo_slm.phase_ppo_training import train_phase_ppo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize one 64x64 SLM phase with a paper-style Gaussian "
            "phase-distribution PPO and a Fourier focal-plane simulator."
        )
    )
    parser.add_argument("--target", default="data/1.bmp", help="Target image path")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="results", help="Result directory root")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--quick", action="store_true", help="Short pipeline check")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    return parser


def _config_from_args(args: argparse.Namespace) -> PhasePPOConfig:
    config = PhasePPOConfig.quick() if args.quick else PhasePPOConfig()
    overrides: dict[str, int | float | str] = {
        "device": args.device,
        "seed": args.seed,
    }
    for argument, field_name in (
        (args.rounds, "rounds"),
        (args.population_size, "population_size"),
        (args.sigma, "sigma"),
        (args.learning_rate, "learning_rate"),
        (args.update_epochs, "update_epochs"),
        (args.minibatch_size, "minibatch_size"),
    ):
        if argument is not None:
            overrides[field_name] = argument
    population_size = int(
        overrides.get("population_size", config.population_size)
    )
    if args.minibatch_size is None and population_size < config.minibatch_size:
        overrides["minibatch_size"] = population_size
    return replace(config, **overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    device = select_device(config.device)
    target_path = project_path(args.target)
    output_root = project_path(args.output)
    print(f"Device: {device}")
    print(f"Target: {target_path}")
    print(f"Rounds: {config.rounds}")
    print(
        "Optical evaluations: "
        f"{1 + config.rounds * config.population_size}"
    )
    result = train_phase_ppo(config, target_path)
    run_dir = create_run_directory(output_root, args.run_name)
    save_phase_ppo_result(result, run_dir)
    print(f"Output: {run_dir}")
    print(f"Initial composite loss: {result.initial_loss:.6f}")
    print(f"Best composite loss: {result.best_loss:.6f}")
    print(f"Best NCC: {result.best_ncc:.4f}")
    print(f"Best PSNR: {result.best_psnr_db:.2f} dB")
    print(f"Best round: {result.best_round}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
