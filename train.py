from __future__ import annotations

import argparse
from dataclasses import replace

from ppo_slm.config import PPOConfig, project_path, select_device
from ppo_slm.outputs import create_run_directory, save_training_result
from ppo_slm.training import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO to optimize a 64x64 phase-only SLM through a simulated "
            "Fourier focal-plane optical path."
        )
    )
    parser.add_argument("--target", default="data/1.bmp", help="Target image path")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="results", help="Result directory root")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--quick", action="store_true", help="Short pipeline check")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--target-mse", type=float, default=None)
    return parser


def _config_from_args(args: argparse.Namespace) -> PPOConfig:
    config = PPOConfig.quick() if args.quick else PPOConfig()
    overrides: dict[str, int | float | str] = {
        "device": args.device,
        "seed": args.seed,
    }
    for argument, field_name in (
        (args.total_steps, "total_steps"),
        (args.rollout_steps, "rollout_steps"),
        (args.update_epochs, "update_epochs"),
        (args.minibatch_size, "minibatch_size"),
        (args.episode_length, "episode_length"),
        (args.target_mse, "target_mse"),
    ):
        if argument is not None:
            overrides[field_name] = argument
    total_steps = int(overrides.get("total_steps", config.total_steps))
    if args.rollout_steps is None and total_steps < config.rollout_steps:
        overrides["rollout_steps"] = total_steps
    rollout_steps = int(overrides.get("rollout_steps", config.rollout_steps))
    if args.minibatch_size is None and rollout_steps < config.minibatch_size:
        overrides["minibatch_size"] = rollout_steps
    return replace(config, **overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    device = select_device(config.device)
    target_path = project_path(args.target)
    output_root = project_path(args.output)
    print(f"Device: {device}")
    print(f"Target: {target_path}")
    print(f"Training steps: up to {config.total_steps}")
    result = train(config, target_path)
    run_dir = create_run_directory(output_root, args.run_name)
    save_training_result(result, run_dir)
    improvement = result.initial_mse - result.best_mse
    percentage = 100.0 * improvement / max(result.initial_mse, 1e-12)
    print(f"Output: {run_dir}")
    print(f"Initial weighted MSE: {result.initial_mse:.6f}")
    print(f"Best weighted MSE: {result.best_mse:.6f}")
    print(f"Best PSNR: {result.best_psnr:.2f} dB")
    print(f"Improvement: {percentage:.2f}%")
    print(f"Stopped by: {result.stop_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
