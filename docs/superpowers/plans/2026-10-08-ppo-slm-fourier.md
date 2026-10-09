# PPO SLM Fourier Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent `PPO` project that trains PPO from a random `64×64` phase map to reproduce one selected `256×256` target through a simulated Fourier focal-plane optical path.

**Architecture:** Keep image loading, detached optical simulation, RL environment, Actor-Critic/PPO, training orchestration, and artifact output in separate modules. The environment exposes a three-channel observation and 4096-dimensional continuous phase increments; training tracks the best observed phase globally and stops on either the requested foreground-weighted MSE or a finite step budget.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Pillow, Matplotlib, pytest

**Spec:** `PPO/docs/superpowers/specs/2026-10-08-ppo-slm-fourier-design.md`

## Global Constraints

- Do not modify `moni` or any file under `PPO/data`.
- The controlled phase is exactly `64×64`; optical propagation and observations are exactly `256×256`.
- Phase expansion is nearest-neighbor and phase values are wrapped into `[0, 2π)`.
- Each invocation trains against exactly one selected target image.
- The default run and quick run both have finite step limits; reaching `target_mse` stops training early.
- Optical captures and rewards are detached from autograd; PPO gradients update only the Actor-Critic.
- All saved and resolved paths must remain portable when the whole `PPO` directory is copied to Linux.

## Review Focus

- All-black or unreadable targets must fail before training; covered in Task 2 data tests.
- Phase values at `2π`, negative values, NaN, and Inf must not leak invalid SLM/camera output; covered in Task 3 optics tests.
- A zero action and a very large action must preserve phase shape/range and use the exact reward formula; covered in Task 4 environment tests.
- A partial final rollout and a rollout ending at an episode boundary must compute bootstrap values correctly; covered in Task 6 training tests.
- Existing output directories and unavailable explicit CUDA selections must never be silently reused/fallback; covered in Tasks 1 and 7 tests.

---

### Task 1: Project Configuration and Device Selection

**Files:**
- Create: `PPO/ppo_slm/__init__.py`
- Create: `PPO/ppo_slm/config.py`
- Create: `PPO/tests/conftest.py`
- Create: `PPO/tests/test_config.py`

**Interfaces:**
- Produces: immutable `PPOConfig`, `PPOConfig.quick()`, `select_device(requested: str) -> torch.device`, `project_path(value: str | Path) -> Path`, and `seed_everything(seed: int) -> None`.
- Consumes: no project-local interfaces.

- [ ] **Step 1: Write failing configuration tests**

Assert exact geometry (`phase_size=64`, `image_size=256`, `action_size=4096`), positive/coherent PPO limits, quick-mode reductions, project-relative path resolution, repeatable random seeding, automatic device selection, and an error for explicit CUDA when unavailable.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL because `ppo_slm.config` does not exist.

- [ ] **Step 3: Implement the immutable configuration API**

Use defaults: `action_scale=π/16`, `foreground_weight=8.0`, `total_steps=20_000`, `rollout_steps=256`, `episode_length=256`, `target_mse=0.005`, `learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_ratio=0.2`, `update_epochs=10`, `minibatch_size=64`, `value_coefficient=0.5`, `entropy_coefficient=0.01`, `max_grad_norm=0.5`, `improvement_gain=100.0`, `mse_weight=1.0`, `action_penalty=0.001`, and `seed=7`. Quick mode uses 16 total steps, rollout/episode length 8, one update epoch, and minibatches of 4.

- [ ] **Step 4: Run configuration tests and verify GREEN**

Run: `python -m pytest tests/test_config.py -v`

Expected: all Task 1 tests PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: add PPO SLM configuration`

### Task 2: Target Image Loading

**Files:**
- Create: `PPO/ppo_slm/data.py`
- Create: `PPO/tests/test_data.py`

**Interfaces:**
- Consumes: `PPOConfig.image_size` from Task 1.
- Produces: `load_target(path: Path, image_size: int = 256) -> torch.Tensor` returning finite `[1, H, W]` float32 data normalized to `[0, 1]` with maximum 1.

- [ ] **Step 1: Write failing image-loading tests**

Create temporary RGB and grayscale images and assert output shape/type/range/max normalization. Assert `FileNotFoundError` for missing paths and `ValueError` for corrupt or all-black files.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_data.py -v`

Expected: FAIL because `load_target` is missing.

- [ ] **Step 3: Implement `load_target` with Pillow**

Convert to `L`, resize to the configured square only when necessary using bilinear resampling, convert to float32, reject non-finite/all-black content, and divide by its maximum.

- [ ] **Step 4: Run image-loading tests and verify GREEN**

Run: `python -m pytest tests/test_data.py -v`

Expected: all Task 2 tests PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: load normalized target images`

### Task 3: Fourier Optical Model and SLM Conversion

**Files:**
- Create: `PPO/ppo_slm/optics.py`
- Create: `PPO/tests/test_optics.py`

**Interfaces:**
- Produces: `upsample_phase(phase: torch.Tensor, output_size: int = 256) -> torch.Tensor`, `fourier_intensity(phase: torch.Tensor, output_size: int = 256) -> torch.Tensor`, and `phase_to_uint8(phase: torch.Tensor) -> np.ndarray`.
- Consumes: no project-local interfaces.

- [ ] **Step 1: Write failing optical behavior tests**

Assert `64×64 → 256×256` nearest-neighbor expansion, macro-pixel preservation, FFT output `[1,256,256]`/finite/`[0,1]`/detached, and uint8 conversion shape and phase wrapping. Reject wrong dimensions, non-floating phase, NaN/Inf, nonpositive output sizes, and output sizes smaller than the phase map.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_optics.py -v`

Expected: FAIL because the optical functions are missing.

- [ ] **Step 3: Implement the optical functions**

Use `torch.nn.functional.interpolate(..., mode="nearest")`, `torch.polar`, orthonormal `fft2`, `fftshift`, squared magnitude, and maximum normalization. Run propagation under `torch.no_grad()` and return a cloned detached tensor.

- [ ] **Step 4: Run optical tests and verify GREEN**

Run: `python -m pytest tests/test_optics.py -v`

Expected: all Task 3 tests PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: simulate Fourier SLM optics`

### Task 4: Reinforcement-Learning Environment

**Files:**
- Create: `PPO/ppo_slm/environment.py`
- Create: `PPO/tests/test_environment.py`

**Interfaces:**
- Consumes: `PPOConfig`, `fourier_intensity`, and `upsample_phase`.
- Produces: `SLMEnvironment(target: Tensor, config: PPOConfig, device: torch.device)`, properties `phase`, `camera`, `mse`, `reached_target`, `steps`, plus `reset(seed: int | None = None) -> Tensor` and `step(action: Tensor) -> tuple[Tensor, float, bool, dict[str, float | bool]]`.

- [ ] **Step 1: Write failing reset tests**

Assert deterministic seeded reset, different default resets, `[0,2π)` phase range, `[3,256,256]` detached observation range/type, and strict validation of target shape/type/range/finiteness.

- [ ] **Step 2: Run reset tests and verify RED**

Run: `python -m pytest tests/test_environment.py -k reset -v`

Expected: FAIL because `SLMEnvironment` does not exist.

- [ ] **Step 3: Implement reset and observation construction**

Keep internal phase/camera private; expose defensive clones. Use CPU generator seeds and then move the phase to the selected device.

- [ ] **Step 4: Run reset tests and verify GREEN**

Run: `python -m pytest tests/test_environment.py -k reset -v`

Expected: reset tests PASS.

- [ ] **Step 5: Write failing step/reward/termination tests**

Assert exact phase update/wrapping, zero and huge action behavior, exact reward equation from the returned metrics, action validation, target-threshold termination, and episode-length termination.

- [ ] **Step 6: Run step tests and verify RED**

Run: `python -m pytest tests/test_environment.py -k 'step or reward or terminates' -v`

Expected: FAIL because step behavior is absent.

- [ ] **Step 7: Implement `step` and termination metrics**

Return at least `mse`, `previous_mse`, `improvement`, `action_penalty`, `psnr`, and `reached_target`; calculate PSNR as `-10*log10(max(mse, 1e-12))`.

- [ ] **Step 8: Run all environment tests and verify GREEN**

Run: `python -m pytest tests/test_environment.py -v`

Expected: all Task 4 tests PASS.

- [ ] **Step 9: Commit**

Commit message: `feat: add iterative SLM environment`

### Task 5: Actor-Critic and PPO Update

**Files:**
- Create: `PPO/ppo_slm/networks.py`
- Create: `PPO/ppo_slm/ppo.py`
- Create: `PPO/tests/test_ppo.py`

**Interfaces:**
- Produces: `ActorCritic(observation_channels: int, action_size: int)`, `PolicyStep`, `RolloutBuffer`, `compute_gae(...)`, and `PPOAgent.act/update`.
- Consumes: `PPOConfig` from Task 1.

- [ ] **Step 1: Write failing network and action tests**

Assert batched network output shapes, finite learned log standard deviation, sampled/deterministic 4096-value actions, scalar mean log probability/value, and rejection of malformed observations.

- [ ] **Step 2: Run network tests and verify RED**

Run: `python -m pytest tests/test_ppo.py -k 'network or act' -v`

Expected: FAIL because the network and agent are missing.

- [ ] **Step 3: Implement the shared convolutional Actor-Critic and `PPOAgent.act`**

Use three strided convolutions, adaptive pooling, a shared dense feature, separate mean/value heads, and one learnable action-sized `log_std` parameter.

- [ ] **Step 4: Run network tests and verify GREEN**

Run: `python -m pytest tests/test_ppo.py -k 'network or act' -v`

Expected: network/action tests PASS.

- [ ] **Step 5: Write failing GAE, buffer, and PPO update tests**

Pin a small hand-calculated GAE example, terminal bootstrap behavior, empty/unfinished buffer errors, non-finite transition rejection, and verify at least one model parameter changes after an update.

- [ ] **Step 6: Run PPO update tests and verify RED**

Run: `python -m pytest tests/test_ppo.py -k 'gae or buffer or update' -v`

Expected: FAIL because rollout completion/update behavior is missing.

- [ ] **Step 7: Implement rollout storage, GAE, and clipped PPO update**

Normalize advantages, iterate random minibatches for configured epochs, apply clipped policy loss/value loss/entropy term, reject non-finite batches/loss, and clip gradient norm.

- [ ] **Step 8: Run all PPO tests and verify GREEN**

Run: `python -m pytest tests/test_ppo.py -v`

Expected: all Task 5 tests PASS.

- [ ] **Step 9: Commit**

Commit message: `feat: implement actor critic PPO`

### Task 6: Training Orchestration and Best-State Tracking

**Files:**
- Create: `PPO/ppo_slm/training.py`
- Create: `PPO/tests/test_training.py`

**Interfaces:**
- Consumes: configuration, target loading, environment, Actor-Critic, and PPO agent from Tasks 1–5.
- Produces: immutable `TrainingResult` and `train(config: PPOConfig, target_path: Path) -> TrainingResult`.

- [ ] **Step 1: Write failing training-loop tests**

Using quick configuration and a temporary valid target, assert finite histories of equal length, exact step cap, recorded initial state, global best MSE/phase/camera consistency, finite PSNR, deterministic reruns, partial final rollout handling, and stop reason `target_mse` versus `total_steps`.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_training.py -v`

Expected: FAIL because `train` is missing.

- [ ] **Step 3: Implement deterministic training and result collection**

Reset only after episode-length termination; stop the entire run when `reached_target` is true. Finish/update both full and partial rollouts with zero bootstrap for terminal transitions and critic bootstrap otherwise. Clone the best phase/camera whenever MSE strictly improves.

- [ ] **Step 4: Run training tests and verify GREEN**

Run: `python -m pytest tests/test_training.py -v`

Expected: all Task 6 tests PASS.

- [ ] **Step 5: Commit**

Commit message: `feat: train PPO against one target`

### Task 7: Artifact Output and Command-Line Entry Point

**Files:**
- Create: `PPO/ppo_slm/outputs.py`
- Create: `PPO/train.py`
- Create: `PPO/tests/test_outputs_cli.py`

**Interfaces:**
- Consumes: `PPOConfig`, `TrainingResult`, `phase_to_uint8`, and `train`.
- Produces: `create_run_directory(root: Path, name: str | None) -> Path`, `save_training_result(result: TrainingResult, run_dir: Path) -> None`, `build_parser() -> ArgumentParser`, and `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write failing output tests**

Assert collision-safe run directory naming and the exact seven files required by the spec. Load `best_phase.npy`, images, JSON, and checkpoint to verify shapes, fields, finite values, and portable plain-data config serialization.

- [ ] **Step 2: Run output tests and verify RED**

Run: `python -m pytest tests/test_outputs_cli.py -k output -v`

Expected: FAIL because output helpers are missing.

- [ ] **Step 3: Implement artifact saving and plots**

Use Matplotlib's noninteractive `Agg` backend, Pillow for grayscale images, JSON-safe scalar conversion, and `torch.save` for the checkpoint. Close every figure after saving.

- [ ] **Step 4: Run output tests and verify GREEN**

Run: `python -m pytest tests/test_outputs_cli.py -k output -v`

Expected: output tests PASS.

- [ ] **Step 5: Write failing CLI tests**

Assert parser defaults/overrides, quick configuration, relative path resolution from the project root, explicit invalid CUDA behavior, end-to-end quick execution on `data/1.bmp`, and printed initial/best MSE plus output location.

- [ ] **Step 6: Run CLI tests and verify RED**

Run: `python -m pytest tests/test_outputs_cli.py -k cli -v`

Expected: FAIL because the entry point is incomplete.

- [ ] **Step 7: Implement `train.py`**

Support `--target`, `--device`, `--seed`, `--output`, `--run-name`, `--quick`, `--total-steps`, `--rollout-steps`, `--update-epochs`, `--minibatch-size`, `--episode-length`, and `--target-mse`; return exit code zero after successful save.

- [ ] **Step 8: Run all output/CLI tests and verify GREEN**

Run: `python -m pytest tests/test_outputs_cli.py -v`

Expected: all Task 7 tests PASS.

- [ ] **Step 9: Commit**

Commit message: `feat: save PPO results and add CLI`

### Task 8: Documentation, Dependencies, and Full Verification

**Files:**
- Create: `PPO/README.md`
- Create: `PPO/requirements.txt`
- Create: `PPO/.gitignore`
- Modify: tests only if full-suite integration exposes a specification mismatch.

**Interfaces:**
- Consumes: all project behavior from Tasks 1–7.
- Produces: server-copy instructions and a reproducible install/run workflow.

- [ ] **Step 1: Write the user documentation and dependency manifest**

Document Windows/Linux environment setup, preserving an existing GPU PyTorch install, test command, quick run, full run, every CLI option, output interpretation, expected limitations of a `64×64` phase map, and copying the whole `PPO` directory to the server. Pin minimum compatible versions without forcing a CUDA-specific PyTorch wheel.

- [ ] **Step 2: Run the complete automated suite**

Run: `python -m pytest tests -v`

Expected: all tests PASS with no warnings originating from project code.

- [ ] **Step 3: Run the real-data quick workflow**

Run: `python train.py --target data/1.bmp --quick --device cpu --run-name verification`

Expected: exit code 0, finite initial/best MSE, and all seven artifacts under a newly created results directory.

- [ ] **Step 4: Inspect generated artifacts and source boundaries**

Confirm the comparison/curve images are readable, `best_phase.npy` is `64×64`, `simulated_camera.png` is `256×256`, `metrics.json` reports a valid stop reason, and no tracked content outside `PPO` was modified by implementation.

- [ ] **Step 5: Compile all Python sources**

Run: `python -m compileall ppo_slm train.py`

Expected: successful compilation with no syntax errors.

- [ ] **Step 6: Commit**

Commit message: `docs: explain PPO SLM training workflow`
