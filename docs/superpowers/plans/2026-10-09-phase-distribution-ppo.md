# Phase Distribution PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-style, stateless phase-distribution PPO that optimizes one `64×64` phase mask for one target image through the existing simulated Fourier focal-plane model.

**Architecture:** Keep the existing Actor-Critic PPO unchanged. Add an independent Gaussian phase policy whose trainable mean is updated with a pixel-wise clipped PPO surrogate, a batch optical simulator, a composite amplitude/NCC/background objective, dedicated training/output modules, and a separate CLI.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Pillow, Matplotlib, pytest

**Spec:** `docs/superpowers/specs/2026-10-09-phase-distribution-ppo-design.md`

## Global Constraints

- Preserve the behavior and public interfaces of `train.py` and the existing Actor-Critic PPO modules.
- Optimize exactly one selected target per invocation.
- Use a `64×64` trainable phase and a `256×256` simulated camera image.
- Wrap phase only for optical display/output; keep the trainable Gaussian mean and sampled variables unwrapped for PPO probability calculations.
- Sample optical measurements under `torch.no_grad()`; PPO gradients flow only through Gaussian log probabilities to the policy mean.
- Do not add GA, GS, direct phase-gradient optimization, real hardware I/O, resume CLI, or new third-party dependencies.
- Resolve target/output paths relative to the repository root and never overwrite an existing run directory.

## Review Focus

- A Gaussian sample crossing `0/2π` must retain its unwrapped value for log probability while the displayed phase wraps correctly; covered in Task 4 policy tests.
- Constant or nearly constant batch rewards must standardize to finite zero advantages rather than NaN; covered in Task 4 update tests.
- A target with a very small foreground must still produce a non-empty ROI and finite NCC/background metrics; covered in Task 3 objective tests.
- A partial final minibatch must be updated rather than dropped when population size is not divisible by minibatch size; covered in Task 4 update tests.
- CUDA out-of-memory during a large optical batch must be surfaced with a clear recommendation to reduce `--population-size`, not silently change the requested device; covered in Task 5 error-path tests.

---

### Task 1: Phase PPO Configuration

**Files:**
- Create: `ppo_slm/phase_ppo_config.py`
- Create: `tests/test_phase_ppo_config.py`

**Interfaces:**
- Consumes: `select_device`, `project_path`, and `seed_everything` remain in `ppo_slm.config`.
- Produces: immutable `PhasePPOConfig`, `PhasePPOConfig.quick()`, and `action_size`.

- [ ] **Step 1: Write failing configuration tests**

Assert the exact defaults from the spec, `action_size == 4096`, quick values `(rounds=2, population_size=8, update_epochs=1, minibatch_size=4)`, and rejection of non-positive dimensions/rounds/sigma/learning rate/gradient norm, population below 2, minibatch above population, invalid clip ratio, negative coefficients, all-zero objective coefficients, invalid threshold, seed, or device.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_phase_ppo_config.py -v`

Expected: FAIL because `ppo_slm.phase_ppo_config` does not exist.

- [ ] **Step 3: Implement the immutable config**

Create `PhasePPOConfig` with the exact values in the spec and the same validation style as `PPOConfig`. `quick()` changes only workload fields and preserves geometry, objective coefficients, seed, and device.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_phase_ppo_config.py -v`

Expected: all Task 1 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ppo_slm/phase_ppo_config.py tests/test_phase_ppo_config.py
git commit -m "feat: configure phase distribution PPO"
```

### Task 2: Batched Fourier Capture

**Files:**
- Modify: `ppo_slm/optics.py`
- Modify: `tests/test_optics.py`

**Interfaces:**
- Consumes: existing phase validation conventions.
- Produces: `upsample_phase_batch(phases: Tensor, output_size: int = 256) -> Tensor` and `fourier_intensity_batch(phases: Tensor, output_size: int = 256) -> Tensor` returning raw nonnegative `[B,1,H,W]` intensity.

- [ ] **Step 1: Write failing batch-optics tests**

Assert `[B,64,64] -> [B,256,256]` nearest-neighbor macro-pixels; raw intensity shape/type/finiteness/non-negativity/detachment; and that normalizing each batch result by its own maximum equals the existing `fourier_intensity` result. Reject empty batches, wrong dimensions, integer tensors, NaN/Inf, and invalid output sizes.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_optics.py -k batch -v`

Expected: FAIL because the two batch functions are missing.

- [ ] **Step 3: Implement batched expansion and propagation**

Use nearest interpolation on `phases[:, None]`, `torch.polar`, `fft2(..., dim=(-2,-1), norm="ortho")`, `fftshift(..., dim=(-2,-1))`, and squared magnitude inside `torch.no_grad()`. Do not divide raw batch intensity by per-image maxima.

- [ ] **Step 4: Run optics tests to verify GREEN**

Run: `python -m pytest tests/test_optics.py -v`

Expected: all old and new optics tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ppo_slm/optics.py tests/test_optics.py
git commit -m "feat: batch Fourier phase captures"
```

### Task 3: Composite Phase Objective

**Files:**
- Create: `ppo_slm/phase_objective.py`
- Create: `tests/test_phase_objective.py`

**Interfaces:**
- Consumes: `PhasePPOConfig` objective coefficients and target threshold.
- Produces: frozen `PhaseObjectiveBatch(total_loss, amplitude_mse, ncc, outside_energy_ratio, intensity_mse, psnr_db)` with each field shaped `[B]`, and `evaluate_phase_objective(intensities: Tensor, target: Tensor, config: PhasePPOConfig) -> PhaseObjectiveBatch`.

- [ ] **Step 1: Write failing normalization and validation tests**

Assert accepted shapes `[B,1,256,256]` and `[1,256,256]`; finite output vectors; invariance of amplitude metrics to positive intensity scaling; and rejection of negative/non-float/non-finite intensity, malformed target, empty batch, or an empty ROI.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_phase_objective.py -k 'normalization or rejects' -v`

Expected: FAIL because the objective module is missing.

- [ ] **Step 3: Implement RMS amplitude normalization and input validation**

Use `sqrt(clamp(intensity, min=0))`, per-image RMS with epsilon `1e-8`, target weights `1 + foreground_weight * target`, and ROI `target > target_threshold`.

- [ ] **Step 4: Write failing metric/composite tests**

Pin hand-computable small tensors for weighted amplitude MSE, Pearson NCC, outside-energy ratio, plain intensity MSE, PSNR, and the exact `0.60/0.30/0.10` weighted total. Assert a perfect reconstruction beats a centered impulse and random image.

- [ ] **Step 5: Run metric tests to verify RED**

Run: `python -m pytest tests/test_phase_objective.py -k 'metric or perfect or composite' -v`

Expected: FAIL because the metrics are incomplete.

- [ ] **Step 6: Implement all objective metrics**

Vectorize over batch, clamp NCC to `[-1,1]`, calculate outside energy from raw intensity, and compute PSNR only from plain intensity MSE after independently normalizing predicted and target intensity by RMS.

- [ ] **Step 7: Run objective tests to verify GREEN**

Run: `python -m pytest tests/test_phase_objective.py -v`

Expected: all Task 3 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add ppo_slm/phase_objective.py tests/test_phase_objective.py
git commit -m "feat: score phase reconstructions"
```

### Task 4: Gaussian Phase Policy and Pixel-Wise PPO

**Files:**
- Create: `ppo_slm/phase_distribution.py`
- Create: `tests/test_phase_distribution.py`

**Interfaces:**
- Consumes: `PhasePPOConfig`.
- Produces: `PhaseSamples(unwrapped, display, old_log_probs)`, `PhaseDistributionPolicy(config, device)`, `standardize_advantages(rewards: Tensor) -> Tensor`, and `update_phase_policy(policy, optimizer, samples, advantages, config) -> dict[str,float]`.

- [ ] **Step 1: Write failing policy initialization/sampling tests**

Assert seeded uniform mean shape/range, fixed non-trainable sigma, reproducible samples, sample shapes `[B,64,64]`, display values in `[0,2π)`, and an explicit crossing-boundary example where unwrapped and display values differ but old log probabilities correspond to unwrapped values. Assert returned samples are detached.

- [ ] **Step 2: Run sampling tests to verify RED**

Run: `python -m pytest tests/test_phase_distribution.py -k 'initial or sample or boundary' -v`

Expected: FAIL because the policy module is missing.

- [ ] **Step 3: Implement policy construction, sampling, and log probability**

Store `mean` as the only `nn.Parameter`, sigma as a scalar buffer, and a device-matched `torch.Generator`. `log_prob(unwrapped)` returns the elementwise normal log probability without summing dimensions.

- [ ] **Step 4: Write failing advantage/update tests**

Assert reward standardization has mean zero and finite unit population standard deviation; constant rewards yield zeros; exact pixel-wise clipped surrogate matches a hand calculation; all minibatches including a partial final one are used; parameters change for informative advantages; parameters do not change for all-zero advantages; metrics and gradients stay finite; malformed/non-finite inputs are rejected.

- [ ] **Step 5: Run update tests to verify RED**

Run: `python -m pytest tests/test_phase_distribution.py -k 'advantage or update or clip or minibatch' -v`

Expected: FAIL because update behavior is missing.

- [ ] **Step 6: Implement advantage standardization and PPO updates**

Cache old log probabilities detached, recompute elementwise new log probabilities, clamp log-ratio to `[-20,20]` before exponentiation, broadcast scalar sample advantages across phase pixels, average surrogate over sample/pixel dimensions, clip gradients, validate finiteness, and return policy loss, approximate KL, clip fraction, and gradient norm averaged across updates.

- [ ] **Step 7: Run policy tests to verify GREEN**

Run: `python -m pytest tests/test_phase_distribution.py -v`

Expected: all Task 4 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add ppo_slm/phase_distribution.py tests/test_phase_distribution.py
git commit -m "feat: implement phase distribution PPO"
```

### Task 5: Training Orchestration

**Files:**
- Create: `ppo_slm/phase_ppo_training.py`
- Create: `tests/test_phase_ppo_training.py`

**Interfaces:**
- Consumes: target loading, device/seed helpers, batch optics, objective metrics, policy sampling/update.
- Produces: frozen `PhasePPOResult` and `train_phase_ppo(config: PhasePPOConfig, target_path: Path) -> PhasePPOResult`.

- [ ] **Step 1: Write failing one-round and best-tracking tests**

With a small `phase_size/image_size` test config and temporary target, assert initial mean evaluation, exactly `rounds * population_size + 1` optical evaluations, equal-length histories, global best selected from initial/sample candidates, wrapped best/final mean phase, raw final mean retained for checkpointing, and finite objective/update metrics.

- [ ] **Step 2: Run training tests to verify RED**

Run: `python -m pytest tests/test_phase_ppo_training.py -k 'round or best or evaluations' -v`

Expected: FAIL because training orchestration is missing.

- [ ] **Step 3: Implement the training loop and result dataclass**

Evaluate initial mean once, then sample/capture/score/update per round. Store CPU clones for returned tensors and optimizer state. Track per-round best/mean loss, best NCC, mean outside-energy ratio, policy loss, approximate KL, and clip fraction. Catch CUDA OOM and raise `RuntimeError` recommending a lower population size while preserving explicit device selection.

- [ ] **Step 4: Write failing reproducibility and quick-workload tests**

Assert identical histories/best phases for repeated CPU runs with the same seed, different initial means for different seeds, quick mode completes two rounds, and legacy `train` remains importable and unchanged.

- [ ] **Step 5: Run reproducibility tests to verify RED**

Run: `python -m pytest tests/test_phase_ppo_training.py -v`

Expected: reproducibility test FAIL because all training randomness is not yet routed through deterministic generators.

- [ ] **Step 6: Complete deterministic random handling**

Use the policy's device-matched generator for Gaussian sampling and a seeded generator for minibatch permutations so identical CPU runs consume identical random streams.

- [ ] **Step 7: Run training tests to verify GREEN**

Run: `python -m pytest tests/test_phase_ppo_training.py -v`

Expected: all Task 5 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add ppo_slm/phase_ppo_training.py tests/test_phase_ppo_training.py
git commit -m "feat: train phase distribution policy"
```

### Task 6: Outputs and Command-Line Entry Point

**Files:**
- Create: `ppo_slm/phase_ppo_outputs.py`
- Create: `train_phase_ppo.py`
- Create: `tests/test_phase_ppo_outputs_cli.py`

**Interfaces:**
- Consumes: `PhasePPOConfig`, `PhasePPOResult`, existing `create_run_directory`, `phase_to_uint8`, and project path/device helpers.
- Produces: `save_phase_ppo_result(result, run_dir) -> None`, CLI `build_parser()`, `_config_from_args()`, and `main() -> int`.

- [ ] **Step 1: Write failing artifact tests**

Construct a small real `PhasePPOResult`, save it, and assert exactly eight expected artifacts; phase/image shapes and grayscale mode; JSON fields including objective components, total evaluations, runtime, and best round; checkpoint raw/wrapped phase distinction and plain-data config serialization; dynamic `64×64` phase title; collision-safe run directories.

- [ ] **Step 2: Run output tests to verify RED**

Run: `python -m pytest tests/test_phase_ppo_outputs_cli.py -k output -v`

Expected: FAIL because output helpers are missing.

- [ ] **Step 3: Implement result saving and plots**

Reuse the existing noninteractive Matplotlib/Pillow conventions without changing legacy outputs. Plot best/mean composite loss, best NCC, outside energy, approximate KL, and clip fraction against round number. Save actual SLM output as single-channel grayscale.

- [ ] **Step 4: Write failing parser and quick CLI tests**

Assert all required arguments/defaults/overrides; quick workload adjustments when population or minibatch overrides conflict; project-relative paths; printed device/target/rounds/evaluations/best loss/NCC/output; and end-to-end quick generation from `data/1.bmp`.

- [ ] **Step 5: Run CLI tests to verify RED**

Run: `python -m pytest tests/test_phase_ppo_outputs_cli.py -k cli -v`

Expected: FAIL because the entry point is incomplete.

- [ ] **Step 6: Implement `train_phase_ppo.py`**

Support every CLI option in the spec, use `PhasePPOConfig.quick()` for `--quick`, apply explicit overrides with `dataclasses.replace`, run training, create a non-overwriting directory, save results, print concise metrics, and return zero.

- [ ] **Step 7: Run output/CLI tests to verify GREEN**

Run: `python -m pytest tests/test_phase_ppo_outputs_cli.py -v`

Expected: all Task 6 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add ppo_slm/phase_ppo_outputs.py train_phase_ppo.py tests/test_phase_ppo_outputs_cli.py
git commit -m "feat: expose phase PPO training CLI"
```

### Task 7: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Test: all files under `tests/`

**Interfaces:**
- Consumes: all behavior from Tasks 1-6.
- Produces: server workflow, output interpretation, and comparison guidance for the two PPO implementations.

- [ ] **Step 1: Document the new algorithm**

Add a clear distinction between legacy Actor-Critic PPO and recommended phase-distribution PPO; quick/full commands; parameter descriptions; output meanings; evaluation-count calculation; expected GPU memory trade-off; and the fact that quick mode is not a reconstruction-quality test.

- [ ] **Step 2: Run the complete suite**

Run: `python -m pytest tests -v`

Expected: all legacy and phase PPO tests PASS with no project warnings.

- [ ] **Step 3: Run syntax compilation**

Run: `python -m compileall ppo_slm train.py train_phase_ppo.py`

Expected: exit code 0 with no syntax errors.

- [ ] **Step 4: Run the server quick workflow**

Run: `python train_phase_ppo.py --target data/1.bmp --quick --device cuda --run-name phase-ppo-quick`

Expected: exit code 0, 17 optical evaluations, finite metrics, and all eight output artifacts.

- [ ] **Step 5: Run one bounded quality smoke experiment**

Run: `python train_phase_ppo.py --target data/1.bmp --device cuda --rounds 20 --population-size 32 --run-name phase-ppo-smoke`

Expected: no NaN/Inf, `best_loss < initial_loss`, and `comparison.png` is readable. Record NCC and loss without claiming that 20 rounds prove final reconstruction quality.

- [ ] **Step 6: Inspect repository boundaries and diff**

Confirm no data image changed, no result artifact is tracked, legacy CLI still imports, `git diff --check` is clean, and only planned files changed.

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs: explain phase distribution PPO"
```
