# 论文式相位分布 PPO 设计

## 目标

在现有 `ppo-slm-fourier` 项目中新增一套独立的相位分布 PPO，用于针对一张指定目标图优化 `64×64` 相位，并通过简化的傅里叶焦平面模型生成接近目标的 `256×256` 模拟相机图。现有 Actor-Critic PPO、`train.py` 和既有输出格式继续保留，作为对照，不改变其行为。

本阶段只验证模拟光路。最终输出的 `64×64` 相位可在后续根据真实 SLM 的显示分辨率放大，并将模拟捕获接口替换为真实 SLM/相机反馈。

## 算法选择

新算法采用无状态相位分布 PPO。策略不是神经网络，而是以可训练平均相位 `mu` 和固定标准差 `sigma` 定义的逐像素独立高斯分布：

```text
unwrapped_phase ~ Normal(mu, sigma²)
display_phase = unwrapped_phase mod 2π
```

`mu` 的形状为 `64×64`，初始值使用固定随机种子在 `[0, 2π)` 均匀采样。默认 `sigma=0.04` 弧度。每轮从当前策略采样一批完整相位图，经过模拟光路得到图像损失，以负损失作为奖励，并在同一批测量上执行多轮 PPO 更新。

算法不包含状态编码器、Actor、Critic、GAE、episode 或环境重置。整个训练过程中平均相位持续更新，全局最佳实测样本独立保存。

## PPO 更新

默认每轮采样 64 张相位图，PPO 更新 4 个 epoch，minibatch 为 16，clip ratio 为 `0.02`。优化器使用 Adam，默认学习率 `0.3`，最大梯度范数为 `1.0`。

每张样本的优势值由奖励在当前采样批次内标准化得到：

```text
advantage = (reward - batch_mean) / (batch_std + 1e-8)
```

为避免 4096 维联合概率比的指数溢出，对每个样本、每个相位像素分别计算新旧概率比，并将该样本的标量优势广播到全部像素：

```text
ratio = exp(clamp(new_log_prob - old_log_prob, -20, 20))
surrogate = min(ratio * advantage,
                clip(ratio, 1-epsilon, 1+epsilon) * advantage)
loss = -mean(surrogate)
```

平均操作同时覆盖样本维和像素维。该更新是面向大规模独立相位参数的逐像素 PPO surrogate，不使用整幅相位图联合概率的乘积。

训练参数 `mu` 保持为未环绕的实数，避免跨越 `0/2π` 边界时破坏同一批样本的 PPO probability ratio；只有进入光路、比较和保存图片时才通过模运算生成 `[0, 2π)` 的显示相位。固定 `sigma` 不参与训练。训练和 checkpoint 必须保留未环绕样本及其旧 log probability，确保 PPO ratio 使用同一采样变量计算。

## 批量傅里叶光路

新增批量光路接口，输入形状为 `[B, 64, 64]`：

1. 使用最近邻插值扩展为 `[B, 256, 256]`；
2. 构造单位振幅复光场 `exp(i·phase)`；
3. 执行正交归一化二维 FFT 和 `fftshift`；
4. 计算模平方得到未按单图最大值归一化的光强；
5. 输出有限、非负的 `[B, 1, 256, 256]` 光强。

批量接口必须与既有单张光路的傅里叶计算一致，但不修改旧接口的返回行为。它在模拟阶段不需要对相位求光学梯度，采样测量在 `no_grad` 下执行；PPO 的梯度只经过高斯策略的 log probability 流向 `mu`。

## 图像目标函数

目标 BMP 被解释为目标光强。对模拟光强和目标光强分别取平方根得到振幅，再按各自 RMS 归一化，避免单个最大亮点决定整张图的尺度。

复合损失为：

```text
total_loss = 0.60 * foreground_amplitude_mse
           + 0.30 * (1 - ncc)
           + 0.10 * outside_energy_ratio
```

其中：

- `foreground_amplitude_mse` 使用 `1 + 8 * target_intensity` 作为逐像素权重；
- `ncc` 是归一化振幅图与目标振幅图的 Pearson 相关系数，范围裁剪到 `[-1, 1]`；
- 目标区域由 `target_intensity > 0.05` 定义；
- `outside_energy_ratio` 是目标区域外模拟光强占总模拟光强的比例；
- 所有分母使用 `1e-8` 防止除零；
- 全黑目标继续由现有图片加载器拒绝。

用于选取最佳相位和 PPO 奖励的指标是复合损失。报告用的普通强度 MSE 先将模拟强度和目标强度分别按 RMS 归一化，再进行无权重逐像素比较；PSNR只由该普通强度 MSE 计算。二者不参与奖励，从而保证 PSNR 含义正确并避免光强总尺度影响比较。

## 训练流程

默认训练 500 轮，每轮 64 次光路评估，总计 32,000 次模拟捕获。流程如下：

1. 加载并归一化指定目标图；
2. 创建相位分布策略并评估初始平均相位；
3. 每轮采样一批未环绕相位，同时生成用于显示的环绕相位；
4. 批量捕获模拟相机图并计算逐样本损失、奖励和报告指标；
5. 若任一样本的复合损失优于历史最佳值，则保存该样本及其相机图；
6. 使用同一批样本和旧 log probability 执行 PPO 更新；
7. 记录本轮样本最佳损失、批次平均损失、平均相位损失、最佳 NCC 和策略更新指标；
8. 完成指定轮数后保存结果。

快速模式使用 2 轮、每轮 8 个样本、1 个更新 epoch 和 minibatch 4，只验证完整链路，不承诺形成可辨认目标。

相同设备、相同 PyTorch 实现和相同随机种子下，训练结果应可复现。CUDA 不可用但用户显式选择 CUDA 时继续明确报错。

## 配置与命令行

新增不可变 `PhasePPOConfig`，与旧 `PPOConfig` 分离。主要默认值为：

```text
phase_size=64
image_size=256
rounds=500
population_size=64
sigma=0.04
clip_ratio=0.02
update_epochs=4
minibatch_size=16
learning_rate=0.3
max_grad_norm=1.0
foreground_weight=8.0
mse_coefficient=0.60
ncc_coefficient=0.30
outside_coefficient=0.10
target_threshold=0.05
seed=7
device=auto
```

新入口为 `train_phase_ppo.py`，至少支持：

```text
--target
--device
--seed
--output
--run-name
--quick
--rounds
--population-size
--sigma
--learning-rate
--update-epochs
--minibatch-size
```

示例：

```bash
python train_phase_ppo.py \
  --target data/1.bmp \
  --device cuda \
  --rounds 500 \
  --population-size 64 \
  --run-name phase-ppo-target-1
```

## 模块边界

新增文件：

```text
ppo_slm/phase_ppo_config.py
ppo_slm/phase_objective.py
ppo_slm/phase_distribution.py
ppo_slm/phase_ppo_training.py
ppo_slm/phase_ppo_outputs.py
train_phase_ppo.py
tests/test_phase_ppo_config.py
tests/test_phase_objective.py
tests/test_phase_distribution.py
tests/test_phase_ppo_training.py
tests/test_phase_ppo_outputs_cli.py
```

现有 `ppo_slm/optics.py` 只新增批量光路函数。旧训练器和旧入口不调用新模块，新训练器也不复用旧 Actor-Critic PPO 类。

## 输出

每次训练写入独立的 `results/<run-name>/`，同名目录自动追加编号。输出包括：

- `best_phase.npy`：历史最佳 `64×64` 环绕相位；
- `mean_phase.npy`：最终策略平均相位环绕到 `[0, 2π)` 后的结果；
- `slm_8bit.png`：最佳相位线性映射后的 8 位灰度图；
- `simulated_camera.png`：最佳模拟相机光强图；
- `comparison.png`：目标、初始平均相位成像、最佳成像、绝对误差和最佳相位；
- `training_curves.png`：复合损失、NCC、背景能量比例和 PPO 更新指标；
- `metrics.json`：配置、初始/最佳指标、最佳轮数、总光路评估次数和运行时间；
- `checkpoint_best.pt`：策略的原始未环绕均值、优化器、配置、历史和最佳状态，为后续增加恢复训练能力保留完整状态；本次命令行不提供断点恢复。

相位彩色显示只用于 `comparison.png` 的可视化；实际 SLM 文件 `slm_8bit.png` 必须是单通道灰度图。

## 错误处理

- 配置拒绝非正轮数、样本数、标准差、学习率和梯度范数；
- `population_size` 必须不小于 2，`minibatch_size` 不得超过它；
- loss 系数必须非负且总和大于零；
- 策略拒绝形状错误、非浮点、NaN 或 Inf 的样本、奖励和旧 log probability；
- 光路和目标函数发现 NaN、Inf、负光强或错误形状时立即报错；
- PPO 更新后如果原始 `mu`、loss 或梯度非有限，立即停止并报错，不保存为最佳结果；
- 输出目录不得覆盖已有实验。

## 测试与验收

采用测试驱动开发。测试至少覆盖：

- 新配置的默认值、快速模式和非法组合；
- 批量相位扩展与单张光路的一致性；
- 振幅 RMS 归一化、加权 MSE、NCC、背景能量和复合损失；
- 目标相同的图像优于中心亮点和随机散斑；
- 固定种子的策略采样可复现，显示相位始终位于 `[0, 2π)`；
- 逐像素 PPO ratio、clip、优势标准化和参数更新均为有限值；
- 一轮训练追踪全局最佳样本，总评估次数准确；
- 快速模式端到端输出全部八个文件；
- 旧测试套件保持通过。

正式模拟验收不以某一次随机运行的绝对阈值作为单元测试，而使用固定目标进行不少于三个种子的实验比较。每次正式实验应满足 `best_loss < initial_loss`，并报告最佳 NCC；是否形成可辨认图像由 `comparison.png` 人工确认。

## 非目标与后续扩展

本次不实现遗传算法、Gerchberg–Saxton、直接梯度相位优化、真实 SLM 控制、真实相机采集、硬件标定或多目标通用策略。后续硬件接入应实现与批量模拟捕获等价的接口，并保持相位策略、目标函数和输出模块不变。
