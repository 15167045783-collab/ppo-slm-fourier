# 用 PPO 优化相位型 SLM 的傅里叶成像

本项目针对一张指定目标图，优化 `64×64` 相位并用最近邻方式放大到 `256×256`，再经过简化的二维傅里叶焦平面模型得到模拟相机图。

项目现在包含两套互不影响的算法：

- **相位分布 PPO（推荐）**：`train_phase_ppo.py`。直接学习一张相位分布的平均相位，每轮采样多张完整相位图并比较成像质量，更适合当前“一张目标图对应一张相位图”的任务。
- **原 Actor-Critic PPO（保留作对照）**：`train.py`。逐步输出 4096 个相位增量，原代码和原命令均未改动。

当前阶段只运行模拟光路，不控制真实 SLM 或相机。模拟结果稳定后，可以把输出相位按真实 SLM 分辨率放大，并根据设备的相位—灰度标定曲线映射。

## 推荐算法的流程

```text
指定目标图（按光强解释）
        ↓
随机初始化 64×64 平均相位 μ
        ↓
每轮从 Normal(μ, σ²) 采样一批完整相位
        ↓
相位环绕到 [0, 2π)，最近邻放大到 256×256
        ↓
exp(i·phase) → 2D FFT → 模拟相机光强
        ↓
振幅误差 + NCC 结构相似度 + 目标外能量
        ↓
逐像素 PPO 更新 μ，并始终保存历史最佳实测候选
```

该 PPO 不通过 FFT 对相位直接反向传播。PPO 梯度只经过高斯分布的 log probability，因此后续可以把批量模拟捕获替换为真实 SLM/相机反馈。

## 服务器准备

```bash
cd /4T/home/lsy/KS/PPO
conda activate jit
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

服务器已有可用 GPU 版 PyTorch 时不要覆盖它。只在缺少其他包时安装：

```bash
python -m pip install numpy Pillow matplotlib pytest
```

运行全部测试：

```bash
python -m pytest tests -v
```

## 先做快速链路检查

```bash
python train_phase_ppo.py --target data/1.bmp --quick --device cuda --run-name phase-ppo-quick
```

快速模式只有 2 轮、每轮 8 张候选，总计 `1 + 2×8 = 17` 次光路评估。它只验证训练、保存和绘图能否正常完成，不用于判断最终重建质量。

## 正式训练

默认设置是 500 轮、每轮 64 张候选，共 `1 + 500×64 = 32001` 次光路评估：

```bash
python train_phase_ppo.py \
  --target data/1.bmp \
  --device cuda \
  --rounds 500 \
  --population-size 64 \
  --run-name phase-ppo-target-1
```

建议先做一个有界的 20 轮实验，确认 loss 下降并查看图像：

```bash
python train_phase_ppo.py \
  --target data/1.bmp \
  --device cuda \
  --rounds 20 \
  --population-size 32 \
  --run-name phase-ppo-smoke
```

常用参数：

- `--target`：本次单独训练的目标图。
- `--rounds`：采样和 PPO 更新的轮数，默认 500。
- `--population-size`：每轮完整相位候选数，默认 64；显存不足时优先减小它。
- `--sigma`：采样标准差，默认 0.04 弧度。
- `--learning-rate`：平均相位的 Adam 学习率，默认 0.3。
- `--update-epochs`：同一批样本重复 PPO 更新次数，默认 4。
- `--minibatch-size`：PPO 小批量大小，默认 16，不得大于候选数。
- `--device`：`auto`、`cpu` 或 `cuda`。
- `--seed`：随机种子；相同环境和种子用于复现实验。
- `--output`：结果根目录，默认 `results`。
- `--run-name`：实验名；重名时自动追加编号，不覆盖旧结果。

如出现 CUDA 显存不足，程序会明确提示减小 `--population-size`，不会偷偷切换到 CPU。

## 推荐算法的输出

每次训练生成独立的 `results/<run-name>/`：

- `best_phase.npy`：历史最佳候选的环绕相位，形状 `64×64`，单位弧度。
- `mean_phase.npy`：训练结束时策略平均相位环绕到 `[0, 2π)` 的结果。
- `slm_8bit.png`：历史最佳相位线性映射成的单通道 8 位灰度图；这是当前模拟下的 SLM 投图文件。
- `simulated_camera.png`：最佳模拟相机光强图。
- `comparison.png`：目标、初始成像、最佳成像、绝对误差和最佳相位。
- `training_curves.png`：复合损失、NCC、目标外能量和 PPO 更新指标。
- `metrics.json`：初始/最佳指标、最佳轮数、评估次数、运行时间和完整配置。
- `checkpoint_best.pt`：未环绕平均相位、环绕平均相位、优化器、历史和最佳状态。

重点查看 `comparison.png` 和 `metrics.json`。正式实验至少应满足：

```text
best.total_loss < initial.total_loss
```

同时观察 `best.ncc` 是否提高以及最佳相机图是否形成目标结构。不能只看 PSNR 或中心亮点。

相位在 `comparison.png` 中用彩色仅为了看清相位差异；真正的 `slm_8bit.png` 是灰度图。

## 原 Actor-Critic PPO 对照

原入口仍可照常运行：

```bash
python train.py --target data/1.bmp --quick --device cuda --run-name legacy-quick
python train.py --target data/1.bmp --device cuda --run-name legacy-full
```

它的参数包括 `--total-steps`、`--rollout-steps`、`--episode-length`、`--target-mse` 等，输出格式保持原样。新旧算法建议使用不同的 `run-name`，便于比较。

## 从 GitHub 更新服务器代码

如果服务器目录没有本地修改：

```bash
cd /4T/home/lsy/KS/PPO
git status
git pull origin main
```

拉取后先执行快速检查，再开始长时间训练。

## 物理限制

- `64×64` 有 4096 个相位自由度，但相位型 SLM 和傅里叶传播本身仍限制可实现的光强图，不能保证任意图像精确重建。
- 当前模型没有包含真实波长、像素尺寸、传播距离、孔径、像差、零级光、噪声和相机响应。
- 每张目标图必须单独训练，得到的相位不会自动泛化到其他图片。
- `slm_8bit.png` 暂时假设灰度和 `[0, 2π)` 线性对应；上真实硬件前应使用设备标定曲线重新映射。
- 最终结论应至少比较 3 个随机种子，并人工检查 `comparison.png`，不能只凭一次快速运行判断算法效果。
