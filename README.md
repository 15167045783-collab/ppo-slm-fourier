# PPO 优化 8×8 相位型 SLM 的傅里叶成像

这个项目针对 `data` 中指定的一张目标图，训练 PPO 逐步修改 `8×8` 相位。每一步都把相位双线性插值成 `256×256`，经过二维 FFT 模拟傅里叶焦平面成像，再根据模拟相机图与目标图的误差给出奖励。训练会保存整个过程中误差最低的相位，而不是只保存最后一步。

`PPO` 是独立项目，不依赖也不修改旁边的 `moni` 项目。

## 工作流程

```text
指定目标图（256×256）
        ↓
随机 8×8 相位，范围 [0, 2π)
        ↓
PPO 输出 64 个连续相位增量
        ↓
双线性插值到 256×256
        ↓
exp(i·phase) → 2D FFT → 归一化光强
        ↓
模拟相机图与目标图计算 MSE
        ↓
继续迭代，直到达到 target_mse 或 total_steps
```

观测包含目标图、当前模拟相机图和当前相位图三个通道。PPO 只对策略网络求梯度，不通过 FFT 光路直接对相位反向传播，因此后续可以把模拟成像替换为真实 SLM/相机反馈。

## 目录

```text
PPO/
├── data/                  # 12 张现有 BMP 目标图
├── ppo_slm/               # 光路、环境、PPO、训练与结果保存
├── tests/                 # 自动化测试
├── train.py               # 训练入口
├── requirements.txt
└── README.md
```

## 复制到服务器

复制整个 `PPO` 文件夹即可。不要只复制 `train.py`，因为它还需要 `ppo_slm`、`data` 和其他文件。

例如在本地 PowerShell 中：

```powershell
scp -r "D:\桌面\学习\科研\第二个工作\PPO" 用户名@服务器地址:/目标父目录/
```

也可以压缩 `PPO` 后通过 VS Code 或 SFTP 上传。`results`、`.venv` 和缓存目录不需要上传。

## 检查服务器环境

代码要求 Python 3.10 或更高版本。

进入服务器上的项目目录：

```bash
cd /你的路径/PPO
python3 --version
python3 -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"
```

如果服务器已经有能使用 GPU 的 PyTorch，请优先保留该版本，不要盲目覆盖。只补缺少的包：

```bash
python3 -m pip install numpy Pillow matplotlib pytest
```

如果使用 Conda：

```bash
conda activate 你的环境名
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install numpy Pillow matplotlib pytest
```

只有在服务器没有 PyTorch 时，才根据服务器 CUDA 版本到 PyTorch 官方安装页选择对应命令。`requirements.txt` 中没有绑定某个 CUDA wheel；已有满足版本要求的 PyTorch 不会被强制替换。

## 先运行测试

```bash
python -m pytest tests -v
```

测试覆盖目标图读取、8×8 相位插值、FFT 模拟成像、环境奖励与停止、GAE/PPO 更新、最佳结果追踪、输出文件和命令行快速流程。

## 快速检查

快速模式只用 16 个环境步检查完整链路和输出，不保证图像已经明显接近目标：

```bash
python train.py --target data/1.bmp --quick --device auto --run-name first-check
```

## 正式训练

```bash
python train.py --target data/1.bmp --device auto --run-name target-1
```

换另一张目标图：

```bash
python train.py --target data/6.bmp --device auto --run-name target-6
```

显式使用 GPU：

```bash
python train.py --target data/1.bmp --device cuda --run-name target-1-gpu
```

如果显式指定 `cuda` 但 PyTorch 无法使用 CUDA，程序会直接报错。`auto` 会在 CUDA 可用时选 GPU，否则用 CPU。

## 常用参数

```bash
python train.py \
  --target data/1.bmp \
  --device auto \
  --total-steps 50000 \
  --rollout-steps 256 \
  --episode-length 256 \
  --update-epochs 10 \
  --minibatch-size 64 \
  --target-mse 0.005 \
  --seed 7 \
  --output results \
  --run-name experiment-1
```

- `--target`：本次单独训练的目标图。
- `--device`：`auto`、`cpu` 或 `cuda`。
- `--total-steps`：整个训练最多执行的环境步数，默认 20,000。
- `--rollout-steps`：每次 PPO 更新收集的步数，默认 256。
- `--episode-length`：一次随机相位 episode 的最大步数，默认 256。
- `--update-epochs`：每批 rollout 重复优化的轮数，默认 10。
- `--minibatch-size`：PPO 小批量大小，默认 64。
- `--target-mse`：达到该 MSE 时提前结束整个训练，默认 0.005。
- `--seed`：随机种子。
- `--output`：结果根目录，相对路径以 `PPO` 为根解析。
- `--run-name`：本次实验名；同名目录已存在时自动追加编号，不覆盖旧结果。

## 输出结果

每次训练会创建 `results/<run-name>/`，其中包括：

- `best_phase.npy`：最佳 `8×8` 浮点相位，单位为弧度；
- `slm_8bit.png`：最佳相位映射到 0–255 后的 `8×8` SLM 投图；
- `simulated_camera.png`：最佳 `256×256` 模拟相机光强；
- `comparison.png`：目标、初始图、最佳图、绝对误差和最佳相位；
- `training_curves.png`：MSE、奖励和 PSNR 曲线；
- `checkpoint_best.pt`：模型、优化器、配置和最佳相位；
- `metrics.json`：初始/最佳 MSE、PSNR、改善百分比、最佳步数和停止原因。

查看关键数字：

```bash
cat results/target-1/metrics.json
```

`stop_reason` 为 `target_mse` 表示已经达到设置的误差阈值；为 `total_steps` 表示用完了训练步数。正式训练的最基本判断是 `best_mse < initial_mse`。

## 重要限制

- `8×8` 只有 64 个相位自由度，无法精确重建任意复杂的 `256×256` 图像；它更适合验证闭环和粗结构重建。
- 二维 FFT 是简化的傅里叶焦平面模型，没有包含真实波长、像素尺寸、传播距离、像差、噪声和相机响应。
- 每张目标图单独训练，模型不会自动泛化到其他图片。
- PPO 面对连续 64 维动作可能需要较多训练步数；快速模式只验证代码能运行。
- 若长期没有改善，可先尝试增加 `--total-steps`，再调整 `target_mse`、相位步长或奖励权重；这些高级参数集中在 `ppo_slm/config.py`。

## 从 checkpoint 读取最佳相位

```python
import torch

checkpoint = torch.load("results/target-1/checkpoint_best.pt", map_location="cpu")
best_phase = checkpoint["best_phase"]      # shape: [8, 8]，弧度
best_camera = checkpoint["best_camera"]    # shape: [1, 256, 256]
print(checkpoint["best_mse"])
```

实际投到相位型 SLM 时，可直接使用输出的 `slm_8bit.png`，或根据真实设备的相位—灰度标定曲线重新映射 `best_phase.npy`。当前 0–255 映射是假设灰度与 `[0, 2π)` 线性对应。
