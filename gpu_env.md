# GPU 环境部署说明

本文档用于客户在 NVIDIA GPU 上运行已经训练/转换好的 Torch 模型，例如 `final.safetensors`。

结论：

- 走本仓库的 Torch serve 路径，不需要把权重转回 JAX。
- 不需要官方 JAX checkpoint 的 `params` 目录。
- 需要保留训练配置对应的归一化统计文件，例如 `assets/acot_libero_action_cot_explicit_implicit_co_fusion/libero/norm_stats.json`。
- GPU 上使用 `--backend torch --device cuda`，不要安装 `torch_npu`，不要设置 `ASCEND_RT_VISIBLE_DEVICES`。

## 1. 基础要求

推荐环境：

- Ubuntu 22.04
- Python 3.11
- NVIDIA Driver 可正常识别 GPU
- CUDA 12.x 对应的 PyTorch 2.7.x

先确认 GPU 可见：

```bash
nvidia-smi
```

## 2. 拉取代码

如果使用已经交付的迁移版代码，直接进入代码目录即可：

```bash
cd ACoT-VLA
git submodule update --init --recursive
```

如果客户从官方仓库重新拉取，需要先应用我们交付的 Torch/NPU 迁移补丁；否则官方脚本只支持 JAX 路径，会要求 `params` 和 checkpoint assets。

## 3. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

## 4. 安装系统依赖

```bash
sudo apt update
sudo apt install -y git git-lfs ffmpeg libgl1 libglib2.0-0
git lfs install
```

如果安装依赖时报 `libavcodec` 版本不满足，再按 `refer.md` 中的 FFmpeg 7 源码编译方式处理。

## 5. 安装 Python 依赖

先同步项目依赖：

```bash
GIT_LFS_SKIP_SMUDGE=1 \
UV_CACHE_DIR=/tmp/uv-cache \
UV_LINK_MODE=copy \
uv sync --no-dev --default-index https://pypi.org/simple
```

再安装 CUDA 版 PyTorch。以下以 CUDA 12.6 wheel 为例：

```bash
uv pip install -U pip setuptools wheel
uv pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1 torchvision==0.22.1
uv pip install safetensors
uv pip install -e .
```

验证 CUDA 版 PyTorch：

```bash
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

`cuda available` 必须为 `True`。

## 6. 准备模型和归一化文件

假设目录如下：

```text
ACoT-VLA/
  tmp/libero_acot_train/final.safetensors
  assets/acot_libero_action_cot_explicit_implicit_co_fusion/libero/norm_stats.json
```

说明：

- `final.safetensors` 是训练得到的 Torch 权重。
- `assets/.../norm_stats.json` 是输入/动作归一化统计，推理时要与训练配置匹配。
- 这里的 `assets` 不是官方 JAX checkpoint 里的 `assets` 目录；Torch 权重不需要 JAX 的 `params`。

## 7. 启动单卡 policy server

```bash
export GPU_ID=0
export PORT=8025
export DATA_ROOT=./libero
export CKPT_PATH=tmp/libero_acot_train/final.safetensors

CUDA_VISIBLE_DEVICES=${GPU_ID} \
LIBERO_LEROBOT_ROOT=${DATA_ROOT} \
HF_HOME=/tmp/hf-home \
HF_DATASETS_CACHE=/tmp/hf-datasets \
UV_CACHE_DIR=/tmp/uv-cache \
UV_LINK_MODE=copy \
MPLCONFIGDIR=/tmp/matplotlib-cache \
uv run python scripts/serve_policy.py \
  --backend torch \
  --env libero \
  --device cuda \
  --port ${PORT} \
  --torch-backbone full \
  --torch-dtype bfloat16 \
  --torch-checkpoint-path ${CKPT_PATH} \
  --torch-strict-checkpoint \
  --torch-num-steps 10 \
  policy:checkpoint \
  --policy.config acot_libero_action_cot_explicit_implicit_co_fusion \
  --policy.dir .
```

注意：

- `DATA_ROOT` 建议保持 `./libero`，这样会匹配仓库中的 `assets/acot_libero_action_cot_explicit_implicit_co_fusion/libero/norm_stats.json`。
- 如果改成绝对路径，需要同步调整/放置对应的 norm stats，否则服务可能在无归一化统计的情况下启动，动作数值会不可靠。
- 如果 4090 上 `bfloat16` 兼容性或性能不符合预期，可把 `--torch-dtype bfloat16` 改为 `--torch-dtype float16` 做对比验证。

## 8. 多卡启动

当前 serve 是单进程单 GPU。多卡部署采用多个 policy server 副本，每张卡一个进程、一个端口：

```bash
export DATA_ROOT=./libero
export CKPT_PATH=tmp/libero_acot_train/final.safetensors

for GPU_ID in 0 1 2 3; do
  PORT=$((8025 + GPU_ID))
  CUDA_VISIBLE_DEVICES=${GPU_ID} \
  LIBERO_LEROBOT_ROOT=${DATA_ROOT} \
  HF_HOME=/tmp/hf-home-gpu-${GPU_ID} \
  HF_DATASETS_CACHE=/tmp/hf-datasets-gpu-${GPU_ID} \
  UV_CACHE_DIR=/tmp/uv-cache \
  UV_LINK_MODE=copy \
  MPLCONFIGDIR=/tmp/matplotlib-cache-gpu-${GPU_ID} \
  nohup uv run python scripts/serve_policy.py \
    --backend torch \
    --env libero \
    --device cuda \
    --port ${PORT} \
    --torch-backbone full \
    --torch-dtype bfloat16 \
    --torch-checkpoint-path ${CKPT_PATH} \
    --torch-strict-checkpoint \
    --torch-num-steps 10 \
    policy:checkpoint \
    --policy.config acot_libero_action_cot_explicit_implicit_co_fusion \
    --policy.dir . \
    > serve_gpu_${GPU_ID}.log 2>&1 &
done
```

客户端按实际连接的服务端口访问，例如 `8025/8026/8027/8028`。

## 9. 常见问题

`CUDA requested, but torch.cuda is not available.`

说明当前环境不是 CUDA 版 PyTorch，或容器没有挂载 GPU。先检查 `nvidia-smi`，再重新安装 CUDA 版 `torch/torchvision`。

`No converted Torch checkpoint found`

说明没有找到 Torch 权重。请确认 `--torch-checkpoint-path` 指向真实的 `.safetensors` 或 `.pt` 文件。

`Missing key` / `Unexpected key`

建议保留 `--torch-strict-checkpoint`。如果这里报错，说明权重和当前模型结构/配置不一致，不能直接忽略。

动作余弦相似度很高但实机效果异常

还需要检查 `norm_stats.json`、输入图像顺序、state/action 维度顺序、`--torch-num-steps`、dtype，以及客户端发送的 observation 字段是否与训练时一致。
