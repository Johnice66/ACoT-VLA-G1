容器基础镜像：

```text
quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11
```

容器启动命令：

```text
docker run -itd \
   -u root \
   --privileged \
   --device=/dev/davinci0 \
   --device=/dev/davinci1 \
   --device=/dev/davinci2 \
   --device=/dev/davinci3 \
   --device=/dev/davinci4 \
   --device=/dev/davinci5 \
   --device=/dev/davinci6 \
   --device=/dev/davinci7 \
   --device=/dev/davinci_manager \
   --device=/dev/devmm_svm \
   --device=/dev/hisi_hdc \
   -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
   -v /usr/local/dcmi:/usr/local/dcmi \
   -v /etc/ascend_install.info:/etc/ascend_install.info \
   -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
   -v /etc/hccn.conf:/etc/hccn.conf \
   --shm-size 1024g \
   --net=host \
   -v /home/XXX:/home/XXX \
   -v /disk1:/disk1 \
   --name ACoT-VLA \
   quay.io/ascend/cann:8.5.1-910b-ubuntu22.04-py3.11 \
   /bin/bash
```

拉取源码：

```bash
git clone https://github.com/AgibotTech/ACoT-VLA.git
cd ACoT-VLA
git submodule update --init --recursive
```

安装补丁：
```bash
  git checkout 64eb9bd
  git apply acot_vla_npu_adaptation.patch
```

安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

安装 `ffmpeg7`：

1. 安装工具和编译依赖
```bash
sudo apt update
sudo apt install -y software-properties-common nasm yasm gcc make pkg-config python3-dev
```
2. 从源码编译 FFmpeg 7
```bash
cd /tmp
wget https://ffmpeg.org/releases/ffmpeg-7.1.tar.gz
tar -xf ffmpeg-7.1.tar.gz
cd ffmpeg-7.1
./configure --enable-shared --enable-pic --prefix=/usr/local
make -j$(nproc)
make install
ldconfig
```
3. 让 pkg-config 找到新版头文件
```bash
echo 'export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
4. 验证
```bash
pkg-config --modversion libavcodec # 应显示 61.x
```

安装项目依赖：

```bash
GIT_LFS_SKIP_SMUDGE=1 \
UV_CACHE_DIR=/tmp/uv-cache \
UV_LINK_MODE=copy \
uv sync --no-dev --default-index https://pypi.org/simple
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

安装 NPU 版 PyTorch：

```bash
uv pip install -U pip setuptools wheel
uv pip install torch==2.7.1 torch_npu==2.7.1.post2
uv pip install torchvision==0.22.1 --no-deps
```
