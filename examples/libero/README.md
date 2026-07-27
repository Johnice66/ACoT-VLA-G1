# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

### Ascend/aarch64 client environment

The default `examples/libero/requirements.txt` pins CUDA 11.3 PyTorch wheels for
x86_64 and will not resolve on aarch64. For an Ascend container, run the LIBERO
simulation as a websocket client and keep the Torch/NPU policy server in the main
project environment.

```bash
apt-get update && apt-get install -y \
  make \
  g++ \
  clang \
  libosmesa6-dev \
  libgl1-mesa-glx \
  libglew-dev \
  libglfw3-dev \
  libgles2-mesa-dev \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6

uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements-aarch64.txt
uv pip install -e packages/openpi-client --no-deps
uv pip install -e third_party/libero --no-deps
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

mkdir -p /tmp/libero
cat > /tmp/libero/config.yaml <<EOF
benchmark_root: $PWD/third_party/libero/libero/libero
bddl_files: $PWD/third_party/libero/libero/libero/bddl_files
init_states: $PWD/third_party/libero/libero/libero/init_files
datasets: $PWD/third_party/libero/libero/datasets
assets: $PWD/third_party/libero/libero/libero/assets
EOF
export LIBERO_CONFIG_PATH=/tmp/libero

MUJOCO_GL=egl examples/libero/.venv/bin/python examples/libero/main.py \
  --host 127.0.0.1 \
  --port 8000 \
  --task-suite-name libero_spatial \
  --task-start 0 \
  --task-count 1 \
  --num-trials-per-task 1 \
  --replan-steps 1 \
  --server-wait-timeout-s 120 \
  --exp-name torch_npu_smoke
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```
