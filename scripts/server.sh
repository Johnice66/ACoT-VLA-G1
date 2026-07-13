cart_num=${1}
port=${2}
env_mode=${3:-G2SIM}
checkpoint_dir=${4:-${G01_CHECKPOINT_DIR:-}}
policy_config=${G01_POLICY_CONFIG:-acot_g01_task_5093}

export TF_NUM_INTRAOP_THREADS=16
export CUDA_VISIBLE_DEVICES=${cart_num}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_autotune_level=0"

export PYTHONPATH=/root/openpi/src:${PYTHONPATH:-/app:/app/src}

if [ "${env_mode}" = "G01" ] || [ "${env_mode}" = "g01" ]; then
    if [ -z "${checkpoint_dir}" ]; then
        echo "G01 checkpoint is required. Usage: bash scripts/server.sh <GPU_ID> <PORT> G01 <CHECKPOINT_STEP_DIR>"
        echo "Or set G01_CHECKPOINT_DIR=/path/to/checkpoint_step before running: bash scripts/server.sh <GPU_ID> <PORT> G01"
        exit 1
    fi
    export G01_CHECKPOINT_DIR="${checkpoint_dir}"
    export G01_POLICY_CONFIG="${policy_config}"
    GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/serve_policy.py \
        --env G01 \
        --port "${port}" \
        --default-prompt "Fixed-point Non-generalized Door Opening"
else
    GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/serve_policy.py --env "${env_mode}" --port "${port}"
fi
