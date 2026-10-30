#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Desktop/CTWM_V2
source /home/administrator/.venvs/ctwm-r2i-gpu/bin/activate
unset LD_LIBRARY_PATH CUDA_ROOT CUDA_PATH JAX_PLATFORMS JAX_PLATFORM_NAME
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$PWD/outputs/.jax_cache"
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=true

# Run-specific overrides: 64 history ticks / 2 ticks = about 32 environment steps.
run_dir="${1:-outputs/repeatprev_medium_v13_small_n64_m64_k2_b16_seed0_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)}"
if [[ -e "$run_dir/checkpoint.ckpt" || -e "$run_dir/replay" ]]; then
  printf 'Refusing to overwrite/resume an existing run: %s\n' "$run_dir" >&2
  exit 2
fi
mkdir -p "$run_dir"
printf '%s\n' "$$" > "$run_dir/launcher.pid"
printf 'Run directory: %s\n' "$run_dir"
python scripts/gpu_check.py --require-gpu
exec python -u CTWM.py --configs small popgym ctwm \
  --jax.platform gpu --jax.precision float32 --jax.prealloc False \
  --task gym_popgym:popgym-RepeatPreviousMedium-v0 \
  --envs.amount 1 --envs.parallel none \
  --batch_size 16 --batch_length 1024 \
  --replay_size 10000000 --imag_horizon 15 \
  --run.train_ratio 64 --run.train_fill 32768 \
  --nrsm.neurons 64 --nrsm.deter 128 --nrsm.sync_neurons 16 \
  --nrsm.memory 64 --nrsm.ticks 2 --nrsm_tick_scale 0 \
  --seed 0 --run.steps 50000000 \
  --logdir "$run_dir"
