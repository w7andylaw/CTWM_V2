#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Desktop/CTWM_V2
source /home/administrator/.venvs/ctwm-r2i-gpu/bin/activate
unset LD_LIBRARY_PATH CUDA_ROOT CUDA_PATH JAX_PLATFORMS JAX_PLATFORM_NAME
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$PWD/outputs/.jax_cache"
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=true

# DreamerV3 current public batch/training-ratio settings, with Yinorn's model.
# tiny is the former small network: N64, history8, four internal ticks.
run_dir="${1:-outputs/atari100k_pong_v13_n64_m8_k4_b16_l64_seed0_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$run_dir" ]]; then
  printf 'Choose a new output directory: %s\n' "$run_dir" >&2
  exit 2
fi
python -B - <<'PY'
from importlib.util import find_spec
missing = [name for name in ('ale_py', 'cv2') if find_spec(name) is None]
if missing:
    raise SystemExit('Missing Atari dependencies: ' + ', '.join(missing) +
                     '\nInstall requirements-atari.txt and run AutoROM first.')
PY
python scripts/gpu_check.py --require-gpu
# This native check fails early if Pong ROMs or the environment are unavailable.
python scripts/smoke_env.py --task atari_pong --steps 4 --seed 0
mkdir -p "$run_dir"
printf 'Output directory: %s\n' "$run_dir"
python -u CTWM.py --configs tiny atari100k ctwm \
  --task atari_pong --seed 0 \
  --jax.platform gpu --jax.precision float32 --jax.prealloc False \
  --envs.amount 1 --envs.parallel none \
  --batch_size 16 --batch_length 64 \
  --replay lfs --replay_size 200000 \
  --run.script train_eval --run.steps 100000 \
  --run.train_ratio 256 --run.train_fill 1024 \
  --run.eval_every 100000 --run.eval_initial False --run.eval_eps 100 \
  --run.log_every 60 --run.save_every 900 \
  --nrsm.neurons 64 --nrsm.deter 128 --nrsm.sync_neurons 16 \
  --nrsm.memory 8 --nrsm.ticks 4 --nrsm_tick_scale 0 \
  --imag_horizon 15 --logdir "$run_dir" \
  2>&1 | tee "$run_dir/console.log"
