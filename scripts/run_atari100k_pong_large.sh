#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Administrator/Desktop/CTWM_V2
source /home/administrator/.venvs/ctwm-r2i-gpu/bin/activate
unset LD_LIBRARY_PATH CUDA_ROOT CUDA_PATH JAX_PLATFORMS JAX_PLATFORM_NAME
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$PWD/outputs/.jax_cache"
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=true

# DreamerV3 (2023) batch/training ratio/replay capacity, with Yinorn's model.
# Reference: danijar/dreamerv3 commit 84ecf191d967f787f5cc36298e69974854b0df9c.
# Keep the existing disk-backed replay; this is not an exact Dreamer reproduction.
# large: N256, deter512, z32x32, prior/posterior512, sync64, memory8, four ticks.
# RGB Pong with six actions: 18,149,800 trainable parameters by shape accounting.
run_dir="${1:-outputs/atari100k_pong_v13_large_n256_d512_z32x32_m8_k4_b16_l64_tr1024_r1m_seed0_$(TZ=America/Chicago date +%Y%m%d_%H%M%S)}"
if [[ -e "$run_dir" ]]; then
  printf 'Choose a new output directory: %s\n' "$run_dir" >&2
  exit 2
fi
python -B - <<'PY'
from importlib.util import find_spec
from shutil import disk_usage

# FIFO_LFS rounds a 64-record RGB image chunk to seven 128-KiB units.
# Its file also reserves a 2-GiB prefix. No replay file is allocated here.
gib = 1024 ** 3
replay_bytes = ((1000000 + 63) // 64) * (7 * 128 * 1024) + 2 * gib
free_bytes = disk_usage('.').free
print('YiNorn large: 256 neurons, deter512, z32x32, memory8, 4 ticks, float32.', flush=True)
print('DreamerV3 2023 training settings: batch16 x length64, ratio1024, '
      'replay1000000, budget100000 actions.', flush=True)
print(f'LFS replay preallocation estimate: {replay_bytes / gib:.2f} GiB; '
      f'disk free: {free_bytes / gib:.2f} GiB. '
      'Preallocation on /mnt/c may take time.', flush=True)
if free_bytes < replay_bytes + 2 * gib:
    raise SystemExit('Insufficient disk space for replay plus 2 GiB of headroom.')

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
python -u CTWM.py --configs large atari100k ctwm \
  --task atari_pong --seed 0 \
  --jax.platform gpu --jax.precision float32 --jax.prealloc False \
  --envs.amount 1 --envs.parallel none \
  --batch_size 16 --batch_length 64 \
  --replay lfs --replay_size 1000000 \
  --run.script train_eval --run.steps 100000 \
  --run.train_ratio 1024 --run.train_fill 1024 \
  --run.eval_every 100000 --run.eval_initial False --run.eval_eps 100 \
  --run.log_every 60 --run.save_every 900 \
  --nrsm.memory 8 --nrsm.ticks 4 --nrsm_tick_scale 0 \
  --imag_horizon 15 --logdir "$run_dir" \
  2>&1 | tee "$run_dir/console.log"
