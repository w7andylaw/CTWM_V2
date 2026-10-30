# Memory Length 运行命令

所有 Memory Length 实验统一使用 batch32、length1024、train_ratio1024、100万条交互记录。以下以 Memory Length 100、seed0 为例；YiNorn、R2I与DreamerV3式GRU基线使用相同训练配置。

## WSL环境

```bash
cd /mnt/c/Users/Administrator/Desktop/CTWM_V2
source /home/administrator/.venvs/ctwm-r2i-gpu/bin/activate
unset LD_LIBRARY_PATH CUDA_ROOT CUDA_PATH JAX_PLATFORMS JAX_PLATFORM_NAME
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=true
export JAX_COMPILATION_CACHE_DIR="$PWD/outputs/.jax_cache"
SEED=0
RUN_TAG=$(date +%Y%m%d_%H%M%S)
python scripts/gpu_check.py --require-gpu
```

## YiNorn v13：Memory Length 100

```bash
python -u CTWM.py --configs small ctwm \
  --task bsuite_memory_len/100 --seed "$SEED" \
  --jax.platform gpu --jax.precision float32 --jax.prealloc False \
  --envs.amount 1 --envs.parallel none \
  --batch_size 32 --batch_length 1024 \
  --replay lfs --replay_size 10000000 \
  --run.script train --run.steps 1000000 \
  --run.train_ratio 1024 --run.train_fill 32768 \
  --nrsm.neurons 64 --nrsm.deter 128 --nrsm.sync_neurons 16 \
  --nrsm.memory 8 --nrsm.ticks 4 \
  --nrsm.stoch 8 --nrsm.classes 8 --nrsm.units 128 \
  --nrsm.nlm_hidden 8 --nrsm_tick_scale 0 \
  --imag_horizon 15 --run.log_every 60 --run.save_every 900 \
  --logdir "outputs/yinorn_memory_len100_b32_l1024_tr1024_seed${SEED}_${RUN_TAG}"
```

## R2I：Memory Length 100

```bash
python -u recall2imagine/train.py --configs small \
  --ssm_type mimo --method R2I --wdb_name R2I \
  --task bsuite_memory_len/100 --seed "$SEED" \
  --jax.platform gpu --jax.precision float32 --jax.prealloc False \
  --rssm.deter 512 --rssm.hidden 512 --rssm.units 1024 \
  --rssm.stoch 32 --rssm.classes 32 --rssm.nonrecurrent_enc True \
  --ssm.n_layers 3 --ssm.parallel True \
  --actor.inputs deter stoch --critic.inputs deter stoch \
  --encoder.mlp_keys observation --encoder.cnn_keys '$^' \
  --decoder.mlp_keys observation --decoder.cnn_keys '$^' \
  --envs.amount 1 --envs.parallel none \
  --batch_size 32 --batch_length 1024 \
  --replay lfs --replay_size 10000000 \
  --run.script train --run.steps 1000000 \
  --run.train_ratio 1024 --run.train_fill 32768 \
  --imag_horizon 15 --run.log_every 60 --run.save_every 900 \
  --logdir "outputs/r2i_memory_len100_b32_l1024_tr1024_seed${SEED}_${RUN_TAG}"
```

## DreamerV3式GRU：Memory Length 100

GRU512、deter512、z32×32，Encoder/Decoder及Actor/Critic/reward均为2×512。使用上述同一WSL环境与SEED变量。

```bash
python -u recall2imagine/train.py --configs small \
  --ssm_type gru --method DreamerV3_GRU --wdb_name DreamerV3_GRU \
  --task bsuite_memory_len/100 --seed "$SEED" \
  --jax.platform gpu --jax.precision float32 --jax.prealloc False \
  --rssm.deter 512 --rssm.units 512 --rssm.stoch 32 --rssm.classes 32 \
  --rssm.nonrecurrent_enc False --rssm.unroll False \
  --actor.inputs deter stoch --critic.inputs deter stoch \
  --decoder.inputs deter stoch --reward_head.inputs deter stoch \
  --cont_head.inputs deter stoch \
  --encoder.mlp_keys observation --encoder.cnn_keys '$^' \
  --decoder.mlp_keys observation --decoder.cnn_keys '$^' \
  --encoder.mlp_layers 2 --encoder.mlp_units 512 \
  --decoder.mlp_layers 2 --decoder.mlp_units 512 \
  --envs.amount 1 --envs.parallel none \
  --batch_size 32 --batch_length 1024 \
  --replay lfs --replay_size 10000000 \
  --run.script train --run.steps 1000000 \
  --run.train_ratio 1024 --run.train_fill 32768 \
  --imag_horizon 15 --run.log_every 60 --run.save_every 900 \
  --logdir "outputs/dreamerv3_memory_len100_gru512_b32_l1024_tr1024_seed${SEED}_${RUN_TAG}"
```

## 切换 Memory Length 或 seed

- 跑 Memory Length 30：将任务名改为 `bsuite_memory_len/30`，同时更换输出目录中的任务标识。
- 跑其他长度：更换任务名中的长度和输出目录，保持 **batch32、length1024、train_ratio1024、LFS回放1000万、Encoder/Decoder 2×512、预算100万**。
- 跑其他 seed：修改 `SEED`，使用独立输出目录。

改变命令不会改变已启动进程的参数。以各run保存的 `config.yaml` 为准；每个输出目录只由一个训练进程写入。
