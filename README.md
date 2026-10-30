# YiNorn：实验配置与基线

本页统一列出本项目采用的实验配置。Memory Length 使用指定的统一配置；POPGym 的训练参数与核心规模采用朋友的 RepeatPrevious 配置，显式历史窗口按任务所需跨度设置；Atari 100k 使用当前 large 配置。

## 训练参数

以下预算均为每个 seed 的预算，全部使用 **float32**。

| 实验 | 训练预算 | batch × length | train_ratio | 回放容量 | 环境数 |
|---|---|---|---|---|---|
| Memory Length（所有长度） | 100万条交互记录 | **32 × 1024** | **1024** | LFS 1000万 | 1 |
| POPGym（含 RepeatPrevious） | **5000万条交互记录** | **16 × 1024** | **64** | LFS 1000万 | 8，process |
| Atari 100k（当前 Pong） | 10万次动作决策 | **16 × 64** | **1024** | LFS 100万 | 1 |

同一任务中的 YiNorn 与基线共用上表的训练预算、batch、序列长度、train_ratio、回放容量与精度。POPGym 使用 seed 0、1、2；Atari 当前使用 seed 0；Memory Length 的 seed 由运行命令指定。

## 模型大小与输入

| 实验 | 模型 | 递归核心 | deter（D） | 随机潜变量（Z） | 显式历史覆盖（环境步） | Actor / Critic 输入 |
|---|---|---|---|---|---|---|
| Memory Length | YiNorn | **64 神经元** | **128** | **8×8** | 约2步（memory8，ticks4） | deter + z |
| Memory Length | R2I | 3层 SSM，每层 hidden512、units1024 | 512 | 32×32 | — | deter + z |
| Memory Length | DreamerV3式 GRU 基线 | GRU512、RSSM units512 | 512 | 32×32 | — | deter + z |
| POPGym | YiNorn | **64 神经元** | **128** | **8×8** | **按任务所需历史步数覆盖**；RepeatPrevious Medium为约32步（memory64，ticks2） | pre_trace + post_trace + z |
| POPGym | R2I | 3层 SSM，每层 hidden512、units1024 | 512 | 32×32 | — | SSM hidden + z |
| POPGym | DreamerV3式 GRU 基线 | GRU1024、RSSM units1024 | 1024 | 32×32 | — | deter + z |
| Atari 100k | **YiNorn large** | **256 神经元** | **512** | **32×32** | 约2步（memory8，ticks4） | deter + z |
| Atari 100k | DreamerV3式 GRU 基线 | GRU512、RSSM units512 | 512 | 32×32 | — | deter + z |

`Z=8×8` 表示8个分类变量、每个8类，展平后64维。`memory` 是神经活动的 tick 历史长度，显式窗口约为 `memory / ticks` 个环境步，不是递推记忆的最长持续时间。POPGym 按任务需要保留的历史跨度设置窗口：`nrsm.memory = 所需环境步数 × 每步ticks`；每步2 ticks时，RepeatPrevious Medium的32步跨度对应memory64。

| 网络参数 | Memory Length | POPGym | Atari 100k |
|---|---|---|---|
| Encoder / Decoder | 三种模型均为 MLP 2×512 | YiNorn/R2I：MLP 2×512；GRU：MLP 5×1024 | CNN depth32，MLP 2×512 |
| Actor / Critic / reward 隐藏层 | 2×512 | 2×512 | 2×512 |
| YiNorn 先验/后验宽度 | 128 | 128 | 512 |
| YiNorn NLM 隐藏宽度 | 8 | 8 | 8 |
| YiNorn 同步采样神经元 | 16 | 16 | 64 |

当前 Pong 的可训练参数量：YiNorn large **18,149,800**，GRU 基线 **17,922,952**，不计优化器状态和慢速 Critic 副本。

## DreamerV3 基线：各实验参数

以下为本项目共享训练框架中的 **DreamerV3式 GRU-RSSM 基线**。每项实验与YiNorn使用相同的交互预算、batch、序列长度、train_ratio及float32精度。

| 参数 | Memory Length（所有长度） | POPGym（所有任务） | Atari 100k（当前 Pong） |
|---|---|---|---|
| 每seed训练预算 | 100万条记录 | 5000万条记录 | 10万次动作决策 |
| seed | 运行命令指定，示例0 | 0、1、2 | 0 |
| batch × length | **32×1024** | **16×1024** | **16×64** |
| train_ratio | **1024** | **64** | **1024** |
| 回放 | LFS 1000万 | LFS 1000万 | LFS 100万 |
| 环境数 / 并行方式 | 1 / none | 8 / process | 1 / none |
| train_fill / 有效预采集阈值 | 32768 / 32768 | 0 / 131072 | 1024 / 1024 |
| GRU隐藏状态 / deter | **512** | **1024** | **512** |
| RSSM中间层宽度 units | 512 | 1024 | 512 |
| 随机潜变量 z | 32×32 | 32×32 | 32×32 |
| Encoder / Decoder | MLP 2×512 | MLP 5×1024 | CNN depth32，MLP 2×512 |
| Actor / Critic / reward / continuation隐藏层 | 2×512 | 2×512 | 2×512 |
| Actor / Critic及预测头输入 | deter + z | deter + z | deter + z |
| 后验 | q(z\|deter, observation) | q(z\|deter, observation) | q(z\|deter, observation) |
| 奖励头 | 普通MLP | 普通MLP | 普通MLP |
| 精度 / 想象长度 | float32 / 15 | float32 / 15 | float32 / 15 |
| 运行方式 | train，训练回报 | train，训练回报 | train_eval，结束后100回合评估 |

三项基线均设置 `ssm_type=gru`、`rssm.nonrecurrent_enc=False`、`rssm.unroll=False`；没有YiNorn的nrsm.memory或多tick参数。优化器、KL、熵正则与折扣使用下节共用设置。

Memory Length的GRU512核心参考[DreamerV3 2023 BSuite预设](https://github.com/danijar/dreamerv3/blob/84ecf191d967f787f5cc36298e69974854b0df9c/dreamerv3/configs.yaml)，Encoder/Decoder固定为本项目统一的2×512。POPGym沿用GRU1024和5×1024 Encoder/Decoder的对比配置；Atari沿用现有启动脚本。

## 共用优化设置

| 参数 | 值 |
|---|---|
| 世界模型 Adam | lr `1e-4`，eps `1e-8`，clip1000，wd0 |
| Actor / Critic Adam | lr `3e-5`，eps `1e-5`，clip100，wd0 |
| loss 权重 | 观测/奖励/continuation 各1；dyn KL0.5、rep KL0.1；free nats1 |
| 想象与回报 | 想象长度15，horizon333（γ≈0.997），λ0.95 |
| 策略 | 离散 REINFORCE，unimix0.01，熵权重 `3e-4` |
| Critic | 255-bin symlog 分布，slow 更新比例0.02 |
| YiNorn | tick KL权重0；动作条件奖励首层；无历史花色监督loss |

## 运行与结果

- **Memory Length：** 在本项目运行，见 [运行命令](docs/EXPERIMENT_RUN_COMMANDS.md)。所有长度保持 B32/L1024、train_ratio1024。
- **POPGym：** 在独立的 `CTWM_V2_PopGym` 项目运行，见 [POPGym README](../CTWM_V2_PopGym/README.md) 和 [运行命令](../CTWM_V2_PopGym/docs/EXPERIMENT_RUN_COMMANDS.md)。POPGym 沿用 B16/L1024、train_ratio64、N64/D128/Z8×8、每步2 ticks；memory按任务所需历史步数设置，RepeatPrevious Medium为64。
- **Atari：** YiNorn 使用 [large 启动脚本](scripts/run_atari100k_pong_large.sh)，Dreamer 使用 [GRU 启动脚本](scripts/run_atari100k_pong_dreamerv3.sh)。Pong 观测为64×64 RGB，动作重复4、最小动作集合、sticky关闭、NOOP上限30；10万次动作结束后独立评估100回合。

有效预采集阈值：Memory Length 为32768条，POPGym 为131072条，Atari 为1024条。Memory Length/POPGym 的 `scores.jsonl` 记录训练回报；Atari 另有独立评估。最终生效参数保存在每次运行的 `config.yaml`，loss 在 `metrics.jsonl`。

表中的 R2I 与 GRU 参数是本项目的对比配置；GRU 基线运行在共享训练框架内。方法参考：[R2I](https://arxiv.org/html/2403.04253v1)、[DreamerV3 2023配置](https://github.com/danijar/dreamerv3/blob/84ecf191d967f787f5cc36298e69974854b0df9c/dreamerv3/configs.yaml)。

方法与公式：[METHOD](docs/METHOD.md)、[FORMULA_ALIGNMENT](docs/FORMULA_ALIGNMENT.md)。GPU环境：[GPU_RUNTIME](docs/GPU_RUNTIME.md)。其他任务：[TASK_SUPPORT](docs/TASK_SUPPORT.md)。
