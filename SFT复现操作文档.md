# ShellAgent SFT 训练复现操作文档

> 项目路径：`/data/dhsun/shellagent`
> 梳理日期：2026-09-02
> 目标：复现 ShellAgent 的 SFT（Supervised Fine-Tuning）训练

---

## 〇、SFT 在整条流水线中的位置

ShellAgent 训练 terminal agent 分两大阶段，SFT 是**第一步（热启动）**，RL 在 SFT 产出的模型之上继续：

```
task generation (任务生成)  →  trajectory generation (求解轨迹)  →  SFT 训练  →  RL 训练
    rl_data/generate_tasks      rl_data/generate_solutions           本阶段     DPPO/GRPO
```

SFT 的**目标**：用 agent 成功解决问题的轨迹（prompt + 思考 + bash 动作序列），教会 base model（Qwen）如何调用 bash 工具、在终端环境里完成任务。

---

## 一、SFT 数据从哪来（上游依赖）

SFT 不是凭空训练的，它依赖前两步产出的数据：

| 阶段           | 产物                                                                              | 位置/脚本                                     |
| -------------- | --------------------------------------------------------------------------------- | --------------------------------------------- |
| ① 任务生成     | `task_*/task.json`、容器、测试                                                    | `rl_data/scripts/generate_tasks/`             |
| ② 轨迹生成     | `task_*/solutions/<MODEL_TAG>_<harness>_summary.json`（成功的轨迹）               | `rl_data/scripts/generate_solutions/`         |
| ③ 轨迹→SFT格式 | `skill_tax_20260505_2.2k_combined_balanced_thinking_all` 这样的 HF dataset config | `sft/preprocessing/convert_trajectories.py` ⚠️ |
| ④ 上传 HF      | `allenai/shellagent-sft`                                                                | 见训练脚本 `--dataset_mixer_list`             |

### ⚠️ 关键缺口 1：`sft/preprocessing/convert_trajectories.py` 不在仓库

本地仓库 `rl_data/generator/sample_solutions.py`、多个脚本都引用
`sft/preprocessing/convert_trajectories.py`（以及 `sft/preprocessing/config/`），但**该目录不在本仓库中**（可能在私有 repo 或被 gitignore）。

它的作用：把 `generate_solutions` 输出的每任务 `*_summary.json`（原始轨迹），转换成 open-instruct SFT 训练格式（messages 列 + tools 列），并组合成
`skill_tax_..._thinking_all` 这类 dataset config。

> **复现前必须先找师兄确认这个脚本的位置**，否则无法从自己生成的轨迹得到训练数据。

### ⚠️ 关键缺口 2：SFT 训练数据基座已由别人生成并上传 HF

训练脚本直接引用现成的 HF 数据集：
- `allenai/shellagent-sft` → small（只用 shellagent 自己的 ~2.2k 轨迹）
- `allenai/shellagent-sft-big` → big（shellagent + 之前工作的大混合）

**如果只是想跑通 SFT 训练**：直接用这些公开数据集即可，不需要自己从 0 生成。
**如果要完整复现数据流水线**：需要上面缺口 1 的转换脚本。

---

## 二、训练脚本解读

SFT 启动脚本在 `training/open-instruct/scripts/shellagent/SFT/`：

| 脚本                     | 基座模型               | 数据集                 | 规模           |
| ------------------------ | ---------------------- | ---------------------- | -------------- |
| `sft_qwen3_8b_small.sh`  | `Qwen/Qwen3-8B`        | `allenai/shellagent-sft`     | 只用 shellagent 数据 |
| `sft_qwen3_8b_big.sh`    | `Qwen/Qwen3-8B`        | `allenai/shellagent-sft-big` | 大混合         |
| `sft_qwen35_9b_small.sh` | `hamishivi/Qwen3.5-9B` | `allenai/shellagent-sft`     | 只用 shellagent 数据 |
| `sft_qwen35_9b_big.sh`   | `hamishivi/Qwen3.5-9B` | `allenai/shellagent-sft-big` | 大混合         |

> ⚠️ `hamishivi/Qwen3.5-9B` 可能是**私有/需授权模型**（Qwen3.5 未在 HF 公开），需确认是否有访问权限。`Qwen/Qwen3-8B` 是公开的，最容易上手复现。

### 脚本结构 = mason 启动器 + `--` + 训练命令

每个脚本都分两段（以 `sft_qwen3_8b_small.sh` 为例）：

```bash
uv run python mason.py \
    --cluster ai2/jupiter --workspace ai2/open-instruct-dev --priority urgent \
    --image "$BEAKER_IMAGE" --pure_docker_mode --preemptible \
    --num_nodes 4 --budget ai2/oe-adapt --gpus 8 \
    -- \
    accelerate launch \
    --mixed_precision bf16 --num_processes 8 --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name sft_qwen3_8b_shellagent \
    --model_name_or_path Qwen/Qwen3-8B \
    --tokenizer_name Qwen/Qwen3-8B \
    --use_flash_attn \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list allenai/shellagent-sft 1.0 \
    --dataset_mixer_list_config_names skill_tax_20260505_2.2k_combined_balanced_thinking_all \
    --add_bos \
    --gradient_checkpointing \
    --report_to wandb --with_tracking --logging_steps 1 --seed 42
```

### 关键训练超参

| 参数                              | 值                                                       | 说明                               |
| --------------------------------- | -------------------------------------------------------- | ---------------------------------- |
| `max_seq_length`                  | 32768                                                    | 长轨迹序列                         |
| `per_device_train_batch_size`     | 1 × 8 GPU × 4 nodes                                      | 小 batch                           |
| `gradient_accumulation_steps`     | 4                                                        |                                    |
| `learning_rate`                   | 2e-5                                                     |                                    |
| `num_train_epochs`                | 2                                                        |                                    |
| `dataset_mixer_list`              | `allenai/shellagent-sft` × 1.0                                 | 训练数据集                         |
| `dataset_mixer_list_config_names` | `skill_tax_20260505_2.2k_combined_balanced_thinking_all` | **选择数据集里的哪个 config/子集** |
| `add_bos`                         | 开                                                       |                                    |
| `use_flash_attn`                  | 开                                                       | 需要 flash-attn 环境               |
| `report_to wandb`                 | 开                                                       | 需要 wandb 账号                    |

> ⚠️ **`--dataset_mixer_list_config_names` 参数只在 small 脚本出现**，big 脚本没有——两个脚本数据集内部结构不同。

### `dataset_mixer_list_config_names` 是什么

HF 数据集可以有多个 **config**（配置/子集）。`allenai/shellagent-sft` 里
`skill_tax_20260505_2.2k_combined_balanced_thinking_all` 是其中一个 config，表示：
- 20260505 那批 ~2.2k 轨迹
- combined balanced（legacy+v2 均衡混合）
- **thinking_all**（保留所有思考痕迹的变体 —— Qwen3 用交错 reasoning chat template）

看脚本注释：`sft_qwen35_9b_small.sh` 明确说 "We use a version of Qwen 3.5 with an interleaved reasoning chat template"。

---

## 三、完整复现步骤（两条路线）

### 路线 A：直接用公开数据跑训练（推荐先跑通）

适合：验证代码能跑、环境正确。不碰数据生成上游。

```bash
cd /data/dhsun/shellagent/training/open-instruct

# 1. 安装依赖
uv sync            # 或 uv sync --extra 等（看 pyproject 里的 extras）

# 2. 登录 wandb（脚本会 report）
wandb login

# 3. 准备 HF 数据集（allenai/shellagent-sft 若 gated 需要 token）
export HF_TOKEN=...

# 4. 直接跑（4 节点×8 GPU 版，见下）或改成小规模
bash scripts/shellagent/SFT/sft_qwen3_8b_small.sh <你的beaker镜像或本地跑>
```

### 路线 B：完整复现（数据→轨迹→SFT）

在路线 A 基础上，先把上游自己跑出来：

```bash
# ① 任务生成（你自己已在 tasks_smoke 跑通过 2 个）
NUM_TASKS=2200 OUT_DIR=rl_data/output/tasks_skill_tax_v2_... \
    bash rl_data/scripts/generate_tasks/run_generate_tasks_sft_v2_1k.sh

# ② 生成轨迹（每任务 8 条，取成功轨迹）
#    需 GPU + vLLM 服务 + agent harness
TASKS_DIR=rl_data/output/tasks_... \
    bash rl_data/scripts/generate_solutions/run_generate_solutions_skill_tax_combined_2.5k.sh

# ③ 轨迹 → SFT 格式（⚠️ 缺脚本，需找师兄要 sft/preprocessing/convert_trajectories.py）
# ④ 上传 HF dataset（换成自己的 repo）
bash rl_data/scripts/upload/upload_data_to_hf.sh --repo <your>/shellagent-sft
```

---

## 四、在"你自己的实验室集群"上跑的改造要点

原脚本是为 **AI2 Beaker** 写的。实验室若用 **Slurm / 本地 GPU**，需改：

### 1. 去掉 mason.py 部分

mason.py 只是 Beaker 调度器。去掉 `uv run python mason.py ... --` 之前的部分，
保留 `--` 之后的训练命令即可（这是真正的训练逻辑）。

### 2. 本地/Slurm 跑训练命令示例

```bash
cd /data/dhsun/shellagent/training/open-instruct

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --exp_name sft_qwen3_8b_shellagent_local \
    --model_name_or_path Qwen/Qwen3-8B \
    --tokenizer_name Qwen/Qwen3-8B \
    --use_flash_attn \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list allenai/shellagent-sft 1.0 \
    --dataset_mixer_list_config_names skill_tax_20260505_2.2k_combined_balanced_thinking_all \
    --add_bos \
    --gradient_checkpointing \
    --report_to wandb --with_tracking --logging_steps 1 --seed 42
```

> 按你自己的 GPU 数量调整 `--num_processes`（= 总 GPU 数）。

### 3. Slurm 提交（参考其他 slurm 脚本）

`training/open-instruct/scripts/slurm/` 下有现成 Slurm 示例（sft 相关），可参考
`scripts/slurm/sft/train_dolci_*.sh` 的提交头写法（`#SBATCH`）和 accelerate 启动方式。

### 4. 环境依赖核对

| 依赖           | 说明                                     |
| -------------- | ---------------------------------------- |
| CUDA + PyTorch | 按 open-instruct requirements            |
| flash-attn     | `--use_flash_attn` 需要                  |
| deepspeed      | stage3 offloading config                 |
| wandb          | `--report_to wandb`                      |
| HF hub 访问    | 拉 `Qwen/Qwen3-8B` 和 `allenai/shellagent-sft` |

---

## 五、复现清单核对表（照着打勾）

- [ ] 确认 ① 师兄能否提供 `sft/preprocessing/convert_trajectories.py`（或轨迹已在 HF）
- [ ] 确认 ② 选择哪个模型基座：`Qwen/Qwen3-8B`（公开，推荐）还是 `Qwen3.5`（可能需授权）
- [ ] 确认 ③ `allenai/shellagent-sft` HF 数据集能否访问（gated？）
- [ ] 确认 ④ 训练用哪条路线：A 直接公开数据 / B 完整流水线
- [ ] 确认 ⑤ 实验室有没有 4×8=32 张 GPU（没有就调小 `--num_processes`）
- [ ] 确认 ⑥ `uv sync` 依赖 + wandb + HF token 配好
- [ ] 先跑 **小规模冒烟**：1-2 个 GPU、`num_train_epochs` 调小、`max_seq_length` 调小，确认能跑通不报错
- [ ] 再全量跑

---

## 六、和 RL 的衔接（了解即可）

SFT 产出的 checkpoint 是 RL（DPPO/GRPO）的起点：
- RL 脚本：`training/open-instruct/scripts/shellagent/RL/`
- RL 训练数据：`swerl-shellagent-15k`（RL 格式，带 sandbox 环境）
- RL 里 `--model_name_or_path` 指向 SFT 产物

建议先跑 `qwen35_2b_1gpu.sh`（1 GPU 调试脚本）验证 RL 环境。

---

## 七、当前复现的已知障碍汇总

| 障碍                            | 严重度 | 处理                           |
| ------------------------------- | ------ | ------------------------------ |
| `sft/preprocessing/` 不在仓库   | 🔴 高   | 找师兄要                       |
| 依赖 AI2 Beaker/mason           | 🟡 中   | 去掉 mason 部分，本地/Slurm 跑 |
| `hamishivi/Qwen3.5-9B` 可能私有 | 🟡 中   | 改用公开 Qwen3-8B 或申请权限   |
| `allenai/shellagent-sft(-big)` gated? | 🟡 中   | 确认 HF 权限                   |
| 需 32 GPU（4节点×8）            | 🟡 中   | 减小规模验证，再按资源跑       |
| Dockerhub PAT / sandbox         | ⚪ 低   | 仅 RL 阶段需要                 |
