<h1 align="center">ShellAgent</h1>



<p align="center">
  <em>一套简单而强大的终端 Agent 配方：数据生成 → Agent 实现 → 模型训练 → 评估</em>
</p>



---

ShellAgent 是一个围绕**终端使用 Agent（terminal agent）**构建的完整项目，覆盖从零到部署的全流程：

- **数据生成**：以组合式采样自动合成终端任务，构建自包含容器环境，带程序化验证器
- **Agent 实现**：基于 Harbor + LiteLLM 的 `Vanillux2Agent`，通过 bash 工具与容器交互
- **模型训练**：基于 open-instruct 分支的 SFT + DPPO RL，训练 shellagent 系列模型（2B / 4B / 9B / 27B）
- **评估**：本地 vLLM 或 Beaker 集群上跑 Terminal-Bench / TB-Lite / SWE-bench 等基准

## 核心思路

| 设计选择 | 做法 |
|---------|------|
| **可扩展性（软过滤）** | *跳过*教师模型的质量校验。管线只保证任务**可执行**（容器能构建、测试能跑通），任务质量交给 RL 的软过滤——零通过率的 rollout 不贡献梯度，训练时直接丢弃 |
| **多样性（独立采样）** | 每个任务是从一组**正交轴**独立采样得到的一次抽样：领域 × 技能 × 角色 × 夹具 × 任务复杂度 × 命令复杂度 × 验证器类型 |
| **难度（显式标定）** | 两条复杂度轴 + **分级验证器**（metric_threshold / adversarial_corpus / fuzz_equivalence / multi_protocol），提供连续难度旋钮，避免"要么太简单要么无解"的双峰任务池 |
| **多模态夹具** | 容器内注入具体工件（图片 / 音频 / 视频 / stripped 二进制 / vendored 包 / 多服务 compose）。模型本身仍是纯文本模型，通过终端工具（OCR、ASR、`ffmpeg` 等）去检查这些工件 |

## 项目结构

```
shellagent/
│
├── rl_data/                 # ① 数据生成管线（四阶段）
│   ├── generate_tasks.py        #    阶段1：组合采样 → 模板 → 测试 → 容器构建 + 冒烟测试
│   ├── generate_solutions.py    #    阶段2：对每个任务跑 Agent 求解，收集 pass@k
│   ├── analyze.py               #    阶段3：任务组成/难度/平衡性统计与绘图
│   ├── upload_to_hf.py          #    阶段4：发布语料到 Hugging Face Hub
│   ├── estimate_cost.py         #    API 成本预估
│   ├── generator/               #    生成器构建模块（轴采样器、测试生成、夹具生成、求解 harness）
│   ├── containers/              #    各领域预构建的 Apptainer 基础镜像定义
│   ├── comparison/              #    与外部基线数据集的对比分析
│   ├── decontamination/         #    13-gram 重叠去污检测
│   └── scripts/                 #    各阶段的 shell 启动脚本
│
├── Vanillux2Agent/          # ② Harbor Agent 实现
│   └── agent.py                 #    基于 LiteLLM + vanillux 提示模板的 bash 工具 Agent
│
├── training/open-instruct/  # ③ 模型训练（open-instruct 分支）
│   └── scripts/shellagent/
│       ├── SFT/                 #    监督微调脚本（Qwen3-8B / Qwen3.5-9B）
│       └── RL/                  #    DPPO 强化学习脚本（2B / 4B / 9B / 27B）
│
├── beaker_configs/          # ④ Beaker 集群评估配置
│   ├── launch_eval.sh           #    一键评估：装 podman → 起 vLLM → harbor run → 收结果
│   ├── launch_vllm.sh           #    Beaker 上的 vLLM 服务启动器
│   └── vllm_serve.yaml          #    vLLM Beaker 任务配置
│
├── scripts/                 # 评估与工具脚本
│   ├── compute_stats.py         #    从 Harbor job 目录统计平均 reward / pass@k
│   ├── setup_podman_harbor.sh   #    Podman + Harbor 本地环境配置
│   ├── publish_shellagent15k.sh #    发布 ShellAgent-15K-Harbor 数据集
│   ├── beaker/                  #    Beaker 评估流程与 Harbor 补丁
│   └── plot/                    #    论文图表生成
│
├── assets/                  # 横幅图片与论文 PDF
├── pyproject.toml           # 依赖配置（uv 管理）
└── LICENSE                  # Apache 2.0
```

## 快速开始

Python 依赖通过 [`uv`](https://github.com/astral-sh/uv) 管理，以下命令均在仓库根目录执行。

```bash
# 安装依赖
uv sync
```

### 1. 生成任务数据

```bash
# 先预估成本（可选）
uv run python -m rl_data.estimate_cost --num-tasks 1000 --num-solutions 8

# 生成一个小规模任务语料（默认 legacy 模式）
NUM_TASKS=10 OUT_DIR=rl_data/output/tasks_smoke \
    bash rl_data/scripts/generate_tasks/run_generate_tasks.sh

# 用 Agent 求解，收集 pass@k
TASKS_DIR=rl_data/output/tasks_smoke \
    bash rl_data/scripts/generate_solutions/run_generate_solutions.sh

# 统计任务分布与通过率（结果写入 <TASKS_DIR>/analysis）
TASKS_DIR=rl_data/output/tasks_smoke \
    bash rl_data/scripts/analyze/run_analyze.sh
```

每个任务产出一个自包含目录：

```
task_000123_ab12cd34/
├── task.json              # 任务提示、正确答案、采样轴信息
├── test_initial_state.py  # 初始状态断言
├── test_final_state.py    # 程序化验证器（pass/fail 信号）
├── container.def          # Apptainer 容器定义
├── setup.sh               # 镜像构建时执行的环境初始化
├── fixtures/              # 多模态工件（当采样命中时）
└── solutions/             # 各模型求解结果 + pass@k 汇总
```

> 详细说明见 [`rl_data/README.md`](rl_data/README.md)：组合式轴采样器、四阶段管线、语料类型（`legacy` / `sft_v2` / `rl_v2`）、SFT warm-start 数据集等。

### 2. 训练模型

SFT 与 DPPO RL 都通过 `training/open-instruct/` 下的 open-instruct 分支运行，启动脚本位于 `training/open-instruct/scripts/shellagent/`。

```bash
# 在 training/open-instruct/ 目录下执行，例如 Qwen3.5-4B 的 RL
bash scripts/shellagent/RL/qwen35_4b.sh <beaker-image>
```

建议先用 1 张 GPU 的调试脚本 `qwen35_2b_1gpu.sh` 跑通（跑 3 步后正常退出即表示环境无误），再放大规模。
脚本分为 `mason.py` 启动器与真实训练命令两部分，详见 [`training/open-instruct/scripts/shellagent/README.md`](training/open-instruct/scripts/shellagent/README.md)。

### 3. 评估模型

```bash
# 在 Beaker 上评估：起 vLLM 服务 + 跑 Harbor 数据集
./beaker_configs/launch_eval.sh allenai/open_instruct_dev \
    --revision sft_qwen3_4b_shellagent_4node \
    --name sft-4b \
    --dataset terminal-bench@2.0
```

完整的评估流程、参数与排错见 [`scripts/beaker/README.md`](scripts/beaker/README.md)。

**手动评估**（在带 GPU 的节点上，不用 Beaker）：

```bash
# 1. 用 vLLM 在 localhost:8008 起服务（另开一个进程）
uvx vllm==0.19.1 serve allenai/shellagent-9b \
    --served-model-name shellagent-9b \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --tensor-parallel-size 8 --port 8008

# 2. 配置 Daytona 沙箱密钥
export DAYTONA_API_KEY='xxx'

# 3. 用 Harbor 跑数据集（换 --env docker 可在本地构建容器）
uv run harbor run \
  --dataset terminal-bench@2.0 \
  --env daytona \
  --agent-import-path Vanillux2Agent:Vanillux2Agent \
  --model openai/shellagent-9b \
  --agent-kwarg api_base=http://localhost:8008/v1 \
  --agent-kwarg max_format_errors=64 \
  --n-concurrent 16 \
  -k 5 \
  --job-name shellagent-9b-tb2
```



## 环境要求

- **Python ≥ 3.12** 与 [`uv`](https://github.com/astral-sh/uv)（依赖已固定于 `pyproject.toml` / `uv.lock`）
- **LLM API Key**：默认模型为 `gemini/gemini-3.1-pro-preview`，需 `GEMINI_API_KEY`；也支持通过环境变量接入本地 vLLM / Ollama / OpenAI 兼容端点
- **`apptainer`**：数据生成阶段构建与运行任务容器所必需（需在 PATH 中）
- **`HF_TOKEN`**：上传阶段与拉取 gated 模型所需
- **容器运行时**：评估时本地用 Docker/podman，或在无 Docker 的机器上用 Daytona 沙箱（`DAYTONA_API_KEY`）
- **训练**：需要 Docker Hub 登录凭证（大规模拉取镜像通常需要企业账号）、`wandb` 账号，以及 Beaker / Slurm 集群

## 常用工具

```bash
# 统计某个 Harbor job 的平均 reward 与 pass@k
python scripts/compute_stats.py jobs/<job-name> --per-task

# 配置本地 podman + harbor 环境
source scripts/setup_podman_harbor.sh
```

## 文档索引

| 文档 | 内容 |
|------|------|
| [`rl_data/README.md`](rl_data/README.md) | 数据生成管线总览 |
| [`rl_data/scripts/README.md`](rl_data/scripts/README.md) | 数据管线各阶段启动脚本 |
| [`training/open-instruct/scripts/shellagent/README.md`](training/open-instruct/scripts/shellagent/README.md) | 训练脚本解读与运行方式 |
| [`scripts/beaker/README.md`](scripts/beaker/README.md) | Beaker 评估流程与 Harbor 补丁说明 |
| [`项目文档.md`](项目文档.md) | 项目架构与完整路径树（中文） |
| [`SFT复现操作文档.md`](SFT复现操作文档.md) | SFT 训练复现步骤（中文） |
| [`笔记.md`](笔记.md) | 任务生成流程复现记录（中文） |



