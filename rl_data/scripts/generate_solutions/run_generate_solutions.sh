#!/bin/bash
# 指定使用 Bash 解释器执行

#SBATCH --job-name=rl-gen-solutions
# Slurm 作业名：rl-gen-solutions

#SBATCH --output=logs/gen_solutions_%j.out
# Slurm 标准输出日志路径，%j 替换为 job ID

#SBATCH --error=logs/gen_solutions_%j.err
# Slurm 标准错误日志路径，%j 替换为 job ID

#SBATCH --time=48:00:00
# 作业最长运行时间 48 小时

#SBATCH --ntasks=1
# 只启动 1 个任务（1 个主进程）

#SBATCH --cpus-per-task=32
# 每个任务分配 32 个 CPU 核心

#SBATCH --mem=64G
# 申请 64 GB 内存

set -euo pipefail
# 严格模式：
#   -e 任何命令失败就退出
#   -u 使用未定义变量报错
#   -o pipefail 管道中任一段失败则整个管道失败

# ---- Parameters (edit here or override via env) ----
# 参数区：下面这些参数可在此修改，或用环境变量覆盖

TASKS_DIR="${TASKS_DIR:-rl_data/output/tasks_skill_tax_20260327_toy}"
# 任务目录，默认 rl_data/output/tasks_skill_tax_20260327_toy

MODEL="${MODEL:-gemini/gemini-3-flash-preview}" #gemini-3-flash-preview, gemini-3.1-flash-lite-preview, "gemini/gemini-3.1-pro-preview"
# 使用的模型，默认 gemini/gemini-3-flash-preview，注释列出其他可选模型

NUM_SOLUTIONS="${NUM_SOLUTIONS:-8}"
# 每个任务生成多少个解，默认 8（pass@8）

MAX_ACTIONS="${MAX_ACTIONS:-16}" # max turns
# agent 每个解最多执行多少轮动作，默认 16 轮

MAX_TOKENS="${MAX_TOKENS:-65536}"
# 单次 LLM 生成最大 token 数，默认 65536

NUM_TASKS="${NUM_TASKS:-10}"
# 本次处理多少个任务，默认 10

START_AT="${START_AT:-0}"
# 从第几个任务开始，默认 0，用于分批或续跑

WORKERS="${WORKERS:-10}"                   # parallel tasks (each runs NUM_SOLUTIONS agent loops)
# 并行处理多少个任务，默认 10；每个任务内部跑 NUM_SOLUTIONS 个 agent 循环

NUM_POOL_WORKERS="${NUM_POOL_WORKERS:-128}"        # parallel LLM calls within each turn
# 每个 turn 内部并行的 LLM 调用数上限，默认 128

SOLUTION_TEMPERATURE="${SOLUTION_TEMPERATURE:-0.7}"
# 解生成时的采样温度，默认 0.7

COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-600}"         # per-command timeout in seconds inside containers
# 容器内每条命令的超时时间（秒），默认 600

                            # (was 30 — too aggressive once v2 corpus tasks
                            # with apt/pip-heavy setup.sh joined the mix)
# 续注释：以前是 30 秒，v2 语料里有 apt/pip 重的 setup，30 秒太激进

# First shell prompt: under WORKERS×NUM_SOLUTIONS concurrent Apptainers, raise if you see "Shell init timed out"
SHELL_INIT_TIMEOUT=120
# 容器 shell 初始化超时，默认 120 秒
# 提示：在 WORKERS×NUM_SOLUTIONS 并发 Apptainer 下，若见 "Shell init timed out" 就调大

SHELL_INIT_ATTEMPTS=3
# shell 初始化失败时的重试次数，默认 3

BUILD_WORKERS=1             # concurrent SIF builds in pre-pass (1 = serial, safe; bump to 2-3 if I/O allows)
# 预构建 SIF 时的并发 worker 数，默认 1（串行、安全）；I/O 允许可调到 2-3

BUILD_RETRIES=3             # retries per SIF build with exponential backoff
# 每次 SIF 构建失败后的重试次数，默认 3，带指数退避

BASE_SIFS_DIR="rl_data/containers"  # shared base SIFs; set empty to use per-task SIF builds
# 共享基础 SIF 目录，默认 rl_data/containers；设为空则改用每任务单独构建 SIF

FORCE_RERUN=1               # set to 1 to re-run all tasks even if *_summary.json exists
# 设为 1 时，即使任务已有 *_summary.json 也强制重跑

LOG_COMMANDS=0              # 1: append bash I/O to per-task log dir (default: solutions/debug_commands)
# 设为 1 时，把 bash I/O 追加到每任务日志目录（默认 solutions/debug_commands）

# COMMAND_LOG_DIR=output/debug_commands   # optional; relative to each task dir if not absolute
# 可选：自定义命令日志目录；非绝对路径则相对于每个任务目录

# Full copy of stdout+stderr from this Python process (see also SBATCH --output above):
DISABLE_TERMINAL_LOG=0      # set to 1 to skip --terminal-log
# 是否禁用 terminal log，默认 0（启用）；设为 1 则跳过 --terminal-log

# Each run gets a unique log: <model>_<timestamp>.log
_MODEL_TAG=$(echo "$MODEL" | tr '/' '_')
# 把模型名里的 / 替换成 _，用于日志文件名

_RUN_TS=$(date -u +%Y%m%d_%H%M%S)
# 生成 UTC 时间戳，格式 YYYYMMDD_HHMMSS

TERMINAL_LOG="${TASKS_DIR}/logs/${_MODEL_TAG}_${_RUN_TS}.log"
# 拼接 terminal log 路径：任务目录下 logs/<模型标签>_<时间戳>.log

# --------------------------------
# 参数区结束分隔线

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 获取当前脚本所在目录的绝对路径

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# 从脚本目录往上三层，得到项目根目录

cd "$PROJECT_ROOT"
# 切换到项目根目录

mkdir -p logs
# 创建 logs 目录，用于 Slurm 输出

# OCI blob cache on GPFS (persistent across jobs — avoids re-pulling from Docker Hub)
export APPTAINER_CACHEDIR="/gpfs/projects/h2lab/osey/apptainer_cache"
# 设置 Apptainer 缓存目录到 GPFS，跨作业持久，避免重复从 Docker Hub 拉取

# Build scratch on local NVMe (fast I/O, ephemeral — cleaned up when allocation ends)
export APPTAINER_TMPDIR="/tmp/apptainer_tmp"
# 设置 Apptainer 临时目录到本地 NVMe，I/O 快，作业结束后清理

mkdir -p "$APPTAINER_TMPDIR"
# 确保 Apptainer 临时目录存在

EXTRA_ARGS=()
# 初始化额外参数数组

if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--force-rerun)
fi
# 若 FORCE_RERUN=1，追加 --force-rerun

if [[ "${LOG_COMMANDS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--log-commands)
fi
# 若 LOG_COMMANDS=1，追加 --log-commands

if [[ -n "${COMMAND_LOG_DIR:-}" ]]; then
  EXTRA_ARGS+=(--command-log-dir "$COMMAND_LOG_DIR")
fi
# 若设置了 COMMAND_LOG_DIR，追加 --command-log-dir

if [[ -n "${BASE_SIFS_DIR:-}" ]]; then
  EXTRA_ARGS+=(--base-sifs-dir "$BASE_SIFS_DIR")
fi
# 若 BASE_SIFS_DIR 非空，追加 --base-sifs-dir

if [[ "${DISABLE_TERMINAL_LOG:-0}" != "1" ]]; then
  TL="${TERMINAL_LOG:-logs/gen_solutions_terminal.log}"
  if [[ "$TL" != /* ]]; then
    TL="$PROJECT_ROOT/$TL"
  fi
  mkdir -p "$(dirname "$TL")"
  EXTRA_ARGS+=(--terminal-log "$TL")
fi
# 若未禁用 terminal log：
#   取 TERMINAL_LOG，缺省为 logs/gen_solutions_terminal.log
#   如果是相对路径，拼到项目根下
#   创建日志父目录
#   追加 --terminal-log <路径>

uv run python -m rl_data.generate_solutions \
# 用 uv run 运行 rl_data.generate_solutions 模块

    --tasks-dir "$TASKS_DIR" \
# 任务目录

    --model "$MODEL" \
# 模型名

    --num-solutions "$NUM_SOLUTIONS" \
# 每任务的解数

    --max-actions "$MAX_ACTIONS" \
# 最大动作轮数

    --max-tokens "$MAX_TOKENS" \
# 最大 token 数

    --num-tasks "$NUM_TASKS" \
# 处理任务数

    --start-at "$START_AT" \
# 起始任务索引

    --workers "$WORKERS" \
# 并行任务数

    --num-pool-workers "$NUM_POOL_WORKERS" \
# turn 内并行 LLM 调用数

    --solution-temperature "$SOLUTION_TEMPERATURE" \
# 解生成温度

    --command-timeout "$COMMAND_TIMEOUT" \
# 容器内命令超时

    --shell-init-timeout "$SHELL_INIT_TIMEOUT" \
# shell 初始化超时

    --shell-init-attempts "$SHELL_INIT_ATTEMPTS" \
# shell 初始化重试次数

    --build-workers "$BUILD_WORKERS" \
# SIF 构建并发数

    --build-retries "$BUILD_RETRIES" \
# SIF 构建重试次数

    --verbose \
# 详细日志

    "${EXTRA_ARGS[@]}"
# 展开之前拼接的额外参数