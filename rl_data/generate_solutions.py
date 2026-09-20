"""Build, test, and run solutions for generated tasks."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, TextIO

from tqdm import tqdm

from rl_data import DEFAULT_MODEL
from rl_data.generate_tasks import _safe_write_text
from rl_data.generator.env import _fakeroot_flags
from rl_data.generator.sample_solutions import run_n_solutions
from rl_data.generator.vanillux_solver import run_n_solutions_vanillux


def _summary_basename(model: str, harness: str, thinking: bool = False) -> str:
    """Return the per-task solutions-summary filename for a (model, harness, thinking) triple.

    Bash, no-thinking runs preserve the legacy ``<MODEL_TAG>_summary.json`` name
    byte-for-byte so existing summaries in skill_tax 1k / 10k corpora keep
    matching. Non-bash runs (currently just ``vanillux``) get a harness suffix,
    and runs with reasoning (``<think>...</think>``) traces enabled get an
    additional ``_thinking`` suffix, so all four (harness, thinking) combinations
    can coexist on disk in the same task dir without overwriting one another.

    Resulting filenames:
      * ``<MODEL_TAG>_summary.json``                       — bash, thinking off
      * ``<MODEL_TAG>_thinking_summary.json``              — bash, thinking on
      * ``<MODEL_TAG>_<harness>_summary.json``             — non-bash, thinking off
      * ``<MODEL_TAG>_<harness>_thinking_summary.json``    — non-bash, thinking on
    """
    model_tag = model.replace("/", "_")
    parts = [model_tag]
    if harness != "bash":
        parts.append(harness)
    if thinking:
        parts.append("thinking")
    parts.append("summary.json")
    return "_".join(parts)


@dataclass
class SolutionConfig:
    """Configuration for running solutions on tasks."""

    tasks_dir: str
    num_solutions: int = 128
    max_actions: int = 16
    model: str = DEFAULT_MODEL
    solution_temperature: float = 0.7
    verbose: bool = False
    num_tasks: int = 1
    start_at: int = 0
    num_pool_workers: int = 128
    workers: int = 1
    force_build: bool = False
    max_tokens: int = 65536
    filter_solved: bool = False
    use_parquet: bool = False
    command_timeout: float = 30.0
    #: If False, skip task dirs that already have a `*_summary.json` (default).
    #: Set True (--force-rerun) to regenerate solutions and overwrite summaries.
    force_rerun: bool = False
    shell_init_timeout: float = 120.0
    shell_init_attempts: int = 3
    log_commands: bool = False
    #: Relative to each task dir if not absolute; default when log_commands: solutions/debug_commands
    command_log_dir: Optional[str] = None
    #: If set, copy everything printed to stdout/stderr (terminal) into this file (append).
    terminal_log: Optional[str] = None
    #: Concurrency limit for the SIF build pre-pass (default 1 = serial; safe for shared cache).
    build_workers: int = 1
    #: Retries per SIF build (with exponential backoff). Transient failures are common under load.
    build_retries: int = 3
    #: Directory containing pre-built base SIFs (base_{domain}.sif). When set, per-task SIFs
    #: are not needed; the env uses a shared base + task-specific delta script.
    base_sifs_dir: Optional[str] = None
    #: Random-sample at most this many tasks from ``tasks_dir`` (0 = disabled;
    #: use ``num_tasks``/``start_at`` for sequential sampling instead).
    #: Applied **after** ``filter_solved`` and ``use_parquet`` so the random
    #: subsample is drawn from the already-filtered set.
    sample_size: int = 0
    #: Seed for the random sample; keep fixed across runs for reproducibility.
    sample_seed: int = 0
    #: Solution-sampling harness. ``"bash"`` (default) is the legacy
    #: bash-tool-calling harness in :func:`sample_solutions.run_n_solutions`
    #: (terse system prompt, 16-action default).
    #: ``"vanillux"`` switches to the mini-swe-agent-style bash-tool harness
    #: in :func:`vanillux_solver.run_n_solutions_vanillux` (vendored prompts
    #: from upstream mini-swe-agent's ``config/default.yaml``, 64-action
    #: default, head/tail observation truncation). Both share the same single
    #: ``bash`` tool surface and the same ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``
    #: submit marker, so the A/B is a clean prompt-richness + budget test.
    harness: str = "bash"
    #: Whether the underlying chat completions are being sampled with reasoning
    #: traces (``<think>...</think>``) enabled. Purely a NAMING knob from this
    #: module's perspective: it adds a ``_thinking`` infix to the per-task
    #: summary filename so thinking and non-thinking trajectories don't clobber
    #: each other on shared task dirs (see :func:`_summary_basename`). The
    #: actual on/off switch lives in the vLLM launcher / litellm extra-body
    #: (``LITELLM_EXTRA_BODY_JSON={"chat_template_kwargs": {"enable_thinking": true}}``);
    #: pass ``--thinking`` here ONLY when that's also set, otherwise the
    #: filename will misrepresent the contents.
    thinking: bool = False


class _TeeTextStream:
    """Write to the real terminal stream and to a log file (for debugging full run output)."""

    def __init__(self, primary: TextIO, log_file: TextIO) -> None:
        self._primary = primary
        self._log = log_file

    def write(self, data: str) -> int:
        n = self._primary.write(data)
        self._primary.flush()
        self._log.write(data)
        self._log.flush()
        return int(n) if isinstance(n, int) else len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return getattr(self._primary, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._primary.fileno()


def _patch_def_chmod(def_path: Path) -> None:
    """Ensure ``/home/user`` exists with 755 perms by the end of %post.

    shellagent-wide convention is that every container exposes a writable
    ``/home/user``. Most base SIFs / task defs create it themselves, but we
    defensively inject a ``mkdir -p /home/user && chmod 755 /home/user``
    idiom at the top of %post so adapters that skip the convention (e.g.
    upstream Dockerfiles that only create ``/workspace``) still build. The
    previous form (``chmod 755 /home/user`` alone) would fail on such
    containers because ``chmod`` errors when the path doesn't exist yet.
    """
    patch_line = "mkdir -p /home/user && chmod 755 /home/user"
    with open(def_path, "r") as f:
        def_text = f.read()
    # Already-patched (new or legacy form) -> nothing to do.
    if patch_line in def_text or "chmod 755 /home/user" in def_text:
        return
    section_headers = [line for line in def_text.split("\n") if line.strip().startswith("%")]
    post_idx = [i for i, line in enumerate(section_headers) if "post" in line.lower()]
    if post_idx:
        idx = post_idx[0]
        if idx + 1 < len(section_headers):
            next_header = section_headers[idx + 1]
            def_text = def_text.replace(
                next_header, f"    {patch_line}\n" + next_header
            )
        else:
            def_text = def_text.rstrip() + f"\n    {patch_line}\n"
        with open(def_path, "w") as f:
            f.write(def_text)


def build_sif(
    sif_path: Path,
    def_path: Path,
    *,
    retries: int = 3,
    timeout: int = 300,
    verbose: bool = False,
) -> tuple[bool, str]:
    """Build a SIF from a .def with retries and exponential backoff.

    Returns (success, error_message_or_empty).

    On ``subprocess.TimeoutExpired`` we bail **without retrying**: a per-attempt
    wall-clock timeout usually means the def is installing something huge
    (e.g. ``build-essential``, ~400 MB) over a slow link, and retrying will
    burn another full timeout for the same outcome. The caller gets a clean
    ``(False, ...)`` so the overall pre-build phase continues with other tasks.
    """
    _patch_def_chmod(def_path)
    for attempt in range(1, retries + 1):
        try:
            proc = subprocess.run(
                ["apptainer", "build", "--force", str(sif_path), str(def_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            msg = (f"apptainer build exceeded {timeout}s on attempt {attempt}/"
                   f"{retries}; skipping task to avoid blocking other builds.")
            if verbose:
                print(f"⏱️  [{sif_path.parent.name}] {msg}")
            return False, msg
        if proc.returncode == 0:
            return True, ""
        err = (proc.stdout or "") + (proc.stderr or "")
        short_err = err.strip()[-500:] if err.strip() else "(no output)"
        if verbose:
            print(f"⚠️  [{sif_path.parent.name}] build attempt {attempt}/{retries} "
                  f"failed (exit {proc.returncode}): {short_err}")
        if attempt < retries:
            delay = 2 ** attempt
            time.sleep(delay)
    return False, f"Apptainer build failed after {retries} attempts"


def build_and_test(
    sif_path: Path, def_path: Path, test_py: str, run_initial_tests: bool = True
) -> tuple[bool, str]:
    """Build container and optionally run initial tests."""
    ok, msg = build_sif(sif_path, def_path, verbose=True)
    if not ok:
        return False, msg

    if not run_initial_tests:
        return True, ""

    test_file = sif_path.parent / "test_initial_state.py"
    test_file.write_text(test_py)

    proc = subprocess.run(
        [
            "apptainer", "exec",
            *_fakeroot_flags(), "--userns",
            "--writable-tmpfs", "--cleanenv",
            str(sif_path),
            "pytest", "-q", str(test_file.name),
        ],
        capture_output=True,
        text=True,
    )

    return proc.returncode == 0, proc.stdout + proc.stderr


def process_task(task_dir: str, cfg: SolutionConfig):  # 定义函数：处理单个任务；task_dir 是任务目录字符串，cfg 是 SolutionConfig 配置对象
    """Process a single task: build, test, run solutions, and cleanup."""  # 文档字符串：处理单个任务，包括构建、测试、运行解答和清理
    task_dir = Path(task_dir)  # 将任务目录字符串转换为 pathlib.Path 对象，便于后续路径操作
    print(f"\nProcessing task: {task_dir.name}")  # 打印当前正在处理的任务名称，前面加换行

    sif_path = task_dir / "container.sif"  # 拼接得到容器 SIF 文件路径
    def_path = task_dir / "container.def"  # 拼接得到容器定义文件路径
    initial_test_path = task_dir / "test_initial_state.py"  # 拼接得到初始状态测试脚本路径
    final_test_path = task_dir / "test_final_state.py"  # 拼接得到最终状态测试脚本路径
    task_json_path = task_dir / "task.json"  # 拼接得到任务描述 JSON 文件路径
    solutions_dir = task_dir / "solutions"  # 拼接得到解答输出目录路径

    if not cfg.base_sifs_dir:  # 如果配置中没有基础 SIF 目录，则走本地构建流程
        print(f"{task_dir} sif_path: {sif_path}")  # 打印任务目录和 SIF 文件路径
        if not sif_path.exists():  # 如果 SIF 文件不存在
            if not def_path.exists():  # 如果容器定义文件也不存在
                print(f"[{task_dir.name}] No def file found, skipping.")  # 打印没有 def 文件，跳过任务
                return "no def"  # 返回状态字符串 "no def"
            print(f"[{task_dir.name}] Building SIF from def...")  # 打印开始从 def 文件构建 SIF
            ok, msg = build_and_test(sif_path, def_path, initial_test_path.read_text(), run_initial_tests=False)  # 调用构建和测试函数；传入 SIF 路径、def 路径、初始测试内容，不运行初始测试；返回是否成功和消息
            if not ok:  # 如果构建或测试失败
                print(f"[{task_dir.name}] SIF build failed: {msg}")  # 打印 SIF 构建失败信息
                return "no sif"  # 返回状态字符串 "no sif"
    else:  # 如果配置了基础 SIF 目录
        if not def_path.exists():  # 只检查容器定义文件是否存在
            print(f"[{task_dir.name}] No def file found, skipping.")  # 不存在则打印跳过信息
            return "no def"  # 返回状态字符串 "no def"

    pass_at_k = None  # 初始化 pass@k 结果变量为 None

    try:  # 开始异常捕获块，处理后续解答采样流程
        print(f"[{task_dir.name}] Running {cfg.num_solutions} solutions...")  # 打印即将运行多少个解答
        solutions_dir.mkdir(exist_ok=True)  # 创建 solutions 目录；如果已存在则忽略错误

        cmd_log_resolved: Optional[Path] = None  # 声明命令日志解析后的路径变量，类型为 Optional[Path]，初值为 None
        if cfg.log_commands:  # 如果配置开启记录命令日志
            if cfg.command_log_dir:  # 如果配置指定了命令日志目录
                p = Path(cfg.command_log_dir).expanduser()  # 将配置的日志目录转为 Path 对象，并展开 ~ 为用户目录
                cmd_log_resolved = p if p.is_absolute() else (task_dir / p)  # 如果 p 是绝对路径则直接使用，否则相对任务目录拼接
            else:  # 如果未指定命令日志目录
                cmd_log_resolved = solutions_dir / "debug_commands"  # 默认使用 solutions/debug_commands 作为日志目录
            cmd_log_resolved = cmd_log_resolved.resolve()  # 将日志路径解析为绝对路径
            print(f"[{task_dir.name}] Command debug logs -> {cmd_log_resolved}")  # 打印命令调试日志最终输出位置

        # Pick the solution-sampling harness. ``bash`` (default) reproduces  # 英文注释：选择解答采样执行框架；bash 为默认，可复现旧流程
        # the legacy 1k/10k pipeline byte-for-byte; ``vanillux`` switches to  # 英文注释：vanillux 会切换到另一种框架
        # the mini-swe-agent-style bash-tool harness used for v2 RL solutions  # 英文注释：该框架是用于 v2 强化学习解答的 mini-swe-agent 风格 bash 工具框架
        # (see vanillux_solver.py for the architectural rationale).  # 英文注释：架构设计理由见 vanillux_solver.py
        if cfg.harness == "vanillux":  # 如果配置的 harness 是 "vanillux"
            solver_fn = run_n_solutions_vanillux  # 使用 vanillux 版本的解答采样函数
        elif cfg.harness == "bash":  # 否则如果配置的 harness 是 "bash"
            solver_fn = run_n_solutions  # 使用 bash 版本的解答采样函数
        else:  # 如果是其他未知值
            raise ValueError(  # 抛出 ValueError 异常
                f"Unknown harness {cfg.harness!r}; expected 'bash' or 'vanillux'."  # 异常信息：未知 harness，期望 bash 或 vanillux
            )
        summary = solver_fn(  # 调用选定的解答采样函数，返回汇总结果
            num_solutions=cfg.num_solutions,  # 要生成的解答数量
            container_sif_path=str(sif_path),  # 容器 SIF 文件路径字符串
            initial_test_path=str(initial_test_path),  # 初始状态测试脚本路径字符串
            final_test_path=str(final_test_path),  # 最终状态测试脚本路径字符串
            def_path=str(def_path),  # 容器定义文件路径字符串
            task_path=str(task_json_path),  # 任务 JSON 文件路径字符串
            max_actions=cfg.max_actions,  # 最大动作数
            model=cfg.model,  # 使用的模型
            temperature=cfg.solution_temperature,  # 解答采样温度
            max_tokens=cfg.max_tokens,  # 最大 token 数
            save_dir=str(solutions_dir),  # 解答保存目录
            verbose=cfg.verbose,  # 是否输出详细日志
            num_pool_workers=cfg.num_pool_workers,  # 并行工作进程数量
            run_initial_tests=False,  # 不运行初始测试
            command_timeout=cfg.command_timeout,  # 命令超时时间
            shell_init_timeout=cfg.shell_init_timeout,  # shell 初始化超时时间
            shell_init_attempts=cfg.shell_init_attempts,  # shell 初始化尝试次数
            log_commands=cfg.log_commands,  # 是否记录命令日志
            command_log_dir=str(cmd_log_resolved) if cmd_log_resolved else None,  # 命令日志目录字符串；如果未解析出则为 None
            base_sifs_dir=cfg.base_sifs_dir,  # 基础 SIF 目录
        )

        summary_name = _summary_basename(cfg.model, cfg.harness, cfg.thinking)  # 根据模型、harness 和 thinking 配置生成汇总文件名
        _safe_write_text(  # 调用安全写入文本函数
            task_dir / "solutions" / summary_name,  # 写入路径：任务目录下的 solutions 目录中的汇总文件
            json.dumps(summary, indent=4),  # 将 summary 字典序列化为缩进 4 个空格的 JSON 字符串
        )
        pass_at_k = summary.get("pass_at_k", {})  # 从汇总结果中获取 pass_at_k 字段；如果不存在则使用空字典

    except Exception as exc:  # 捕获 try 块中发生的任何异常
        print(f"[{task_dir.name}] ❌ Task failed: {exc}")  # 打印任务失败信息和异常内容
        return "error"  # 返回状态字符串 "error"

    finally:  # 无论是否发生异常，都会执行的清理块
        if sif_path.exists():  # 如果 SIF 文件存在
            print(f"[{task_dir.name}] Not deleting SIF file.")  # 打印不删除 SIF 文件；代码实际也没有删除

    return pass_at_k  # 返回 pass@k 结果


def parse_args(argv: Optional[List[str]] = None) -> SolutionConfig:
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description="Build, test, and run solutions for generated tasks."
    )
    ap.add_argument("--tasks-dir", type=str, required=True, help="Directory containing generated tasks")
    ap.add_argument("--start-at", type=int, default=0, help="Start at task number")
    ap.add_argument("--num-tasks", type=int, default=200, help="Number of tasks to process")
    ap.add_argument("--num-solutions", type=int, default=16, help="Number of solution attempts per task")
    ap.add_argument("--max-actions", type=int, default=16, help="Max shell actions per solution attempt")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--solution-temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=65536, help="Max tokens for the solution agent")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    ap.add_argument("--num-pool-workers", type=int, default=128, help="Number of pool workers")
    ap.add_argument("--workers", type=int, default=1, help="Number of concurrent tasks to process")
    ap.add_argument("--force-build", action="store_true", help="Force build the SIF file")
    ap.add_argument("--filter-solved", action="store_true", help="Only solve tasks that have been solved already")
    ap.add_argument("--use-parquet", action="store_true", help="Use parquet file for tasks")
    ap.add_argument("--command-timeout", type=float, default=30.0, help="Per-command timeout in seconds inside containers (default: 30)")
    ap.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run solution generation even when a *_summary.json already exists (overwrites on success)",
    )
    ap.add_argument(
        "--shell-init-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for Apptainer interactive shell init marker (raise if many concurrent containers)",
    )
    ap.add_argument(
        "--shell-init-attempts",
        type=int,
        default=3,
        help="Retries if shell init times out (default: 3)",
    )
    ap.add_argument(
        "--log-commands",
        action="store_true",
        help="Write each container bash command and raw output to per-solution log files under the task (see --command-log-dir)",
    )
    ap.add_argument(
        "--command-log-dir",
        type=str,
        default=None,
        help="Directory for command debug logs: absolute path, or relative to each task folder (default: <task>/solutions/debug_commands)",
    )
    ap.add_argument(
        "--terminal-log",
        type=str,
        default=None,
        metavar="PATH",
        help="Append full stdout/stderr of this process (everything on your central terminal) to PATH; still prints to terminal",
    )
    ap.add_argument(
        "--build-workers",
        type=int,
        default=1,
        help="Max concurrent SIF builds in the pre-build phase (default: 1 = serial, safe for shared cache/tmp)",
    )
    ap.add_argument(
        "--build-retries",
        type=int,
        default=3,
        help="Retries per SIF build with exponential backoff (default: 3)",
    )
    ap.add_argument(
        "--base-sifs-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory with pre-built base_{domain}.sif files. When set, per-task SIF builds "
             "are skipped; the env uses a shared base SIF + task-specific delta script.",
    )
    ap.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Random-sample at most N tasks from --tasks-dir (0 = disabled). "
             "Great for cost-bounded comparison runs against large baselines.",
    )
    ap.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for --sample-size so reruns pick the same tasks (default: 0).",
    )
    ap.add_argument(
        "--harness",
        type=str,
        default="bash",
        choices=["bash", "vanillux"],
        help=(
            "Solution-sampling harness. 'bash' (default) reproduces the legacy "
            "bash-tool-calling pipeline byte-for-byte (terse prompt, 16-action "
            "budget). 'vanillux' switches to the mini-swe-agent-style harness "
            "(vendored upstream prompts, 64-action budget, head/tail observation "
            "truncation; same single bash tool). Used for v2 RL solutions."
        ),
    )
    ap.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "Tag the per-task summary filename with a `_thinking` infix so "
            "trajectories sampled with reasoning traces (Qwen3 <think>...</think>) "
            "enabled don't clobber non-thinking summaries on the same task "
            "dir. Use whenever the upstream chat completion is configured "
            "with `enable_thinking=true` (e.g. via "
            "LITELLM_EXTRA_BODY_JSON='{\"chat_template_kwargs\": {\"enable_thinking\": true}}'). "
            "Pass the SAME flag to sft/preprocessing/convert_trajectories.py "
            "at conversion time so it picks the right summary file."
        ),
    )

    args = ap.parse_args(argv)
    return SolutionConfig(**vars(args))


_BOOTSTRAP_RE = re.compile(r"^\s*Bootstrap\s*:\s*(\S+)", re.IGNORECASE | re.MULTILINE)
_FROM_RE = re.compile(r"^\s*From\s*:\s*(\S+)", re.IGNORECASE | re.MULTILINE)


def _prepull_base_images(def_paths: list[Path]) -> None:
    """Pre-pull unique Docker base images into the Apptainer OCI cache.

    Without this, every ``apptainer build`` with ``Bootstrap: docker`` fetches
    from Docker Hub independently, quickly exhausting the unauthenticated rate
    limit (100 pulls / 6 h / IP).  A single ``apptainer pull`` per unique image
    populates the cache; subsequent builds reuse it.
    """
    images: set[str] = set()
    for dp in def_paths:
        try:
            text = dp.read_text()
        except OSError:
            continue
        m_bootstrap = _BOOTSTRAP_RE.search(text)
        m_from = _FROM_RE.search(text)
        if m_bootstrap and m_from and m_bootstrap.group(1).lower() == "docker":
            images.add(m_from.group(1))

    if not images:
        return

    print(f"\n📦 Pre-pulling {len(images)} base image(s) into Apptainer cache...")
    for img in sorted(images):
        uri = f"docker://{img}"
        print(f"  pulling {uri} ...", end=" ", flush=True)
        proc = subprocess.run(
            ["apptainer", "pull", "--disable-cache=false", uri],
            capture_output=True,
            text=True,
            timeout=600,
            cwd="/tmp",
        )
        if proc.returncode == 0:
            print("done")
            sif_name = img.replace("/", "_").replace(":", "_") + ".sif"
            sif_artifact = Path("/tmp") / sif_name
            sif_artifact.unlink(missing_ok=True)
        else:
            err = (proc.stderr or "").strip()[-300:]
            print(f"warning ({err or 'exit ' + str(proc.returncode)})")
    print()


def _run_generate_solutions(cfg: SolutionConfig) -> None:  # 定义核心驱动函数，接收 SolutionConfig 配置，无返回值
    """Core driver (stdout/stderr may be teed by main())."""  # 文档字符串：核心驱动，stdout/stderr 可能被 main() 分流
    all_entries = list(Path(cfg.tasks_dir).iterdir())  # 列出任务目录下所有条目，并转成列表
    # Accept either the canonical `task_*` prefix (skill-tax / endless-terminals)  # 原注释：接受规范的 task_* 前缀
    # or any directory that ships a `task.json` (adapter-produced dirs like  # 原注释：也接受任何包含 task.json 的目录
    # `otrl_task_1008`, `otb_*`, etc.). Mirrors the more permissive predicate  # 原注释：例如 adapter 生成的 otrl_task_1008、otb_* 等
    # used by rl_data.comparison.taxonomy_classifier.  # 原注释：与 taxonomy_classifier 中更宽松的判断保持一致
    task_dirs = [  # 开始通过列表推导筛选任务目录
        d  # 当前遍历到的目录条目
        for d in tqdm(all_entries, desc="Scanning task directories", total=len(all_entries))  # 遍历所有条目，并显示扫描进度条
        if d.is_dir()  # 条件一：该条目必须是目录
        and (d.name.startswith("task_") or (d / "task.json").exists())  # 条件二：目录名以 task_ 开头，或目录内存在 task.json
    ]  # 列表推导结束，得到候选任务目录列表

    if cfg.filter_solved:  # 如果配置要求过滤掉已解决任务
        print(f"Filtering to tasks with existing pass@16 > 0, prefilter: {len(task_dirs)}")  # 打印过滤前任务数量

        def _pass16_gt_zero(task_dir: str) -> bool:  # 定义内部函数：判断某任务已有 pass@16 是否大于 0
            task_dir = Path(task_dir)  # 将传入路径转换为 Path 对象
            try:  # 开始异常保护，读取失败时返回 False
                model_summary_path = task_dir / "solutions" / _summary_basename(cfg.model, cfg.harness, cfg.thinking)  # 构造当前模型对应的 summary 路径
                if model_summary_path.exists():  # 如果当前模型 summary 已存在
                    return False  # 返回 False，表示不应保留该任务用于过滤
                # Check any existing summary  # 原注释：检查任意已存在的 summary
                summaries = list((task_dir / "solutions").glob("*_summary.json"))  # 查找 solutions 下所有 summary JSON 文件
                if not summaries:  # 如果一个 summary 都没有
                    return False  # 返回 False
                with open(summaries[0], "r") as f:  # 打开第一个 summary 文件
                    data = json.load(f)  # 读取并解析 JSON 数据
                pass_at_k = data.get("pass_at_k", {})  # 获取 pass_at_k 字段，默认空字典
                value = pass_at_k.get("16") or pass_at_k.get(16)  # 兼容字符串键 "16" 和整数键 16
                if value is None:  # 如果没找到 pass@16 的值
                    return False  # 返回 False
                return float(value) > 0.0  # 将值转 float，判断是否大于 0
            except Exception:  # 捕获任何异常
                return False  # 异常时返回 False

        from concurrent.futures import ThreadPoolExecutor, as_completed  # 导入线程池和 as_completed，用于并发读取 summary

        with ThreadPoolExecutor(max_workers=len(task_dirs)) as executor:  # 创建线程池，最大线程数为任务目录数量
            futures = {executor.submit(_pass16_gt_zero, d): i for i, d in enumerate(task_dirs)}  # 为每个任务目录提交判断任务，并记录原始索引
            mask = [False] * len(task_dirs)  # 初始化布尔掩码，默认全部 False
            with tqdm(total=len(task_dirs), desc="Reading summaries") as pbar:  # 创建进度条，显示读取 summary 进度
                for fut in as_completed(futures):  # 按完成顺序遍历 future
                    idx = futures[fut]  # 取出该 future 对应的原始索引
                    try:  # 开始异常保护
                        mask[idx] = fut.result()  # 获取判断结果并写入对应掩码位置
                    except Exception:  # 如果 future 执行异常
                        mask[idx] = False  # 将该任务标记为 False
                    finally:  # 无论成功失败都执行
                        pbar.update(1)  # 进度条加一
        task_dirs = [d for d, ok in zip(task_dirs, mask) if ok]  # 根据掩码过滤任务目录，只保留 ok 为 True 的

        print(f"Filtering to tasks with pass@16 > 0, postfilter: {len(task_dirs)}")  # 打印过滤后任务数量
        time.sleep(5)  # 暂停 5 秒，可能用于等待文件系统或外部状态稳定

    if cfg.use_parquet:  # 如果配置要求从 parquet 数据集读取任务
        from datasets import load_dataset  # 导入 Hugging Face datasets 的 load_dataset

        dataset = load_dataset(  # 加载 parquet 数据集
            "parquet", data_files=os.path.join(cfg.tasks_dir, "train.parquet")  # 指定格式为 parquet，文件为 tasks_dir/train.parquet
        )["train"]  # 取 train 分割
        task_dirs = [d["extra_info"]["task_dir"] for d in dataset]  # 从每条数据中提取 extra_info.task_dir 作为任务目录

    task_dirs = list(sorted(task_dirs))  # 对任务目录排序并转成列表，保证后续处理顺序确定

    # Optional random subsample for cost-bounded runs.  Runs BEFORE the  # 原注释：可选随机抽样，用于控制成本
    # start_at/num_tasks window so the sample is drawn once, then the usual  # 原注释：抽样发生在 start_at/num_tasks 窗口之前
    # slice still applies if someone wants to chunk the sampled set.  # 原注释：如果有人想分块，后续切片仍然适用
    if cfg.sample_size and cfg.sample_size > 0 and cfg.sample_size < len(task_dirs):  # 如果配置了有效且小于总数的抽样数量
        import random as _random  # 导入 random 模块并别名为 _random
        rng = _random.Random(cfg.sample_seed)  # 使用配置的随机种子创建随机数生成器
        task_dirs = rng.sample(task_dirs, cfg.sample_size)  # 从任务目录中无放回随机抽取指定数量
        task_dirs = list(sorted(task_dirs))  # deterministic ordering post-sample  # 抽样后排序，保证顺序确定
        print(f"Random sample (seed={cfg.sample_seed}): {cfg.sample_size} tasks")  # 打印随机抽样信息

    task_dirs = task_dirs[cfg.start_at : min(cfg.start_at + cfg.num_tasks, len(task_dirs))]  # 按 start_at 和 num_tasks 切片，确定本轮处理窗口

    if not task_dirs:  # 如果切片后没有任务目录
        print(f"No task directories found in {cfg.tasks_dir}")  # 打印未找到任务目录
        return  # 直接返回

    # ------------------------------------------------------------------  # 分隔线
    # Pre-build phase  # 原注释：预构建阶段
    # ------------------------------------------------------------------  # 分隔线
    if cfg.base_sifs_dir:  # 如果配置了基础 SIF 目录，则使用基础镜像模式
        # Ensure all 9 base SIFs exist; build any that are missing.  # 原注释：确保所有 9 个基础 SIF 存在，缺失则构建
        from rl_data.generator.apptainer_def_gen import BASE_IMAGES, CONTAINERS_DIR  # 导入基础镜像定义和容器目录常量，CONTAINERS_DIR 当前未使用
        base_dir = Path(cfg.base_sifs_dir).resolve()  # 将基础 SIF 目录转为绝对路径
        missing_bases: list[tuple[Path, Path]] = []  # 初始化缺失基础 SIF 列表，元素为 (sif路径, def路径)
        for domain in BASE_IMAGES:  # 遍历所有基础镜像域
            sif = base_dir / f"base_{domain}.sif"  # 构造该域对应的 SIF 文件路径
            defp = base_dir / f"base_{domain}.def"  # 构造该域对应的 def 文件路径
            if not sif.exists() and defp.exists():  # 如果 SIF 不存在但 def 存在
                missing_bases.append((sif, defp))  # 加入待构建列表
        if missing_bases:  # 如果存在缺失的基础 SIF
            _prepull_base_images([d for _, d in missing_bases])  # 预拉取这些 def 所需的基础镜像
            print(f"\n🔨 Building {len(missing_bases)} missing base SIF(s)...")  # 打印即将构建的缺失基础 SIF 数量
            for sif, defp in tqdm(missing_bases, desc="Building base SIFs"):  # 遍历待构建基础 SIF，并显示进度条
                ok, msg = build_sif(sif, defp, retries=cfg.build_retries, verbose=True)  # 调用 build_sif 构建，带重试和详细输出
                tag = sif.stem  # 取 SIF 文件名主干作为标签
                if ok:  # 如果构建成功
                    print(f"  ✅ {tag}")  # 打印成功标记
                else:  # 如果构建失败
                    print(f"  ❌ {tag}: {msg}")  # 打印失败标记和错误信息
        existing = sum(1 for d in BASE_IMAGES if (base_dir / f"base_{d}.sif").exists())  # 统计已存在的基础 SIF 数量
        print(f"🔨 Base SIFs ready: {existing}/{len(BASE_IMAGES)} (per-task SIF builds skipped)\n")  # 打印基础 SIF 就绪情况，并说明跳过每任务 SIF 构建
    else:  # 否则进入旧版模式：按任务构建 SIF
        # Legacy: build per-task SIFs for tasks that don't have one yet.  # 原注释：旧逻辑，为尚无 SIF 的任务构建每任务 SIF
        to_build: list[tuple[Path, Path]] = []  # 初始化待构建列表，元素为 (sif路径, def路径)
        for td in task_dirs:  # 遍历每个任务目录
            td = Path(td)  # 转为 Path 对象
            sif = td / "container.sif"  # 构造任务容器 SIF 路径
            defp = td / "container.def"  # 构造任务容器 def 路径
            if not sif.exists() and defp.exists():  # 如果 SIF 不存在但 def 存在
                to_build.append((sif, defp))  # 加入待构建列表

        if to_build:  # 如果存在待构建任务
            _prepull_base_images([defp for _, defp in to_build])  # 预拉取所有待构建 def 所需的基础镜像
            print(f"\n🔨 Pre-build phase: {len(to_build)} SIF(s) to build "  # 打印预构建阶段信息：待构建数量
                  f"(workers={cfg.build_workers}, retries={cfg.build_retries})")  # 继续打印并发 worker 数和重试次数

            def _build_one(pair: tuple[Path, Path]) -> tuple[str, bool, str]:  # 定义单个 SIF 构建函数
                sif, defp = pair  # 解包 SIF 路径和 def 路径
                tag = sif.parent.name  # 用 SIF 所在父目录名作为任务标签
                try:  # 开始异常保护
                    ok, msg = build_sif(  # 调用 build_sif 构建 SIF
                        sif, defp,  # 传入 SIF 路径和 def 路径
                        retries=cfg.build_retries,  # 传入重试次数
                        verbose=True,  # 开启详细输出
                    )  # build_sif 调用结束
                except Exception as exc:  # noqa: BLE001 -- never crash the pre-build phase  # 捕获异常，避免预构建阶段崩溃
                    ok, msg = False, f"unexpected exception: {exc!r}"  # 标记失败并记录异常信息
                if ok:  # 如果构建成功
                    print(f"  ✅ {tag}")  # 打印成功标记
                else:  # 如果构建失败
                    print(f"  ❌ {tag}: {msg}")  # 打印失败标记和消息
                return tag, ok, msg  # 返回标签、成功状态和消息

            if cfg.build_workers <= 1:  # 如果构建 worker 数小于等于 1，则串行构建
                for pair in tqdm(to_build, desc="Building SIFs"):  # 遍历待构建列表，显示进度条
                    _build_one(pair)  # 调用单任务构建函数
            else:  # 否则并发构建
                from concurrent.futures import ThreadPoolExecutor, as_completed  # 导入线程池和 as_completed

                with ThreadPoolExecutor(max_workers=cfg.build_workers) as bld_exec:  # 创建构建线程池
                    futs = {bld_exec.submit(_build_one, p): p for p in to_build}  # 提交所有构建任务，并记录对应参数
                    with tqdm(total=len(to_build), desc="Building SIFs") as bld_pbar:  # 创建构建进度条
                        for fut in as_completed(futs):  # 按完成顺序遍历 future
                            try:  # 开始异常保护
                                fut.result()  # 获取结果，触发异常如有
                            except Exception as exc:  # noqa: BLE001  # 捕获异常
                                # Defensive: _build_one already catches everything,  # 原注释：防御性处理，_build_one 已捕获所有异常
                                # but if a future is somehow cancelled or raises we  # 原注释：但如果 future 被取消或抛异常
                                # still want to drain the rest of the batch.  # 原注释：仍希望继续处理剩余批次
                                print(f"  ❌ (future error): {exc!r}")  # 打印 future 级错误
                            finally:  # 无论成功失败都执行
                                bld_pbar.update(1)  # 构建进度条加一

            built = sum(1 for td in task_dirs if (Path(td) / "container.sif").exists())  # 统计已有 SIF 的任务数量
            print(f"🔨 Pre-build done: {built}/{len(task_dirs)} tasks have a SIF\n")  # 打印预构建完成情况

    # ------------------------------------------------------------------  # 分隔线
    # Solution phase (high concurrency)  # 原注释：解生成阶段，高并发
    # ------------------------------------------------------------------  # 分隔线
    model_summary_name = _summary_basename(cfg.model, cfg.harness, cfg.thinking)  # 生成当前模型/框架/思考模式对应的 summary 文件名

    def process_task_with_retry(task_dir: str, cfg: SolutionConfig):  # 定义带重试的单个任务处理函数
        """Wrap per-task retry logic so it can run in parallel."""  # 文档字符串：封装每任务重试逻辑，便于并行运行
        task_dir = Path(task_dir)  # 将任务目录转为 Path 对象

        sol_dir = task_dir / "solutions"  # 构造 solutions 子目录路径
        model_summary = sol_dir / model_summary_name  # 构造当前模型的 summary 文件路径
        if model_summary.exists() and not cfg.force_rerun:  # 如果 summary 已存在且不强制重跑
            print(f"Skipping {task_dir.name} (already has {model_summary_name})")  # 打印跳过信息
            return task_dir, "skipped"  # 返回任务目录和 skipped 状态
        if model_summary.exists() and cfg.force_rerun:  # 如果 summary 已存在且强制重跑
            print(f"Re-running {task_dir.name} (--force-rerun; overwriting {model_summary_name})")  # 打印重跑并覆盖信息

        max_retries = 1  # 设置最大重试次数为 1
        result = None  # 初始化结果变量

        while max_retries > 0:  # 当还有重试次数时循环
            result = process_task(task_dir, cfg)  # 调用 process_task 处理该任务
            if result is None:  # 如果结果为 None
                print(f"Retrying task {task_dir.name}...")  # 打印重试信息
                max_retries -= 1  # 重试次数减一
            elif result in ("no def", "no sif", "no initial test"):  # 如果结果表示缺少 def、sif 或初始测试
                print(f"No def, sif, or initial test for task {task_dir.name}, skipping.")  # 打印跳过信息
                break  # 跳出循环，不再重试
            elif result == "error":  # 如果结果为 error
                print(f"Retrying task {task_dir.name} after error...")  # 打印错误后重试信息
                max_retries -= 1  # 重试次数减一
            else:  # 其他情况视为成功结果
                print(f"Pass@k: {result} for task {task_dir.name}")  # 打印该任务的 pass@k 结果
                break  # 跳出循环

        return task_dir, result  # 返回任务目录和处理结果

    if cfg.workers <= 1:  # 如果处理 worker 数小于等于 1，则串行处理
        for task_dir in tqdm(task_dirs, desc="Processing Tasks"):  # 遍历任务目录，显示处理进度条
            process_task_with_retry(task_dir, cfg)  # 调用带重试的任务处理函数
    else:  # 否则并发处理
        from concurrent.futures import ThreadPoolExecutor, as_completed  # 导入线程池和 as_completed

        with ThreadPoolExecutor(max_workers=cfg.workers) as executor:  # 创建处理线程池
            futures = {executor.submit(process_task_with_retry, td, cfg): td for td in task_dirs}  # 提交所有任务处理，并记录对应任务目录
            with tqdm(total=len(task_dirs), desc="Processing Tasks") as pbar:  # 创建处理进度条
                for fut in as_completed(futures):  # 按完成顺序遍历 future
                    try:  # 开始异常保护
                        _td, _res = fut.result()  # 获取 future 结果，解包任务目录和结果
                    finally:  # 无论成功失败都执行
                        pbar.update(1)  # 处理进度条加一


def main() -> None:
    """Main entry point; optionally tee terminal output to a single log file."""
    cfg = parse_args()
    log_f: Optional[TextIO] = None
    if cfg.terminal_log:
        log_path = Path(cfg.terminal_log).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w", encoding="utf-8", errors="replace")
        log_f.write(f"terminal log opened {datetime.now(timezone.utc).isoformat()}\n")
        log_f.write(f"cwd={os.getcwd()}\n")
        log_f.write(f"argv={' '.join(sys.argv)}\n")
        log_f.flush()
        sys.stdout = _TeeTextStream(sys.__stdout__, log_f)
        sys.stderr = _TeeTextStream(sys.__stderr__, log_f)
        print(f"📝 Also logging terminal output to: {log_path}", flush=True)

    try:
        _run_generate_solutions(cfg)
    finally:
        if log_f is not None:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            try:
                log_f.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
