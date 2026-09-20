# ============================================================================
# 功能概述
# ----------------------------------------------------------------------------
# 本模块实现“通过批量 LLM 调用生成任务”的完整 pipeline：
#   task templates -> initial tests -> final tests -> container defs -> save
#
# 分两大阶段：
#   Phase 1（阶段 1-3）：纯 LLM 生成，产出中间结果并 checkpoint 到
#                        <out_dir>/_intermediates.jsonl
#   Phase 2（阶段 4）：生成 Apptainer def 并 build+smoke test，逐条落盘，
#                      进度记录在 <out_dir>/_stage4_done.jsonl
#
# 重要限制（LLM-only ground truth）：
#   truth 和 test_final_state.py 都是模型生成的文本，本模块不真正执行 setup
#   也不重新计算 golden，因此派生量错误、setup 与期望不一致等问题可能漏过。
#   final test 由第二个模型基于 truth 生成，也可能出现抄错或漂移。
#   硬化建议：发布前增加外部校验（执行 setup、参考解或自动检查）。
#   task_template_gen / completion_test_gen 中的 prompt 已编码一致性原则。
# ============================================================================

from __future__ import annotations   # 允许在类型注解中使用前向引用（如 "Foo"）

import argparse                       # 命令行参数解析
import json                           # JSON 读写
import subprocess                     # 调用外部命令（apptainer build）
import time                           # 计时
from dataclasses import dataclass     # 数据类装饰器
from datetime import datetime, timezone  # 时间戳
from pathlib import Path              # 路径对象
from typing import Any, Dict, List, Optional, Tuple  # 类型注解
import uuid                           # 生成随机后缀

from tqdm import tqdm                 # 进度条

from rl_data import DEFAULT_MODEL     # 默认模型名
from rl_data.generator.task_template_gen import generate_templates_batch
# 生成任务模板（阶段 1）

from rl_data.generator.initial_state_test_gen import generate_test_templates_batch as generate_initial_tests_batch
# 生成初始状态测试（阶段 2），重命名为 generate_initial_tests_batch

from rl_data.generator.apptainer_def_gen import iterate_def_template_batch, save_setup_artifacts
# 阶段 4：批量生成 def、build+test；保存 setup 产物

from rl_data.generator.completion_test_gen import generate_test_templates_batch as generate_final_tests_batch
# 生成最终状态测试（阶段 3），重命名为 generate_final_tests_batch

from rl_data.generator.container_def_patch import inject_files_section
# 向 container.def 注入 %files 段

from rl_data.generator.fixture_gen import (
    materialize as fixture_materialize,   # 在宿主机上物化 fixture 文件
    emit_files_section,                    # 生成 %files 段文本
    fixture_seed_for_task,                 # 为任务生成稳定随机种子
    NOOP_FIXTURE_KINDS,                    # 不需要物化的 fixture 类型集合
)


@dataclass
class PipelineConfig:
    """pipeline 基础配置（同步版）。"""
    num_tasks: int                         # 要生成的任务数
    out_dir: Path                          # 输出目录
    max_def_retries: int = 3               # def 生成最大重试次数
    max_num_completions: int = 4           # 最大 completion 数
    num_solutions: int = 256               # 生成解的数量
    max_actions: int = 20                  # 最大动作数
    model: str = DEFAULT_MODEL             # 使用的模型
    max_tokens: int = 32768                # 单次生成最大 token
    task_temperature: float = 1.0          # 任务生成温度
    test_temperature: float = 0.6          # 测试生成温度
    solution_temperature: float = 1.0      # 解生成温度
    parallel_jobs: int = 1                 # 并行作业数
    verbose: bool = False                  # 是否详细日志
    #: 阶段 4 中最大并发 Apptainer build+test worker 数。
    #: 每个 worker 约用 1 CPU + 4 GB RAM。安全默认 4；可按 CPU 扩容。
    def_build_workers: int = 4
    #: 传给 random_user_msg 的 corpus kind。
    #: "legacy"（默认）与 v2 之前字节级一致；
    #: "sft_v2" / "rl_v2" 启用新的 verifier_kind / fixture_kind / intricate
    #: complexity 轴，通过 bucket-upweight sampler 采样。
    corpus_kind: str = "legacy"


def _safe_write_text(path: Path, content: str) -> None:
    """安全写文本：自动创建父目录，以 UTF-8 写入。"""
    path.parent.mkdir(parents=True, exist_ok=True)   # 确保父目录存在
    path.write_text(content, encoding="utf-8")       # 写文件


def _build_sif(def_path: Path, sif_path: Path) -> bool:
    """用 apptainer build 从 def 构建 SIF，返回是否成功。"""
    sif_path.parent.mkdir(parents=True, exist_ok=True)   # 确保 SIF 父目录存在
    try:
        rc = subprocess.run(                             # 调用 apptainer build
            ["apptainer", "build", str(sif_path), str(def_path)],
            stdout=subprocess.DEVNULL,                   # 丢弃 stdout
            stderr=subprocess.DEVNULL,                   # 丢弃 stderr
        ).returncode                                     # 取返回码
        return rc == 0                                   # 0 表示成功
    except FileNotFoundError:                            # apptainer 不存在
        return False
    except subprocess.TimeoutExpired:                    # 超时
        return False


def _format_task_dir(base: Path, idx: int, width: int = 6) -> Path:
    """生成任务目录名：task_<idx 补零>_<8位随机后缀>。"""
    suffix = uuid.uuid4().hex[:8]                        # 随机 8 位十六进制
    return base / f"task_{idx:0{width}d}_{suffix}"       # 拼接路径


def _save_task_bundle(
    task_dir: Path,
    task_obj: Dict[str, Any],
    initial_test_code: str,
    def_text: str,
    final_test_code: str,
    summary: Dict[str, Any],
) -> Tuple[Path, Path, Path, Path, Path]:
    """把单个任务的各个文件写入磁盘，返回关键路径元组。"""
    task_json = task_dir / "task.json"                   # 任务元数据 JSON
    init_py = task_dir / "test_initial_state.py"         # 初始状态测试
    final_py = task_dir / "test_final_state.py"          # 最终状态测试
    def_file = task_dir / "container.def"                # Apptainer def
    sif_file = task_dir / "container.sif"                # 目标 SIF 路径（此处未构建）
    sol_dir = task_dir / "solutions"                     # 解目录
    sol_dir.mkdir(parents=True, exist_ok=True)           # 创建解目录

    _safe_write_text(task_json, json.dumps(task_obj, indent=4))  # 写 task.json
    _safe_write_text(init_py, initial_test_code)                 # 写初始测试
    _safe_write_text(final_py, final_test_code)                  # 写最终测试
    _safe_write_text(def_file, def_text)                         # 写 def
    _safe_write_text(sol_dir / "summary.json", json.dumps(summary, indent=4))  # 写 summary

    domain = task_obj.get("domain", "software_engineering")      # 取 domain
    save_setup_artifacts(task_dir, def_text, domain)             # 保存 setup 产物

    return task_json, init_py, final_py, def_file, sif_file      # 返回路径元组


@dataclass
class AsyncBatchConfig(PipelineConfig):
    """异步批量配置：在基础配置上增加 batch_size 与 max_concurrency。"""
    batch_size: int = 64          # 每批任务数
    max_concurrency: int = 64     # 最大并发 LLM 调用数


def _generate_intermediates_batch(
    cfg: AsyncBatchConfig, batch_count: int,
) -> List[Dict[str, Any]]:
    """运行阶段 1-3（模板、初始测试、最终测试），返回中间结果列表。"""

    # 1) 任务模板
    print(
        f"Generating {batch_count} task templates with {cfg.max_concurrency} "
        f"concurrency (corpus_kind={cfg.corpus_kind})"
    )
    task_templates = generate_templates_batch(          # 批量生成任务模板
        batch_count,
        model=cfg.model,
        temperature=cfg.task_temperature,
        max_tokens=cfg.max_tokens,
        max_concurrency=cfg.max_concurrency,
        corpus_kind=cfg.corpus_kind,
    )

    if not task_templates:                              # 没有生成任何模板
        print("No task templates generated")
        return []

    descriptions: List[str] = [t.get("description", "").strip() for t in task_templates]
    # 提取 description 并去空白

    truths: List[str] = [t.get("truth", "").strip() for t in task_templates]
    # 提取 truth 并去空白

    meta: List[Dict[str, Any]] = [                      # 提取元数据
        {
            "domain": t.get("domain", ""),
            "skill_type": t.get("skill_type", ""),
            "primitive_skills": t.get("primitive_skills", []),
            "task_complexity": t.get("task_complexity", ""),
            "command_complexity": t.get("command_complexity", ""),
            "scenario": t.get("scenario", ""),
            "language": t.get("language", ""),
            "anchor": t.get("anchor"),
            # v2 轴：每个模板都有；legacy 模板携带 legacy 默认值
            # ("exact_text", "text_only", "legacy", None)。
            "verifier_kind": t.get("verifier_kind", "exact_text"),
            "fixture_kind": t.get("fixture_kind", "text_only"),
            "corpus_kind": t.get("corpus_kind", "legacy"),
            "base_image": t.get("base_image"),
        }
        for t in task_templates
    ]

    valid_indices = [i for i, (d, tr) in enumerate(zip(descriptions, truths)) if d and tr]
    # 过滤出 description 和 truth 都非空的下标

    if not valid_indices:                               # 没有有效模板
        print("No valid task templates generated")
        return []

    descriptions = [descriptions[i] for i in valid_indices]   # 保留有效 description
    truths = [truths[i] for i in valid_indices]               # 保留有效 truth
    meta = [meta[i] for i in valid_indices]                   # 保留有效 meta

    print(f"Task templates generated: {len(descriptions)}")

    # 2) 初始测试（批量）
    print(f"Generating {len(descriptions)} initial tests with {cfg.max_concurrency} concurrency")
    init_tests = generate_initial_tests_batch(          # 批量生成初始测试
        list(zip(descriptions, truths)),                # 输入 (description, truth) 对
        model=cfg.model,
        temperature=cfg.test_temperature,
        max_tokens=cfg.max_tokens,
        max_concurrency=cfg.max_concurrency,
    )

    valid_indices = [i for i, test in enumerate(init_tests) if test]
    # 过滤出非空初始测试

    descriptions = [descriptions[i] for i in valid_indices]   # 同步过滤
    truths = [truths[i] for i in valid_indices]
    meta = [meta[i] for i in valid_indices]
    init_tests = [init_tests[i] for i in valid_indices]

    print(f"Generated {len(init_tests)} initial tests")

    # 3) 最终测试（批量）
    # 对于 v2 语料，把每个任务的 verifier_kind 作为第 4 个元组元素传入，
    # 这样 completion_test_gen 能按模板选择条件化 system prompt
    # （并对非 legacy verifier kind 允许第三方 import）。
    print(f"Generating {len(descriptions)} final tests with {cfg.max_concurrency} concurrency")
    final_test_items: List[tuple] = [
        (descriptions[i], truths[i], init_tests[i], meta[i].get("verifier_kind", "exact_text"))
        for i in range(len(descriptions))
    ]
    final_tests = generate_final_tests_batch(           # 批量生成最终测试
        final_test_items,
        model=cfg.model,
        temperature=cfg.test_temperature,
        max_tokens=cfg.max_tokens,
        max_concurrency=cfg.max_concurrency,
    )

    print(f"Generated {len(final_tests)} final tests")
    valid_indices = [i for i, test in enumerate(final_tests) if test]
    # 过滤出非空最终测试

    descriptions = [descriptions[i] for i in valid_indices]   # 同步过滤
    truths = [truths[i] for i in valid_indices]
    meta = [meta[i] for i in valid_indices]
    init_tests = [init_tests[i] for i in valid_indices]
    final_tests = [final_tests[i] for i in valid_indices]

    return [                                            # 组装中间结果
        {
            "description": descriptions[i],
            "truth": truths[i],
            "init_test": init_tests[i],
            "final_test": final_tests[i],
            "meta": meta[i],
        }
        for i in range(len(descriptions))
    ]


# ---------------------------------------------------------------------------
# 中间结果 checkpoint 读写
# ---------------------------------------------------------------------------

_INTERMEDIATES_FILENAME = "_intermediates.jsonl"        # 中间结果文件名


def _save_intermediates(path: Path, items: List[Dict[str, Any]]) -> None:
    """覆盖写入中间结果 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)      # 确保父目录存在
    with open(path, "w", encoding="utf-8") as f:         # 以写模式打开
        for item in items:                               # 逐条写
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _append_intermediates(path: Path, items: List[Dict[str, Any]]) -> None:
    """追加写入中间结果 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)      # 确保父目录存在
    with open(path, "a", encoding="utf-8") as f:         # 以追加模式打开
        for item in items:                               # 逐条写
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _load_intermediates(path: Path) -> List[Dict[str, Any]]:
    """从 JSONL 读取中间结果。"""
    items: List[Dict[str, Any]] = []                    # 结果列表
    with open(path, "r", encoding="utf-8") as f:         # 以读模式打开
        for line in f:                                   # 逐行读
            line = line.strip()                          # 去空白
            if line:                                     # 跳过空行
                items.append(json.loads(line))           # 解析 JSON
    return items                                         # 返回列表


# ---------------------------------------------------------------------------
# 把单个任务保存到磁盘
# ---------------------------------------------------------------------------

def _save_one_task(
    item: Dict[str, Any],
    def_text: str,
    out_dir: Path,
    idx: int,
) -> Path:
    """持久化单个任务（中间结果 + def 文本），返回任务目录。"""
    m = item["meta"]                                     # 元数据
    desc = item["description"]                           # 描述
    tr = item["truth"]                                   # 真值

    task_dir = _format_task_dir(out_dir, idx=idx)        # 生成任务目录
    task_obj = {                                         # 组装 task.json 内容
        "name": task_dir.name,
        "domain": m["domain"],
        "skill_type": m["skill_type"],
        "primitive_skills": m["primitive_skills"],
        "task_complexity": m["task_complexity"],
        "command_complexity": m["command_complexity"],
        "scenario": m["scenario"],
        "language": m.get("language", ""),
        "anchor": m.get("anchor"),
        # v2 轴：始终存在，便于下游统一按 legacy/v2 分组分析。
        "verifier_kind": m.get("verifier_kind", "exact_text"),
        "fixture_kind": m.get("fixture_kind", "text_only"),
        "corpus_kind": m.get("corpus_kind", "legacy"),
        # 路由提示，供 env._resolve_runtime_sif 使用。
        # None 保持 legacy 行为（用 base_<domain>.sif）；v2 任务设为 "intricate"。
        "base_image": m.get("base_image"),
        "description": desc,
        "truth": tr,
    }

    # v2：在宿主机物化非 legacy fixture，并向 def 注入 %files 段，
    # 使其被烘焙进每个任务的 SIF。legacy 任务（text_only / 未知类型）为空操作。
    fixture_kind = m.get("fixture_kind", "text_only")
    if fixture_kind not in NOOP_FIXTURE_KINDS:           # 需要物化 fixture
        # 稳定种子（不用 hash()，因为 Python 3 中每进程加盐）。
        fixture_seed = fixture_seed_for_task(idx, task_dir.name)
        fixture_pairs = fixture_materialize(             # 物化 fixture 文件
            fixture_kind,
            task_description=desc,
            truth=tr,
            dest_dir=task_dir,
            seed=fixture_seed,
        )
        if fixture_pairs:                                # 有产物才注入
            files_section = emit_files_section(fixture_pairs)  # 生成 %files 段
            # 把 %files 段插到现有 %post 之前。Apptainer 允许 %files 在
            # %post 前任意位置，但惯例是紧跟在 Bootstrap/From 头之后。
            def_text = inject_files_section(def_text, files_section)

    _save_task_bundle(                                   # 写任务各文件
        task_dir, task_obj, item["init_test"], def_text,
        item["final_test"], summary={},
    )

    skills_str = ", ".join(m["primitive_skills"])        # 拼接技能字符串
    summary_txt = (                                      # 生成可读摘要
        f"Task: {task_dir.name}\n"
        f"Domain: {m['domain']}\n"
        f"Skill Type: {m['skill_type']}\n"
        f"Primitive Skills: {skills_str}\n"
        f"Task Complexity: {m['task_complexity']}\n"
        f"Command Complexity: {m['command_complexity']}\n"
        f"Scenario: {m['scenario']}\n"
        f"\n{'='*60}\n"
        f"DESCRIPTION\n{'='*60}\n\n"
        f"{desc}\n"
        f"\n{'='*60}\n"
        f"GROUND TRUTH\n{'='*60}\n\n"
        f"{tr}\n"
    )
    _safe_write_text(task_dir / "task_summary.txt", summary_txt)  # 写摘要
    return task_dir                                      # 返回任务目录


# ---------------------------------------------------------------------------
# 阶段 4 进度跟踪
# ---------------------------------------------------------------------------

_STAGE4_PROGRESS_FILENAME = "_stage4_done.jsonl"         # 阶段 4 进度文件名


def _load_stage4_done(path: Path) -> Dict[int, str]:
    """加载已完成的阶段 4 索引 → 任务目录映射。"""
    done: Dict[int, str] = {}                            # 结果字典
    if not path.exists():                                # 文件不存在直接返回
        return done
    with open(path, "r", encoding="utf-8") as f:         # 打开进度文件
        for line in f:                                   # 逐行读
            line = line.strip()                          # 去空白
            if line:                                     # 跳过空行
                entry = json.loads(line)                 # 解析 JSON
                done[entry["idx"]] = entry["task_dir"]   # 记录 idx -> dir
    return done                                          # 返回映射


def _append_stage4_done(path: Path, entries: List[Dict[str, Any]]) -> None:
    """追加写入阶段 4 完成记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)      # 确保父目录存在
    with open(path, "a", encoding="utf-8") as f:         # 追加模式打开
        for entry in entries:                            # 逐条写
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 两阶段 pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: AsyncBatchConfig) -> Dict[str, Any]:
    """两阶段生成任务，每个阶段都有 checkpoint。

    **Phase 1 — 阶段 1-3**（纯 LLM：模板、初始测试、最终测试）：
    很快；每批后把结果保存到 ``<out_dir>/_intermediates.jsonl``。
    若进入时该文件已存在，则整个阶段跳过。

    **Phase 2 — 阶段 4**（def 生成 + Apptainer build/test）：
    CPU 密集；对所有中间结果一次性运行。任务在**每轮重试后**落盘
    （流式保存），进度记录在 ``<out_dir>/_stage4_done.jsonl``。
    重启时已完成项自动跳过。
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)       # 确保输出目录存在

    intermediates_path = cfg.out_dir / _INTERMEDIATES_FILENAME  # 中间结果路径
    progress_path = cfg.out_dir / _STAGE4_PROGRESS_FILENAME     # 进度路径
    batch_size = max(1, cfg.batch_size)                  # 批大小至少 1

    # ── Phase 1: 中间结果（阶段 1-3） ──
    if intermediates_path.exists():                      # 已有 checkpoint
        all_intermediates = _load_intermediates(intermediates_path)  # 读取
        print(
            f"Checkpoint found: loaded {len(all_intermediates)} intermediates "
            f"from {intermediates_path} (stages 1-3 skipped)"
        )
    else:                                                # 没有则生成
        all_intermediates: List[Dict[str, Any]] = []     # 中间结果列表
        remaining = cfg.num_tasks                        # 剩余任务数
        num_batches = (cfg.num_tasks + batch_size - 1) // batch_size  # 批数
        for batch_idx in tqdm(range(num_batches), desc="Stages 1-3"):  # 逐批
            count = min(batch_size, remaining)           # 本批数量
            items = _generate_intermediates_batch(cfg, count)  # 生成
            all_intermediates.extend(items)              # 累积
            _append_intermediates(intermediates_path, items)  # 追加 checkpoint
            remaining -= count                           # 更新剩余
        print(
            f"Stages 1-3 complete: {len(all_intermediates)} intermediates "
            f"(saved to {intermediates_path})"
        )

    if not all_intermediates:                            # 没有中间结果
        print("No intermediates to process")
        return {
            "requested": cfg.num_tasks,
            "intermediates": 0,
            "succeeded": 0,
            "success_rate": 0.0,
            "saved_dirs": [],
        }

    # ── Phase 2: def 生成（阶段 4）+ 流式保存 ──
    done_map = _load_stage4_done(progress_path)          # 读取已完成映射
    done_indices = set(done_map.keys())                  # 已完成索引集合
    all_saved_dirs: List[str] = list(done_map.values())  # 已保存目录列表
    round_stats: List[Dict[str, Any]] = []               # 每轮统计

    if done_indices:                                     # 有已完成项
        print(f"Stage 4 checkpoint: {len(done_indices)} items already completed, resuming")

    descriptions = [item["description"] for item in all_intermediates]  # 描述
    truths = [item["truth"] for item in all_intermediates]              # 真值
    init_tests = [item["init_test"] for item in all_intermediates]      # 初始测试
    domains = [item["meta"]["domain"] for item in all_intermediates]    # domain

    n_total = len(all_intermediates)                     # 总数
    n_pending = n_total - len(done_indices)              # 待处理数
    print(
        f"Stage 4: {n_pending} defs to process ({n_total} total, "
        f"{len(done_indices)} already done)\n"
        f"  build_workers={cfg.def_build_workers}, "
        f"llm_concurrency={cfg.max_concurrency}, "
        f"retries={cfg.max_def_retries}"
    )

    stage4_start = time.monotonic()                      # 阶段 4 开始时间
    _round_start = [stage4_start]                        # 当前轮开始时间（列表以便闭包修改）
    _save_lock = __import__("threading").Lock()          # 保存锁，避免并发写冲突

    def _on_item_success(idx: int, def_text: str) -> None:
        """单个任务 build+test 通过后立即落盘。"""
        task_dir = _save_one_task(                       # 保存任务
            all_intermediates[idx], def_text, cfg.out_dir, idx=idx,
        )
        with _save_lock:                                 # 加锁更新共享状态
            all_saved_dirs.append(str(task_dir))         # 记录目录
            done_indices.add(idx)                        # 标记完成
            _append_stage4_done(progress_path, [{"idx": idx, "task_dir": str(task_dir)}])
            # 追加进度记录

    def _on_round_complete(round_idx: int, newly_succeeded: Dict[int, str]) -> None:
        """记录轮级统计（保存已在每项完成时发生）。"""
        round_elapsed = time.monotonic() - _round_start[0]  # 本轮耗时
        _round_start[0] = time.monotonic()               # 重置轮开始时间

        round_stats.append({                             # 记录统计
            "round": round_idx,
            "succeeded_this_round": len(newly_succeeded),
            "cumulative_succeeded": len(all_saved_dirs),
            "remaining": n_total - len(done_indices),
            "elapsed_s": round(round_elapsed, 1),
        })

    iterate_def_template_batch(                          # 批量生成 def 并 build+test
        list(zip(descriptions, truths, init_tests)),     # 输入三元组
        domains=domains,
        model=cfg.model,
        temperature=cfg.test_temperature,
        max_tokens=cfg.max_tokens,
        max_concurrency=cfg.max_concurrency,
        max_retries=cfg.max_def_retries,
        max_build_workers=cfg.def_build_workers,
        skip_indices=done_indices if done_indices else None,  # 跳过已完成
        on_round_complete=_on_round_complete,            # 轮完成回调
        on_item_success=_on_item_success,                # 单项成功回调
    )

    stage4_elapsed = time.monotonic() - stage4_start     # 阶段 4 总耗时

    # ── 写生成日志 ──
    log_path = cfg.out_dir / "_generation_log.txt"       # 日志路径
    n_succeeded = len(all_saved_dirs)                    # 成功数
    n_failed = n_total - n_succeeded                     # 失败数
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")  # 时间戳
    W = 64                                               # 宽度
    sep = "=" * W                                        # 粗分隔线
    thin = "-" * W                                       # 细分隔线

    log_lines = [                                        # 日志行列表
        sep,
        "  TASK GENERATION LOG",
        sep,
        f"  Timestamp:         {ts}",
        f"  Model:             {cfg.model}",
        f"  Max tokens:        {cfg.max_tokens}",
        f"  Task temperature:  {cfg.task_temperature}",
        f"  Test temperature:  {cfg.test_temperature}",
        "",
        sep,
        "  PHASE 1: STAGES 1-3 (LLM-only)",
        sep,
        f"  Tasks requested:   {cfg.num_tasks}",
        f"  Batch size:        {cfg.batch_size}",
        f"  LLM concurrency:   {cfg.max_concurrency}",
        f"  Intermediates:     {n_total}",
        f"  Survival rate:     {n_total / cfg.num_tasks:.1%}" if cfg.num_tasks else "",
        # 存活率，避免除零
        "",
        sep,
        "  PHASE 2: STAGE 4 (def gen + Apptainer build/test)",
        sep,
        f"  Build workers:     {cfg.def_build_workers}",
        f"  Max retries:       {cfg.max_def_retries}",
        f"  Total time:        {stage4_elapsed / 60:.1f} min",
        "",
        f"  {'Round':<8} {'Succeeded':>10} {'Cumulative':>11} {'Remaining':>10} {'Time':>10}",
        f"  {thin}",
    ]
    for rs in round_stats:                               # 逐轮追加
        log_lines.append(
            f"  {rs['round']:<8} {rs['succeeded_this_round']:>10} "
            f"{rs['cumulative_succeeded']:>11} {rs['remaining']:>10} "
            f"{rs['elapsed_s']:>8.1f}s"
        )
    log_lines += [                                       # 追加总结
        f"  {thin}",
        "",
        sep,
        "  SUMMARY",
        sep,
        f"  Requested:         {cfg.num_tasks}",
        f"  Intermediates:     {n_total}",
        f"  Succeeded:         {n_succeeded}",
        f"  Failed:            {n_failed}",
        f"  Overall rate:      {n_succeeded / cfg.num_tasks:.1%}" if cfg.num_tasks else "",
        f"  Output dir:        {cfg.out_dir}",
        sep,
        "",
    ]

    log_text = "\n".join(log_lines)                      # 拼成文本
    _safe_write_text(log_path, log_text)                 # 写日志
    print(log_text)                                      # 同时打印

    return {                                             # 返回 summary
        "requested": cfg.num_tasks,
        "intermediates": n_total,
        "succeeded": n_succeeded,
        "success_rate": (n_succeeded / cfg.num_tasks) if cfg.num_tasks else 0.0,
        "saved_dirs": all_saved_dirs,
    }


def parse_args(argv: Optional[List[str]] = None) -> AsyncBatchConfig:
    """解析命令行参数，返回 AsyncBatchConfig。"""
    ap = argparse.ArgumentParser(description="Generate tasks via async-batched LLM calls.")
    ap.add_argument("--num-tasks", type=int, default=100, help="How many tasks to request")
    # 请求任务数
    ap.add_argument("--out-dir", type=Path, default=Path("tasks"), help="Output directory")
    # 输出目录
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)   # 模型
    ap.add_argument("--task-temperature", type=float, default=1.0)  # 任务温度
    ap.add_argument("--test-temperature", type=float, default=0.6)  # 测试温度
    ap.add_argument("--solution-temperature", type=float, default=1.0)  # 解温度
    ap.add_argument("--batch-size", type=int, default=100)        # 批大小
    ap.add_argument("--max-concurrency", type=int, default=128)   # 最大并发
    ap.add_argument(
        "--def-build-workers", type=int, default=4,
        help="Max concurrent Apptainer build+test workers in stage 4 "
             "(each uses ~1 CPU + ~4 GB RAM; default: 4)",
    )  # 阶段 4 构建 worker 数
    ap.add_argument(
        "--corpus-kind", type=str, default="legacy",
        choices=["legacy", "sft_v2", "rl_v2"],
        help=(
            "Corpus generation mode. 'legacy' (default) reproduces the "
            "pre-v2 pipeline byte-for-byte. 'sft_v2' / 'rl_v2' enable the "
            "verifier_kind / fixture_kind / intricate-complexity axes via "
            "the bucket-upweight sampler (M=2 / M=1.5 respectively)."
        ),
    )  # 语料类型
    ap.add_argument("--verbose", action="store_true")   # 详细日志
    ap.add_argument("--quiet", action="store_true")     # 安静模式

    args = ap.parse_args(argv)                          # 解析
    verbose = args.verbose and not args.quiet           # verbose 且非 quiet 才详细

    return AsyncBatchConfig(                            # 构造配置
        num_tasks=args.num_tasks,
        out_dir=args.out_dir,
        model=args.model,
        task_temperature=args.task_temperature,
        test_temperature=args.test_temperature,
        solution_temperature=args.solution_temperature,
        parallel_jobs=1,
        verbose=verbose,
        batch_size=max(1, args.batch_size),
        max_concurrency=max(1, args.max_concurrency),
        def_build_workers=max(1, args.def_build_workers),
        corpus_kind=args.corpus_kind,
    )


if __name__ == "__main__":                              # 脚本入口
    cfg = parse_args()                                  # 解析参数
    summary = run_pipeline(cfg)                         # 运行 pipeline
    print(json.dumps(summary, indent=4))                # 打印 summary JSON