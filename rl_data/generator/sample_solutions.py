"""Solution sampling & verification using tool-calling format.

Runs N parallel solution attempts inside Apptainer containers, driving an LLM
agent that uses the same bash tool-calling harness as shellagent's SFT training data.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from rl_data import chat_completion_batch_with_tools, DEFAULT_MODEL
from rl_data.generator.env import InteractiveContainerEnvironment as ContainerEnvironment

MAX_OUTPUT_LENGTH = 50_000
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

HARNESS_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "sft" / "preprocessing" / "config"
_DEFAULT_SYSTEM_PROMPT = """You are a helpful coding assistant. You have access to a bash terminal.
Use it to explore the codebase, understand the problem, implement a solution, and verify it works.

IMPORTANT RULES:
- Every response must include a THOUGHT section explaining your reasoning, followed by exactly one bash command.
- Directory or environment variable changes are not persistent. Every command runs in a new subshell. Use `cd /path && <command>` to run commands in a specific directory.
- Edit files using bash commands like `sed`, `cat > file << 'EOF'`, etc.
- Long running commands: Wrap with `timeout`, e.g., `timeout 10 <command>`.
- Interactive commands are not possible. Use `yes`/`no`, etc. as appropriate.
- Output may be truncated. Use `head`/`tail`/`grep` to filter large outputs.
- When you are confident your solution is correct, submit by running: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
- After submitting you cannot continue working on the task.
"""
_DEFAULT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command. Each command runs in a new subshell.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
                "required": ["command"],
            },
        },
    }
]


def _load_harness_config() -> tuple[str, list[dict[str, Any]]]:
    """Load legacy SFT harness config, falling back to the checked-in shellagent defaults."""
    system_prompt_path = HARNESS_CONFIG_DIR / "system_prompt.txt"
    tool_schemas_path = HARNESS_CONFIG_DIR / "tool_schemas.json"
    if system_prompt_path.is_file() and tool_schemas_path.is_file():
        return (
            system_prompt_path.read_text(encoding="utf-8").strip(),
            json.loads(tool_schemas_path.read_text(encoding="utf-8")),
        )
    return _DEFAULT_SYSTEM_PROMPT.strip(), _DEFAULT_TOOL_SCHEMAS


SYSTEM_PROMPT, TOOL_SCHEMAS = _load_harness_config()

# Max characters per log entry (full stdout/stderr from container); avoids huge files.
_MAX_CMD_DEBUG_CHARS = 512_000


class CommandDebugLogger:
    """Append-only per-solution command/output logs for debugging (thread-safe per env index)."""

    def __init__(self, base_dir: Path, num_envs: int, task_path: str) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._locks = [threading.Lock() for _ in range(num_envs)]
        readme = self.base_dir / "README.txt"
        if not readme.exists():
            readme.write_text(
                "Per-solution bash command logs from generate_solutions / run_n_solutions.\n"
                f"task.json: {task_path}\n"
                "Files: env_0000.log, env_0001.log, ... (one parallel solution attempt each).\n"
                "Each block: timestamp, turn, success, command, raw PTY output.\n",
                encoding="utf-8",
            )

    def log(
        self,
        env_idx: int,
        turn: int,
        command: str,
        success: bool,
        output: str,
        *,
        note: str = "",
    ) -> None:
        if env_idx < 0 or env_idx >= len(self._locks):
            return
        path = self.base_dir / f"env_{env_idx:04d}.log"
        body = output or ""
        if len(body) > _MAX_CMD_DEBUG_CHARS:
            tail = len(body) - _MAX_CMD_DEBUG_CHARS
            body = body[:_MAX_CMD_DEBUG_CHARS] + f"\n... [{tail} characters truncated for log]\n"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        extra = f"  note={note}" if note else ""
        block = (
            f"\n{'=' * 80}\n"
            f"time={ts}  solution={env_idx}  turn={turn}  success={success}{extra}\n"
            f"$ {command}\n"
            f"{'-' * 80}\n"
            f"{body}\n"
        )
        with self._locks[env_idx]:
            with open(path, "a", encoding="utf-8", errors="replace") as f:
                f.write(block)


def _truncate(text: str, limit: int = MAX_OUTPUT_LENGTH) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    n_elided = len(text) - limit
    return f"{text[:half]}\n\n... [{n_elided} characters elided] ...\n\n{text[-half:]}"


def _extract_tool_call(response_msg: dict) -> Dict[str, Optional[str]]:
    """Parse a tool-calling response message.

    Returns dict with:
      type: "command" | "done" | "no_tool_call"
      command: the bash command string (if type=="command")
      tool_call_id: the id needed for the tool response message
    """
    tool_calls = response_msg.get("tool_calls")
    if not tool_calls:
        return {"type": "no_tool_call", "command": None, "tool_call_id": None}

    tc = tool_calls[0]
    func = tc.get("function", {})
    func_name = func.get("name", "")

    if func_name != "bash":
        return {"type": "no_tool_call", "command": None, "tool_call_id": tc.get("id")}

    args_raw = func.get("arguments", "{}")
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            return {"type": "no_tool_call", "command": None, "tool_call_id": None}
    else:
        args = args_raw

    command = args.get("command", "").strip()
    tool_call_id = tc.get("id")

    if SUBMIT_MARKER in command:
        return {"type": "done", "command": command, "tool_call_id": tool_call_id}

    return {"type": "command", "command": command, "tool_call_id": tool_call_id}


def run_n_solutions(  # 定义函数：生成 n 个交互式解决方案
    num_solutions: int,  # 要生成的解决方案数量
    container_sif_path: str,  # 容器 SIF 文件路径
    initial_test_path: str,  # 初始状态测试脚本路径
    final_test_path: str,  # 最终状态测试脚本路径
    def_path: str,  # 容器定义文件路径
    task_path: str,  # 任务 JSON 文件路径
    max_actions: int = 16,  # 最大动作数（交互轮数），默认 16
    model: str = DEFAULT_MODEL,  # 使用的模型，默认为 DEFAULT_MODEL
    temperature: float = 0.7,  # 采样温度，默认 0.7
    max_tokens: int = 65536,  # 最大生成 token 数，默认 65536
    save_dir: Optional[str] = None,  # 保存目录，可选
    verbose: bool = True,  # 是否输出详细日志，默认 True
    num_pool_workers: int = 128,  # 线程池工作线程数，默认 128
    run_initial_tests: bool = True,  # 是否运行初始测试，默认 True
    command_timeout: float = 120.0,  # 命令超时时间（秒），默认 120
    shell_init_timeout: float = 120.0,  # shell 初始化超时时间，默认 120
    shell_init_attempts: int = 3,  # shell 初始化尝试次数，默认 3
    log_commands: bool = False,  # 是否记录命令日志，默认 False
    command_log_dir: Optional[str] = None,  # 命令日志目录，可选
    base_sifs_dir: Optional[str] = None,  # 基础 SIF 目录，可选
    max_timeouts_per_solution: int = 2,  # 每个解决方案允许的最大超时次数，默认 2
) -> Dict[str, Any]:  # 返回类型为字典
    """Produce n interactive solutions for the given task using tool-calling format."""  # 文档字符串：使用工具调用格式为给定任务生成 n 个交互式解决方案

    task_data = json.loads(Path(task_path).read_text(encoding="utf-8"))  # 读取任务 JSON 文件并解析为字典
    task_description: str = task_data.get("description", "").strip()  # 获取任务描述并去除首尾空白
    print(f"running {num_solutions} solutions for task")  # 打印正在为任务运行多少个解决方案
    results: List[Dict[str, Any]] = []  # 初始化结果列表
    num_success = 0  # 成功计数器初始化为 0

    # Per-solution token usage accumulators  # 每个解决方案的 token 使用量累加器
    usage_accum: List[Dict[str, int]] = [  # 初始化 token 使用量列表
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,  # 提示 token、补全 token、总 token
         "reasoning_tokens": 0}  # 推理 token
        for _ in range(num_solutions)  # 为每个解决方案创建一个累加器
    ]

    out_dir: Optional[Path] = None  # 输出目录初始化为 None
    if save_dir:  # 如果提供了保存目录
        out_dir = Path(save_dir)  # 转换为 Path 对象
        out_dir.mkdir(parents=True, exist_ok=True)  # 创建目录，如果已存在则忽略

    messages: List[List[Dict[str, Any]]] = [  # 初始化消息列表
        [  # 每个解决方案的消息列表
            {"role": "system", "content": SYSTEM_PROMPT},  # 系统消息
            {"role": "user", "content": task_description},  # 用户消息（任务描述）
        ]
        for _ in range(num_solutions)  # 为每个解决方案创建初始消息
    ]

    envs: List[ContainerEnvironment] = []  # 容器环境列表
    cmd_logger: Optional[CommandDebugLogger] = None  # 命令调试日志记录器，初始为 None
    if log_commands:  # 如果开启命令日志
        if command_log_dir:  # 如果指定了命令日志目录
            log_root = Path(command_log_dir).expanduser().resolve()  # 展开用户目录并解析为绝对路径
        elif save_dir:  # 否则如果提供了保存目录
            log_root = (Path(save_dir).expanduser().resolve() / "debug_commands")  # 在保存目录下创建 debug_commands 子目录
        else:  # 否则
            log_root = None  # 日志根目录为 None
        if log_root is not None:  # 如果日志根目录存在
            cmd_logger = CommandDebugLogger(log_root, num_solutions, str(Path(task_path).resolve()))  # 创建命令调试日志记录器
        elif verbose:  # 否则如果详细模式
            print("⚠️  log_commands=True but no command_log_dir and no save_dir; command debug logs disabled.")  # 打印警告：未指定日志目录，命令调试日志禁用

    try:  # 开始 try 块
        start_time = time.time()  # 记录开始时间

        def _init_env(i: int) -> ContainerEnvironment:  # 定义初始化环境的内部函数
            env = ContainerEnvironment(  # 创建容器环境
                container_sif_path=container_sif_path,  # 容器 SIF 路径
                initial_test_path=initial_test_path,  # 初始测试路径
                final_test_path=final_test_path,  # 最终测试路径
                def_path=def_path,  # 定义文件路径
                max_actions=max_actions,  # 最大动作数
                verbose=verbose,  # 详细模式
                read_timeout=command_timeout,  # 读取超时
                shell_init_timeout=shell_init_timeout,  # shell 初始化超时
                shell_init_attempts=shell_init_attempts,  # shell 初始化尝试次数
                base_sifs_dir=base_sifs_dir,  # 基础 SIF 目录
            )
            ok = env.initialize(run_initial_tests=False)  # 初始化环境，不运行初始测试
            if not ok:  # 如果初始化失败
                raise RuntimeError(f"Failed to initialize environment #{i}")  # 抛出运行时错误
            return env  # 返回环境对象

        with ThreadPoolExecutor(max_workers=num_pool_workers) as executor:  # 创建线程池执行器
            envs = list(executor.map(_init_env, range(num_solutions)))  # 并行初始化所有环境
        end_time = time.time()  # 记录结束时间
        print(f"environments initialized in {end_time - start_time:.1f} seconds")  # 打印环境初始化耗时

        if run_initial_tests:  # 如果需要运行初始测试
            if not envs[0].run_initial_tests():  # 在第一个环境上运行初始测试
                raise AssertionError("Initial state tests failed for env")  # 如果失败则抛出断言错误

        is_done: List[bool] = [False] * num_solutions  # 每个解决方案是否完成的标志列表
        not_done_idx: List[int] = list(range(num_solutions))  # 未完成的解决方案索引列表
        timeout_counts: List[int] = [0] * num_solutions  # 每个解决方案的超时计数
        num_steps = 0  # 步数计数器

        while not all(is_done):  # 当并非所有解决方案都完成时循环
            if not not_done_idx:  # 如果没有未完成的解决方案
                break  # 跳出循环

            prompt_messages = [messages[i] for i in not_done_idx]  # 获取未完成解决方案的消息
            print(f"generating solutions...for task {task_path} turn {num_steps}")  # 打印正在生成解决方案
            start_time = time.time()  # 记录开始时间
            responses_raw = chat_completion_batch_with_tools(  # 批量调用聊天补全接口
                prompt_messages,  # 提示消息
                tools=TOOL_SCHEMAS,  # 工具模式
                model=model,  # 模型
                temperature=temperature,  # 温度
                max_tokens=max_tokens,  # 最大 token 数
                max_concurrency=len(prompt_messages),  # 最大并发数
            )
            end_time = time.time()  # 记录结束时间
            print(f"solutions generated in {end_time - start_time:.1f} seconds")  # 打印生成耗时

            response_msgs: List[dict] = []  # 响应消息列表
            for local_i, r in enumerate(responses_raw):  # 遍历原始响应
                if r is None:  # 如果响应为 None
                    response_msgs.append({})  # 添加空字典
                else:  # 否则
                    response_msgs.append(r.choices[0].message.model_dump())  # 提取消息并转为字典
                    sol_idx = not_done_idx[local_i]  # 获取对应的解决方案索引
                    if hasattr(r, "usage") and r.usage is not None:  # 如果有 usage 信息
                        u = r.usage  # 获取 usage
                        usage_accum[sol_idx]["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0  # 累加提示 token
                        usage_accum[sol_idx]["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0  # 累加补全 token
                        usage_accum[sol_idx]["total_tokens"] += getattr(u, "total_tokens", 0) or 0  # 累加总 token
                        usage_accum[sol_idx]["reasoning_tokens"] += getattr(u, "reasoning_tokens", 0) or 0  # 累加推理 token

            actions = [_extract_tool_call(msg) for msg in response_msgs]  # 从响应消息中提取工具调用

            to_mark_done: List[int] = []  # 要标记为完成的索引列表
            to_exec: List[tuple[int, str, str]] = []  # 要执行的命令列表（索引，命令，工具调用 ID）

            for i, n in enumerate(not_done_idx):  # 遍历未完成的解决方案
                msg = response_msgs[i]  # 获取响应消息
                act = actions[i]  # 获取动作

                if not msg:  # 如果消息为空
                    messages[n].append({  # 添加助手消息
                        "role": "assistant",  # 角色为助手
                        "content": "I encountered an error. Let me try again.",  # 内容为错误信息
                    })
                    continue  # 继续下一个

                messages[n].append(msg)  # 将响应消息添加到消息历史

                if act["type"] == "done":  # 如果动作类型为 done
                    is_done[n] = True  # 标记为完成
                    to_mark_done.append(n)  # 添加到完成列表
                    if act["tool_call_id"] and act["command"]:  # 如果有工具调用 ID 和命令
                        success, output = envs[n].exec(act["command"])  # 执行命令
                        if cmd_logger:  # 如果命令日志记录器存在
                            cmd_logger.log(  # 记录日志
                                n, num_steps, act["command"], success, output or "", note="submit"  # 参数
                            )
                        messages[n].append({  # 添加工具响应消息
                            "role": "tool",  # 角色为工具
                            "tool_call_id": act["tool_call_id"],  # 工具调用 ID
                            "content": _truncate(output) if output else "(no output)",  # 内容为截断的输出或提示
                        })

                elif act["type"] == "command":  # 如果动作类型为命令
                    command = act["command"] or ""  # 获取命令
                    tool_call_id = act["tool_call_id"] or ""  # 获取工具调用 ID
                    to_exec.append((n, command, tool_call_id))  # 添加到待执行列表

                else:  # 其他类型
                    pass  # 不做处理

            start_time = time.time()  # 记录开始时间
            if to_exec:  # 如果有待执行的命令
                def _exec_one(item: tuple[int, str, str]) -> tuple[int, bool, str, str]:  # 定义执行单个命令的函数
                    idx, cmd, tc_id = item  # 解包
                    success, output = envs[idx].exec(cmd)  # 执行命令
                    if cmd_logger:  # 如果命令日志记录器存在
                        cmd_logger.log(idx, num_steps, cmd, success, output or "")  # 记录日志
                    return idx, success, output, tc_id  # 返回结果

                with ThreadPoolExecutor(max_workers=num_pool_workers) as pool:  # 创建线程池
                    exec_results: list[tuple[int, bool, str, str]] = list(pool.map(_exec_one, to_exec))  # 并行执行命令

                for idx, success, output, tc_id in exec_results:  # 遍历执行结果
                    truncated = _truncate(output) if output else "(no output)"  # 截断输出

                    if success:  # 如果成功
                        result_back = f"{truncated}\n\n(exit_code=0)"  # 结果加上退出码 0
                    else:  # 否则
                        result_back = f"{truncated}\n\n(exit_code=1)"  # 结果加上退出码 1

                    if "Command timed out" in output:  # 如果输出包含超时信息
                        timeout_counts[idx] += 1  # 超时计数加一
                        if timeout_counts[idx] >= max_timeouts_per_solution:  # 如果超过最大超时次数
                            is_done[idx] = True  # 标记为完成
                            if idx not in to_mark_done:  # 如果不在完成列表
                                to_mark_done.append(idx)  # 添加到完成列表
                            if verbose:  # 如果详细模式
                                print(f"⏹️  Solution {idx} aborted after {timeout_counts[idx]} timeouts")  # 打印中止信息

                    if SUBMIT_MARKER in output:  # 如果输出包含提交标记
                        is_done[idx] = True  # 标记为完成
                        if idx not in to_mark_done:  # 如果不在完成列表
                            to_mark_done.append(idx)  # 添加到完成列表

                    messages[idx].append({  # 添加工具响应消息
                        "role": "tool",  # 角色为工具
                        "tool_call_id": tc_id,  # 工具调用 ID
                        "content": result_back,  # 内容为结果
                    })

            end_time = time.time()  # 记录结束时间
            print(f"commands executed in {end_time - start_time:.1f} seconds")  # 打印命令执行耗时

            if to_mark_done:  # 如果有要标记为完成的
                done_set = set(to_mark_done)  # 转换为集合
                not_done_idx = [idx for idx in not_done_idx if idx not in done_set]  # 更新未完成列表

            num_steps += 1  # 步数加一
            if num_steps >= max_actions:  # 如果达到最大动作数
                is_done = [True] * num_solutions  # 全部标记为完成
                not_done_idx = []  # 清空未完成列表
                break  # 跳出循环

        start_time = time.time()  # 记录开始时间

        def _run_final(i: int) -> tuple[bool, str]:  # 定义运行最终测试的函数
            return envs[i].run_final_tests()  # 返回最终测试结果

        with ThreadPoolExecutor(max_workers=num_pool_workers) as pool:  # 创建线程池
            finals: list[tuple[bool, str]] = list(pool.map(_run_final, range(num_solutions)))  # 并行运行最终测试

        for i in range(num_solutions):  # 遍历所有解决方案
            success, output = finals[i]  # 获取最终测试结果
            if success:  # 如果成功
                num_success += 1  # 成功计数加一
            results.append({  # 添加结果
                "success": success,  # 是否成功
                "messages": messages[i],  # 消息历史
                "output": output,  # 输出
                "reward": 1 if success else 0,  # 奖励
                "usage": usage_accum[i],  # token 使用量
            })
        end_time = time.time()  # 记录结束时间
        print(f"final tests executed in {end_time - start_time:.1f} seconds")  # 打印最终测试耗时

    finally:  # 无论是否发生异常
        for env in envs:  # 遍历所有环境
            try:  # 尝试
                env.cleanup()  # 清理环境
            except Exception:  # 捕获异常
                pass  # 忽略

    n = num_solutions  # 总解决方案数
    c = num_success  # 成功数
    pass_at_k: Dict[int, float] = {}  # 初始化 pass@k 字典
    for k in range(1, n + 1):  # 遍历 k 从 1 到 n
        if c == 0:  # 如果成功数为 0
            p = 0.0  # 概率为 0
        else:  # 否则
            p = 1.0 - (comb(n - c, k) / comb(n, k))  # 计算 pass@k
        pass_at_k[k] = float(p)  # 存储结果

    total_usage = {  # 总 token 使用量
        "prompt_tokens": sum(u["prompt_tokens"] for u in usage_accum),  # 提示 token 总和
        "completion_tokens": sum(u["completion_tokens"] for u in usage_accum),  # 补全 token 总和
        "total_tokens": sum(u["total_tokens"] for u in usage_accum),  # 总 token 总和
        "reasoning_tokens": sum(u["reasoning_tokens"] for u in usage_accum),  # 推理 token 总和
    }

    summary: Dict[str, Any] = {  # 汇总字典
        "num_runs": num_solutions,  # 运行次数
        "num_success": num_success,  # 成功次数
        "pass_at_k": pass_at_k,  # pass@k 结果
        "usage": total_usage,  # token 使用量
        "results": results,  # 详细结果
    }

    return summary  # 返回汇总


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--task-dir", type=str, default="tasks")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)

    args = ap.parse_args()
    n = args.n
    task_dir = args.task_dir
    task_path = os.path.join(task_dir, "task.json")
    container_sif_path = os.path.join(task_dir, "container.sif")
    initial_test_path = os.path.join(task_dir, "test_initial_state.py")
    final_test_path = os.path.join(task_dir, "test_final_state.py")
    def_path_str = os.path.join(task_dir, "container.def")

    max_actions = 16

    summary = run_n_solutions(
        n,
        container_sif_path=container_sif_path,
        initial_test_path=initial_test_path,
        final_test_path=final_test_path,
        def_path=def_path_str,
        task_path=task_path,
        max_actions=max_actions,
        model=args.model,
        temperature=0.7,
        save_dir=task_dir,
        verbose=True,
        run_initial_tests=True,
    )

    print(json.dumps({
        "num_runs": summary["num_runs"],
        "num_success": summary["num_success"],
        "pass_at_k": summary["pass_at_k"],
    }, indent=4))
