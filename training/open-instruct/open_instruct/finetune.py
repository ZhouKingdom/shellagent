# !/usr/bin/env python
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# isort: off
import contextlib
import os
import warnings

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
with contextlib.suppress(Exception):
    import deepspeed

# isort: on
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

import datasets
import torch
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.accelerator import GradientAccumulationPlugin
from accelerate.logging import get_logger
from accelerate.utils import DeepSpeedSequenceParallelConfig, InitProcessGroupKwargs, ParallelismConfig, set_seed
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.pretty import pprint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq, get_scheduler
from transformers.training_args import _convert_str_dict

from open_instruct import logger_utils, model_utils, utils
from open_instruct.dataset_transformation import (
    INPUT_IDS_KEY,
    TOKENIZED_SFT_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.grpo_utils import build_fla_cp_context_for_sample
from open_instruct.model_utils import push_folder_to_hub, save_with_accelerate
from open_instruct.padding_free_collator import TensorDataCollatorWithFlattening
from open_instruct.qwen3_5_packing_patch import patch_qwen3_5_packing
from open_instruct.utils import (
    ArgumentParserPlus,
    clean_last_n_checkpoints,
    get_last_checkpoint_path,
    get_optimizer_grouped_parameters,
    get_wandb_tags,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_update_beaker_description,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
)

logger = get_logger(__name__)


_MAX_SEQ_LENGTH_TRANSFORM_FNS = {
    "sft_tulu_tokenize_and_truncate_v1",
    "last_turn_tulu_tokenize_and_truncate_v1",
}
_MAX_TOKEN_LENGTH_FILTER_FNS = {"sft_length_and_label_filter_v1"}


def build_transform_fn_args(dataset_transform_fn: list[str], max_seq_length: int | None) -> list[dict[str, int | None]]:
    transform_fn_args = []
    for fn_name in dataset_transform_fn:
        if fn_name in _MAX_SEQ_LENGTH_TRANSFORM_FNS:
            transform_fn_args.append({"max_seq_length": max_seq_length})
        elif fn_name in _MAX_TOKEN_LENGTH_FILTER_FNS:
            transform_fn_args.append({"max_token_length": max_seq_length})
        else:
            transform_fn_args.append({})
    return transform_fn_args


@dataclass
class FlatArguments:
    """
    Full arguments class for all fine-tuning jobs.
    """

    # Sometimes users will pass in a `str` repr of a dict in the CLI
    # We need to track what fields those can be. Each time a new arg
    # has a dict type, it must be added to this list.
    # Important: These should be typed with Optional[Union[dict,str,...]]
    # Note: the suggested ellipses typing above causes errors on python 3.10, so they are omitted.
    _VALID_DICT_FIELDS = ["additional_model_arguments"]

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    do_not_randomize_output_dir: bool = False
    """By default the output directory will be randomized"""
    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: str | None = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    model_revision: str | None = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    additional_model_arguments: dict | str | None = field(
        default_factory=dict, metadata={"help": "A dictionary of additional model args used to construct the model."}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, "
                "then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    dataset_name: str | None = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_mixer: dict | None = field(
        default=None, metadata={"help": "A dictionary of datasets (local or HF) to sample from."}
    )
    dataset_mixer_list: list[str] = field(default_factory=lambda: ["allenai/tulu-3-sft-personas-algebra", "1.0"])
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    dataset_mixer_list_config_names: list[str] = field(default_factory=list)
    """The Hugging Face dataset config names to use for training datasets"""
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )
    """The list of transform functions to apply to the dataset."""
    dataset_target_columns: list[str] = field(default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS)
    """The columns to use for the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_hash: str | None = None
    """The hash of the dataset configuration."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    dataset_mix_dir: str | None = field(
        default=None, metadata={"help": "The directory to save the mixed dataset to disk."}
    )
    dataset_config_name: str | None = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_seq_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    clip_grad_norm: float = field(
        default=-1,
        metadata={"help": "Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead)."},
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW optimizer."})
    logging_steps: int | None = field(
        default=None, metadata={"help": "Log the training loss and learning rate every logging_steps steps."}
    )
    lora_rank: int = field(default=64, metadata={"help": "The rank of lora."})
    lora_alpha: float = field(default=16, metadata={"help": "The alpha parameter of lora."})
    lora_dropout: float = field(default=0.1, metadata={"help": "The dropout rate of lora modules."})
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use for learning rate adjustment.",
            "choices": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        },
    )
    num_train_epochs: int = field(default=2, metadata={"help": "Total number of training epochs to perform."})
    output_dir: str = field(
        default="output/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "If True, will use LORA (low-rank parameter-efficient training) to train the model."},
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Use qLoRA training - initializes model in quantized form. Not compatible with deepspeed."},
    )
    use_8bit_optimizer: bool = field(
        default=False, metadata={"help": "Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed."}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    final_lr_ratio: float | None = field(
        default=None,
        metadata={
            "help": "Set the final lr value at the end of training to be final_lr_ratio * learning_rate."
            " Only for linear schedulers, currently."
        },
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout for the training process in seconds."
            "Useful if tokenization process is long. Default is 1800 seconds (30 minutes)."
        },
    )
    resume_from_checkpoint: str | None = field(
        default=None, metadata={"help": "If the training should continue from a checkpoint folder."}
    )
    report_to: str | list[str] = field(
        default="all",
        metadata={
            "help": "The integration(s) to report results and logs to. "
            "Can be a single string or a list of strings. "
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Specify multiple by listing them: e.g., ['tensorboard', 'wandb']"
        },
    )
    save_to_hub: str | None = field(
        default=None, metadata={"help": "Save the model to the Hub under this name. E.g allenai/your-model"}
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Turn on gradient checkpointing. Saves memory but slows training."}
    )
    use_liger_kernel: bool = field(default=False, metadata={"help": "Whether to use LigerKernel for training."})
    max_train_steps: int | None = field(
        default=None,
        metadata={"help": "If set, overrides the number of training steps. Otherwise, num_train_epochs is used."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for initialization and dataset shuffling."})
    checkpointing_steps: str | None = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
        },
    )
    keep_last_n_checkpoints: int = field(
        default=3, metadata={"help": "How many checkpoints to keep in the output directory. -1 for all."}
    )
    fused_optimizer: bool = field(default=True, metadata={"help": "Whether to use fused AdamW or not."})
    load_balancing_loss: bool = field(
        default=False, metadata={"help": "Whether to include a load balancing loss (for OLMoE) or not."}
    )
    load_balancing_weight: float = field(
        default=0.5, metadata={"help": "Weight for load balancing loss if applicable."}
    )
    clean_checkpoints_at_end: bool = field(
        default=True, metadata={"help": "Whether to clean up all previous checkpoints at the end of the run."}
    )

    # Experiment tracking
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: str | None = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: str | None = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: str | None = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: str | None = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: str | None = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    try_launch_beaker_eval_jobs: bool = True
    """Whether to launch beaker evaluation jobs after training"""
    hf_metadata_dataset: str | None = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""

    # Ai2 specific settings
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: str | None = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: list[str] | None = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""

    sync_each_batch: bool = False
    """Optionaly sync grads every batch when using grad accumulation. Can significantly reduce memory costs."""
    packing: bool = field(
        default=False,
        metadata={"help": "Whether to use packing/padding-free collation via TensorDataCollatorWithFlattening"},
    )
    verbose: bool = field(
        default=False, metadata={"help": "Optionally print additional statistics at each reporting period"}
    )
    sequence_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Degree of Ulysses sequence parallelism. 1 means disabled. Requires DeepSpeed ZeRO-3 and flash attention."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixer is None and self.dataset_mixer_list is None:
            raise ValueError("Need either a dataset name, dataset mixer, or dataset mixer list.")
        if (
            self.dataset_name is not None and (self.dataset_mixer is not None or self.dataset_mixer_list is not None)
        ) or (self.dataset_mixer is not None and self.dataset_mixer_list is not None):
            raise ValueError("Cannot provide two dataset selection mechanisms.")
        if self.try_launch_beaker_eval_jobs and not self.push_to_hub:
            raise ValueError("Cannot launch Beaker evaluation jobs without pushing to the Hub.")
        if self.final_lr_ratio is not None:
            if self.lr_scheduler_type != "linear":
                raise NotImplementedError("final_lr_ratio only currently implemented for linear schedulers")
            if not (1.0 >= self.final_lr_ratio >= 0.0):
                raise ValueError(f"final_lr_ratio must be between 0 and 1, not {self.final_lr_ratio=}")

        # Parse in args that could be `dict` sent in from the CLI as a string
        for dict_feld in self._VALID_DICT_FIELDS:
            passed_value = getattr(self, dict_feld)
            # We only want to do this if the str starts with a bracket to indicate a `dict`
            # else its likely a filename if supported
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                # Convert str values to types if applicable
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, dict_feld, loaded_dict)


def _create_scheduler(args: FlatArguments, optimizer, num_training_steps: int):
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    if args.final_lr_ratio is not None and args.lr_scheduler_type == "linear":
        num_training_steps = (num_training_steps - args.final_lr_ratio * num_warmup_steps) / (1 - args.final_lr_ratio)
    return get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )


def main(args: FlatArguments, tc: TokenizerConfig):  # 主函数：接收训练参数 args 和分词器配置 tc
    # ------------------------------------------------------------  # 分隔线
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.  # 初始化加速器，由其处理设备放置
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers  # 若启用跟踪，也在此初始化，默认会拾取所有支持的追踪器
    # in the environment  # 环境中的追踪器
    accelerator_log_kwargs = {}  # 初始化加速器日志参数为空字典
    if args.with_tracking:  # 如果启用了跟踪
        accelerator_log_kwargs["log_with"] = args.report_to  # 设置日志记录目标为 args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir  # 设置项目目录为输出目录
    # if you get timeouts (e.g. due to long tokenization) increase this.  # 如果遇到超时（如长分词），可增大此值
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))  # 创建进程组初始化超时参数
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)  # 数据加载器配置，使用可设置种子的采样器

    parallelism_config = None  # 并行配置初始为 None
    if args.sequence_parallel_size > 1 and not args.cache_dataset_only:  # 若序列并行大小 >1 且不只缓存数据集
        world_size = int(os.environ.get("WORLD_SIZE", 1))  # 获取全局进程数，默认为 1
        if world_size % args.sequence_parallel_size != 0:  # 若全局进程数不能被序列并行大小整除
            raise ValueError(  # 抛出异常
                f"WORLD_SIZE ({world_size}) must be divisible by sequence_parallel_size ({args.sequence_parallel_size})"  # 错误信息
            )
        dp_shard_size = world_size // args.sequence_parallel_size  # 计算数据并行分片大小
        parallelism_config = ParallelismConfig(  # 创建并行配置
            sp_backend="deepspeed",  # 序列并行后端为 deepspeed
            sp_size=args.sequence_parallel_size,  # 序列并行大小
            dp_shard_size=dp_shard_size,  # 数据并行分片大小
            sp_handler=DeepSpeedSequenceParallelConfig(  # 序列并行处理器配置
                sp_seq_length_is_variable=True, sp_attn_implementation=model_utils.detect_hf_attn_implementation()  # 序列长度可变，注意力实现自动检测
            ),
        )

    accelerator = Accelerator(  # 创建加速器实例
        dataloader_config=dataloader_config,  # 传入数据加载器配置
        parallelism_config=parallelism_config,  # 传入并行配置
        **accelerator_log_kwargs,  # 展开日志参数
        kwargs_handlers=[timeout_kwargs],  # 传入超时处理器
        gradient_accumulation_plugin=GradientAccumulationPlugin(  # 梯度累积插件
            num_steps=args.gradient_accumulation_steps, sync_each_batch=args.sync_each_batch  # 累积步数，是否每批同步
        ),
    )

    # ------------------------------------------------------------  # 分隔线
    # Setup tokenizer  # 设置分词器
    tc.tokenizer_revision = args.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision  # 若分词器版本未设置，则使用模型版本
    tc.tokenizer_name_or_path = (  # 设置分词器名称或路径
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path  # 若未设置则使用模型名称或路径
    )
    if tc.tokenizer_revision != args.model_revision and tc.tokenizer_name_or_path != args.model_name_or_path:  # 若分词器与模型的版本或名称不一致
        # Warn user if tokenizer and model use different revisions; this is an unusual  # 警告用户：分词器与模型使用不同版本，这并不常见
        # use case.  # 使用场景
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different  # 警告信息：请求的分词器版本不同
                   from the model revision `{args.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`  # 与模型版本或分词器名称不同
                   is different from the model name `{args.model_name_or_path=}`."""  # 与模型名称不同
        logger.warning(warning)  # 记录警告日志
    tokenizer = tc.tokenizer  # 获取分词器对象

    # ------------------------------------------------------------  # 分隔线
    # Set up runtime variables  # 设置运行时变量

    if not args.do_not_randomize_output_dir:  # 如果不禁止随机化输出目录
        args.output_dir = os.path.join(args.output_dir, args.exp_name)  # 将实验名拼接到输出目录后
    logger.info("using the output directory: %s", args.output_dir)  # 记录使用的输出目录
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)  # 将数据集本地缓存目录转为绝对路径
    if is_beaker_job():  # 如果是 Beaker 任务
        args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"  # 使用指定缓存目录
    if args.push_to_hub and accelerator.is_main_process:  # 若推送到 Hub 且是主进程
        if args.hf_repo_id is None:  # auto-generate one  # 若未指定 HF 仓库 ID，则自动生成
            args.hf_repo_id = "open_instruct_dev"  # 默认仓库 ID
        if args.hf_entity is None:  # first try to use AI2 entity  # 若未指定实体，先尝试 AI2 实体
            args.hf_entity = maybe_use_ai2_hf_entity()  # 尝试获取 AI2 HF 实体
        if args.hf_entity is None:  # then try to use the user's entity  # 若仍无，则尝试用户实体
            args.hf_entity = HfApi().whoami()["name"]  # 通过 HF API 获取当前用户名
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"  # 拼接实体和仓库 ID
        if args.hf_repo_revision is None:  # 若未指定仓库修订版本
            args.hf_repo_revision = args.exp_name  # 使用实验名作为修订版本
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"  # 构造仓库 URL
        beaker_config = maybe_get_beaker_config()  # 获取 Beaker 配置

    # ------------------------------------------------------------  # 分隔线
    # Initialize the trackers we use, and also store our configuration.  # 初始化使用的追踪器，并存储配置
    # The trackers initializes automatically on the main process.  # 追踪器会在主进程自动初始化
    if args.with_tracking:  # 若启用跟踪
        experiment_config = vars(args)  # 获取 args 的字典形式作为实验配置
        # TensorBoard cannot log Enums, need the raw value  # TensorBoard 不能记录枚举，需要原始值
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]  # 将学习率调度器类型设为原始值

        # (Optional) Ai2 internal tracking  # （可选）Ai2 内部跟踪
        if args.wandb_entity is None:  # 若未指定 wandb 实体
            args.wandb_entity = maybe_use_ai2_wandb_entity()  # 尝试使用 AI2 wandb 实体
        if accelerator.is_main_process and is_beaker_job():  # 若是主进程且为 Beaker 任务
            beaker_config = maybe_get_beaker_config()  # 获取 Beaker 配置
            experiment_config.update(vars(beaker_config))  # 更新实验配置
        experiment_config.update(vars(tc))  # 更新分词器配置
        accelerator.init_trackers(  # 初始化追踪器
            args.wandb_project_name,  # wandb 项目名
            experiment_config,  # 实验配置
            init_kwargs={  # 初始化参数
                "wandb": {  # wandb 配置
                    "name": args.exp_name,  # 运行名称
                    "entity": args.wandb_entity,  # 实体
                    "tags": [args.exp_name] + get_wandb_tags(),  # 标签
                }
            },
        )
        wandb_tracker = accelerator.get_tracker("wandb")  # 获取 wandb 追踪器
        if accelerator.is_main_process:  # 若是主进程
            maybe_update_beaker_description(wandb_url=wandb_tracker.run.url)  # 更新 Beaker 描述中的 wandb URL
    else:  # 若未启用跟踪
        wandb_tracker = None  # for later eval launching  # 设为 None，供后续评估启动使用

    if accelerator.is_main_process:  # 若是主进程
        pprint([args, tc])  # 打印参数和分词器配置

    # Make one log on every process with the configuration for debugging.  # 在每个进程上记录一次日志，用于调试
    logger_utils.setup_logger()  # 设置日志记录器
    logger.info(accelerator.state, main_process_only=False)  # 记录加速器状态，非仅主进程
    if accelerator.is_local_main_process:  # 若是本地主进程
        datasets.utils.logging.set_verbosity_warning()  # 设置 datasets 日志级别为警告
        transformers.utils.logging.set_verbosity_info()  # 设置 transformers 日志级别为信息
    else:  # 否则
        datasets.utils.logging.set_verbosity_error()  # 设置 datasets 日志级别为错误
        transformers.utils.logging.set_verbosity_error()  # 设置 transformers 日志级别为错误

    # If passed along, set the training seed now.  # 如果传入了种子，现在设置训练种子
    if args.seed is not None:  # 若种子不为 None
        set_seed(args.seed)  # 设置随机种子

    if accelerator.is_main_process and args.output_dir is not None:  # 若是主进程且输出目录不为 None
        os.makedirs(args.output_dir, exist_ok=True)  # 创建输出目录

    accelerator.wait_for_everyone()  # 等待所有进程

    if args.dataset_mixer is not None:  # 若数据集混合器不为 None
        args.dataset_mixer_list = [item for pair in args.dataset_mixer.items() for item in pair]  # 将混合器字典展平为列表
    dataset_mixer_list_config_names = args.dataset_mixer_list_config_names  # 获取数据集混合器配置名称列表
    if not dataset_mixer_list_config_names and args.dataset_config_name is not None:  # 若列表为空且数据集配置名不为 None
        dataset_mixer_list_config_names = [args.dataset_config_name]  # 使用数据集配置名作为列表
    with accelerator.main_process_first():  # 在主进程优先执行上下文中
        transform_fn_args = build_transform_fn_args(args.dataset_transform_fn, args.max_seq_length)  # 构建转换函数参数
        train_dataset = get_cached_dataset_tulu(  # 获取缓存数据集
            dataset_mixer_list=args.dataset_mixer_list,  # 数据集混合列表
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,  # 数据集分割
            tc=tc,  # 分词器配置
            dataset_transform_fn=args.dataset_transform_fn,  # 数据集转换函数
            transform_fn_args=transform_fn_args,  # 转换函数参数
            target_columns=args.dataset_target_columns,  # 目标列
            dataset_cache_mode=args.dataset_cache_mode,  # 缓存模式
            dataset_config_hash=args.dataset_config_hash,  # 配置哈希
            hf_entity=args.hf_entity,  # HF 实体
            dataset_local_cache_dir=args.dataset_local_cache_dir,  # 本地缓存目录
            dataset_skip_cache=args.dataset_skip_cache,  # 是否跳过缓存
            dataset_mixer_list_config_names=dataset_mixer_list_config_names,  # 混合器配置名称列表
        )
        train_dataset = train_dataset.shuffle(seed=args.seed)  # 打乱数据集
        train_dataset.set_format(type="pt")  # 设置格式为 PyTorch 张量
    if accelerator.is_main_process:  # 若是主进程
        visualize_token(train_dataset[0][INPUT_IDS_KEY], tokenizer)  # 可视化第一个样本的 token

    if args.cache_dataset_only:  # 若仅缓存数据集
        return  # 直接返回

    # Pre-download model files on main process to avoid race conditions  # 在主进程预下载模型文件，避免竞态条件
    # when multiple ranks on a shared filesystem all try to access the  # 当共享文件系统上的多个 rank 同时尝试访问
    # HF hub cache concurrently.  # HF 中心缓存时
    model_path = args.config_name or args.model_name_or_path  # 模型路径：配置名或模型名
    if model_path and accelerator.is_main_process:  # 若模型路径存在且是主进程
        snapshot_download(model_path, revision=args.model_revision)  # 下载模型快照
    accelerator.wait_for_everyone()  # 等待所有进程

    # Load pretrained model and tokenizer  # 加载预训练模型和分词器
    if args.config_name:  # 若指定了配置名
        config = AutoConfig.from_pretrained(  # 从预训练加载配置
            args.config_name,  # 配置名
            revision=args.model_revision,  # 模型修订版本
            trust_remote_code=tc.trust_remote_code,  # 是否信任远程代码
            local_files_only=True,  # 仅使用本地文件
            **args.additional_model_arguments,  # 额外模型参数
        )
    elif args.model_name_or_path:  # 否则若指定了模型名或路径
        config = AutoConfig.from_pretrained(  # 从预训练加载配置
            args.model_name_or_path,  # 模型名或路径
            revision=args.model_revision,  # 模型修订版本
            trust_remote_code=tc.trust_remote_code,  # 是否信任远程代码
            local_files_only=True,  # 仅使用本地文件
            **args.additional_model_arguments,  # 额外模型参数
        )
    else:  # 否则
        raise ValueError(  # 抛出异常
            "You are instantiating a new config instance from scratch. This is not supported by this script."  # 不支持从头实例化新配置
        )

    if args.model_name_or_path:  # 若指定了模型名或路径
        if args.use_qlora:  # 若使用 QLoRA
            bnb_config = BitsAndBytesConfig(  # 创建 BitsAndBytes 量化配置
                load_in_4bit=True,  # 4 比特加载
                bnb_4bit_use_double_quant=True,  # 使用双量化
                bnb_4bit_quant_type="nf4",  # 量化类型 nf4
                bnb_4bit_compute_dtype=torch.bfloat16,  # 计算数据类型 bfloat16
            )
            device_index = accelerator.local_process_index  # 本地进程索引
            device_map = {"": device_index}  # force data-parallel training.  # 强制数据并行训练
            model = AutoModelForCausalLM.from_pretrained(  # 加载因果语言模型
                args.model_name_or_path,  # 模型名或路径
                revision=args.model_revision,  # 修订版本
                from_tf=bool(".ckpt" in args.model_name_or_path),  # 是否从 TensorFlow 检查点加载
                config=config,  # 配置
                trust_remote_code=tc.trust_remote_code,  # 是否信任远程代码
                quantization_config=bnb_config,  # 量化配置
                device_map=device_map,  # 设备映射
                dtype=torch.bfloat16,  # 数据类型
                attn_implementation=model_utils.detect_hf_attn_implementation(),  # 注意力实现
                local_files_only=True,  # 仅本地文件
            )
        elif args.use_liger_kernel:  # 否则若使用 Liger 内核
            from liger_kernel.transformers import AutoLigerKernelForCausalLM  # noqa: PLC0415  # 导入 Liger 内核模型
            logger.info("Attempting to apply liger-kernel. fused_linear_cross_entropy=True")  # 记录日志
            # Supported models: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py#L948  # 支持的模型链接
            model = AutoLigerKernelForCausalLM.from_pretrained(  # 加载 Liger 内核模型
                args.model_name_or_path,  # 模型名或路径
                revision=args.model_revision,  # 修订版本
                from_tf=bool(".ckpt" in args.model_name_or_path),  # 是否从 TF 检查点
                config=config,  # 配置
                trust_remote_code=tc.trust_remote_code,  # 信任远程代码
                low_cpu_mem_usage=args.low_cpu_mem_usage,  # 低 CPU 内存使用
                attn_implementation=model_utils.detect_hf_attn_implementation(),  # 注意力实现
                local_files_only=True,  # 仅本地文件
                # liger-kernel specific args  # Liger 内核特定参数
                fused_linear_cross_entropy=True,  # 融合线性交叉熵
            )
        else:  # 否则
            model = AutoModelForCausalLM.from_pretrained(  # 加载因果语言模型
                args.model_name_or_path,  # 模型名或路径
                revision=args.model_revision,  # 修订版本
                from_tf=bool(".ckpt" in args.model_name_or_path),  # 是否从 TF 检查点
                config=config,  # 配置
                trust_remote_code=tc.trust_remote_code,  # 信任远程代码
                low_cpu_mem_usage=args.low_cpu_mem_usage,  # 低 CPU 内存使用
                dtype=torch.bfloat16,  # 数据类型
                attn_implementation=model_utils.detect_hf_attn_implementation(),  # 注意力实现
                local_files_only=True,  # 仅本地文件
            )
    else:  # 否则
        logger.info("Training new model from scratch")  # 记录从头训练新模型
        model = AutoModelForCausalLM.from_config(config)  # 从配置创建模型

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch  # 仅在必要时调整嵌入大小以避免索引错误。如果从头创建模型
    # on a small vocab and want a smaller embedding size, remove this test.  # 在小词表上并希望更小的嵌入大小，可移除此测试
    # gather deepspeed to get "real" embedding size  # 使用 deepspeed 收集以获取真实嵌入大小
    embeddings = model.get_input_embeddings()  # 获取输入嵌入
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):  # 收集参数
        embedding_size = embeddings.weight.shape[0]  # 获取嵌入大小
    # resize does its own gather  # resize 会自行收集
    if len(tokenizer) > embedding_size:  # 若分词器长度大于嵌入大小
        # pad to multiple for tensor cores.  # 为张量核心填充到倍数
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)  # 调整 token 嵌入
    # update embedding size after resizing for sum loss  # 调整后更新嵌入大小以计算 sum loss
    embeddings = model.get_input_embeddings()  # 重新获取输入嵌入
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):  # 收集参数
        embedding_size = embeddings.weight.shape[0]  # 更新嵌入大小

    if args.use_lora:  # 若使用 LoRA
        if args.use_qlora:  # 若使用 QLoRA
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)  # 准备 k 比特训练模型
        elif args.gradient_checkpointing:  # 否则若启用梯度检查点
            # Enable gradient checkpointing for LoRA (non-QLoRA) too  # 也为 LoRA（非 QLoRA）启用梯度检查点
            model.gradient_checkpointing_enable()  # 启用梯度检查点

        logger.info("Initializing LORA model...")  # 记录初始化 LoRA 模型
        peft_config = LoraConfig(  # 创建 LoRA 配置
            task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言模型
            inference_mode=False,  # 非推理模式
            r=args.lora_rank,  # LoRA 秩
            lora_alpha=args.lora_alpha,  # LoRA alpha
            lora_dropout=args.lora_dropout,  # LoRA dropout
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],  # 目标模块
        )
        model = get_peft_model(model, peft_config)  # 获取 PEFT 模型
        model.print_trainable_parameters()  # 打印可训练参数
    elif args.gradient_checkpointing:  # 否则若启用梯度检查点
        model.gradient_checkpointing_enable()  # 启用梯度检查点

    _conv_kernel_size = getattr(model.config, "linear_conv_kernel_dim", None) or getattr(  # 获取卷积核大小
        getattr(model.config, "text_config", None), "linear_conv_kernel_dim", None  # 从 text_config 中获取
    )
    _is_hybrid = _conv_kernel_size is not None  # 是否为混合模型
    _is_hybrid_sp = _is_hybrid and args.sequence_parallel_size > 1  # 是否为混合序列并行
    if _is_hybrid:  # 若是混合模型
        patch_qwen3_5_packing()  # 打补丁 qwen3_5 packing
    _sp_group = accelerator.torch_device_mesh["sp"].get_group() if args.sequence_parallel_size > 1 else None  # 获取序列并行组

    # DataLoaders creation:  # 创建数据加载器
    if args.packing and args.sequence_parallel_size > 1:  # 若启用 packing 且序列并行 >1
        raise ValueError(  # 抛出异常
            "packing=True is not compatible with sequence_parallel_size > 1: the Ulysses SP "  # packing 与序列并行不兼容
            "adapter cannot split variable-length packing tensors (cu_seq_lens_q, max_length, "  # 适配器无法分割变长张量
            "etc.) across ranks. Use packing=False with SP."  # 请在 SP 下使用 packing=False
        )
    if args.packing:  # 若启用 packing
        collate_fn = TensorDataCollatorWithFlattening()  # 使用展平张量数据整理器
    else:  # 否则
        base_collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")  # 基础数据整理器
        if args.sequence_parallel_size > 1:  # 若序列并行 >1
            sp = args.sequence_parallel_size  # 序列并行大小

            def collate_fn(features):  # 定义整理函数
                batch = base_collate_fn(features)  # 使用基础整理器
                batch.pop("index", None)  # 移除 index
                # Pad seq dim to be divisible by SP size so the adapter can split evenly.  # 填充序列维度使其可被 SP 大小整除
                seq_len = next(iter(batch.values())).shape[1]  # 获取序列长度
                remainder = seq_len % sp  # 计算余数
                if remainder != 0:  # 若有余数
                    pad_len = sp - remainder  # 计算填充长度
                    for k in batch:  # 遍历批次
                        pad_value = -100 if k == "labels" else 0  # 标签填充 -100，其他填充 0
                        batch[k] = torch.nn.functional.pad(batch[k], (0, pad_len), value=pad_value)  # 填充
                if "attention_mask" not in batch:  # 若没有 attention_mask
                    raise ValueError("Expected attention_mask in batch when sequence_parallel_size > 1.")  # 抛出异常
                # Ulysses shards this tensor after collation, so create positions  # Ulysses 在整理后分片，因此创建位置
                # for the full pre-shard sequence.  # 为完整预分片序列
                seq_len = batch["input_ids"].shape[1]  # 获取输入 ID 的序列长度
                batch["position_ids"] = (  # 创建 position_ids
                    torch.arange(seq_len, dtype=torch.long)  # 生成序列
                    .unsqueeze(0)  # 增加维度
                    .expand(batch["input_ids"].shape[0], -1)  # 扩展
                    .contiguous()  # 连续化
                )
                return batch  # 返回批次
        else:  # 否则
            collate_fn = base_collate_fn  # 直接使用基础整理器

    accelerator.print("Creating dataloader")  # 打印创建数据加载器
    train_dataloader = DataLoader(  # 创建数据加载器
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size  # 数据集，打乱，整理函数，批大小
    )

    # Optimizer  # 优化器
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)  # 获取分组优化参数

    if args.use_qlora:  # 若使用 QLoRA
        from bitsandbytes.optim import AdamW  # noqa: PLC0415  # 导入 bitsandbytes AdamW
        optimizer = AdamW(  # 创建优化器
            optimizer_grouped_parameters,  # 参数
            lr=args.learning_rate,  # 学习率
            optim_bits=8 if args.use_8bit_optimizer else 32,  # 优化器比特数
            is_paged=True,  # 分页
        )
    else:  # 否则
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, fused=args.fused_optimizer)  # 创建 PyTorch AdamW

    # Scheduler and math around the number of training steps.  # 调度器和训练步数计算
    overrode_max_train_steps = False  # 是否覆盖最大训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)  # 每轮更新步数
    if args.max_train_steps is None:  # 若最大训练步数为 None
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch  # 根据轮数计算
        overrode_max_train_steps = True  # 标记覆盖

    # Create the learning rate scheduler.  # 创建学习率调度器
    # Note: the current accelerator.step() calls the .step() of the real scheduler  # 注意：当前 accelerator.step() 会调用真实调度器的 .step()
    # for the `num_processes` times. This is because they assume  # 共 num_processes 次。因为他们假设
    # the user initialize the scheduler with the entire training set.  # 用户用整个训练集初始化调度器
    # In the case of data parallel training, each process only  # 在数据并行训练中，每个进程只
    # sees a subset (1/num_processes) of the training set.  # 看到训练集的子集（1/num_processes）
    # So each time the process needs to update the lr multiple times so that the total  # 所以每次进程需要多次更新学习率，以使总
    # number of updates in the end matches the num_training_steps here.  # 更新次数最终与这里的 num_training_steps 匹配
    # Here we need to set the num_training_steps to either using the  # 这里我们需要将 num_training_steps 设置为使用
    # entire training set (when epochs is specified) or we need to multiply the  # 整个训练集（指定 epochs 时）或乘以
    # num_training_steps by num_processes so that the total number of  # num_training_steps 乘以 num_processes，使总
    # updates matches the num_training_steps.  # 更新次数匹配 num_training_steps
    num_training_steps_for_scheduler = (  # 调度器训练步数
        args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes  # 根据是否覆盖计算
    )
    lr_scheduler = _create_scheduler(args, optimizer, num_training_steps_for_scheduler)  # 创建调度器

    # Prepare everything with `accelerator`.  # 使用 accelerator 准备所有对象
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(  # 准备
        model, optimizer, train_dataloader, lr_scheduler  # 模型、优化器、数据加载器、调度器
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.  # 需要重新计算总训练步数，因为数据加载器大小可能改变
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)  # 重新计算每轮更新步数
    if overrode_max_train_steps:  # 若覆盖了最大训练步数
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch  # 重新计算
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)  # 重新计算轮数

    if args.sequence_parallel_size > 1:  # 若序列并行 >1
        # SP changes the dataloader length post-prepare. Recreate the scheduler using  # SP 改变 prepare 后的数据加载器长度。重新创建调度器使用
        # the post-prepare max_train_steps. Multiply by gradient_accumulation_steps because  # prepare 后的 max_train_steps。乘以梯度累积步数因为
        # the scheduler is called every micro-batch (not just on optimizer steps).  # 调度器每个微批次调用（不仅优化器步）
        lr_scheduler = _create_scheduler(args, optimizer, args.max_train_steps * args.gradient_accumulation_steps)  # 重新创建调度器

    # Figure out how many steps we should save the Accelerator states  # 确定保存 Accelerator 状态的步数
    checkpointing_steps = args.checkpointing_steps  # 检查点步数
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":  # 若不为 None 且不是 "epoch"
        checkpointing_steps = int(checkpointing_steps)  # 转为整数

    # Train!  # 训练！
    dp_world_size = accelerator.num_processes // args.sequence_parallel_size  # 数据并行世界大小
    total_batch_size = args.per_device_train_batch_size * dp_world_size * args.gradient_accumulation_steps  # 总批大小
    logger.info("***** Running training *****")  # 记录训练开始
    logger.info(f"  Num examples = {len(train_dataset)}")  # 样本数
    logger.info(f"  Num Epochs = {args.num_train_epochs}")  # 轮数
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")  # 每设备瞬时批大小
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")  # 总训练批大小
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")  # 梯度累积步数
    logger.info(f"  Total optimization steps = {args.max_train_steps}")  # 总优化步数
    # Only show the progress bar once on each machine.  # 每台机器只显示一次进度条
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)  # 进度条
    completed_steps = 0  # 已完成步数
    starting_epoch = 0  # 起始轮数

    # Potentially load in the weights and states from a previous save  # 可能从之前的保存加载权重和状态
    last_checkpoint_path = get_last_checkpoint_path(args)  # 获取最后检查点路径
    if last_checkpoint_path:  # 若存在
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")  # 打印从检查点恢复
        accelerator.load_state(last_checkpoint_path)  # 加载状态
        # Extract `epoch_{i}` or `step_{i}`  # 提取 epoch_{i} 或 step_{i}
        last_checkpoint_path = os.path.basename(last_checkpoint_path)  # 获取基名
        training_difference = os.path.splitext(last_checkpoint_path)[0]  # 去除扩展名

        if "epoch" in training_difference:  # 若包含 epoch
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1  # 起始轮数
            resume_batch_idx = 0  # 恢复批次索引
            completed_steps = starting_epoch * num_update_steps_per_epoch  # 已完成步数
        else:  # 否则
            # need to multiply `gradient_accumulation_steps` to reflect real steps  # 需要乘以梯度累积步数以反映真实步数
            resume_batch_idx = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps  # 恢复批次索引
            starting_epoch = resume_batch_idx // len(train_dataloader)  # 起始轮数
            completed_steps = resume_batch_idx // args.gradient_accumulation_steps  # 已完成步数
            resume_batch_idx -= starting_epoch * len(train_dataloader)  # 调整批次索引

    else:  # 否则
        resume_batch_idx = 0  # 恢复批次索引为 0

    resume_step = resume_batch_idx // args.gradient_accumulation_steps  # 恢复步数

    print(f"Starting {starting_epoch=}, {resume_batch_idx=}, {resume_step=}, {completed_steps=}.")  # 打印起始信息
    # update the progress_bar if load from checkpoint  # 若从检查点加载则更新进度条
    progress_bar.update(completed_steps)  # 更新进度条
    local_total_tokens = torch.tensor(0, dtype=torch.int64, device=accelerator.device)  # 本地总 token 数
    local_pred_tokens = torch.tensor(0, dtype=torch.int64, device=accelerator.device)  # 本地预测 token 数
    local_total_tokens_this_log_period = torch.tensor(0, dtype=torch.int64, device=accelerator.device)  # 本日志周期总 token 数
    local_pred_tokens_this_log_period = torch.tensor(0, dtype=torch.int64, device=accelerator.device)  # 本日志周期预测 token 数
    total_token_including_padding = torch.tensor(0, dtype=torch.int64, device=accelerator.device)  # 包含填充的总 token 数
    start_time = time.perf_counter()  # 开始时间
    skipped_batches = False  # 是否跳过批次
    for epoch in range(starting_epoch, args.num_train_epochs):  # 遍历轮数
        model.train()  # 模型训练模式
        # UlyssesSPDataLoaderAdapter wraps the real dataloader but doesn't proxy set_epoch  # UlyssesSPDataLoaderAdapter 包装真实数据加载器但不代理 set_epoch
        getattr(train_dataloader, "dl", train_dataloader).set_epoch(epoch)  # 设置轮数
        total_loss = 0  # 总损失
        total_aux_loss = 0  # 总辅助损失
        if last_checkpoint_path and resume_batch_idx and not skipped_batches:  # 若从检查点恢复且需跳过批次
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint.  # 从检查点恢复时跳过前 n 个批次
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_batch_idx)  # 跳过批次
            # Only perform this skip once  # 只执行一次跳过
            skipped_batches = True  # 标记已跳过
        else:  # 否则
            active_dataloader = train_dataloader  # 使用完整数据加载器
        for batch in active_dataloader:  # 遍历批次
            batch = {k: v.to(accelerator.device) if hasattr(v, "to") else v for k, v in batch.items()}  # 将批次数据移到设备
            if args.sequence_parallel_size > 1 and "shift_labels" not in batch:  # 若序列并行 >1 且无 shift_labels
                raise ValueError(  # 抛出异常
                    "`shift_labels` not found in batch with sequence parallelism enabled. "  # 未找到 shift_labels
                    "Check that UlyssesSPDataLoaderAdapter is wrapping the dataloader correctly."  # 检查适配器
                )
            if "shift_labels" in batch and "labels" not in batch:  # 若有 shift_labels 但无 labels
                batch["labels"] = batch["shift_labels"]  # 将 shift_labels 赋给 labels
            pred_tokens_in_batch = (batch["labels"] != -100).sum()  # 批次中预测 token 数
            if "attention_mask" in batch:  # 若有 attention_mask
                tokens_in_batch = batch["attention_mask"].sum()  # 批次 token 数
                total_token_including_padding += batch["attention_mask"].numel()  # 累加包含填充的总 token 数
            elif "position_ids" in batch:  # 否则若有 position_ids
                tokens_in_batch = batch["position_ids"].numel()  # 批次 token 数
                total_token_including_padding += tokens_in_batch  # 累加
            elif "cu_seq_lens_q" in batch:  # 否则若有 cu_seq_lens_q
                tokens_in_batch = batch["cu_seq_lens_q"][-1]  # 批次 token 数
                total_token_including_padding += tokens_in_batch  # 累加
            else:  # 否则
                raise ValueError(f"Expected attention_mask or position_ids or cu_seq_lens_q in batch, found {batch=}")  # 抛出异常
            local_total_tokens += tokens_in_batch  # 累加本地总 token 数
            local_total_tokens_this_log_period += tokens_in_batch  # 累加本日志周期总 token 数
            local_pred_tokens += pred_tokens_in_batch  # 累加本地预测 token 数
            local_pred_tokens_this_log_period += pred_tokens_in_batch  # 累加本日志周期预测 token 数

            fwd_extra: dict = {}  # 前向额外参数
            if _is_hybrid_sp:  # 若是混合序列并行
                if "position_ids" not in batch:  # 若无 position_ids
                    raise ValueError(  # 抛出异常
                        "Qwen3.5 hybrid sequence-parallel training requires pre-shard position_ids. "  # 需要预分片 position_ids
                        "Check that the SP collator added position_ids before Ulysses sharding."  # 检查 SP 整理器
                    )
                local_pos = batch["position_ids"]  # 本地位置
                if local_pos.shape[0] != 1:  # 若批次大小不为 1
                    raise ValueError("Qwen3.5 hybrid sequence-parallel training currently requires batch size 1.")  # 抛出异常
                local_pos_row = local_pos[0:1].contiguous()  # 取第一行并连续化
                gathered = [torch.zeros_like(local_pos_row) for _ in range(args.sequence_parallel_size)]  # 收集列表
                torch.distributed.all_gather(gathered, local_pos_row, group=_sp_group)  # 全部收集
                global_pos = torch.cat(gathered, dim=1)  # 拼接全局位置
                fwd_extra["cp_context"] = build_fla_cp_context_for_sample(  # 构建上下文
                    global_position_ids=global_pos,  # 全局位置 ID
                    sp_world_size=args.sequence_parallel_size,  # 序列并行世界大小
                    sp_group=_sp_group,  # 序列并行组
                    conv_kernel_size=_conv_kernel_size,  # 卷积核大小
                    local_seq_len=local_pos.shape[1],  # 本地序列长度
                )

            with accelerator.accumulate(model):  # 梯度累积上下文
                if args.load_balancing_loss:  # 若使用负载均衡损失
                    outputs = model(**batch, use_cache=False, output_router_logits=True, **fwd_extra)  # 前向传播
                    total_aux_loss += outputs.aux_loss.detach().float()  # 累加辅助损失
                else:  # 否则
                    outputs = model(**batch, use_cache=False, **fwd_extra)  # 前向传播

                loss = outputs.loss  # 获取损失
                del outputs  # 删除输出以释放内存

                if args.sequence_parallel_size > 1:  # 若序列并行 >1
                    losses_per_rank = torch.distributed.nn.functional.all_gather(loss.unsqueeze(0), group=_sp_group)  # 收集各 rank 损失
                    labels_for_counting = batch["shift_labels"]  # 用于计数的标签
                    good_tokens = (labels_for_counting != -100).view(-1).sum().float()  # 有效 token 数
                    good_tokens_per_rank = torch.distributed.nn.functional.all_gather(  # 收集各 rank 有效 token 数
                        good_tokens.unsqueeze(0), group=_sp_group  # 收集
                    )
                    total_loss_sp = sum(  # 总损失
                        losses_per_rank[rank] * good_tokens_per_rank[rank]  # 加权损失
                        for rank in range(args.sequence_parallel_size)  # 遍历 rank
                        if good_tokens_per_rank[rank] > 0  # 仅有效 token
                    )
                    total_good_tokens = sum(good_tokens_per_rank)  # 总有效 token
                    loss = total_loss_sp / torch.clamp(total_good_tokens, min=1)  # 计算平均损失

                # We keep track of the loss at each logged step  # 在每个记录步跟踪损失
                total_loss += loss.detach().float()  # 累加损失
                accelerator.backward(loss)  # 反向传播
                # clip gradient norm. don't do this with deepspeed  # 裁剪梯度范数。不要与 deepspeed 一起使用
                if accelerator.sync_gradients and args.clip_grad_norm > 0:  # 若同步梯度且裁剪范数 >0
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)  # 裁剪梯度
                optimizer.step()  # 优化器步进
                optimizer.zero_grad()  # 清零梯度
                lr_scheduler.step()  # 调度器步进

            # Checks if the accelerator has performed an optimization step behind the scenes  # 检查加速器是否已在幕后执行优化步
            if accelerator.sync_gradients:  # 若同步梯度
                progress_bar.update(1)  # 更新进度条
                completed_steps += 1  # 已完成步数加一
                if args.logging_steps and completed_steps % args.logging_steps == 0:  # 若达到记录步数
                    sum_loss = accelerator.gather(total_loss).sum().item()  # 收集总损失
                    total_tokens = accelerator.gather(local_total_tokens).sum().item()  # 收集总 token
                    total_pred_tokens = accelerator.gather(local_pred_tokens).sum().item()  # 收集预测 token
                    total_tokens_including_padding = accelerator.gather(total_token_including_padding).sum().item()  # 收集含填充 token
                    total_tokens_this_log_period = accelerator.gather(local_total_tokens_this_log_period).sum().item()  # 收集本周期 token
                    local_total_tokens_this_log_period.zero_()  # 清零
                    accelerator.gather(local_pred_tokens_this_log_period).sum().item()  # 收集本周期预测 token（未使用）
                    local_pred_tokens_this_log_period.zero_()  # 清零

                    avg_tokens_per_batch = (  # 平均每批 token
                        total_tokens  # 总 token
                        / accelerator.num_processes  # 除以进程数
                        / args.per_device_train_batch_size  # 除以每设备批大小
                        / args.gradient_accumulation_steps  # 除以梯度累积
                        / completed_steps  # 除以已完成步数
                    )
                    avg_tokens_per_batch_including_padding = (  # 平均每批含填充 token
                        total_tokens_including_padding  # 含填充总 token
                        / accelerator.num_processes  # 除以进程数
                        / args.per_device_train_batch_size  # 除以每设备批大小
                        / args.gradient_accumulation_steps  # 除以梯度累积
                        / completed_steps  # 除以已完成步数
                    )
                    avg_pred_tokens_per_batch = (  # 平均每批预测 token
                        total_pred_tokens  # 预测 token
                        / accelerator.num_processes  # 除以进程数
                        / args.per_device_train_batch_size  # 除以每设备批大小
                        / args.gradient_accumulation_steps  # 除以梯度累积
                        / completed_steps  # 除以已完成步数
                    )
                    metrics_to_log = {  # 要记录的指标
                        "learning_rate": lr_scheduler.get_last_lr()[0],  # 学习率
                        "total_tokens": total_tokens,  # 总 token
                        "total_tokens_including_padding": total_tokens_including_padding,  # 含填充总 token
                        "total_pred_tokens": total_pred_tokens,  # 预测 token
                        "total_tokens_this_log_period": total_tokens_this_log_period,  # 本周期 token
                        "avg_tokens_per_batch": avg_tokens_per_batch,  # 平均每批 token
                        "avg_tokens_per_batch_including_padding": avg_tokens_per_batch_including_padding,  # 平均每批含填充
                        "avg_pred_tokens_per_batch": avg_pred_tokens_per_batch,  # 平均每批预测
                        "per_device_tps": total_tokens  # 每设备 TPS
                        / accelerator.num_processes  # 除以进程数
                        / (time.perf_counter() - start_time),  # 除以时间
                        "per_device_tps_including_padding": total_tokens_including_padding  # 每设备含填充 TPS
                        / accelerator.num_processes  # 除以进程数
                        / (time.perf_counter() - start_time),  # 除以时间
                        "reserved_mem_GiB": torch.cuda.max_memory_reserved(device=torch.cuda.current_device()) / 2**30,  # 保留内存
                        "allocated_mem_GiB": torch.cuda.max_memory_allocated(device=torch.cuda.current_device())  # 分配内存
                        / 2**30,  # 转换为 GiB
                    }

                    # [Loss Reporting]  # [损失报告]
                    #  # 
                    # It is useful to handle loss-reporting for the "mean" and "sum" loss cases  # 处理 "mean" 和 "sum" 损失情况很有用
                    # differently.  Cases:  # 不同情况：
                    #  # 
                    # 1) "mean" loss: `sum_loss` takes individual losses which were *averaged* over  # 1) "mean" 损失：sum_loss 取在各自序列上平均的损失
                    #    the toks in their sequence and sums them over all fwd passes in the logging  # 并在日志周期内对所有前向传播求和
                    #    period.  We instead want the avg over these passes. Report avg_loss =  # 我们想要这些传播的平均。报告 avg_loss =
                    #    sum_loss / total_fwd_passes, which is roughly independent of global batch  # sum_loss / total_fwd_passes，大致独立于全局批
                    #    size.  # 大小
                    #  # 
                    # 2) "sum" loss: `sum_loss` takes individual losses which were *summed* over the  # 2) "sum" 损失：sum_loss 取在各自序列上求和的损失
                    #    toks in their sequence and sums them over all fwd passes in the logging  # 并在日志周期内对所有前向传播求和
                    #    period.  We want the avg over each optimizer step (which scales with the  # 我们想要每个优化器步的平均（随全局批大小缩放）
                    #    global batch size), and the average loss per token and per prediction  # 以及每 token 和每预测 token 的平均损失
                    #    token (which are roughly independent of global batch size).  # （大致独立于全局批大小）
                    total_fwd_passes = (  # 总前向传播次数
                        args.logging_steps * args.gradient_accumulation_steps * accelerator.num_processes  # 计算
                    )
                    avg_loss = sum_loss / total_fwd_passes  # 平均损失
                    metrics_to_log["train_loss"] = avg_loss  # 记录训练损失
                    if args.verbose:  # 若详细模式
                        sec_per_step = (time.perf_counter() - start_time) / (completed_steps - resume_step)  # 每步秒数
                        steps_remaining = args.max_train_steps - completed_steps  # 剩余步数
                        secs_remaining = steps_remaining * sec_per_step  # 剩余秒数
                        accelerator.print(  # 打印
                            f"Approx. time remaining: {timedelta(seconds=secs_remaining)}. {args.max_train_steps=}, {completed_steps=}, {steps_remaining=}"  # 剩余时间
                        )

                    if args.load_balancing_loss:  # 若使用负载均衡损失
                        avg_aux_loss = (  # 平均辅助损失
                            accelerator.gather(total_aux_loss).mean().item()  # 收集并平均
                            / args.gradient_accumulation_steps  # 除以梯度累积
                            / args.logging_steps  # 除以记录步数
                        )
                        logger.info(  # 记录日志
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, Aux Loss: {avg_aux_loss}, TPS: {total_tokens / (time.perf_counter() - start_time)}"  # 步、LR、损失、辅助损失、TPS
                        )
                        metrics_to_log["aux_loss"] = avg_aux_loss  # 记录辅助损失
                    else:  # 否则
                        logger.info(  # 记录日志
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, TPS: {total_tokens / (time.perf_counter() - start_time)}"  # 步、LR、损失、TPS
                        )
                    if args.verbose:  # 若详细模式
                        accelerator.print(f"{metrics_to_log=}")  # 打印指标
                    if args.with_tracking:  # 若启用跟踪
                        accelerator.log(metrics_to_log, step=completed_steps)  # 记录指标
                    maybe_update_beaker_description(  # 更新 Beaker 描述
                        current_step=completed_steps,  # 当前步
                        total_steps=args.max_train_steps,  # 总步
                        start_time=start_time,  # 开始时间
                        wandb_url=wandb_tracker.run.url  # wandb URL
                        if wandb_tracker is not None and accelerator.is_main_process  # 条件
                        else None,  # 否则 None
                    )
                    total_loss = 0  # 重置总损失
                    total_aux_loss = 0  # 重置总辅助损失

                if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:  # 若检查点步数为整数且达到
                    output_dir = f"step_{completed_steps}"  # 输出目录
                    if args.output_dir is not None:  # 若输出目录不为 None
                        output_dir = os.path.join(args.output_dir, output_dir)  # 拼接
                    accelerator.save_state(output_dir)  # 保存状态
                    with open(os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"), "w") as f:  # 写入完成标记
                        f.write("COMPLETED")  # 写入
                    if accelerator.is_local_main_process:  # 若是本地主进程
                        clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)  # 清理旧检查点
                    accelerator.wait_for_everyone()  # 等待所有进程

                if completed_steps >= args.max_train_steps:  # 若达到最大训练步数
                    break  # 跳出循环

        if checkpointing_steps == "epoch":  # 若检查点按轮保存
            output_dir = f"epoch_{epoch}"  # 输出目录
            if args.output_dir is not None:  # 若输出目录不为 None
                output_dir = os.path.join(args.output_dir, output_dir)  # 拼接
            accelerator.save_state(output_dir)  # 保存状态
            # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints  # 标记检查点完全保存，避免恢复损坏的检查点
            with open(os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"), "w") as f:  # 写入完成标记
                f.write("COMPLETED")  # annoyingly, empty files arent uploaded by beaker.  # 空文件不会被 beaker 上传
            if accelerator.is_local_main_process:  # 若是本地主进程
                clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)  # 清理旧检查点
            accelerator.wait_for_everyone()  # 等待所有进程

    if args.output_dir is not None:  # 若输出目录不为 None
        save_with_accelerate(  # 使用 accelerate 保存
            accelerator, model, tokenizer, args.output_dir, args.use_lora, chat_template_name=tc.chat_template_name  # 参数
        )

    # remove all checkpoints to save space  # 移除所有检查点以节省空间
    if args.clean_checkpoints_at_end and accelerator.is_main_process:  # 若结束时清理检查点且是主进程
        clean_last_n_checkpoints(args.output_dir, keep_last_n_checkpoints=0)  # 清理

    if (  # 若满足以下条件
        args.try_auto_save_to_beaker  # 尝试自动保存到 Beaker
        and accelerator.is_main_process  # 且是主进程
        and is_beaker_job()  # 且是 Beaker 任务
        and len(maybe_get_beaker_config().beaker_dataset_id_urls) > 0  # 且 Beaker 数据集 URL 非空
        and args.output_dir.rstrip("/") != "/output"  # 且输出目录不是 /output
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)  # 复制目录

    if is_beaker_job() and accelerator.is_main_process and args.try_launch_beaker_eval_jobs:  # 若为 Beaker 任务且主进程且尝试启动评估
        launch_ai2_evals_on_weka(  # 启动 AI2 评估
            path=args.output_dir,  # 路径
            leaderboard_name=args.hf_repo_revision,  # 排行榜名称
            oe_eval_max_length=args.oe_eval_max_length,  # 最大长度
            wandb_url=wandb_tracker.run.url if wandb_tracker is not None else None,  # wandb URL
            oe_eval_tasks=args.oe_eval_tasks,  # 评估任务
        )
    if args.push_to_hub and accelerator.is_main_process:  # 若推送到 Hub 且主进程
        push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)  # 推送文件夹到 Hub
    accelerator.wait_for_everyone()  # 等待所有进程
    if args.with_tracking:  # 若启用跟踪
        accelerator.end_training()  # 结束训练


if __name__ == "__main__":
    warnings.warn(
        "finetune.py is deprecated. Use the OLMo-core SFT implementation instead: "
        "https://github.com/allenai/OLMo-core/tree/main/src/scripts/train/sft",
        DeprecationWarning,
        stacklevel=1,
    )

    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
