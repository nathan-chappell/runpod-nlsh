#!/usr/bin/env python
from __future__ import annotations

import inspect
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import typer

from nlsh.dataio import DEFAULT_EVAL_FRACTION, default_dataset_path, load_jsonl, partition_records

DEFAULT_MODEL_ID = "microsoft/Phi-4-mini-instruct"
DEFAULT_DATASET = Path("data/samples")
VALID_ATTN_IMPLEMENTATIONS = ("auto", "flash_attention_2", "sdpa", "eager")
VALID_TORCH_DTYPES = ("bf16", "fp16", "fp32")
OOM_TEXT_PATTERNS = (
    "out of memory",
    "cuda error: out of memory",
    "cuda out of memory",
    "hip out of memory",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
    "insufficient memory",
)


@dataclass(frozen=True, slots=True)
class PreparedDatasets:
    train: list[dict[str, Any]]
    eval: list[dict[str, Any]] | None


@dataclass(slots=True)
class TrainingArgs:
    model_id: str
    dataset: Path
    train_dataset: Path | None
    eval_dataset: Path | None
    eval_fraction: float
    output_dir: Path
    workspace: Path | None
    dry_run: bool
    attn_implementation: str
    torch_dtype: str
    trust_remote_code: bool
    max_length: int
    packing: bool
    no_eval: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    num_train_epochs: float
    max_steps: int
    warmup_ratio: float
    lr_scheduler_type: str
    logging_steps: int
    save_steps: int
    save_total_limit: int
    seed: int
    resume_from_checkpoint: Path | None
    report_to: str
    dataset_num_proc: int
    overwrite_output_dir: bool
    oom_retries: int
    min_train_batch_size: int
    min_eval_batch_size: int
    min_max_length: int


app = typer.Typer(
    add_completion=False,
    help="Fine-tune Phi-4-mini-instruct on nlsh JSONL data with single-GPU LoRA.",
    rich_markup_mode=None,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def default_output_dir() -> Path:
    workspace = Path("/workspace")
    if workspace.exists():
        return workspace / "nlsh-finetune/phi-4-mini-instruct-lora"
    return Path("outputs/phi-4-mini-instruct-nlsh-lora")


def parse_target_modules(raw: str) -> list[str]:
    modules = [item.strip() for item in raw.split(",") if item.strip()]
    if not modules:
        raise typer.BadParameter("--target-modules must name at least one module")
    return modules


def parse_eval_fraction(value: float) -> float:
    if value < 0 or value >= 1:
        raise typer.BadParameter("--eval-fraction must be >= 0 and < 1")
    return value


def parse_positive_int(value: int) -> int:
    if value < 1:
        raise typer.BadParameter("value must be at least 1")
    return value


def parse_non_negative_int(value: int) -> int:
    if value < 0:
        raise typer.BadParameter("value must be at least 0")
    return value


def parse_attn_implementation(value: str) -> str:
    if value not in VALID_ATTN_IMPLEMENTATIONS:
        choices = ", ".join(VALID_ATTN_IMPLEMENTATIONS)
        raise typer.BadParameter(f"--attn-implementation must be one of: {choices}")
    return value


def parse_torch_dtype(value: str) -> str:
    if value not in VALID_TORCH_DTYPES:
        choices = ", ".join(VALID_TORCH_DTYPES)
        raise typer.BadParameter(f"--torch-dtype must be one of: {choices}")
    return value


def _normalize_role(role: str) -> str:
    if role == "developer":
        return "system"
    if role in {"system", "user", "assistant"}:
        return role
    raise ValueError(f"Unsupported chat role: {role!r}")


def _normalize_message(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("Each message must be an object")
    role = raw.get("role")
    content = raw.get("content")
    if not isinstance(role, str):
        raise ValueError("Each message must contain a string role")
    if not isinstance(content, str):
        raise ValueError("Each message must contain string content")
    return {"role": _normalize_role(role), "content": content}


def to_prompt_completion_record(record: dict[str, Any], *, row_number: int, path: Path) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path} row {row_number} must contain a non-empty messages list")
    normalized = [_normalize_message(message) for message in messages]
    assistant_index = None
    for index, message in enumerate(normalized):
        if message["role"] == "assistant":
            assistant_index = index
    if assistant_index is None:
        raise ValueError(f"{path} row {row_number} must contain an assistant message")
    if assistant_index == 0:
        raise ValueError(f"{path} row {row_number} assistant message cannot be first")

    return {
        "prompt": normalized[:assistant_index],
        "completion": [normalized[assistant_index]],
    }


def load_prompt_completion_dataset(path: Path) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    return to_prompt_completion_records(records, path=path)


def to_prompt_completion_records(records: list[dict[str, Any]], *, path: Path) -> list[dict[str, Any]]:
    return [to_prompt_completion_record(record, row_number=index, path=path) for index, record in enumerate(records, start=1)]


def prepare_datasets(args: TrainingArgs) -> PreparedDatasets:
    using_explicit_datasets = args.train_dataset is not None or args.eval_dataset is not None
    if using_explicit_datasets:
        if args.train_dataset is None:
            raise ValueError("--train-dataset is required when --eval-dataset is set")
        if not args.no_eval and args.eval_dataset is None:
            raise ValueError("--eval-dataset is required with --train-dataset unless --no-eval is set")
        return PreparedDatasets(
            train=load_prompt_completion_dataset(args.train_dataset),
            eval=None if args.no_eval else load_prompt_completion_dataset(args.eval_dataset),
        )

    records = load_jsonl(args.dataset)
    if args.no_eval:
        train_records = records
        eval_records = None
    else:
        train_records, eval_records = partition_records(records, eval_fraction=args.eval_fraction)
    return PreparedDatasets(
        train=to_prompt_completion_records(train_records, path=args.dataset),
        eval=None if eval_records is None else to_prompt_completion_records(eval_records, path=args.dataset),
    )


def configure_workspace_cache(workspace: Path | None) -> None:
    if workspace is None:
        return
    cache_paths = {
        "HF_HOME": workspace / "hf-cache",
        "TMPDIR": workspace / "tmp",
        "TRITON_CACHE_DIR": workspace / "triton-cache",
        "XDG_CACHE_HOME": workspace / ".cache",
    }
    for key, path in cache_paths.items():
        os.environ.setdefault(key, str(path))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])


def _dtype(torch_module: Any, name: str) -> Any:
    if name == "bf16":
        return torch_module.bfloat16
    if name == "fp16":
        return torch_module.float16
    return torch_module.float32


def _training_imports() -> dict[str, Any]:
    missing: list[str] = []
    imported: dict[str, Any] = {}
    for module_name in ("datasets", "peft", "torch", "transformers", "trl"):
        try:
            imported[module_name] = __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        package_list = ", ".join(missing)
        raise SystemExit(
            f'Missing training dependencies: {package_list}. Install them with `pip install -e ".[dev,train]"`.'
        )
    return imported


def _sft_config_kwargs(args: TrainingArgs, has_eval: bool) -> dict[str, Any]:
    eval_strategy = "steps" if has_eval else "no"
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "do_eval": has_eval,
        "learning_rate": args.learning_rate,
        "log_level": "info",
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "lr_scheduler_type": args.lr_scheduler_type,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "bf16": args.torch_dtype == "bf16",
        "fp16": args.torch_dtype == "fp16",
        "packing": args.packing,
        "completion_only_loss": True,
        "dataset_num_proc": args.dataset_num_proc,
        "report_to": args.report_to,
        "remove_unused_columns": True,
    }
    kwargs["eval_steps"] = args.save_steps
    kwargs["eval_strategy"] = eval_strategy
    kwargs["evaluation_strategy"] = eval_strategy
    kwargs["max_length"] = args.max_length
    kwargs["max_seq_length"] = args.max_length
    return kwargs


def _make_sft_config(config_class: Any, args: TrainingArgs, has_eval: bool) -> Any:
    supported = set(inspect.signature(config_class).parameters)
    kwargs = {key: value for key, value in _sft_config_kwargs(args, has_eval).items() if key in supported}
    return config_class(**kwargs)


def _load_model(
    *,
    args: TrainingArgs,
    torch_module: Any,
    auto_model_class: Any,
) -> tuple[Any, str]:
    if args.attn_implementation == "auto":
        implementations = ["sdpa"]
        if find_spec("flash_attn") is not None:
            implementations.insert(0, "flash_attention_2")
    else:
        implementations = [args.attn_implementation]

    model_kwargs = {
        "use_cache": False,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": _dtype(torch_module, args.torch_dtype),
        "device_map": None,
    }
    last_error: Exception | None = None
    for implementation in implementations:
        try:
            model = auto_model_class.from_pretrained(
                args.model_id,
                attn_implementation=implementation,
                **model_kwargs,
            )
            model.config.use_cache = False
            return model, implementation
        except Exception as exc:
            if args.attn_implementation != "auto":
                raise
            last_error = exc
            print(f"Could not load with {implementation}: {exc}", file=sys.stderr)
    raise RuntimeError("Could not load model with any attention implementation") from last_error


def _make_trainer_kwargs(
    *,
    trainer_class: Any,
    model: Any,
    sft_config: Any,
    peft_config: Any,
    train_dataset: Any,
    eval_dataset: Any | None,
    tokenizer: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "peft_config": peft_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    supported = set(inspect.signature(trainer_class).parameters)
    if "processing_class" in supported:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported:
        kwargs["tokenizer"] = tokenizer
    return kwargs


def build_dry_run_payload(args: TrainingArgs, datasets: PreparedDatasets) -> dict[str, Any]:
    return {
        "model_id": args.model_id,
        "dataset": str(args.dataset),
        "train_dataset": None if args.train_dataset is None else str(args.train_dataset),
        "train_records": len(datasets.train),
        "eval_dataset": None if args.no_eval or args.eval_dataset is None else str(args.eval_dataset),
        "eval_records": None if datasets.eval is None else len(datasets.eval),
        "eval_fraction": args.eval_fraction,
        "output_dir": str(args.output_dir),
        "attn_implementation": args.attn_implementation,
        "torch_dtype": args.torch_dtype,
        "trust_remote_code": args.trust_remote_code,
        "max_length": args.max_length,
        "packing": args.packing,
        "peft_config": {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": args.target_modules,
            "modules_to_save": None,
        },
        "training": {
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_checkpointing": args.gradient_checkpointing,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "max_length": args.max_length,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "oom_retries": args.oom_retries,
        },
    }


def train(args: TrainingArgs, datasets: PreparedDatasets) -> None:
    imports = _training_imports()
    hf_datasets = imports["datasets"]
    peft = imports["peft"]
    torch = imports["torch"]
    transformers = imports["transformers"]
    trl = imports["trl"]

    if args.torch_dtype == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("Warning: CUDA is available but bf16 is not reported as supported.", file=sys.stderr)

    model, resolved_attention = _load_model(
        args=args,
        torch_module=torch,
        auto_model_class=transformers.AutoModelForCausalLM,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.model_max_length = args.max_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "right"

    train_dataset = hf_datasets.Dataset.from_list(datasets.train)
    eval_dataset = None if datasets.eval is None else hf_datasets.Dataset.from_list(datasets.eval)
    peft_config = peft.LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
        modules_to_save=None,
    )
    sft_config = _make_sft_config(trl.SFTConfig, args, has_eval=eval_dataset is not None)
    trainer = trl.SFTTrainer(
        **_make_trainer_kwargs(
            trainer_class=trl.SFTTrainer,
            model=model,
            sft_config=sft_config,
            peft_config=peft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
    )

    train_result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    info = {
        "base_model": args.model_id,
        "adapter_type": "lora",
        "resolved_attention_implementation": resolved_attention,
        "source": "scripts/phi_4_training.py",
        "args": build_dry_run_payload(args, datasets),
    }
    (args.output_dir / "adapter_run_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _is_oom_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(pattern in text for pattern in OOM_TEXT_PATTERNS)


def _clear_cuda_cache() -> None:
    try:
        import torch  # type: ignore
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    except Exception:
        return


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-", 1)[1]) if path.name.split("-", 1)[1].isdigit() else -1,
    )
    return checkpoints[-1] if checkpoints else None


def _reduced_batch_size(current: int, minimum: int) -> int | None:
    if current <= minimum:
        return None
    reduced = max(minimum, current // 2)
    if reduced >= current:
        reduced = current - 1
    return reduced if reduced >= minimum else None


def _reduce_training_footprint(args: TrainingArgs) -> dict[str, dict[str, int]] | None:
    changes: dict[str, dict[str, int]] = {}

    next_train_batch = _reduced_batch_size(args.per_device_train_batch_size, args.min_train_batch_size)
    if next_train_batch is not None:
        previous_batch = args.per_device_train_batch_size
        previous_accumulation = args.gradient_accumulation_steps
        factor = max(1, math.ceil(previous_batch / next_train_batch))
        args.per_device_train_batch_size = next_train_batch
        args.gradient_accumulation_steps = max(1, previous_accumulation * factor)
        changes["per_device_train_batch_size"] = {"from": previous_batch, "to": args.per_device_train_batch_size}
        changes["gradient_accumulation_steps"] = {"from": previous_accumulation, "to": args.gradient_accumulation_steps}

        next_eval_batch = _reduced_batch_size(args.per_device_eval_batch_size, args.min_eval_batch_size)
        if next_eval_batch is not None:
            previous_eval_batch = args.per_device_eval_batch_size
            args.per_device_eval_batch_size = next_eval_batch
            changes["per_device_eval_batch_size"] = {"from": previous_eval_batch, "to": args.per_device_eval_batch_size}
        return changes

    next_eval_batch = _reduced_batch_size(args.per_device_eval_batch_size, args.min_eval_batch_size)
    if next_eval_batch is not None:
        previous_eval_batch = args.per_device_eval_batch_size
        args.per_device_eval_batch_size = next_eval_batch
        changes["per_device_eval_batch_size"] = {"from": previous_eval_batch, "to": args.per_device_eval_batch_size}
        return changes

    if args.max_length > args.min_max_length:
        previous_max_length = args.max_length
        args.max_length = max(args.min_max_length, args.max_length // 2)
        changes["max_length"] = {"from": previous_max_length, "to": args.max_length}
        return changes

    return None


def _training_state_payload(args: TrainingArgs, datasets: PreparedDatasets) -> dict[str, Any]:
    return {
        "started_at": _utc_now(),
        "finished_at": None,
        "status": "running",
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "attempts": [],
        "resolved_args": build_dry_run_payload(args, datasets),
    }


def train_with_retries(args: TrainingArgs, datasets: PreparedDatasets) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "training_state.json"
    state = _training_state_payload(args, datasets)
    _write_json(state_path, state)

    for attempt in range(1, args.oom_retries + 2):
        attempt_state: dict[str, Any] = {
            "attempt": attempt,
            "started_at": _utc_now(),
            "status": "running",
            "config": {
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "per_device_eval_batch_size": args.per_device_eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_length": args.max_length,
                "resume_from_checkpoint": None if args.resume_from_checkpoint is None else str(args.resume_from_checkpoint),
            },
        }
        state["attempts"].append(attempt_state)
        state["resolved_args"] = build_dry_run_payload(args, datasets)
        _write_json(state_path, state)

        try:
            train(args, datasets)
            attempt_state["status"] = "completed"
            attempt_state["finished_at"] = _utc_now()
            state["status"] = "completed"
            state["finished_at"] = attempt_state["finished_at"]
            _write_json(state_path, state)
            return
        except Exception as exc:
            attempt_state["finished_at"] = _utc_now()
            attempt_state["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "oom": _is_oom_error(exc),
            }
            latest_checkpoint = _latest_checkpoint(args.output_dir)
            if latest_checkpoint is not None:
                attempt_state["latest_checkpoint"] = str(latest_checkpoint)

            if _is_oom_error(exc):
                _clear_cuda_cache()
                changes = _reduce_training_footprint(args)
                if changes is not None and attempt <= args.oom_retries:
                    if latest_checkpoint is not None:
                        args.resume_from_checkpoint = latest_checkpoint
                    attempt_state["status"] = "retrying"
                    attempt_state["retry_adjustments"] = changes
                    print(
                        f"retrying training after OOM with adjustments: {json.dumps(changes, ensure_ascii=False)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    state["resolved_args"] = build_dry_run_payload(args, datasets)
                    _write_json(state_path, state)
                    continue

            attempt_state["status"] = "failed"
            state["status"] = "failed"
            state["finished_at"] = attempt_state["finished_at"]
            _write_json(state_path, state)
            raise


@app.command()
def run(
    model_id: str = typer.Option(DEFAULT_MODEL_ID),
    dataset: Path = typer.Option(
        DEFAULT_DATASET if DEFAULT_DATASET.exists() else default_dataset_path(),
        help="Canonical JSONL file or directory to partition for training and eval.",
    ),
    train_dataset: Path | None = typer.Option(None, help="Explicit training JSONL file or directory."),
    eval_dataset: Path | None = typer.Option(None, help="Explicit eval JSONL file or directory."),
    eval_fraction: float = typer.Option(DEFAULT_EVAL_FRACTION),
    output_dir: Path = typer.Option(default_output_dir()),
    workspace: Path | None = typer.Option(Path("/workspace") if Path("/workspace").exists() else None),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate inputs and print the resolved config."),
    attn_implementation: str = typer.Option("auto", help="One of: auto, flash_attention_2, sdpa, eager."),
    torch_dtype: str = typer.Option("bf16", help="One of: bf16, fp16, fp32."),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code/--no-trust-remote-code"),
    max_length: int = typer.Option(2048),
    packing: bool = typer.Option(False, "--packing/--no-packing"),
    no_eval: bool = typer.Option(False, "--no-eval"),
    lora_r: int = typer.Option(8),
    lora_alpha: int = typer.Option(16),
    lora_dropout: float = typer.Option(0.05),
    target_modules: str = typer.Option("qkv_proj"),
    per_device_train_batch_size: int = typer.Option(1),
    per_device_eval_batch_size: int = typer.Option(1),
    gradient_accumulation_steps: int = typer.Option(8),
    gradient_checkpointing: bool = typer.Option(True, "--gradient-checkpointing/--no-gradient-checkpointing"),
    learning_rate: float = typer.Option(2.0e-4),
    num_train_epochs: float = typer.Option(3.0),
    max_steps: int = typer.Option(-1),
    warmup_ratio: float = typer.Option(0.03),
    lr_scheduler_type: str = typer.Option("cosine"),
    logging_steps: int = typer.Option(10),
    save_steps: int = typer.Option(10),
    save_total_limit: int = typer.Option(1),
    seed: int = typer.Option(0),
    resume_from_checkpoint: Path | None = typer.Option(None),
    report_to: str = typer.Option("none"),
    dataset_num_proc: int = typer.Option(1),
    overwrite_output_dir: bool = typer.Option(True, "--overwrite-output-dir/--no-overwrite-output-dir"),
    oom_retries: int = typer.Option(3),
    min_train_batch_size: int = typer.Option(1),
    min_eval_batch_size: int = typer.Option(1),
    min_max_length: int = typer.Option(512, "--min-max-length"),
) -> None:
    args = TrainingArgs(
        model_id=model_id,
        dataset=dataset,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_fraction=parse_eval_fraction(eval_fraction),
        output_dir=output_dir,
        workspace=workspace,
        dry_run=dry_run,
        attn_implementation=parse_attn_implementation(attn_implementation),
        torch_dtype=parse_torch_dtype(torch_dtype),
        trust_remote_code=trust_remote_code,
        max_length=max_length,
        packing=packing,
        no_eval=no_eval,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=parse_target_modules(target_modules),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        resume_from_checkpoint=resume_from_checkpoint,
        report_to=report_to,
        dataset_num_proc=dataset_num_proc,
        overwrite_output_dir=overwrite_output_dir,
        oom_retries=parse_non_negative_int(oom_retries),
        min_train_batch_size=parse_positive_int(min_train_batch_size),
        min_eval_batch_size=parse_positive_int(min_eval_batch_size),
        min_max_length=parse_positive_int(min_max_length),
    )

    configure_workspace_cache(args.workspace)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        datasets = prepare_datasets(args)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if args.dry_run:
        print(json.dumps(build_dry_run_payload(args, datasets), indent=2, ensure_ascii=False))
        return

    train_with_retries(args, datasets)


if __name__ == "__main__":
    app()
