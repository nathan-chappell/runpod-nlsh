#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "microsoft/Phi-4-mini-instruct"
DEFAULT_TRAIN_DATASET = Path("data/train.messages.jsonl")
DEFAULT_EVAL_DATASET = Path("data/dev.messages.jsonl")


@dataclass(frozen=True, slots=True)
class PreparedDatasets:
    train: list[dict[str, Any]]
    eval: list[dict[str, Any]] | None


def default_output_dir() -> Path:
    workspace = Path("/workspace")
    if workspace.exists():
        return workspace / "nlsh-finetune/phi-4-mini-instruct-lora"
    return Path("outputs/phi-4-mini-instruct-nlsh-lora")


def parse_target_modules(raw: str) -> list[str]:
    modules = [item.strip() for item in raw.split(",") if item.strip()]
    if not modules:
        raise argparse.ArgumentTypeError("--target-modules must name at least one module")
    return modules


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
    return records


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
    records = _read_jsonl(path)
    return [
        to_prompt_completion_record(record, row_number=index, path=path)
        for index, record in enumerate(records, start=1)
    ]


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Phi-4-mini-instruct on nlsh JSONL data with single-GPU LoRA.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--eval-dataset", type=Path, default=DEFAULT_EVAL_DATASET)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--workspace", type=Path, default=Path("/workspace") if Path("/workspace").exists() else None)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the resolved config.")

    parser.add_argument("--attn-implementation", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--torch-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-eval", action="store_true")

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=parse_target_modules, default=["qkv_proj"])

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--overwrite-output-dir", action=argparse.BooleanOptionalAction, default=True)
    return parser


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
            f"Missing training dependencies: {package_list}. "
            "Install them with `pip install -e .[train]` on the Runpod PyTorch image."
        )
    return imported


def _sft_config_kwargs(args: argparse.Namespace, has_eval: bool) -> dict[str, Any]:
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


def _make_sft_config(config_class: Any, args: argparse.Namespace, has_eval: bool) -> Any:
    supported = set(inspect.signature(config_class).parameters)
    kwargs = {
        key: value
        for key, value in _sft_config_kwargs(args, has_eval).items()
        if key in supported
    }
    return config_class(**kwargs)


def _load_model(
    *,
    args: argparse.Namespace,
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


def build_dry_run_payload(args: argparse.Namespace, datasets: PreparedDatasets) -> dict[str, Any]:
    return {
        "model_id": args.model_id,
        "train_dataset": str(args.train_dataset),
        "train_records": len(datasets.train),
        "eval_dataset": None if args.no_eval else str(args.eval_dataset),
        "eval_records": None if datasets.eval is None else len(datasets.eval),
        "output_dir": str(args.output_dir),
        "attn_implementation": args.attn_implementation,
        "torch_dtype": args.torch_dtype,
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
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_checkpointing": args.gradient_checkpointing,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
        },
    }


def train(args: argparse.Namespace, datasets: PreparedDatasets) -> None:
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

    train_result = trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    )
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
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


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_workspace_cache(args.workspace)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    datasets = PreparedDatasets(
        train=load_prompt_completion_dataset(args.train_dataset),
        eval=None if args.no_eval else load_prompt_completion_dataset(args.eval_dataset),
    )

    if args.dry_run:
        print(json.dumps(build_dry_run_payload(args, datasets), indent=2, ensure_ascii=False))
        return 0

    train(args, datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
