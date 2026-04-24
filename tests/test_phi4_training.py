import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from nlsh.dataio import load_jsonl, partition_records


def load_training_module() -> ModuleType:
    path = Path("scripts/phi_4_training.py")
    spec = importlib.util.spec_from_file_location("phi_4_training", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phi4_training_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/phi_4_training.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--target-modules" in result.stdout
    assert "--attn-implementation" in result.stdout
    assert "--oom-retries" in result.stdout
    assert "--min-max-length" in result.stdout


def test_phi4_training_dry_run() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/phi_4_training.py", "--dry-run", "--no-eval"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    expected_records = load_jsonl(Path("data/samples"))
    assert payload["model_id"] == "microsoft/Phi-4-mini-instruct"
    assert payload["dataset"] == "data/samples"
    assert payload["train_records"] == len(expected_records)
    assert payload["eval_records"] is None
    assert payload["trust_remote_code"] is False
    assert payload["peft_config"]["r"] == 8
    assert payload["peft_config"]["lora_alpha"] == 16
    assert payload["peft_config"]["target_modules"] == ["qkv_proj"]
    assert payload["training"]["per_device_train_batch_size"] == 4
    assert payload["training"]["per_device_eval_batch_size"] == 4
    assert payload["training"]["gradient_accumulation_steps"] == 4
    assert payload["training"]["learning_rate"] == 5.0e-4
    assert payload["training"]["num_train_epochs"] == 10.0
    expected_steps_per_epoch = math.ceil(math.ceil(len(expected_records) / 4) / 4)
    assert payload["training"]["steps_per_epoch"] == expected_steps_per_epoch
    assert payload["training"]["logging_steps"] == min(10, expected_steps_per_epoch)
    assert payload["training"]["evaluation_strategy"] == "no"
    assert payload["training"]["save_strategy"] == "epoch"


def test_phi4_training_auto_partitions_default_dataset() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/phi_4_training.py", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    train_records, eval_records = partition_records(load_jsonl(Path("data/samples")))
    assert payload["dataset"] == "data/samples"
    assert payload["train_records"] == len(train_records)
    assert payload["eval_records"] == len(eval_records)


def test_phi4_training_maps_developer_to_system() -> None:
    module = load_training_module()
    record = {
        "messages": [
            {"role": "developer", "content": "Return JSON only."},
            {"role": "user", "content": "find pdfs"},
            {"role": "assistant", "content": '{"kind":"plan","steps":[]}'},
        ]
    }

    converted = module.to_prompt_completion_record(record, row_number=1, path=Path("example.jsonl"))

    assert converted == {
        "prompt": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "find pdfs"},
        ],
        "completion": [{"role": "assistant", "content": '{"kind":"plan","steps":[]}'}],
    }


def test_phi4_training_target_module_parser() -> None:
    module = load_training_module()

    assert module.parse_target_modules("qkv_proj,o_proj") == ["qkv_proj", "o_proj"]


def test_phi4_training_reduces_footprint_before_retry() -> None:
    module = load_training_module()
    args = SimpleNamespace(
        per_device_train_batch_size=8,
        min_train_batch_size=1,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=4,
        min_eval_batch_size=1,
        max_length=2048,
        min_max_length=512,
    )

    changes = module._reduce_training_footprint(args)

    assert changes == {
        "per_device_train_batch_size": {"from": 8, "to": 4},
        "gradient_accumulation_steps": {"from": 2, "to": 4},
        "per_device_eval_batch_size": {"from": 4, "to": 2},
    }
    assert args.per_device_train_batch_size == 4
    assert args.gradient_accumulation_steps == 4
    assert args.per_device_eval_batch_size == 2


def test_phi4_training_rejects_incompatible_torchao() -> None:
    module = load_training_module()

    with (
        patch.object(module, "find_spec", return_value=object()),
        patch.object(module.importlib_metadata, "version", return_value="0.9.0"),
    ):
        try:
            module.ensure_compatible_torchao()
        except RuntimeError as exc:
            message = str(exc)
        else:  # pragma: no cover - defensive assertion.
            raise AssertionError("expected ensure_compatible_torchao() to fail")

    assert "torchao 0.9.0" in message
    assert "torchao >= 0.16.0" in message
    assert "Rebuild the pod image" in message


def test_phi4_training_normalizes_metric_history(tmp_path: Path) -> None:
    module = load_training_module()
    rows = module._normalize_metric_history(
        [
            {
                "step": 10,
                "epoch": 2.0,
                "loss": 1.23,
                "learning_rate": 0.001,
                "grad_norm": 0.4,
                "mean_token_accuracy": 0.8,
                "entropy": 1.5,
            },
            {
                "step": 10,
                "epoch": 2.0,
                "eval_loss": 0.5,
                "eval_mean_token_accuracy": 0.9,
                "eval_entropy": 1.2,
                "eval_runtime": 0.2,
            },
        ]
    )

    assert rows == [
        {
            "phase": "train",
            "step": 10,
            "epoch": 2.0,
            "loss": 1.23,
            "token_accuracy": 0.8,
            "entropy": 1.5,
            "learning_rate": 0.001,
            "grad_norm": 0.4,
            "num_tokens": None,
        },
        {
            "phase": "eval",
            "step": 10,
            "epoch": 2.0,
            "loss": 0.5,
            "token_accuracy": 0.9,
            "entropy": 1.2,
            "learning_rate": None,
            "grad_norm": None,
            "num_tokens": None,
            "runtime": 0.2,
            "samples_per_second": None,
            "steps_per_second": None,
        },
    ]

    json_path, csv_path = module._write_metrics_history(tmp_path, rows)

    assert json.loads(json_path.read_text()) == rows
    csv_text = csv_path.read_text()
    assert "phase,step,epoch,loss" in csv_text
    assert "train,10,2.0,1.23" in csv_text
