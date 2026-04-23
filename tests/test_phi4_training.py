import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType


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


def test_phi4_training_dry_run() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/phi_4_training.py", "--dry-run", "--no-eval"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["model_id"] == "microsoft/Phi-4-mini-instruct"
    assert payload["train_records"] == 5
    assert payload["eval_records"] is None
    assert payload["peft_config"]["r"] == 8
    assert payload["peft_config"]["lora_alpha"] == 16
    assert payload["peft_config"]["target_modules"] == ["qkv_proj"]


def test_phi4_training_maps_developer_to_system() -> None:
    module = load_training_module()
    record = {
        "messages": [
            {"role": "developer", "content": "Return JSON only."},
            {"role": "user", "content": "find pdfs"},
            {"role": "assistant", "content": "{\"kind\":\"plan\",\"steps\":[]}"},
        ]
    }

    converted = module.to_prompt_completion_record(record, row_number=1, path=Path("example.jsonl"))

    assert converted == {
        "prompt": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "find pdfs"},
        ],
        "completion": [
            {"role": "assistant", "content": "{\"kind\":\"plan\",\"steps\":[]}"}
        ],
    }


def test_phi4_training_target_module_parser() -> None:
    module = load_training_module()

    assert module.parse_target_modules("qkv_proj,o_proj") == ["qkv_proj", "o_proj"]
