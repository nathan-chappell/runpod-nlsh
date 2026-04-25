from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from nlsh.prompts import TRAINING_DEVELOPER_PROMPT


SKILL_SCRIPT = Path(".codex/skills/runpod-live-test/scripts/runpod_live_test.py")


def _load_skill_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("runpod_live_test_skill", SKILL_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runpod_live_test_skill_exists() -> None:
    assert Path(".codex/skills/runpod-live-test/SKILL.md").exists()
    assert SKILL_SCRIPT.exists()


def test_runpod_live_test_resolve_url() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL_SCRIPT),
            "resolve-url",
            "--pod-id",
            "abc123xyz",
            "--port",
            "8000",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "RUNPOD_SERVE_API_KEY": "", "RUNPOD_POD_ID": ""},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["proxy_url"] == "https://abc123xyz-8000.proxy.runpod.net"
    assert payload["openai_base_url"] == "https://abc123xyz-8000.proxy.runpod.net/v1"
    assert payload["api_key_configured"] is False


def test_runpod_live_test_resolve_url_accepts_v1_proxy_url() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL_SCRIPT),
            "resolve-url",
            "--proxy-url",
            "https://abc123xyz-8000.proxy.runpod.net/v1",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "RUNPOD_SERVE_API_KEY": "", "RUNPOD_POD_ID": ""},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["proxy_url"] == "https://abc123xyz-8000.proxy.runpod.net"
    assert payload["openai_base_url"] == "https://abc123xyz-8000.proxy.runpod.net/v1"


def test_probe_dataset_writes_runtime_artifact(monkeypatch, tmp_path, capsys) -> None:
    module = _load_skill_module()
    records = module._load_probe_records(Path("data/samples"))
    prompt_to_response = {
        record["prompt"]: json.dumps(record["plan"], ensure_ascii=False)
        for record in records
    }

    def fake_chat_completion_text(**kwargs: object) -> str:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        prompt = messages[-1]["content"]
        return prompt_to_response[prompt]

    monkeypatch.setattr(module, "chat_completion_text", fake_chat_completion_text)
    args = argparse.Namespace(
        pod_id=None,
        proxy_url="https://abc123xyz-8000.proxy.runpod.net",
        port=8000,
        api_key="",
        model="nlsh-phi4-ft",
        timeout=5.0,
        max_tokens=600,
        dataset=str(Path("data/samples")),
        seed=7,
        mode="runtime",
        artifact_dir=str(tmp_path),
    )

    exit_code = module.command_probe_dataset(args)

    assert exit_code == 0
    payload = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert payload["summary"]["count"] == 10
    assert payload["summary"]["exact_matches"] == 10
    assert payload["summary"]["transport_or_parse_failures"] == 0
    assert {item["bucket"] for item in payload["results"]} == set(module.PROBE_BUCKETS)
    for item in payload["results"]:
        assert item["sent_messages"][0]["role"] == "system"
        assert item["sent_messages"][0]["content"] == TRAINING_DEVELOPER_PROMPT
    output = capsys.readouterr().out
    assert "artifact" in output


def test_interactive_demo_skips_clarification_then_executes(monkeypatch, tmp_path) -> None:
    module = _load_skill_module()
    monkeypatch.setattr(
        module,
        "_demo_prompt_entries",
        lambda: [
            {"name": "clarify", "prompt": "ambiguous prompt", "primary_output": ""},
            {
                "name": "csv_to_json",
                "prompt": "convert orders.csv to JSON in orders.json",
                "primary_output": "orders.json",
            },
        ],
    )

    responses = {
        "ambiguous prompt": json.dumps({"kind": "clarification", "question": "Which file?"}),
        "convert orders.csv to JSON in orders.json": json.dumps(
            {
                "kind": "plan",
                "steps": [
                    {
                        "kind": "csv_to_json",
                        "input_file": "orders.csv",
                        "output_file": "orders.json",
                    }
                ],
            }
        ),
    }

    monkeypatch.setattr(
        module,
        "chat_completion_text",
        lambda **kwargs: responses[kwargs["messages"][-1]["content"]],
    )

    args = argparse.Namespace(
        pod_id=None,
        proxy_url="https://abc123xyz-8000.proxy.runpod.net",
        port=8000,
        api_key="",
        model="nlsh-phi4-ft",
        timeout=5.0,
        max_tokens=600,
        sandbox_root=str(tmp_path),
        max_attempts=2,
        yes=True,
        allow_overwrite=False,
    )

    exit_code = module.command_interactive_demo(args)

    assert exit_code == 0
    sandboxes = sorted(tmp_path.iterdir())
    assert len(sandboxes) == 1
    sandbox = sandboxes[0]
    transcript = json.loads((sandbox / "execution-transcript.json").read_text(encoding="utf-8"))
    assert transcript["success"] is True
    assert transcript["attempts"][0]["status"] == "clarification"
    assert transcript["attempts"][1]["status"] == "executed"
    orders_json = json.loads((sandbox / "orders.json").read_text(encoding="utf-8"))
    assert len(orders_json) == 3
    assert orders_json[0]["order_id"] == "1001"
