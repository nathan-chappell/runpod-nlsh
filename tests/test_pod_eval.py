import json
import os
import subprocess
import sys
from pathlib import Path


def test_pod_eval_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/pod_eval.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "download-models" in result.stdout
    assert "run-suite" in result.stdout


def test_pod_eval_manifest_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/pod_eval.py",
            "--manifest",
            "configs/pod_eval_models.json",
            "download-models",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["models"] == [
        "microsoft/Phi-4-mini-instruct",
        "Qwen/Qwen3-8B",
        "HuggingFaceTB/SmolLM3-3B",
    ]


def test_pod_eval_dockerfile_is_lean_runpod_image() -> None:
    dockerfile = Path("Dockerfile.pod-eval").read_text(encoding="utf-8")

    assert "WORKDIR /opt/nlsh" in dockerfile
    assert "HF_HOME=/workspace/hf-cache" in dockerfile
    assert "python scripts/pod_eval.py download-models" not in dockerfile
    assert 'CMD ["bash", "scripts/runpod_pod_eval.sh"]' in dockerfile


def test_runpod_startup_script_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", "scripts/runpod_pod_eval.sh"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_runpod_startup_requires_hf_token(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.update(
        {
            "POD_EVAL_APP_DIR": str(Path.cwd()),
            "POD_EVAL_WORKSPACE": str(tmp_path),
            "POD_EVAL_EXIT_AFTER": "1",
        }
    )

    result = subprocess.run(
        ["bash", "scripts/runpod_pod_eval.sh"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 2
    assert "HF_TOKEN is required" in result.stdout
    assert (tmp_path / "nlsh-artifacts" / "last_exit_code").read_text().strip() == "2"


def test_pod_eval_gold_model_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "pod_eval"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/pod_eval.py",
            "run-model",
            "--planner",
            "gold",
            "--model",
            "microsoft/Phi-4-mini-instruct",
            "--dataset",
            "data/dev.messages.jsonl",
            "--limit",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((output_dir / "microsoft_Phi-4-mini-instruct" / "report.json").read_text())
    assert report["count"] == 2
    assert report["exact_match_rate"] == 1.0
    assert report["compile_valid_rate"] == 1.0
    assert report["slot_accuracy"] == 1.0
