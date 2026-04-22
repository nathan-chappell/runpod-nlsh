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


def test_pod_eval_dockerfile_uses_runpod_base() -> None:
    dockerfile = Path("Dockerfile.pod-eval").read_text(encoding="utf-8")

    assert "ARG RUNPOD_BASE_IMAGE=runpod/pytorch:" in dockerfile
    assert "FROM ${RUNPOD_BASE_IMAGE}" in dockerfile
    assert "vllm/vllm-openai" not in dockerfile
    assert "build-essential" in dockerfile
    assert "iproute2" in dockerfile
    assert "openssh-server" in dockerfile
    assert "python3-venv" in dockerfile
    assert "CC=/usr/bin/gcc" in dockerfile
    assert "WORKDIR /opt/nlsh" in dockerfile
    assert "HF_HOME=/workspace/hf-cache" in dockerfile
    assert "TMPDIR=/workspace/tmp" in dockerfile
    assert "TRITON_CACHE_DIR=/workspace/triton-cache" in dockerfile
    assert "mkdir -p /workspace/hf-cache /workspace/tmp" in dockerfile
    assert "chmod 1777 /workspace/tmp" in dockerfile
    assert "POD_EVAL_VENV=/workspace/nlsh-venv" in dockerfile
    assert "python scripts/pod_eval.py download-models" not in dockerfile
    assert 'CMD ["bash", "scripts/runpod_pod_eval.sh"]' in dockerfile


def test_runpod_startup_script_uses_volume_caches_and_base_services() -> None:
    script = Path("scripts/runpod_pod_eval.sh").read_text(encoding="utf-8")

    assert 'export TMPDIR="${TMPDIR:-${WORKSPACE_DIR}/tmp}"' in script
    assert 'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${WORKSPACE_DIR}/triton-cache}"' in script
    assert 'export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${WORKSPACE_DIR}/vllm-cache}"' in script
    assert 'export CC="${CC:-/usr/bin/gcc}"' in script
    assert '&& -x /start.sh' in script
    assert "/start.sh &" in script


def test_runpod_startup_script_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", "scripts/runpod_pod_eval.sh"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_pod_eval_manifest_uses_conservative_vllm_args() -> None:
    manifest = json.loads(Path("configs/pod_eval_models.json").read_text(encoding="utf-8"))

    assert "--enforce-eager" in manifest["defaults"]["vllm_args"]
    assert "--disable-custom-all-reduce" in manifest["defaults"]["vllm_args"]


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
