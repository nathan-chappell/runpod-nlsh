import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any


BOOTSTRAP_SCRIPT = Path("scripts/runpod_bootstrap.py")


def load_pod_eval_module() -> ModuleType:
    path = Path("scripts/pod_eval.py")
    spec = importlib.util.spec_from_file_location("pod_eval", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    assert "--request-model" in subprocess.run(
        [sys.executable, "scripts/pod_eval.py", "run-model", "--help"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout


def test_pod_eval_run_model_help_includes_oom_controls() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/pod_eval.py", "run-model", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--oom-retries" in result.stdout
    assert "--min-max-model-len" in result.stdout
    assert "--max-model-len" in result.stdout
    assert "--gpu-memory-utilization" in result.stdout


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
        "HuggingFaceTB/SmolLM3-3B",
        "Qwen/Qwen3-8B",
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
    assert 'CMD ["python", "scripts/runpod_bootstrap.py"]' in dockerfile


def test_runpod_bootstrap_uses_volume_caches_and_base_services() -> None:
    script = BOOTSTRAP_SCRIPT.read_text(encoding="utf-8")

    assert '"TMPDIR": WORKSPACE_DIR / "tmp"' in script
    assert '"TRITON_CACHE_DIR": WORKSPACE_DIR / "triton-cache"' in script
    assert '"VLLM_CACHE_ROOT": WORKSPACE_DIR / "vllm-cache"' in script
    assert 'os.environ.setdefault("CC", "/usr/bin/gcc")' in script
    assert '_env_bool("POD_EVAL_START_RUNPOD_SERVICES", True)' in script
    assert 'Path("/start.sh")' in script
    assert '"-m", "nlsh.pod_workflow", "run"' in script


def test_pod_workflow_defaults_to_priority_order() -> None:
    from nlsh.pod_workflow import DEFAULT_MODEL_ORDER, Workflow, WorkflowConfig, load_manifest

    workflow = Workflow(WorkflowConfig(artifact_dir=Path("/tmp/nlsh-test-artifacts")), dry_run=True)
    ordered = workflow._ordered_models(load_manifest(Path("configs/pod_eval_models.json")))

    assert [model.id for model in ordered] == list(DEFAULT_MODEL_ORDER)


def test_pyproject_declares_training_extra() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    train_deps = pyproject["project"]["optional-dependencies"]["train"]
    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.startswith("typer") for dep in dependencies)
    assert any(dep.startswith("datasets") for dep in train_deps)
    assert any(dep.startswith("peft") for dep in train_deps)
    assert any(dep.startswith("transformers") for dep in train_deps)
    assert any(dep.startswith("trl") for dep in train_deps)


def test_runpod_startup_python_syntax() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(BOOTSTRAP_SCRIPT),
            "src/nlsh/pod_workflow.py",
            "scripts/pod_eval.py",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_pod_eval_manifest_uses_conservative_vllm_args() -> None:
    manifest = json.loads(Path("configs/pod_eval_models.json").read_text(encoding="utf-8"))

    assert "--enforce-eager" in manifest["defaults"]["vllm_args"]
    assert "--disable-custom-all-reduce" in manifest["defaults"]["vllm_args"]


def test_pod_workflow_requires_hf_token(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env.update(
        {
            "POD_EVAL_APP_DIR": str(Path.cwd()),
            "POD_EVAL_WORKSPACE": str(tmp_path),
            "POD_EVAL_EXIT_AFTER": "1",
            "POD_EVAL_SKIP_DOWNLOADS": "1",
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "nlsh.pod_workflow", "run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 2
    assert "HF_TOKEN is required" in result.stdout
    assert (tmp_path / "nlsh-artifacts" / "last_exit_code").read_text().strip() == "2"


def test_pod_workflow_dry_run(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HF_TOKEN": "dummy",
            "POD_EVAL_APP_DIR": str(Path.cwd()),
            "POD_EVAL_WORKSPACE": str(tmp_path),
            "POD_EVAL_EXIT_AFTER": "1",
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "nlsh.pod_workflow", "run", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "planned model order" in result.stdout
    state = json.loads((tmp_path / "nlsh-artifacts" / "workflow_state.json").read_text())
    assert state["dry_run"] is True
    assert (tmp_path / "nlsh-artifacts" / "last_exit_code").read_text().strip() == "0"


def test_pod_workflow_runs_eval_then_training_then_post_training_eval(tmp_path: Path, monkeypatch: Any) -> None:
    from nlsh import pod_workflow
    from nlsh.pod_workflow import Workflow, WorkflowConfig

    order: list[str] = []

    monkeypatch.setenv("HF_TOKEN", "dummy")

    def fake_start_downloads(self: Any, models: list[Any]) -> dict[str, Any]:
        assert models
        order.append("downloads")
        return {}

    def fake_run_baseline(self: Any, models: list[Any], futures: dict[str, Any]) -> int:
        assert models
        assert futures == {}
        order.append("baseline_eval")
        return 1

    def fake_run_training(self: Any) -> int:
        order.append("training")
        return 0

    def fake_run_post_training_eval(self: Any, training_status: int) -> int:
        assert training_status == 0
        order.append("post_training_eval")
        return 0

    def fake_finish(self: Any, status: int) -> None:
        order.append(f"finish:{status}")

    monkeypatch.setattr(Workflow, "_start_downloads", fake_start_downloads)
    monkeypatch.setattr(Workflow, "_run_priority_gated_baseline_eval", fake_run_baseline)
    monkeypatch.setattr(Workflow, "_run_training", fake_run_training)
    monkeypatch.setattr(Workflow, "_run_post_training_eval", fake_run_post_training_eval)
    monkeypatch.setattr(Workflow, "_finish", fake_finish)

    workflow = Workflow(
        WorkflowConfig(
            app_dir=Path.cwd(),
            workspace_dir=tmp_path,
            artifact_dir=tmp_path / "artifacts",
            dataset=Path("data/samples"),
            manifest=Path("configs/pod_eval_models.json"),
        ),
        dry_run=False,
    )

    status = workflow.run()

    assert status == 1
    assert order == [
        "downloads",
        "baseline_eval",
        "training",
        "post_training_eval",
        "finish:1",
    ]


def test_pod_workflow_lora_eval_command(tmp_path: Path, monkeypatch: Any) -> None:
    from nlsh import pod_workflow
    from nlsh.pod_workflow import Workflow, WorkflowConfig, WorkflowModel

    captured: list[list[str]] = []

    def fake_run_logged_command(command: list[str], *_args: Any, **_kwargs: Any) -> int:
        captured.append(command)
        return 0

    monkeypatch.setattr(pod_workflow, "run_logged_command", fake_run_logged_command)
    workflow = Workflow(
        WorkflowConfig(
            app_dir=Path.cwd(),
            workspace_dir=tmp_path,
            artifact_dir=tmp_path / "artifacts",
            dataset=Path("data/samples"),
            manifest=Path("configs/pod_eval_models.json"),
            train_output_dir=tmp_path / "adapter",
        ),
        dry_run=False,
    )

    workflow._eval_model(
        WorkflowModel("microsoft/Phi-4-mini-instruct", "Phi-4 Mini Instruct"),
        output_dir=tmp_path / "post",
        log_name="post.log",
        request_model="nlsh-phi4-ft",
        extra_vllm_args=("--enable-lora", "--lora-modules", f"nlsh-phi4-ft={tmp_path / 'adapter'}"),
    )

    command = captured[0]
    assert "--request-model" in command
    assert "nlsh-phi4-ft" in command
    assert "--vllm-arg=--enable-lora" in command
    assert "--vllm-arg=--lora-modules" in command


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
            "data/samples",
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


def test_pod_eval_reduces_spec_for_oom() -> None:
    module = load_pod_eval_module()
    spec = module.ModelSpec(
        id="example/model",
        display_name="Example",
        trust_remote_code=False,
        max_model_len=4096,
        max_num_seqs=4,
        gpu_memory_utilization=0.85,
        generation_config="vllm",
        vllm_args=(),
    )

    next_spec, changes = module._reduce_spec_for_oom(
        spec,
        min_max_model_len=512,
        min_gpu_memory_utilization=0.55,
    )

    assert next_spec is not None
    assert next_spec.max_num_seqs == 2
    assert next_spec.max_model_len == 2048
    assert next_spec.gpu_memory_utilization == 0.8
    assert changes == {
        "max_num_seqs": {"from": 4, "to": 2},
        "max_model_len": {"from": 4096, "to": 2048},
        "gpu_memory_utilization": {"from": 0.85, "to": 0.8},
    }
