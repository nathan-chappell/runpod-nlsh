import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

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


def load_runpod_bootstrap_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("runpod_bootstrap", BOOTSTRAP_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_requirements(path: str) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


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
    assert "--min-context-length" in result.stdout
    assert "--context-length" in result.stdout
    assert "--mem-fraction-static" in result.stdout


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
    assert "requirements/pod-sglang.txt" in dockerfile
    assert "build-essential" in dockerfile
    assert "iproute2" in dockerfile
    assert "openssh-client" in dockerfile
    assert "CC=/usr/bin/gcc" in dockerfile
    assert "WORKDIR /opt/nlsh" in dockerfile
    assert "HF_HOME=/workspace/hf-cache" in dockerfile
    assert "SGLANG_STORAGE_PATH=/workspace/sglang-storage" in dockerfile
    assert "TMPDIR=/workspace/tmp" in dockerfile
    assert "TRITON_CACHE_DIR=/workspace/triton-cache" in dockerfile
    assert "mkdir -p /workspace/hf-cache /workspace/sglang-storage /workspace/tmp" in dockerfile
    assert "chmod 1777 /workspace/tmp" in dockerfile
    assert "POD_EVAL_IMAGE_VENV" not in dockerfile
    assert "POD_EVAL_VENV" not in dockerfile
    assert "COPY requirements ./requirements" in dockerfile
    assert dockerfile.index("COPY requirements ./requirements") < dockerfile.index("COPY src ./src")
    assert "PIP_DISABLE_PIP_VERSION_CHECK=1" in dockerfile
    assert "PIP_NO_CACHE_DIR=1" not in dockerfile
    assert "python -m venv" not in dockerfile
    assert "--mount=type=cache,target=/root/.cache/pip" in dockerfile
    assert "python -m pip install --upgrade pip setuptools wheel" in dockerfile
    assert "requirements/pod-core.txt" in dockerfile
    assert "requirements/pod-train.txt" in dockerfile
    assert "requirements/pod-sglang.txt" in dockerfile
    assert "python -m pip install --no-deps ." in dockerfile
    assert dockerfile.index("COPY src ./src") < dockerfile.index("python -m pip install --no-deps .")
    assert dockerfile.index("python -m pip install --no-deps .") < dockerfile.index("COPY data ./data")
    assert "python -m pip install --no-cache-dir -e ." not in dockerfile
    assert "python scripts/pod_eval.py download-models" not in dockerfile
    assert "openssh-server" not in dockerfile
    assert "python3-venv" not in dockerfile
    assert 'CMD ["python", "scripts/runpod_bootstrap.py"]' in dockerfile


def test_runpod_bootstrap_uses_volume_caches_and_base_services() -> None:
    script = BOOTSTRAP_SCRIPT.read_text(encoding="utf-8")

    assert '"SGLANG_STORAGE_PATH": WORKSPACE_DIR / "sglang-storage"' in script
    assert '"TMPDIR": WORKSPACE_DIR / "tmp"' in script
    assert '"TRITON_CACHE_DIR": WORKSPACE_DIR / "triton-cache"' in script
    assert 'os.environ.setdefault("CC", "/usr/bin/gcc")' in script
    assert '_env_bool("POD_EVAL_START_RUNPOD_SERVICES", True)' in script
    assert 'Path("/start.sh")' in script
    assert '"-m", "nlsh.pod_workflow", "run"' in script
    assert "_workflow_command" in script
    assert "_workflow_environment" in script
    assert "os.execvpe(command[0], command, env)" in script
    assert "POD_EVAL_VENV" not in script
    assert "POD_EVAL_IMAGE_VENV" not in script
    assert "BOOTSTRAP_VERSION_FILE" not in script
    assert "site.addsitedir(" not in script
    assert '"-m", "pip", "install"' not in script
    assert '"-m", "venv"' not in script


def test_pod_workflow_defaults_to_priority_order() -> None:
    from nlsh.pod_workflow import DEFAULT_MODEL_ORDER, Workflow, WorkflowConfig, load_manifest

    workflow = Workflow(
        WorkflowConfig(
            artifact_dir=Path("/tmp/nlsh-test-artifacts"),
            selected_models=(),
        ),
        dry_run=True,
    )
    ordered = workflow._ordered_models(load_manifest(Path("configs/pod_eval_models.json")))

    assert [model.id for model in ordered] == list(DEFAULT_MODEL_ORDER)


def test_pod_workflow_defaults_to_phi4_only_selection() -> None:
    from nlsh.pod_workflow import DEFAULT_SELECTED_MODELS, Workflow, WorkflowConfig, load_manifest

    workflow = Workflow(WorkflowConfig(artifact_dir=Path("/tmp/nlsh-test-artifacts")), dry_run=True)
    ordered = workflow._ordered_models(load_manifest(Path("configs/pod_eval_models.json")))

    assert workflow.config.selected_models == DEFAULT_SELECTED_MODELS
    assert [model.id for model in ordered] == list(DEFAULT_SELECTED_MODELS)


def test_pod_workflow_selected_models_filters_manifest() -> None:
    from nlsh.pod_workflow import Workflow, WorkflowConfig, load_manifest

    workflow = Workflow(
        WorkflowConfig(
            artifact_dir=Path("/tmp/nlsh-test-artifacts"),
            selected_models=("microsoft/Phi-4-mini-instruct",),
        ),
        dry_run=True,
    )
    ordered = workflow._ordered_models(load_manifest(Path("configs/pod_eval_models.json")))

    assert [model.id for model in ordered] == ["microsoft/Phi-4-mini-instruct"]


def test_pyproject_declares_training_extra() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    train_deps = pyproject["project"]["optional-dependencies"]["train"]
    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.startswith("huggingface-hub") for dep in dependencies)
    assert any(dep.startswith("typer") for dep in dependencies)
    assert any(dep.startswith("datasets") for dep in train_deps)
    assert any(dep.startswith("hf_transfer") for dep in train_deps)
    assert any(dep.startswith("peft") for dep in train_deps)
    assert any(dep.startswith("transformers") for dep in train_deps)
    assert any(dep.startswith("trl") for dep in train_deps)


def test_pod_requirements_split_runtime_training_and_sglang() -> None:
    assert read_requirements("requirements/pod-core.txt") == [
        "openai==2.32.0",
        "pydantic==2.13.3",
        "pypdf==6.10.2",
        "typer==0.24.2",
        "huggingface-hub==0.36.2",
    ]
    assert read_requirements("requirements/pod-train.txt") == [
        "accelerate==1.13.0",
        "datasets==4.8.4",
        "hf_transfer==0.1.9",
        "peft==0.19.1",
        "transformers==5.3.0",
        "trl==1.2.0",
    ]
    assert read_requirements("requirements/pod-sglang.txt") == [
        "sglang==0.5.10.post1",
        "sglang-kernel==0.4.1",
    ]


def test_pod_workflow_does_not_install_training_dependencies_at_runtime() -> None:
    workflow = Path("src/nlsh/pod_workflow.py").read_text(encoding="utf-8")

    assert "training_modules_available" not in workflow
    assert '"training-deps.log"' not in workflow
    assert '"pip", "install", "-e"' not in workflow


def test_runpod_bootstrap_ensures_workspace_defaults(tmp_path: Path, monkeypatch: Any) -> None:
    module = load_runpod_bootstrap_module()
    workspace_dir = tmp_path / "workspace"
    artifact_dir = workspace_dir / "nlsh-artifacts"
    monkeypatch.setattr(module, "WORKSPACE_DIR", workspace_dir)
    monkeypatch.setattr(module, "ARTIFACT_DIR", artifact_dir)
    for key in ("HF_HOME", "SGLANG_STORAGE_PATH", "TMPDIR", "TRANSFORMERS_CACHE", "TRITON_CACHE_DIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(key, raising=False)

    module._ensure_workspace()

    assert artifact_dir.exists()
    assert (workspace_dir / "hf-cache").exists()
    assert (workspace_dir / "sglang-storage").exists()
    assert (workspace_dir / "tmp").exists()
    assert (workspace_dir / "triton-cache").exists()
    assert (workspace_dir / ".cache").exists()
    assert os.environ["HF_HOME"] == str(workspace_dir / "hf-cache")
    assert os.environ["SGLANG_STORAGE_PATH"] == str(workspace_dir / "sglang-storage")
    assert os.environ["TMPDIR"] == str(workspace_dir / "tmp")
    assert os.environ["TRITON_CACHE_DIR"] == str(workspace_dir / "triton-cache")


def test_runpod_bootstrap_starts_runpod_services_when_available(tmp_path: Path, monkeypatch: Any) -> None:
    module = load_runpod_bootstrap_module()
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(module, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.delenv("RUNPOD_START_BASE_SERVICES", raising=False)
    monkeypatch.setattr(module.os, "access", lambda path, mode: str(path) == "/start.sh")
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    def fake_popen(command: list[str], **kwargs: Any) -> Any:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    module._start_runpod_services()

    assert captured["command"] == ["/start.sh"]
    assert captured["kwargs"]["stderr"] is module.subprocess.STDOUT
    assert (artifact_dir / "runpod-base-services.log").exists()


def test_runpod_bootstrap_main_execs_image_python(tmp_path: Path, monkeypatch: Any) -> None:
    module = load_runpod_bootstrap_module()
    app_dir = tmp_path / "app"
    artifact_dir = tmp_path / "artifacts"
    app_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(module, "APP_DIR", app_dir)
    monkeypatch.setattr(module, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(module, "_ensure_workspace", lambda: captured.setdefault("workspace", True))
    monkeypatch.setattr(module, "_start_runpod_services", lambda: captured.setdefault("services", True))
    monkeypatch.setattr(module.sys, "executable", "/usr/bin/image-python")
    monkeypatch.setattr(module.os, "chdir", lambda path: captured.setdefault("chdir", path))
    monkeypatch.delenv("POD_EVAL_EXIT_AFTER", raising=False)

    def fake_execvpe(executable: str, command: list[str], env: dict[str, str]) -> None:
        captured["executable"] = executable
        captured["command"] = command
        captured["env"] = dict(env)
        raise SystemExit(0)

    monkeypatch.setattr(module.os, "execvpe", fake_execvpe)

    with pytest.raises(SystemExit):
        module.main()

    assert captured["workspace"] is True
    assert captured["services"] is True
    assert captured["chdir"] == app_dir
    assert captured["executable"] == "/usr/bin/image-python"
    assert captured["command"] == ["/usr/bin/image-python", "-m", "nlsh.pod_workflow", "run"]
    assert captured["env"]["POD_EVAL_EXIT_AFTER"] == "1"


def test_runpod_bootstrap_preserves_explicit_exit_after(tmp_path: Path, monkeypatch: Any) -> None:
    module = load_runpod_bootstrap_module()
    app_dir = tmp_path / "app"
    artifact_dir = tmp_path / "artifacts"
    app_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(module, "APP_DIR", app_dir)
    monkeypatch.setattr(module, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(module, "_ensure_workspace", lambda: None)
    monkeypatch.setattr(module, "_start_runpod_services", lambda: None)
    monkeypatch.setattr(module.sys, "executable", "/usr/bin/image-python")
    monkeypatch.setattr(module.os, "chdir", lambda _path: None)
    monkeypatch.setenv("POD_EVAL_EXIT_AFTER", "1")

    def fake_execvpe(executable: str, command: list[str], env: dict[str, str]) -> None:
        captured["executable"] = executable
        captured["command"] = command
        captured["env"] = dict(env)
        raise SystemExit(0)

    monkeypatch.setattr(module.os, "execvpe", fake_execvpe)

    with pytest.raises(SystemExit):
        module.main()

    assert captured["executable"] == "/usr/bin/image-python"
    assert captured["command"] == ["/usr/bin/image-python", "-m", "nlsh.pod_workflow", "run"]
    assert captured["env"]["POD_EVAL_EXIT_AFTER"] == "1"


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


def test_pod_eval_manifest_uses_conservative_sglang_defaults() -> None:
    manifest = json.loads(Path("configs/pod_eval_models.json").read_text(encoding="utf-8"))

    assert manifest["defaults"]["context_length"] == 4096
    assert manifest["defaults"]["max_running_requests"] == 1
    assert manifest["defaults"]["mem_fraction_static"] == 0.85
    assert manifest["defaults"]["sglang_args"] == [
        "--disable-cuda-graph",
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
    ]


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


def test_pod_workflow_defaults_to_materialized_splits_when_available(monkeypatch: Any) -> None:
    from nlsh.pod_workflow import WorkflowConfig

    monkeypatch.setenv("POD_EVAL_APP_DIR", str(Path.cwd()))

    config = WorkflowConfig()

    assert config.dataset == Path.cwd() / "data/splits/v1/test"
    assert config.train_dataset == Path.cwd() / "data/splits/v1/train"
    assert config.train_eval_dataset == Path.cwd() / "data/splits/v1/eval"
    assert config.exit_after is True
    assert config.selected_models == ("microsoft/Phi-4-mini-instruct",)


def test_pod_workflow_runs_eval_then_training_then_post_training_eval(tmp_path: Path, monkeypatch: Any) -> None:
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
        request_model="microsoft/Phi-4-mini-instruct:nlsh-phi4-ft",
        extra_sglang_args=("--enable-lora", "--lora-paths", f"nlsh-phi4-ft={tmp_path / 'adapter'}"),
    )

    command = captured[0]
    assert "--request-model" in command
    assert "microsoft/Phi-4-mini-instruct:nlsh-phi4-ft" in command
    assert "--sglang-arg=--enable-lora" in command
    assert "--sglang-arg=--lora-paths" in command


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
        context_length=4096,
        max_running_requests=4,
        mem_fraction_static=0.85,
        sglang_args=(),
    )

    next_spec, changes = module._reduce_spec_for_oom(
        spec,
        min_context_length=512,
        min_mem_fraction_static=0.55,
    )

    assert next_spec is not None
    assert next_spec.max_running_requests == 2
    assert next_spec.context_length == 2048
    assert next_spec.mem_fraction_static == 0.8
    assert changes == {
        "max_running_requests": {"from": 4, "to": 2},
        "context_length": {"from": 4096, "to": 2048},
        "mem_fraction_static": {"from": 0.85, "to": 0.8},
    }
