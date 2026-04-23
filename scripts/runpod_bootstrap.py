#!/usr/bin/env python
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

APP_DIR = Path(os.environ.get("POD_EVAL_APP_DIR", "/opt/nlsh"))
WORKSPACE_DIR = Path(os.environ.get("POD_EVAL_WORKSPACE", "/workspace"))
ARTIFACT_DIR = Path(os.environ.get("POD_EVAL_OUTPUT_DIR", WORKSPACE_DIR / "nlsh-artifacts"))
VENV_DIR = Path(os.environ.get("POD_EVAL_VENV", WORKSPACE_DIR / "nlsh-venv"))
IMAGE_VENV_DIR = Path(os.environ.get("POD_EVAL_IMAGE_VENV", "/opt/nlsh-image-venv"))
BOOTSTRAP_PYTHON = os.environ.get("POD_EVAL_BOOTSTRAP_PYTHON", sys.executable)
BOOTSTRAP_STATE_VERSION = "3"
BOOTSTRAP_VERSION_FILE = WORKSPACE_DIR / ".nlsh-bootstrap-version"
SOURCE_PTH_FILENAME = "00-nlsh-src.pth"
IMAGE_DEPS_PTH_FILENAME = "01-nlsh-image-deps.pth"


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _log(message: str = "") -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}" if message else ""
    print(line, flush=True)
    with (ARTIFACT_DIR / "startup.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    _log("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _python_version_dir() -> str:
    return f"python{sys.version_info.major}.{sys.version_info.minor}"


def _site_packages_dir(venv_dir: Path) -> Path:
    return venv_dir / "lib" / _python_version_dir() / "site-packages"


def _write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _image_dependency_paths() -> list[Path]:
    image_site_packages = _site_packages_dir(IMAGE_VENV_DIR)
    image_python = IMAGE_VENV_DIR / "bin/python"
    paths = [image_site_packages]

    if not image_python.exists():
        return paths

    try:
        result = subprocess.run(
            [
                str(image_python),
                "-c",
                "import json, site; print(json.dumps(site.getsitepackages()))",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        discovered_paths = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return paths

    for raw_path in discovered_paths:
        candidate = Path(raw_path)
        if candidate.exists() and candidate not in paths:
            paths.append(candidate)
    return paths


def _ensure_workspace() -> None:
    cache_paths = {
        "HF_HOME": WORKSPACE_DIR / "hf-cache",
        "TMPDIR": WORKSPACE_DIR / "tmp",
        "TRANSFORMERS_CACHE": WORKSPACE_DIR / "hf-cache",
        "TRITON_CACHE_DIR": WORKSPACE_DIR / "triton-cache",
        "VLLM_CACHE_ROOT": WORKSPACE_DIR / "vllm-cache",
        "XDG_CACHE_HOME": WORKSPACE_DIR / ".cache",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for key, default_path in cache_paths.items():
        path = Path(os.environ.setdefault(key, str(default_path)))
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NLSH_VLLM_LIKE", "1")
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("CC", "/usr/bin/gcc")
    os.environ.setdefault("CXX", "/usr/bin/g++")


def _start_runpod_services() -> None:
    should_start = _env_bool(
        "RUNPOD_START_BASE_SERVICES",
        _env_bool("POD_EVAL_START_RUNPOD_SERVICES", True),
    )
    if not should_start:
        _log("Runpod base services disabled by RUNPOD_START_BASE_SERVICES=0")
        return
    start_script = Path("/start.sh")
    if not os.access(start_script, os.X_OK):
        _log("/start.sh is not present or executable; continuing without base services")
        return
    log_path = ARTIFACT_DIR / "runpod-base-services.log"
    _log(f"starting Runpod base services with {start_script}; log={log_path}")
    log_file = log_path.open("a", encoding="utf-8")
    subprocess.Popen(
        [str(start_script)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(float(os.environ.get("POD_EVAL_RUNPOD_SERVICE_DELAY", "2")))


def _sync_runtime_paths() -> None:
    source_dir = APP_DIR / "src"
    image_site_packages = _site_packages_dir(IMAGE_VENV_DIR)
    workspace_site_packages = _site_packages_dir(VENV_DIR)

    if not source_dir.exists():
        raise RuntimeError(f"expected source tree at {source_dir}")
    if not image_site_packages.exists():
        raise RuntimeError(f"expected image site-packages at {image_site_packages}")

    _write_text(workspace_site_packages / SOURCE_PTH_FILENAME, str(source_dir) + "\n")
    _write_text(
        workspace_site_packages / IMAGE_DEPS_PTH_FILENAME,
        "".join(
            f"import site; site.addsitedir({str(path)!r})\n"
            for path in _image_dependency_paths()
        ),
    )


def _prepare_python() -> Path:
    python_bin = VENV_DIR / "bin/python"
    current_version = BOOTSTRAP_VERSION_FILE.read_text(encoding="utf-8").strip() if BOOTSTRAP_VERSION_FILE.exists() else None
    if current_version != BOOTSTRAP_STATE_VERSION and VENV_DIR.exists():
        _log(
            "resetting persistent Python environment at "
            f"{VENV_DIR} for bootstrap state version {BOOTSTRAP_STATE_VERSION}"
        )
        shutil.rmtree(VENV_DIR)

    if not python_bin.exists():
        _log(f"creating persistent Python environment at {VENV_DIR}")
        _run([BOOTSTRAP_PYTHON, "-m", "venv", str(VENV_DIR)])
    else:
        _log(f"using persistent Python environment at {python_bin}")

    _sync_runtime_paths()
    BOOTSTRAP_VERSION_FILE.write_text(BOOTSTRAP_STATE_VERSION + "\n", encoding="utf-8")

    return python_bin


def main() -> int:
    _ensure_workspace()
    _log("nlsh Runpod bootstrap starting")
    _log(f"app_dir={APP_DIR}")
    _log(f"workspace_dir={WORKSPACE_DIR}")
    _log(f"artifact_dir={ARTIFACT_DIR}")
    _start_runpod_services()
    python_bin = _prepare_python()
    command = [str(python_bin), "-m", "nlsh.pod_workflow", "run"]
    if _env_bool("POD_EVAL_DRY_RUN", False):
        command.append("--dry-run")
    _log("handoff to Typer workflow")
    _log("+ " + " ".join(command))
    os.chdir(APP_DIR)
    os.execvpe(str(python_bin), command, os.environ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
