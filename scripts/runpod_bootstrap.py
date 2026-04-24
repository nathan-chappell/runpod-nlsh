#!/usr/bin/env python
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

APP_DIR = Path(os.environ.get("POD_EVAL_APP_DIR", "/opt/nlsh"))
WORKSPACE_DIR = Path(os.environ.get("POD_EVAL_WORKSPACE", "/workspace"))
ARTIFACT_DIR = Path(os.environ.get("POD_EVAL_OUTPUT_DIR", WORKSPACE_DIR / "nlsh-artifacts"))


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


def _ensure_workspace() -> None:
    cache_paths = {
        "HF_HOME": WORKSPACE_DIR / "hf-cache",
        "SGLANG_STORAGE_PATH": WORKSPACE_DIR / "sglang-storage",
        "TMPDIR": WORKSPACE_DIR / "tmp",
        "TRANSFORMERS_CACHE": WORKSPACE_DIR / "hf-cache",
        "TRITON_CACHE_DIR": WORKSPACE_DIR / "triton-cache",
        "XDG_CACHE_HOME": WORKSPACE_DIR / ".cache",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for key, default_path in cache_paths.items():
        path = Path(os.environ.setdefault(key, str(default_path)))
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CC", "/usr/bin/gcc")
    os.environ.setdefault("CXX", "/usr/bin/g++")


def _workflow_environment() -> dict[str, str]:
    # Default to exiting after the baseline eval -> training -> post-training eval
    # sequence so serious runs finish once and hand control back to the platform.
    env = dict(os.environ)
    env.setdefault("POD_EVAL_EXIT_AFTER", "1")
    return env


def _boot_mode() -> str:
    return os.environ.get("RUNPOD_BOOT_MODE", "serve").strip().lower()


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


def _workflow_command(*, dry_run: bool) -> list[str]:
    command = [sys.executable, "-m", "nlsh.pod_workflow", "run"]
    if dry_run:
        command.append("--dry-run")
    return command


def _serve_command() -> list[str]:
    bundled_adapter_dir = APP_DIR / "bundled-adapter" / "current"
    command = [
        sys.executable,
        "scripts/pod_eval.py",
        "--manifest",
        str(APP_DIR / "configs/pod_eval_models.json"),
        "serve-lora",
        "--adapter-dir",
        str(bundled_adapter_dir),
        "--host",
        os.environ.get("RUNPOD_SERVE_HOST", "0.0.0.0"),
        "--port",
        os.environ.get("RUNPOD_SERVE_PORT", "8000"),
    ]
    model_id = os.environ.get("RUNPOD_SERVE_MODEL_ID")
    adapter_name = os.environ.get("RUNPOD_SERVE_ADAPTER_NAME")
    if model_id:
        command.extend(["--model", model_id])
    if adapter_name:
        command.extend(["--adapter-name", adapter_name])
    for arg in shlex.split(os.environ.get("RUNPOD_SERVE_SGLANG_ARGS", "")):
        command.append(f"--sglang-arg={arg}")
    return command


def main() -> int:
    _ensure_workspace()
    _log("nlsh Runpod bootstrap starting")
    _log(f"app_dir={APP_DIR}")
    _log(f"workspace_dir={WORKSPACE_DIR}")
    _log(f"artifact_dir={ARTIFACT_DIR}")
    _start_runpod_services()
    mode = _boot_mode()
    if mode == "workflow":
        command = _workflow_command(dry_run=_env_bool("POD_EVAL_DRY_RUN", False))
        env = _workflow_environment()
    elif mode == "serve":
        command = _serve_command()
        env = dict(os.environ)
    else:
        raise SystemExit(f"Unsupported RUNPOD_BOOT_MODE={mode!r}; expected 'serve' or 'workflow'")
    _log(f"boot_mode={mode}")
    _log("handoff to runtime command")
    _log("+ " + " ".join(command))
    os.chdir(APP_DIR)
    os.execvpe(command[0], command, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
