from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

DEFAULT_MODEL_ORDER = (
    "microsoft/Phi-4-mini-instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "Qwen/Qwen3-8B",
)
DEFAULT_SELECTED_MODELS = (
    "microsoft/Phi-4-mini-instruct",
)
DEFAULT_API_KEY = "EMPTY"
POST_TRAINING_ADAPTER_NAME = "nlsh-phi4-ft"

app = typer.Typer(help="Runpod eval, fine-tune, and post-training eval workflow.")


@app.callback()
def main() -> None:
    """Runpod eval, fine-tune, and post-training eval workflow."""


def _default_app_dir() -> Path:
    return Path(os.environ.get("POD_EVAL_APP_DIR", Path.cwd()))


def _default_split_dir() -> Path:
    return _default_app_dir() / "data" / "splits" / "v1"


def _default_workspace_dir() -> Path:
    raw_value = os.environ.get("POD_EVAL_WORKSPACE")
    if raw_value:
        return Path(raw_value)
    workspace = Path("/workspace")
    if workspace.exists():
        return workspace
    return Path("artifacts/pod-workspace")


def _split_path(name: str) -> Path:
    return _default_split_dir() / name


def _default_eval_dataset() -> Path:
    raw_value = os.environ.get("POD_EVAL_DATASET")
    if raw_value:
        return Path(raw_value)
    split_path = _split_path("test")
    if split_path.exists():
        return split_path
    return _default_app_dir() / "data/samples"


def _default_train_dataset() -> Path | None:
    raw_value = os.environ.get("POD_EVAL_TRAIN_DATASET")
    if raw_value:
        return Path(raw_value)
    split_path = _split_path("train")
    return split_path if split_path.exists() else None


def _default_train_eval_dataset() -> Path | None:
    raw_value = os.environ.get("POD_EVAL_TRAIN_EVAL_DATASET")
    if raw_value:
        return Path(raw_value)
    split_path = _split_path("eval")
    return split_path if split_path.exists() else None


@dataclass(frozen=True, slots=True)
class WorkflowModel:
    id: str
    display_name: str

    @property
    def slug(self) -> str:
        slug = "".join(char if char.isalnum() or char in "_.-" else "_" for char in self.id).strip("_")
        return slug or "model"


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    app_dir: Path = field(default_factory=_default_app_dir)
    workspace_dir: Path = field(default_factory=_default_workspace_dir)
    artifact_dir: Path = field(default_factory=lambda: Path(
        os.environ.get(
            "POD_EVAL_OUTPUT_DIR",
            _default_workspace_dir() / "nlsh-artifacts",
        )
    ))
    dataset: Path = field(default_factory=_default_eval_dataset)
    manifest: Path = field(default_factory=lambda: Path(
        os.environ.get(
            "POD_EVAL_MANIFEST",
            _default_app_dir() / "configs/pod_eval_models.json",
        )
    ))
    training_dataset: Path = field(default_factory=lambda: Path(
        os.environ.get(
            "POD_EVAL_TRAINING_DATASET",
            _default_app_dir() / "data/samples",
        )
    ))
    train_dataset: Path | None = field(default_factory=_default_train_dataset)
    train_eval_dataset: Path | None = field(default_factory=_default_train_eval_dataset)
    train_output_dir: Path = field(default_factory=lambda: Path(
        os.environ.get(
            "POD_EVAL_TRAIN_OUTPUT_DIR",
            _default_workspace_dir() / "nlsh-finetune/phi-4-mini-instruct-lora",
        )
    ))
    train_model_id: str = os.environ.get("POD_EVAL_TRAIN_MODEL_ID", "microsoft/Phi-4-mini-instruct")
    limit: int | None = field(default_factory=lambda: _optional_int("POD_EVAL_LIMIT"))
    timeout: float = field(default_factory=lambda: float(os.environ.get("POD_EVAL_TIMEOUT", "90")))
    startup_timeout: float = field(default_factory=lambda: float(os.environ.get("POD_EVAL_STARTUP_TIMEOUT", "900")))
    download_workers: int = field(default_factory=lambda: int(os.environ.get("POD_EVAL_DOWNLOAD_WORKERS", "3")))
    run_baseline_eval: bool = field(default_factory=lambda: _env_bool("POD_EVAL_RUN_BASELINE_EVAL", True))
    run_training: bool = field(default_factory=lambda: _env_bool("POD_EVAL_RUN_TRAINING", True))
    train_no_eval: bool = field(default_factory=lambda: _env_bool("POD_EVAL_TRAIN_NO_EVAL", False))
    train_dry_run: bool = field(default_factory=lambda: _env_bool("POD_EVAL_TRAIN_DRY_RUN", False))
    skip_downloads: bool = field(default_factory=lambda: _env_bool("POD_EVAL_SKIP_DOWNLOADS", False))
    skip_post_training_eval: bool = field(default_factory=lambda: _env_bool("POD_EVAL_SKIP_POST_TRAINING_EVAL", False))
    local_files_only: bool = field(default_factory=lambda: _env_bool("POD_EVAL_LOCAL_FILES_ONLY", False))
    exit_after: bool = field(default_factory=lambda: _env_bool("POD_EVAL_EXIT_AFTER", True))
    api_key: str = field(default_factory=lambda: os.environ.get("POD_EVAL_API_KEY", DEFAULT_API_KEY))
    python_executable: str = field(default_factory=lambda: os.environ.get("POD_EVAL_PYTHON", sys.executable))
    model_order: tuple[str, ...] = field(default_factory=lambda: tuple(
        item.strip()
        for item in os.environ.get("POD_EVAL_MODEL_ORDER", ",".join(DEFAULT_MODEL_ORDER)).split(",")
        if item.strip()
    ))
    selected_models: tuple[str, ...] = field(default_factory=lambda: tuple(
        item.strip()
        for item in os.environ.get("POD_EVAL_SELECTED_MODELS", ",".join(DEFAULT_SELECTED_MODELS)).split(",")
        if item.strip()
    ))
    eval_args: tuple[str, ...] = field(default_factory=lambda: tuple(shlex.split(os.environ.get("POD_EVAL_EVAL_ARGS", ""))))
    sglang_args: tuple[str, ...] = field(default_factory=lambda: tuple(shlex.split(os.environ.get("POD_EVAL_SGLANG_ARGS", ""))))
    train_args: tuple[str, ...] = field(default_factory=lambda: tuple(shlex.split(os.environ.get("POD_EVAL_TRAIN_ARGS", ""))))
    post_training_sglang_args: tuple[str, ...] = field(default_factory=lambda: tuple(
        shlex.split(os.environ.get("POD_EVAL_POST_TRAINING_SGLANG_ARGS", ""))
    ))


class Workflow:
    def __init__(self, config: WorkflowConfig, *, dry_run: bool) -> None:
        self.config = config
        self.dry_run = dry_run
        self.state_lock = threading.Lock()
        self.state: dict[str, Any] = {
            "started_at": _utc_now(),
            "finished_at": None,
            "dry_run": dry_run,
            "config": self._config_payload(),
            "downloads": {},
            "baseline_eval": {},
            "training": None,
            "post_training_eval": None,
            "exit_codes": {},
        }
        self.logger = _setup_logging(config.artifact_dir)
        self.download_executor: ThreadPoolExecutor | None = None

    def run(self) -> int:
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._write_state()
        self._log_configuration()

        if not self.dry_run and not os.environ.get("HF_TOKEN"):
            self.logger.error("HF_TOKEN is required. Set it from a Runpod secret before launching the pod.")
            self._write_exit_code("last_exit_code", 2)
            self._finish(2)
            return 2

        models = self._ordered_models(load_manifest(self.config.manifest))
        if self.dry_run:
            self.logger.info("dry run: planned model order=%s", [model.id for model in models])
            self._finish(0)
            return 0

        futures = self._start_downloads(models)
        eval_status = self._run_priority_gated_baseline_eval(models, futures)
        training_status = self._run_training()
        post_training_status = self._run_post_training_eval(training_status)
        status = first_nonzero(eval_status, training_status, post_training_status)
        self._finish(status)
        return status

    def _config_payload(self) -> dict[str, Any]:
        return {
            "app_dir": str(self.config.app_dir),
            "workspace_dir": str(self.config.workspace_dir),
            "artifact_dir": str(self.config.artifact_dir),
            "dataset": str(self.config.dataset),
            "manifest": str(self.config.manifest),
            "training_dataset": str(self.config.training_dataset),
            "train_dataset": None if self.config.train_dataset is None else str(self.config.train_dataset),
            "train_eval_dataset": None if self.config.train_eval_dataset is None else str(self.config.train_eval_dataset),
            "train_output_dir": str(self.config.train_output_dir),
            "train_model_id": self.config.train_model_id,
            "limit": self.config.limit,
            "timeout": self.config.timeout,
            "startup_timeout": self.config.startup_timeout,
            "download_workers": self.config.download_workers,
            "run_baseline_eval": self.config.run_baseline_eval,
            "run_training": self.config.run_training,
            "train_no_eval": self.config.train_no_eval,
            "train_dry_run": self.config.train_dry_run,
            "skip_downloads": self.config.skip_downloads,
            "skip_post_training_eval": self.config.skip_post_training_eval,
            "local_files_only": self.config.local_files_only,
            "exit_after": self.config.exit_after,
            "model_order": list(self.config.model_order),
            "selected_models": list(self.config.selected_models),
            "eval_args": list(self.config.eval_args),
            "sglang_args": list(self.config.sglang_args),
            "train_args": list(self.config.train_args),
            "post_training_sglang_args": list(self.config.post_training_sglang_args),
        }

    def _log_configuration(self) -> None:
        self.logger.info("nlsh Runpod Typer workflow starting")
        for key, value in self._config_payload().items():
            self.logger.info("config.%s=%s", key, value)
        self.logger.info("execution plan: parallel downloads -> priority-gated eval -> Phi-4 fine-tune -> SGLang adapter eval")

    def _write_state(self) -> None:
        with self.state_lock:
            _write_json(self.config.artifact_dir / "workflow_state.json", self.state)

    def _write_exit_code(self, name: str, value: int) -> None:
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.config.artifact_dir / name).write_text(f"{value}\n", encoding="utf-8")
        self.state["exit_codes"][name] = value
        self._write_state()

    def _finish(self, status: int) -> None:
        self.state["finished_at"] = _utc_now()
        self._write_exit_code("last_exit_code", status)
        self._write_state()
        self.logger.info("workflow finished with exit code %s", status)
        if not self.config.exit_after:
            self.logger.info("keeping container alive for inspection; set POD_EVAL_EXIT_AFTER=1 to exit")
            while True:
                time.sleep(3600)

    def _ordered_models(self, models: list[WorkflowModel]) -> list[WorkflowModel]:
        by_id = {model.id: model for model in models}
        if self.config.selected_models:
            missing = [model_id for model_id in self.config.selected_models if model_id not in by_id]
            if missing:
                missing_list = ", ".join(missing)
                raise ValueError(f"selected model ids not found in manifest: {missing_list}")
            return [by_id[model_id] for model_id in self.config.selected_models]

        ordered: list[WorkflowModel] = []
        for model_id in self.config.model_order:
            model = by_id.get(model_id)
            if model is not None:
                ordered.append(model)
        ordered_ids = {model.id for model in ordered}
        ordered.extend(model for model in models if model.id not in ordered_ids)
        return ordered

    def _start_downloads(self, models: list[WorkflowModel]) -> dict[str, Future[DownloadResult]]:
        if self.config.skip_downloads:
            self.logger.info("skipping downloads because POD_EVAL_SKIP_DOWNLOADS=1")
            return {}
        self.logger.info("starting %s model downloads with %s workers", len(models), self.config.download_workers)
        executor = ThreadPoolExecutor(max_workers=max(self.config.download_workers, 1), thread_name_prefix="download")
        self.download_executor = executor
        return {
            model.id: executor.submit(self._download_model, model)
            for model in models
        }

    def _download_model(self, model: WorkflowModel) -> "DownloadResult":
        log_path = self.config.artifact_dir / "downloads" / f"{model.slug}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_file:
            _emit(log_file, f"downloading {model.id}")
            _emit(log_file, f"hf_home={os.environ.get('HF_HOME')}")
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=model.id,
                    token=os.environ.get("HF_TOKEN") or None,
                    local_files_only=self.config.local_files_only,
                )
                duration = time.perf_counter() - started
                _emit(log_file, f"download ready in {duration:.3f}s")
                result = DownloadResult(model=model, status="ready", duration=duration, log_path=log_path)
            except Exception as exc:
                duration = time.perf_counter() - started
                _emit(log_file, f"download failed after {duration:.3f}s: {type(exc).__name__}: {exc}")
                result = DownloadResult(
                    model=model,
                    status="failed",
                    duration=duration,
                    log_path=log_path,
                    error=f"{type(exc).__name__}: {exc}",
                )
        self.state["downloads"][model.id] = result.to_state()
        self._write_state()
        self.logger.info("download %s: %s", model.id, result.status)
        return result

    def _run_priority_gated_baseline_eval(
        self,
        models: list[WorkflowModel],
        futures: dict[str, Future["DownloadResult"]],
    ) -> int:
        if not self.config.run_baseline_eval:
            self.logger.info("skipping baseline eval because POD_EVAL_RUN_BASELINE_EVAL=0")
            for model in models:
                self._wait_for_download(model, futures)
            if self.download_executor is not None:
                self.download_executor.shutdown(wait=True)
                self.download_executor = None
            self._write_exit_code("last_eval_exit_code", 0)
            return 0

        aggregate_status = 0
        for model in models:
            download_result = self._wait_for_download(model, futures)
            if download_result.status == "failed":
                aggregate_status = 1
                self.state["baseline_eval"][model.id] = {
                    "status": "skipped_download_failed",
                    "download_error": download_result.error,
                }
                self._write_state()
                continue
            status = self._eval_model(model, output_dir=self.config.artifact_dir, log_name=f"{model.slug}.eval-command.log")
            if status != 0:
                aggregate_status = 1
            self.state["baseline_eval"][model.id] = {
                "status": "completed" if status == 0 else "failed",
                "exit_code": status,
            }
            self._write_state()

        self._write_exit_code("last_eval_exit_code", aggregate_status)
        if self.download_executor is not None:
            self.download_executor.shutdown(wait=True)
            self.download_executor = None
        return aggregate_status

    def _wait_for_download(
        self,
        model: WorkflowModel,
        futures: dict[str, Future["DownloadResult"]],
    ) -> "DownloadResult":
        if self.config.skip_downloads:
            return DownloadResult(model=model, status="ready", duration=0.0, log_path=None)
        self.logger.info("waiting for preferred model download: %s", model.id)
        return futures[model.id].result()

    def _eval_model(
        self,
        model: WorkflowModel,
        *,
        output_dir: Path,
        log_name: str,
        request_model: str | None = None,
        extra_sglang_args: tuple[str, ...] = (),
    ) -> int:
        command = [
            sys.executable,
            "scripts/pod_eval.py",
            "--manifest",
            str(self.config.manifest),
            "run-model",
            "--model",
            model.id,
            "--dataset",
            str(self.config.dataset),
            "--output-dir",
            str(output_dir),
            "--timeout",
            str(self.config.timeout),
            "--startup-timeout",
            str(self.config.startup_timeout),
            "--python-executable",
            self.config.python_executable,
        ]
        if self.config.limit is not None:
            command.extend(["--limit", str(self.config.limit)])
        if request_model is not None:
            command.extend(["--request-model", request_model])
        command.extend(self.config.eval_args)
        for arg in (*self.config.sglang_args, *extra_sglang_args):
            command.append(f"--sglang-arg={arg}")
        return run_logged_command(command, self.config.artifact_dir / log_name, self.logger, cwd=self.config.app_dir)

    def _run_training(self) -> int:
        if not self.config.run_training:
            self.logger.info("skipping fine-tuning because POD_EVAL_RUN_TRAINING=0")
            self._write_exit_code("last_training_exit_code", 0)
            self.state["training"] = {"status": "skipped"}
            self._write_state()
            return 0

        command = [
            sys.executable,
            "scripts/phi_4_training.py",
            "--model-id",
            self.config.train_model_id,
            "--dataset",
            str(self.config.training_dataset),
            "--output-dir",
            str(self.config.train_output_dir),
            "--workspace",
            str(self.config.workspace_dir),
            "--no-trust-remote-code",
        ]
        if self.config.train_dataset is not None:
            command.extend(["--train-dataset", str(self.config.train_dataset)])
        if self.config.train_eval_dataset is not None:
            command.extend(["--eval-dataset", str(self.config.train_eval_dataset)])
        if self.config.train_no_eval:
            command.append("--no-eval")
        if self.config.train_dry_run:
            command.append("--dry-run")
        command.extend(self.config.train_args)
        status = run_logged_command(command, self.config.artifact_dir / "training.log", self.logger, cwd=self.config.app_dir)

        self._write_exit_code("last_training_exit_code", status)
        self.state["training"] = {
            "status": "completed" if status == 0 else "failed",
            "exit_code": status,
            "output_dir": str(self.config.train_output_dir),
        }
        self._write_state()
        return status

    def _run_post_training_eval(self, training_status: int) -> int:
        if not self.config.run_training or self.config.skip_post_training_eval:
            self.logger.info("skipping post-training eval")
            self._write_exit_code("last_post_training_eval_exit_code", 0)
            return 0
        if training_status != 0:
            self.logger.warning("skipping post-training eval because training failed")
            self._write_exit_code("last_post_training_eval_exit_code", 1)
            return 1
        if self.config.train_dry_run:
            self.logger.info("skipping post-training eval because training was a dry run")
            self._write_exit_code("last_post_training_eval_exit_code", 0)
            return 0

        models = load_manifest(self.config.manifest)
        by_id = {model.id: model for model in models}
        base_model = by_id.get(self.config.train_model_id)
        if base_model is None:
            self.logger.error("training model %s is not in manifest", self.config.train_model_id)
            self._write_exit_code("last_post_training_eval_exit_code", 1)
            return 1

        lora_arg = f"{POST_TRAINING_ADAPTER_NAME}={self.config.train_output_dir}"
        status = self._eval_model(
            base_model,
            output_dir=self.config.artifact_dir / "post-training-eval",
            log_name="post_training_eval.log",
            request_model=f"{base_model.id}:{POST_TRAINING_ADAPTER_NAME}",
            extra_sglang_args=(
                "--enable-lora",
                "--lora-paths",
                lora_arg,
                *self.config.post_training_sglang_args,
            ),
        )
        self._write_post_training_summary(base_model, status)
        self._write_exit_code("last_post_training_eval_exit_code", status)
        return status

    def _write_post_training_summary(self, base_model: WorkflowModel, status: int) -> None:
        baseline_report = self.config.artifact_dir / base_model.slug / "report.json"
        adapter_report = self.config.artifact_dir / "post-training-eval" / base_model.slug / "report.json"
        baseline = _read_json_or_none(baseline_report)
        adapter = _read_json_or_none(adapter_report)
        summary = {
            "status": "completed" if status == 0 else "failed",
            "base_model": base_model.id,
            "adapter_model": f"{base_model.id}:{POST_TRAINING_ADAPTER_NAME}",
            "adapter_path": str(self.config.train_output_dir),
            "baseline_report": str(baseline_report),
            "adapter_report": str(adapter_report),
            "baseline": _compact_report(baseline),
            "fine_tuned": _compact_report(adapter),
            "delta": _metric_delta(baseline, adapter),
        }
        _write_json(self.config.artifact_dir / "post_training_summary.json", summary)
        self.state["post_training_eval"] = summary
        self._write_state()


@dataclass(frozen=True, slots=True)
class DownloadResult:
    model: WorkflowModel
    status: str
    duration: float
    log_path: Path | None
    error: str | None = None

    def to_state(self) -> dict[str, Any]:
        return {
            "model": self.model.id,
            "status": self.status,
            "duration_seconds": self.duration,
            "log_path": None if self.log_path is None else str(self.log_path),
            "error": self.error,
        }


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_int(name: str) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return None
    return int(raw_value)


def _optional_path(name: str) -> Path | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return None
    return Path(raw_value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _setup_logging(artifact_dir: Path) -> logging.Logger:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nlsh.pod_workflow")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(artifact_dir / "workflow.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _emit(handle: Any, text: str) -> None:
    handle.write(text + "\n")
    handle.flush()


def load_manifest(path: Path) -> list[WorkflowModel]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list) or not models:
        raise ValueError(f"{path} must contain a non-empty models list")
    parsed: list[WorkflowModel] = []
    for index, item in enumerate(models, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"{path} models[{index}] must contain an id")
        parsed.append(WorkflowModel(
            id=item["id"],
            display_name=item.get("display_name") or item["id"],
        ))
    return parsed


def run_logged_command(command: list[str], log_path: Path, logger: logging.Logger, *, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("+ %s", " ".join(shlex.quote(part) for part in command))
    with log_path.open("w", encoding="utf-8") as log_file:
        _emit(log_file, "+ " + " ".join(shlex.quote(part) for part in command))
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.rstrip()
            logger.info(stripped)
            _emit(log_file, stripped)
        return process.wait()


def first_nonzero(*values: int) -> int:
    for value in values:
        if value != 0:
            return value
    return 0


def _compact_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "count": report.get("count"),
        "exact_match_rate": report.get("exact_match_rate"),
        "compile_valid_rate": report.get("compile_valid_rate"),
        "slot_accuracy": report.get("slot_accuracy"),
        "latency_seconds": report.get("latency_seconds"),
        "error_counts": report.get("error_counts"),
    }


def _metric_delta(
    baseline: dict[str, Any] | None,
    adapter: dict[str, Any] | None,
) -> dict[str, float] | None:
    if baseline is None or adapter is None:
        return None
    deltas: dict[str, float] = {}
    for key in ("exact_match_rate", "compile_valid_rate", "slot_accuracy"):
        baseline_value = baseline.get(key)
        adapter_value = adapter.get(key)
        if isinstance(baseline_value, int | float) and isinstance(adapter_value, int | float):
            deltas[key] = float(adapter_value) - float(baseline_value)
    return deltas


@app.command()
def run(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print and record the planned workflow without downloads, evals, or training.",
    ),
) -> None:
    workflow = Workflow(WorkflowConfig(), dry_run=dry_run or _env_bool("POD_EVAL_DRY_RUN", False))
    raise typer.Exit(code=workflow.run())


if __name__ == "__main__":
    app()
