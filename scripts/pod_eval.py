#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nlsh.compiler import CompileError, compile_plan
from nlsh.dataio import load_jsonl
from nlsh.planner import GoldPlanner, OpenAIPlanner, Planner, PlannerConfig
from nlsh.schema import PlanV1, normalize_plan, validate_plan_payload


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data/dev.messages.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "configs/pod_eval_models.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/pod_eval"
DEFAULT_API_KEY = "EMPTY"


class VLLMStartupError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    display_name: str
    trust_remote_code: bool
    max_model_len: int
    max_num_seqs: int
    gpu_memory_utilization: float
    generation_config: str
    vllm_args: tuple[str, ...]

    @property
    def slug(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.id).strip("_")
        return slug or "model"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _diff_expected_vs_actual(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    diffs: dict[str, dict[str, Any]] = {}
    for key in sorted(set(expected) | set(actual)):
        expected_value = expected.get(key, "<missing>")
        actual_value = actual.get(key, "<missing>")
        if _normalize(expected_value) != _normalize(actual_value):
            diffs[key] = {"expected": expected_value, "actual": actual_value}
    return diffs


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        items: dict[str, Any] = {}
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            items.update(_flatten(item, next_prefix))
        return items
    if isinstance(value, list):
        items = {}
        for index, item in enumerate(value):
            items.update(_flatten(item, f"{prefix}[{index}]"))
        return items
    return {prefix: value}


def _slot_score(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[int, int]:
    expected_slots = _flatten(expected)
    actual_slots = _flatten(actual)
    keys = set(expected_slots) | set(actual_slots)
    correct = sum(1 for key in keys if expected_slots.get(key) == actual_slots.get(key))
    return correct, len(keys)


def _tail_text(path: Path, line_count: int = 220) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _emit(log_file: Any, text: str = "") -> None:
    print(text, flush=True)
    log_file.write(text + "\n")
    log_file.flush()


def _expect_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be a boolean")


def _expect_int(value: Any, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"{field} must be an integer")


def _expect_float(value: Any, field: str) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"{field} must be a number")


def _expect_string(value: Any, field: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{field} must be a non-empty string")


def _expect_string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of strings")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(value)


def load_manifest(path: Path) -> list[ModelSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    defaults = payload.get("defaults", {})
    models = payload.get("models")
    if not isinstance(defaults, dict):
        raise ValueError("manifest defaults must be an object")
    if not isinstance(models, list) or not models:
        raise ValueError("manifest models must be a non-empty list")

    specs: list[ModelSpec] = []
    seen_ids: set[str] = set()
    for index, raw_model in enumerate(models, start=1):
        if not isinstance(raw_model, dict):
            raise ValueError(f"manifest model #{index} must be an object")
        model_id = _expect_string(raw_model.get("id"), f"models[{index}].id")
        if model_id in seen_ids:
            raise ValueError(f"duplicate model id in manifest: {model_id}")
        seen_ids.add(model_id)

        def value(name: str, fallback: Any) -> Any:
            return raw_model.get(name, defaults.get(name, fallback))

        specs.append(
            ModelSpec(
                id=model_id,
                display_name=_expect_string(
                    raw_model.get("display_name", model_id),
                    f"models[{index}].display_name",
                ),
                trust_remote_code=_expect_bool(
                    value("trust_remote_code", False),
                    f"models[{index}].trust_remote_code",
                ),
                max_model_len=_expect_int(value("max_model_len", 4096), f"models[{index}].max_model_len"),
                max_num_seqs=_expect_int(value("max_num_seqs", 1), f"models[{index}].max_num_seqs"),
                gpu_memory_utilization=_expect_float(
                    value("gpu_memory_utilization", 0.85),
                    f"models[{index}].gpu_memory_utilization",
                ),
                generation_config=_expect_string(
                    value("generation_config", "vllm"),
                    f"models[{index}].generation_config",
                ),
                vllm_args=_expect_string_tuple(value("vllm_args", []), f"models[{index}].vllm_args"),
            )
        )
    return specs


def find_model(specs: list[ModelSpec], selector: str) -> ModelSpec:
    matches = [
        spec for spec in specs
        if selector in {spec.id, spec.display_name, spec.slug}
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        available = ", ".join(spec.id for spec in specs)
        raise ValueError(f"Unknown model {selector!r}. Available models: {available}")
    raise ValueError(f"Ambiguous model selector {selector!r}")


def _vllm_command(
    spec: ModelSpec,
    *,
    host: str,
    port: int,
    extra_args: list[str],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        spec.id,
        "--max-model-len",
        str(spec.max_model_len),
        "--max-num-seqs",
        str(spec.max_num_seqs),
        "--gpu-memory-utilization",
        str(spec.gpu_memory_utilization),
        "--generation-config",
        spec.generation_config,
    ]
    if spec.trust_remote_code:
        command.append("--trust-remote-code")
    command.extend(spec.vllm_args)
    command.extend(extra_args)
    return command


def _connect_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def wait_for_models(
    *,
    base_url: str,
    api_key: str,
    startup_timeout: float,
    process: subprocess.Popen[Any] | None,
) -> dict[str, Any]:
    deadline = time.monotonic() + startup_timeout
    last_error = ""
    url = _models_url(base_url)

    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise VLLMStartupError(
                f"vLLM exited before serving {url} with exit code {process.returncode}"
            )
        try:
            headers = {"Accept": "application/json"}
            if api_key and api_key != DEFAULT_API_KEY:
                headers["Authorization"] = f"Bearer {api_key}"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=5) as response:
                body = response.read().decode("utf-8")
                if 200 <= response.status < 300:
                    return json.loads(body or "{}")
                last_error = f"HTTP {response.status}: {body[:500]}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(2)

    detail = f"; last error: {last_error}" if last_error else ""
    raise VLLMStartupError(f"Timed out after {startup_timeout:.0f}s waiting for {url}{detail}")


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _error_category(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    status_code = getattr(exc, "status_code", None)
    if isinstance(exc, ValidationError) or "validation" in name:
        return "validation"
    if "timeout" in name:
        return "timeout"
    if isinstance(status_code, int):
        if status_code >= 500:
            return "server_error"
        return "api_error"
    if "server" in name or "api" in name or "http" in name:
        return "server_error"
    return "planner_error"


def _update_report_metrics(report: dict[str, Any]) -> None:
    count = report["count"]
    latencies = [
        item["latency_seconds"]
        for item in report["items"]
        if isinstance(item.get("latency_seconds"), int | float)
    ]
    report["exact_match_rate"] = report["exact_matches"] / count if count else 0.0
    report["compile_valid_rate"] = report["compile_valid"] / count if count else 0.0
    report["slot_accuracy"] = (
        report["slot_correct"] / report["slot_total"]
        if report["slot_total"]
        else 0.0
    )
    report["latency_seconds"] = {
        "total": sum(latencies),
        "average": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "min": min(latencies) if latencies else 0.0,
        "max": max(latencies) if latencies else 0.0,
    }


def _prepare_records(dataset_path: Path, limit: int | None) -> list[dict[str, Any]]:
    if limit is not None and limit < 0:
        raise ValueError("--limit must be greater than or equal to 0")
    records = load_jsonl(dataset_path)
    if limit is not None:
        records = records[:limit]
    prepared: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if "prompt" not in record or "plan" not in record:
            raise ValueError(f"{dataset_path} row {index} must contain prompt and plan")
        expected_plan = validate_plan_payload(record["plan"])
        prepared.append({
            "prompt": record["prompt"],
            "expected": normalize_plan(expected_plan),
        })
    return prepared


def _make_planner(
    *,
    planner_name: str,
    dataset_path: Path,
    spec: ModelSpec,
    base_url: str,
    api_key: str,
    request_timeout: float,
) -> Planner:
    if planner_name == "gold":
        return GoldPlanner(dataset_path=dataset_path)
    if planner_name != "openai":
        raise ValueError(f"Unsupported planner: {planner_name}")
    config = PlannerConfig(
        model=spec.id,
        base_url=base_url,
        api_key=api_key,
        request_timeout=request_timeout,
        force_vllm_like=True,
    )
    return OpenAIPlanner(config)


def evaluate_model(
    *,
    spec: ModelSpec,
    planner_name: str,
    dataset_path: Path,
    output_dir: Path,
    limit: int | None,
    base_url: str,
    api_key: str,
    request_timeout: float,
    python_executable: str,
) -> dict[str, Any]:
    records = _prepare_records(dataset_path, limit)
    model_dir = output_dir / spec.slug
    report_path = model_dir / "report.json"
    log_path = model_dir / "eval.log"
    model_dir.mkdir(parents=True, exist_ok=True)
    planner = _make_planner(
        planner_name=planner_name,
        dataset_path=dataset_path,
        spec=spec,
        base_url=base_url,
        api_key=api_key,
        request_timeout=request_timeout,
    )

    report: dict[str, Any] = {
        "started_at": _utc_now(),
        "finished_at": None,
        "dataset": str(dataset_path),
        "limit": limit,
        "planner": planner_name,
        "base_url": base_url,
        "request_timeout": request_timeout,
        "model": {
            "id": spec.id,
            "display_name": spec.display_name,
            "trust_remote_code": spec.trust_remote_code,
            "max_model_len": spec.max_model_len,
            "max_num_seqs": spec.max_num_seqs,
            "gpu_memory_utilization": spec.gpu_memory_utilization,
            "generation_config": spec.generation_config,
            "vllm_args": list(spec.vllm_args),
        },
        "count": len(records),
        "exact_matches": 0,
        "compile_valid": 0,
        "slot_correct": 0,
        "slot_total": 0,
        "exact_match_rate": 0.0,
        "compile_valid_rate": 0.0,
        "slot_accuracy": 0.0,
        "latency_seconds": {
            "total": 0.0,
            "average": 0.0,
            "min": 0.0,
            "max": 0.0,
        },
        "error_counts": {},
        "items": [],
    }

    with log_path.open("w", encoding="utf-8") as log_file:
        _emit(log_file, f"model: {spec.id}")
        _emit(log_file, f"display_name: {spec.display_name}")
        _emit(log_file, f"planner: {planner_name}")
        _emit(log_file, f"base_url: {base_url}")
        _emit(log_file, f"dataset: {dataset_path}")
        _emit(log_file, f"request_timeout: {request_timeout}")
        _emit(log_file, "=" * 100)

        _write_json(report_path, report)
        for index, record in enumerate(records, start=1):
            prompt = record["prompt"]
            expected = record["expected"]
            item: dict[str, Any] = {
                "index": index,
                "prompt": prompt,
                "expected": expected,
                "predicted": None,
                "exact_match": False,
                "compile_valid": False,
                "compile_error": None,
                "error": None,
                "diffs": None,
                "slot_correct": 0,
                "slot_total": 0,
                "latency_seconds": None,
            }

            _emit(log_file, f"[{index}/{len(records)}] PROMPT")
            _emit(log_file, prompt)
            started = time.perf_counter()
            try:
                predicted_plan = planner.plan(prompt)
                predicted = normalize_plan(predicted_plan)
                item["predicted"] = predicted
                item["diffs"] = _diff_expected_vs_actual(expected, predicted)
                item["exact_match"] = not item["diffs"]
                correct_slots, total_slots = _slot_score(expected, predicted)
                item["slot_correct"] = correct_slots
                item["slot_total"] = total_slots

                if isinstance(predicted_plan, PlanV1):
                    try:
                        compile_plan(predicted_plan, python_executable=python_executable)
                        item["compile_valid"] = True
                    except CompileError as exc:
                        item["compile_error"] = str(exc)
                else:
                    item["compile_valid"] = True

                report["exact_matches"] += int(item["exact_match"])
                report["compile_valid"] += int(item["compile_valid"])

                _emit(log_file)
                _emit(log_file, f"RESULT: {'PASS' if item['exact_match'] else 'FAIL'}")
                _emit(log_file, "EXPECTED:")
                _emit(log_file, json.dumps(expected, indent=2, ensure_ascii=False))
                _emit(log_file, "ACTUAL:")
                _emit(log_file, json.dumps(predicted, indent=2, ensure_ascii=False))
                if item["diffs"]:
                    _emit(log_file, "DIFFS:")
                    _emit(log_file, json.dumps(item["diffs"], indent=2, ensure_ascii=False))
                if item["compile_error"]:
                    _emit(log_file, f"COMPILE ERROR: {item['compile_error']}")
            except Exception as exc:
                category = _error_category(exc)
                item["error"] = {
                    "category": category,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                item["slot_correct"], item["slot_total"] = _slot_score(expected, {})
                report["error_counts"][category] = report["error_counts"].get(category, 0) + 1
                _emit(log_file)
                _emit(log_file, "RESULT: ERROR")
                _emit(log_file, f"{category}: {type(exc).__name__}: {exc}")
            finally:
                item["latency_seconds"] = time.perf_counter() - started
                report["slot_correct"] += item["slot_correct"]
                report["slot_total"] += item["slot_total"]
                report["items"].append(item)
                _update_report_metrics(report)
                _write_json(report_path, report)
                _emit(log_file, f"LATENCY: {item['latency_seconds']:.3f}s")
                _emit(log_file, "=" * 100)

        report["finished_at"] = _utc_now()
        _update_report_metrics(report)
        _write_json(report_path, report)

        _emit(log_file, f"SUMMARY: {report['exact_matches']}/{report['count']} exact matches")
        _emit(log_file, f"COMPILE VALID: {report['compile_valid']}/{report['count']}")
        _emit(log_file, f"SLOT ACCURACY: {report['slot_accuracy']:.3f}")
        _emit(log_file, f"AVERAGE LATENCY: {report['latency_seconds']['average']:.3f}s")
        _emit(log_file, f"JSON report: {report_path}")
        _emit(log_file, f"Log: {log_path}")

    report["artifact_path"] = str(report_path)
    report["log_path"] = str(log_path)
    _write_json(report_path, report)
    return report


def run_with_optional_server(
    *,
    spec: ModelSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base_url = args.base_url
    process: subprocess.Popen[Any] | None = None
    server_log_path = args.output_dir / spec.slug / "vllm-server.log"
    api_key = args.api_key or DEFAULT_API_KEY

    if args.planner == "openai" and base_url is None:
        server_log_path.parent.mkdir(parents=True, exist_ok=True)
        command = _vllm_command(
            spec,
            host=args.host,
            port=args.port,
            extra_args=args.vllm_arg or [],
        )
        print(f"starting vLLM for {spec.id}", flush=True)
        print(" ".join(command), flush=True)
        server_log_file = server_log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                stdout=server_log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            base_url = f"http://{_connect_host(args.host)}:{args.port}/v1"
            wait_for_models(
                base_url=base_url,
                api_key=api_key,
                startup_timeout=args.startup_timeout,
                process=process,
            )
        except Exception as exc:
            if process is not None:
                stop_process(process)
            server_log_file.close()
            tail = _tail_text(server_log_path)
            if tail:
                print("vLLM server log tail:", file=sys.stderr)
                print(tail, file=sys.stderr)
                raise VLLMStartupError(
                    f"{exc}\n\nvLLM server log tail from {server_log_path}:\n{tail}"
                ) from exc
            raise
        server_log_file.close()
    elif args.planner == "openai":
        wait_for_models(
            base_url=base_url,
            api_key=api_key,
            startup_timeout=args.startup_timeout,
            process=None,
        )

    try:
        return evaluate_model(
            spec=spec,
            planner_name=args.planner,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            limit=args.limit,
            base_url=base_url or "gold://dataset",
            api_key=api_key,
            request_timeout=args.timeout,
            python_executable=args.python_executable,
        )
    finally:
        if process is not None:
            stop_process(process)


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": report["model"]["id"],
        "display_name": report["model"]["display_name"],
        "status": "completed",
        "count": report["count"],
        "exact_match_rate": report["exact_match_rate"],
        "compile_valid_rate": report["compile_valid_rate"],
        "slot_accuracy": report["slot_accuracy"],
        "latency_seconds": report["latency_seconds"],
        "error_counts": report["error_counts"],
        "artifact_path": report.get("artifact_path"),
        "log_path": report.get("log_path"),
    }


def command_download_models(args: argparse.Namespace) -> int:
    specs = load_manifest(args.manifest)
    if args.hf_home is not None:
        os.environ["HF_HOME"] = str(args.hf_home)

    print(json.dumps({
        "manifest": str(args.manifest),
        "hf_home": os.environ.get("HF_HOME"),
        "models": [spec.id for spec in specs],
        "dry_run": args.dry_run,
    }, indent=2))

    if args.dry_run:
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("download-models requires huggingface_hub, which is included in vLLM images") from exc

    for spec in specs:
        print(f"downloading {spec.id}", flush=True)
        snapshot_download(
            repo_id=spec.id,
            token=os.environ.get("HF_TOKEN") or None,
            local_files_only=args.local_files_only,
        )
    return 0


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--planner", choices=["openai", "gold"], default="openai")
    parser.add_argument("--base-url", help="Use an already-running OpenAI-compatible endpoint instead of starting vLLM.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--timeout", type=float, default=90.0, help="Per-row OpenAI request timeout in seconds.")
    parser.add_argument("--startup-timeout", type=float, default=900.0, help="Seconds to wait for /v1/models.")
    parser.add_argument("--host", default="0.0.0.0", help="vLLM bind host.")
    parser.add_argument("--port", type=int, default=8000, help="vLLM port.")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--vllm-arg",
        action="append",
        help="Additional single vLLM server argument. Repeat for multiple arguments.",
    )


def command_run_model(args: argparse.Namespace) -> int:
    specs = load_manifest(args.manifest)
    spec = find_model(specs, args.model)
    report = run_with_optional_server(spec=spec, args=args)
    print(json.dumps(_report_summary(report), indent=2))
    return 0


def command_run_suite(args: argparse.Namespace) -> int:
    specs = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary: dict[str, Any] = {
        "started_at": _utc_now(),
        "finished_at": None,
        "manifest": str(args.manifest),
        "dataset": str(args.dataset),
        "limit": args.limit,
        "planner": args.planner,
        "models": [],
    }
    _write_json(summary_path, summary)

    for spec in specs:
        try:
            report = run_with_optional_server(spec=spec, args=args)
            summary["models"].append(_report_summary(report))
            _write_json(summary_path, summary)
        except VLLMStartupError as exc:
            summary["models"].append({
                "id": spec.id,
                "display_name": spec.display_name,
                "status": "vllm_start_failed",
                "error": str(exc),
                "server_log_path": str(args.output_dir / spec.slug / "vllm-server.log"),
            })
            summary["finished_at"] = _utc_now()
            _write_json(summary_path, summary)
            print(json.dumps(summary, indent=2), file=sys.stderr)
            return 1

    summary["finished_at"] = _utc_now()
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/pod_eval.py",
        description="Run the dev NLSH eval against local vLLM OpenAI-compatible model servers.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-models", help="Prefetch all manifest models into HF_HOME.")
    download_parser.add_argument("--hf-home", type=Path)
    download_parser.add_argument("--local-files-only", action="store_true")
    download_parser.add_argument("--dry-run", action="store_true")
    download_parser.set_defaults(func=command_download_models)

    run_model_parser = subparsers.add_parser("run-model", help="Run one model from the manifest.")
    run_model_parser.add_argument("--model", required=True, help="Model id, display name, or manifest slug.")
    _add_eval_args(run_model_parser)
    run_model_parser.set_defaults(func=command_run_model)

    run_suite_parser = subparsers.add_parser("run-suite", help="Run all manifest models sequentially.")
    _add_eval_args(run_suite_parser)
    run_suite_parser.set_defaults(func=command_run_suite)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
