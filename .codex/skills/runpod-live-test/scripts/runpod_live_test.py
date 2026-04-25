#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nlsh.compiler import CompileError, compile_plan, required_tools_for_plan  # noqa: E402
from nlsh.dataio import ensure_parent, jsonl_paths  # noqa: E402
from nlsh.planner import (  # noqa: E402
    PlannerConfig,
    chat_completion_text,
    planner_chat_messages,
    planner_response_format,
    validate_planner_payload,
)
from nlsh.preflight import MissingToolsError, ensure_required_tools  # noqa: E402
from nlsh.runner import confirm_execution, print_compile_preview  # noqa: E402
from nlsh.schema import Clarification, PlanV1, normalize_plan, validate_plan_payload  # noqa: E402
from nlsh.settings import RunpodServeSettings, load_dotenv, runpod_proxy_url  # noqa: E402

DEFAULT_MODEL = "microsoft/Phi-4-mini-instruct:nlsh-phi4-ft"
PROBE_BUCKETS = [
    "clarification",
    "csv_to_json",
    "find_files",
    "json_filter",
    "json_group_count",
    "json_select_fields",
    "json_sort",
    "pdf_extract_pages",
    "pdf_merge",
    "pdf_search_text",
]
LIVE_PROBE_DIR = REPO_ROOT / "artifacts" / "live-probe"
DEFAULT_SANDBOX_ROOT = REPO_ROOT / "tmp" / "live-demo"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _request(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None,
    api_key: str | None,
    timeout: float,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/136.0.0.0 Safari/537.36"
        ),
    }
    data: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, method=method, headers=headers, data=data)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: dict[str, Any] | list[Any] | str = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return exc.code, parsed


def _resolve_urls(pod_id: str | None, proxy_url: str | None, port: int) -> tuple[str, str]:
    if proxy_url:
        root = proxy_url.rstrip("/")
        if root.endswith("/v1"):
            return root[:-3], root
        return root, f"{root}/v1"
    if not pod_id:
        raise SystemExit("Pass --pod-id or --proxy-url.")
    root = runpod_proxy_url(pod_id, port)
    return root, f"{root}/v1"


def _planner_config_from_args(args: argparse.Namespace) -> tuple[str | None, str, PlannerConfig]:
    load_dotenv()
    settings = RunpodServeSettings.from_env()
    pod_id = getattr(args, "pod_id", None) or settings.pod_id
    port = getattr(args, "port", None) or settings.port
    api_key = getattr(args, "api_key", None) or settings.api_key or ""
    proxy_root, openai_base = _resolve_urls(pod_id, getattr(args, "proxy_url", None), port)
    config = PlannerConfig(
        model=getattr(args, "model", None) or DEFAULT_MODEL,
        base_url=openai_base,
        api_key=api_key,
        request_timeout=getattr(args, "timeout", 60.0),
    )
    return pod_id, proxy_root, config


def _load_probe_records(dataset_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for jsonl_path in jsonl_paths(dataset_path):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                record["_source_path"] = str(jsonl_path)
                record["_source_line"] = line_number
                records.append(record)
    return records


def _expected_output_bucket(record: dict[str, Any]) -> str:
    output = validate_plan_payload(record["plan"])
    if isinstance(output, Clarification):
        return "clarification"
    return output.steps[-1].kind


def _messages_to_send(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("record is missing messages")
    if isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
        payload = messages[:-1]
    else:
        payload = messages
    if not payload:
        raise ValueError("record has no messages to send")
    return payload


def _expected_assistant_content(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
        content = messages[-1].get("content")
        if isinstance(content, str):
            return content
    return json.dumps(record["plan"], ensure_ascii=False)


def _stratified_probe_records(
    records: list[dict[str, Any]],
    *,
    buckets: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in buckets}
    for record in records:
        bucket = _expected_output_bucket(record)
        if bucket in grouped:
            grouped[bucket].append(record)
    missing = [bucket for bucket, items in grouped.items() if not items]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Dataset is missing required probe buckets: {joined}")

    chooser = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for bucket in buckets:
        candidates = sorted(
            grouped[bucket],
            key=lambda item: (item["_source_path"], item["_source_line"], item["prompt"]),
        )
        chosen.append(chooser.choice(candidates))
    return chosen


def _write_json_artifact(output_dir: Path, payload: dict[str, Any]) -> Path:
    timestamped_path = output_dir / f"live-probe-{_timestamp()}.json"
    latest_path = output_dir / "latest.json"
    ensure_parent(timestamped_path)
    body = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    timestamped_path.write_text(body, encoding="utf-8")
    latest_path.write_text(body, encoding="utf-8")
    return timestamped_path


def _preview_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.suffix in {".json", ".txt", ".csv"}:
        text = path.read_text(encoding="utf-8")
        return text[:500]
    return f"{path.name} ({path.stat().st_size} bytes)"


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
        stat = file_path.stat()
        snapshot[str(file_path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _diff_snapshots(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[str]:
    changed: list[str] = []
    for relative_path, after_meta in after.items():
        if before.get(relative_path) != after_meta:
            changed.append(relative_path)
    return changed


def _write_csv_fixture(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_parent(path)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json_fixture(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_blank_pdf(path: Path, *, page_count: int) -> None:
    from pypdf import PdfWriter

    ensure_parent(path)
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as handle:
        writer.write(handle)


def _create_demo_fixtures(sandbox: Path) -> None:
    _write_csv_fixture(
        sandbox / "orders.csv",
        [
            {"order_id": "1001", "status": "paid", "total": "19.95"},
            {"order_id": "1002", "status": "pending", "total": "42.00"},
            {"order_id": "1003", "status": "paid", "total": "10.50"},
        ],
    )
    _write_json_fixture(
        sandbox / "users.json",
        [
            {"user_id": 1, "role": "admin", "email": "alice@example.com"},
            {"user_id": 2, "role": "guest", "email": "bob@example.com"},
            {"user_id": 3, "role": "member", "email": "cora@example.com"},
        ],
    )
    _write_json_fixture(
        sandbox / "tickets.json",
        [
            {"ticket_id": "T-100", "priority": "high", "owner": "alice", "status": "open"},
            {"ticket_id": "T-101", "priority": "medium", "owner": "bob", "status": "open"},
        ],
    )
    _write_json_fixture(
        sandbox / "invoices.json",
        [
            {"invoice_id": "INV-1", "total": 15.0},
            {"invoice_id": "INV-2", "total": 99.5},
            {"invoice_id": "INV-3", "total": 48.2},
        ],
    )
    _write_json_fixture(
        sandbox / "incidents.json",
        [
            {"incident_id": "I-1", "severity": "high"},
            {"incident_id": "I-2", "severity": "low"},
            {"incident_id": "I-3", "severity": "high"},
        ],
    )
    _write_blank_pdf(sandbox / "contract_a.pdf", page_count=1)
    _write_blank_pdf(sandbox / "contract_b.pdf", page_count=1)
    _write_blank_pdf(sandbox / "handbook.pdf", page_count=4)
    _write_blank_pdf(sandbox / "contracts" / "executed_alpha.pdf", page_count=1)
    _write_blank_pdf(sandbox / "contracts" / "executed_beta.pdf", page_count=1)
    _write_blank_pdf(sandbox / "contracts" / "draft.pdf", page_count=1)


def _demo_prompt_entries() -> list[dict[str, str]]:
    return [
        {
            "name": "csv_to_json",
            "prompt": "convert orders.csv to JSON in orders.json",
            "primary_output": "orders.json",
        },
        {
            "name": "json_filter",
            "prompt": "from users.json keep rows where role is not guest into member_users.json",
            "primary_output": "member_users.json",
        },
        {
            "name": "json_select_fields",
            "prompt": "from tickets.json keep only ticket_id priority and owner into ticket_queue.json",
            "primary_output": "ticket_queue.json",
        },
        {
            "name": "json_sort",
            "prompt": "sort invoices.json by total descending into invoices_sorted.json",
            "primary_output": "invoices_sorted.json",
        },
        {
            "name": "json_group_count",
            "prompt": "count incidents.json by severity into incidents_by_severity.json",
            "primary_output": "incidents_by_severity.json",
        },
        {
            "name": "find_files",
            "prompt": "find every PDF under ./contracts with executed in the path",
            "primary_output": "",
        },
        {
            "name": "pdf_merge",
            "prompt": "merge contract_a.pdf and contract_b.pdf into contract_packet.pdf",
            "primary_output": "contract_packet.pdf",
        },
        {
            "name": "pdf_extract_pages",
            "prompt": "extract pages 2 through 3 from handbook.pdf into handbook_excerpt.pdf",
            "primary_output": "handbook_excerpt.pdf",
        },
    ]


def _run_compiled_in_dir(compiled_script: str, *, cwd: Path, allow_overwrite: bool) -> int:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    repo_pythonpath = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        f"{repo_pythonpath}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else repo_pythonpath
    )
    if allow_overwrite:
        env["NLSH_ALLOW_OVERWRITE"] = "1"
    completed = subprocess.run(
        ["bash", "-lc", compiled_script],
        check=False,
        cwd=cwd,
        env=env,
    )
    return completed.returncode


def command_resolve_url(args: argparse.Namespace) -> int:
    load_dotenv()
    settings = RunpodServeSettings.from_env()
    pod_id = args.pod_id or settings.pod_id
    port = args.port or settings.port
    proxy_root, openai_base = _resolve_urls(pod_id, args.proxy_url, port)
    payload = {
        "pod_id": pod_id,
        "proxy_url": proxy_root,
        "openai_base_url": openai_base,
        "model": args.model or DEFAULT_MODEL,
        "api_key_configured": bool(args.api_key or settings.api_key),
    }
    print(json.dumps(payload, indent=2))
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    load_dotenv()
    settings = RunpodServeSettings.from_env()
    pod_id = args.pod_id or settings.pod_id
    port = args.port or settings.port
    api_key = args.api_key or settings.api_key
    model = args.model or DEFAULT_MODEL
    proxy_root, openai_base = _resolve_urls(pod_id, args.proxy_url, port)

    model_info_status, model_info = _request(
        "GET",
        f"{proxy_root}/model_info",
        payload=None,
        api_key=api_key,
        timeout=args.timeout,
    )
    models_status, models = _request(
        "GET",
        f"{openai_base}/models",
        payload=None,
        api_key=api_key,
        timeout=args.timeout,
    )
    chat_status, chat = _request(
        "POST",
        f"{openai_base}/chat/completions",
        payload={
            "model": model,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        },
        api_key=api_key,
        timeout=args.timeout,
    )

    payload = {
        "pod_id": pod_id,
        "proxy_url": proxy_root,
        "openai_base_url": openai_base,
        "model": model,
        "api_key_configured": bool(api_key),
        "checks": {
            "model_info": {"status": model_info_status, "body": model_info},
            "models": {"status": models_status, "body": models},
            "chat_completions": {"status": chat_status, "body": chat},
        },
    }
    print(json.dumps(payload, indent=2))
    if model_info_status != 200 or models_status != 200 or chat_status != 200:
        return 1
    return 0


def command_probe_dataset(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    records = _load_probe_records(dataset_path)
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 1_000_000_000)
    chosen = _stratified_probe_records(records, buckets=PROBE_BUCKETS, seed=seed)
    pod_id, proxy_root, config = _planner_config_from_args(args)

    results: list[dict[str, Any]] = []
    exact_matches = 0
    plan_outputs = 0
    clarification_outputs = 0
    compile_valid_plan_outputs = 0
    failures = 0

    print(f"Probe model: {config.model}")
    print(f"Probe base URL: {config.base_url}")
    print(f"Proxy URL: {proxy_root}")
    print(f"Seed: {seed}")
    print(f"Sample count: {len(chosen)}")

    for index, record in enumerate(chosen, start=1):
        expected = validate_plan_payload(record["plan"])
        expected_dump = normalize_plan(expected)
        expected_assistant = _expected_assistant_content(record)
        if args.mode == "runtime":
            sent_messages = planner_chat_messages(record["prompt"])
            response_format = planner_response_format()
        else:
            sent_messages = _messages_to_send(record)
            response_format = None

        print("")
        print("=" * 100)
        print(f"Sample {index}/{len(chosen)}")
        print(f"Bucket: {_expected_output_bucket(record)}")
        print(f"Source: {record['_source_path']}:{record['_source_line']}")
        print(f"Prompt: {record['prompt']}")

        raw_content: str | None = None
        parse_error: str | None = None
        compile_error: str | None = None
        actual_dump: dict[str, Any] | None = None
        compile_valid: bool | None = None
        output_kind = "error"

        try:
            raw_content = chat_completion_text(
                messages=sent_messages,
                config=config,
                response_format=response_format,
                temperature=0,
                max_tokens=args.max_tokens,
            )
            actual = validate_planner_payload(raw_content, extract_json_fragment=True)
            actual_dump = normalize_plan(actual)
            exact_match = actual_dump == expected_dump
            exact_matches += int(exact_match)
            if isinstance(actual, PlanV1):
                output_kind = "plan"
                plan_outputs += 1
                try:
                    compile_plan(actual, python_executable=sys.executable)
                    compile_valid = True
                    compile_valid_plan_outputs += 1
                except CompileError as exc:
                    compile_valid = False
                    compile_error = str(exc)
            else:
                output_kind = "clarification"
                clarification_outputs += 1
            print(f"Exact match: {'yes' if exact_match else 'no'}")
            if compile_valid is None:
                print("Compile valid: n/a (clarification)")
            else:
                print(f"Compile valid: {'yes' if compile_valid else 'no'}")
        except Exception as exc:
            failures += 1
            exact_match = False
            compile_valid = None
            parse_error = f"{type(exc).__name__}: {exc}"
            print(f"Failure: {parse_error}")

        results.append(
            {
                "bucket": _expected_output_bucket(record),
                "source_path": record["_source_path"],
                "source_line": record["_source_line"],
                "focus": record.get("focus"),
                "prompt": record["prompt"],
                "sent_messages": sent_messages,
                "expected_assistant": expected_assistant,
                "expected_normalized": expected_dump,
                "actual_assistant": raw_content,
                "actual_normalized": actual_dump,
                "exact_match": exact_match,
                "compile_valid": compile_valid,
                "compile_error": compile_error,
                "parse_or_transport_error": parse_error,
                "output_kind": output_kind,
            }
        )

    summary = {
        "dataset": str(dataset_path),
        "mode": args.mode,
        "seed": seed,
        "count": len(chosen),
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / len(chosen),
        "plan_outputs": plan_outputs,
        "clarification_outputs": clarification_outputs,
        "compile_valid_plan_outputs": compile_valid_plan_outputs,
        "compile_valid_plan_rate": (compile_valid_plan_outputs / plan_outputs if plan_outputs else 0.0),
        "transport_or_parse_failures": failures,
        "model": config.model,
        "base_url": config.base_url,
        "proxy_url": proxy_root,
        "pod_id": pod_id,
    }
    artifact_payload = {
        "summary": summary,
        "results": results,
    }
    artifact_path = _write_json_artifact(Path(args.artifact_dir), artifact_payload)
    print("")
    print("=" * 100)
    print(json.dumps({"summary": summary, "artifact": str(artifact_path)}, indent=2, ensure_ascii=False))
    return 1 if failures else 0


def command_interactive_demo(args: argparse.Namespace) -> int:
    pod_id, proxy_root, config = _planner_config_from_args(args)
    sandbox = Path(args.sandbox_root) / _timestamp()
    sandbox.mkdir(parents=True, exist_ok=True)
    _create_demo_fixtures(sandbox)
    prompt_entries = _demo_prompt_entries()[: args.max_attempts]
    transcript: dict[str, Any] = {
        "pod_id": pod_id,
        "proxy_url": proxy_root,
        "openai_base_url": config.base_url,
        "model": config.model,
        "sandbox": str(sandbox),
        "attempts": [],
        "success": False,
    }

    print(f"Sandbox: {sandbox}")
    print(f"Proxy URL: {proxy_root}")
    print(f"Model: {config.model}")

    for index, entry in enumerate(prompt_entries, start=1):
        prompt = entry["prompt"]
        sent_messages = planner_chat_messages(prompt)
        attempt: dict[str, Any] = {
            "index": index,
            "name": entry["name"],
            "prompt": prompt,
            "sent_messages": sent_messages,
        }

        print("")
        print("=" * 100)
        print(f"Attempt {index}/{len(prompt_entries)}")
        print(f"Prompt: {prompt}")

        try:
            raw_content = chat_completion_text(
                messages=sent_messages,
                config=config,
                response_format=planner_response_format(),
                temperature=0,
                max_tokens=args.max_tokens,
            )
            attempt["actual_assistant"] = raw_content
            actual = validate_planner_payload(raw_content, extract_json_fragment=True)
            attempt["normalized_output"] = normalize_plan(actual)
        except Exception as exc:
            attempt["status"] = "transport_or_parse_error"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            transcript["attempts"].append(attempt)
            print(f"Failure: {attempt['error']}")
            continue

        if isinstance(actual, Clarification):
            attempt["status"] = "clarification"
            transcript["attempts"].append(attempt)
            print("Clarification:")
            print(actual.question)
            continue

        try:
            compiled = compile_plan(actual, python_executable=sys.executable)
            attempt["compiled_script"] = compiled.script
            attempt["summary"] = compiled.summary
            attempt["output_files"] = compiled.output_files
        except CompileError as exc:
            attempt["status"] = "compile_error"
            attempt["error"] = str(exc)
            transcript["attempts"].append(attempt)
            print(f"Compile error: {exc}")
            continue

        try:
            ensure_required_tools(required_tools_for_plan(actual))
        except MissingToolsError as exc:
            attempt["status"] = "missing_tools"
            attempt["error"] = str(exc)
            transcript["attempts"].append(attempt)
            print(str(exc))
            continue

        print_compile_preview(actual, compiled)
        confirmed = args.yes or confirm_execution()
        attempt["confirmed"] = confirmed
        if not confirmed:
            attempt["status"] = "declined"
            transcript["attempts"].append(attempt)
            print("Skipped at confirmation prompt.")
            continue

        before = _snapshot_tree(sandbox)
        exit_code = _run_compiled_in_dir(compiled.script, cwd=sandbox, allow_overwrite=args.allow_overwrite)
        after = _snapshot_tree(sandbox)
        changed_files = _diff_snapshots(before, after)
        attempt["execution_exit_code"] = exit_code
        attempt["changed_files"] = changed_files

        primary_output = entry.get("primary_output") or ""
        if primary_output:
            primary_path = sandbox / primary_output
            attempt["primary_output"] = primary_output
            attempt["primary_output_preview"] = _preview_file(primary_path)

        transcript["attempts"].append(attempt)

        print(f"Execution exit code: {exit_code}")
        if changed_files:
            print("Changed files:")
            for relative_path in changed_files:
                print(f"  - {relative_path}")
        if primary_output and primary_path.exists():
            print("Primary output preview:")
            print(_preview_file(primary_path))

        if exit_code == 0:
            attempt["status"] = "executed"
            transcript["success"] = True
            transcript["successful_attempt"] = index
            break

        attempt["status"] = "execution_failed"

    transcript_path = sandbox / "execution-transcript.json"
    transcript_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("")
    print(json.dumps({"success": transcript["success"], "sandbox": str(sandbox), "transcript": str(transcript_path)}, indent=2))
    return 0 if transcript["success"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runpod_live_test.py",
        description="Resolve, smoke-test, probe, and sandbox-test the public Runpod proxy URL for the bundled NLSH serving image.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve-url", help="Resolve the proxy URL and OpenAI base URL.")
    resolve_parser.add_argument("--pod-id", help="Runpod pod id. Defaults to RUNPOD_POD_ID when available.")
    resolve_parser.add_argument("--proxy-url", help="Existing proxy root or /v1 URL.")
    resolve_parser.add_argument("--port", type=int, default=8000)
    resolve_parser.add_argument("--api-key", help="Bearer token. Defaults to RUNPOD_SERVE_API_KEY.")
    resolve_parser.add_argument("--model", default=DEFAULT_MODEL)
    resolve_parser.set_defaults(func=command_resolve_url)

    smoke_parser = subparsers.add_parser("smoke", help="Call /model_info, /v1/models, and /v1/chat/completions.")
    smoke_parser.add_argument("--pod-id", help="Runpod pod id. Defaults to RUNPOD_POD_ID when available.")
    smoke_parser.add_argument("--proxy-url", help="Existing proxy root or /v1 URL.")
    smoke_parser.add_argument("--port", type=int, default=8000)
    smoke_parser.add_argument("--api-key", help="Bearer token. Defaults to RUNPOD_SERVE_API_KEY.")
    smoke_parser.add_argument("--model", default=DEFAULT_MODEL)
    smoke_parser.add_argument("--timeout", type=float, default=60.0)
    smoke_parser.add_argument("--max-tokens", type=int, default=24)
    smoke_parser.add_argument("--prompt", default="Reply with READY and nothing else.")
    smoke_parser.set_defaults(func=command_smoke)

    probe_parser = subparsers.add_parser(
        "probe-dataset",
        help="Run a saved 10-sample stratified probe against the live runtime prompt shape.",
    )
    probe_parser.add_argument("--pod-id", help="Runpod pod id. Defaults to RUNPOD_POD_ID when available.")
    probe_parser.add_argument("--proxy-url", help="Existing proxy root or /v1 URL.")
    probe_parser.add_argument("--port", type=int, default=8000)
    probe_parser.add_argument("--api-key", help="Bearer token. Defaults to RUNPOD_SERVE_API_KEY.")
    probe_parser.add_argument("--model", default=DEFAULT_MODEL)
    probe_parser.add_argument("--timeout", type=float, default=60.0)
    probe_parser.add_argument("--max-tokens", type=int, default=600)
    probe_parser.add_argument("--dataset", default=str(REPO_ROOT / "data" / "samples"))
    probe_parser.add_argument("--seed", type=int)
    probe_parser.add_argument("--mode", choices=["runtime", "replay-messages"], default="runtime")
    probe_parser.add_argument("--artifact-dir", default=str(LIVE_PROBE_DIR))
    probe_parser.set_defaults(func=command_probe_dataset)

    demo_parser = subparsers.add_parser(
        "interactive-demo",
        help="Create a disposable sandbox, prompt the live endpoint, confirm, and execute one successful plan.",
    )
    demo_parser.add_argument("--pod-id", help="Runpod pod id. Defaults to RUNPOD_POD_ID when available.")
    demo_parser.add_argument("--proxy-url", help="Existing proxy root or /v1 URL.")
    demo_parser.add_argument("--port", type=int, default=8000)
    demo_parser.add_argument("--api-key", help="Bearer token. Defaults to RUNPOD_SERVE_API_KEY.")
    demo_parser.add_argument("--model", default=DEFAULT_MODEL)
    demo_parser.add_argument("--timeout", type=float, default=60.0)
    demo_parser.add_argument("--max-tokens", type=int, default=600)
    demo_parser.add_argument("--sandbox-root", default=str(DEFAULT_SANDBOX_ROOT))
    demo_parser.add_argument("--max-attempts", type=int, default=len(_demo_prompt_entries()))
    demo_parser.add_argument("--yes", action="store_true", help="Auto-confirm the first execution-ready plan.")
    demo_parser.add_argument("--allow-overwrite", action="store_true")
    demo_parser.set_defaults(func=command_interactive_demo)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
