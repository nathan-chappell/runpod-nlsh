#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nlsh.compiler import CompileError, compile_plan
from nlsh.dataio import load_jsonl
from nlsh.planner import OpenAIPlanner, PlannerConfig
from nlsh.schema import PlanV1, normalize_plan, validate_plan_payload


DEFAULT_DATASET = Path("data/dev.messages.jsonl")
DEFAULT_JSON_OUTPUT = Path("artifacts/runpod_dev_eval.json")
DEFAULT_LOG_OUTPUT = Path("artifacts/runpod_dev_eval.txt")


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


def _write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/runpod_dev_eval.py",
        description="Run the dev message set against the configured Runpod/OpenAI-compatible endpoint.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--log-output", type=Path, default=DEFAULT_LOG_OUTPUT)
    parser.add_argument("--timeout", type=float, help="Override NLSH_REQUEST_TIMEOUT for this run.")
    parser.add_argument("--limit", type=int, help="Only run the first N dataset rows.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.timeout is not None:
        os.environ["NLSH_REQUEST_TIMEOUT"] = str(args.timeout)

    config = PlannerConfig.from_env()
    planner = OpenAIPlanner(config)
    records = load_jsonl(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "dataset": str(args.dataset),
        "model": config.model,
        "base_url": config.base_url,
        "request_timeout": config.request_timeout,
        "count": len(records),
        "exact_matches": 0,
        "compile_valid": 0,
        "slot_correct": 0,
        "slot_total": 0,
        "items": [],
    }

    args.log_output.parent.mkdir(parents=True, exist_ok=True)
    with args.log_output.open("w", encoding="utf-8") as log_file:
        def emit(text: str = "") -> None:
            print(text, flush=True)
            log_file.write(text + "\n")
            log_file.flush()

        emit(f"model: {config.model}")
        emit(f"base_url: {config.base_url}")
        emit(f"request_timeout: {config.request_timeout}")
        emit(f"dataset: {args.dataset}")
        emit("=" * 100)

        _write_json_report(args.json_output, report)

        for index, record in enumerate(records, start=1):
            prompt = record["prompt"]
            expected_plan = validate_plan_payload(record["plan"])
            expected = normalize_plan(expected_plan)
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
            }

            emit(f"[{index}/{len(records)}] PROMPT")
            emit(prompt)

            try:
                predicted_plan = planner.plan(prompt)
                predicted = normalize_plan(predicted_plan)
                item["predicted"] = predicted
                item["diffs"] = _diff_expected_vs_actual(expected, predicted)
                item["exact_match"] = not item["diffs"]
                correct_slots, total_slots = _slot_score(expected, predicted)
                report["slot_correct"] += correct_slots
                report["slot_total"] += total_slots

                if isinstance(predicted_plan, PlanV1):
                    try:
                        compile_plan(predicted_plan, python_executable=sys.executable)
                        item["compile_valid"] = True
                    except CompileError as exc:
                        item["compile_error"] = str(exc)
                elif predicted_plan.kind == "clarification":
                    item["compile_valid"] = True

                report["exact_matches"] += int(item["exact_match"])
                report["compile_valid"] += int(item["compile_valid"])

                emit()
                emit(f"RESULT: {'PASS' if item['exact_match'] else 'FAIL'}")
                emit("EXPECTED:")
                emit(json.dumps(expected, indent=2, ensure_ascii=False))
                emit("ACTUAL:")
                emit(json.dumps(predicted, indent=2, ensure_ascii=False))
                if item["diffs"]:
                    emit("DIFFS:")
                    emit(json.dumps(item["diffs"], indent=2, ensure_ascii=False))
                if item["compile_error"]:
                    emit(f"COMPILE ERROR: {item['compile_error']}")
            except Exception as exc:
                item["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                emit()
                emit("RESULT: ERROR")
                emit(f"{type(exc).__name__}: {exc}")

            report["items"].append(item)
            _write_json_report(args.json_output, report)
            emit("=" * 100)

        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["exact_match_rate"] = report["exact_matches"] / report["count"] if report["count"] else 0.0
        report["compile_valid_rate"] = report["compile_valid"] / report["count"] if report["count"] else 0.0
        report["slot_accuracy"] = (
            report["slot_correct"] / report["slot_total"]
            if report["slot_total"]
            else 0.0
        )
        _write_json_report(args.json_output, report)

        emit(f"SUMMARY: {report['exact_matches']}/{report['count']} exact matches")
        emit(f"COMPILE VALID: {report['compile_valid']}/{report['count']}")
        emit(f"SLOT ACCURACY: {report['slot_accuracy']:.3f}")
        emit(f"JSON report: {args.json_output}")
        emit(f"Log: {args.log_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
