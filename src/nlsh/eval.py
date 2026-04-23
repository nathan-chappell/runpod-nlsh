from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nlsh.compiler import CompileError, compile_plan
from nlsh.dataio import default_dataset_path, ensure_parent, load_jsonl
from nlsh.planner import Planner
from nlsh.schema import PlanV1, PlannerOutput, normalize_plan, validate_plan_payload


@dataclass(slots=True)
class EvalItem:
    prompt: str
    expected: PlannerOutput
    predicted: PlannerOutput
    exact_match: bool
    compile_valid: bool
    compile_error: str | None


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            items.update(_flatten(value, next_prefix))
        return items
    if isinstance(obj, list):
        for index, value in enumerate(obj):
            next_prefix = f"{prefix}[{index}]"
            items.update(_flatten(value, next_prefix))
        return items
    items[prefix] = obj
    return items


def load_eval_records(dataset_path: Path | None = None) -> list[dict[str, Any]]:
    path = dataset_path or default_dataset_path()
    return load_jsonl(path)


def evaluate_planner(
    planner: Planner,
    dataset_path: Path | None = None,
    label: str = "samples",
    python_executable: str = "python",
) -> dict[str, Any]:
    path = dataset_path or default_dataset_path()
    raw_records = load_eval_records(dataset_path=path)
    items: list[EvalItem] = []
    total_slots = 0
    correct_slots = 0

    for record in raw_records:
        expected = validate_plan_payload(record["plan"])
        predicted = planner.plan(record["prompt"])
        expected_dump = normalize_plan(expected)
        predicted_dump = normalize_plan(predicted)
        exact_match = expected_dump == predicted_dump

        expected_slots = _flatten(expected_dump)
        predicted_slots = _flatten(predicted_dump)
        slot_keys = set(expected_slots) | set(predicted_slots)
        total_slots += len(slot_keys)
        correct_slots += sum(1 for key in slot_keys if expected_slots.get(key) == predicted_slots.get(key))

        compile_valid = True
        compile_error: str | None = None
        if isinstance(predicted, PlanV1):
            try:
                compile_plan(predicted, python_executable=python_executable)
            except CompileError as exc:
                compile_valid = False
                compile_error = str(exc)

        items.append(
            EvalItem(
                prompt=record["prompt"],
                expected=expected,
                predicted=predicted,
                exact_match=exact_match,
                compile_valid=compile_valid,
                compile_error=compile_error,
            )
        )

    results = {
        "dataset": label,
        "dataset_path": str(path),
        "exact_match_rate": (sum(item.exact_match for item in items) / len(items)) if items else 0.0,
        "compile_valid_rate": (sum(item.compile_valid for item in items) / len(items)) if items else 0.0,
        "slot_accuracy": (correct_slots / total_slots) if total_slots else 0.0,
        "count": len(items),
        "items": [
            {
                "prompt": item.prompt,
                "exact_match": item.exact_match,
                "compile_valid": item.compile_valid,
                "compile_error": item.compile_error,
                "expected": normalize_plan(item.expected),
                "predicted": normalize_plan(item.predicted),
            }
            for item in items
        ],
    }
    return results


def write_eval_artifact(results: dict[str, Any], output_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    label = str(results.get("dataset", "samples"))
    output_path = output_dir / f"{label}-eval-{timestamp}.json"
    latest_path = output_dir / "latest.json"
    ensure_parent(output_path)
    payload = json.dumps(results, indent=2, ensure_ascii=False)
    output_path.write_text(payload + "\n", encoding="utf-8")
    latest_path.write_text(payload + "\n", encoding="utf-8")
    return output_path
