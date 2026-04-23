from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_FRACTION = 0.2


def default_dataset_path() -> Path:
    return REPO_ROOT / "data" / "samples"


def jsonl_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        paths = sorted(path.rglob("*.jsonl"))
        if not paths:
            raise ValueError(f"No JSONL files found under {path}")
        return paths
    raise FileNotFoundError(f"Dataset path does not exist: {path}")


def _load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
    return records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for jsonl_path in jsonl_paths(path):
        records.extend(_load_jsonl_file(jsonl_path))
    return records


def partition_records(
    records: list[dict[str, Any]],
    *,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if eval_fraction < 0 or eval_fraction >= 1:
        raise ValueError("eval_fraction must be >= 0 and < 1")
    if len(records) < 2 or eval_fraction == 0:
        return list(records), []

    eval_count = max(1, round(len(records) * eval_fraction))
    eval_count = min(eval_count, len(records) - 1)
    ranked = sorted(
        enumerate(records),
        key=lambda item: (_partition_key(item[1]), item[0]),
    )
    eval_indices = {index for index, _record in ranked[:eval_count]}
    train_records = [record for index, record in enumerate(records) if index not in eval_indices]
    eval_records = [record for index, record in enumerate(records) if index in eval_indices]
    return train_records, eval_records


def _partition_key(record: dict[str, Any]) -> str:
    payload = {
        "focus": record.get("focus"),
        "prompt": record.get("prompt"),
        "plan": record.get("plan"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
