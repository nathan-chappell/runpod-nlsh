from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_SPLIT_ROOT = REPO_ROOT / "data" / "splits" / "v1"


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


def partition_records_three_way(
    records: list[dict[str, Any]],
    *,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if eval_fraction < 0 or test_fraction < 0:
        raise ValueError("eval_fraction and test_fraction must be >= 0")
    if eval_fraction + test_fraction >= 1:
        raise ValueError("eval_fraction + test_fraction must be < 1")
    if len(records) < 3:
        raise ValueError("Need at least 3 records to create train/eval/test splits")

    ranked = sorted(
        enumerate(records),
        key=lambda item: (_partition_key(item[1]), item[0]),
    )
    train_count, eval_count, test_count = _three_way_split_counts(
        len(records),
        eval_fraction=eval_fraction,
        test_fraction=test_fraction,
    )
    test_indices = {index for index, _record in ranked[:test_count]}
    eval_indices = {index for index, _record in ranked[test_count : test_count + eval_count]}
    train_indices = {index for index, _record in ranked[test_count + eval_count : test_count + eval_count + train_count]}
    train_records = [record for index, record in enumerate(records) if index in train_indices]
    eval_records = [record for index, record in enumerate(records) if index in eval_indices]
    test_records = [record for index, record in enumerate(records) if index in test_indices]
    return train_records, eval_records, test_records


def materialize_dataset_splits(
    source_dir: Path,
    output_dir: Path,
    *,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> dict[str, Any]:
    resolved_source = source_dir.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == resolved_source:
        raise ValueError("output_dir must be different from source_dir")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    summary: dict[str, Any] = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "eval_fraction": eval_fraction,
        "test_fraction": test_fraction,
        "splits": {"train": 0, "eval": 0, "test": 0},
        "files": [],
    }

    for source_path in jsonl_paths(source_dir):
        records = _load_jsonl_file(source_path)
        train_records, eval_records, test_records = partition_records_three_way(
            records,
            eval_fraction=eval_fraction,
            test_fraction=test_fraction,
        )
        relative_path = source_path.relative_to(source_dir)
        split_payloads = {
            "train": train_records,
            "eval": eval_records,
            "test": test_records,
        }
        for split_name, split_records in split_payloads.items():
            destination = output_dir / split_name / relative_path
            _write_jsonl_file(destination, split_records)
            summary["splits"][split_name] += len(split_records)
        summary["files"].append(
            {
                "path": str(relative_path),
                "source_records": len(records),
                "train_records": len(train_records),
                "eval_records": len(eval_records),
                "test_records": len(test_records),
            }
        )

    ensure_parent(output_dir / "split_manifest.json")
    (output_dir / "split_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


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


def _three_way_split_counts(record_count: int, *, eval_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    if record_count < 3:
        raise ValueError("record_count must be at least 3")
    eval_count = _fractional_count(record_count, eval_fraction)
    test_count = _fractional_count(record_count, test_fraction)

    while eval_count + test_count > record_count - 1:
        if eval_count >= test_count and eval_count > 1:
            eval_count -= 1
            continue
        if test_count > 1:
            test_count -= 1
            continue
        raise ValueError("Not enough records to reserve non-empty train/eval/test splits")

    train_count = record_count - eval_count - test_count
    if train_count < 1:
        raise ValueError("Not enough records to reserve a non-empty training split")
    return train_count, eval_count, test_count


def _fractional_count(record_count: int, fraction: float) -> int:
    if fraction == 0:
        return 0
    return max(1, math.floor(record_count * fraction))


def _write_jsonl_file(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
