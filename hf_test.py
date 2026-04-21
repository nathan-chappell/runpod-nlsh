from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from nlsh.dataio import load_jsonl
from nlsh.planner import OpenAIPlanner
from nlsh.schema import PlanV1


DATASET_PATH = Path(os.environ.get("NLSH_DEV_DATASET", "data/dev.messages.jsonl"))
RESULT_PATH = Path(os.environ.get("NLSH_RESULT_PATH", "artifacts/hf_test_result.txt"))


def normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: normalize(value) for key, value in sorted(obj.items())}
    if isinstance(obj, list):
        return [normalize(value) for value in obj]
    return obj


def diff_expected_vs_actual(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, dict[str, Any]]:
    diffs: dict[str, dict[str, Any]] = {}
    for key in sorted(set(expected) | set(actual)):
        expected_value = expected.get(key, "<missing>")
        actual_value = actual.get(key, "<missing>")
        if normalize(expected_value) != normalize(actual_value):
            diffs[key] = {"expected": expected_value, "actual": actual_value}
    return diffs


def main() -> None:
    records = load_jsonl(DATASET_PATH)
    planner = OpenAIPlanner()

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as result_file:
        def emit(*parts: object, sep: str = " ", end: str = "\n") -> None:
            text = sep.join(str(part) for part in parts) + end
            print(*parts, sep=sep, end=end)
            result_file.write(text)

        emit(f"model: {planner.config.model}")
        emit(f"base_url: {planner.config.base_url}")
        emit(f"dataset: {DATASET_PATH}")
        emit("=" * 100)

        passed = 0
        for index, record in enumerate(records, start=1):
            prompt = record["prompt"]
            expected_plan = PlanV1.model_validate(record["plan"])
            expected = expected_plan.model_dump(mode="json", exclude_none=False)

            emit(f"[{index}/{len(records)}] PROMPT")
            emit(prompt)

            try:
                actual_plan = planner.plan(prompt)
                actual = actual_plan.model_dump(mode="json", exclude_none=False)
                diffs = diff_expected_vs_actual(expected, actual)
                ok = not diffs
                passed += int(ok)

                emit()
                emit(f"RESULT: {'PASS' if ok else 'FAIL'}")
                emit()
                emit("EXPECTED:")
                emit(json.dumps(expected, indent=2, ensure_ascii=False))
                emit()
                emit("ACTUAL:")
                emit(json.dumps(actual, indent=2, ensure_ascii=False))
                if diffs:
                    emit()
                    emit("DIFFS:")
                    emit(json.dumps(diffs, indent=2, ensure_ascii=False))
            except Exception as exc:
                emit()
                emit("RESULT: ERROR")
                emit(repr(exc))

            emit("=" * 100)

        emit(f"SUMMARY: {passed}/{len(records)} exact matches")

    print(f"\nResults written to {RESULT_PATH}")


if __name__ == "__main__":
    main()
