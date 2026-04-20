from __future__ import annotations

import re
import subprocess
from copy import deepcopy
from typing import Any

from nlsh.compiler import CompileError, CompileResult, compile_plan
from nlsh.schema import ClarificationQuestion, PlanV1


FIELD_PATH_RE = re.compile(r"steps\[(\d+)\]\.(.+)")


def _coerce_answer(raw: str, expected_type: str) -> Any:
    value = raw.strip()
    if expected_type == "string":
        return value
    if expected_type == "integer":
        return int(value)
    if expected_type == "number":
        return float(value)
    if expected_type == "boolean":
        lowered = value.lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
        raise ValueError("Expected yes/no or true/false")
    if expected_type == "string_list":
        return [item.strip() for item in value.split(",") if item.strip()]
    raise ValueError(f"Unsupported expected type: {expected_type}")


def _apply_field_value(plan: PlanV1, field_path: str, value: Any) -> PlanV1:
    data = deepcopy(plan.model_dump(mode="json", exclude_none=False))
    match = FIELD_PATH_RE.fullmatch(field_path)
    if not match:
        raise ValueError(f"Unsupported field path: {field_path}")
    step_index = int(match.group(1))
    field_name = match.group(2)
    data["steps"][step_index][field_name] = value
    data["questions"] = [item for item in data["questions"] if item["field_path"] != field_path]
    return PlanV1.model_validate(data)


def ask_clarifying_questions(plan: PlanV1) -> PlanV1:
    updated_plan = plan
    for question in list(updated_plan.questions):
        while True:
            answer = input(f"{question.prompt.strip()} ").strip()
            if not answer and not question.required:
                break
            try:
                coerced = _coerce_answer(answer, question.expected_type)
                updated_plan = _apply_field_value(updated_plan, question.field_path, coerced)
                break
            except ValueError as exc:
                print(f"Invalid answer: {exc}")
    return updated_plan


def print_compile_preview(plan: PlanV1, compiled: CompileResult) -> None:
    print("Plan:")
    print(plan.model_dump_json(indent=2))
    print()
    print("Compiled script:")
    print(compiled.script)


def confirm_execution() -> bool:
    answer = input("Execute this plan? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def run_compiled_script(
    compiled: CompileResult,
    *,
    allow_overwrite: bool = False,
) -> int:
    env = None
    if allow_overwrite:
        import os

        env = dict(os.environ)
        env["NLSH_ALLOW_OVERWRITE"] = "1"
    completed = subprocess.run(
        ["bash", "-lc", compiled.script],
        check=False,
        env=env,
    )
    return completed.returncode


def prepare_plan_for_execution(
    plan: PlanV1,
    *,
    python_executable: str,
) -> tuple[PlanV1, CompileResult]:
    current_plan = plan
    if current_plan.questions:
        current_plan = ask_clarifying_questions(current_plan)
    try:
        compiled = compile_plan(current_plan, python_executable=python_executable)
    except CompileError as exc:
        raise CompileError(f"Plan is not execution-ready: {exc}") from exc
    return current_plan, compiled

