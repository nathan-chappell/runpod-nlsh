from __future__ import annotations

import subprocess

from nlsh.compiler import CompileError, CompileResult, compile_plan
from nlsh.schema import PlanV1


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
    try:
        compiled = compile_plan(plan, python_executable=python_executable)
    except CompileError as exc:
        raise CompileError(f"Plan is not execution-ready: {exc}") from exc
    return plan, compiled
