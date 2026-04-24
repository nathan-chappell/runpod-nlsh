from __future__ import annotations

import json
import random
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from nlsh.compiler import CompileError, compile_plan, required_tools_for_plan
from nlsh.dataio import default_dataset_path, jsonl_paths
from nlsh.eval import evaluate_planner, write_eval_artifact
from nlsh.planner import (
    PlannerConfig,
    chat_completion_text,
    load_planner,
    plan_to_pretty_json,
    planner_chat_messages,
    planner_response_format,
    validate_planner_payload,
)
from nlsh.preflight import MissingToolsError, ensure_required_tools
from nlsh.runner import confirm_execution, prepare_plan_for_execution, print_compile_preview, run_compiled_script
from nlsh.schema import Clarification, PlanV1, normalize_plan, validate_plan_payload

app = typer.Typer(
    name="nlsh",
    no_args_is_help=True,
    help="Natural-language file, PDF, CSV, and JSON workflow compiler.",
)


class ProbeMode(str, Enum):
    runtime = "runtime"
    replay_messages = "replay-messages"


def plan_prompt(prompt: str, *, planner_name: str, dataset_path: Path | None) -> int:
    planner = load_planner(planner_name, dataset_path=dataset_path)
    output = planner.plan(prompt)
    typer.echo(plan_to_pretty_json(output))
    if isinstance(output, Clarification):
        return 0
    try:
        compiled = compile_plan(output, python_executable=sys.executable)
    except CompileError:
        return 0
    typer.echo("")
    typer.echo("# Compiled preview")
    typer.echo(compiled.script)
    return 0


def run_prompt(
    prompt: str,
    *,
    planner_name: str,
    dataset_path: Path | None,
    yes: bool,
    allow_overwrite: bool,
) -> int:
    planner = load_planner(planner_name, dataset_path=dataset_path)
    output = planner.plan(prompt)
    if isinstance(output, Clarification):
        typer.echo(output.question)
        return 1
    plan, compiled = prepare_plan_for_execution(output, python_executable=sys.executable)
    try:
        ensure_required_tools(required_tools_for_plan(plan))
    except MissingToolsError as exc:
        typer.echo(str(exc), err=True)
        return 2
    print_compile_preview(plan, compiled)
    if not yes and not confirm_execution():
        typer.echo("Cancelled.")
        return 1
    return run_compiled_script(compiled, allow_overwrite=allow_overwrite)


def compile_plan_file(plan_path: Path) -> int:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    output = validate_plan_payload(payload)
    if not isinstance(output, PlanV1):
        raise CompileError("Cannot compile a clarification response")
    compiled = compile_plan(output, python_executable=sys.executable)
    typer.echo(compiled.script)
    return 0


def evaluate_dataset(*, planner_name: str, dataset_path: Path, artifact_dir: Path) -> int:
    planner = load_planner(
        planner_name,
        dataset_path=dataset_path if planner_name == "gold" else None,
        strict=planner_name == "openai",
    )
    results = evaluate_planner(
        planner,
        dataset_path=dataset_path,
        label=dataset_path.name,
        python_executable=sys.executable,
    )
    artifact_path = write_eval_artifact(results, artifact_dir)
    typer.echo(
        json.dumps(
            {
                "dataset": results["dataset_path"],
                "count": results["count"],
                "exact_match_rate": results["exact_match_rate"],
                "compile_valid_rate": results["compile_valid_rate"],
                "slot_accuracy": results["slot_accuracy"],
                "artifact": str(artifact_path),
            },
            indent=2,
        )
    )
    return 0


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


def _sample_probe_records(records: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be at least 1")
    sample_size = min(count, len(records))
    if sample_size == len(records):
        return list(records)
    return random.Random(seed).sample(records, sample_size)


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


def probe_live_dataset(*, dataset_path: Path, count: int, seed: int | None, mode: ProbeMode) -> int:
    records = _load_probe_records(dataset_path)
    if not records:
        typer.echo(f"No probe records found under {dataset_path}", err=True)
        return 1

    effective_seed = seed if seed is not None else random.SystemRandom().randrange(1, 1_000_000_000)
    chosen = _sample_probe_records(records, count=count, seed=effective_seed)
    config = PlannerConfig.from_env()

    exact_matches = 0
    plan_outputs = 0
    clarification_outputs = 0
    compile_valid_plan_outputs = 0
    failures = 0

    typer.echo(f"Probe model: {config.model}")
    typer.echo(f"Probe base URL: {config.base_url}")
    typer.echo(f"Dataset: {dataset_path}")
    typer.echo(f"Mode: {mode.value}")
    typer.echo(f"Seed: {effective_seed}")
    typer.echo(f"Sample count: {len(chosen)}")

    for index, record in enumerate(chosen, start=1):
        source_path = record["_source_path"]
        source_line = record["_source_line"]
        expected = validate_plan_payload(record["plan"])
        expected_dump = normalize_plan(expected)
        expected_assistant = _expected_assistant_content(record)

        typer.echo("")
        typer.echo("=" * 100)
        typer.echo(f"Sample {index}/{len(chosen)}")
        typer.echo(f"Source: {source_path}:{source_line}")
        if record.get("focus"):
            typer.echo(f"Focus: {record['focus']}")
        typer.echo(f"Prompt: {record['prompt']}")

        if mode is ProbeMode.runtime:
            sent_messages = planner_chat_messages(record["prompt"])
            response_format = planner_response_format()
        else:
            try:
                sent_messages = _messages_to_send(record)
            except Exception as exc:
                failures += 1
                typer.echo(f"Message preparation error: {type(exc).__name__}: {exc}", err=True)
                continue
            response_format = None

        typer.echo("Sent messages:")
        typer.echo(json.dumps(sent_messages, indent=2, ensure_ascii=False))
        typer.echo("Expected assistant:")
        typer.echo(expected_assistant)
        typer.echo("Expected normalized:")
        typer.echo(json.dumps(expected_dump, indent=2, ensure_ascii=False))

        try:
            actual_content = chat_completion_text(
                messages=sent_messages,
                config=config,
                response_format=response_format,
                temperature=0,
                max_tokens=600,
            )
        except Exception as exc:
            failures += 1
            typer.echo(f"Transport error: {type(exc).__name__}: {exc}", err=True)
            continue

        typer.echo("Actual assistant:")
        typer.echo(actual_content)

        try:
            actual = validate_planner_payload(actual_content, extract_json_fragment=True)
        except ValidationError as exc:
            failures += 1
            typer.echo(f"Parse error: {exc}", err=True)
            continue

        actual_dump = normalize_plan(actual)
        exact_match = actual_dump == expected_dump
        exact_matches += int(exact_match)

        typer.echo("Actual normalized:")
        typer.echo(json.dumps(actual_dump, indent=2, ensure_ascii=False))
        typer.echo(f"Exact match: {'yes' if exact_match else 'no'}")

        if isinstance(actual, PlanV1):
            plan_outputs += 1
            try:
                compile_plan(actual, python_executable=sys.executable)
                compile_valid = True
                compile_valid_plan_outputs += 1
                compile_error = None
            except CompileError as exc:
                compile_valid = False
                compile_error = str(exc)
            typer.echo(f"Compile valid: {'yes' if compile_valid else 'no'}")
            if compile_error:
                typer.echo(f"Compile error: {compile_error}")
        else:
            clarification_outputs += 1
            typer.echo("Compile valid: n/a (clarification)")

    typer.echo("")
    typer.echo("=" * 100)
    typer.echo("Summary:")
    typer.echo(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "mode": mode.value,
                "seed": effective_seed,
                "count": len(chosen),
                "exact_matches": exact_matches,
                "exact_match_rate": exact_matches / len(chosen),
                "plan_outputs": plan_outputs,
                "clarification_outputs": clarification_outputs,
                "compile_valid_plan_outputs": compile_valid_plan_outputs,
                "compile_valid_plan_rate": (
                    compile_valid_plan_outputs / plan_outputs if plan_outputs else 0.0
                ),
                "transport_or_parse_failures": failures,
                "model": config.model,
                "base_url": config.base_url,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 1 if failures else 0


@app.command("plan")
def plan_command(
    prompt: str = typer.Argument(...),
    planner: str = typer.Option("openai", "--planner"),
    dataset: Path | None = typer.Option(None, "--dataset"),
) -> None:
    code = plan_prompt(prompt, planner_name=planner, dataset_path=dataset)
    if code:
        raise typer.Exit(code=code)


@app.command("run")
def run_command(
    prompt: str = typer.Argument(...),
    planner: str = typer.Option("openai", "--planner"),
    dataset: Path | None = typer.Option(None, "--dataset"),
    yes: bool = typer.Option(False, "--yes"),
    allow_overwrite: bool = typer.Option(False, "--allow-overwrite"),
) -> None:
    code = run_prompt(
        prompt,
        planner_name=planner,
        dataset_path=dataset,
        yes=yes,
        allow_overwrite=allow_overwrite,
    )
    if code:
        raise typer.Exit(code=code)


@app.command("compile")
def compile_command(plan_path: Path = typer.Argument(...)) -> None:
    try:
        code = compile_plan_file(plan_path)
    except CompileError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    if code:
        raise typer.Exit(code=code)


@app.command("eval")
def eval_command(
    planner: str = typer.Option("gold", "--planner"),
    dataset: Path = typer.Option(default_dataset_path(), "--dataset"),
    artifact_dir: Path = typer.Option(Path("artifacts/eval"), "--artifact-dir"),
) -> None:
    code = evaluate_dataset(planner_name=planner, dataset_path=dataset, artifact_dir=artifact_dir)
    if code:
        raise typer.Exit(code=code)


@app.command("probe-live")
def probe_live_command(
    dataset: Path = typer.Option(default_dataset_path(), "--dataset"),
    count: int = typer.Option(5, "--count"),
    seed: int | None = typer.Option(None, "--seed"),
    mode: ProbeMode = typer.Option(ProbeMode.runtime, "--mode"),
) -> None:
    try:
        code = probe_live_dataset(dataset_path=dataset, count=count, seed=seed, mode=mode)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    if code:
        raise typer.Exit(code=code)


def main() -> None:
    app()
