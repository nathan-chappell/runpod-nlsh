from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nlsh.compiler import CompileError, compile_plan, required_tools_for_plan
from nlsh.dataio import default_split_path
from nlsh.eval import evaluate_planner, write_eval_artifact
from nlsh.planner import load_planner, plan_to_pretty_json
from nlsh.preflight import MissingToolsError, ensure_required_tools
from nlsh.runner import confirm_execution, prepare_plan_for_execution, print_compile_preview, run_compiled_script
from nlsh.schema import Clarification, PlanV1, validate_plan_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nlsh")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("prompt")
    plan_parser.add_argument("--planner", choices=["openai", "gold"], default="openai")
    plan_parser.add_argument("--dataset", type=Path)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("prompt")
    run_parser.add_argument("--planner", choices=["openai", "gold"], default="openai")
    run_parser.add_argument("--dataset", type=Path)
    run_parser.add_argument("--yes", action="store_true")
    run_parser.add_argument("--allow-overwrite", action="store_true")

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("plan_path", type=Path)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--planner", choices=["openai", "gold"], default="gold")
    eval_parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    eval_parser.add_argument("--dataset", type=Path)
    eval_parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/eval"))

    return parser


def command_plan(args: argparse.Namespace) -> int:
    planner = load_planner(args.planner, dataset_path=args.dataset)
    output = planner.plan(args.prompt)
    print(plan_to_pretty_json(output))
    if isinstance(output, Clarification):
        return 0
    try:
        compiled = compile_plan(output, python_executable=sys.executable)
    except CompileError:
        return 0
    print()
    print("# Compiled preview")
    print(compiled.script)
    return 0


def command_run(args: argparse.Namespace) -> int:
    planner = load_planner(args.planner, dataset_path=args.dataset)
    output = planner.plan(args.prompt)
    if isinstance(output, Clarification):
        print(output.question)
        return 1
    plan, compiled = prepare_plan_for_execution(output, python_executable=sys.executable)
    try:
        ensure_required_tools(required_tools_for_plan(plan))
    except MissingToolsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print_compile_preview(plan, compiled)
    if not args.yes and not confirm_execution():
        print("Cancelled.")
        return 1
    return run_compiled_script(compiled, allow_overwrite=args.allow_overwrite)


def command_compile(args: argparse.Namespace) -> int:
    payload = json.loads(args.plan_path.read_text(encoding="utf-8"))
    output = validate_plan_payload(payload)
    if not isinstance(output, PlanV1):
        raise CompileError("Cannot compile a clarification response")
    plan = output
    compiled = compile_plan(plan, python_executable=sys.executable)
    print(compiled.script)
    return 0


def command_eval(args: argparse.Namespace) -> int:
    dataset_path = args.dataset or default_split_path(args.split)
    planner = load_planner(args.planner, dataset_path=dataset_path if args.planner == "gold" else None)
    results = evaluate_planner(
        planner,
        split=args.split,
        dataset_path=dataset_path,
        python_executable=sys.executable,
    )
    artifact_path = write_eval_artifact(results, args.artifact_dir)
    print(json.dumps({
        "split": results["split"],
        "count": results["count"],
        "exact_match_rate": results["exact_match_rate"],
        "compile_valid_rate": results["compile_valid_rate"],
        "slot_accuracy": results["slot_accuracy"],
        "artifact": str(artifact_path),
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        return command_plan(args)
    if args.command == "run":
        return command_run(args)
    if args.command == "compile":
        return command_compile(args)
    if args.command == "eval":
        return command_eval(args)
    parser.error(f"Unknown command: {args.command}")
    return 2
