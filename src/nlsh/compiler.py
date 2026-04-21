from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nlsh.schema import (
    CsvToJsonStep,
    FindFilesStep,
    JsonFilterStep,
    JsonGroupCountStep,
    JsonSelectFieldsStep,
    JsonSortStep,
    PdfExtractPagesStep,
    PdfMergeStep,
    PdfSearchTextStep,
    PlanV1,
)


class CompileError(ValueError):
    pass


@dataclass(slots=True)
class CompileResult:
    script: str
    summary: str
    output_files: list[str]


TOOL_PACKAGES = {
    "find": "findutils",
    "jq": "jq",
    "qpdf": "qpdf",
}


JsonTerminalStep = JsonFilterStep | JsonSelectFieldsStep | JsonSortStep | JsonGroupCountStep


def _quote(value: str) -> str:
    return shlex.quote(value)


def _basename_with_suffix(path: str, suffix: str, ext: str | None = None) -> str:
    source = Path(path)
    extension = ext if ext is not None else source.suffix
    return f"{source.stem}{suffix}{extension}"


def _path_expr(path: str) -> str:
    if path.startswith("$") or path.startswith('"$') or path.startswith('"${'):
        return path
    return _quote(path)


def _default_json_output(input_file: str, suffix: str) -> str:
    if input_file.startswith("$") or input_file.startswith('"$') or input_file.startswith('"${'):
        return f"output{suffix}.json"
    return _basename_with_suffix(input_file, suffix, ".json")


def _check_output_lines(output_file: str) -> list[str]:
    quoted = _quote(output_file)
    return [
        f'if [ -e {quoted} ] && [ "${{NLSH_ALLOW_OVERWRITE:-0}}" != "1" ]; then',
        f'  echo "Refusing to overwrite existing file: {output_file}" >&2',
        "  exit 1",
        "fi",
    ]


def _require_single_match_lines(var_name: str, label: str) -> list[str]:
    return [
        f'if [ "${{#{var_name}[@]}}" -eq 0 ]; then',
        f'  echo "No matches found for {label}" >&2',
        "  exit 1",
        "fi",
        f'if [ "${{#{var_name}[@]}}" -ne 1 ]; then',
        f'  echo "Expected exactly one match for {label}, got ${{#{var_name}[@]}}" >&2',
        "  exit 1",
        "fi",
    ]


def _find_command(step: FindFilesStep) -> str:
    parts: list[str] = ["find", _quote(step.root)]
    if step.max_depth is not None:
        parts.extend(["-maxdepth", str(step.max_depth)])
    parts.extend(["-type", "f"])
    if step.glob:
        root_prefix = step.root.rstrip("/") or "."
        parts.extend(["-path", _quote(f"{root_prefix}/{step.glob}")])
    parts.append("-print0")
    return " ".join(parts) + " | sort -z"


def _find_lines(step: FindFilesStep, var_name: str = "MATCHES") -> list[str]:
    command = _find_command(step)
    return [
        f"mapfile -d '' -t {var_name} < <({command})",
        f'if [ "${{#{var_name}[@]}}" -eq 0 ]; then',
        '  echo "No matching files found." >&2',
        "  exit 1",
        "fi",
    ]


def _compile_pdf_merge(step: PdfMergeStep, previous_find: bool) -> tuple[list[str], list[str], str]:
    output_file = step.output_file or "merged.pdf"
    lines = _check_output_lines(output_file)
    if previous_find:
        inputs_expr = '"${MATCHES[@]}"'
    else:
        if not step.input_files:
            raise CompileError("pdf_merge requires input_files unless it follows find_files")
        inputs_expr = " ".join(_quote(item) for item in step.input_files)
    lines.append(f"qpdf --empty --pages {inputs_expr} -- {_quote(output_file)}")
    return lines, [output_file], f"Merge PDFs into {output_file}"


def _compile_pdf_extract_pages(
    step: PdfExtractPagesStep,
    previous_find: bool,
) -> tuple[list[str], list[str], str]:
    if step.page_start is None or step.page_end is None:
        raise CompileError("pdf_extract_pages requires page_start and page_end")
    if previous_find:
        lines = _require_single_match_lines("MATCHES", "pdf input")
        input_expr = '"${MATCHES[0]}"'
        source_for_name = "input.pdf"
    else:
        if not step.input_file:
            raise CompileError("pdf_extract_pages requires input_file")
        lines = []
        input_expr = _quote(step.input_file)
        source_for_name = step.input_file
    output_file = step.output_file or _basename_with_suffix(
        source_for_name,
        f"_pages_{step.page_start}_{step.page_end}",
        ".pdf",
    )
    lines.extend(_check_output_lines(output_file))
    lines.append(f"qpdf {input_expr} --pages . {step.page_start}-{step.page_end} -- {_quote(output_file)}")
    return lines, [output_file], f"Extract PDF pages into {output_file}"


def _compile_pdf_search_text(
    step: PdfSearchTextStep,
    previous_find: bool,
    *,
    python_executable: str,
) -> tuple[list[str], list[str], str]:
    if not step.query:
        raise CompileError("pdf_search_text requires query")
    output_file = step.output_file or "pdf_search_matches.json"
    lines = _check_output_lines(output_file)
    if previous_find:
        inputs_expr = '"${MATCHES[@]}"'
    else:
        if not step.input_files:
            raise CompileError("pdf_search_text requires input_files unless it follows find_files")
        inputs_expr = " ".join(_quote(item) for item in step.input_files)
    lines.append(
        f"{_quote(python_executable)} -m nlsh.pdf_search "
        f"--query {_quote(step.query)} "
        f"--context-chars {step.context_chars} "
        f"--output {_quote(output_file)} "
        f"{inputs_expr}"
    )
    return lines, [output_file], f"Search PDF text into {output_file}"


def _compile_csv_to_json(
    step: CsvToJsonStep,
    previous_find: bool,
    *,
    python_executable: str,
    output_file: str | None = None,
    check_output: bool = True,
) -> tuple[list[str], list[str], str]:
    if previous_find:
        lines = _require_single_match_lines("MATCHES", "csv input")
        input_expr = '"${MATCHES[0]}"'
        source_for_name = "input.csv"
    else:
        if not step.input_file:
            raise CompileError("csv_to_json requires input_file")
        lines = []
        input_expr = _quote(step.input_file)
        source_for_name = step.input_file
    target = output_file or step.output_file or _basename_with_suffix(source_for_name, "", ".json")
    if check_output:
        lines.extend(_check_output_lines(target))
    lines.append(f"{_quote(python_executable)} -m nlsh.csv_to_json {input_expr} > {_path_expr(target)}")
    return lines, [target], f"Convert CSV to JSON in {target}"


def _jq_value_arg(value: str | int | float | bool) -> str:
    return _quote(json.dumps(value))


def _compile_json_filter(step: JsonFilterStep, input_file: str) -> tuple[list[str], list[str], str]:
    if step.field is None or step.operator is None or step.value is None:
        raise CompileError("json_filter requires field, operator, and value")
    output_file = step.output_file or _default_json_output(input_file, "_filtered")
    is_number = isinstance(step.value, (int, float)) and not isinstance(step.value, bool)
    if step.operator in {"gt", "gte", "lt", "lte"} and is_number:
        field_expr = "(.[$field] | tonumber?)"
    else:
        field_expr = ".[$field]"
    operator_expr = {
        "eq": ".[$field] == $value",
        "ne": ".[$field] != $value",
        "gt": f"{field_expr} > $value",
        "gte": f"{field_expr} >= $value",
        "lt": f"{field_expr} < $value",
        "lte": f"{field_expr} <= $value",
        "contains": "(.[$field] | tostring | contains($value | tostring))",
    }[step.operator]
    expression = f"map(select({operator_expr}))"
    lines = _check_output_lines(output_file)
    lines.append(
        "jq "
        f"--arg field {_quote(step.field)} "
        f"--argjson value {_jq_value_arg(step.value)} "
        f"{_quote(expression)} {_path_expr(input_file)} > {_quote(output_file)}"
    )
    return lines, [output_file], f"Filter JSON into {output_file}"


def _compile_json_select_fields(step: JsonSelectFieldsStep, input_file: str) -> tuple[list[str], list[str], str]:
    if not step.fields:
        raise CompileError("json_select_fields requires fields")
    output_file = step.output_file or _default_json_output(input_file, "_selected")
    expression = 'map(. as $row | reduce $fields[] as $field ({}; . + {($field): $row[$field]}))'
    lines = _check_output_lines(output_file)
    lines.append(
        "jq "
        f"--argjson fields {_quote(json.dumps(step.fields))} "
        f"{_quote(expression)} {_path_expr(input_file)} > {_quote(output_file)}"
    )
    return lines, [output_file], f"Select JSON fields into {output_file}"


def _compile_json_sort(step: JsonSortStep, input_file: str) -> tuple[list[str], list[str], str]:
    if not step.field:
        raise CompileError("json_sort requires field")
    output_file = step.output_file or _default_json_output(input_file, "_sorted")
    expression = "sort_by(.[$field])"
    if step.descending:
        expression = f"{expression} | reverse"
    lines = _check_output_lines(output_file)
    lines.append(
        "jq "
        f"--arg field {_quote(step.field)} "
        f"{_quote(expression)} {_path_expr(input_file)} > {_quote(output_file)}"
    )
    return lines, [output_file], f"Sort JSON into {output_file}"


def _compile_json_group_count(step: JsonGroupCountStep, input_file: str) -> tuple[list[str], list[str], str]:
    if not step.group_by:
        raise CompileError("json_group_count requires group_by")
    output_file = step.output_file or _default_json_output(input_file, "_counts")
    expression = (
        "group_by([.[$fields[]]]) | "
        "map(.[0] as $first | "
        "(reduce $fields[] as $field ({}; . + {($field): $first[$field]})) + "
        "{($count_field): length})"
    )
    lines = _check_output_lines(output_file)
    lines.append(
        "jq "
        f"--argjson fields {_quote(json.dumps(step.group_by))} "
        f"--arg count_field {_quote(step.count_field)} "
        f"{_quote(expression)} {_path_expr(input_file)} > {_quote(output_file)}"
    )
    return lines, [output_file], f"Count JSON groups into {output_file}"


def _compile_json_terminal(step: JsonTerminalStep, input_file: str) -> tuple[list[str], list[str], str]:
    if isinstance(step, JsonFilterStep):
        return _compile_json_filter(step, input_file)
    if isinstance(step, JsonSelectFieldsStep):
        return _compile_json_select_fields(step, input_file)
    if isinstance(step, JsonSortStep):
        return _compile_json_sort(step, input_file)
    if isinstance(step, JsonGroupCountStep):
        return _compile_json_group_count(step, input_file)
    raise CompileError(f"Unsupported JSON step: {type(step)!r}")


def _terminal_input_file(step: JsonTerminalStep) -> str | None:
    return step.input_file


def summarize_outputs(outputs: Iterable[str]) -> list[str]:
    return [str(Path(item)) for item in outputs]


def required_tools_for_plan(plan: PlanV1) -> list[str]:
    tools: set[str] = set()
    for step in plan.steps:
        if isinstance(step, FindFilesStep):
            tools.add("find")
        elif isinstance(step, (PdfMergeStep, PdfExtractPagesStep)):
            tools.add("qpdf")
        elif isinstance(step, (JsonFilterStep, JsonSelectFieldsStep, JsonSortStep, JsonGroupCountStep)):
            tools.add("jq")
    return sorted(tools)


def compile_plan(plan: PlanV1, python_executable: str = "python") -> CompileResult:
    lines = ["set -euo pipefail"]
    output_files: list[str] = []

    if len(plan.steps) == 1 and isinstance(plan.steps[0], FindFilesStep):
        lines.extend(_find_lines(plan.steps[0]))
        lines.append('printf "%s\\n" "${MATCHES[@]}"')
        return CompileResult("\n".join(lines) + "\n", "Find matching paths", [])

    steps = plan.steps
    previous_find = False
    if isinstance(steps[0], FindFilesStep):
        lines.extend(_find_lines(steps[0]))
        previous_find = True
        steps = steps[1:]

    if len(steps) == 1:
        step = steps[0]
        if isinstance(step, PdfMergeStep):
            terminal_lines, outputs, summary = _compile_pdf_merge(step, previous_find)
        elif isinstance(step, PdfExtractPagesStep):
            terminal_lines, outputs, summary = _compile_pdf_extract_pages(step, previous_find)
        elif isinstance(step, PdfSearchTextStep):
            terminal_lines, outputs, summary = _compile_pdf_search_text(
                step,
                previous_find,
                python_executable=python_executable,
            )
        elif isinstance(step, CsvToJsonStep):
            terminal_lines, outputs, summary = _compile_csv_to_json(
                step,
                previous_find,
                python_executable=python_executable,
            )
        elif isinstance(step, (JsonFilterStep, JsonSelectFieldsStep, JsonSortStep, JsonGroupCountStep)):
            if previous_find:
                terminal_lines = _require_single_match_lines("MATCHES", "json input")
                input_file = '"${MATCHES[0]}"'
                json_lines, outputs, summary = _compile_json_terminal(step, input_file)
                terminal_lines.extend(json_lines)
            else:
                input_file = _terminal_input_file(step)
                if not input_file:
                    raise CompileError(f"{step.kind} requires input_file")
                terminal_lines, outputs, summary = _compile_json_terminal(step, input_file)
        else:
            raise CompileError(f"Unsupported step kind: {type(step)!r}")
        lines.extend(terminal_lines)
        output_files.extend(outputs)
        return CompileResult("\n".join(lines) + "\n", summary, summarize_outputs(output_files))

    if len(steps) == 2 and isinstance(steps[0], CsvToJsonStep) and isinstance(
        steps[1],
        (JsonFilterStep, JsonSelectFieldsStep, JsonSortStep, JsonGroupCountStep),
    ):
        csv_step = steps[0]
        json_step = steps[1]
        if csv_step.output_file:
            intermediate = csv_step.output_file
            csv_lines, csv_outputs, _ = _compile_csv_to_json(
                csv_step,
                previous_find,
                python_executable=python_executable,
                output_file=intermediate,
                check_output=True,
            )
            output_files.extend(csv_outputs)
        else:
            lines.extend([
                'NLSH_TMP_JSON="$(mktemp --suffix=.json)"',
                'trap \'rm -f "$NLSH_TMP_JSON"\' EXIT',
            ])
            intermediate = "$NLSH_TMP_JSON"
            csv_lines, _, _ = _compile_csv_to_json(
                csv_step,
                previous_find,
                python_executable=python_executable,
                output_file=intermediate,
                check_output=False,
            )
        lines.extend(csv_lines)
        json_lines, json_outputs, summary = _compile_json_terminal(json_step, intermediate)
        lines.extend(json_lines)
        output_files.extend(json_outputs)
        return CompileResult("\n".join(lines) + "\n", summary, summarize_outputs(output_files))

    raise CompileError("Unsupported plan shape")
