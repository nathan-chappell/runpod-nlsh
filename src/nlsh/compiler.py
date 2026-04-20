from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nlsh.schema import (
    CsvFilterRowsStep,
    CsvGroupCountStep,
    CsvJoinStep,
    CsvSelectColumnsStep,
    CsvSortRowsStep,
    FindFilesStep,
    MediaClipStep,
    MediaExtractAudioMp3Step,
    MediaTranscodeForTvStep,
    PdfCombineStep,
    PdfCompressStep,
    PdfExtractPagesStep,
    PdfRotateStep,
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
    "ffmpeg": "ffmpeg",
    "find": "findutils",
    "gs": "ghostscript",
    "mlr": "miller",
    "qpdf": "qpdf",
}


def _quote(value: str) -> str:
    return shlex.quote(value)


def _basename_with_suffix(path: str, suffix: str, ext: str | None = None) -> str:
    source = Path(path)
    extension = ext if ext is not None else source.suffix
    return f"{source.stem}{suffix}{extension}"


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
        f'if [ "${{{var_name}[@]}}" = "" ]; then',
        f'  echo "No matches found for {label}" >&2',
        "  exit 1",
        "fi",
        f'if [ "${{#{var_name}[@]}}" -ne 1 ]; then',
        f'  echo "Expected exactly one match for {label}, got ${{#{var_name}[@]}}" >&2',
        "  exit 1",
        "fi",
    ]


def _require_no_commas(values: list[str], label: str) -> None:
    for value in values:
        if "," in value:
            raise CompileError(f"{label} values cannot contain commas for shell compilation: {value!r}")


def _mlr_field_list(values: list[str], label: str) -> str:
    _require_no_commas(values, label)
    return ",".join(values)


def _mlr_field_ref(field_name: str) -> str:
    return f"$[{json.dumps(field_name)}]"


def _mlr_value_literal(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def _mlr_filter_expression(step: CsvFilterRowsStep) -> str:
    if step.filter_column is None or step.filter_operator is None or step.filter_value is None:
        raise CompileError("csv_filter_rows requires filter_column, filter_operator, and filter_value")
    field_ref = _mlr_field_ref(step.filter_column)
    if step.filter_operator == "contains":
        pattern = re.escape(str(step.filter_value))
        return f"{field_ref} =~ {json.dumps(pattern)}"

    operator_map = {
        "eq": "==",
        "ne": "!=",
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
    }
    operator = operator_map[step.filter_operator]
    literal = _mlr_value_literal(step.filter_value)
    return f"{field_ref} {operator} {literal}"


def _find_command(step: FindFilesStep) -> str:
    parts: list[str] = ["find"]
    parts.extend(_quote(root) for root in step.roots)
    if step.max_depth is not None:
        parts.extend(["-maxdepth", str(step.max_depth)])
    parts.extend(["-type", "d" if step.file_type == "directory" else "f"])
    if step.name_pattern:
        parts.extend(["-name", _quote(step.name_pattern)])
    if step.extension:
        normalized_ext = step.extension if step.extension.startswith(".") else f".{step.extension}"
        parts.extend(["-name", _quote(f"*{normalized_ext}")])
    if step.path_contains:
        parts.extend(["-path", _quote(f"*{step.path_contains}*")])
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


def _compile_pdf_combine(
    step: PdfCombineStep,
    previous_find: bool,
) -> tuple[list[str], list[str], str]:
    output_file = step.output_file or "combined.pdf"
    lines = _check_output_lines(output_file)
    if previous_find:
        inputs_expr = '"${MATCHES[@]}"'
    else:
        if not step.input_files:
            raise CompileError("pdf_combine requires input_files unless it follows find_files")
        inputs_expr = " ".join(_quote(item) for item in step.input_files)
    lines.append(
        f"qpdf --empty --pages {inputs_expr} -- {_quote(output_file)}"
    )
    return lines, [output_file], f"Combine PDFs into {output_file}"


def _compile_pdf_compress(
    step: PdfCompressStep,
    previous_find: bool,
) -> tuple[list[str], list[str], str]:
    if previous_find:
        lines = _require_single_match_lines("MATCHES", "pdf input")
        input_expr = '"${MATCHES[0]}"'
        input_name = "matched input PDF"
        source_for_name = "input.pdf"
    else:
        if not step.input_file:
            raise CompileError("pdf_compress requires input_file")
        lines = []
        input_expr = _quote(step.input_file)
        input_name = step.input_file
        source_for_name = step.input_file
    output_file = step.output_file or _basename_with_suffix(source_for_name, "_compressed", ".pdf")
    lines.extend(_check_output_lines(output_file))
    lines.append(
        "gs -q -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 "
        "-dPDFSETTINGS=/ebook "
        f"-sOutputFile={_quote(output_file)} {input_expr}"
    )
    return lines, [output_file], f"Compress {input_name} into {output_file}"


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
    page_range = f"{step.page_start}-{step.page_end}"
    lines.append(f"qpdf {input_expr} --pages . {page_range} -- {_quote(output_file)}")
    return lines, [output_file], f"Extract pages {step.page_start}-{step.page_end} into {output_file}"


def _compile_pdf_rotate(
    step: PdfRotateStep,
    previous_find: bool,
) -> tuple[list[str], list[str], str]:
    if step.rotation_degrees is None:
        raise CompileError("pdf_rotate requires rotation_degrees")
    if previous_find:
        lines = _require_single_match_lines("MATCHES", "pdf input")
        input_expr = '"${MATCHES[0]}"'
        source_for_name = "input.pdf"
    else:
        if not step.input_file:
            raise CompileError("pdf_rotate requires input_file")
        lines = []
        input_expr = _quote(step.input_file)
        source_for_name = step.input_file
    output_file = step.output_file or _basename_with_suffix(source_for_name, "_rotated", ".pdf")
    lines.extend(_check_output_lines(output_file))
    lines.append(
        f"qpdf {input_expr} --rotate=+{step.rotation_degrees}:1-z -- {_quote(output_file)}"
    )
    return lines, [output_file], f"Rotate PDF into {output_file}"


def _compile_media_transcode(
    step: MediaTranscodeForTvStep,
    previous_find: bool,
) -> tuple[list[str], list[str], str]:
    if previous_find:
        lines = _require_single_match_lines("MATCHES", "media input")
        input_expr = '"${MATCHES[0]}"'
        source_for_name = "input.mp4"
    else:
        if not step.input_file:
            raise CompileError("media_transcode_for_tv requires input_file")
        lines = []
        input_expr = _quote(step.input_file)
        source_for_name = step.input_file
    output_file = step.output_file or _basename_with_suffix(source_for_name, "_tv", ".mp4")
    lines.extend(_check_output_lines(output_file))
    lines.append(
        "ffmpeg -hide_banner -loglevel error -nostdin -n "
        f"-i {input_expr} -c:v libx264 -preset medium -crf 23 "
        f"-c:a aac -b:a 192k -movflags +faststart {_quote(output_file)}"
    )
    return lines, [output_file], f"Transcode media for TV playback into {output_file}"


def _compile_media_extract_audio(
    step: MediaExtractAudioMp3Step,
    previous_find: bool,
) -> tuple[list[str], list[str], str]:
    if previous_find:
        lines = _require_single_match_lines("MATCHES", "media input")
        input_expr = '"${MATCHES[0]}"'
        source_for_name = "input.mp4"
    else:
        if not step.input_file:
            raise CompileError("media_extract_audio_mp3 requires input_file")
        lines = []
        input_expr = _quote(step.input_file)
        source_for_name = step.input_file
    output_file = step.output_file or _basename_with_suffix(source_for_name, "", ".mp3")
    lines.extend(_check_output_lines(output_file))
    lines.append(
        "ffmpeg -hide_banner -loglevel error -nostdin -n "
        f"-i {input_expr} -vn -codec:a libmp3lame -qscale:a 2 {_quote(output_file)}"
    )
    return lines, [output_file], f"Extract MP3 audio into {output_file}"


def _compile_media_clip(
    step: MediaClipStep,
    previous_find: bool,
) -> tuple[list[str], list[str], str]:
    if step.start_seconds is None or step.duration_seconds is None:
        raise CompileError("media_clip requires start_seconds and duration_seconds")
    if previous_find:
        lines = _require_single_match_lines("MATCHES", "media input")
        input_expr = '"${MATCHES[0]}"'
        source_for_name = "input.mp4"
    else:
        if not step.input_file:
            raise CompileError("media_clip requires input_file")
        lines = []
        input_expr = _quote(step.input_file)
        source_for_name = step.input_file
    output_file = step.output_file or _basename_with_suffix(source_for_name, "_clip", ".mp4")
    lines.extend(_check_output_lines(output_file))
    lines.append(
        "ffmpeg -hide_banner -loglevel error -nostdin -n "
        f"-ss {step.start_seconds} -i {input_expr} -t {step.duration_seconds} "
        f"-c copy {_quote(output_file)}"
    )
    return lines, [output_file], f"Clip media into {output_file}"


def _compile_csv_join(
    step: CsvJoinStep,
    previous_find: bool,
) -> tuple[list[str], list[str], str]:
    if not step.join_keys:
        raise CompileError("csv_join requires join_keys")
    if previous_find:
        lines = [
            'if [ "${#MATCHES[@]}" -ne 2 ]; then',
            '  echo "csv_join expects exactly 2 matches from find_files." >&2',
            "  exit 1",
            "fi",
        ]
        left_expr = '"${MATCHES[0]}"'
        right_expr = '"${MATCHES[1]}"'
        source_for_name = "joined.csv"
    else:
        if not step.left_file or not step.right_file:
            raise CompileError("csv_join requires left_file and right_file")
        lines = []
        left_expr = _quote(step.left_file)
        right_expr = _quote(step.right_file)
        source_for_name = step.left_file
    output_file = step.output_file or _basename_with_suffix(source_for_name, "_joined", ".csv")
    lines.extend(_check_output_lines(output_file))
    join_keys = _mlr_field_list(step.join_keys, "join_keys")
    join_modifiers = {
        "inner": "",
        "left": " --ul",
        "right": " --ur",
        "outer": " --ul --ur",
    }[step.how]
    lines.append(
        f"mlr --csv join -f {left_expr} -j {_quote(join_keys)}{join_modifiers} {right_expr} > {_quote(output_file)}"
    )
    return lines, [output_file], f"Join CSV files into {output_file}"


def _compile_csv_filter(
    step: CsvFilterRowsStep,
    previous_find: bool,
) -> tuple[list[str], list[str], str]:
    if step.filter_column is None or step.filter_operator is None or step.filter_value is None:
        raise CompileError("csv_filter_rows requires filter_column, filter_operator, and filter_value")
    if previous_find:
        lines = _require_single_match_lines("MATCHES", "csv input")
        input_expr = '"${MATCHES[0]}"'
        source_for_name = "input.csv"
    else:
        if not step.input_file:
            raise CompileError("csv_filter_rows requires input_file")
        lines = []
        input_expr = _quote(step.input_file)
        source_for_name = step.input_file
    output_file = step.output_file or _basename_with_suffix(source_for_name, "_filtered", ".csv")
    lines.extend(_check_output_lines(output_file))
    expression = _mlr_filter_expression(step)
    lines.append(
        f"mlr --csv filter {_quote(expression)} {input_expr} > {_quote(output_file)}"
    )
    return lines, [output_file], f"Filter CSV rows into {output_file}"


def _compile_csv_select(
    step: CsvSelectColumnsStep,
    previous_find: bool,
) -> tuple[list[str], list[str], str]:
    if not step.columns:
        raise CompileError("csv_select_columns requires columns")
    if previous_find:
        lines = _require_single_match_lines("MATCHES", "csv input")
        input_expr = '"${MATCHES[0]}"'
        source_for_name = "input.csv"
    else:
        if not step.input_file:
            raise CompileError("csv_select_columns requires input_file")
        lines = []
        input_expr = _quote(step.input_file)
        source_for_name = step.input_file
    output_file = step.output_file or _basename_with_suffix(source_for_name, "_selected", ".csv")
    lines.extend(_check_output_lines(output_file))
    columns = _mlr_field_list(step.columns, "columns")
    lines.append(
        f"mlr --csv cut -o -f {_quote(columns)} {input_expr} > {_quote(output_file)}"
    )
    return lines, [output_file], f"Select CSV columns into {output_file}"


def _compile_csv_sort(
    step: CsvSortRowsStep,
    previous_find: bool,
) -> tuple[list[str], list[str], str]:
    if not step.sort_by:
        raise CompileError("csv_sort_rows requires sort_by")
    if previous_find:
        lines = _require_single_match_lines("MATCHES", "csv input")
        input_expr = '"${MATCHES[0]}"'
        source_for_name = "input.csv"
    else:
        if not step.input_file:
            raise CompileError("csv_sort_rows requires input_file")
        lines = []
        input_expr = _quote(step.input_file)
        source_for_name = step.input_file
    output_file = step.output_file or _basename_with_suffix(source_for_name, "_sorted", ".csv")
    lines.extend(_check_output_lines(output_file))
    sort_fields = _mlr_field_list(step.sort_by, "sort_by")
    sort_flag = "-r" if step.descending else "-f"
    lines.append(
        f"mlr --csv sort {sort_flag} {_quote(sort_fields)} {input_expr} > {_quote(output_file)}"
    )
    return lines, [output_file], f"Sort CSV rows into {output_file}"


def _compile_csv_group_count(
    step: CsvGroupCountStep,
    previous_find: bool,
) -> tuple[list[str], list[str], str]:
    if not step.group_by:
        raise CompileError("csv_group_count requires group_by")
    if previous_find:
        lines = _require_single_match_lines("MATCHES", "csv input")
        input_expr = '"${MATCHES[0]}"'
        source_for_name = "input.csv"
    else:
        if not step.input_file:
            raise CompileError("csv_group_count requires input_file")
        lines = []
        input_expr = _quote(step.input_file)
        source_for_name = step.input_file
    output_file = step.output_file or _basename_with_suffix(source_for_name, "_group_count", ".csv")
    lines.extend(_check_output_lines(output_file))
    group_fields = _mlr_field_list(step.group_by, "group_by")
    lines.append(
        f"mlr --csv count -g {_quote(group_fields)} -o {_quote(step.count_column)} {input_expr} > {_quote(output_file)}"
    )
    return lines, [output_file], f"Group and count CSV rows into {output_file}"


def _compile_terminal(step: object, previous_find: bool) -> tuple[list[str], list[str], str]:
    if isinstance(step, PdfCombineStep):
        return _compile_pdf_combine(step, previous_find)
    if isinstance(step, PdfCompressStep):
        return _compile_pdf_compress(step, previous_find)
    if isinstance(step, PdfExtractPagesStep):
        return _compile_pdf_extract_pages(step, previous_find)
    if isinstance(step, PdfRotateStep):
        return _compile_pdf_rotate(step, previous_find)
    if isinstance(step, MediaTranscodeForTvStep):
        return _compile_media_transcode(step, previous_find)
    if isinstance(step, MediaExtractAudioMp3Step):
        return _compile_media_extract_audio(step, previous_find)
    if isinstance(step, MediaClipStep):
        return _compile_media_clip(step, previous_find)
    if isinstance(step, CsvJoinStep):
        return _compile_csv_join(step, previous_find)
    if isinstance(step, CsvFilterRowsStep):
        return _compile_csv_filter(step, previous_find)
    if isinstance(step, CsvSelectColumnsStep):
        return _compile_csv_select(step, previous_find)
    if isinstance(step, CsvSortRowsStep):
        return _compile_csv_sort(step, previous_find)
    if isinstance(step, CsvGroupCountStep):
        return _compile_csv_group_count(step, previous_find)
    raise CompileError(f"Unsupported step kind: {type(step)!r}")


def summarize_outputs(outputs: Iterable[str]) -> list[str]:
    return [str(Path(item)) for item in outputs]


def required_tools_for_plan(plan: PlanV1) -> list[str]:
    tools: set[str] = set()
    for step in plan.steps:
        if isinstance(step, FindFilesStep):
            tools.add("find")
        elif isinstance(step, (PdfCombineStep, PdfExtractPagesStep, PdfRotateStep)):
            tools.add("qpdf")
        elif isinstance(step, PdfCompressStep):
            tools.add("gs")
        elif isinstance(step, (MediaTranscodeForTvStep, MediaExtractAudioMp3Step, MediaClipStep)):
            tools.add("ffmpeg")
        elif isinstance(step, (
            CsvJoinStep,
            CsvFilterRowsStep,
            CsvSelectColumnsStep,
            CsvSortRowsStep,
            CsvGroupCountStep,
        )):
            tools.add("mlr")
        else:
            raise CompileError(f"Unsupported step kind: {type(step)!r}")
    return sorted(tools)


def compile_plan(plan: PlanV1, python_executable: str = "python") -> CompileResult:
    lines = ["set -euo pipefail"]
    output_files: list[str] = []

    if len(plan.steps) == 1 and isinstance(plan.steps[0], FindFilesStep):
        lines.extend(_find_lines(plan.steps[0]))
        lines.append('printf "%s\\n" "${MATCHES[@]}"')
        return CompileResult(
            script="\n".join(lines) + "\n",
            summary="Find matching paths",
            output_files=[],
        )

    previous_find = False
    terminal_step = plan.steps[0]
    if len(plan.steps) == 2:
        if not isinstance(plan.steps[0], FindFilesStep):
            raise CompileError("2-step plans must start with find_files")
        lines.extend(_find_lines(plan.steps[0]))
        previous_find = True
        terminal_step = plan.steps[1]

    terminal_lines, outputs, summary = _compile_terminal(
        terminal_step,
        previous_find=previous_find,
    )
    lines.extend(terminal_lines)
    output_files.extend(outputs)
    return CompileResult(
        script="\n".join(lines) + "\n",
        summary=summary,
        output_files=summarize_outputs(output_files),
    )
