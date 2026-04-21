from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from nlsh.dataio import default_split_path, load_jsonl
from nlsh.prompts import REPAIR_DEVELOPER_PROMPT, TRAINING_DEVELOPER_PROMPT
from nlsh.schema import PlanV1, plan_json_schema, validate_plan_payload, validation_error_text
from pydantic import ValidationError

STEP_KINDS = {
    "find_files",
    "pdf_combine",
    "pdf_compress",
    "pdf_extract_pages",
    "pdf_rotate",
    "media_transcode_for_tv",
    "media_extract_audio_mp3",
    "media_clip",
    "csv_join",
    "csv_filter_rows",
    "csv_select_columns",
    "csv_sort_rows",
    "csv_group_count",
}

FIELD_NAMES = {"extension", "name_pattern", "path_contains", "max_depth", "file_type", "roots"}
PLAN_LEVEL_KEYS = {"needs_confirmation", "questions", "risk_level", "notes", "version"}
MEDIA_INPUT_QUESTION_PROMPTS = {
    "media_transcode_for_tv": "Which media file should be transcoded for TV playback?",
    "media_extract_audio_mp3": "Which media file should have audio extracted to MP3?",
    "media_clip": "Which media file should be clipped?",
}
MEDIA_INPUT_NOTES = {
    "media_transcode_for_tv": "The source media file is missing.",
    "media_extract_audio_mp3": "The source media file is missing.",
    "media_clip": "The source media file is missing.",
}
CSV_JOIN_KEYS_PROMPT = "Which column or columns should be used as join keys? Separate multiple keys with commas."
CSV_JOIN_KEYS_NOTE = "A join key is required before compilation."
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


class Planner(Protocol):
    def plan(self, prompt: str) -> PlanV1:
        ...


@dataclass(slots=True)
class PlannerConfig:
    model: str
    base_url: str
    api_key: str

    @classmethod
    def from_env(cls) -> "PlannerConfig":
        api_key = (
            os.environ.get("NLSH_API_KEY")
            or os.environ.get("HF_TOKEN")
            or ""
        )
        return cls(
            model=os.environ.get("NLSH_MODEL", "openai/gpt-oss-20b:together"),
            base_url=os.environ.get("NLSH_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=api_key,
        )

    def is_vllm_like(self) -> bool:
        lowered = self.base_url.lower()
        return (
            "runpod.ai" in lowered
            or "/openai/v1" in lowered
            or "vllm" in lowered
        )


def _extract_message_text(message: object) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
                continue
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if not lines:
        return stripped
    if lines[0].startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_fragment(text: str) -> str:
    stripped = _strip_code_fences(text)
    if not stripped:
        return stripped

    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    start: int | None = None
    stack: list[str] = []
    in_string = False
    escape = False
    closing_for = {"{": "}", "[": "]"}

    for index, char in enumerate(stripped):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char in closing_for:
            if start is None:
                start = index
            stack.append(closing_for[char])
            continue
        if char in ("}", "]") and stack:
            if char != stack[-1]:
                continue
            stack.pop()
            if start is not None and not stack:
                return stripped[start : index + 1]

    return stripped


def _infer_step_kind(step: dict[str, Any]) -> str | None:
    if any(key in step for key in ("roots", "extension", "path_contains", "max_depth", "file_type", "path", "depth", "pattern")):
        return "find_files"
    if "input_files" in step:
        return "pdf_combine"
    if "page_start" in step or "page_end" in step:
        return "pdf_extract_pages"
    if "rotation_degrees" in step:
        return "pdf_rotate"
    if any(key in step for key in ("left_file", "right_file", "join_keys")):
        return "csv_join"
    if any(key in step for key in ("filter_column", "filter_operator", "filter_value")):
        return "csv_filter_rows"
    if "columns" in step:
        return "csv_select_columns"
    if "sort_by" in step or "descending" in step:
        return "csv_sort_rows"
    if "group_by" in step or "count_column" in step:
        return "csv_group_count"
    if "start_seconds" in step or "duration_seconds" in step:
        return "media_clip"

    for key, value in step.items():
        if isinstance(value, str) and value in STEP_KINDS:
            return value
    return None


def _coerce_find_files_step(step: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(step)
    if coerced.get("name_pattern") == "find_files":
        coerced.pop("name_pattern", None)

    file_type = coerced.get("file_type")
    if file_type in {"csv", "pdf"}:
        coerced.setdefault("extension", f".{file_type}")
        coerced["file_type"] = "file"

    path = coerced.pop("path", None)
    if path is not None and "roots" not in coerced:
        coerced["roots"] = [path] if isinstance(path, str) else path

    depth = coerced.pop("depth", None)
    if depth is not None and "max_depth" not in coerced:
        coerced["max_depth"] = depth

    pattern = coerced.pop("pattern", None)
    if pattern is not None and "name_pattern" not in coerced and "extension" not in coerced:
        coerced["name_pattern"] = pattern

    return coerced


def _coerce_step_payload(step: Any) -> Any:
    if not isinstance(step, dict):
        return step

    coerced = {key: value for key, value in step.items() if key not in PLAN_LEVEL_KEYS}
    if "kind" not in coerced:
        nested_kind = next(
            (key for key, value in coerced.items() if key in STEP_KINDS and isinstance(value, dict)),
            None,
        )
        if nested_kind is not None:
            nested_payload = dict(coerced.pop(nested_kind))
            coerced = {"kind": nested_kind, **nested_payload, **coerced}
        else:
            inferred_kind = _infer_step_kind(coerced)
            if inferred_kind is not None:
                coerced["kind"] = inferred_kind

    if coerced.get("kind") == "find_files":
        coerced = _coerce_find_files_step(coerced)

    return coerced


def _coerce_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(payload)
    coerced.setdefault("version", "1")
    steps = coerced.get("steps")
    if coerced.get("questions") is None:
        coerced["questions"] = []
    else:
        coerced.setdefault("questions", [])
        if isinstance(coerced["questions"], list):
            normalized_questions: list[Any] = []
            for question in coerced["questions"]:
                if isinstance(question, dict):
                    normalized = dict(question)
                    field_path = normalized.get("field_path")
                    if (
                        isinstance(field_path, str)
                        and not field_path.startswith("steps[")
                        and isinstance(steps, list)
                        and len(steps) == 1
                    ):
                        normalized["field_path"] = f"steps[0].{field_path}"
                    if "prompt" not in normalized and "question" in normalized:
                        normalized["prompt"] = normalized.pop("question")
                    if "prompt" not in normalized:
                        normalized["prompt"] = f"Please provide {normalized.get('field_path', 'the missing value')}."
                    normalized_questions.append(normalized)
                else:
                    normalized_questions.append(question)
            coerced["questions"] = normalized_questions
    if coerced.get("notes") is None:
        coerced["notes"] = []
    elif isinstance(coerced.get("notes"), str):
        coerced["notes"] = [coerced["notes"]]
    else:
        coerced.setdefault("notes", [])
    if coerced.get("needs_confirmation") is None:
        coerced["needs_confirmation"] = bool(coerced["questions"])
    else:
        coerced.setdefault("needs_confirmation", False)
    coerced.setdefault("risk_level", "medium")

    if isinstance(steps, list):
        coerced["steps"] = [_coerce_step_payload(step) for step in steps]

    return coerced


def _validate_or_coerce_plan_payload(payload: str | bytes | dict[str, Any]) -> PlanV1:
    if isinstance(payload, dict):
        try:
            return validate_plan_payload(payload)
        except ValidationError:
            return validate_plan_payload(_coerce_plan_payload(payload))

    try:
        return validate_plan_payload(payload)
    except ValidationError as original_error:
        text = payload.decode() if isinstance(payload, bytes) else payload
        extracted = _extract_json_fragment(text)
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError:
            raise original_error
        try:
            return validate_plan_payload(parsed)
        except ValidationError:
            return validate_plan_payload(_coerce_plan_payload(parsed))


def _extract_depth_hint(prompt: str) -> int | None:
    lowered = prompt.lower()
    match = re.search(r"no deeper than (\d+) levels?", lowered)
    if match:
        return int(match.group(1))

    match = re.search(
        r"no deeper than (one|two|three|four|five|six|seven|eight|nine|ten) levels?",
        lowered,
    )
    if match:
        return NUMBER_WORDS[match.group(1)]
    return None


def _extract_file_mentions(prompt: str) -> set[str]:
    return set(re.findall(r"(?<![\w/.-])([./\w-]+\.[A-Za-z0-9]{2,5})(?![\w/.-])", prompt))


def _stem_for_path(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate or candidate.startswith("{{"):
        return None
    return Path(candidate).stem or None


def _canonical_output_file(step: dict[str, Any], previous_step: dict[str, Any] | None) -> str | None:
    kind = step.get("kind")
    previous_path_hint = None
    if previous_step and previous_step.get("kind") == "find_files":
        previous_path_hint = previous_step.get("path_contains")

    if kind == "csv_filter_rows":
        basis = previous_path_hint or _stem_for_path(step.get("input_file")) or "filtered"
        return f"{basis}_filtered.csv"
    if kind == "csv_join":
        basis = _stem_for_path(step.get("left_file")) or "joined"
        return f"{basis}_joined.csv"
    if kind == "csv_select_columns":
        basis = _stem_for_path(step.get("input_file")) or "selected"
        return f"{basis}_selected.csv"
    if kind == "csv_sort_rows":
        basis = _stem_for_path(step.get("input_file")) or "sorted"
        return f"{basis}_sorted.csv"
    if kind == "csv_group_count":
        basis = previous_path_hint or _stem_for_path(step.get("input_file")) or "group"
        groups = step.get("group_by") or []
        if groups:
            return f"{basis}_{'_'.join(groups)}_counts.csv"
        return f"{basis}_counts.csv"
    if kind == "pdf_extract_pages":
        basis = _stem_for_path(step.get("input_file")) or "extract"
        start = step.get("page_start")
        end = step.get("page_end")
        if start is not None and end is not None:
            return f"{basis}_pages_{start}_{end}.pdf"
        return f"{basis}_excerpt.pdf"
    if kind == "pdf_compress":
        basis = _stem_for_path(step.get("input_file")) or "document"
        return f"{basis}_compressed.pdf"
    if kind == "media_extract_audio_mp3":
        basis = _stem_for_path(step.get("input_file")) or "audio"
        return f"{basis}.mp3"
    return None


def _apply_prompt_hints(plan: PlanV1, prompt: str) -> PlanV1:
    lowered = prompt.lower()
    file_mentions = _extract_file_mentions(prompt)
    data = plan.model_dump(mode="json", exclude_none=False)
    mutated = False

    if len(data["steps"]) == 1 and data["steps"][0]["kind"] == "find_files":
        data["risk_level"] = "low"
        mutated = True
    elif data.get("risk_level") == "low":
        data["risk_level"] = "medium"
        mutated = True

    for index, step in enumerate(data["steps"]):
        previous_step = data["steps"][index - 1] if index > 0 else None
        if step.get("kind") != "find_files":
            if previous_step and previous_step.get("kind") == "find_files":
                input_file = step.get("input_file")
                if isinstance(input_file, str) and (input_file.startswith("{{") or input_file not in file_mentions):
                    step["input_file"] = None
                    mutated = True

            output_file = step.get("output_file")
            if not (isinstance(output_file, str) and output_file in file_mentions):
                canonical_output = _canonical_output_file(step, previous_step)
                if canonical_output and step.get("output_file") != canonical_output:
                    step["output_file"] = canonical_output
                    mutated = True

            kind = step.get("kind")
            if kind == "csv_join":
                explicit_join_keys = re.search(r"\bon\s+[A-Za-z_][\w, ]*", lowered)
                if not explicit_join_keys:
                    if step.get("join_keys") is not None:
                        step["join_keys"] = None
                        mutated = True
                    questions = data.setdefault("questions", [])
                    filtered_questions = [
                        question
                        for question in questions
                        if not (
                            isinstance(question, dict)
                            and question.get("field_path") in {"steps[0].left_file", "steps[0].right_file"}
                        )
                    ]
                    if filtered_questions != questions:
                        data["questions"] = filtered_questions
                        questions = filtered_questions
                        mutated = True
                    field_path = f"steps[{index}].join_keys"
                    if not any(
                        isinstance(question, dict) and question.get("field_path") == field_path
                        for question in questions
                    ):
                        questions.append(
                            {
                                "field_path": field_path,
                                "prompt": CSV_JOIN_KEYS_PROMPT,
                                "expected_type": "string_list",
                                "required": True,
                            }
                        )
                        mutated = True
                    else:
                        for question in questions:
                            if isinstance(question, dict) and question.get("field_path") == field_path:
                                normalized_question = {
                                    "field_path": field_path,
                                    "prompt": CSV_JOIN_KEYS_PROMPT,
                                    "expected_type": "string_list",
                                    "required": True,
                                }
                                if question != normalized_question:
                                    question.clear()
                                    question.update(normalized_question)
                                    mutated = True
                                break
                    if CSV_JOIN_KEYS_NOTE not in data["notes"]:
                        data["notes"].append(CSV_JOIN_KEYS_NOTE)
                        mutated = True
                    if not data.get("needs_confirmation"):
                        data["needs_confirmation"] = True
                        mutated = True
            if kind in MEDIA_INPUT_QUESTION_PROMPTS:
                output_file = step.get("output_file")
                non_output_mentions = {item for item in file_mentions if item != output_file}
                deictic_media_reference = any(
                    phrase in lowered
                    for phrase in ("this movie", "this video", "this media", "this file")
                )
                input_file = step.get("input_file")
                if (
                    input_file is not None
                    and input_file not in file_mentions
                    and (deictic_media_reference or not non_output_mentions)
                ):
                    step["input_file"] = None
                    mutated = True

                if step.get("input_file") is None and (deictic_media_reference or not non_output_mentions):
                    questions = data.setdefault("questions", [])
                    field_path = f"steps[{index}].input_file"
                    if not any(
                        isinstance(question, dict) and question.get("field_path") == field_path
                        for question in questions
                    ):
                        questions.append(
                            {
                                "field_path": field_path,
                                "prompt": MEDIA_INPUT_QUESTION_PROMPTS[kind],
                                "expected_type": "string",
                                "required": True,
                            }
                        )
                        mutated = True
                    else:
                        for question in questions:
                            if isinstance(question, dict) and question.get("field_path") == field_path:
                                normalized_question = {
                                    "field_path": field_path,
                                    "prompt": MEDIA_INPUT_QUESTION_PROMPTS[kind],
                                    "expected_type": "string",
                                    "required": True,
                                }
                                if question != normalized_question:
                                    question.clear()
                                    question.update(normalized_question)
                                    mutated = True
                                break
                    note = MEDIA_INPUT_NOTES[kind]
                    if data["notes"] != [note]:
                        data["notes"] = [note]
                        mutated = True
                    if not data.get("needs_confirmation"):
                        data["needs_confirmation"] = True
                        mutated = True
            continue

        roots = step.get("roots") or []
        cleaned_roots = [root for root in roots if root not in FIELD_NAMES]
        if cleaned_roots != roots:
            step["roots"] = cleaned_roots or ["."]
            mutated = True

        if step.get("path_contains") == "":
            step["path_contains"] = None
            mutated = True

        if step.get("extension") is None:
            if " pdf" in f" {lowered}" or lowered.startswith("pdf"):
                step["extension"] = ".pdf"
                mutated = True
            elif " csv" in f" {lowered}" or lowered.startswith("csv"):
                step["extension"] = ".csv"
                mutated = True
        elif isinstance(step.get("extension"), str) and not step["extension"].startswith("."):
            step["extension"] = f".{step['extension']}"
            mutated = True

        depth_hint = _extract_depth_hint(prompt)
        if depth_hint is not None and step.get("max_depth") is None:
            step["max_depth"] = depth_hint
            mutated = True

    if not mutated:
        return plan
    return PlanV1.model_validate(data)


class OpenAIPlanner:
    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig.from_env()
        if not self.config.api_key:
            raise ValueError("Set NLSH_API_KEY or HF_TOKEN before using the OpenAI planner.")

    def _request_kwargs(
        self,
        *,
        developer_prompt: str,
        prompt: str,
        response_format: dict[str, Any],
        use_structured_outputs: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "temperature": 0,
            "max_completion_tokens": 500,
            "messages": [
                {"role": "developer", "content": developer_prompt},
                {"role": "user", "content": prompt},
            ],
            "response_format": response_format,
        }
        if self.config.is_vllm_like():
            kwargs["reasoning_effort"] = "low"
            kwargs["max_completion_tokens"] = 220
            kwargs["response_format"] = {"type": "json_object"}
            extra_body: dict[str, Any] = {
                "include_reasoning": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            kwargs["extra_body"] = extra_body
        return kwargs

    def _repair_payload(
        self,
        *,
        client: Any,
        prompt: str,
        raw_text: str,
        error: ValidationError,
    ) -> str:
        repair_prompt = (
            f"User request:\n{prompt}\n\n"
            f"Schema:\n{json.dumps(plan_json_schema(), ensure_ascii=False)}\n\n"
            f"Invalid JSON:\n{raw_text}\n\n"
            f"Validation error:\n{validation_error_text(error)}"
        )
        response = client.chat.completions.create(
            **self._request_kwargs(
                developer_prompt=REPAIR_DEVELOPER_PROMPT,
                prompt=repair_prompt,
                response_format={"type": "json_object"},
                use_structured_outputs=False,
            )
        )
        return _extract_message_text(response.choices[0].message).strip()

    def plan(self, prompt: str) -> PlanV1:
        from openai import OpenAI

        client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
        response = client.chat.completions.create(
            **self._request_kwargs(
                developer_prompt=TRAINING_DEVELOPER_PROMPT,
                prompt=prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "nlsh_plan_v1",
                        "strict": True,
                        "schema": plan_json_schema(),
                    },
                },
            )
        )
        raw_text = _extract_message_text(response.choices[0].message).strip()
        try:
            return _apply_prompt_hints(_validate_or_coerce_plan_payload(raw_text), prompt)
        except ValidationError as error:
            repaired_text = self._repair_payload(
                client=client,
                prompt=prompt,
                raw_text=raw_text,
                error=error,
            )
            if not repaired_text:
                raise ValueError("Planner returned empty content while attempting to repair invalid JSON.")
            return _apply_prompt_hints(_validate_or_coerce_plan_payload(repaired_text), prompt)


class GoldPlanner:
    def __init__(self, dataset_path: Path | None = None) -> None:
        path = dataset_path or default_split_path("train")
        self.records = load_jsonl(path)
        self.prompt_to_plan = {
            record["prompt"].strip(): PlanV1.model_validate(record["plan"])
            for record in self.records
        }

    def plan(self, prompt: str) -> PlanV1:
        key = prompt.strip()
        if key not in self.prompt_to_plan:
            raise KeyError(f"Prompt not found in gold dataset: {prompt!r}")
        return self.prompt_to_plan[key]


def load_planner(name: str, dataset_path: Path | None = None) -> Planner:
    if name == "openai":
        return OpenAIPlanner()
    if name == "gold":
        return GoldPlanner(dataset_path=dataset_path)
    raise ValueError(f"Unsupported planner: {name}")


def plan_to_pretty_json(plan: PlanV1) -> str:
    return json.dumps(plan.model_dump(mode="json", exclude_none=False), indent=2, ensure_ascii=False)
