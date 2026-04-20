from __future__ import annotations

import json
import os
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

    coerced = dict(step)
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
    if coerced.get("questions") is None:
        coerced["questions"] = []
    else:
        coerced.setdefault("questions", [])
    if coerced.get("notes") is None:
        coerced["notes"] = []
    else:
        coerced.setdefault("notes", [])
    if coerced.get("needs_confirmation") is None:
        coerced["needs_confirmation"] = bool(coerced["questions"])
    else:
        coerced.setdefault("needs_confirmation", False)
    coerced.setdefault("risk_level", "medium")

    steps = coerced.get("steps")
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
            extra_body: dict[str, Any] = {"include_reasoning": False}
            if use_structured_outputs:
                extra_body["structured_outputs"] = {"json": plan_json_schema()}
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
                use_structured_outputs=True,
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
            return _validate_or_coerce_plan_payload(raw_text)
        except ValidationError as error:
            repaired_text = self._repair_payload(
                client=client,
                prompt=prompt,
                raw_text=raw_text,
                error=error,
            )
            if not repaired_text:
                raise ValueError("Planner returned empty content while attempting to repair invalid JSON.")
            return _validate_or_coerce_plan_payload(repaired_text)


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
