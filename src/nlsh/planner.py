from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import ValidationError

from nlsh.dataio import default_dataset_path, load_jsonl
from nlsh.prompts import REPAIR_DEVELOPER_PROMPT, TRAINING_DEVELOPER_PROMPT
from nlsh.schema import (
    PlannerOutput,
    normalize_plan,
    plan_json_schema,
    validate_plan_payload,
    validation_error_text,
)


STEP_KINDS = {
    "find_files",
    "pdf_merge",
    "pdf_extract_pages",
    "pdf_search_text",
    "csv_to_json",
    "json_filter",
    "json_select_fields",
    "json_sort",
    "json_group_count",
}
PLAN_LEVEL_KEYS = {"needs_confirmation", "questions", "risk_level", "notes", "version"}


class Planner(Protocol):
    def plan(self, prompt: str) -> PlannerOutput: ...


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(slots=True)
class PlannerConfig:
    model: str
    base_url: str
    api_key: str
    request_timeout: float

    @classmethod
    def from_env(cls) -> "PlannerConfig":
        _load_dotenv()
        api_key = os.environ.get("NLSH_API_KEY") or os.environ.get("HF_TOKEN") or ""
        return cls(
            model=os.environ.get("NLSH_MODEL", "microsoft/Phi-4-mini-instruct"),
            base_url=os.environ.get("NLSH_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=api_key,
            request_timeout=float(os.environ.get("NLSH_REQUEST_TIMEOUT", "60")),
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
    if lines and lines[0].startswith("```"):
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


def _coerce_step_payload(step: Any) -> Any:
    if not isinstance(step, dict):
        return step

    coerced = {key: value for key, value in step.items() if key not in PLAN_LEVEL_KEYS}
    nested_kind = next(
        (key for key, value in coerced.items() if key in STEP_KINDS and isinstance(value, dict)),
        None,
    )
    if "kind" not in coerced and nested_kind is not None:
        nested_payload = dict(coerced.pop(nested_kind))
        coerced = {"kind": nested_kind, **nested_payload, **coerced}

    kind = coerced.get("kind")
    if isinstance(kind, str):
        normalized_kind = kind
        if normalized_kind not in STEP_KINDS and normalized_kind.endswith("Step"):
            normalized_kind = re.sub(r"(?<!^)(?=[A-Z])", "_", normalized_kind[:-4]).lower()
        if normalized_kind in STEP_KINDS:
            coerced["kind"] = normalized_kind
            kind = normalized_kind

    if kind == "find_files":
        roots = coerced.pop("roots", None)
        if roots is not None and "root" not in coerced:
            coerced["root"] = roots[0] if isinstance(roots, list) and roots else roots
        path = coerced.pop("path", None)
        if path is not None and "root" not in coerced:
            coerced["root"] = path
        depth = coerced.pop("depth", None)
        if depth is not None and "max_depth" not in coerced:
            coerced["max_depth"] = depth
        pattern = coerced.pop("pattern", None)
        name_pattern = coerced.pop("name_pattern", None)
        extension = coerced.pop("extension", None)
        path_contains = coerced.pop("path_contains", None)
        file_type = coerced.pop("file_type", None)
        if extension is None and file_type in {"csv", "pdf", "json"}:
            extension = f".{file_type}"
        if "glob" not in coerced:
            if pattern is not None:
                coerced["glob"] = pattern
            elif name_pattern is not None:
                coerced["glob"] = name_pattern
            elif extension is not None:
                normalized_ext = extension if str(extension).startswith(".") else f".{extension}"
                if path_contains:
                    coerced["glob"] = f"*{path_contains}*{normalized_ext}"
                else:
                    coerced["glob"] = f"*{normalized_ext}"
            elif path_contains is not None:
                coerced["glob"] = f"*{path_contains}*"
    elif kind == "json_group_count" and isinstance(coerced.get("group_by"), str):
        coerced["group_by"] = [coerced["group_by"]]
    return coerced


def _coerce_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    coerced = {key: value for key, value in payload.items() if key not in PLAN_LEVEL_KEYS}
    candidate_step = _coerce_step_payload(coerced)
    if (
        isinstance(candidate_step, dict)
        and candidate_step.get("kind") in STEP_KINDS
        and "steps" not in coerced
        and "question" not in coerced
        and "clarifying_question" not in coerced
    ):
        return {"kind": "plan", "steps": [candidate_step]}

    if "kind" not in coerced:
        if "clarifying_question" in coerced:
            return {"kind": "clarification", "question": coerced["clarifying_question"]}
        if "question" in coerced and "steps" not in coerced:
            return {"kind": "clarification", "question": coerced["question"]}
        coerced["kind"] = "plan"
    steps = coerced.get("steps")

    if isinstance(steps, list):
        coerced["steps"] = [_coerce_step_payload(step) for step in steps]
    for key in PLAN_LEVEL_KEYS:
        coerced.pop(key, None)
    return coerced


def _validate_or_coerce_plan_payload(
    payload: str | bytes | dict[str, Any],
) -> PlannerOutput:
    if isinstance(payload, dict):
        try:
            return validate_plan_payload(payload)
        except ValidationError:
            return validate_plan_payload(_coerce_plan_payload(payload))

    try:
        return validate_plan_payload(payload)
    except ValidationError as original_error:
        if TYPE_CHECKING:
            assert isinstance(payload, (str, bytes))
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
    ) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": 600,
            "messages": [
                {"role": "system", "content": developer_prompt},
                {"role": "user", "content": prompt},
            ],
            "response_format": response_format,
        }

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
            )
        )
        return _extract_message_text(response.choices[0].message).strip()

    def plan(self, prompt: str) -> PlannerOutput:
        from openai import OpenAI

        client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.request_timeout,
            max_retries=0,
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
        path = dataset_path or default_dataset_path()
        self.records = load_jsonl(path)
        self.prompt_to_plan = {record["prompt"].strip(): validate_plan_payload(record["plan"]) for record in self.records}

    def plan(self, prompt: str) -> PlannerOutput:
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


def plan_to_pretty_json(plan: PlannerOutput) -> str:
    return json.dumps(normalize_plan(plan), indent=2, ensure_ascii=False)
