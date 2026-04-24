from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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


def _validate_planner_payload(
    payload: str | bytes | dict[str, Any],
    *,
    extract_json_fragment: bool,
) -> PlannerOutput:
    if isinstance(payload, dict):
        return validate_plan_payload(payload)

    try:
        return validate_plan_payload(payload)
    except ValidationError as original_error:
        if not extract_json_fragment:
            raise
        text = payload.decode() if isinstance(payload, bytes) else payload
        extracted = _extract_json_fragment(text)
        if extracted == text:
            raise original_error
        return validate_plan_payload(extracted)


class OpenAIPlanner:
    def __init__(self, config: PlannerConfig | None = None, *, strict: bool = False) -> None:
        self.config = config or PlannerConfig.from_env()
        self.strict = strict
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
            return _validate_planner_payload(raw_text, extract_json_fragment=not self.strict)
        except ValidationError as error:
            if self.strict:
                raise
            repaired_text = self._repair_payload(
                client=client,
                prompt=prompt,
                raw_text=raw_text,
                error=error,
            )
            if not repaired_text:
                raise ValueError("Planner returned empty content while attempting to repair invalid JSON.")
            return _validate_planner_payload(repaired_text, extract_json_fragment=True)


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


def load_planner(name: str, dataset_path: Path | None = None, *, strict: bool = False) -> Planner:
    if name == "openai":
        return OpenAIPlanner(strict=strict)
    if name == "gold":
        return GoldPlanner(dataset_path=dataset_path)
    raise ValueError(f"Unsupported planner: {name}")


def plan_to_pretty_json(plan: PlannerOutput) -> str:
    return json.dumps(normalize_plan(plan), indent=2, ensure_ascii=False)
