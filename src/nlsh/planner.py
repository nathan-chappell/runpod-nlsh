from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nlsh.dataio import default_split_path, load_jsonl
from nlsh.prompts import TRAINING_DEVELOPER_PROMPT
from nlsh.schema import PlanV1, plan_json_schema, validate_plan_payload


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


class OpenAIPlanner:
    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig.from_env()
        if not self.config.api_key:
            raise ValueError("Set NLSH_API_KEY or HF_TOKEN before using the OpenAI planner.")

    def plan(self, prompt: str) -> PlanV1:
        from openai import OpenAI

        client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
        response = client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            messages=[
                {"role": "developer", "content": TRAINING_DEVELOPER_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "nlsh_plan_v1",
                    "strict": True,
                    "schema": plan_json_schema(),
                },
            },
        )
        raw_text = _extract_message_text(response.choices[0].message).strip()
        return validate_plan_payload(raw_text)


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

