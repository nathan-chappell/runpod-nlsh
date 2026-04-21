from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator


def _reject_control_chars(value: str) -> str:
    if any(char in value for char in ("\x00", "\r", "\n")):
        raise ValueError("control characters are not allowed")
    return value.strip()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FindFilesStep(StrictModel):
    kind: Literal["find_files"] = "find_files"
    root: str = "."
    glob: str | None = None
    max_depth: int | None = Field(default=None, ge=0)

    @field_validator("root", "glob")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)


class PdfMergeStep(StrictModel):
    kind: Literal["pdf_merge"] = "pdf_merge"
    input_files: list[str] | None = None
    output_file: str | None = None

    @field_validator("input_files", mode="after")
    @classmethod
    def validate_input_files(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("input_files cannot be empty")
        return [_reject_control_chars(item) for item in value]

    @field_validator("output_file")
    @classmethod
    def validate_output_file(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)


class PdfExtractPagesStep(StrictModel):
    kind: Literal["pdf_extract_pages"] = "pdf_extract_pages"
    input_file: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    output_file: str | None = None

    @field_validator("input_file", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)

    @model_validator(mode="after")
    def validate_range(self) -> "PdfExtractPagesStep":
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_start > self.page_end
        ):
            raise ValueError("page_start must be <= page_end")
        return self


class PdfSearchTextStep(StrictModel):
    kind: Literal["pdf_search_text"] = "pdf_search_text"
    input_files: list[str] | None = None
    query: str | None = None
    output_file: str | None = None
    context_chars: int = Field(default=160, ge=0)

    @field_validator("input_files", mode="after")
    @classmethod
    def validate_input_files(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("input_files cannot be empty")
        return [_reject_control_chars(item) for item in value]

    @field_validator("query", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)


JsonScalar = str | int | float | bool


class CsvToJsonStep(StrictModel):
    kind: Literal["csv_to_json"] = "csv_to_json"
    input_file: str | None = None
    output_file: str | None = None

    @field_validator("input_file", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)


class JsonFilterStep(StrictModel):
    kind: Literal["json_filter"] = "json_filter"
    input_file: str | None = None
    field: str | None = None
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains"] | None = None
    value: JsonScalar | None = None
    output_file: str | None = None

    @field_validator("input_file", "field", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)


class JsonSelectFieldsStep(StrictModel):
    kind: Literal["json_select_fields"] = "json_select_fields"
    input_file: str | None = None
    fields: list[str] | None = None
    output_file: str | None = None

    @field_validator("input_file", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)

    @field_validator("fields", mode="after")
    @classmethod
    def validate_fields(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("fields cannot be empty")
        return [_reject_control_chars(item) for item in value]


class JsonSortStep(StrictModel):
    kind: Literal["json_sort"] = "json_sort"
    input_file: str | None = None
    field: str | None = None
    descending: bool = False
    output_file: str | None = None

    @field_validator("input_file", "field", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)


class JsonGroupCountStep(StrictModel):
    kind: Literal["json_group_count"] = "json_group_count"
    input_file: str | None = None
    group_by: list[str] | None = None
    output_file: str | None = None
    count_field: str = "count"

    @field_validator("input_file", "output_file", "count_field")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)

    @field_validator("group_by", mode="after")
    @classmethod
    def validate_group_by(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("group_by cannot be empty")
        return [_reject_control_chars(item) for item in value]


JsonStep = JsonFilterStep | JsonSelectFieldsStep | JsonSortStep | JsonGroupCountStep

Step = Annotated[
    FindFilesStep
    | PdfMergeStep
    | PdfExtractPagesStep
    | PdfSearchTextStep
    | CsvToJsonStep
    | JsonFilterStep
    | JsonSelectFieldsStep
    | JsonSortStep
    | JsonGroupCountStep,
    Field(discriminator="kind"),
]


class PlanV1(StrictModel):
    kind: Literal["plan"] = "plan"
    steps: list[Step]

    @model_validator(mode="after")
    def validate_shape(self) -> "PlanV1":
        if not self.steps:
            raise ValueError("at least one step is required")
        if len(self.steps) > 3:
            raise ValueError("PlanV1 supports at most 3 steps")
        for index, step in enumerate(self.steps):
            if isinstance(step, FindFilesStep) and index != 0:
                raise ValueError("find_files can only be the first step")
        if len(self.steps) > 1 and isinstance(self.steps[-1], FindFilesStep):
            raise ValueError("find_files cannot be the terminal step in a multi-step plan")
        return self


class Clarification(StrictModel):
    kind: Literal["clarification"] = "clarification"
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        return _reject_control_chars(value)


PlannerOutput = Annotated[PlanV1 | Clarification, Field(discriminator="kind")]
PLANNER_OUTPUT_ADAPTER = TypeAdapter(PlannerOutput)


def plan_json_schema() -> dict[str, Any]:
    return PLANNER_OUTPUT_ADAPTER.json_schema()


def validate_plan_payload(payload: str | bytes | dict[str, Any]) -> PlannerOutput:
    if isinstance(payload, dict):
        return PLANNER_OUTPUT_ADAPTER.validate_python(payload)
    return PLANNER_OUTPUT_ADAPTER.validate_json(payload)


def normalize_plan(plan: PlannerOutput) -> dict[str, Any]:
    return PLANNER_OUTPUT_ADAPTER.dump_python(plan, mode="json", exclude_none=False)


def validation_error_text(error: ValidationError) -> str:
    return error.json(indent=2)
