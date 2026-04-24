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

    @staticmethod
    def _require_fields(step: Step, *fields: str) -> None:
        missing = [field for field in fields if getattr(step, field) is None]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{step.kind} requires {joined}")

    @staticmethod
    def _require_omitted(step: Step, *fields: str) -> None:
        provided = [field for field in fields if field in step.model_fields_set]
        if provided:
            joined = ", ".join(provided)
            raise ValueError(f"{step.kind} must omit {joined} in this plan shape")

    @classmethod
    def _validate_executable_step(cls, step: Step) -> None:
        if isinstance(step, FindFilesStep):
            return
        if isinstance(step, PdfMergeStep):
            cls._require_fields(step, "input_files")
            return
        if isinstance(step, PdfExtractPagesStep):
            cls._require_fields(step, "input_file", "page_start", "page_end")
            return
        if isinstance(step, PdfSearchTextStep):
            cls._require_fields(step, "input_files", "query")
            return
        if isinstance(step, CsvToJsonStep):
            cls._require_fields(step, "input_file")
            return
        if isinstance(step, JsonFilterStep):
            cls._require_fields(step, "input_file", "field", "operator", "value")
            return
        if isinstance(step, JsonSelectFieldsStep):
            cls._require_fields(step, "input_file", "fields")
            return
        if isinstance(step, JsonSortStep):
            cls._require_fields(step, "input_file", "field")
            return
        if isinstance(step, JsonGroupCountStep):
            cls._require_fields(step, "input_file", "group_by")
            return
        raise ValueError(f"Unsupported step kind: {type(step)!r}")

    @classmethod
    def _validate_find_pipeline_step(cls, step: Step) -> None:
        if isinstance(step, FindFilesStep):
            raise ValueError("find_files cannot follow find_files")
        if isinstance(step, PdfMergeStep):
            cls._require_omitted(step, "input_files")
            return
        if isinstance(step, PdfExtractPagesStep):
            cls._require_omitted(step, "input_file")
            cls._require_fields(step, "page_start", "page_end")
            return
        if isinstance(step, PdfSearchTextStep):
            cls._require_omitted(step, "input_files")
            cls._require_fields(step, "query")
            return
        if isinstance(step, CsvToJsonStep):
            cls._require_omitted(step, "input_file")
            return
        if isinstance(step, JsonFilterStep):
            cls._require_omitted(step, "input_file")
            cls._require_fields(step, "field", "operator", "value")
            return
        if isinstance(step, JsonSelectFieldsStep):
            cls._require_omitted(step, "input_file")
            cls._require_fields(step, "fields")
            return
        if isinstance(step, JsonSortStep):
            cls._require_omitted(step, "input_file")
            cls._require_fields(step, "field")
            return
        if isinstance(step, JsonGroupCountStep):
            cls._require_omitted(step, "input_file")
            cls._require_fields(step, "group_by")
            return
        raise ValueError(f"Unsupported step kind after find_files: {type(step)!r}")

    @classmethod
    def _validate_csv_pipeline_step(cls, step: Step) -> None:
        if isinstance(step, JsonFilterStep):
            cls._require_omitted(step, "input_file")
            cls._require_fields(step, "field", "operator", "value")
            return
        if isinstance(step, JsonSelectFieldsStep):
            cls._require_omitted(step, "input_file")
            cls._require_fields(step, "fields")
            return
        if isinstance(step, JsonSortStep):
            cls._require_omitted(step, "input_file")
            cls._require_fields(step, "field")
            return
        if isinstance(step, JsonGroupCountStep):
            cls._require_omitted(step, "input_file")
            cls._require_fields(step, "group_by")
            return
        raise ValueError("csv_to_json pipelines must end with a JSON step")

    @model_validator(mode="after")
    def validate_shape(self) -> "PlanV1":
        if not self.steps:
            raise ValueError("at least one step is required")
        if len(self.steps) == 1:
            self._validate_executable_step(self.steps[0])
            return self
        if len(self.steps) != 2:
            raise ValueError("PlanV1 supports only 1-step plans or 2-step pipelines")

        first, second = self.steps
        if isinstance(first, FindFilesStep):
            self._validate_find_pipeline_step(second)
            return self
        if isinstance(first, CsvToJsonStep):
            self._require_fields(first, "input_file")
            self._validate_csv_pipeline_step(second)
            return self
        raise ValueError("Two-step plans must start with find_files or csv_to_json")
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


def _json_string_schema() -> dict[str, Any]:
    return {"type": "string"}


def _json_scalar_schema() -> dict[str, Any]:
    return {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"},
            {"type": "number"},
            {"type": "boolean"},
        ]
    }


def _array_of_strings_schema() -> dict[str, Any]:
    return {"type": "array", "items": _json_string_schema(), "minItems": 1}


def _step_schema(
    kind: str,
    *,
    required: list[str],
    properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"const": kind, "type": "string"},
            **properties,
        },
        "required": ["kind", *required],
    }


def plan_json_schema() -> dict[str, Any]:
    find_files_step = _step_schema(
        "find_files",
        required=[],
        properties={
            "root": _json_string_schema(),
            "glob": _json_string_schema(),
            "max_depth": {"type": "integer", "minimum": 0},
        },
    )
    csv_to_json_step = _step_schema(
        "csv_to_json",
        required=["input_file"],
        properties={
            "input_file": _json_string_schema(),
            "output_file": _json_string_schema(),
        },
    )
    pdf_merge_step = _step_schema(
        "pdf_merge",
        required=["input_files"],
        properties={
            "input_files": _array_of_strings_schema(),
            "output_file": _json_string_schema(),
        },
    )
    pdf_extract_step = _step_schema(
        "pdf_extract_pages",
        required=["input_file", "page_start", "page_end"],
        properties={
            "input_file": _json_string_schema(),
            "page_start": {"type": "integer", "minimum": 1},
            "page_end": {"type": "integer", "minimum": 1},
            "output_file": _json_string_schema(),
        },
    )
    pdf_search_step = _step_schema(
        "pdf_search_text",
        required=["input_files", "query"],
        properties={
            "input_files": _array_of_strings_schema(),
            "query": _json_string_schema(),
            "output_file": _json_string_schema(),
            "context_chars": {"type": "integer", "minimum": 0},
        },
    )
    json_filter_step = _step_schema(
        "json_filter",
        required=["input_file", "field", "operator", "value"],
        properties={
            "input_file": _json_string_schema(),
            "field": _json_string_schema(),
            "operator": {"enum": ["eq", "ne", "gt", "gte", "lt", "lte", "contains"]},
            "value": _json_scalar_schema(),
            "output_file": _json_string_schema(),
        },
    )
    json_select_fields_step = _step_schema(
        "json_select_fields",
        required=["input_file", "fields"],
        properties={
            "input_file": _json_string_schema(),
            "fields": _array_of_strings_schema(),
            "output_file": _json_string_schema(),
        },
    )
    json_sort_step = _step_schema(
        "json_sort",
        required=["input_file", "field"],
        properties={
            "input_file": _json_string_schema(),
            "field": _json_string_schema(),
            "descending": {"type": "boolean"},
            "output_file": _json_string_schema(),
        },
    )
    json_group_count_step = _step_schema(
        "json_group_count",
        required=["input_file", "group_by"],
        properties={
            "input_file": _json_string_schema(),
            "group_by": _array_of_strings_schema(),
            "output_file": _json_string_schema(),
            "count_field": _json_string_schema(),
        },
    )
    find_pipeline_step = {
        "oneOf": [
            _step_schema("csv_to_json", required=[], properties={"output_file": _json_string_schema()}),
            _step_schema("pdf_merge", required=[], properties={"output_file": _json_string_schema()}),
            _step_schema(
                "pdf_extract_pages",
                required=["page_start", "page_end"],
                properties={
                    "page_start": {"type": "integer", "minimum": 1},
                    "page_end": {"type": "integer", "minimum": 1},
                    "output_file": _json_string_schema(),
                },
            ),
            _step_schema(
                "pdf_search_text",
                required=["query"],
                properties={
                    "query": _json_string_schema(),
                    "output_file": _json_string_schema(),
                    "context_chars": {"type": "integer", "minimum": 0},
                },
            ),
            _step_schema(
                "json_filter",
                required=["field", "operator", "value"],
                properties={
                    "field": _json_string_schema(),
                    "operator": {"enum": ["eq", "ne", "gt", "gte", "lt", "lte", "contains"]},
                    "value": _json_scalar_schema(),
                    "output_file": _json_string_schema(),
                },
            ),
            _step_schema(
                "json_select_fields",
                required=["fields"],
                properties={
                    "fields": _array_of_strings_schema(),
                    "output_file": _json_string_schema(),
                },
            ),
            _step_schema(
                "json_sort",
                required=["field"],
                properties={
                    "field": _json_string_schema(),
                    "descending": {"type": "boolean"},
                    "output_file": _json_string_schema(),
                },
            ),
            _step_schema(
                "json_group_count",
                required=["group_by"],
                properties={
                    "group_by": _array_of_strings_schema(),
                    "output_file": _json_string_schema(),
                    "count_field": _json_string_schema(),
                },
            ),
        ]
    }
    csv_pipeline_step = {
        "oneOf": [
            _step_schema(
                "json_filter",
                required=["field", "operator", "value"],
                properties={
                    "field": _json_string_schema(),
                    "operator": {"enum": ["eq", "ne", "gt", "gte", "lt", "lte", "contains"]},
                    "value": _json_scalar_schema(),
                    "output_file": _json_string_schema(),
                },
            ),
            _step_schema(
                "json_select_fields",
                required=["fields"],
                properties={
                    "fields": _array_of_strings_schema(),
                    "output_file": _json_string_schema(),
                },
            ),
            _step_schema(
                "json_sort",
                required=["field"],
                properties={
                    "field": _json_string_schema(),
                    "descending": {"type": "boolean"},
                    "output_file": _json_string_schema(),
                },
            ),
            _step_schema(
                "json_group_count",
                required=["group_by"],
                properties={
                    "group_by": _array_of_strings_schema(),
                    "output_file": _json_string_schema(),
                    "count_field": _json_string_schema(),
                },
            ),
        ]
    }
    single_step_plan = {
        "type": "array",
        "minItems": 1,
        "maxItems": 1,
        "prefixItems": [
            {
                "oneOf": [
                    find_files_step,
                    csv_to_json_step,
                    pdf_merge_step,
                    pdf_extract_step,
                    pdf_search_step,
                    json_filter_step,
                    json_select_fields_step,
                    json_sort_step,
                    json_group_count_step,
                ]
            }
        ],
    }
    find_pipeline_plan = {
        "type": "array",
        "minItems": 2,
        "maxItems": 2,
        "prefixItems": [find_files_step, find_pipeline_step],
    }
    csv_pipeline_plan = {
        "type": "array",
        "minItems": 2,
        "maxItems": 2,
        "prefixItems": [csv_to_json_step, csv_pipeline_step],
    }
    return {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"const": "clarification", "type": "string"},
                    "question": _json_string_schema(),
                },
                "required": ["kind", "question"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"const": "plan", "type": "string"},
                    "steps": {
                        "oneOf": [
                            single_step_plan,
                            find_pipeline_plan,
                            csv_pipeline_plan,
                        ]
                    },
                },
                "required": ["kind", "steps"],
            },
        ]
    }


def validate_plan_payload(payload: str | bytes | dict[str, Any]) -> PlannerOutput:
    if isinstance(payload, dict):
        return PLANNER_OUTPUT_ADAPTER.validate_python(payload)
    return PLANNER_OUTPUT_ADAPTER.validate_json(payload)


def normalize_plan(plan: PlannerOutput) -> dict[str, Any]:
    return PLANNER_OUTPUT_ADAPTER.dump_python(plan, mode="json", exclude_none=True)


def validation_error_text(error: ValidationError) -> str:
    return error.json(indent=2)
