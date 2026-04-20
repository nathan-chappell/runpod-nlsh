from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


def _reject_control_chars(value: str) -> str:
    if any(char in value for char in ("\x00", "\r", "\n")):
        raise ValueError("control characters are not allowed")
    return value.strip()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ClarificationQuestion(StrictModel):
    field_path: str
    prompt: str
    expected_type: Literal["string", "integer", "number", "boolean", "string_list"] = "string"
    required: bool = True

    @field_validator("field_path", "prompt")
    @classmethod
    def validate_text_fields(cls, value: str) -> str:
        return _reject_control_chars(value)


class FindFilesStep(StrictModel):
    kind: Literal["find_files"] = "find_files"
    roots: list[str] = Field(default_factory=lambda: ["."])
    name_pattern: str | None = None
    extension: str | None = None
    path_contains: str | None = None
    max_depth: int | None = None
    file_type: Literal["file", "directory"] = "file"

    @field_validator("roots", mode="after")
    @classmethod
    def validate_roots(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("roots must contain at least one path")
        return [_reject_control_chars(item) for item in value]

    @field_validator("name_pattern", "extension", "path_contains")
    @classmethod
    def validate_optional_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)


class PdfCombineStep(StrictModel):
    kind: Literal["pdf_combine"] = "pdf_combine"
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


class PdfCompressStep(StrictModel):
    kind: Literal["pdf_compress"] = "pdf_compress"
    input_file: str | None = None
    output_file: str | None = None

    @field_validator("input_file", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)


class PdfExtractPagesStep(StrictModel):
    kind: Literal["pdf_extract_pages"] = "pdf_extract_pages"
    input_file: str | None = None
    page_start: int | None = None
    page_end: int | None = None
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


class PdfRotateStep(StrictModel):
    kind: Literal["pdf_rotate"] = "pdf_rotate"
    input_file: str | None = None
    rotation_degrees: Literal[90, 180, 270] | None = None
    output_file: str | None = None

    @field_validator("input_file", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)


class MediaTranscodeForTvStep(StrictModel):
    kind: Literal["media_transcode_for_tv"] = "media_transcode_for_tv"
    input_file: str | None = None
    output_file: str | None = None

    @field_validator("input_file", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)


class MediaExtractAudioMp3Step(StrictModel):
    kind: Literal["media_extract_audio_mp3"] = "media_extract_audio_mp3"
    input_file: str | None = None
    output_file: str | None = None

    @field_validator("input_file", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)


class MediaClipStep(StrictModel):
    kind: Literal["media_clip"] = "media_clip"
    input_file: str | None = None
    start_seconds: int | None = None
    duration_seconds: int | None = None
    output_file: str | None = None

    @field_validator("input_file", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)


class CsvJoinStep(StrictModel):
    kind: Literal["csv_join"] = "csv_join"
    left_file: str | None = None
    right_file: str | None = None
    join_keys: list[str] | None = None
    how: Literal["inner", "left", "right", "outer"] = "inner"
    output_file: str | None = None

    @field_validator("left_file", "right_file", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)

    @field_validator("join_keys", mode="after")
    @classmethod
    def validate_join_keys(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("join_keys cannot be empty")
        return [_reject_control_chars(item) for item in value]


class CsvFilterRowsStep(StrictModel):
    kind: Literal["csv_filter_rows"] = "csv_filter_rows"
    input_file: str | None = None
    filter_column: str | None = None
    filter_operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains"] | None = None
    filter_value: str | int | float | bool | None = None
    output_file: str | None = None

    @field_validator("input_file", "filter_column", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)


class CsvSelectColumnsStep(StrictModel):
    kind: Literal["csv_select_columns"] = "csv_select_columns"
    input_file: str | None = None
    columns: list[str] | None = None
    output_file: str | None = None

    @field_validator("input_file", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)

    @field_validator("columns", mode="after")
    @classmethod
    def validate_columns(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("columns cannot be empty")
        return [_reject_control_chars(item) for item in value]


class CsvSortRowsStep(StrictModel):
    kind: Literal["csv_sort_rows"] = "csv_sort_rows"
    input_file: str | None = None
    sort_by: list[str] | None = None
    descending: bool = False
    output_file: str | None = None

    @field_validator("input_file", "output_file")
    @classmethod
    def validate_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_control_chars(value)

    @field_validator("sort_by", mode="after")
    @classmethod
    def validate_sort_by(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("sort_by cannot be empty")
        return [_reject_control_chars(item) for item in value]


class CsvGroupCountStep(StrictModel):
    kind: Literal["csv_group_count"] = "csv_group_count"
    input_file: str | None = None
    group_by: list[str] | None = None
    output_file: str | None = None
    count_column: str = "count"

    @field_validator("input_file", "output_file", "count_column")
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


Step = Annotated[
    FindFilesStep
    | PdfCombineStep
    | PdfCompressStep
    | PdfExtractPagesStep
    | PdfRotateStep
    | MediaTranscodeForTvStep
    | MediaExtractAudioMp3Step
    | MediaClipStep
    | CsvJoinStep
    | CsvFilterRowsStep
    | CsvSelectColumnsStep
    | CsvSortRowsStep
    | CsvGroupCountStep,
    Field(discriminator="kind"),
]


class PlanV1(StrictModel):
    version: Literal["1"] = "1"
    steps: list[Step]
    needs_confirmation: bool = False
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"
    notes: list[str] = Field(default_factory=list)

    @field_validator("notes", mode="after")
    @classmethod
    def validate_notes(cls, value: list[str]) -> list[str]:
        return [_reject_control_chars(item) for item in value]

    @model_validator(mode="after")
    def validate_shape(self) -> "PlanV1":
        if not self.steps:
            raise ValueError("at least one step is required")
        if len(self.steps) > 2:
            raise ValueError("PlanV1 supports at most 2 steps")
        if len(self.steps) == 2:
            if self.steps[0].kind != "find_files":
                raise ValueError("the first step must be find_files for a 2-step plan")
            if self.steps[1].kind == "find_files":
                raise ValueError("find_files cannot be the terminal step in a 2-step plan")
        if self.questions and not self.needs_confirmation:
            raise ValueError("questions require needs_confirmation=true")
        return self


def plan_json_schema() -> dict[str, Any]:
    return PlanV1.model_json_schema()


def validate_plan_payload(payload: str | bytes | dict[str, Any]) -> PlanV1:
    if isinstance(payload, dict):
        return PlanV1.model_validate(payload)
    return PlanV1.model_validate_json(payload)


def normalize_plan(plan: PlanV1) -> dict[str, Any]:
    return plan.model_dump(mode="json", exclude_none=False)


def validation_error_text(error: ValidationError) -> str:
    return error.json(indent=2)

