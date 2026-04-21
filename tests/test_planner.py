from nlsh.planner import _apply_prompt_hints, _coerce_plan_payload, _extract_json_fragment
from nlsh.schema import PlanV1


def test_extract_json_fragment_handles_markdown_fences() -> None:
    payload = """
Here you go:

```json
{"version":"1","steps":[{"kind":"find_files","roots":["./invoices"]}],"needs_confirmation":false,"questions":[],"risk_level":"low","notes":[]}
```
""".strip()

    extracted = _extract_json_fragment(payload)

    assert extracted.startswith("{")
    assert extracted.endswith("}")


def test_coerce_plan_payload_flattens_nested_find_files_step() -> None:
    payload = {
        "steps": [
            {
                "find_files": {
                    "path": "./invoices",
                    "depth": 2,
                    "pattern": "*.pdf",
                }
            }
        ]
    }

    plan = PlanV1.model_validate(_coerce_plan_payload(payload))
    step = plan.steps[0]

    assert step.kind == "find_files"
    assert step.roots == ["./invoices"]
    assert step.max_depth == 2
    assert step.name_pattern == "*.pdf"


def test_coerce_plan_payload_infers_find_files_kind_from_fields() -> None:
    payload = {
        "version": "1",
        "steps": [
            {
                "name_pattern": "find_files",
                "roots": ["./invoices"],
                "path_contains": "invoices",
            }
        ],
    }

    plan = PlanV1.model_validate(_coerce_plan_payload(payload))
    step = plan.steps[0]

    assert step.kind == "find_files"
    assert step.roots == ["./invoices"]
    assert step.path_contains == "invoices"
    assert step.name_pattern is None


def test_coerce_plan_payload_removes_step_level_plan_fields() -> None:
    payload = {
        "version": "1",
        "steps": [
            {
                "kind": "find_files",
                "roots": ["./exports"],
                "file_type": "csv",
                "needs_confirmation": False,
                "risk_level": "low",
            }
        ],
        "needs_confirmation": False,
        "questions": [],
        "risk_level": "medium",
        "notes": [],
    }

    plan = PlanV1.model_validate(_coerce_plan_payload(payload))
    step = plan.steps[0]

    assert step.kind == "find_files"
    assert step.extension == ".csv"
    assert step.file_type == "file"


def test_apply_prompt_hints_fills_missing_find_fields() -> None:
    plan = PlanV1.model_validate(
        {
            "version": "1",
            "steps": [
                {
                    "kind": "find_files",
                    "roots": ["./invoices", "extension"],
                    "extension": None,
                    "path_contains": "",
                    "max_depth": None,
                    "file_type": "file",
                }
            ],
            "needs_confirmation": False,
            "questions": [],
            "risk_level": "medium",
            "notes": [],
        }
    )

    hinted = _apply_prompt_hints(plan, "find every pdf under ./invoices no deeper than two levels")
    step = hinted.steps[0]

    assert hinted.risk_level == "low"
    assert step.roots == ["./invoices"]
    assert step.extension == ".pdf"
    assert step.path_contains is None
    assert step.max_depth == 2


def test_coerce_plan_payload_renames_question_field() -> None:
    payload = {
        "steps": [
            {
                "kind": "media_transcode_for_tv",
                "input_file": None,
                "output_file": "living_room_copy.mp4",
            }
        ],
        "needs_confirmation": True,
        "questions": [
            {
                "field_path": "steps[0].input_file",
                "question": "Which movie file should be converted?",
            }
        ],
    }

    plan = PlanV1.model_validate(_coerce_plan_payload(payload))

    assert plan.questions[0].prompt == "Which movie file should be converted?"


def test_coerce_plan_payload_normalizes_sparse_question_and_string_notes() -> None:
    payload = {
        "steps": [
            {
                "kind": "media_transcode_for_tv",
                "input_file": None,
                "output_file": "living_room_copy.mp4",
            }
        ],
        "needs_confirmation": True,
        "questions": [{"field_path": "input_file"}],
        "notes": "Ensure the original file is backed up before conversion.",
    }

    plan = PlanV1.model_validate(_coerce_plan_payload(payload))

    assert plan.questions[0].field_path == "steps[0].input_file"
    assert plan.questions[0].prompt == "Please provide steps[0].input_file."
    assert plan.notes == ["Ensure the original file is backed up before conversion."]


def test_apply_prompt_hints_marks_missing_media_input() -> None:
    plan = PlanV1.model_validate(
        {
            "version": "1",
            "steps": [
                {
                    "kind": "media_transcode_for_tv",
                    "input_file": "movie.mp4",
                    "output_file": "living_room_copy.mp4",
                }
            ],
            "needs_confirmation": False,
            "questions": [],
            "risk_level": "low",
            "notes": [],
        }
    )

    hinted = _apply_prompt_hints(plan, "convert this movie for the TV and name it living_room_copy.mp4")
    step = hinted.steps[0]

    assert hinted.needs_confirmation is True
    assert hinted.risk_level == "medium"
    assert step.input_file is None
    assert step.output_file == "living_room_copy.mp4"
    assert hinted.questions[0].field_path == "steps[0].input_file"
    assert hinted.questions[0].prompt == "Which media file should be transcoded for TV playback?"
    assert hinted.notes == ["The source media file is missing."]


def test_apply_prompt_hints_canonicalizes_existing_media_question() -> None:
    plan = PlanV1.model_validate(
        {
            "version": "1",
            "steps": [
                {
                    "kind": "media_transcode_for_tv",
                    "input_file": None,
                    "output_file": "living_room_copy.mp4",
                }
            ],
            "needs_confirmation": True,
            "questions": [
                {
                    "field_path": "steps[0].input_file",
                    "prompt": "Please provide steps[0].input_file.",
                }
            ],
            "risk_level": "medium",
            "notes": ["Ensure the original file is backed up before conversion."],
        }
    )

    hinted = _apply_prompt_hints(plan, "convert this movie for the TV and name it living_room_copy.mp4")

    assert hinted.questions[0].prompt == "Which media file should be transcoded for TV playback?"
    assert hinted.notes == ["The source media file is missing."]


def test_apply_prompt_hints_normalizes_find_then_filter_plan() -> None:
    plan = PlanV1.model_validate(
        {
            "version": "1",
            "steps": [
                {
                    "kind": "find_files",
                    "roots": ["./exports"],
                    "extension": "csv",
                    "path_contains": "june",
                    "file_type": "file",
                },
                {
                    "kind": "csv_filter_rows",
                    "input_file": "{{steps[0].output_file}}",
                    "filter_column": "region",
                    "filter_operator": "eq",
                    "filter_value": "eu",
                    "output_file": "filtered_june_eu.csv",
                },
            ],
            "needs_confirmation": False,
            "questions": [],
            "risk_level": "medium",
            "notes": [],
        }
    )

    hinted = _apply_prompt_hints(
        plan,
        "find the csv in ./exports with june in the path and keep only rows where region is eu",
    )
    find_step = hinted.steps[0]
    filter_step = hinted.steps[1]

    assert find_step.extension == ".csv"
    assert filter_step.input_file is None
    assert filter_step.output_file == "june_filtered.csv"


def test_apply_prompt_hints_marks_missing_join_keys() -> None:
    plan = PlanV1.model_validate(
        {
            "version": "1",
            "steps": [
                {
                    "kind": "csv_join",
                    "left_file": "customers.csv",
                    "right_file": "orders.csv",
                    "join_keys": ["customer_id"],
                    "output_file": "customers_joined.csv",
                }
            ],
            "needs_confirmation": True,
            "questions": [
                {
                    "field_path": "steps[0].left_file",
                    "prompt": "Which file should be used as the left input for the join operation?",
                },
                {
                    "field_path": "steps[0].join_keys",
                    "prompt": "What is the join key column name in both customers.csv and orders.csv?",
                },
            ],
            "risk_level": "medium",
            "notes": [],
        }
    )

    hinted = _apply_prompt_hints(plan, "join customers.csv and orders.csv")
    step = hinted.steps[0]

    assert step.join_keys is None
    assert hinted.needs_confirmation is True
    assert [question.model_dump(mode="json") for question in hinted.questions] == [
        {
            "field_path": "steps[0].join_keys",
            "prompt": "Which column or columns should be used as join keys? Separate multiple keys with commas.",
            "expected_type": "string_list",
            "required": True,
        }
    ]
    assert hinted.notes == ["A join key is required before compilation."]
