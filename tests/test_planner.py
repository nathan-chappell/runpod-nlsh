from nlsh.planner import _coerce_plan_payload, _extract_json_fragment
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


def test_coerce_plan_payload_flattens_nested_json_filter_step() -> None:
    payload = {
        "steps": [
            {
                "json_filter": {
                    "input_file": "orders.json",
                    "field": "status",
                    "operator": "eq",
                    "value": "paid",
                    "output_file": "paid_orders.json",
                }
            }
        ]
    }

    plan = PlanV1.model_validate(_coerce_plan_payload(payload))
    step = plan.steps[0]

    assert step.kind == "json_filter"
    assert step.input_file == "orders.json"
    assert step.field == "status"
    assert step.value == "paid"


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


def test_coerce_plan_payload_renames_question_field() -> None:
    payload = {
        "steps": [
            {
                "kind": "pdf_extract_pages",
                "input_file": "report.pdf",
                "page_start": None,
                "page_end": None,
                "output_file": "report_excerpt.pdf",
            }
        ],
        "needs_confirmation": True,
        "questions": [
            {
                "field_path": "steps[0].page_start",
                "question": "What is the first page to extract?",
            }
        ],
    }

    plan = PlanV1.model_validate(_coerce_plan_payload(payload))

    assert plan.questions[0].prompt == "What is the first page to extract?"


def test_coerce_plan_payload_normalizes_sparse_question_and_string_notes() -> None:
    payload = {
        "steps": [
            {
                "kind": "pdf_extract_pages",
                "input_file": "report.pdf",
                "page_start": None,
                "page_end": 5,
                "output_file": "report_excerpt.pdf",
            }
        ],
        "needs_confirmation": True,
        "questions": [{"field_path": "page_start", "expected_type": "integer"}],
        "notes": "A page range is required before compilation.",
    }

    plan = PlanV1.model_validate(_coerce_plan_payload(payload))

    assert plan.questions[0].field_path == "steps[0].page_start"
    assert plan.questions[0].prompt == "Please provide steps[0].page_start."
    assert plan.notes == ["A page range is required before compilation."]
