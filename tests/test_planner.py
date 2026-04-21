from nlsh.planner import _coerce_plan_payload, _extract_json_fragment
from nlsh.schema import Clarification, PlanV1, validate_plan_payload


def test_extract_json_fragment_handles_markdown_fences() -> None:
    payload = """
Here you go:

```json
{"kind":"plan","steps":[{"kind":"find_files","root":"./invoices","glob":"*.pdf"}]}
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
    assert step.root == "./invoices"
    assert step.glob == "*.pdf"
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
    assert step.root == "./exports"
    assert step.glob == "*.csv"


def test_coerce_plan_payload_converts_clarifying_question() -> None:
    payload = {"clarifying_question": "Which page range should I extract?"}

    output = validate_plan_payload(_coerce_plan_payload(payload))

    assert isinstance(output, Clarification)
    assert output.question == "Which page range should I extract?"


def test_coerce_plan_payload_defaults_missing_kind_to_plan() -> None:
    payload = {
        "steps": [
            {
                "kind": "pdf_extract_pages",
                "input_file": "report.pdf",
                "page_start": 1,
                "page_end": 5,
                "output_file": "report_excerpt.pdf",
            }
        ],
    }

    plan = PlanV1.model_validate(_coerce_plan_payload(payload))

    assert plan.kind == "plan"
