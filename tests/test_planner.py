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
