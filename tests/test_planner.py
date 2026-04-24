from pydantic import ValidationError

from nlsh.planner import _extract_json_fragment, _validate_planner_payload
from nlsh.schema import Clarification, PlanV1


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


def test_validate_planner_payload_extracts_json_when_allowed() -> None:
    payload = """
Here you go:

```json
{"kind":"plan","steps":[{"kind":"find_files","root":"./invoices","glob":"*.pdf"}]}
```
""".strip()

    plan = _validate_planner_payload(payload, extract_json_fragment=True)

    assert isinstance(plan, PlanV1)
    assert plan.steps[0].kind == "find_files"


def test_validate_planner_payload_rejects_wrapped_json_in_strict_mode() -> None:
    payload = """
Here you go:

```json
{"kind":"plan","steps":[{"kind":"find_files","root":"./invoices","glob":"*.pdf"}]}
```
""".strip()

    try:
        _validate_planner_payload(payload, extract_json_fragment=False)
    except ValidationError:
        pass
    else:
        raise AssertionError("Strict planner validation accepted wrapped JSON")


def test_validate_planner_payload_accepts_clarification_dict() -> None:
    output = _validate_planner_payload(
        {"kind": "clarification", "question": "Which page range should I extract?"},
        extract_json_fragment=False,
    )

    assert isinstance(output, Clarification)
    assert output.question == "Which page range should I extract?"
