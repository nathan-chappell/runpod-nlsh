from pydantic import ValidationError

from nlsh.schema import Clarification, PlanV1, validate_plan_payload


def test_plan_rejects_unknown_key() -> None:
    payload = {
        "kind": "plan",
        "steps": [
            {
                "kind": "pdf_merge",
                "input_files": ["a.pdf", "b.pdf"],
                "output_file": "merged.pdf",
                "surprise": True,
            }
        ],
    }

    try:
        PlanV1.model_validate(payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("PlanV1 accepted an unexpected key")


def test_plan_allows_find_then_json_sort_pipeline() -> None:
    plan = PlanV1.model_validate(
        {
            "kind": "plan",
            "steps": [
                {"kind": "find_files", "root": "./exports", "glob": "*revenue*.json"},
                {
                    "kind": "json_sort",
                    "field": "due_date",
                    "descending": True,
                    "output_file": "revenue_sorted.json",
                },
            ],
        }
    )

    assert len(plan.steps) == 2


def test_plan_rejects_three_step_csv_json_pipeline() -> None:
    payload = {
        "kind": "plan",
        "steps": [
            {"kind": "find_files", "root": "./exports", "glob": "*june*.csv"},
            {"kind": "csv_to_json"},
            {
                "kind": "json_filter",
                "field": "region",
                "operator": "eq",
                "value": "eu",
                "output_file": "june_eu.json",
            },
        ],
    }

    try:
        PlanV1.model_validate(payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("PlanV1 accepted an unsupported 3-step pipeline")


def test_plan_rejects_null_handoff_fields_in_find_pipeline() -> None:
    payload = {
        "kind": "plan",
        "steps": [
            {"kind": "find_files", "root": "./contracts", "glob": "*.pdf"},
            {
                "kind": "pdf_merge",
                "input_files": None,
                "output_file": "bundle.pdf",
            },
        ],
    }

    try:
        PlanV1.model_validate(payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("PlanV1 accepted an explicit null pipeline handoff")


def test_plan_rejects_single_step_csv_to_json_without_input_file() -> None:
    payload = {
        "kind": "plan",
        "steps": [{"kind": "csv_to_json", "output_file": "orders.json"}],
    }

    try:
        PlanV1.model_validate(payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("PlanV1 accepted csv_to_json without input_file")


def test_pdf_page_numbers_must_be_positive() -> None:
    payload = {
        "kind": "plan",
        "steps": [
            {
                "kind": "pdf_extract_pages",
                "input_file": "a.pdf",
                "page_start": 0,
                "page_end": 2,
                "output_file": "excerpt.pdf",
            }
        ],
    }

    try:
        PlanV1.model_validate(payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("PlanV1 accepted a non-positive PDF page")


def test_validate_plan_payload_accepts_clarification() -> None:
    output = validate_plan_payload(
        {"kind": "clarification", "question": "Which PDF files should I merge?"}
    )

    assert isinstance(output, Clarification)
    assert output.question == "Which PDF files should I merge?"
