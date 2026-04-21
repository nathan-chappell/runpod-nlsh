from pydantic import ValidationError

from nlsh.schema import PlanV1


def test_plan_rejects_unknown_key() -> None:
    payload = {
        "version": "1",
        "steps": [
            {
                "kind": "pdf_merge",
                "input_files": ["a.pdf", "b.pdf"],
                "output_file": "merged.pdf",
                "surprise": True,
            }
        ],
        "needs_confirmation": False,
        "questions": [],
        "risk_level": "medium",
        "notes": [],
    }

    try:
        PlanV1.model_validate(payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("PlanV1 accepted an unexpected key")


def test_plan_requires_find_as_first_step_for_two_step_plan() -> None:
    payload = {
        "version": "1",
        "steps": [
            {"kind": "pdf_merge", "input_files": ["a.pdf"], "output_file": "merged.pdf"},
            {"kind": "find_files", "roots": ["."], "extension": ".pdf"},
        ],
        "needs_confirmation": False,
        "questions": [],
        "risk_level": "medium",
        "notes": [],
    }

    try:
        PlanV1.model_validate(payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("PlanV1 accepted an invalid step ordering")


def test_plan_allows_three_step_csv_json_pipeline() -> None:
    plan = PlanV1.model_validate(
        {
            "version": "1",
            "steps": [
                {"kind": "find_files", "roots": ["./exports"], "extension": ".csv"},
                {"kind": "csv_to_json", "input_file": None, "output_file": None},
                {
                    "kind": "json_filter",
                    "input_file": None,
                    "field": "region",
                    "operator": "eq",
                    "value": "eu",
                    "output_file": "eu.json",
                },
            ],
            "needs_confirmation": False,
            "questions": [],
            "risk_level": "medium",
            "notes": [],
        }
    )

    assert len(plan.steps) == 3


def test_pdf_page_numbers_must_be_positive() -> None:
    payload = {
        "version": "1",
        "steps": [
            {
                "kind": "pdf_extract_pages",
                "input_file": "a.pdf",
                "page_start": 0,
                "page_end": 2,
                "output_file": "excerpt.pdf",
            }
        ],
        "needs_confirmation": False,
        "questions": [],
        "risk_level": "medium",
        "notes": [],
    }

    try:
        PlanV1.model_validate(payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("PlanV1 accepted a non-positive PDF page")
