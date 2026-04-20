from pydantic import ValidationError

from nlsh.schema import PlanV1


def test_plan_rejects_unknown_key() -> None:
    payload = {
        "version": "1",
        "steps": [
            {
                "kind": "pdf_compress",
                "input_file": "a.pdf",
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
            {"kind": "pdf_compress", "input_file": "a.pdf"},
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

