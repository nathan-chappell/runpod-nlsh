from nlsh.compiler import compile_plan
from nlsh.schema import PlanV1


def test_compile_find_then_pdf_combine() -> None:
    plan = PlanV1.model_validate(
        {
            "version": "1",
            "steps": [
                {"kind": "find_files", "roots": ["./contracts"], "extension": ".pdf"},
                {"kind": "pdf_combine", "output_file": "bundle.pdf"},
            ],
            "needs_confirmation": False,
            "questions": [],
            "risk_level": "medium",
            "notes": [],
        }
    )

    compiled = compile_plan(plan, python_executable="/usr/bin/python3")
    assert "mapfile -d '' -t MATCHES" in compiled.script
    assert "qpdf --empty --pages" in compiled.script
    assert "-- bundle.pdf" in compiled.script
    assert '"${MATCHES[@]}"' in compiled.script


def test_compile_csv_filter_uses_mlr() -> None:
    plan = PlanV1.model_validate(
        {
            "version": "1",
            "steps": [
                {
                    "kind": "csv_filter_rows",
                    "input_file": "orders.csv",
                    "filter_column": "status",
                    "filter_operator": "eq",
                    "filter_value": "paid",
                    "output_file": "paid.csv",
                }
            ],
            "needs_confirmation": False,
            "questions": [],
            "risk_level": "medium",
            "notes": [],
        }
    )

    compiled = compile_plan(plan, python_executable="/usr/bin/python3")
    assert "mlr --csv filter" in compiled.script
    assert 'status"] == "paid"' in compiled.script
    assert "nlsh.backends.csv_ops" not in compiled.script


def test_compile_find_then_media_clip_requires_single_match() -> None:
    plan = PlanV1.model_validate(
        {
            "version": "1",
            "steps": [
                {"kind": "find_files", "roots": ["./media"], "extension": ".mp4"},
                {
                    "kind": "media_clip",
                    "start_seconds": 0,
                    "duration_seconds": 30,
                    "output_file": "preview.mp4",
                },
            ],
            "needs_confirmation": False,
            "questions": [],
            "risk_level": "medium",
            "notes": [],
        }
    )

    compiled = compile_plan(plan, python_executable="/usr/bin/python3")
    assert "Expected exactly one match for media input" in compiled.script
    assert "ffmpeg" in compiled.script


def test_compile_pdf_rotate_uses_qpdf() -> None:
    plan = PlanV1.model_validate(
        {
            "version": "1",
            "steps": [
                {
                    "kind": "pdf_rotate",
                    "input_file": "invoice.pdf",
                    "rotation_degrees": 90,
                    "output_file": "invoice_fixed.pdf",
                }
            ],
            "needs_confirmation": False,
            "questions": [],
            "risk_level": "medium",
            "notes": [],
        }
    )

    compiled = compile_plan(plan)
    assert "qpdf invoice.pdf --rotate=+90:1-z -- invoice_fixed.pdf" in compiled.script
    assert "nlsh.backends.pdf_ops" not in compiled.script
