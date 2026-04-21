from nlsh.compiler import compile_plan, required_tools_for_plan
from nlsh.schema import PlanV1


def test_compile_find_then_pdf_merge() -> None:
    plan = PlanV1.model_validate(
        {
            "kind": "plan",
            "steps": [
                {"kind": "find_files", "root": "./contracts", "glob": "*.pdf"},
                {"kind": "pdf_merge", "output_file": "bundle.pdf"},
            ],
        }
    )

    compiled = compile_plan(plan, python_executable="/usr/bin/python3")
    assert "mapfile -d '' -t MATCHES" in compiled.script
    assert "find ./contracts -type f -path './contracts/*.pdf' -print0" in compiled.script
    assert "qpdf --empty --pages" in compiled.script
    assert "-- bundle.pdf" in compiled.script
    assert '"${MATCHES[@]}"' in compiled.script
    assert required_tools_for_plan(plan) == ["find", "qpdf"]


def test_compile_pdf_extract_uses_qpdf() -> None:
    plan = PlanV1.model_validate(
        {
            "kind": "plan",
            "steps": [
                {
                    "kind": "pdf_extract_pages",
                    "input_file": "invoice.pdf",
                    "page_start": 2,
                    "page_end": 4,
                    "output_file": "invoice_pages.pdf",
                }
            ],
        }
    )

    compiled = compile_plan(plan)
    assert "qpdf invoice.pdf --pages . 2-4 -- invoice_pages.pdf" in compiled.script
    assert required_tools_for_plan(plan) == ["qpdf"]


def test_compile_pdf_search_uses_python_helper() -> None:
    plan = PlanV1.model_validate(
        {
            "kind": "plan",
            "steps": [
                {
                    "kind": "pdf_search_text",
                    "input_files": ["handbook.pdf"],
                    "query": "warranty",
                    "output_file": "warranty_matches.json",
                }
            ],
        }
    )

    compiled = compile_plan(plan, python_executable="/usr/bin/python3")
    assert "/usr/bin/python3 -m nlsh.pdf_search" in compiled.script
    assert "--query warranty" in compiled.script
    assert "--output warranty_matches.json" in compiled.script
    assert required_tools_for_plan(plan) == []


def test_compile_csv_to_json_then_filter_uses_jq() -> None:
    plan = PlanV1.model_validate(
        {
            "kind": "plan",
            "steps": [
                {"kind": "csv_to_json", "input_file": "orders.csv", "output_file": None},
                {
                    "kind": "json_filter",
                    "input_file": None,
                    "field": "status",
                    "operator": "eq",
                    "value": "paid",
                    "output_file": "paid_orders.json",
                },
            ],
        }
    )

    compiled = compile_plan(plan, python_executable="/usr/bin/python3")
    assert 'NLSH_TMP_JSON="$(mktemp --suffix=.json)"' in compiled.script
    assert "/usr/bin/python3 -m nlsh.csv_to_json orders.csv > $NLSH_TMP_JSON" in compiled.script
    assert "jq --arg field status --argjson value '\"paid\"'" in compiled.script
    assert "map(select(.[$field] == $value))" in compiled.script
    assert "> paid_orders.json" in compiled.script
    assert required_tools_for_plan(plan) == ["jq"]


def test_compile_find_then_csv_json_pipeline_requires_single_match() -> None:
    plan = PlanV1.model_validate(
        {
            "kind": "plan",
            "steps": [
                {"kind": "find_files", "root": "./exports", "glob": "*june*.csv"},
                {"kind": "csv_to_json", "input_file": None, "output_file": None},
                {
                    "kind": "json_filter",
                    "input_file": None,
                    "field": "region",
                    "operator": "eq",
                    "value": "eu",
                    "output_file": "june_eu.json",
                },
            ],
        }
    )

    compiled = compile_plan(plan, python_executable="/usr/bin/python3")
    assert "Expected exactly one match for csv input" in compiled.script
    assert '"${MATCHES[0]}"' in compiled.script
    assert "> $NLSH_TMP_JSON" in compiled.script
    assert "> june_eu.json" in compiled.script
    assert required_tools_for_plan(plan) == ["find", "jq"]


def test_compile_numeric_json_filter_casts_string_values() -> None:
    plan = PlanV1.model_validate(
        {
            "kind": "plan",
            "steps": [
                {
                    "kind": "json_filter",
                    "input_file": "orders.json",
                    "field": "total",
                    "operator": "gt",
                    "value": 100,
                    "output_file": "large_orders.json",
                }
            ],
        }
    )

    compiled = compile_plan(plan)
    assert "map(select((.[$field] | tonumber?) > $value))" in compiled.script


def test_compile_json_group_count_uses_jq() -> None:
    plan = PlanV1.model_validate(
        {
            "kind": "plan",
            "steps": [
                {
                    "kind": "json_group_count",
                    "input_file": "orders.json",
                    "group_by": ["status"],
                    "output_file": "counts.json",
                }
            ],
        }
    )

    compiled = compile_plan(plan)
    assert "jq --argjson fields '[\"status\"]' --arg count_field count" in compiled.script
    assert "group_by([.[$fields[]]])" in compiled.script
    assert "> counts.json" in compiled.script
    assert required_tools_for_plan(plan) == ["jq"]
