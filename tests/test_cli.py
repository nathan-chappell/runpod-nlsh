import json
from pathlib import Path

from typer.testing import CliRunner

from nlsh.cli import ProbeMode, _sample_probe_records, app, probe_live_dataset, run_prompt
from nlsh.preflight import MissingToolsError
from nlsh.schema import PlannerOutput, PlanV1

RUNNER = CliRunner()


class StubPlanner:
    def __init__(self, plan: PlannerOutput) -> None:
        self._plan = plan

    def plan(self, prompt: str) -> PlannerOutput:
        return self._plan


def test_cli_help_lists_existing_commands_and_probe_live() -> None:
    result = RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "plan" in result.stdout
    assert "run" in result.stdout
    assert "compile" in result.stdout
    assert "eval" in result.stdout
    assert "probe-live" in result.stdout


def test_run_prompt_returns_2_when_required_tool_is_missing(monkeypatch) -> None:
    plan = PlanV1.model_validate(
        {
            "kind": "plan",
            "steps": [
                {
                    "kind": "json_filter",
                    "input_file": "orders.json",
                    "field": "status",
                    "operator": "eq",
                    "value": "paid",
                    "output_file": "paid_orders.json",
                }
            ],
        }
    )

    monkeypatch.setattr("nlsh.cli.load_planner", lambda planner, dataset_path=None: StubPlanner(plan))
    monkeypatch.setattr(
        "nlsh.cli.ensure_required_tools",
        lambda tools: (_ for _ in ()).throw(MissingToolsError(["jq"])),
    )

    assert run_prompt(
        "keep paid rows",
        planner_name="gold",
        dataset_path=None,
        yes=False,
        allow_overwrite=False,
    ) == 2


def test_sample_probe_records_is_deterministic() -> None:
    records = [{"prompt": f"prompt-{index}"} for index in range(5)]

    first = _sample_probe_records(records, count=3, seed=123)
    second = _sample_probe_records(records, count=3, seed=123)

    assert [item["prompt"] for item in first] == [item["prompt"] for item in second]


def test_probe_live_dataset_reports_expected_and_actual(tmp_path: Path, monkeypatch, capsys) -> None:
    dataset_path = tmp_path / "samples.jsonl"
    record = {
        "focus": "simple csv conversion",
        "prompt": "convert orders.csv to JSON in orders.json",
        "messages": [
            {
                "role": "developer",
                "content": "Translate the request into PlanV1 JSON. Return either {\"kind\":\"plan\",\"steps\":[...]} or {\"kind\":\"clarification\",\"question\":\"...\"}. Never emit shell.",
            },
            {"role": "user", "content": "convert orders.csv to JSON in orders.json"},
            {
                "role": "assistant",
                "content": "{\"kind\":\"plan\",\"steps\":[{\"kind\":\"csv_to_json\",\"input_file\":\"orders.csv\",\"output_file\":\"orders.json\"}]}",
            },
        ],
        "plan": {
            "kind": "plan",
            "steps": [
                {
                    "kind": "csv_to_json",
                    "input_file": "orders.csv",
                    "output_file": "orders.json",
                }
            ],
        },
    }
    dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_chat_completion_text(**kwargs: object) -> str:
        captured["messages"] = kwargs["messages"]
        captured["response_format"] = kwargs["response_format"]
        return "{\"kind\":\"plan\",\"steps\":[{\"kind\":\"csv_to_json\",\"input_file\":\"orders.csv\",\"output_file\":\"orders.json\"}]}"

    monkeypatch.setattr("nlsh.cli.chat_completion_text", fake_chat_completion_text)

    exit_code = probe_live_dataset(dataset_path=dataset_path, count=1, seed=7, mode=ProbeMode.runtime)
    output = capsys.readouterr().out

    assert exit_code == 0
    assert isinstance(captured["messages"], list)
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["content"] == "convert orders.csv to JSON in orders.json"
    assert captured["response_format"] is not None
    assert "Expected assistant:" in output
    assert "Actual assistant:" in output
    assert "Exact match: yes" in output
    assert "Compile valid: yes" in output
    assert '"transport_or_parse_failures": 0' in output


def test_probe_live_dataset_returns_1_on_parse_error(tmp_path: Path, monkeypatch, capsys) -> None:
    dataset_path = tmp_path / "samples.jsonl"
    record = {
        "focus": "simple csv conversion",
        "prompt": "convert orders.csv to JSON in orders.json",
        "messages": [
            {
                "role": "developer",
                "content": "Translate the request into PlanV1 JSON. Return either {\"kind\":\"plan\",\"steps\":[...]} or {\"kind\":\"clarification\",\"question\":\"...\"}. Never emit shell.",
            },
            {"role": "user", "content": "convert orders.csv to JSON in orders.json"},
            {
                "role": "assistant",
                "content": "{\"kind\":\"plan\",\"steps\":[{\"kind\":\"csv_to_json\",\"input_file\":\"orders.csv\",\"output_file\":\"orders.json\"}]}",
            },
        ],
        "plan": {
            "kind": "plan",
            "steps": [
                {
                    "kind": "csv_to_json",
                    "input_file": "orders.csv",
                    "output_file": "orders.json",
                }
            ],
        },
    }
    dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    monkeypatch.setattr("nlsh.cli.chat_completion_text", lambda **_kwargs: "not json at all")

    exit_code = probe_live_dataset(dataset_path=dataset_path, count=1, seed=7, mode=ProbeMode.runtime)
    output = capsys.readouterr()

    assert exit_code == 1
    assert "Parse error:" in output.err


def test_probe_live_dataset_replay_messages_mode_uses_record_messages(tmp_path: Path, monkeypatch, capsys) -> None:
    dataset_path = tmp_path / "samples.jsonl"
    record = {
        "focus": "simple csv conversion",
        "prompt": "convert orders.csv to JSON in orders.json",
        "messages": [
            {
                "role": "developer",
                "content": "Translate the request into PlanV1 JSON. Return either {\"kind\":\"plan\",\"steps\":[...]} or {\"kind\":\"clarification\",\"question\":\"...\"}. Never emit shell.",
            },
            {"role": "user", "content": "convert orders.csv to JSON in orders.json"},
            {
                "role": "assistant",
                "content": "{\"kind\":\"plan\",\"steps\":[{\"kind\":\"csv_to_json\",\"input_file\":\"orders.csv\",\"output_file\":\"orders.json\"}]}",
            },
        ],
        "plan": {
            "kind": "plan",
            "steps": [
                {
                    "kind": "csv_to_json",
                    "input_file": "orders.csv",
                    "output_file": "orders.json",
                }
            ],
        },
    }
    dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_chat_completion_text(**kwargs: object) -> str:
        captured["messages"] = kwargs["messages"]
        captured["response_format"] = kwargs["response_format"]
        return "{\"kind\":\"plan\",\"steps\":[{\"kind\":\"csv_to_json\",\"input_file\":\"orders.csv\",\"output_file\":\"orders.json\"}]}"

    monkeypatch.setattr("nlsh.cli.chat_completion_text", fake_chat_completion_text)

    exit_code = probe_live_dataset(dataset_path=dataset_path, count=1, seed=7, mode=ProbeMode.replay_messages)
    _ = capsys.readouterr()

    assert exit_code == 0
    assert captured["messages"] == record["messages"][:-1]
    assert captured["response_format"] is None
