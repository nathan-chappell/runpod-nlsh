import argparse

from nlsh.cli import command_run
from nlsh.preflight import MissingToolsError
from nlsh.schema import PlanV1


class StubPlanner:
    def __init__(self, plan: PlanV1) -> None:
        self._plan = plan

    def plan(self, prompt: str) -> PlanV1:
        return self._plan


def test_command_run_returns_2_when_required_tool_is_missing(monkeypatch) -> None:
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

    monkeypatch.setattr("nlsh.cli.load_planner", lambda planner, dataset_path=None: StubPlanner(plan))
    monkeypatch.setattr(
        "nlsh.cli.ensure_required_tools",
        lambda tools: (_ for _ in ()).throw(MissingToolsError(["mlr"])),
    )

    args = argparse.Namespace(
        prompt="keep paid rows",
        planner="gold",
        dataset=None,
        yes=False,
        allow_overwrite=False,
    )

    assert command_run(args) == 2
