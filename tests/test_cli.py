import argparse

from nlsh.cli import command_run
from nlsh.preflight import MissingToolsError
from nlsh.schema import PlannerOutput, PlanV1


class StubPlanner:
    def __init__(self, plan: PlannerOutput) -> None:
        self._plan = plan

    def plan(self, prompt: str) -> PlannerOutput:
        return self._plan


def test_command_run_returns_2_when_required_tool_is_missing(monkeypatch) -> None:
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

    args = argparse.Namespace(
        prompt="keep paid rows",
        planner="gold",
        dataset=None,
        yes=False,
        allow_overwrite=False,
    )

    assert command_run(args) == 2
