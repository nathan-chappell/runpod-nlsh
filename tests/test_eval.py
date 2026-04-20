from pathlib import Path

from nlsh.eval import evaluate_planner
from nlsh.planner import GoldPlanner


def test_gold_eval_smoke() -> None:
    dataset_path = Path("data/test.messages.jsonl")
    planner = GoldPlanner(dataset_path=dataset_path)
    results = evaluate_planner(
        planner,
        split="test",
        dataset_path=dataset_path,
        python_executable="/usr/bin/python3",
    )

    assert results["count"] == 6
    assert results["exact_match_rate"] == 1.0
    assert results["compile_valid_rate"] == 1.0
