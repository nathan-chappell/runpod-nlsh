from pathlib import Path

from nlsh.dataio import load_jsonl
from nlsh.eval import evaluate_planner
from nlsh.planner import GoldPlanner


def test_gold_eval_smoke() -> None:
    dataset_path = Path("data/samples")
    planner = GoldPlanner(dataset_path=dataset_path)
    results = evaluate_planner(
        planner,
        dataset_path=dataset_path,
        label="samples",
        python_executable="/usr/bin/python3",
    )

    assert results["count"] == len(load_jsonl(dataset_path))
    assert results["exact_match_rate"] == 1.0
    assert results["compile_valid_rate"] == 1.0
