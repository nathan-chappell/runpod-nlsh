import json
from pathlib import Path

from nlsh.dataio import load_jsonl, materialize_dataset_splits, partition_records_three_way


def test_partition_records_three_way_keeps_disjoint_splits() -> None:
    records = [
        {"focus": f"example-{index}", "prompt": f"prompt-{index}", "plan": {"kind": "plan", "steps": []}}
        for index in range(5)
    ]

    train_records, eval_records, test_records = partition_records_three_way(records)

    assert len(train_records) == 3
    assert len(eval_records) == 1
    assert len(test_records) == 1
    encoded = lambda rows: {json.dumps(row, sort_keys=True) for row in rows}
    assert encoded(train_records).isdisjoint(encoded(eval_records))
    assert encoded(train_records).isdisjoint(encoded(test_records))
    assert encoded(eval_records).isdisjoint(encoded(test_records))


def test_materialized_dataset_splits_cover_canonical_dataset(tmp_path: Path) -> None:
    source_dir = Path("data/samples")
    output_dir = tmp_path / "splits"

    manifest = materialize_dataset_splits(source_dir, output_dir)

    train_records = load_jsonl(output_dir / "train")
    eval_records = load_jsonl(output_dir / "eval")
    test_records = load_jsonl(output_dir / "test")
    source_records = load_jsonl(source_dir)

    assert manifest["splits"] == {
        "train": len(train_records),
        "eval": len(eval_records),
        "test": len(test_records),
    }
    assert len(source_records) == len(train_records) + len(eval_records) + len(test_records)
    encoded = lambda rows: {json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows}
    assert encoded(train_records).isdisjoint(encoded(eval_records))
    assert encoded(train_records).isdisjoint(encoded(test_records))
    assert encoded(eval_records).isdisjoint(encoded(test_records))


def test_materialized_dataset_splits_rejects_source_as_output(tmp_path: Path) -> None:
    source_dir = tmp_path / "samples"
    source_dir.mkdir()
    (source_dir / "examples.jsonl").write_text(
        '{"focus":"a","prompt":"a","plan":{"kind":"plan","steps":[]}}\n'
        '{"focus":"b","prompt":"b","plan":{"kind":"plan","steps":[]}}\n'
        '{"focus":"c","prompt":"c","plan":{"kind":"plan","steps":[]}}\n',
        encoding="utf-8",
    )

    try:
        materialize_dataset_splits(source_dir, source_dir)
    except ValueError as exc:
        assert "output_dir must be different from source_dir" in str(exc)
    else:  # pragma: no cover - defensive assertion.
        raise AssertionError("expected materialize_dataset_splits to reject identical source/output paths")
