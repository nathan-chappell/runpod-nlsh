#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import typer

from nlsh.dataio import DEFAULT_EVAL_FRACTION, DEFAULT_SPLIT_ROOT, DEFAULT_TEST_FRACTION, materialize_dataset_splits

app = typer.Typer(
    add_completion=False,
    help="Create deterministic train/eval/test dataset splits from the canonical data/samples tree.",
    rich_markup_mode=None,
)


@app.command()
def run(
    source: Path = typer.Option(Path("data/samples"), help="Canonical JSONL source directory."),
    output: Path = typer.Option(DEFAULT_SPLIT_ROOT, help="Directory where train/eval/test splits will be written."),
    eval_fraction: float = typer.Option(DEFAULT_EVAL_FRACTION, help="Fraction of each source file reserved for eval."),
    test_fraction: float = typer.Option(DEFAULT_TEST_FRACTION, help="Fraction of each source file reserved for test."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the split manifest without writing files."),
) -> None:
    if dry_run:
        with TemporaryDirectory(prefix="nlsh-splits-") as temp_dir:
            manifest = materialize_dataset_splits(
                source,
                Path(temp_dir),
                eval_fraction=eval_fraction,
                test_fraction=test_fraction,
            )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    manifest = materialize_dataset_splits(
        source,
        output,
        eval_fraction=eval_fraction,
        test_fraction=test_fraction,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
