from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def convert_csv_to_json(input_file: Path) -> list[dict[str, str]]:
    with input_file.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nlsh.csv_to_json")
    parser.add_argument("input_file", type=Path)
    args = parser.parse_args(argv)

    rows = convert_csv_to_json(args.input_file)
    json.dump(rows, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
