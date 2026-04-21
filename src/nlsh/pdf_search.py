from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def _excerpt(text: str, start: int, end: int, context_chars: int) -> str:
    left = max(0, start - context_chars)
    right = min(len(text), end + context_chars)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def search_pdf(path: Path, query: str, context_chars: int) -> list[dict[str, Any]]:
    lowered_query = query.lower()
    matches: list[dict[str, Any]] = []
    reader = PdfReader(str(path))
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        lowered_text = text.lower()
        match_index = lowered_text.find(lowered_query)
        if match_index == -1:
            continue
        matches.append(
            {
                "file": str(path),
                "page": index,
                "query": query,
                "text_excerpt": _excerpt(
                    text,
                    match_index,
                    match_index + len(query),
                    context_chars,
                ),
            }
        )
    return matches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nlsh.pdf_search")
    parser.add_argument("--query", required=True)
    parser.add_argument("--context-chars", type=int, default=160)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("input_files", nargs="+", type=Path)
    args = parser.parse_args(argv)

    matches: list[dict[str, Any]] = []
    for input_file in args.input_files:
        matches.extend(search_pdf(input_file, args.query, args.context_chars))

    args.output.write_text(json.dumps(matches, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
