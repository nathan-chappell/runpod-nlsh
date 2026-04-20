import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple
from openai import OpenAI


ACTION_NAMES = [
    "pdf_combine",
    "pdf_compress",
    "pdf_extract_pages",
    "pdf_rotate",
    "video_transcode_for_tv",
    "video_extract_audio_mp3",
    "video_clip",
    "csv_join",
    "csv_filter_rows",
]

ARG_PROPERTIES = {
    "input_pattern": {"type": ["string", "null"]},
    "input_file": {"type": ["string", "null"]},
    "page_start": {"type": ["integer", "null"]},
    "page_end": {"type": ["integer", "null"]},
    "rotation_degrees": {"type": ["integer", "null"], "enum": [90, 180, 270, None]},
    "left_file": {"type": ["string", "null"]},
    "right_file": {"type": ["string", "null"]},
    "join_keys": {"type": ["array", "null"], "items": {"type": "string"}},
    "filter_column": {"type": ["string", "null"]},
    "filter_operator": {
        "type": ["string", "null"],
        "enum": ["eq", "ne", "gt", "gte", "lt", "lte", "contains", None],
    },
    "filter_value": {"type": ["string", "number", "boolean", "null"]},
    "start_seconds": {"type": ["integer", "null"]},
    "duration_seconds": {"type": ["integer", "null"]},
    "output_file": {"type": ["string", "null"]},
}

MISSING_FIELD_NAMES = [
    "input_pattern",
    "input_file",
    "page_start",
    "page_end",
    "rotation_degrees",
    "left_file",
    "right_file",
    "join_keys",
    "filter_column",
    "filter_operator",
    "filter_value",
    "start_seconds",
    "duration_seconds",
]


def expected_action(
    action: str,
    *,
    needs_confirmation: bool = False,
    missing_fields: list[str] | None = None,
    **args: Any,
) -> Dict[str, Any]:
    payload_args = {key: None for key in ARG_PROPERTIES}
    payload_args.update(args)
    return {
        "action": action,
        "needs_confirmation": needs_confirmation,
        "missing_fields": list(missing_fields or []),
        "args": payload_args,
    }


examples = [
    {
        "prompt": "combine the files that end with _foo into a single pdf",
        "expected": expected_action("pdf_combine", input_pattern="*_foo.pdf"),
    },
    {
        "prompt": "combine all PDF files in this folder into one file called everything.pdf",
        "expected": expected_action(
            "pdf_combine",
            input_pattern="*.pdf",
            output_file="everything.pdf",
        ),
    },
    {
        "prompt": "compress scanned_contract.pdf so it's easier to email",
        "expected": expected_action("pdf_compress", input_file="scanned_contract.pdf"),
    },
    {
        "prompt": "extract pages 2 through 5 from report.pdf into a new file",
        "expected": expected_action(
            "pdf_extract_pages",
            input_file="report.pdf",
            page_start=2,
            page_end=5,
        ),
    },
    {
        "prompt": "rotate invoice.pdf clockwise and save it as invoice_fixed.pdf",
        "expected": expected_action(
            "pdf_rotate",
            input_file="invoice.pdf",
            rotation_degrees=90,
            output_file="invoice_fixed.pdf",
        ),
    },
    {
        "prompt": "convert vacation.mkv to a format that works on most TVs",
        "expected": expected_action(
            "video_transcode_for_tv",
            input_file="vacation.mkv",
        ),
    },
    {
        "prompt": "extract the audio from lecture.mp4 into an mp3",
        "expected": expected_action(
            "video_extract_audio_mp3",
            input_file="lecture.mp4",
        ),
    },
    {
        "prompt": "clip the first 90 seconds from movie.mp4 and save it as intro.mp4",
        "expected": expected_action(
            "video_clip",
            input_file="movie.mp4",
            start_seconds=0,
            duration_seconds=90,
            output_file="intro.mp4",
        ),
    },
    {
        "prompt": "join sales.csv with customers.csv on customer_id",
        "expected": expected_action(
            "csv_join",
            left_file="sales.csv",
            right_file="customers.csv",
            join_keys=["customer_id"],
        ),
    },
    {
        "prompt": "keep only the rows in orders.csv where status is paid",
        "expected": expected_action(
            "csv_filter_rows",
            input_file="orders.csv",
            filter_column="status",
            filter_operator="eq",
            filter_value="paid",
        ),
    },
    # Ambiguous
    {
        "prompt": "convert this movie to a format that works on my tv",
        "expected": expected_action(
            "video_transcode_for_tv",
            needs_confirmation=True,
            missing_fields=["input_file"],
        ),
    },
    {
        "prompt": "join these two csvs together",
        "expected": expected_action(
            "csv_join",
            needs_confirmation=True,
            missing_fields=["left_file", "right_file", "join_keys"],
        ),
    },
]


MODEL = os.environ.get("NLSH_MODEL", "openai/gpt-oss-20b:together")
RESULT_PATH = Path("artifacts/hf_test_result.txt")

SYSTEM_PROMPT = """
Return only valid JSON matching the schema.

Rules:
- Use exactly one action from the schema.
- Put arguments only in args.
- Never invent placeholder filenames like "movie_file".
- Never emit shell commands.
- Never emit extra keys.
- If required information is missing, set needs_confirmation=true and list the missing fields.
- Because the schema is strict, include every arg key and use null for any value that is unknown, irrelevant, or not explicitly requested.
"""

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)
schema = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTION_NAMES},
        "needs_confirmation": {"type": "boolean"},
        "missing_fields": {
            "type": "array",
            "items": {"type": "string", "enum": MISSING_FIELD_NAMES},
        },
        "args": {
            "type": "object",
            "properties": ARG_PROPERTIES,
            "required": list(ARG_PROPERTIES),
            "additionalProperties": False,
        },
    },
    "required": ["action", "needs_confirmation", "missing_fields", "args"],
    "additionalProperties": False,
}


def extract_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
        return "\n".join(parts)

    return str(content)


def normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    return obj


def diff_expected_vs_actual(
    expected: Dict[str, Any], actual: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    diffs = {}
    for key in sorted(set(expected) | set(actual)):
        ev = expected.get(key, "<missing>")
        av = actual.get(key, "<missing>")
        if normalize(ev) != normalize(av):
            diffs[key] = {"expected": ev, "actual": av}
    return diffs


def run_one(prompt: str) -> Tuple[Dict[str, Any], str]:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "shell_action",
                "strict": True,
                "schema": schema,
            },
        },
    )

    raw_text = extract_text(resp.choices[0].message).strip()
    parsed = json.loads(raw_text)
    return parsed, raw_text


def main() -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as result_file:
        def emit(*parts: object, sep: str = " ", end: str = "\n") -> None:
            text = sep.join(str(part) for part in parts) + end
            print(*parts, sep=sep, end=end)
            result_file.write(text)

        total = len(examples)
        passed = 0

        for i, ex in enumerate(examples, start=1):
            prompt = ex["prompt"]
            expected = ex["expected"]

            emit("=" * 100)
            emit(f"[{i}/{total}] PROMPT")
            emit(prompt)

            try:
                actual, raw_text = run_one(prompt)
                diffs = diff_expected_vs_actual(expected, actual)
                ok = not diffs

                if ok:
                    passed += 1
                    emit("\nRESULT: PASS")
                else:
                    emit("\nRESULT: FAIL")

                emit("\nEXPECTED:")
                emit(json.dumps(expected, indent=2, ensure_ascii=False))

                emit("\nACTUAL:")
                emit(json.dumps(actual, indent=2, ensure_ascii=False))

                if diffs:
                    emit("\nDIFFS:")
                    emit(json.dumps(diffs, indent=2, ensure_ascii=False))

            except Exception as e:
                emit("\nRESULT: ERROR")
                emit(repr(e))

        emit("=" * 100)
        emit(f"SUMMARY: {passed}/{total} exact matches")

    print(f"\nResults written to {RESULT_PATH}")


if __name__ == "__main__":
    main()
