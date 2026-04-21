# nlsh

`nlsh` is a local CLI that turns natural-language requests into a typed execution plan for a small, whitelisted set of file, PDF, CSV, and JSON workflows. The model plans; the compiler and runner stay deterministic.

## What Is In Here

- A typed schema for either a short execution plan or one clarifying question
- A compiler that turns plans into safe bash scripts
- A runner with clarification and confirmation gates
- A shell runtime built on `find`, `qpdf`, and `jq`
- Python helpers for CSV-to-JSON conversion and simple PDF text search
- Seed `train/dev/test` message datasets for fine-tuning and eval
- A starter Axolotl config for `microsoft/Phi-4-mini-instruct` LoRA

## Supported Actions

- `find_files`
- `pdf_merge`
- `pdf_extract_pages`
- `pdf_search_text`
- `csv_to_json`
- `json_filter`
- `json_select_fields`
- `json_sort`
- `json_group_count`

Planner output is one of two shapes:

```json
{"kind": "clarification", "question": "Which PDF files should I merge?"}
```

```json
{"kind": "plan", "steps": [{"kind": "find_files", "root": "./contracts", "glob": "*.pdf", "max_depth": null}]}
```

Plans allow up to three linear steps. `find_files` is optional, but when used it must be first. CSV analysis should flow through `csv_to_json` and then one JSON transform.

## Setup

Create or reuse a virtualenv, then install the package in editable mode:

```bash
./.venv/bin/pip install -e .[dev]
```

Install required system tools:

```bash
./scripts/bootstrap_system_deps.sh
```

You can inspect the current machine state without installing anything:

```bash
./scripts/bootstrap_system_deps.sh --check
```

Set credentials in `.env` or your shell:

```bash
export NLSH_API_KEY=...
export NLSH_BASE_URL=https://api.runpod.ai/v2/ENDPOINT_ID/openai/v1
export NLSH_MODEL=microsoft/Phi-4-mini-instruct
```

Run the configured Runpod endpoint against the dev messages:

```bash
./.venv/bin/python scripts/runpod_dev_eval.py --timeout 30
```

The script writes a readable log to `artifacts/runpod_dev_eval.txt` and a structured report to `artifacts/runpod_dev_eval.json`.

## CLI

Plan a request:

```bash
./.venv/bin/nlsh plan "merge every PDF under ./contracts into contracts.pdf"
```

Run a request with confirmation:

```bash
./.venv/bin/nlsh run "convert orders.csv to JSON and keep rows where status is paid in paid_orders.json"
```

Compile a saved plan:

```bash
./.venv/bin/nlsh compile plan.json
```

Evaluate against the seed dataset:

```bash
./.venv/bin/nlsh eval --planner gold --split test
```

## Dataset And Training

The canonical examples live in:

- `data/train.messages.jsonl`
- `data/dev.messages.jsonl`
- `data/test.messages.jsonl`

Each row includes:

- `prompt`
- `tags`
- `messages`
- `plan`

For now, the active prompt-iteration set is `data/dev.messages.jsonl`, with 20 examples.

The Axolotl starter config is at `configs/axolotl/phi-4-mini-instruct-lora.yaml`. It is intentionally small and meant for a demo LoRA run after the prompting and eval examples feel stable.

## Notes

- The compiler refuses to overwrite outputs unless `NLSH_ALLOW_OVERWRITE=1`.
- `pdf_merge` and `pdf_extract_pages` compile to `qpdf`.
- `pdf_search_text` uses `pypdf` and writes JSON matches with `file`, `page`, `query`, and `text_excerpt`.
- JSON transforms compile to `jq`.
- CSV transforms first convert to JSON with `python -m nlsh.csv_to_json`.
- `nlsh run` checks for missing system tools before asking for execution confirmation.
