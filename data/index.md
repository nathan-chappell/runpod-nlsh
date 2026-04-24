# Dataset Index

This is the browseable map for agents adding examples to the NLSH planner dataset.

The canonical dataset lives under `data/samples/`, organized by planner behavior:

- `data/samples/`: 128 canonical examples.
- `data/splits/v1/`: materialized `train/eval/test` splits for fairer fine-tuning and held-out pod eval.

All migrated rows include `focus` as the first JSON key.

## Canonical Layout

Plans:

- `data/samples/plans/find_files.jsonl`: 10 examples.
- `data/samples/plans/pdf_merge.jsonl`: 10 examples.
- `data/samples/plans/pdf_extract_pages.jsonl`: 10 examples.
- `data/samples/plans/pdf_search_text.jsonl`: 10 examples.
- `data/samples/plans/csv_to_json.jsonl`: 10 examples.
- `data/samples/plans/json_filter.jsonl`: 10 examples.
- `data/samples/plans/json_select_fields.jsonl`: 10 examples.
- `data/samples/plans/json_sort.jsonl`: 10 examples.
- `data/samples/plans/json_group_count.jsonl`: 10 examples.
- `data/samples/plans/pipelines.jsonl`: 18 multi-step examples.

Clarifications:

- `data/samples/clarifications/pdf_extract_pages.jsonl`: 10 examples.
- `data/samples/clarifications/pdf_merge.jsonl`: 10 examples.

## Materialized Splits

The Runpod workflow now defaults to the committed split tree under `data/splits/v1/`:

- `data/splits/v1/train/`: 78 examples
- `data/splits/v1/eval/`: 25 examples
- `data/splits/v1/test/`: 25 examples

These splits are generated deterministically per source file so every canonical JSONL file contributes held-out coverage. Regenerate them after changing `data/samples/`:

```bash
./.venv/bin/python scripts/materialize_dataset_splits.py
```

## Step Coverage

Current plan-step coverage across all canonical plan samples:

- `find_files`: 21
- `pdf_merge`: 13
- `pdf_extract_pages`: 10
- `pdf_search_text`: 13
- `csv_to_json`: 17
- `json_filter`: 14
- `json_select_fields`: 13
- `json_sort`: 12
- `json_group_count`: 13

## Row Convention

Use `focus` to state the specific behavior being tested. Keep it short, lowercase, and singular: 2-6 words is usually enough.

```json
{"focus":"find pdf max depth","prompt":"find every PDF under ./invoices no deeper than two levels","tags":["find","pdf"],"messages":[...],"plan":{...}}
```

Use `tags` for broad grouping and `focus` for the narrow reason the sample exists.

When adding a row, keep `messages[-1].content` exactly aligned with `plan`. The assistant message should be the compact JSON string form of the same expected output.

## Add-Sample Checklist

1. Choose one specific `focus` before writing the row.
2. Pick the canonical file under `data/samples/` by output type:
   - one-step plans go in `data/samples/plans/<step-kind>.jsonl`
   - multi-step plans go in `data/samples/plans/pipelines.jsonl`
   - clarifications go in `data/samples/clarifications/<task>.jsonl`
3. Inspect nearby examples by tags and focus:

   ```bash
   jq -r '[.focus, .prompt] | @tsv' data/samples/**/*.jsonl
   ```

4. Search for duplicate prompts and near-duplicate expected plans:

   ```bash
   jq -r '.prompt' data/samples/**/*.jsonl | sort | uniq -cd
   ```

5. Add the new row to the canonical file only; then regenerate `data/splits/v1/`.
6. Validate JSONL and run focused tests before finishing.

## Useful Commands

Show current canonical coverage:

```bash
for file in data/samples/**/*.jsonl; do printf '%s ' "$file"; wc -l < "$file"; done
```

Show current tag coverage:

```bash
jq -r '[.tags[]] | join("/")' data/samples/**/*.jsonl | sort | uniq -c | sort -nr
```

Validate all dataset rows parse:

```bash
find data -name '*.jsonl' -print0 | xargs -0 jq -c . >/dev/null
```
