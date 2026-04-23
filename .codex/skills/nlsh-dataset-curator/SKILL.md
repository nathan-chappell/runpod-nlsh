---
name: nlsh-dataset-curator
description: Add, inspect, and curate examples for the NLSH planner dataset in oss-ft-bash-testing/data. Use when asked to add dataset samples, improve coverage, avoid duplicate examples, or organize planner examples for file, PDF, CSV, JSON, and clarification tasks.
metadata:
  short-description: Curate NLSH planner dataset examples
---

# NLSH Dataset Curator

Use this skill when adding or reviewing dataset examples for the `oss-ft-bash-testing` NLSH planner. The goal is useful coverage, not just more rows.

## Start Here

1. Open `data/index.md` in the repo to identify the relevant canonical file under `data/samples/`.
2. Inspect nearby rows in `data/samples/**/*.jsonl` before adding anything.
3. Add new examples to the canonical `data/samples/` hierarchy first.
4. Do not add split files; eval and training load `data/samples/` directly.
5. Do not repartition data unless the user explicitly asks.

## Row Shape

New rows should keep `focus` as the first JSON key:

```json
{"focus":"find pdf max depth","prompt":"find every PDF under ./invoices no deeper than two levels","tags":["find","pdf"],"messages":[...],"plan":{...}}
```

Rules:

- `focus`: one short lowercase phrase, usually 2-6 words.
- `prompt`: the natural-language request.
- `tags`: broad grouping labels such as `find`, `pdf`, `csv`, `json`, `filter`, `clarification`.
- `messages`: developer/user/assistant training messages.
- `plan`: the expected `PlanV1` or clarification JSON object.

Keep `messages[-1].content` exactly aligned with `plan`. The assistant message should be the same expected output serialized as compact JSON.

## Choosing A Useful Sample

Prefer samples that add one clear behavior:

- A new slot value pattern, such as `max_depth`, output file wording, or numeric filter value.
- A new pipeline handoff, especially where an input field should be `null`.
- A clarification case where required information is missing.
- A wording variation that is meaningfully different from existing prompts.

Avoid samples that only rename files or repeat the same plan shape with no new behavior.

## Duplicate Checks

Before editing, run targeted searches:

```bash
jq -r '.prompt' data/samples/**/*.jsonl | sort | uniq -cd
jq -r 'select((.tags // []) | index("pdf")) | [.focus, .prompt] | @tsv' data/samples/**/*.jsonl
jq -r '[.tags[]] | join("/")' data/samples/**/*.jsonl | sort | uniq -c | sort -nr
```

Adapt the tag search to the task being added. If a near duplicate exists, either choose a different behavior or make the new `focus` clearly distinct.

## Validation

After changing data, run:

```bash
find data -name '*.jsonl' -print0 | xargs -0 jq -c . >/dev/null
pytest tests/test_eval.py
pytest tests/test_phi4_training.py
```

If tests fail because existing tests assert old row counts, inspect whether the count expectation should be updated for the new sample.
