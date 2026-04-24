# Runpod Phi-4 Results - 2026-04-24

This note captures the key findings from the April 24, 2026 Runpod Phi-4 fine-tune runs so they survive beyond the local `tmp/` artifact bundles.

## Runs

Artifacts were exfiltrated from these pods:

- `dbdyfefx5gmdgn-64410d30`
- `68oi72inl7zsj6-64411b67`

Local artifact bundles live under:

- `tmp/runpod-downloads/2026-04-24/dbdyfefx5gmdgn-64410d30/`
- `tmp/runpod-downloads/2026-04-24/68oi72inl7zsj6-64411b67/`

## Headline Result

The important change is not `compile_valid`. That metric is already saturated at `1.0` on the fair `25`-example held-out test split for both baseline and fine-tuned Phi-4.

The real gain is exactness and slot filling:

| Run | Baseline Exact | Fine-Tuned Exact | Baseline Compile | Fine-Tuned Compile | Baseline Slot | Fine-Tuned Slot |
| --- | --- | --- | --- | --- | --- | --- |
| `dbdyfefx5gmdgn-64410d30` | `0.48` | `0.92` | `1.0` | `1.0` | `0.396` | `0.905` |
| `68oi72inl7zsj6-64411b67` | `0.48` | `0.88` | `1.0` | `1.0` | `0.396` | `0.898` |

So the result looks stable across reruns: fine-tuning is worth about `+0.40` to `+0.44` exact match and about `+0.50` slot accuracy on the current fair split.

## What Improved

On the `68oi72inl7zsj6-64411b67` run:

- `11/25` prompts improved from non-exact to exact
- `1/25` prompt regressed from exact to non-exact
- `12/25` prompts improved slot accuracy
- `2/25` prompts regressed slot accuracy

The dominant improvement pattern is simple: the baseline model over-clarified on prompts that were already fully specified, while the fine-tuned model executed the intended plan directly.

Improvement categories:

- `find_files`: `2`
- `json_filter`: `2`
- `json_group_count`: `1`
- `json_sort`: `2`
- `pdf_extract_pages`: `2`
- `pdf_search_text`: `2`

## Representative Improvements

### 1. Stop Clarifying Fully Specified `find_files`

Prompt:

```text
find every PDF under ./invoices no deeper than two levels
```

Gold:

```json
{"kind":"plan","steps":[{"kind":"find_files","root":"./invoices","glob":"*.pdf","max_depth":2}]}
```

Baseline output:

```json
{"kind":"clarification","question":"Do you want to find all PDF files under the './invoices' directory without going deeper than two levels?"}
```

Fine-tuned output:

```json
{"kind":"plan","steps":[{"kind":"find_files","root":"./invoices","glob":"*.pdf","max_depth":2}]}
```

### 2. Stop Clarifying Obvious Filter Fields

Prompt:

```text
from payments.json keep rows where amount is greater than or equal to 500 into large_payments.json
```

Gold:

```json
{"kind":"plan","steps":[{"kind":"json_filter","input_file":"payments.json","field":"amount","operator":"gte","value":500,"output_file":"large_payments.json"}]}
```

Baseline output:

```json
{"kind":"clarification","question":"What is the field name for the amount in the payments.json file?"}
```

Fine-tuned output:

```json
{"kind":"plan","steps":[{"kind":"json_filter","input_file":"payments.json","field":"amount","operator":"gte","value":500,"output_file":"large_payments.json"}]}
```

### 3. Handle Explicit Single-Page Extraction Directly

Prompt:

```text
extract page 1 from cover_letter.pdf into cover_letter_page1.pdf
```

Gold:

```json
{"kind":"plan","steps":[{"kind":"pdf_extract_pages","input_file":"cover_letter.pdf","page_start":1,"page_end":1,"output_file":"cover_letter_page1.pdf"}]}
```

Baseline output:

```json
{"kind":"clarification","question":"Do you want to extract only page 1 from cover_letter.pdf, or do you need a specific range of pages?"}
```

Fine-tuned output:

```json
{"kind":"plan","steps":[{"kind":"pdf_extract_pages","input_file":"cover_letter.pdf","page_start":1,"page_end":1,"output_file":"cover_letter_page1.pdf"}]}
```

### 4. Stop Asking for Confirmation on Straightforward Search

Prompt:

```text
search handbook.pdf for warranty and write warranty_matches.json
```

Gold:

```json
{"kind":"plan","steps":[{"kind":"pdf_search_text","input_files":["handbook.pdf"],"query":"warranty","output_file":"warranty_matches.json","context_chars":160}]}
```

Baseline output:

```json
{"kind":"clarification","question":"Do you want to search for the exact phrase 'warranty' in handbook.pdf and output the results to warranty_matches.json?"}
```

Fine-tuned output:

```json
{"kind":"plan","steps":[{"kind":"pdf_search_text","input_files":["handbook.pdf"],"query":"warranty","output_file":"warranty_matches.json","context_chars":160}]}
```

## Regression To Fix

The main regression is a harmful one:

Prompt:

```text
merge the PDFs into packet.pdf
```

Gold:

```json
{"kind":"clarification","question":"Which PDF files should I merge into packet.pdf?"}
```

Baseline output:

```json
{"kind":"clarification","question":"Which PDF files should I merge into packet.pdf?"}
```

Fine-tuned output:

```json
{"kind":"plan","steps":[{"kind":"pdf_merge","input_files":["pdf1.pdf","pdf2.pdf","pdf3.pdf"],"output_file":"packet.pdf"}]}
```

This is the clearest sign that the fine-tune improved the "do the task" instinct but slightly weakened the "do not invent missing inputs" boundary for underspecified merge requests.

## Near-Miss To Fix

Prompt:

```text
find the JSON file in ./exports with june in the path and keep only rows where region is eu in june_eu.json
```

Baseline output:

- over-clarified instead of executing

Fine-tuned output:

- got the two-step shape right
- got the downstream `json_filter` right
- missed exact match because it used `june*.json` instead of `*june*.json`

This is not a planning-shape problem anymore; it is a slot-value precision problem around glob semantics.

## Interpretation

The current dataset and prompt changes appear to have solved the compile-valid problem already. The next bottleneck is narrower:

1. preserve the new confidence on fully specified prompts
2. restore correct clarification behavior on underspecified prompts
3. tighten slot-value precision on globs and similar small string fields

## Dataset Enhancement Plan

### 1. Add Paired "Clarify vs Execute" Contrasts

For each task family, add explicit pairs:

- underspecified merge -> clarification
- fully specified merge -> plan
- underspecified extract -> clarification
- explicit page/range extract -> plan
- underspecified search -> clarification
- explicit query + inputs -> plan

The goal is to teach the boundary, not just add more positive examples.

### 2. Add Anti-Hallucination Merge Cases

The regression on `merge the PDFs into packet.pdf` suggests the model now wants to be helpful too aggressively. Add more clarification examples where the output filename is given but the input PDF set is not.

Good candidates:

- `merge the PDFs into packet.pdf`
- `merge the quarterly PDFs into q1_packet.pdf`
- `combine the board packets into board_packet.pdf`

These should all stay clarifications unless the prompt provides either explicit files or a concrete `find_files` cue.

### 3. Add More Explicit "Enough Information" Positives

The wins came from tasks that were already specific enough but baseline still questioned:

- direct `find_files`
- direct `json_filter`
- direct `json_sort`
- direct `json_group_count`
- direct `pdf_extract_pages`
- direct `pdf_search_text`

Add more one-step positives in exactly those families so the model keeps learning "execute when the slots are already present."

### 4. Add Glob-Semantics Precision Cases

The `june*.json` vs `*june*.json` miss shows we need denser coverage for path-cue phrasing:

- `with june in the path` -> `*june*`
- `starting with june` -> `june*`
- `ending with june` -> `*june`
- `named exactly june.json` -> `june.json`

This should include both standalone `find_files` and `find_files -> terminal` pipeline examples.

### 5. Add Regression-Focused Eval Coverage

Keep a small regression slice that always includes:

- underspecified merge clarification
- underspecified quarterly merge clarification
- explicit single-page extraction
- explicit multi-page extraction
- explicit PDF search
- path-cue wildcard precision

These are the behaviors that moved the most in this experiment.

### 6. Prefer Dataset Work Over More Schema Work Right Now

Given the current results:

- `compile_valid` is already `1.0`
- the remaining mistakes are boundary and slot-value mistakes

So the next effort should mostly be dataset curation, not another schema rewrite.

## Suggested Next Experiment

1. Add the clarification-vs-execution contrast rows above.
2. Regenerate `data/splits/v1`.
3. Re-run the same Phi-4-only fair split workflow.
4. Require:
   - `compile_valid` stays at `1.0`
   - exact match does not regress from the current `0.88` to `0.92` range
   - underspecified merge prompts stop hallucinating input files
