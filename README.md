# nlsh

`nlsh` is an experiment in teaching a small language model to turn natural-language file tasks into a tiny workflow language and then into deterministic shell execution. Instead of letting the model emit arbitrary bash, the planner is constrained to a small action set and is expected to ask for clarification when the prompt is underspecified.

## Command Model

The supported action vocabulary is intentionally narrow:

- `find_files` for file discovery by `root`, `glob`, and optional `max_depth`
- `pdf_merge`, `pdf_extract_pages`, and `pdf_search_text` for PDF workflows
- `csv_to_json` for CSV ingestion
- `json_filter`, `json_select_fields`, `json_sort`, and `json_group_count` for JSON transforms

The planner never returns shell directly. Its final output is always one of two validated JSON shapes:

```json
{"kind":"clarification","question":"Which PDF files should I merge?"}
```

```json
{"kind":"plan","steps":[{"kind":"find_files","root":"./contracts","glob":"*.pdf"}]}
```

`PlanV1` is also deliberately small:

- a single executable step
- `find_files` followed by exactly one consuming step
- `csv_to_json` followed by exactly one JSON terminal step
- omitted handoff fields in two-step plans instead of `null`

So this is closer to semantic parsing than free-form shell generation: no arbitrary command chains, no three-step plans, and no partial executable plans. Valid plans compile into deterministic `find`, `qpdf`, `jq`, and Python helper invocations, and the runner previews and confirms them before execution.

## End-To-End Flow

The repo is organized around a full dataset-to-runtime loop:

1. `data/samples/` stores canonical JSONL examples where each prompt maps to either a `PlanV1` response or a `Clarification`.
2. `scripts/materialize_dataset_splits.py` builds the committed fair `train/eval/test` split tree under `data/splits/v1/`.
3. `scripts/phi_4_training.py` fine-tunes `microsoft/Phi-4-mini-instruct` with LoRA on those structured examples.
4. The Runpod image can either run the eval/fine-tune workflow or serve the bundled adapter behind an OpenAI-compatible SGLang endpoint.
5. At runtime, `nlsh plan` or `nlsh run` sends a natural-language prompt to that endpoint, validates the returned JSON, compiles it into a shell script, previews it, and executes it locally after confirmation.

## System Idea

- `planner`: turns natural language into either `{"kind":"clarification"}` or `{"kind":"plan"}`
- `schema`: defines the only legal actions and plan shapes
- `compiler`: lowers valid plans into deterministic shell scripts
- `runner`: previews, confirms, and executes compiled plans
- `dataset`: teaches the planner when to execute and when to ask for clarification
- `training`: fine-tunes Phi-4 with LoRA on those prompt/plan examples
- `serving`: packages the fine-tuned adapter behind an OpenAI-compatible SGLang endpoint
- `evaluation`: measures exact match, compile-valid, and slot accuracy on held-out prompts

## Architecture

At a high level, the stack looks like this:

```text
natural language
  -> planner prompt
  -> PlanV1 / Clarification JSON
  -> schema validation
  -> deterministic compiler
  -> confirmation / tool checks
  -> bash + helper scripts
```

## Latest Results

The current headline result is that the fine-tuned Phi-4 model is substantially better on the fair held-out test split, but the gain is in exact task execution and slot filling, not compile-valid.

Held-out test split size:

- `25` examples

Recent Runpod reruns on the same split:

| Run | Baseline Exact | Fine-Tuned Exact | Baseline Compile | Fine-Tuned Compile | Baseline Slot | Fine-Tuned Slot |
| --- | --- | --- | --- | --- | --- | --- |
| `dbdyfefx5gmdgn-64410d30` | `0.48` | `0.92` | `1.0` | `1.0` | `0.396` | `0.905` |
| `68oi72inl7zsj6-64411b67` | `0.48` | `0.88` | `1.0` | `1.0` | `0.396` | `0.898` |

So the result is stable enough to summarize simply:

- compile-valid is already saturated on the current test split
- fine-tuning improves exact match by about `+0.40` to `+0.44`
- fine-tuning improves slot accuracy by about `+0.50`

That means the main problem is no longer “can the model emit legal plans?” It is “does it choose the correct action and fill the slots precisely?”

Representative improvements:

- `find every PDF under ./invoices no deeper than two levels`
  baseline asked a clarification, fine-tuned emits the exact `find_files` plan
- `from payments.json keep rows where amount is greater than or equal to 500 into large_payments.json`
  baseline over-clarified, fine-tuned emits the exact `json_filter` plan
- `extract page 1 from cover_letter.pdf into cover_letter_page1.pdf`
  baseline still asked for confirmation of the page range, fine-tuned executes directly
- `search handbook.pdf for warranty and write warranty_matches.json`
  baseline over-clarified, fine-tuned emits the exact `pdf_search_text` plan

The main regression to watch is still:

- `merge the PDFs into packet.pdf`
  baseline correctly asks which PDFs to merge, while the fine-tuned model can hallucinate input filenames

So the current dataset work is focused on preserving the new confidence on fully specified prompts without teaching the model to invent missing inputs on underspecified ones.

There is now also a live served check against the Runpod-hosted fine-tuned model from April 25, 2026: the endpoint responded successfully to `/model_info`, `/v1/models`, and `/v1/chat/completions`; a 10-sample live probe reached `8/10` exact match with `10/10` compile-valid outputs; and an end-to-end playground run executed `8/8` real prompts successfully across CSV, JSON, PDF, and file-discovery tasks. The brief write-up and saved artifacts are in [docs/live-served-demo-2026-04-25.md](docs/live-served-demo-2026-04-25.md).

## Setup

Create or reuse a virtualenv, then install the package in editable mode:

```bash
./.venv/bin/pip install -e ".[dev]"
```

Install required system tools:

```bash
./scripts/bootstrap_system_deps.sh
```

You can inspect the current machine state without installing anything:

```bash
./scripts/bootstrap_system_deps.sh --check
```

Set credentials in `.env` or your shell when using an OpenAI-compatible planner endpoint:

```bash
export NLSH_API_KEY=...
export NLSH_BASE_URL=https://api.example.com/v1
export NLSH_MODEL=microsoft/Phi-4-mini-instruct
```

## CLI

The main commands are:

```bash
./.venv/bin/nlsh plan "merge every PDF under ./contracts into contracts.pdf"
./.venv/bin/nlsh run "convert orders.csv to JSON in orders.json"
./.venv/bin/nlsh compile plan.json
./.venv/bin/nlsh eval --planner gold
```

For live model checks:

```bash
export NLSH_BASE_URL=https://your-endpoint.example/v1
export NLSH_MODEL=nlsh-phi4-ft
export NLSH_API_KEY=...
./.venv/bin/nlsh probe-live --count 5 --seed 20260424
```

`probe-live` uses the same runtime planner prompt shape as `nlsh plan/run` by default. The repo-local `runpod-live-test` skill adds two higher-level checks:

- `probe-dataset`: a saved 10-sample stratified probe
- `interactive-demo`: a disposable sandbox that asks the live model for a plan, confirms it, and executes it against real temp files

## Pod Eval And Fine-Tuning

There are two Runpod modes:

1. `workflow`: baseline eval -> fine-tune -> post-train eval
2. `serve`: expose the bundled fine-tuned Phi-4 adapter over an OpenAI-compatible endpoint

Build and publish the image with:

```bash
docker build -f Dockerfile.pod-eval \
  -t YOUR_DOCKERHUB_USER/runpod-nlsh:0.4.0 \
  -t YOUR_DOCKERHUB_USER/runpod-nlsh:latest .
docker push YOUR_DOCKERHUB_USER/runpod-nlsh:0.4.0
docker push YOUR_DOCKERHUB_USER/runpod-nlsh:latest
```

Practical pod setup:

```bash
RUNPOD_BOOT_MODE=serve
RUNPOD_SERVE_HOST=0.0.0.0
RUNPOD_SERVE_PORT=8000
RUNPOD_SERVE_API_KEY=replace-with-a-random-64-char-hex-token
```

If you want `workflow` mode instead of serving, also set `HF_TOKEN` and switch `RUNPOD_BOOT_MODE=workflow`.

What the image does:

- bootstraps from `scripts/runpod_bootstrap.py`
- prepares `/workspace` caches and artifact dirs
- defaults to `serve` mode
- serves `microsoft/Phi-4-mini-instruct:nlsh-phi4-ft` on port `8000`

Public access:

- expose HTTP port `8000`
- set `RUNPOD_SERVE_API_KEY` if you want bearer-token protection
- without that token, the endpoint is public once the port is exposed

Bundling the latest adapter into the next image:

```bash
python scripts/stage_serving_adapter.py \
  --bundle-root tmp/runpod-downloads/2026-04-24/68oi72inl7zsj6-64411b67
```

Known hardware/runtime caveat:

- the image is currently based on `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- that setup worked on RTX `5090` pods
- we hit startup failures on at least one RTX `4090` pod before the app even booted, with `nvidia-container-cli` reporting `cuda>=12.8`
- that points to a host-driver / Runpod-base-image compatibility issue rather than an app bug
- so for now, `5090` is the known-good path with the current image; if `4090` support matters, the likely fix is choosing a different Runpod base image / CUDA line

A quick external smoke test:

```bash
curl -H "Authorization: Bearer $RUNPOD_SERVE_API_KEY" \
  https://<pod-id>-8000.proxy.runpod.net/v1/models
```

## Dataset And Training

The canonical, coverage-organized examples live in `data/samples/`; see `data/index.md` for the current map. Materialized fair splits live under `data/splits/v1/`.

The April 24, 2026 Runpod fine-tune findings and the follow-up dataset plan are captured in `docs/runpod-results-2026-04-24.md`.

The dataset is trying to teach one specific boundary over and over:

- execute when the prompt already contains enough information
- ask for clarification when required inputs are actually missing

That is why the examples are organized by action family and why many rows come in contrastive pairs such as:

- underspecified merge -> clarification
- fully specified merge -> plan
- underspecified page extraction -> clarification
- explicit page range -> plan
- vague search request -> clarification
- explicit query + file(s) -> plan

The current fair split is:

- `78` train
- `25` eval
- `25` test

Training in one command:

```bash
pip install -e ".[dev,train]"
python scripts/phi_4_training.py \
  --output-dir /workspace/nlsh-finetune/phi-4-mini-instruct-lora
```

Current default training shape:

- Phi-4 mini LoRA
- native `transformers` classes, no remote-code dependency
- `10` epochs
- batch size `4`
- grad accumulation `4`
- learning rate `5e-4`
- OOM retries that back off batch size and sequence length

Regenerate the committed fair splits after editing `data/samples`:

```bash
./.venv/bin/python scripts/materialize_dataset_splits.py
```

Run artifacts now include enough to compare experiments later:

- baseline vs fine-tuned eval summaries
- checkpoints and adapter files
- `metrics_history.json`
- `metrics_history.csv`
- `training-metrics.svg`
- exfil reports with representative improvements and regressions

## Notes

- The compiler refuses to overwrite outputs unless `NLSH_ALLOW_OVERWRITE=1`.
- `pdf_merge` and `pdf_extract_pages` compile to `qpdf`.
- JSON transforms compile to `jq`.
- CSV transforms use `python -m nlsh.csv_to_json`.
- `pdf_search_text` uses `pypdf`.
- `nlsh run` checks for missing system tools before asking for execution confirmation.
