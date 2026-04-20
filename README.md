# nlsh

`nlsh` is a local CLI that turns natural-language requests into a typed execution plan for a small, whitelisted subset of shell workflows. The model plans; the compiler and runner stay deterministic.

## What is in here

- A typed `PlanV1` schema for short execution plans
- A compiler that turns plans into safe bash scripts
- A runner with clarification and confirmation gates
- A shell-first runtime built on `find`, `qpdf`, `ghostscript`, `ffmpeg`, and `mlr`
- Seed `train/dev/test` message datasets for fine-tuning and eval
- A starter Axolotl config for `openai/gpt-oss-20b` LoRA on Runpod

## Supported actions

- `find_files`
- `pdf_combine`
- `pdf_compress`
- `pdf_extract_pages`
- `pdf_rotate`
- `media_transcode_for_tv`
- `media_extract_audio_mp3`
- `media_clip`
- `csv_join`
- `csv_filter_rows`
- `csv_select_columns`
- `csv_sort_rows`
- `csv_group_count`

V1 allows either one step or two steps where the first step is `find_files`.

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
export NLSH_BASE_URL=https://router.huggingface.co/v1
export NLSH_MODEL=openai/gpt-oss-20b:together
```

For the old Hugging Face router experiment, `hf_test.py` now reads `HF_TOKEN` from the environment.

## CLI

Plan a request:

```bash
./.venv/bin/nlsh plan "combine all the PDFs under contracts into one file"
```

Run a request with confirmation:

```bash
./.venv/bin/nlsh run "extract the audio from lecture.mp4"
```

Compile a saved plan:

```bash
./.venv/bin/nlsh compile plan.json
```

Evaluate against the seed dataset:

```bash
./.venv/bin/nlsh eval --planner gold --split test
```

## Dataset and training

The canonical examples live in:

- `data/train.messages.jsonl`
- `data/dev.messages.jsonl`
- `data/test.messages.jsonl`

Each row includes:

- `prompt`
- `tags`
- `messages`
- `plan`

The Axolotl starter config is at `configs/axolotl/gpt-oss-20b-lora.yaml`. It assumes a single-node LoRA run on `openai/gpt-oss-20b`, which matches the documented `gpt-oss-20b` single-node fine-tuning path and fits the Runpod recommendation that LLM fine-tuning use high-end 40-80 GB GPUs.

## Notes

- The compiler refuses to overwrite outputs unless `NLSH_ALLOW_OVERWRITE=1`.
- `pdf_combine`, `pdf_extract_pages`, and `pdf_rotate` compile to `qpdf`.
- `pdf_compress` compiles to `gs`, and media steps compile to `ffmpeg`.
- CSV transforms compile to `mlr`.
- `nlsh run` checks for missing system tools before asking for execution confirmation.
