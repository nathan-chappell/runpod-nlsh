# nlsh

`nlsh` is a local CLI that turns natural-language requests into a typed execution plan for a small, whitelisted set of file, PDF, CSV, and JSON workflows. The model plans; the compiler and runner stay deterministic.

## What Is In Here

- A typed schema for either a short execution plan or one clarifying question
- A compiler that turns plans into safe bash scripts
- A runner with clarification and confirmation gates
- A shell runtime built on `find`, `qpdf`, and `jq`
- Python helpers for CSV-to-JSON conversion and simple PDF text search
- Coverage-organized message dataset for fine-tuning and eval
- A starter Axolotl config for `microsoft/Phi-4-mini-instruct` LoRA
- A pod eval image and manifest for sequential vLLM model comparisons plus a direct Phi-4 LoRA run

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
./.venv/bin/nlsh eval --planner gold
```

## Pod Eval And Fine-Tuning

The pod eval and fine-tuning flow builds on Runpod's PyTorch base image, adds the project, configs, datasets, system tools, and a compiler toolchain, then installs vLLM and model weights at pod startup into Runpod persistent storage. This keeps the pushed image from baking model weights while matching Runpod's GPU/CUDA environment.

```bash
docker build -f Dockerfile.pod-eval -t YOUR_DOCKERHUB_USER/nlsh-pod-eval:latest .
docker push YOUR_DOCKERHUB_USER/nlsh-pod-eval:latest
```

The default base is `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`. To try another Runpod base, pass `--build-arg RUNPOD_BASE_IMAGE=...`.

Create a Runpod pod with:

- GPU: RTX 4090 or another 32 GB+ GPU
- Image: `YOUR_DOCKERHUB_USER/nlsh-pod-eval:latest`
- Persistent or network volume mounted at `/workspace`
- `HF_TOKEN` set from a Runpod secret
- Recommended volume size: at least 100 GB
- Container disk: use 20 GB for the Runpod PyTorch base image. The model, vLLM, tmp, and cache data still live on `/workspace`, but the base image itself is large enough that 10 GB is likely too tight if Runpod counts image/root filesystem unpacking there.

The container `CMD` is `["python", "scripts/runpod_bootstrap.py"]`. The bootstrap stays stdlib-only so it can prepare the persistent Python environment before project dependencies are available. It keeps the runtime venv at `/workspace/nlsh-venv` so it survives pod restarts, attempts the required pip installs on every startup, and records a simple bootstrap state version in `/workspace/.nlsh-bootstrap-version`. If that version changes in a future image, the bootstrap removes the old venv once and recreates it before continuing. After dependency install it hands off to the Typer workflow at `python -m nlsh.pod_workflow run`.

The Typer workflow logs the resolved configuration and execution plan, then:

1. validates `HF_TOKEN` and enters `/opt/nlsh`
2. starts all manifest model downloads in parallel
3. evaluates models in priority order as each preferred model is ready: Phi-4, SmolLM3, then Qwen3
4. fine-tunes Phi-4-mini with `scripts/phi_4_training.py`
5. serves the trained adapter with vLLM LoRA support and re-evaluates the fine-tuned Phi-4 adapter

Baseline evals and training now keep incremental state on disk so a failure does not erase the run history. The workflow writes `/workspace/nlsh-artifacts/workflow_state.json`; each model eval writes `run_state.json`, `report.json`, and `eval.log`; OOM retries preserve prior attempt artifacts as `*.attempt-N.*`; and training writes `training_state.json` plus regular checkpoints under the output directory. Reports go to `/workspace/nlsh-artifacts`, the adapter goes to `/workspace/nlsh-finetune/phi-4-mini-instruct-lora`, and exit codes are recorded in `/workspace/nlsh-artifacts/last_eval_exit_code`, `/workspace/nlsh-artifacts/last_training_exit_code`, `/workspace/nlsh-artifacts/last_post_training_eval_exit_code`, and `/workspace/nlsh-artifacts/last_exit_code`. Set `POD_EVAL_EXIT_AFTER=1` to exit after the batch instead of keeping the container alive for inspection.

The Runpod PyTorch base image may include `/start.sh` for base-image services such as SSH or notebook processes. Runpod's custom-template docs describe this as the base-service startup path, so the bootstrap runs it by default when it exists. Set `RUNPOD_START_BASE_SERVICES=0` only if you want application-only startup.

Optional runtime knobs:

```bash
POD_EVAL_LIMIT=2
POD_EVAL_TIMEOUT=90
POD_EVAL_STARTUP_TIMEOUT=900
POD_EVAL_DOWNLOAD_WORKERS=3
POD_EVAL_EVAL_ARGS="--oom-retries 3"
RUNPOD_START_BASE_SERVICES=1
POD_EVAL_RUN_BASELINE_EVAL=1
POD_EVAL_RUN_TRAINING=1
POD_EVAL_TRAIN_MODEL_ID=microsoft/Phi-4-mini-instruct
POD_EVAL_TRAIN_OUTPUT_DIR=/workspace/nlsh-finetune/phi-4-mini-instruct-lora
POD_EVAL_TRAIN_ARGS="--max-steps 200 --oom-retries 3"
```

You can also run commands manually in the pod:

```bash
export HF_HOME=/workspace/hf-cache
/workspace/nlsh-venv/bin/python scripts/pod_eval.py download-models
/workspace/nlsh-venv/bin/python scripts/pod_eval.py run-suite --dataset data/samples
/workspace/nlsh-venv/bin/python -m nlsh.pod_workflow run --dry-run
```

The manifest is `configs/pod_eval_models.json`. It currently evaluates `microsoft/Phi-4-mini-instruct`, `HuggingFaceTB/SmolLM3-3B`, and `Qwen/Qwen3-8B` through vLLM with conservative single-GPU settings. Downloads run in parallel; eval stays sequential in the preferred order so one GPU is used predictably. If a vLLM startup or serve attempt fails with an OOM, `scripts/pod_eval.py` retries with smaller `max_num_seqs`, then shorter `max_model_len`, then lower `gpu_memory_utilization` until it hits the configured floor. If training hits an OOM, `scripts/phi_4_training.py` retries with smaller batch sizes, then shorter sequence length, resuming from the latest checkpoint when one exists.

For local non-GPU checks, validate the manifest and exercise the eval path with the gold planner:

```bash
./.venv/bin/python scripts/pod_eval.py download-models --dry-run
./.venv/bin/python scripts/pod_eval.py run-model --planner gold --model microsoft/Phi-4-mini-instruct --limit 2
```

## Dataset And Training

The canonical, coverage-organized examples live in `data/samples/`; see `data/index.md` for the current map. Eval, pod eval, gold planning, and fine-tuning all default to this directory.

Each row includes:

- `focus`
- `prompt`
- `tags`
- `messages`
- `plan`

For a direct single-GPU Phi-4-mini LoRA run on a 32 GB+ Runpod GPU, install the training extras in the pod and run the training script directly:

```bash
pip install -e ".[dev,train]"
python scripts/phi_4_training.py \
  --output-dir /workspace/nlsh-finetune/phi-4-mini-instruct-lora
```

By default the training script deterministically partitions `data/samples/` into train/eval records. Use `--no-eval` to train on all canonical examples.

The script uses regular bf16 LoRA by default, maps dataset `developer` messages to chat `system` messages, trains on prompt/completion examples, and starts with `target_modules=["qkv_proj"]`, `r=8`, `lora_alpha=16`, and `lora_dropout=0.05`. It tries FlashAttention 2 when installed and otherwise falls back to SDPA. Checkpoints go under `--output-dir/checkpoint-*`; the final PEFT adapter is saved directly in `--output-dir`.

The next pod startup uses this script directly, not Axolotl. Set `POD_EVAL_RUN_TRAINING=0` if you only want baseline evals, or set `POD_EVAL_TRAIN_ARGS` to pass extra arguments such as `--max-steps 100` for a short smoke run.

If the adapter underfits, widen LoRA gradually:

```bash
python scripts/phi_4_training.py --target-modules qkv_proj,o_proj
python scripts/phi_4_training.py --target-modules qkv_proj,o_proj,gate_up_proj,down_proj
```

The Axolotl starter config is at `configs/axolotl/phi-4-mini-instruct-lora.yaml`. It mirrors the conservative qkv-only LoRA defaults and is meant for a demo LoRA run after the prompting and eval examples feel stable.

## Notes

- The compiler refuses to overwrite outputs unless `NLSH_ALLOW_OVERWRITE=1`.
- `pdf_merge` and `pdf_extract_pages` compile to `qpdf`.
- `pdf_search_text` uses `pypdf` and writes JSON matches with `file`, `page`, `query`, and `text_excerpt`.
- JSON transforms compile to `jq`.
- CSV transforms first convert to JSON with `python -m nlsh.csv_to_json`.
- `nlsh run` checks for missing system tools before asking for execution confirmation.
