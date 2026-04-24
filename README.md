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
- A pod eval image and manifest for sequential SGLang model comparisons plus a direct Phi-4 LoRA run

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

The pod eval and fine-tuning flow builds on Runpod's PyTorch base image and installs the project plus the full Python dependency stack during `docker build`. Pod startup stays lightweight: it prepares `/workspace`, starts Runpod base services through `/start.sh` when available, and then hands off to the workflow. Mutable state such as Hugging Face cache, downloaded models, checkpoints, and eval artifacts lives on `/workspace`; Python packages do not.

```bash
docker build -f Dockerfile.pod-eval \
  -t YOUR_DOCKERHUB_USER/runpod-nlsh:0.2.4 \
  -t YOUR_DOCKERHUB_USER/runpod-nlsh:latest .
docker push YOUR_DOCKERHUB_USER/runpod-nlsh:0.2.4
docker push YOUR_DOCKERHUB_USER/runpod-nlsh:latest
```

The default base is `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`. To try another Runpod base, pass `--build-arg RUNPOD_BASE_IMAGE=...`.
`requirements/pod-sglang.txt` pins `sglang==0.5.10.post1` and `sglang-kernel==0.4.1` so the image stays aligned with the base layer's Torch 2.9.1 and can reuse the PyTorch already bundled in the Runpod image. `requirements/pod-train.txt` matches that runtime by pinning `transformers==5.3.0`, and the training script uses native Transformers classes for Phi-4 by default instead of the Hub-hosted remote model code.

Create a Runpod pod with:

- GPU: RTX 4090 or another 32 GB+ GPU
- Image: `YOUR_DOCKERHUB_USER/runpod-nlsh:0.2.4`
- Persistent or network volume mounted at `/workspace`
- `HF_TOKEN` set from a Runpod secret
- Recommended volume size: at least 100 GB
- Container disk: use 20 GB for the Runpod PyTorch base image. The model, SGLang, tmp, and cache data still live on `/workspace`, but the base image itself is large enough that 10 GB is likely too tight if Runpod counts image/root filesystem unpacking there.

The container `CMD` is `["python", "scripts/runpod_bootstrap.py"]`. Heavy Python dependencies are baked into the image from `requirements/pod-core.txt`, `requirements/pod-train.txt`, and `requirements/pod-sglang.txt`, and `nlsh` itself is installed into the image Python during the build. The bootstrap stays stdlib-only, creates cache and artifact directories under `/workspace`, starts Runpod base services by default, and then execs `python -m nlsh.pod_workflow run`. Normal pod startup should not need any runtime `pip install`, and `/workspace` should only hold models, caches, checkpoints, and artifacts.

The Typer workflow logs the resolved configuration and execution plan, then:

1. validates `HF_TOKEN` and enters `/opt/nlsh`
2. starts all manifest model downloads in parallel
3. evaluates models in priority order as each preferred model is ready: Phi-4, SmolLM3, then Qwen3
4. fine-tunes Phi-4-mini with `scripts/phi_4_training.py`
5. serves the trained adapter with SGLang LoRA support and re-evaluates the fine-tuned Phi-4 adapter

For serious runs, the workflow now defaults to committed `train/eval/test` splits under `data/splits/v1/`: baseline eval and post-training eval use `data/splits/v1/test`, while training uses `data/splits/v1/train` plus `data/splits/v1/eval`. The canonical source dataset remains `data/samples/`.

Baseline evals and training now keep incremental state on disk so a failure does not erase the run history. The workflow writes `/workspace/nlsh-artifacts/workflow_state.json`; each model eval writes `run_state.json`, `report.json`, and `eval.log`; OOM retries preserve prior attempt artifacts as `*.attempt-N.*`; and training writes `training_state.json` plus regular checkpoints under the output directory. Reports go to `/workspace/nlsh-artifacts`, the adapter goes to `/workspace/nlsh-finetune/phi-4-mini-instruct-lora`, and exit codes are recorded in `/workspace/nlsh-artifacts/last_eval_exit_code`, `/workspace/nlsh-artifacts/last_training_exit_code`, `/workspace/nlsh-artifacts/last_post_training_eval_exit_code`, and `/workspace/nlsh-artifacts/last_exit_code`. The pod bootstrap now defaults `POD_EVAL_EXIT_AFTER=1`, so a serious eval/train/eval run finishes once and exits cleanly. Set `POD_EVAL_EXIT_AFTER=0` if you want the container to stay alive for inspection and artifact exfiltration.

The Runpod PyTorch base image may include `/start.sh` for base-image services such as SSH or notebook processes. Runpod's custom-template docs describe this as the base-service startup path, so the bootstrap runs it by default when it exists. Set `RUNPOD_START_BASE_SERVICES=0` only if you want application-only startup.

Optional runtime knobs:

```bash
POD_EVAL_LIMIT=2
POD_EVAL_TIMEOUT=90
POD_EVAL_STARTUP_TIMEOUT=900
POD_EVAL_DOWNLOAD_WORKERS=3
POD_EVAL_SELECTED_MODELS=microsoft/Phi-4-mini-instruct
POD_EVAL_EVAL_ARGS="--oom-retries 3"
RUNPOD_START_BASE_SERVICES=0
POD_EVAL_RUN_BASELINE_EVAL=1
POD_EVAL_RUN_TRAINING=1
POD_EVAL_TRAIN_MODEL_ID=microsoft/Phi-4-mini-instruct
POD_EVAL_TRAIN_OUTPUT_DIR=/workspace/nlsh-finetune/phi-4-mini-instruct-lora
POD_EVAL_TRAIN_ARGS="--oom-retries 3"
POD_EVAL_EXIT_AFTER=1
```

You can also run commands manually in the pod:

```bash
export HF_HOME=/workspace/hf-cache
python scripts/pod_eval.py download-models
python scripts/pod_eval.py run-suite --dataset data/splits/v1/test
python -m nlsh.pod_workflow run --dry-run
```

The manifest is `configs/pod_eval_models.json`. It currently evaluates `microsoft/Phi-4-mini-instruct`, `HuggingFaceTB/SmolLM3-3B`, and `Qwen/Qwen3-8B` through SGLang with conservative single-GPU settings. The default `sglang_args` disable CUDA graph capture and force Triton attention plus PyTorch sampling, which is a safer starting point on Runpod Blackwell/RTX 5090 pods using the CUDA 12.8 base image. Downloads run in parallel; eval stays sequential in the preferred order so one GPU is used predictably. The workflow now defaults `POD_EVAL_SELECTED_MODELS` to Phi-4 only, which keeps serious training runs focused on the main model unless you explicitly broaden the batch. If an SGLang startup or serve attempt fails with an OOM, `scripts/pod_eval.py` retries with smaller `max_running_requests`, then shorter `context_length`, then lower `mem_fraction_static` until it hits the configured floor. If training hits an OOM, `scripts/phi_4_training.py` retries with smaller batch sizes, then shorter sequence length, resuming from the latest checkpoint when one exists.

Set `POD_EVAL_SELECTED_MODELS` to a comma-separated subset when you want to rerun a different batch, for example `POD_EVAL_SELECTED_MODELS=microsoft/Phi-4-mini-instruct,HuggingFaceTB/SmolLM3-3B`.

For local non-GPU checks, validate the manifest and exercise the eval path with the gold planner:

```bash
./.venv/bin/python scripts/pod_eval.py download-models --dry-run
./.venv/bin/python scripts/pod_eval.py run-model --planner gold --model microsoft/Phi-4-mini-instruct --limit 2
```

## Dataset And Training

The canonical, coverage-organized examples live in `data/samples/`; see `data/index.md` for the current map. Materialized fair splits live under `data/splits/v1/`.

The April 24, 2026 Runpod fine-tune findings and the follow-up dataset plan are captured in `docs/runpod-results-2026-04-24.md`.

Each row includes:

- `focus`
- `prompt`
- `tags`
- `messages`
- `plan`

For a direct single-GPU Phi-4-mini LoRA run outside the pod image on a 32 GB+ Runpod GPU, install the training extras and run the training script directly:

```bash
pip install -e ".[dev,train]"
python scripts/phi_4_training.py \
  --output-dir /workspace/nlsh-finetune/phi-4-mini-instruct-lora
```

By default the standalone training script still deterministically partitions `data/samples/` into train/eval records. The Runpod workflow now prefers the committed `data/splits/v1/train`, `data/splits/v1/eval`, and `data/splits/v1/test` tree so post-training evaluation stays held out from the training examples. The current split is `78` train, `25` eval, and `25` test. Regenerate those splits after editing the canonical data:

```bash
./.venv/bin/python scripts/materialize_dataset_splits.py
```

The script uses regular bf16 LoRA by default, maps dataset `developer` messages to chat `system` messages, trains on prompt/completion examples, and starts with `target_modules=["qkv_proj"]`, `r=8`, `lora_alpha=16`, and `lora_dropout=0.05`. The default training schedule is now more serious: `10` epochs, `per_device_train_batch_size=4`, `per_device_eval_batch_size=4`, `gradient_accumulation_steps=4`, and `learning_rate=5e-4`, with OOM retries reducing batch sizes and sequence length if the GPU cannot hold that footprint. It defaults to `--no-trust-remote-code` for Phi-4 so it can use the native `transformers` `Phi3*` classes and avoid upstream remote-code drift. It tries FlashAttention 2 when installed and otherwise falls back to SDPA. Checkpoints go under `--output-dir/checkpoint-*`; the final PEFT adapter is saved directly in `--output-dir`.

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
