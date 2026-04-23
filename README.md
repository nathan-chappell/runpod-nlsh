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
- A pod eval image and manifest for sequential vLLM model comparisons

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
./.venv/bin/nlsh eval --planner gold --split test
```

## Pod Eval

The pod eval flow builds on Runpod's PyTorch base image, adds the project, configs, datasets, system tools, and a compiler toolchain, then installs vLLM and model weights at pod startup into Runpod persistent storage. This keeps the pushed image from baking model weights while matching Runpod's GPU/CUDA environment.

```bash
docker build -f Dockerfile.pod-eval -t YOUR_DOCKERHUB_USER/nlsh-pod-eval:latest .
docker push YOUR_DOCKERHUB_USER/nlsh-pod-eval:latest
```

The default base is `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`. To try another Runpod base, pass `--build-arg RUNPOD_BASE_IMAGE=...`.

Create a Runpod pod with:

- GPU: RTX 4090
- Image: `YOUR_DOCKERHUB_USER/nlsh-pod-eval:latest`
- Persistent or network volume mounted at `/workspace`
- `HF_TOKEN` set from a Runpod secret
- Recommended volume size: at least 100 GB
- Container disk: use 20 GB for the Runpod PyTorch base image. The model, vLLM, tmp, and cache data still live on `/workspace`, but the base image itself is large enough that 10 GB is likely too tight if Runpod counts image/root filesystem unpacking there.

The image starts Runpod's base services when `/start.sh` is present, then starts a full dev eval automatically. On first boot it creates `/workspace/nlsh-venv`, installs vLLM there, downloads or reuses model weights in `/workspace/hf-cache`, keeps temp/Triton/vLLM caches under `/workspace`, writes reports to `/workspace/nlsh-artifacts`, records the final code in `/workspace/nlsh-artifacts/last_exit_code`, and then keeps the container alive for inspection. Set `POD_EVAL_EXIT_AFTER=1` to exit after the batch instead.

Optional runtime knobs:

```bash
POD_EVAL_LIMIT=2
POD_EVAL_TIMEOUT=90
POD_EVAL_STARTUP_TIMEOUT=900
POD_EVAL_VLLM_SPEC=vllm
POD_EVAL_START_RUNPOD_SERVICES=1
```

You can also run commands manually in the pod:

```bash
export HF_HOME=/workspace/hf-cache
python scripts/pod_eval.py download-models
python scripts/pod_eval.py run-suite --dataset data/dev.messages.jsonl
```

The manifest is `configs/pod_eval_models.json`. It currently runs `microsoft/Phi-4-mini-instruct`, `Qwen/Qwen3-8B`, and `HuggingFaceTB/SmolLM3-3B` sequentially through vLLM with conservative 24GB settings.

For local non-GPU checks, validate the manifest and exercise the eval path with the gold planner:

```bash
./.venv/bin/python scripts/pod_eval.py download-models --dry-run
./.venv/bin/python scripts/pod_eval.py run-model --planner gold --model microsoft/Phi-4-mini-instruct --limit 2
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

For a direct single-GPU Phi-4-mini LoRA run on a 48 GB Runpod GPU, install the training extras in the pod and run the training script directly:

```bash
pip install -e .[train]
python scripts/phi_4_training.py \
  --train-dataset data/train.messages.jsonl \
  --eval-dataset data/dev.messages.jsonl \
  --output-dir /workspace/nlsh-finetune/phi-4-mini-instruct-lora
```

The script uses regular bf16 LoRA by default, maps dataset `developer` messages to chat `system` messages, trains on prompt/completion examples, and starts with `target_modules=["qkv_proj"]`, `r=8`, `lora_alpha=16`, and `lora_dropout=0.05`. It tries FlashAttention 2 when installed and otherwise falls back to SDPA. Checkpoints go under `--output-dir/checkpoint-*`; the final PEFT adapter is saved directly in `--output-dir`.

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
