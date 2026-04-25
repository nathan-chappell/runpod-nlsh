---
name: runpod-live-test
description: Resolve, authenticate, and smoke-test the public Runpod proxy endpoint for the bundled NLSH serving image. Use when Codex needs to turn a Runpod pod id or proxy URL into a live API base URL, load `RUNPOD_SERVE_API_KEY` from `.env`, verify `/model_info` and `/v1/models`, or send a small `/v1/chat/completions` request to the deployed `microsoft/Phi-4-mini-instruct:nlsh-phi4-ft` service.
---

# Runpod Live Test

Use the helper script in `scripts/runpod_live_test.py`.

## Quick Start

Resolve the public proxy URL from a pod id:

```bash
python .codex/skills/runpod-live-test/scripts/runpod_live_test.py resolve-url \
  --pod-id YOUR_POD_ID
```

Smoke-test the deployed service:

```bash
python .codex/skills/runpod-live-test/scripts/runpod_live_test.py smoke \
  --pod-id YOUR_POD_ID
```

Run a saved 10-sample stratified probe against the live runtime prompt shape:

```bash
python .codex/skills/runpod-live-test/scripts/runpod_live_test.py probe-dataset \
  --proxy-url https://YOURPOD-8000.proxy.runpod.net
```

Run a sandboxed interactive demo that keeps trying prompts until one compiles,
gets confirmed, and executes successfully:

```bash
python .codex/skills/runpod-live-test/scripts/runpod_live_test.py interactive-demo \
  --proxy-url https://YOURPOD-8000.proxy.runpod.net
```

The script loads `.env` automatically and uses:

- `RUNPOD_SERVE_API_KEY` for bearer auth
- `RUNPOD_SERVE_PORT` with a default of `8000`
- request model `microsoft/Phi-4-mini-instruct:nlsh-phi4-ft` by default

## URL Resolution

Use one of these inputs:

- `--pod-id`: builds `https://<pod-id>-<port>.proxy.runpod.net`
- `--proxy-url`: use an already-known proxy URL, with or without `/v1`

If you do not know the pod id:

- check the Runpod console Pod details
- check startup logs for `proxy_url=...`
- inside the pod, Runpod exposes `RUNPOD_POD_ID`

Global networking is not enough for laptop/browser access. Public testing requires `8000` in Runpod `Expose HTTP Ports`.

## What Is Being Served

The image serves:

- base model: `microsoft/Phi-4-mini-instruct`
- bundled LoRA adapter: `nlsh-phi4-ft`
- request model: `microsoft/Phi-4-mini-instruct:nlsh-phi4-ft`

Endpoints worth checking:

- native SGLang: `/model_info`
- OpenAI-compatible: `/v1/models`
- OpenAI-compatible: `/v1/chat/completions`

## Probe And Demo Modes

`probe-dataset`:

- samples one canonical example for each major output bucket
- uses the same runtime planner prompt shape as the app by default
- writes timestamped JSON plus `artifacts/live-probe/latest.json`
- supports `--mode replay-messages` for raw dataset-message diagnostics

`interactive-demo`:

- creates a disposable sandbox under `tmp/live-demo/<timestamp>`
- generates local CSV, JSON, and PDF fixtures
- tries real prompts against the served endpoint until one succeeds
- shows the normalized plan and compiled shell preview
- asks for confirmation unless `--yes` is supplied
- writes `execution-transcript.json` inside the sandbox

## Notes

Do not print or paste the bearer token into chat replies. Summarize whether auth is configured instead of echoing the secret.
