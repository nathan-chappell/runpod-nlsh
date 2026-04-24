---
name: runpod-artifact-exfil
description: Fetch Runpod training and eval artifacts over SSH, including any available `logs.txt`, then analyze the extracted bundle and write a Markdown report with workflow status, metrics, improvements, regressions, and dataset follow-up ideas. Use when the user asks to exfiltrate Runpod pod data, pull artifacts from `ssh.runpod.io`, inspect `workflow.log` or `logs.txt`, compare baseline versus fine-tuned evals, or produce a report similar to prior Runpod result summaries.
---

# Runpod Artifact Exfil

## Overview

Use this skill to pull a Runpod pod's artifact directories and logs into the current repo, then generate a reusable analysis report from the extracted bundle, including static training charts and resolved hyperparameters.

The main entry point is `scripts/runpod_exfil_report.py`.

## Workflow

1. Resolve connection parameters.
2. Exfiltrate `/workspace/nlsh-artifacts`, `/workspace/nlsh-finetune/phi-4-mini-instruct-lora`, and any discovered `logs.txt`.
3. Extract the archive under `./tmp/runpod-downloads/<date>/<pod-slug>/`.
4. Analyze the extracted reports and logs.
5. Write `analysis-report.md`, `metrics-history.json`, `metrics-history.csv`, and `training-metrics.svg` inside the extracted bundle and summarize the key findings to the user.

## Connection Rules

- If the user gives a fresh SSH command such as `ssh user@ssh.runpod.io -i ~/.ssh/id_ed25519`, pass it with `--ssh-command`.
- If the user gives discrete parameters, pass `--host` and `--identity-file`.
- If the user gives nothing new, reuse the last saved connection automatically.
- If there is no saved connection and nothing was supplied this turn, ask the user for the connection parameters.

The script persists the last successful connection in `state/last_connection.json` so it can be reused on later turns or on another machine with the same skill folder.

## Commands

Pull and analyze in one step:

```bash
python .codex/skills/runpod-artifact-exfil/scripts/runpod_exfil_report.py pull-and-analyze \
  --ssh-command "ssh user@ssh.runpod.io -i ~/.ssh/id_ed25519"
```

Reuse the saved connection:

```bash
python .codex/skills/runpod-artifact-exfil/scripts/runpod_exfil_report.py pull-and-analyze
```

Analyze an already extracted bundle again:

```bash
python .codex/skills/runpod-artifact-exfil/scripts/runpod_exfil_report.py analyze \
  --bundle-root tmp/runpod-downloads/2026-04-24/pod-id
```

## Output

The script writes:

- the extracted artifact bundle under `./tmp/runpod-downloads/...`
- `analysis-report.md` inside that bundle
- `metrics-history.json`, `metrics-history.csv`, and `training-metrics.svg` inside that bundle when training history is available
- updated connection state under `state/last_connection.json`

The report covers:

- workflow completion and exit codes
- baseline vs fine-tuned metrics when available
- resolved training hyperparameters and dataset sizes
- training metric charts over time
- representative improved items
- regressions and near-misses
- notable log signals from `workflow.log`, `training.log`, `post_training_eval.log`, and `logs.txt`
- suggested dataset follow-up work

## Notes

- The script already searches common Runpod locations for `logs.txt`; do not forget to mention whether one was found.
- If the user wants extra files beyond the default artifact roots, pass `--extra-remote-path` one or more times.
- Prefer `pull-and-analyze` unless the user explicitly wants only the raw exfiltration step.
