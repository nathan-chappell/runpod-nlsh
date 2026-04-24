#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import shlex
import subprocess
import sys
import tarfile
import textwrap
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

SKILL_DIR = Path(__file__).resolve().parents[1]
STATE_FILE = SKILL_DIR / "state" / "last_connection.json"
DEFAULT_ARTIFACT_ROOT = Path("tmp/runpod-downloads")
DEFAULT_REMOTE_ARTIFACT_DIR = "/workspace/nlsh-artifacts"
DEFAULT_REMOTE_FINETUNE_DIR = "/workspace/nlsh-finetune/phi-4-mini-instruct-lora"
DEFAULT_LOG_SEARCH_ROOTS = ("/workspace", "/opt/nlsh")
DEFAULT_STRICT_HOST_KEY = ("-o", "StrictHostKeyChecking=no")
MARKER_META = "__EXFIL_META__"
MARKER_BEGIN = "__BEGIN_BASE64__"
MARKER_END = "__END_BASE64__"
ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
ANSI_OSC_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)")
BASE64_LINE_RE = re.compile(r"^[A-Za-z0-9+/=]+$")
ERROR_PATTERNS = (
    "Traceback",
    "ERROR",
    "RuntimeError",
    "ImportError",
    "out of memory",
    "CUDA",
)
SSH_OPTIONS_WITH_VALUES = {
    "-b",
    "-c",
    "-D",
    "-E",
    "-e",
    "-F",
    "-I",
    "-i",
    "-J",
    "-L",
    "-l",
    "-m",
    "-O",
    "-o",
    "-p",
    "-Q",
    "-R",
    "-S",
    "-W",
    "-w",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_terminal_output(text: str) -> str:
    text = text.replace("\r", "")
    text = ANSI_OSC_RE.sub("", text)
    text = ANSI_CSI_RE.sub("", text)
    return text


def json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return slug or "runpod"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_state(payload: dict[str, Any]) -> None:
    ensure_parent(STATE_FILE)
    STATE_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_state() -> dict[str, Any] | None:
    if not STATE_FILE.exists():
        return None
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def parse_ssh_command(raw: str) -> list[str]:
    tokens = shlex.split(raw)
    if not tokens:
        raise SystemExit("Empty --ssh-command")
    if tokens[0] == "ssh":
        tokens = tokens[1:]
    if not tokens:
        raise SystemExit("No SSH host found in --ssh-command")
    return tokens


def ssh_tokens_from_parts(host: str, identity_file: str | None) -> list[str]:
    tokens: list[str] = []
    if identity_file:
        tokens.extend(["-i", identity_file])
    tokens.append(host)
    return tokens


def extract_ssh_host(tokens: list[str]) -> str | None:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            if index + 1 < len(tokens):
                return tokens[index + 1]
            return None
        if token in SSH_OPTIONS_WITH_VALUES:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return None


def canonicalize_ssh_tokens(raw_tokens: list[str]) -> list[str]:
    host = extract_ssh_host(raw_tokens)
    if not host:
        return list(raw_tokens)

    options: list[str] = []
    host_consumed = False
    index = 0
    while index < len(raw_tokens):
        token = raw_tokens[index]
        if token == "--":
            break
        if token in SSH_OPTIONS_WITH_VALUES:
            if index + 1 < len(raw_tokens):
                options.extend([token, raw_tokens[index + 1]])
            else:
                options.append(token)
            index += 2
            continue
        if token.startswith("-"):
            options.append(token)
            index += 1
            continue
        if not host_consumed and token == host:
            host_consumed = True
            index += 1
            continue
        options.append(token)
        index += 1
    return [*options, host]


def effective_ssh_tokens(raw_tokens: list[str]) -> list[str]:
    tokens = canonicalize_ssh_tokens(raw_tokens)
    if "-tt" not in tokens and "-t" not in tokens:
        tokens.insert(0, "-tt")
    if "StrictHostKeyChecking=no" not in " ".join(tokens):
        tokens = [*DEFAULT_STRICT_HOST_KEY, *tokens]
    return tokens


def resolve_connection(args: argparse.Namespace) -> dict[str, Any]:
    if args.ssh_command:
        raw_tokens = parse_ssh_command(args.ssh_command)
    elif args.host:
        raw_tokens = ssh_tokens_from_parts(args.host, args.identity_file)
    else:
        state = load_state()
        if not state:
            raise SystemExit("No connection parameters supplied and no saved Runpod connection is available.")
        raw_tokens = state["ssh_tokens"]

    tokens = effective_ssh_tokens(raw_tokens)
    host = extract_ssh_host(raw_tokens)
    if not host:
        raise SystemExit("Could not resolve SSH host from the supplied connection parameters.")

    return {
        "host": host,
        "ssh_tokens": raw_tokens,
        "effective_ssh_command": ["ssh", *tokens],
        "resolved_at": utc_now(),
    }


def unique_output_dir(root: Path, slug: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / slug
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = root / f"{slug}-{index}"
        if not candidate.exists():
            return candidate
        index += 1


def default_bundle_root(host: str) -> Path:
    today_root = DEFAULT_ARTIFACT_ROOT / str(date.today())
    if host.endswith("@ssh.runpod.io"):
        slug_source = host.split("@", 1)[0]
    else:
        slug_source = host
    return unique_output_dir(today_root, normalize_slug(slug_source))


def remote_exfil_program(extra_remote_paths: list[str]) -> str:
    extra_paths_json = json.dumps(extra_remote_paths, ensure_ascii=False)
    search_roots_json = json.dumps(list(DEFAULT_LOG_SEARCH_ROOTS), ensure_ascii=False)
    return textwrap.dedent(
        f"""
        python - <<'PY'
        from __future__ import annotations
        import base64
        import io
        import json
        import os
        import tarfile
        from pathlib import Path

        MARKER_META = {MARKER_META!r}
        MARKER_BEGIN = {MARKER_BEGIN!r}
        MARKER_END = {MARKER_END!r}
        archive_path = Path("/workspace/tmp") / f"codex-exfil-{{os.getpid()}}.tgz"
        base_items = [
            Path({DEFAULT_REMOTE_ARTIFACT_DIR!r}),
            Path({DEFAULT_REMOTE_FINETUNE_DIR!r}),
        ]
        search_roots = [Path(item) for item in {search_roots_json}]
        extra_paths = [Path(item) for item in {extra_paths_json}]
        included: list[dict[str, str]] = []
        seen_logs: set[str] = set()

        def log_candidates() -> list[Path]:
            candidates: list[Path] = []
            for root in search_roots:
                if not root.exists():
                    continue
                queue = [(root, 0)]
                while queue:
                    current, depth = queue.pop(0)
                    if depth > 3:
                        continue
                    if current.is_file():
                        if current.name == "logs.txt":
                            resolved = str(current)
                            if resolved not in seen_logs:
                                seen_logs.add(resolved)
                                candidates.append(current)
                        continue
                    try:
                        children = list(current.iterdir())
                    except OSError:
                        continue
                    for child in children:
                        if child.is_file() and child.name == "logs.txt":
                            resolved = str(child)
                            if resolved not in seen_logs:
                                seen_logs.add(resolved)
                                candidates.append(child)
                        elif child.is_dir():
                            queue.append((child, depth + 1))
            return sorted(candidates)

        def add_path(tar: tarfile.TarFile, source: Path, arcname: str) -> None:
            tar.add(str(source), arcname=arcname)
            included.append({{"source": str(source), "arcname": arcname}})

        with tarfile.open(archive_path, "w:gz") as tar:
            for item in base_items:
                if item.exists():
                    add_path(tar, item, item.as_posix().lstrip("/"))
            for item in extra_paths:
                if item.exists():
                    add_path(tar, item, "extra-files/" + item.as_posix().lstrip("/"))
            for log_path in log_candidates():
                add_path(tar, log_path, "extra-files/" + log_path.as_posix().lstrip("/"))
            manifest = {{
                "included": included,
                "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            }}
            data = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
            info = tarfile.TarInfo("exfil_manifest.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        print(MARKER_META + json.dumps({{
            "included_count": len(included),
            "included": included,
            "archive_path": str(archive_path),
        }}, ensure_ascii=False))
        print(MARKER_BEGIN)
        with archive_path.open("rb") as handle:
            while True:
                chunk = handle.read(57 * 1024)
                if not chunk:
                    break
                print(base64.b64encode(chunk).decode("ascii"))
        print(MARKER_END)
        archive_path.unlink(missing_ok=True)
        PY
        exit
        """
    ).strip() + "\n"


def run_remote_exfil(connection: dict[str, Any], output_dir: Path, extra_remote_paths: list[str]) -> dict[str, Any]:
    script = remote_exfil_program(extra_remote_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_b64 = output_dir / "runpod-export.tgz.b64"
    archive_tgz = output_dir / "runpod-export.tgz"
    if archive_b64.exists():
        archive_b64.unlink()
    if archive_tgz.exists():
        archive_tgz.unlink()

    proc = subprocess.Popen(
        connection["effective_ssh_command"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(script)
    proc.stdin.close()

    meta: dict[str, Any] | None = None
    saw_begin = False
    saw_end = False
    in_payload = False
    with archive_b64.open("w", encoding="ascii") as handle:
        for raw_line in proc.stdout:
            line = clean_terminal_output(raw_line).strip()
            if not line:
                continue
            if line.startswith(MARKER_META):
                meta = json.loads(line[len(MARKER_META):])
                continue
            if line == MARKER_BEGIN:
                in_payload = True
                saw_begin = True
                continue
            if line == MARKER_END:
                saw_end = True
                in_payload = False
                continue
            if in_payload and BASE64_LINE_RE.match(line):
                handle.write(line)

    return_code = proc.wait(timeout=300)
    if return_code != 0:
        raise SystemExit(f"SSH exfiltration failed with exit code {return_code}")
    if not saw_begin or not saw_end:
        raise SystemExit("Did not receive a complete archive stream from the Runpod pod.")
    if meta is None:
        meta = {"included": []}

    archive_tgz.write_bytes(base64.b64decode(archive_b64.read_text(encoding="ascii")))
    with tarfile.open(archive_tgz, "r:gz") as tar:
        tar.extractall(path=output_dir)

    meta["archive_path"] = str(archive_tgz)
    meta["output_dir"] = str(output_dir)
    return meta


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def locate_bundle(bundle_root: Path) -> dict[str, Path | None]:
    artifact_dir = bundle_root / "workspace" / "nlsh-artifacts"
    finetune_root = bundle_root / "workspace" / "nlsh-finetune"
    if not artifact_dir.exists():
        artifact_dir = bundle_root / "nlsh-artifacts"
    adapter_dir = None
    if finetune_root.exists():
        adapter_dirs = [path for path in finetune_root.iterdir() if path.is_dir()]
        adapter_dir = adapter_dirs[0] if adapter_dirs else None
    else:
        direct_adapter = bundle_root / "nlsh-finetune" / "phi-4-mini-instruct-lora"
        if direct_adapter.exists():
            adapter_dir = direct_adapter
    return {
        "artifact_dir": artifact_dir if artifact_dir.exists() else None,
        "adapter_dir": adapter_dir,
    }


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def slot_accuracy(item: dict[str, Any]) -> float:
    total = item.get("slot_total", 0)
    if not total:
        return 1.0
    return float(item.get("slot_correct", 0)) / float(total)


def expected_category(item: dict[str, Any]) -> str:
    expected = item.get("expected", {})
    if expected.get("kind") == "clarification":
        return "clarification"
    steps = expected.get("steps", [])
    if len(steps) == 1:
        return steps[0].get("kind", "plan")
    if steps:
        return "pipeline:" + "->".join(step.get("kind", "?") for step in steps)
    return "unknown"


def notable_log_lines(path: Path, limit: int = 8) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits: list[str] = []
    for line in lines:
        if any(pattern in line for pattern in ERROR_PATTERNS):
            hits.append(line.strip())
        if "workflow finished with exit code" in line:
            hits.append(line.strip())
    unique_hits: list[str] = []
    for line in hits:
        if line not in unique_hits:
            unique_hits.append(line)
    return unique_hits[:limit]


def normalize_metric_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in rows:
        step = item.get("step")
        epoch = item.get("epoch")
        if not isinstance(step, (int, float)):
            continue
        if item.get("phase") in {"train", "eval"}:
            normalized.append(item)
            continue
        if "loss" in item or "learning_rate" in item or "grad_norm" in item:
            normalized.append(
                {
                    "phase": "train",
                    "step": step,
                    "epoch": epoch,
                    "loss": item.get("loss"),
                    "token_accuracy": item.get("mean_token_accuracy"),
                    "entropy": item.get("entropy"),
                    "learning_rate": item.get("learning_rate"),
                    "grad_norm": item.get("grad_norm"),
                    "num_tokens": item.get("num_tokens"),
                }
            )
        if "eval_loss" in item:
            normalized.append(
                {
                    "phase": "eval",
                    "step": step,
                    "epoch": epoch,
                    "loss": item.get("eval_loss"),
                    "token_accuracy": item.get("eval_mean_token_accuracy"),
                    "entropy": item.get("eval_entropy"),
                    "learning_rate": None,
                    "grad_norm": None,
                    "num_tokens": item.get("eval_num_tokens"),
                    "runtime": item.get("eval_runtime"),
                    "samples_per_second": item.get("eval_samples_per_second"),
                    "steps_per_second": item.get("eval_steps_per_second"),
                }
            )
    return normalized


def load_metric_history(adapter_dir: Path | None) -> list[dict[str, Any]]:
    if adapter_dir is None:
        return []
    metrics_history = load_json(adapter_dir / "metrics_history.json")
    if isinstance(metrics_history, list):
        return normalize_metric_history(metrics_history)
    trainer_state = load_json(adapter_dir / "trainer_state.json") or {}
    log_history = trainer_state.get("log_history")
    if isinstance(log_history, list):
        return normalize_metric_history(log_history)
    return []


def write_metric_history_artifacts(bundle_root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    json_path = bundle_root / "metrics-history.json"
    csv_path = bundle_root / "metrics-history.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = [
        "phase",
        "step",
        "epoch",
        "loss",
        "token_accuracy",
        "entropy",
        "learning_rate",
        "grad_norm",
        "num_tokens",
        "runtime",
        "samples_per_second",
        "steps_per_second",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return json_path, csv_path


def _metric_points(rows: list[dict[str, Any]], *, phase: str, key: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        if row.get("phase") != phase:
            continue
        step = row.get("step")
        value = row.get(key)
        if isinstance(step, (int, float)) and isinstance(value, (int, float)):
            points.append((float(step), float(value)))
    return points


def _value_range(series: list[list[tuple[float, float]]]) -> tuple[float, float]:
    values = [value for points in series for _step, value in points]
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    if lo == hi:
        padding = 1.0 if lo == 0 else abs(lo) * 0.1
        return lo - padding, hi + padding
    padding = (hi - lo) * 0.1
    return lo - padding, hi + padding


def render_metric_svg(rows: list[dict[str, Any]]) -> str:
    metric_specs = [
        ("Loss", [("Train loss", "train", "loss", "#1f77b4"), ("Eval loss", "eval", "loss", "#d62728")]),
        ("Token accuracy", [("Train token accuracy", "train", "token_accuracy", "#2ca02c"), ("Eval token accuracy", "eval", "token_accuracy", "#ff7f0e")]),
        ("Learning rate", [("Learning rate", "train", "learning_rate", "#9467bd")]),
        ("Gradient norm", [("Grad norm", "train", "grad_norm", "#8c564b")]),
    ]
    active_specs: list[tuple[str, list[tuple[str, str, str, str]], list[list[tuple[float, float]]]]] = []
    for title, series_specs in metric_specs:
        series = [_metric_points(rows, phase=phase, key=key) for _label, phase, key, _color in series_specs]
        if any(series):
            active_specs.append((title, series_specs, series))

    width = 960
    panel_height = 180
    top_margin = 24
    left_margin = 56
    right_margin = 24
    inner_height = 108
    chart_height = max(1, len(active_specs)) * panel_height + 32
    max_step = max((float(row["step"]) for row in rows if isinstance(row.get("step"), (int, float))), default=1.0)

    def x_pos(step: float) -> float:
        usable = width - left_margin - right_margin
        if max_step <= 1:
            return left_margin + usable / 2
        return left_margin + (step / max_step) * usable

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{chart_height}" viewBox="0 0 {width} {chart_height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:monospace;font-size:12px;fill:#222} .axis{stroke:#999;stroke-width:1} .grid{stroke:#eee;stroke-width:1} .series{fill:none;stroke-width:2}</style>',
        '<text x="24" y="18" font-size="16">Runpod training metrics</text>',
    ]

    for panel_index, (title, series_specs, series) in enumerate(active_specs):
        panel_top = 32 + panel_index * panel_height
        panel_bottom = panel_top + inner_height
        lo, hi = _value_range(series)
        svg_lines.append(f'<text x="24" y="{panel_top + 14}">{title}</text>')
        svg_lines.append(f'<line class="axis" x1="{left_margin}" y1="{panel_bottom}" x2="{width - right_margin}" y2="{panel_bottom}"/>')
        svg_lines.append(f'<line class="axis" x1="{left_margin}" y1="{panel_top}" x2="{left_margin}" y2="{panel_bottom}"/>')
        for tick_ratio in (0.0, 0.5, 1.0):
            y = panel_bottom - tick_ratio * inner_height
            value = lo + (hi - lo) * tick_ratio
            svg_lines.append(f'<line class="grid" x1="{left_margin}" y1="{y}" x2="{width - right_margin}" y2="{y}"/>')
            svg_lines.append(f'<text x="8" y="{y + 4}">{value:.3g}</text>')
        legend_x = width - right_margin - 180
        legend_y = panel_top + 12
        for legend_index, (label, phase, key, color) in enumerate(series_specs):
            points = _metric_points(rows, phase=phase, key=key)
            if not points:
                continue
            plotted = []
            for step, value in points:
                if hi == lo:
                    y = panel_top + inner_height / 2
                else:
                    y = panel_bottom - ((value - lo) / (hi - lo)) * inner_height
                plotted.append((x_pos(step), y))
            point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in plotted)
            svg_lines.append(f'<polyline class="series" stroke="{color}" points="{point_text}"/>')
            for x, y in plotted:
                svg_lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" fill="{color}"/>')
            legend_row = legend_y + legend_index * 16
            svg_lines.append(f'<line x1="{legend_x}" y1="{legend_row}" x2="{legend_x + 12}" y2="{legend_row}" stroke="{color}" stroke-width="2"/>')
            svg_lines.append(f'<text x="{legend_x + 18}" y="{legend_row + 4}">{label}</text>')

    svg_lines.append("</svg>")
    return "\n".join(svg_lines) + "\n"


def write_metric_chart(bundle_root: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    chart_path = bundle_root / "training-metrics.svg"
    chart_path.write_text(render_metric_svg(rows), encoding="utf-8")
    return chart_path


def analyze_bundle(bundle_root: Path) -> dict[str, Any]:
    located = locate_bundle(bundle_root)
    artifact_dir = located["artifact_dir"]
    adapter_dir = located["adapter_dir"]
    if artifact_dir is None:
        raise SystemExit(f"Could not find nlsh-artifacts under {bundle_root}")

    workflow_state = load_json(artifact_dir / "workflow_state.json")
    summary = load_json(artifact_dir / "post_training_summary.json")
    baseline_report = load_json(artifact_dir / "microsoft_Phi-4-mini-instruct" / "report.json")
    fine_report = load_json(artifact_dir / "post-training-eval" / "microsoft_Phi-4-mini-instruct" / "report.json")
    extra_logs = sorted(bundle_root.rglob("logs.txt"))
    adapter_run_info = None if adapter_dir is None else load_json(adapter_dir / "adapter_run_info.json")
    training_state = None if adapter_dir is None else load_json(adapter_dir / "training_state.json")
    metrics_history = load_metric_history(adapter_dir)
    metrics_history_json_path, metrics_history_csv_path = write_metric_history_artifacts(bundle_root, metrics_history)
    metric_chart_path = write_metric_chart(bundle_root, metrics_history)

    report_payload: dict[str, Any] = {
        "bundle_root": str(bundle_root),
        "artifact_dir": str(artifact_dir),
        "adapter_dir": None if adapter_dir is None else str(adapter_dir),
        "extra_logs": [str(path) for path in extra_logs],
        "workflow_state": workflow_state,
        "summary": summary,
        "adapter_run_info": adapter_run_info,
        "training_state": training_state,
        "metrics_history": metrics_history,
        "metrics_history_json": str(metrics_history_json_path),
        "metrics_history_csv": str(metrics_history_csv_path),
        "metric_chart": None if metric_chart_path is None else str(metric_chart_path),
    }

    improved_items: list[dict[str, Any]] = []
    regressed_items: list[dict[str, Any]] = []
    improvement_counts = {
        "improved_exact": 0,
        "regressed_exact": 0,
        "same_exact": 0,
        "improved_slot": 0,
        "regressed_slot": 0,
        "same_slot": 0,
    }

    if baseline_report and fine_report:
        for baseline_item, fine_item in zip(baseline_report.get("items", []), fine_report.get("items", [])):
            baseline_exact = bool(baseline_item.get("exact_match"))
            fine_exact = bool(fine_item.get("exact_match"))
            baseline_slot = slot_accuracy(baseline_item)
            fine_slot = slot_accuracy(fine_item)
            if fine_exact and not baseline_exact:
                improvement_counts["improved_exact"] += 1
            elif baseline_exact and not fine_exact:
                improvement_counts["regressed_exact"] += 1
            else:
                improvement_counts["same_exact"] += 1
            if fine_slot > baseline_slot:
                improvement_counts["improved_slot"] += 1
            elif fine_slot < baseline_slot:
                improvement_counts["regressed_slot"] += 1
            else:
                improvement_counts["same_slot"] += 1

            row = {
                "index": baseline_item.get("index"),
                "prompt": baseline_item.get("prompt"),
                "category": expected_category(baseline_item),
                "baseline_exact": baseline_exact,
                "fine_exact": fine_exact,
                "baseline_slot": baseline_slot,
                "fine_slot": fine_slot,
                "expected": baseline_item.get("expected"),
                "baseline_predicted": baseline_item.get("predicted"),
                "fine_predicted": fine_item.get("predicted"),
                "baseline_diffs": baseline_item.get("diffs"),
                "fine_diffs": fine_item.get("diffs"),
            }
            if fine_exact and not baseline_exact:
                improved_items.append(row)
            if baseline_exact and not fine_exact:
                regressed_items.append(row)

    log_summaries: dict[str, list[str]] = {}
    for rel_path in (
        "workflow.log",
        "training.log",
        "post_training_eval.log",
        "microsoft_Phi-4-mini-instruct/eval.log",
        "post-training-eval/microsoft_Phi-4-mini-instruct/eval.log",
    ):
        path = artifact_dir / rel_path
        hits = notable_log_lines(path)
        if hits:
            log_summaries[str(path)] = hits
    for path in extra_logs:
        hits = notable_log_lines(path)
        if hits:
            log_summaries[str(path)] = hits

    report_payload["improvement_counts"] = improvement_counts
    report_payload["improved_items"] = improved_items[:5]
    report_payload["regressed_items"] = regressed_items
    report_payload["log_summaries"] = log_summaries
    return report_payload


def render_report(payload: dict[str, Any]) -> str:
    workflow_state = payload.get("workflow_state") or {}
    summary = payload.get("summary") or {}
    adapter_run_info = payload.get("adapter_run_info") or {}
    training_state = payload.get("training_state") or {}
    resolved_args = adapter_run_info.get("args") or training_state.get("resolved_args") or {}
    training_config = resolved_args.get("training") or {}
    lines: list[str] = []
    lines.append("# Runpod Artifact Report")
    lines.append("")
    lines.append("## Bundle")
    lines.append("")
    lines.append(f"- Bundle root: `{payload['bundle_root']}`")
    lines.append(f"- Artifact dir: `{payload['artifact_dir']}`")
    if payload.get("adapter_dir"):
        lines.append(f"- Adapter dir: `{payload['adapter_dir']}`")
    if payload.get("extra_logs"):
        lines.append("- Extra `logs.txt` files:")
        for path in payload["extra_logs"]:
            lines.append(f"  - `{path}`")
    else:
        lines.append("- Extra `logs.txt` files: none found")
    lines.append(f"- Metrics history JSON: `{payload['metrics_history_json']}`")
    lines.append(f"- Metrics history CSV: `{payload['metrics_history_csv']}`")
    if payload.get("metric_chart"):
        lines.append(f"- Training chart: `{payload['metric_chart']}`")
    lines.append("")

    if workflow_state:
        lines.append("## Workflow")
        lines.append("")
        lines.append(f"- Started: `{workflow_state.get('started_at')}`")
        lines.append(f"- Finished: `{workflow_state.get('finished_at')}`")
        exit_codes = workflow_state.get("exit_codes", {})
        lines.append(f"- Last exit code: `{exit_codes.get('last_exit_code')}`")
        training = workflow_state.get("training") or {}
        post_eval = workflow_state.get("post_training_eval") or {}
        lines.append(f"- Training status: `{training.get('status')}`")
        lines.append(f"- Post-training eval status: `{post_eval.get('status')}`")
        lines.append("")

    if resolved_args or adapter_run_info:
        lines.append("## Training Setup")
        lines.append("")
        if isinstance(adapter_run_info.get("base_model"), str):
            lines.append(f"- Base model: `{adapter_run_info.get('base_model')}`")
        if isinstance(adapter_run_info.get("adapter_type"), str):
            lines.append(f"- Adapter type: `{adapter_run_info.get('adapter_type')}`")
        if isinstance(adapter_run_info.get("resolved_attention_implementation"), str):
            lines.append(f"- Attention implementation: `{adapter_run_info.get('resolved_attention_implementation')}`")
        if resolved_args:
            lines.append(f"- Train dataset: `{resolved_args.get('train_dataset')}`")
            lines.append(f"- Train records: `{resolved_args.get('train_records')}`")
            lines.append(f"- Eval dataset: `{resolved_args.get('eval_dataset')}`")
            lines.append(f"- Eval records: `{resolved_args.get('eval_records')}`")
        if training_config:
            lines.append(f"- Train batch size: `{training_config.get('per_device_train_batch_size')}`")
            lines.append(f"- Eval batch size: `{training_config.get('per_device_eval_batch_size')}`")
            lines.append(f"- Grad accumulation: `{training_config.get('gradient_accumulation_steps')}`")
            lines.append(f"- Learning rate: `{training_config.get('learning_rate')}`")
            lines.append(f"- Epochs: `{training_config.get('num_train_epochs')}`")
            lines.append(f"- Steps per epoch: `{training_config.get('steps_per_epoch')}`")
            lines.append(f"- Logging steps: `{training_config.get('logging_steps')}`")
            lines.append(f"- Eval strategy: `{training_config.get('evaluation_strategy')}`")
            lines.append(f"- Save strategy: `{training_config.get('save_strategy')}`")
            lines.append(f"- Max length: `{training_config.get('max_length')}`")
        lines.append("")

    if summary:
        baseline = summary.get("baseline", {})
        fine_tuned = summary.get("fine_tuned", {})
        delta = summary.get("delta", {})
        lines.append("## Metrics")
        lines.append("")
        lines.append(f"- Baseline exact match: `{baseline.get('exact_match_rate')}`")
        lines.append(f"- Fine-tuned exact match: `{fine_tuned.get('exact_match_rate')}`")
        lines.append(f"- Baseline compile-valid: `{baseline.get('compile_valid_rate')}`")
        lines.append(f"- Fine-tuned compile-valid: `{fine_tuned.get('compile_valid_rate')}`")
        lines.append(f"- Baseline slot accuracy: `{baseline.get('slot_accuracy')}`")
        lines.append(f"- Fine-tuned slot accuracy: `{fine_tuned.get('slot_accuracy')}`")
        lines.append(f"- Exact-match delta: `{delta.get('exact_match_rate')}`")
        lines.append(f"- Slot-accuracy delta: `{delta.get('slot_accuracy')}`")
        lines.append("")

    counts = payload.get("improvement_counts", {})
    if counts:
        lines.append("## Item Deltas")
        lines.append("")
        lines.append(f"- Improved exact-match items: `{counts.get('improved_exact')}`")
        lines.append(f"- Regressed exact-match items: `{counts.get('regressed_exact')}`")
        lines.append(f"- Improved slot-accuracy items: `{counts.get('improved_slot')}`")
        lines.append(f"- Regressed slot-accuracy items: `{counts.get('regressed_slot')}`")
        lines.append("")

    improved_items = payload.get("improved_items", [])
    if improved_items:
        lines.append("## Representative Improvements")
        lines.append("")
        for item in improved_items:
            lines.append(f"### {item['index']}. {item['prompt']}")
            lines.append("")
            lines.append(f"- Category: `{item['category']}`")
            lines.append(f"- Baseline exact: `{item['baseline_exact']}`")
            lines.append(f"- Fine-tuned exact: `{item['fine_exact']}`")
            lines.append(f"- Baseline prediction: `{compact_json(item['baseline_predicted'])}`")
            lines.append(f"- Fine-tuned prediction: `{compact_json(item['fine_predicted'])}`")
            lines.append("")

    regressed_items = payload.get("regressed_items", [])
    if regressed_items:
        lines.append("## Regressions")
        lines.append("")
        for item in regressed_items:
            lines.append(f"### {item['index']}. {item['prompt']}")
            lines.append("")
            lines.append(f"- Expected: `{compact_json(item['expected'])}`")
            lines.append(f"- Baseline prediction: `{compact_json(item['baseline_predicted'])}`")
            lines.append(f"- Fine-tuned prediction: `{compact_json(item['fine_predicted'])}`")
            lines.append("")

    log_summaries = payload.get("log_summaries", {})
    lines.append("## Log Review")
    lines.append("")
    if not log_summaries:
        lines.append("- No notable error or completion lines were detected in the extracted logs.")
        lines.append("")
    else:
        for path, hits in log_summaries.items():
            lines.append(f"### `{path}`")
            lines.append("")
            for hit in hits:
                lines.append(f"- {hit}")
            lines.append("")

    lines.append("## Suggested Follow-Up")
    lines.append("")
    if regressed_items:
        lines.append("- Add more underspecified merge clarifications to reinforce the \"do not invent input PDFs\" boundary.")
    lines.append("- Add paired clarification-vs-execution examples when a prompt is fully specified but still easy to over-clarify.")
    lines.append("- Add more glob-semantic contrast rows when path cues such as `*june*` versus `june*` matter.")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_report(bundle_root: Path, markdown: str) -> Path:
    path = bundle_root / "analysis-report.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def cmd_pull(args: argparse.Namespace) -> int:
    connection = resolve_connection(args)
    output_dir = Path(args.output_dir) if args.output_dir else default_bundle_root(connection["host"])
    if args.dry_run:
        print(json_dump({
            "connection": connection,
            "output_dir": str(output_dir),
            "extra_remote_paths": args.extra_remote_path,
        }))
        return 0

    meta = run_remote_exfil(connection, output_dir, args.extra_remote_path)
    state_payload = {
        "host": connection["host"],
        "ssh_tokens": connection["ssh_tokens"],
        "last_output_dir": str(output_dir),
        "updated_at": utc_now(),
    }
    save_state(state_payload)
    print(json_dump(meta))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    if args.bundle_root:
        bundle_root = Path(args.bundle_root)
    else:
        state = load_state()
        if not state or not state.get("last_output_dir"):
            raise SystemExit("No --bundle-root was supplied and there is no saved last output directory.")
        bundle_root = Path(state["last_output_dir"])

    payload = analyze_bundle(bundle_root)
    markdown = render_report(payload)
    report_path = write_report(bundle_root, markdown)
    print(markdown)
    print(f"Report written to {report_path}", file=sys.stderr)
    return 0


def cmd_pull_and_analyze(args: argparse.Namespace) -> int:
    pull_args = argparse.Namespace(**vars(args))
    cmd_pull(pull_args)
    analyze_args = argparse.Namespace(bundle_root=args.output_dir)
    if not args.output_dir:
        state = load_state()
        analyze_args.bundle_root = None if state is None else state.get("last_output_dir")
    return cmd_analyze(analyze_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exfiltrate Runpod artifacts and generate an analysis report.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_connection_options(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--ssh-command", help="Full SSH connection command, for example: ssh user@ssh.runpod.io -i ~/.ssh/id_ed25519")
        subparser.add_argument("--host", help="Runpod SSH host, for example: user@ssh.runpod.io")
        subparser.add_argument("--identity-file", help="SSH identity file path")
        subparser.add_argument("--output-dir", help="Local extraction directory. Defaults to ./tmp/runpod-downloads/<date>/<pod-slug>")
        subparser.add_argument("--extra-remote-path", action="append", default=[], help="Additional remote path to include in the archive")
        subparser.add_argument("--dry-run", action="store_true", help="Print the resolved connection and exit without connecting")

    pull_parser = subparsers.add_parser("pull", help="Exfiltrate remote artifacts without analyzing them")
    add_connection_options(pull_parser)
    pull_parser.set_defaults(func=cmd_pull)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze an already extracted bundle")
    analyze_parser.add_argument("--bundle-root", help="Extracted bundle directory. Defaults to the last pulled bundle.")
    analyze_parser.set_defaults(func=cmd_analyze)

    combo_parser = subparsers.add_parser("pull-and-analyze", help="Exfiltrate remote artifacts and immediately analyze them")
    add_connection_options(combo_parser)
    combo_parser.set_defaults(func=cmd_pull_and_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
