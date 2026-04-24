#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_STAGING_DIR = Path("deploy/bundled-adapter/current")
DEFAULT_ADAPTER_NAME = "nlsh-phi4-ft"
REQUIRED_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
)
OPTIONAL_FILES = (
    "adapter_run_info.json",
    "training_state.json",
    "trainer_state.json",
    "metrics_history.json",
    "metrics_history.csv",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "all_results.json",
    "eval_results.json",
    "train_results.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def locate_bundle(bundle_root: Path) -> tuple[Path, Path]:
    artifact_dir = bundle_root / "workspace" / "nlsh-artifacts"
    if not artifact_dir.exists():
        artifact_dir = bundle_root / "nlsh-artifacts"
    if not artifact_dir.exists():
        raise SystemExit(f"Could not find nlsh-artifacts under {bundle_root}")

    adapter_dir = bundle_root / "workspace" / "nlsh-finetune" / "phi-4-mini-instruct-lora"
    if not adapter_dir.exists():
        adapter_dir = bundle_root / "nlsh-finetune" / "phi-4-mini-instruct-lora"
    if not adapter_dir.exists():
        raise SystemExit(f"Could not find phi-4-mini-instruct-lora under {bundle_root}")

    return artifact_dir, adapter_dir


def clear_staging_dir(staging_dir: Path) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    for child in staging_dir.iterdir():
        if child.name in {".gitignore", "README.md"}:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_selected_files(adapter_dir: Path, staging_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in (*REQUIRED_FILES, *OPTIONAL_FILES):
        source = adapter_dir / name
        if not source.exists():
            if name in REQUIRED_FILES:
                raise SystemExit(f"Missing required adapter file: {source}")
            continue
        shutil.copy2(source, staging_dir / name)
        copied.append(name)
    return copied


def determine_base_model(adapter_dir: Path, artifact_dir: Path) -> str:
    adapter_run_info = load_json(adapter_dir / "adapter_run_info.json") or {}
    if isinstance(adapter_run_info.get("base_model"), str):
        return adapter_run_info["base_model"]
    summary = load_json(artifact_dir / "post_training_summary.json") or {}
    if isinstance(summary.get("base_model"), str):
        return summary["base_model"]
    raise SystemExit("Could not determine the base model from adapter_run_info.json or post_training_summary.json")


def write_manifest(
    *,
    staging_dir: Path,
    bundle_root: Path,
    artifact_dir: Path,
    adapter_dir: Path,
    base_model: str,
    adapter_name: str,
    copied_files: list[str],
) -> Path:
    summary = load_json(artifact_dir / "post_training_summary.json") or {}
    manifest = {
        "built_at": utc_now(),
        "source_bundle_root": str(bundle_root.resolve()),
        "source_artifact_dir": str(artifact_dir.resolve()),
        "source_adapter_dir": str(adapter_dir.resolve()),
        "base_model": base_model,
        "adapter_name": adapter_name,
        "copied_files": copied_files,
        "post_training_summary": {
            "status": summary.get("status"),
            "baseline": summary.get("baseline"),
            "fine_tuned": summary.get("fine_tuned"),
            "delta": summary.get("delta"),
        },
    }
    path = staging_dir / "bundled_adapter_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage a fine-tuned LoRA adapter into the Docker build context.")
    parser.add_argument("--bundle-root", type=Path, required=True, help="Exfiltrated Runpod bundle root.")
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--adapter-name", default=DEFAULT_ADAPTER_NAME)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle_root = args.bundle_root.resolve()
    staging_dir = args.staging_dir.resolve()
    artifact_dir, adapter_dir = locate_bundle(bundle_root)
    clear_staging_dir(staging_dir)
    copied_files = copy_selected_files(adapter_dir, staging_dir)
    base_model = determine_base_model(adapter_dir, artifact_dir)
    manifest_path = write_manifest(
        staging_dir=staging_dir,
        bundle_root=bundle_root,
        artifact_dir=artifact_dir,
        adapter_dir=adapter_dir,
        base_model=base_model,
        adapter_name=args.adapter_name,
        copied_files=copied_files,
    )
    print(
        json.dumps(
            {
                "bundle_root": str(bundle_root),
                "staging_dir": str(staging_dir),
                "base_model": base_model,
                "adapter_name": args.adapter_name,
                "manifest": str(manifest_path),
                "copied_files": copied_files,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
