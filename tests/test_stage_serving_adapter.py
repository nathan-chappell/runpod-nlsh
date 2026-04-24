import json
import subprocess
import sys
from pathlib import Path


def make_bundle(root: Path) -> tuple[Path, Path]:
    artifact_dir = root / "workspace" / "nlsh-artifacts"
    adapter_dir = root / "workspace" / "nlsh-finetune" / "phi-4-mini-instruct-lora"
    artifact_dir.mkdir(parents=True)
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_run_info.json").write_text(
        json.dumps({"base_model": "microsoft/Phi-4-mini-instruct"}),
        encoding="utf-8",
    )
    (artifact_dir / "post_training_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "base_model": "microsoft/Phi-4-mini-instruct",
                "baseline": {"exact_match_rate": 0.48},
                "fine_tuned": {"exact_match_rate": 0.88},
                "delta": {"exact_match_rate": 0.4},
            }
        ),
        encoding="utf-8",
    )
    return artifact_dir, adapter_dir


def test_stage_serving_adapter(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    make_bundle(bundle_root)
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    (staging_dir / ".gitignore").write_text("*\n", encoding="utf-8")
    (staging_dir / "README.md").write_text("placeholder\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/stage_serving_adapter.py",
            "--bundle-root",
            str(bundle_root),
            "--staging-dir",
            str(staging_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["base_model"] == "microsoft/Phi-4-mini-instruct"
    assert payload["adapter_name"] == "nlsh-phi4-ft"
    assert (staging_dir / "adapter_model.safetensors").exists()
    assert (staging_dir / "adapter_config.json").exists()
    manifest = json.loads((staging_dir / "bundled_adapter_manifest.json").read_text())
    assert manifest["base_model"] == "microsoft/Phi-4-mini-instruct"
    assert manifest["post_training_summary"]["fine_tuned"]["exact_match_rate"] == 0.88
