import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def load_exfil_module() -> ModuleType:
    path = Path(".codex/skills/runpod-artifact-exfil/scripts/runpod_exfil_report.py")
    spec = importlib.util.spec_from_file_location("runpod_exfil_report", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_bundle(root: Path) -> Path:
    artifact_dir = root / "workspace" / "nlsh-artifacts"
    adapter_dir = root / "workspace" / "nlsh-finetune" / "phi-4-mini-instruct-lora"
    (artifact_dir / "microsoft_Phi-4-mini-instruct").mkdir(parents=True)
    (artifact_dir / "post-training-eval" / "microsoft_Phi-4-mini-instruct").mkdir(parents=True)
    adapter_dir.mkdir(parents=True)

    (artifact_dir / "workflow_state.json").write_text(
        json.dumps(
            {
                "started_at": "2026-04-24T11:08:00+00:00",
                "finished_at": "2026-04-24T11:14:29+00:00",
                "exit_codes": {"last_exit_code": 0},
                "training": {"status": "completed"},
                "post_training_eval": {"status": "completed"},
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "post_training_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "baseline": {"exact_match_rate": 0.48, "compile_valid_rate": 1.0, "slot_accuracy": 0.39},
                "fine_tuned": {"exact_match_rate": 0.88, "compile_valid_rate": 1.0, "slot_accuracy": 0.89},
                "delta": {"exact_match_rate": 0.4, "slot_accuracy": 0.5},
            }
        ),
        encoding="utf-8",
    )
    baseline_item = {
        "index": 1,
        "prompt": "find every PDF under ./invoices no deeper than two levels",
        "expected": {"kind": "plan", "steps": [{"kind": "find_files", "root": "./invoices", "glob": "*.pdf", "max_depth": 2}]},
        "predicted": {"kind": "clarification", "question": "Do you want to find all PDF files under the './invoices' directory without going deeper than two levels?"},
        "exact_match": False,
        "slot_correct": 0,
        "slot_total": 1,
        "diffs": {"kind": {"expected": "plan", "actual": "clarification"}},
    }
    fine_item = {
        "index": 1,
        "prompt": baseline_item["prompt"],
        "expected": baseline_item["expected"],
        "predicted": baseline_item["expected"],
        "exact_match": True,
        "slot_correct": 1,
        "slot_total": 1,
        "diffs": {},
    }
    (artifact_dir / "microsoft_Phi-4-mini-instruct" / "report.json").write_text(json.dumps({"items": [baseline_item]}), encoding="utf-8")
    (artifact_dir / "post-training-eval" / "microsoft_Phi-4-mini-instruct" / "report.json").write_text(
        json.dumps({"items": [fine_item]}),
        encoding="utf-8",
    )
    (artifact_dir / "workflow.log").write_text("2026-04-24 11:14:29,860 INFO workflow finished with exit code 0\n", encoding="utf-8")
    (adapter_dir / "adapter_run_info.json").write_text(
        json.dumps(
            {
                "base_model": "microsoft/Phi-4-mini-instruct",
                "adapter_type": "lora",
                "resolved_attention_implementation": "sdpa",
                "args": {
                    "train_dataset": "/opt/nlsh/data/splits/v1/train",
                    "train_records": 78,
                    "eval_dataset": "/opt/nlsh/data/splits/v1/eval",
                    "eval_records": 25,
                    "training": {
                        "per_device_train_batch_size": 4,
                        "per_device_eval_batch_size": 4,
                        "gradient_accumulation_steps": 4,
                        "learning_rate": 5e-4,
                        "num_train_epochs": 10.0,
                        "steps_per_epoch": 5,
                        "logging_steps": 5,
                        "evaluation_strategy": "epoch",
                        "save_strategy": "epoch",
                        "max_length": 2048,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "training_state.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (adapter_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 5, "epoch": 1.0, "loss": 1.0, "learning_rate": 0.001, "grad_norm": 0.5, "mean_token_accuracy": 0.7},
                    {"step": 5, "epoch": 1.0, "eval_loss": 0.4, "eval_mean_token_accuracy": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )
    return root


def test_runpod_exfil_analysis_writes_metric_artifacts(tmp_path: Path) -> None:
    module = load_exfil_module()
    bundle_root = make_bundle(tmp_path / "bundle")

    payload = module.analyze_bundle(bundle_root)
    report = module.render_report(payload)

    assert (bundle_root / "metrics-history.json").exists()
    assert (bundle_root / "metrics-history.csv").exists()
    assert (bundle_root / "training-metrics.svg").exists()
    assert "## Training Setup" in report
    assert "Training chart" in report
    assert "Improved exact-match items" in report
