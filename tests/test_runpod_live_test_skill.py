import json
import os
import subprocess
import sys
from pathlib import Path


SKILL_SCRIPT = Path(".codex/skills/runpod-live-test/scripts/runpod_live_test.py")


def test_runpod_live_test_skill_exists() -> None:
    assert Path(".codex/skills/runpod-live-test/SKILL.md").exists()
    assert SKILL_SCRIPT.exists()


def test_runpod_live_test_resolve_url() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL_SCRIPT),
            "resolve-url",
            "--pod-id",
            "abc123xyz",
            "--port",
            "8000",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "RUNPOD_SERVE_API_KEY": "", "RUNPOD_POD_ID": ""},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["proxy_url"] == "https://abc123xyz-8000.proxy.runpod.net"
    assert payload["openai_base_url"] == "https://abc123xyz-8000.proxy.runpod.net/v1"
    assert payload["api_key_configured"] is False


def test_runpod_live_test_resolve_url_accepts_v1_proxy_url() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SKILL_SCRIPT),
            "resolve-url",
            "--proxy-url",
            "https://abc123xyz-8000.proxy.runpod.net/v1",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "RUNPOD_SERVE_API_KEY": "", "RUNPOD_POD_ID": ""},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["proxy_url"] == "https://abc123xyz-8000.proxy.runpod.net"
    assert payload["openai_base_url"] == "https://abc123xyz-8000.proxy.runpod.net/v1"
