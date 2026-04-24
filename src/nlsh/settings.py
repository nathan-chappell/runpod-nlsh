from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def runpod_proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


@dataclass(frozen=True, slots=True)
class RunpodServeSettings:
    host: str
    port: int
    model_id: str | None
    adapter_name: str | None
    api_key: str | None
    sglang_args: tuple[str, ...]
    pod_id: str | None

    @classmethod
    def from_env(cls) -> "RunpodServeSettings":
        load_dotenv()
        return cls(
            host=os.environ.get("RUNPOD_SERVE_HOST", "0.0.0.0"),
            port=int(os.environ.get("RUNPOD_SERVE_PORT", "8000")),
            model_id=os.environ.get("RUNPOD_SERVE_MODEL_ID") or None,
            adapter_name=os.environ.get("RUNPOD_SERVE_ADAPTER_NAME") or None,
            api_key=os.environ.get("RUNPOD_SERVE_API_KEY") or None,
            sglang_args=tuple(shlex.split(os.environ.get("RUNPOD_SERVE_SGLANG_ARGS", ""))),
            pod_id=os.environ.get("RUNPOD_POD_ID") or None,
        )

    @property
    def proxy_url(self) -> str | None:
        if not self.pod_id:
            return None
        return runpod_proxy_url(self.pod_id, self.port)
