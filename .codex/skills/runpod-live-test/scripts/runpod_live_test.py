#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nlsh.settings import RunpodServeSettings, load_dotenv, runpod_proxy_url  # noqa: E402

DEFAULT_MODEL = "microsoft/Phi-4-mini-instruct:nlsh-phi4-ft"


def _request(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None,
    api_key: str | None,
    timeout: float,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    headers = {"Accept": "application/json"}
    data: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, method=method, headers=headers, data=data)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: dict[str, Any] | list[Any] | str = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return exc.code, parsed


def _resolve_urls(pod_id: str | None, proxy_url: str | None, port: int) -> tuple[str, str]:
    if proxy_url:
        root = proxy_url.rstrip("/")
        if root.endswith("/v1"):
            return root[:-3], root
        return root, f"{root}/v1"
    if not pod_id:
        raise SystemExit("Pass --pod-id or --proxy-url.")
    root = runpod_proxy_url(pod_id, port)
    return root, f"{root}/v1"


def command_resolve_url(args: argparse.Namespace) -> int:
    load_dotenv()
    settings = RunpodServeSettings.from_env()
    pod_id = args.pod_id or settings.pod_id
    port = args.port or settings.port
    proxy_root, openai_base = _resolve_urls(pod_id, args.proxy_url, port)
    payload = {
        "pod_id": pod_id,
        "proxy_url": proxy_root,
        "openai_base_url": openai_base,
        "model": args.model or DEFAULT_MODEL,
        "api_key_configured": bool(args.api_key or settings.api_key),
    }
    print(json.dumps(payload, indent=2))
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    load_dotenv()
    settings = RunpodServeSettings.from_env()
    pod_id = args.pod_id or settings.pod_id
    port = args.port or settings.port
    api_key = args.api_key or settings.api_key
    model = args.model or DEFAULT_MODEL
    proxy_root, openai_base = _resolve_urls(pod_id, args.proxy_url, port)

    model_info_status, model_info = _request(
        "GET",
        f"{proxy_root}/model_info",
        payload=None,
        api_key=api_key,
        timeout=args.timeout,
    )
    models_status, models = _request(
        "GET",
        f"{openai_base}/models",
        payload=None,
        api_key=api_key,
        timeout=args.timeout,
    )
    chat_status, chat = _request(
        "POST",
        f"{openai_base}/chat/completions",
        payload={
            "model": model,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        },
        api_key=api_key,
        timeout=args.timeout,
    )

    payload = {
        "pod_id": pod_id,
        "proxy_url": proxy_root,
        "openai_base_url": openai_base,
        "model": model,
        "api_key_configured": bool(api_key),
        "checks": {
            "model_info": {"status": model_info_status, "body": model_info},
            "models": {"status": models_status, "body": models},
            "chat_completions": {"status": chat_status, "body": chat},
        },
    }
    print(json.dumps(payload, indent=2))
    if model_info_status != 200 or models_status != 200 or chat_status != 200:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runpod_live_test.py",
        description="Resolve and smoke-test the public Runpod proxy URL for the bundled NLSH serving image.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve-url", help="Resolve the proxy URL and OpenAI base URL.")
    resolve_parser.add_argument("--pod-id", help="Runpod pod id. Defaults to RUNPOD_POD_ID when available.")
    resolve_parser.add_argument("--proxy-url", help="Existing proxy root or /v1 URL.")
    resolve_parser.add_argument("--port", type=int, default=8000)
    resolve_parser.add_argument("--api-key", help="Bearer token. Defaults to RUNPOD_SERVE_API_KEY.")
    resolve_parser.add_argument("--model", default=DEFAULT_MODEL)
    resolve_parser.set_defaults(func=command_resolve_url)

    smoke_parser = subparsers.add_parser("smoke", help="Call /model_info, /v1/models, and /v1/chat/completions.")
    smoke_parser.add_argument("--pod-id", help="Runpod pod id. Defaults to RUNPOD_POD_ID when available.")
    smoke_parser.add_argument("--proxy-url", help="Existing proxy root or /v1 URL.")
    smoke_parser.add_argument("--port", type=int, default=8000)
    smoke_parser.add_argument("--api-key", help="Bearer token. Defaults to RUNPOD_SERVE_API_KEY.")
    smoke_parser.add_argument("--model", default=DEFAULT_MODEL)
    smoke_parser.add_argument("--timeout", type=float, default=60.0)
    smoke_parser.add_argument("--max-tokens", type=int, default=24)
    smoke_parser.add_argument("--prompt", default="Reply with READY and nothing else.")
    smoke_parser.set_defaults(func=command_smoke)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
