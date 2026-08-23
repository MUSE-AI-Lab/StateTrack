from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.I | re.S)
    if fenced:
        value = fenced.group(1)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", value):
        try:
            parsed, _ = decoder.raw_decode(value[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("judge response did not contain a JSON object")


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def call_json_judge(
    *,
    system: str,
    prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 500,
    timeout_s: int = 180,
    retries: int = 5,
) -> dict[str, Any]:
    endpoint_root = (base_url or os.environ["MSUI_BASE_URL"]).rstrip("/")
    secret = api_key or os.environ["MSUI_TEXT_KEY"]
    model_name = model or os.environ.get("MSUI_TEXT_MODEL", "gemini-3.6-flash-c2")
    endpoint = f"{endpoint_root}/chat/completions"
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    last_error = ""
    started = time.perf_counter()
    for attempt in range(retries):
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
            content = _message_text(raw["choices"][0]["message"])
            return {
                "model": model_name,
                "parsed": parse_json_object(content),
                "raw_content": content,
                "usage": raw.get("usage", {}),
                "duration_s": time.perf_counter() - started,
                "request_id": raw.get("id", ""),
            }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:800]}"
            if exc.code == 400 and "response_format" in json.dumps(payload):
                payload.pop("response_format", None)
        except Exception as exc:  # provider and transient transport errors
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < retries:
            time.sleep(min(2**attempt, 12))
    raise RuntimeError(last_error or "judge request failed")


def load_or_call(cache_path: Path, **kwargs: Any) -> dict[str, Any]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    result = call_json_judge(**kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(cache_path)
    return result
