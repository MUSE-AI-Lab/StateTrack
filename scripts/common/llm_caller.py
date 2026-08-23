"""统一 LLM API 调用层 v2 - 支持 thinking 模型 + reasoning effort + 高并发"""
import asyncio
import json
import time
import os
from pathlib import Path
from openai import AsyncOpenAI
from typing import Any

EXPERIMENT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = EXPERIMENT_ROOT.parent / "experiment_config.json"
_cost_file = EXPERIMENT_ROOT / "cost_log.jsonl"

def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

PROVIDERS = {
    "gpt-5.4": ("ikuncode_openai", "reasoning_effort"),
    "gpt-5.4-mini": ("msuicode_openai", None),
    "gemini-3.1-pro-preview": ("msuicode_gemini", None),
    "gemini-3.1-pro-preview-thinking": ("msuicode_gemini", "stream_newapi"),
    "gemini-3-pro-preview-thinking": ("msuicode_gemini", "stream_newapi"),
    "gemini-3-flash-preview": ("msuicode_gemini", None),
    "gemini-3-flash-preview-thinking": ("msuicode_gemini", "stream_newapi"),
    "claude-opus-4-6": ("msuicode_claude", None),
    "claude-opus-4-6-thinking": ("msuicode_claude", "stream_newapi"),
    "claude-sonnet-4-6": ("msuicode_claude", None),
    "claude-sonnet-4-6-thinking": ("msuicode_claude", "stream_newapi"),
    "qwen3.6-plus": ("dashscope", "stream_dashscope"),
    "qwen3.6-flash": ("dashscope", "stream_dashscope"),
    "deepseek-v4-pro": ("dashscope", "stream_dashscope"),
    "deepseek-v4-flash": ("dashscope", "stream_dashscope"),
    "kimi/kimi-k2.6": ("dashscope", None),
    "MiniMax/MiniMax-M2.7": ("dashscope", "stream_dashscope"),
}

_clients: dict[str, AsyncOpenAI] = {}
_semaphores: dict[str, asyncio.Semaphore] = {}
PROVIDER_TIMEOUT_SECONDS = {"dashscope": 180.0}

COST_PER_1K = {
    "gpt-5.4": {"in": 0.0025, "out": 0.015}, "gpt-5.4-mini": {"in": 0.0002, "out": 0.0008},
    "claude-opus-4-6": {"in": 0.015, "out": 0.075}, "claude-opus-4-6-thinking": {"in": 0.015, "out": 0.075},
    "claude-sonnet-4-6": {"in": 0.003, "out": 0.015}, "claude-sonnet-4-6-thinking": {"in": 0.003, "out": 0.015},
    "gemini-3.1-pro-preview": {"in": 0.00125, "out": 0.005}, "gemini-3.1-pro-preview-thinking": {"in": 0.00125, "out": 0.005},
    "gemini-3-pro-preview-thinking": {"in": 0.00125, "out": 0.005},
    "gemini-3-flash-preview": {"in": 0.0001, "out": 0.0004}, "gemini-3-flash-preview-thinking": {"in": 0.0001, "out": 0.0004},
    "deepseek-v4-pro": {"in": 0.002, "out": 0.008}, "deepseek-v4-flash": {"in": 0.0003, "out": 0.0012},
    "qwen3.6-plus": {"in": 0.0008, "out": 0.002}, "qwen3.6-flash": {"in": 0.0001, "out": 0.0005},
    "kimi/kimi-k2.6": {"in": 0.001, "out": 0.004}, "MiniMax/MiniMax-M2.7": {"in": 0.001, "out": 0.004},
}


def _get_client(provider_name: str) -> AsyncOpenAI:
    if provider_name not in _clients:
        cfg = load_config()["providers"][provider_name]
        timeout = PROVIDER_TIMEOUT_SECONDS.get(provider_name, 90.0)
        _clients[provider_name] = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=timeout, max_retries=0)
    return _clients[provider_name]


def _get_semaphore(provider_name: str) -> asyncio.Semaphore:
    if provider_name not in _semaphores:
        limits = {"msuicode_claude": 3, "msuicode_gemini": 4, "ikuncode_openai": 4, "msuicode_openai": 4, "dashscope": 8}
        _semaphores[provider_name] = asyncio.Semaphore(limits.get(provider_name, 5))
    return _semaphores[provider_name]


def _log_cost(model, in_tokens, out_tokens, experiment, duration_s):
    rates = COST_PER_1K.get(model, {"in": 0.001, "out": 0.004})
    cost = (in_tokens / 1000) * rates["in"] + (out_tokens / 1000) * rates["out"]
    record = {"ts": time.time(), "model": model, "in_tokens": in_tokens, "out_tokens": out_tokens,
              "cost_usd": round(cost, 6), "duration_s": round(duration_s, 2), "experiment": experiment}
    with open(_cost_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def call_llm(model: str, messages: list[dict], experiment: str = "unknown",
                   max_tokens: int = 2048, temperature: float = 0.0,
                   enable_thinking: bool = True, max_retries: int = 6) -> dict[str, Any]:
    provider_name, thinking_mode = PROVIDERS.get(model, ("dashscope", None))
    client = _get_client(provider_name)
    sem = _get_semaphore(provider_name)

    # 部分 OpenAI-compatible 网关的 non-stream 聚合不稳定，会产生截断/拼接 JSON。
    # ikuncode_openai 与 msuicode_claude 统一直接消费 SSE，绕过网关聚合层。
    force_stream = provider_name in {"ikuncode_openai", "msuicode_claude"}
    use_stream = (enable_thinking and thinking_mode in ("stream_dashscope", "stream_newapi")) or force_stream
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}

    if enable_thinking and thinking_mode == "reasoning_effort":
        kwargs["extra_body"] = {"reasoning": {"effort": "high"}}
    if enable_thinking and thinking_mode == "stream_dashscope":
        kwargs["extra_body"] = {"enable_thinking": True}
    if use_stream:
        kwargs["stream"] = True

    for attempt in range(max_retries):
        try:
            async with sem:
                t0 = time.time()
                if use_stream:
                    thinking_content, answer_content, in_tokens, out_tokens = "", "", 0, 0
                    async def _consume():
                        nonlocal thinking_content, answer_content, in_tokens, out_tokens
                        s = await client.chat.completions.create(**kwargs)
                        async for chunk in s:
                            if not chunk.choices: continue
                            d = chunk.choices[0].delta
                            if hasattr(d, "reasoning_content") and d.reasoning_content:
                                thinking_content += d.reasoning_content
                            if hasattr(d, "content") and d.content:
                                answer_content += d.content
                            if hasattr(chunk, "usage") and chunk.usage:
                                in_tokens = chunk.usage.prompt_tokens or 0
                                out_tokens = chunk.usage.completion_tokens or 0
                    await asyncio.wait_for(_consume(), timeout=120.0)
                    duration = time.time() - t0
                    if not answer_content.strip() and thinking_content.strip() and not enable_thinking:
                        answer_content, thinking_content = thinking_content, ""
                    if not answer_content.strip() and not enable_thinking:
                        raise ValueError("API returned empty streaming content")
                    if not in_tokens: in_tokens = sum(len(m.get("content",""))//4 for m in messages)
                    if not out_tokens: out_tokens = (len(thinking_content)+len(answer_content))//4
                    _log_cost(model, in_tokens, out_tokens, experiment, duration)
                    return {"content": answer_content, "thinking": thinking_content, "model": model,
                            "in_tokens": in_tokens, "out_tokens": out_tokens, "duration_s": round(duration,2)}
                else:
                    resp = await client.chat.completions.create(**kwargs)
                    duration = time.time() - t0
                    # 防御: 某些代理返回 str 而非正常 response
                    if isinstance(resp, str):
                        raise ValueError(f"API returned raw string instead of response object: {resp[:100]}")
                    if not resp.choices:
                        raise ValueError(f"API returned empty choices: {resp}")
                    message = resp.choices[0].message
                    content = message.content or getattr(message, "reasoning_content", "") or ""
                    if not str(content).strip():
                        raise ValueError("API returned empty content")
                    in_t = resp.usage.prompt_tokens if resp.usage else 0
                    out_t = resp.usage.completion_tokens if resp.usage else 0
                    _log_cost(model, in_t, out_t, experiment, duration)
                    return {"content": content, "thinking": "", "model": model,
                            "in_tokens": in_t, "out_tokens": out_t, "duration_s": round(duration,2)}
        except Exception as e:
            err_type = type(e).__name__
            err_str = f"{err_type}: {e}"
            if hasattr(e, '__cause__') and e.__cause__:
                err_str = f"{err_type}: {e.__cause__}"
            elif hasattr(e, 'args') and e.args:
                err_str = f"{err_type}: {e.args[0] if len(e.args)==1 else e.args}"
            err_short = err_str[:300]
            is_connection = "onnect" in err_str or "timed out" in err_str.lower() or "timeout" in err_str.lower()
            # 400/401/403/422 = 客户端错误，重试无意义
            is_client_error = "400" in err_str[:50] or "401" in err_str[:50] or "403" in err_str[:50] or "422" in err_str[:50]
            if is_client_error:
                print(f"  [SKIP] {model}: {err_short}")
                return {"content": "", "thinking": "", "model": model, "error": err_short,
                        "in_tokens": 0, "out_tokens": 0, "duration_s": 0}
            if attempt < max_retries - 1:
                wait = min(2 ** (attempt + 1) + (5 if is_connection else 0), 60)
                if attempt == 0 or attempt % 3 == 0:
                    print(f"  [Retry {attempt+1}/{max_retries-1}] {model}: {err_short}")
                    print(f"    -> waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"  [FAILED after {max_retries} retries] {model}: {err_short}")
                return {"content": "", "thinking": "", "model": model, "error": err_short,
                        "in_tokens": 0, "out_tokens": 0, "duration_s": 0}


async def call_batch(tasks: list[dict], concurrency: int = 10) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    async def _run(t):
        async with sem:
            return await call_llm(**t)
    return await asyncio.gather(*[_run(t) for t in tasks])


async def call_batch_resumable(tasks: list[dict], cache_path: Path, concurrency: int = 10) -> list[dict]:
    """带断点续传的批量调用。已完成的结果从 cache 读取，失败的重跑。"""
    cached = []
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"  [Resume] Loaded {len(cached)} cached results from {cache_path.name}")
        except: cached = []

    # 找出需要重跑的 index（没有缓存 或 缓存中有 error）
    results = [None] * len(tasks)
    pending_indices = []
    for i in range(len(tasks)):
        if i < len(cached) and cached[i] and not cached[i].get("error"):
            results[i] = cached[i]
        else:
            pending_indices.append(i)

    if not pending_indices:
        print(f"  [Resume] All {len(tasks)} tasks already completed!")
        return cached

    print(f"  [Resume] Running {len(pending_indices)}/{len(tasks)} pending tasks (concurrency={concurrency})...")
    sem = asyncio.Semaphore(concurrency)
    async def _run(idx):
        async with sem:
            return idx, await call_llm(**tasks[idx])

    batch_results = await asyncio.gather(*[_run(i) for i in pending_indices])
    for idx, result in batch_results:
        results[idx] = result

    # 保存到 cache（包括成功和失败的）
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    failed = sum(1 for r in results if r and r.get("error"))
    if failed:
        print(f"  [Resume] {failed} tasks still failed after retries. Re-run to retry them.")
    return results


def get_total_cost(experiment: str = None) -> float:
    if not _cost_file.exists(): return 0.0
    total = 0.0
    for line in _cost_file.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip(): continue
        r = json.loads(line)
        if experiment is None or r.get("experiment") == experiment:
            total += r.get("cost_usd", 0)
    return round(total, 4)
