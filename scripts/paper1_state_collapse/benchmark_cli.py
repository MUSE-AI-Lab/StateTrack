"""Portable execution and row-level replay of the released StateTrack protocol.

Only the Python standard library is required. Scenario banks and historical
checkers are unchanged; missing checkers remain unscored rather than successes.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize(rows):
    """Match the original category-then-depth averaging and 3-digit rounding."""
    matrix = {}
    seen = set()
    for row in rows:
        key = (row["model"], row["state_type"], row["scenario_idx"])
        if key in seen:
            raise ValueError(f"Duplicate scenario result: {key}")
        seen.add(key)
        depth_rows = matrix.setdefault(row["model"], {}).setdefault(row["state_type"], {})
        depths = set()
        for probe in row["probes"]:
            turn = str(probe["turn"])
            if turn in depths:
                raise ValueError(f"Duplicate probe depth: {key}, {turn}")
            depths.add(turn)
            value = probe["retained"]
            if value is not None and type(value) is not bool:
                raise ValueError(f"Invalid retention label: {key}, {turn}")
            values = depth_rows.setdefault(turn, [])
            if value is not None:
                values.append(int(value))
    scores = {model: {state: {turn: round(sum(values) / len(values), 3) if values else None
                              for turn, values in depths.items()}
                      for state, depths in states.items()} for model, states in matrix.items()}
    curves = {}
    for model, states in scores.items():
        curves[model] = {}
        turns = sorted({int(t) for depths in states.values() for t in depths})
        for turn in turns:
            values = [depths[str(turn)] for depths in states.values() if depths.get(str(turn)) is not None]
            curves[model][str(turn)] = round(sum(values) / len(values), 3) if values else None
    return scores, curves


def load_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError("Replay input must be a JSON array or JSONL scenario records")
    return rows


def validate_rows(rows, bank, cases, models):
    allowed = {(state, index) for state, index, _ in cases}
    expected_turns = bank["probe_turns"]
    for row in rows:
        if (row["state_type"], row["scenario_idx"]) not in allowed or row["model"] not in models:
            raise ValueError("Result identity does not match this run's selected scenarios/models")
        if [p["turn"] for p in row["probes"]] != expected_turns:
            raise ValueError("Incomplete or out-of-order probe schedule")
        scenario = bank["scenarios"][row["state_type"]][row["scenario_idx"]]
        for probe in row["probes"]:
            if (probe["retained"] is None) != (scenario["check"] is None):
                raise ValueError("Unchecked scenarios must have null labels; checked scenarios need boolean labels")
    summarize(rows)  # also rejects duplicate identities and non-boolean labels


class ChatClient:
    def __init__(self, base_url, api_key, *, timeout=120, retries=2, max_tokens=400, temperature=0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key = api_key
        self.timeout, self.retries = timeout, retries
        self.max_tokens, self.temperature = max_tokens, temperature

    def request(self, model, messages):
        body = json.dumps({"model": model, "messages": messages, "max_tokens": self.max_tokens,
                           "temperature": self.temperature}).encode()
        request = Request(self.url, data=body, headers={"Authorization": f"Bearer {self.key}",
                                                       "Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            data = json.load(response)
        content = data["choices"][0]["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Provider returned an empty or unsupported text response")
        return content

    async def complete(self, model, messages):
        for attempt in range(self.retries + 1):
            try:
                return await asyncio.to_thread(self.request, model, messages)
            except (HTTPError, URLError, TimeoutError) as exc:
                if isinstance(exc, HTTPError) and exc.code != 429 and exc.code < 500:
                    raise RuntimeError(f"Provider HTTP status {exc.code}") from None
                if attempt == self.retries:
                    raise RuntimeError("Provider request failed after retries") from None
                await asyncio.sleep(min(2 ** attempt, 8))


async def legacy_call_llm(model, messages, *, max_tokens=400, temperature=0, **kwargs):
    """Environment-configured adapter for the original importable runners."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY before running model generation")
    client = ChatClient(os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), key,
                        max_tokens=max_tokens, temperature=temperature)
    return {"content": await client.complete(model, messages)}


async def run_case(client, model, state, index, scenario, bank, suite):
    messages = [{"role": "user", "content": scenario["setup"]}]
    answer = await client.complete(model, messages)
    messages.append({"role": "assistant", "content": answer})
    probes = []
    for turn in range(2, bank["max_turns"] + 1):
        prompt = scenario["probe"] if turn in bank["probe_turns"] else bank["fillers"][(turn - 2) % len(bank["fillers"])]
        messages.append({"role": "user", "content": prompt})
        answer = await client.complete(model, messages)
        messages.append({"role": "assistant", "content": answer})
        if turn in bank["probe_turns"]:
            pattern = scenario["check"]
            retained = bool(re.search(pattern, answer, re.MULTILINE | re.DOTALL)) if pattern else None
            probes.append({"turn": turn, "retained": retained, "preview": answer[:150]})
    return {"model": model, "state_type": state, "scenario_idx": index,
            "suite": suite, "probes": probes, "messages": messages}


async def generate(args, bank, cases, models, previous):
    client = ChatClient(args.base_url, os.environ["OPENAI_API_KEY"], timeout=args.timeout,
                        max_tokens=args.max_tokens, temperature=args.temperature)
    rows = list(previous)
    completed = {(r["model"], r["state_type"], r["scenario_idx"]) for r in rows}
    semaphore = asyncio.Semaphore(args.concurrency)
    failures = []

    async def one(model, state, index, scenario):
        async with semaphore:
            if (model, state, index) in completed:
                return
            try:
                row = await run_case(client, model, state, index, scenario, bank, args.suite)
                with (args.output / "raw_results.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                print(f"COMPLETED {model} {state}[{index}]", flush=True)
            except Exception as exc:
                failures.append(f"{model} {state}[{index}]: {type(exc).__name__}: {exc}")

    await asyncio.gather(*(one(model, state, index, scenario) for model in models for state, index, scenario in cases))
    return rows, failures


def main(argv=None, *, default_suite="core"):
    parser = argparse.ArgumentParser(description="Run or replay the original StateTrack core/augmented benchmark", allow_abbrev=False)
    parser.add_argument("--suite", choices=("core", "augmented"), default=default_suite)
    parser.add_argument("--model", action="append", help="Provider model ID; repeat to evaluate multiple models")
    parser.add_argument("--state-type", action="append")
    parser.add_argument("--scenario-index", action="append", type=int)
    parser.add_argument("--limit", type=int, help="Limit the number of selected scenarios per model")
    parser.add_argument("--output", type=Path, default=None)
    offline = parser.add_mutually_exclusive_group()
    offline.add_argument("--dry-run", action="store_true", help="Export a run manifest without provider calls")
    offline.add_argument("--replay", type=Path, help="Recompute original metrics from cached scenario JSON/JSONL records")
    parser.add_argument("--resume", action="store_true", help="Resume complete dialogues in the same output directory")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args(argv)
    if args.resume and (args.dry_run or args.replay):
        parser.error("--resume is for online generation; choose a separate output for offline work")
    bank_path = ROOT / "benchmarks" / f"{args.suite}.json"
    bank = read_json(bank_path)
    if args.concurrency < 1 or args.timeout <= 0 or args.max_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("concurrency, timeout, max-tokens and limit must be positive")
    if args.state_type and set(args.state_type) - set(bank["scenarios"]):
        parser.error("unknown --state-type; choose from " + ", ".join(bank["scenarios"]))
    cases = [(state, index, scenario) for state, scenarios in bank["scenarios"].items() for index, scenario in enumerate(scenarios)
             if (not args.state_type or state in args.state_type) and (not args.scenario_index or index in args.scenario_index)]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        parser.error("no scenarios match the selection")
    args.output = (args.output or ROOT / "outputs" / args.suite).resolve()
    replay = load_rows(args.replay) if args.replay else None
    models = list(dict.fromkeys(args.model or (sorted({r["model"] for r in replay}) if replay else [])))
    if replay is not None and not replay:
        parser.error("replay input has no scenario records")
    if not args.dry_run and replay is None and (not models or not os.environ.get("OPENAI_API_KEY")):
        parser.error("online generation requires --model and OPENAI_API_KEY; use --dry-run or --replay for offline work")
    manifest = {"suite": args.suite, "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                "max_turns": bank["max_turns"], "probe_turns": bank["probe_turns"],
                "models": models, "scenario_count": len(cases),
                "probe_count_per_model": len(cases) * len(bank["probe_turns"]),
                "scenario_ids": [[state, index] for state, index, _ in cases],
                "max_tokens": args.max_tokens, "temperature": args.temperature, "base_url": args.base_url}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    raw_path = args.output / "raw_results.jsonl"
    previous = []
    if not args.dry_run and replay is None and raw_path.exists():
        if not args.resume:
            parser.error("output already contains raw results; use --resume or a new --output directory")
        if not manifest_path.exists() or read_json(manifest_path) != manifest:
            parser.error("resume configuration does not match the existing manifest")
        previous = load_rows(raw_path)
        validate_rows(previous, bank, cases, models)
    elif args.dry_run and raw_path.exists():
        parser.error("use a separate --output for dry runs so an existing run is preserved")
    write_json(manifest_path, manifest)
    if args.dry_run:
        print(f'DRY_RUN: {args.suite}; {len(cases)} scenarios; {manifest["probe_count_per_model"]} probes per model')
        return 0
    if replay is not None:
        selected = {(state, index) for state, index, _ in cases}
        rows = [row for row in replay if row["model"] in models and (row["state_type"], row["scenario_idx"]) in selected]
        failures = []
    else:
        rows, failures = asyncio.run(generate(args, bank, cases, models, previous))
    validate_rows(rows, bank, cases, models)
    matrix, curves = summarize(rows)
    expected = len(cases) * len(models)
    complete = len(rows) == expected and not failures
    write_json(args.output / "state_retention_matrix.json", matrix)
    write_json(args.output / "collapse_curves.json", curves)
    write_json(args.output / "run_summary.json", {"complete": complete, "expected_dialogues": expected,
                                                 "completed_dialogues": len(rows), "errors": failures})
    print(f'{"REPLAY" if replay is not None else "GENERATION"}: {"PASS" if complete else "INCOMPLETE"}; {len(rows)}/{expected} dialogues')
    for failure in failures:
        print(failure, file=sys.stderr)
    return 0 if complete else 1
