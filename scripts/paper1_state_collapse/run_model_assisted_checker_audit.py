from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from experiment.common.json_judge import load_or_call
from scripts.paper1_state_collapse.analyze_checker_calibration import agreement_statistics


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "rebuttal_20260708" / "rebuttal_split_sic_raw.json"
OUTPUT = ROOT / "results" / "model_assisted_checker_audit"
SYSTEM = "You independently label dialogue outputs under an explicit rubric. Use only the supplied text. Return one JSON object and no prose."
VALID = {"pass", "fail", "unclear"}


def build_prompt(item: dict[str, Any], *, pass_name: str) -> str:
    if pass_name == "a":
        rubric = (
            "Judge three dimensions separately: task=whether the behavior response usefully addresses the immediate probe; "
            "constraint=whether that response obeys the persistent setup; recall=whether the recall response accurately states "
            "the active persistent constraint. Partial or merely stylistic evidence is insufficient when the setup requires an exact value."
        )
    else:
        rubric = (
            "Independently inspect exact persistent-rule compliance first, then whether the immediate request was addressed, then "
            "whether the separate recall branch names the active rule. Do not infer hidden intent and do not reuse any external labels."
        )
    return f"""{rubric}

Persistent setup:
{item['setup']}

Immediate probe:
{item['probe']}

Behavior response:
{item['response']}

Separate recall response:
{item['recall_response']}

Return exactly:
{{"task":"pass|fail|unclear","constraint":"pass|fail|unclear","recall":"pass|fail|unclear","task_rationale":"short reason","constraint_rationale":"short reason","recall_rationale":"short reason"}}"""


def _normalize(value: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key in ("task", "constraint", "recall"):
        label = str(value.get(key, "unclear")).strip().casefold()
        output[key] = label if label in VALID else "unclear"
        output[f"{key}_rationale"] = str(value.get(f"{key}_rationale", ""))[:600]
    return output


def reconcile_labels(a: dict[str, Any], b: dict[str, Any], adjudicator: dict[str, Any] | None) -> dict[str, str]:
    a = _normalize(a)
    b = _normalize(b)
    c = _normalize(adjudicator or {})
    return {
        key: a[key] if a[key] == b[key] else (c[key] if adjudicator is not None else "unclear")
        for key in ("task", "constraint", "recall")
    }


def _run_pass(items: list[dict[str, Any]], pass_name: str, workers: int) -> dict[str, dict[str, Any]]:
    def one(item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        result = load_or_call(
            OUTPUT / "raw" / pass_name / f"{item['item_id']}.json",
            system=SYSTEM,
            prompt=build_prompt(item, pass_name=pass_name),
        )
        return item["item_id"], _normalize(result["parsed"])

    output: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, item) for item in items]
        for future in as_completed(futures):
            item_id, result = future.result()
            output[item_id] = result
    return output


def _adjudicate(item: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    prompt = f"""Resolve only disputed labels under the original rubric. Prior passes are advisory.
Persistent setup: {item['setup']}
Immediate probe: {item['probe']}
Behavior response: {item['response']}
Recall response: {item['recall_response']}
Pass A: {json.dumps(a, ensure_ascii=False)}
Pass B: {json.dumps(b, ensure_ascii=False)}
Return exactly:
{{"task":"pass|fail|unclear","constraint":"pass|fail|unclear","recall":"pass|fail|unclear","task_rationale":"short reason","constraint_rationale":"short reason","recall_rationale":"short reason"}}"""
    result = load_or_call(OUTPUT / "raw" / "adjudication" / f"{item['item_id']}.json", system=SYSTEM, prompt=prompt)
    return _normalize(result["parsed"])


def _dimension_agreement(rows: list[dict[str, Any]], registered: str, audited: str) -> dict[str, Any]:
    clear = [row for row in rows if row[audited] in {"pass", "fail"}]
    pairs = [{"registered": bool(row[registered]), "audit": row[audited] == "pass"} for row in clear]
    result = agreement_statistics(pairs)
    result["excluded_unclear"] = len(rows) - len(clear)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    source_rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    items = []
    for row in source_rows:
        identity = f"{row['model']}|{row['scenario_id']}"
        items.append({"item_id": hashlib.sha256(identity.encode()).hexdigest()[:20], **row})
    if args.limit:
        items = items[: args.limit]
    pass_a = _run_pass(items, "a", args.max_workers)
    pass_b = _run_pass(items, "b", args.max_workers)

    rows = []
    for item in items:
        a = pass_a[item["item_id"]]
        b = pass_b[item["item_id"]]
        disputed = any(a[key] != b[key] for key in ("task", "constraint", "recall"))
        adjudicator = _adjudicate(item, a, b) if disputed else None
        final = reconcile_labels(a, b, adjudicator)
        rows.append(
            {
                "item_id": item["item_id"],
                "model": item["model"],
                "scenario_id": item["scenario_id"],
                "category": item["category"],
                "registered_task": bool(item["immediate_task_ok"]),
                "registered_constraint": bool(item["constraint_only_ok"]),
                "registered_recall": bool(item["recall_ok"]),
                "pass_a_task": a["task"],
                "pass_b_task": b["task"],
                "pass_a_constraint": a["constraint"],
                "pass_b_constraint": b["constraint"],
                "pass_a_recall": a["recall"],
                "pass_b_recall": b["recall"],
                "adjudicated": disputed,
                **final,
            }
        )

    payload = {
        "version": "statetrack-model-assisted-checker-audit-v1",
        "judge_model": __import__("os").environ.get("MSUI_TEXT_MODEL", "gemini-3.6-flash-c2"),
        "temperature": 0,
        "items": len(rows),
        "adjudicated_items": sum(bool(row["adjudicated"]) for row in rows),
        "pass_exact_agreement": sum(
            row["pass_a_task"] == row["pass_b_task"]
            and row["pass_a_constraint"] == row["pass_b_constraint"]
            and row["pass_a_recall"] == row["pass_b_recall"]
            for row in rows
        ) / len(rows),
        "prompt_source_sha256": hashlib.sha256(inspect.getsource(build_prompt).encode()).hexdigest(),
        "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "task_agreement": _dimension_agreement(rows, "registered_task", "task"),
        "constraint_agreement": _dimension_agreement(rows, "registered_constraint", "constraint"),
        "recall_agreement": _dimension_agreement(rows, "registered_recall", "recall"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUTPUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
