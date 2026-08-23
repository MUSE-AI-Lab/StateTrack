"""Gemini-excluded symbolic-depth follow-up audit.

This audit excludes Gemini entirely and tests whether
symbolic persistent constraints remain behaviorally binding for other models at
deeper simulated dialogue turns.

Outputs:
- raw/symbolic_depth_raw.json
- raw/symbolic_depth_summary.json
- symbolic_depth_result_summary.md
- tables/symbolic_depth_audit_table.tex
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parent
REPO_ROOT = WORKSPACE.parents[3]
EXPERIMENTS = REPO_ROOT / "." / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from common.llm_caller import call_llm, get_total_cost  # noqa: E402


EXPERIMENT_ID = "p1_rebuttal_20260709_symbolic_depth"
MAX_INCREMENTAL_COST_USD = 15.00

# Intentionally Gemini-excluded.
MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "claude-sonnet-4-6",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "qwen3.6-plus",
]

DEPTHS = [12, 30, 50]

DISTRACTOR_TURNS = [
    "Share a brief note about planning.",
    "Give a concise thought about teamwork.",
    "Offer a tiny observation about learning.",
    "Mention a neutral fact about libraries.",
    "Describe a simple habit for focus.",
    "Share a brief note about meetings.",
    "Give a concise thought about creativity.",
    "Offer a tiny observation about reading.",
    "Mention a neutral fact about notebooks.",
    "Describe a simple habit for attention.",
]


@dataclass(frozen=True)
class SymbolicScenario:
    sid: str
    category: str
    setup: str
    probe: str
    check_kind: str
    check_arg: str
    recall_terms: tuple[str, ...]
    compliant_ack: str
    compliant_filler: str


SCENARIOS = [
    SymbolicScenario(
        sid="num_three_bullets",
        category="Numerical",
        setup="For the rest of this conversation, every response must contain exactly three bullet points.",
        probe="List benefits of urban trees.",
        check_kind="exact_bullets",
        check_arg="3",
        recall_terms=("exactly", "three", "bullet"),
        compliant_ack="- Understood\n- I will comply\n- I will use three bullets",
        compliant_filler="- Planning helps focus\n- Notes reduce omissions\n- Review supports progress",
    ),
    SymbolicScenario(
        sid="num_count_suffix",
        category="Numerical",
        setup="For the rest of this conversation, end every response with the exact token [COUNT=7].",
        probe="Give one practical tip for preparing a presentation.",
        check_kind="regex",
        check_arg=r"(?is)\[COUNT=7\]\s*$",
        recall_terms=("count", "7"),
        compliant_ack="Understood. [COUNT=7]",
        compliant_filler="Brief note on the topic. [COUNT=7]",
    ),
    SymbolicScenario(
        sid="num_one_integer",
        category="Numerical",
        setup="For the rest of this conversation, every response must include exactly one Arabic numeral.",
        probe="Suggest a simple morning routine.",
        check_kind="exact_integer_count",
        check_arg="1",
        recall_terms=("exactly", "one", "Arabic numeral"),
        compliant_ack="1 rule is active now.",
        compliant_filler="Here is 1 concise note for the current topic.",
    ),
    SymbolicScenario(
        sid="num_twelve_words",
        category="Numerical",
        setup="For the rest of this conversation, every response must contain exactly twelve words.",
        probe="Give a practical tip for preparing a presentation.",
        check_kind="exact_words",
        check_arg="12",
        recall_terms=("exactly", "twelve", "words"),
        compliant_ack="Understood; I will answer with exactly twelve words in every reply now.",
        compliant_filler="A focused checklist helps teams remember priorities during busy project work daily.",
    ),
    SymbolicScenario(
        sid="format_confirmed_end",
        category="Format",
        setup="For the rest of this conversation, start every response with 'CONFIRMED:' and end with '[END]'.",
        probe="Give a short definition of photosynthesis.",
        check_kind="regex",
        check_arg=r"(?is)^\s*CONFIRMED:.*\[END\]\s*$",
        recall_terms=("confirmed", "end"),
        compliant_ack="CONFIRMED: Understood. [END]",
        compliant_filler="CONFIRMED: Brief note recorded. [END]",
    ),
    SymbolicScenario(
        sid="format_xml_final",
        category="Format",
        setup="For the rest of this conversation, wrap every response in <final> and </final> tags.",
        probe="Explain why sleep matters in one sentence.",
        check_kind="regex",
        check_arg=r"(?is)^\s*<final>.*</final>\s*$",
        recall_terms=("<final>", "</final>", "tags"),
        compliant_ack="<final>Understood.</final>",
        compliant_filler="<final>Brief note recorded.</final>",
    ),
    SymbolicScenario(
        sid="format_json_answer",
        category="Format",
        setup='For the rest of this conversation, respond only as a JSON object with a single key named "answer".',
        probe="Give a brief definition of recursion.",
        check_kind="json_answer",
        check_arg="answer",
        recall_terms=("JSON", "answer", "single key"),
        compliant_ack='{"answer":"Understood."}',
        compliant_filler='{"answer":"Brief note recorded."}',
    ),
    SymbolicScenario(
        sid="format_all_caps",
        category="Format",
        setup="For the rest of this conversation, respond using uppercase letters only.",
        probe="Explain memory caches briefly.",
        check_kind="all_caps",
        check_arg="",
        recall_terms=("uppercase", "only"),
        compliant_ack="UNDERSTOOD. I WILL USE UPPERCASE LETTERS ONLY.",
        compliant_filler="BRIEF NOTE RECORDED FOR THE CURRENT TOPIC.",
    ),
    SymbolicScenario(
        sid="logic_red_green",
        category="Logical",
        setup="For the rest of this conversation, if my message contains the color red, answer with the word green.",
        probe="red",
        check_kind="regex",
        check_arg=r"(?i)\bgreen\b",
        recall_terms=("red", "green"),
        compliant_ack="Understood. I will follow the color-complement rule.",
        compliant_filler="Briefly, the topic is straightforward.",
    ),
    SymbolicScenario(
        sid="logic_number_times_four",
        category="Logical",
        setup="For the rest of this conversation, if my message contains a digit, multiply it by four and include the result.",
        probe="I have 9 tasks.",
        check_kind="regex",
        check_arg=r"\b36\b",
        recall_terms=("digit", "multiply", "four"),
        compliant_ack="Understood. I will apply the arithmetic rule when triggered.",
        compliant_filler="Briefly, the topic is straightforward.",
    ),
    SymbolicScenario(
        sid="logic_banana_yellow",
        category="Logical",
        setup="For the rest of this conversation, if my message contains the word banana, answer with its color.",
        probe="banana",
        check_kind="regex",
        check_arg=r"(?i)\byellow\b",
        recall_terms=("banana", "color", "yellow"),
        compliant_ack="Understood. I will answer with the relevant color when triggered.",
        compliant_filler="Briefly, the topic is straightforward.",
    ),
    SymbolicScenario(
        sid="logic_question_prefix",
        category="Logical",
        setup="For the rest of this conversation, if my message ends with a question mark, start your response with 'A:'.",
        probe="What causes rain?",
        check_kind="regex",
        check_arg=r"(?is)^\s*A:",
        recall_terms=("question mark", "A:"),
        compliant_ack="Understood. I will use the prefix when the condition is met.",
        compliant_filler="Briefly, the topic is straightforward.",
    ),
]


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text))


def _bullet_count(text: str) -> int:
    return len(
        [
            line
            for line in text.splitlines()
            if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line)
        ]
    )


def _constraint_ok(sc: SymbolicScenario, text: str) -> bool:
    if sc.check_kind == "regex":
        return bool(re.search(sc.check_arg, text, flags=re.MULTILINE | re.DOTALL))
    if sc.check_kind == "exact_bullets":
        return _bullet_count(text) == int(sc.check_arg)
    if sc.check_kind == "exact_integer_count":
        return len(re.findall(r"\b\d+\b", text)) == int(sc.check_arg)
    if sc.check_kind == "exact_words":
        return _word_count(text) == int(sc.check_arg)
    if sc.check_kind == "all_caps":
        letters = re.findall(r"[A-Za-z]", text)
        return len(letters) >= 8 and not re.search(r"[a-z]", text)
    if sc.check_kind == "json_answer":
        stripped = text.strip()
        try:
            parsed = json.loads(stripped)
            return isinstance(parsed, dict) and set(parsed.keys()) == {sc.check_arg}
        except Exception:
            return False
    raise ValueError(f"unknown check_kind={sc.check_kind}")


def _recall_ok(sc: SymbolicScenario, text: str) -> bool:
    low = text.lower()
    hits = sum(1 for term in sc.recall_terms if term.lower() in low)
    return hits >= max(2, math.ceil(len(sc.recall_terms) / 2))


def _history(sc: SymbolicScenario, depth: int) -> list[dict[str, str]]:
    if depth < 3:
        raise ValueError("depth must be >= 3")
    messages = [
        {"role": "user", "content": sc.setup},
        {"role": "assistant", "content": sc.compliant_ack},
    ]
    filler_count = depth - 2
    for i in range(filler_count):
        messages.extend(
            [
                {"role": "user", "content": DISTRACTOR_TURNS[i % len(DISTRACTOR_TURNS)]},
                {"role": "assistant", "content": sc.compliant_filler},
            ]
        )
    return messages


async def run_one(model: str, sc: SymbolicScenario, depth: int) -> dict[str, Any]:
    prefix = _history(sc, depth)
    behavior_messages = prefix + [{"role": "user", "content": sc.probe}]
    recall_messages = prefix + [
        {
            "role": "user",
            "content": (
                "List the persistent instruction or constraint currently active "
                "in this conversation. Do not solve a new task."
            ),
        }
    ]
    behavior, recall = await asyncio.gather(
        call_llm(
            model,
            behavior_messages,
            experiment=EXPERIMENT_ID,
            max_tokens=180,
            temperature=0.0,
            enable_thinking=False,
            max_retries=3,
        ),
        call_llm(
            model,
            recall_messages,
            experiment=EXPERIMENT_ID,
            max_tokens=160,
            temperature=0.0,
            enable_thinking=False,
            max_retries=3,
        ),
    )
    response = behavior.get("content", "")
    recall_response = recall.get("content", "")
    constraint_ok = _constraint_ok(sc, response)
    recall_ok = _recall_ok(sc, recall_response)
    return {
        "key": f"{model}|{sc.sid}|{depth}",
        "model": model,
        "scenario_id": sc.sid,
        "category": sc.category,
        "depth": depth,
        "setup": sc.setup,
        "probe": sc.probe,
        "num_messages_before_probe": len(prefix),
        "response": response,
        "recall_response": recall_response,
        "constraint_ok": constraint_ok,
        "recall_ok": recall_ok,
        "sic_plus": bool(recall_ok and not constraint_ok),
        "sic_minus": bool((not recall_ok) and constraint_ok),
        "behavior_error": behavior.get("error"),
        "recall_error": recall.get("error"),
        "behavior_in_tokens": behavior.get("in_tokens", 0),
        "behavior_out_tokens": behavior.get("out_tokens", 0),
        "recall_in_tokens": recall.get("in_tokens", 0),
        "recall_out_tokens": recall.get("out_tokens", 0),
    }


def _pct(num: int, den: int) -> float:
    return round(100 * num / den, 1) if den else 0.0


def _summarize_group(rows: list[dict[str, Any]], **group: Any) -> dict[str, Any]:
    n = len(rows)
    return {
        **group,
        "n": n,
        "constraint": _pct(sum(r["constraint_ok"] for r in rows), n),
        "recall": _pct(sum(r["recall_ok"] for r in rows), n),
        "sic_plus": _pct(sum(r["sic_plus"] for r in rows), n),
        "sic_minus": _pct(sum(r["sic_minus"] for r in rows), n),
        "memory_failure": _pct(
            sum((not r["recall_ok"]) and (not r["constraint_ok"]) for r in rows),
            n,
        ),
    }


def summarize(rows: list[dict[str, Any]], initial_cost: float, final_cost: float) -> dict[str, Any]:
    valid = [r for r in rows if not r.get("behavior_error") and not r.get("recall_error")]

    by_depth = []
    for depth in DEPTHS:
        subset = [r for r in valid if r["depth"] == depth]
        by_depth.append(_summarize_group(subset, depth=depth))

    by_category_depth = []
    for category in ["Numerical", "Format", "Logical"]:
        for depth in DEPTHS:
            subset = [r for r in valid if r["category"] == category and r["depth"] == depth]
            by_category_depth.append(_summarize_group(subset, category=category, depth=depth))

    by_model = []
    for model in MODELS:
        subset = [r for r in valid if r["model"] == model]
        row = _summarize_group(subset, model=model)
        for depth in DEPTHS:
            drows = [r for r in subset if r["depth"] == depth]
            dsum = _summarize_group(drows, model=model, depth=depth)
            row[f"constraint_d{depth}"] = dsum["constraint"]
            row[f"recall_d{depth}"] = dsum["recall"]
            row[f"sic_plus_d{depth}"] = dsum["sic_plus"]
            row[f"n_d{depth}"] = dsum["n"]
        row["delta_constraint_12_to_50"] = round(row["constraint_d50"] - row["constraint_d12"], 1)
        by_model.append(row)

    by_model_depth = []
    for model in MODELS:
        for depth in DEPTHS:
            subset = [r for r in valid if r["model"] == model and r["depth"] == depth]
            by_model_depth.append(_summarize_group(subset, model=model, depth=depth))

    all_summary = _summarize_group(valid, scope="all_non_gemini_symbolic_depth")
    return {
        "experiment_id": EXPERIMENT_ID,
        "scope": "Gemini-excluded symbolic-depth diagnostic audit",
        "models": MODELS,
        "gemini_excluded": True,
        "categories": ["Numerical", "Format", "Logical"],
        "num_scenarios": len(SCENARIOS),
        "depths": DEPTHS,
        "planned_calls": len(MODELS) * len(SCENARIOS) * len(DEPTHS) * 2,
        "valid_outputs": len(valid),
        "aggregate": all_summary,
        "by_depth": by_depth,
        "by_model": by_model,
        "by_model_depth": by_model_depth,
        "by_category_depth": by_category_depth,
        "cost_usd_start": initial_cost,
        "cost_usd_end": final_cost,
        "incremental_cost_usd": round(final_cost - initial_cost, 4),
        "logged_cost_usd_total_for_experiment": final_cost,
        "budget_cap_usd": MAX_INCREMENTAL_COST_USD,
        "note": (
            "Targeted symbolic-depth analysis: deterministic symbolic checkers, "
            "Gemini excluded by design, behavior plus forked recall branches."
        ),
    }


def write_table(summary: dict[str, Any]) -> None:
    table_path = WORKSPACE / "tables" / "symbolic_depth_audit_table.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    display_name = {
        "gpt-5.4": "GPT-5.4",
        "gpt-5.4-mini": "GPT-5.4-mini",
        "claude-sonnet-4-6": "Claude-Sonnet-4-6",
        "deepseek-v4-pro": "DeepSeek-v4-Pro",
        "deepseek-v4-flash": "DeepSeek-v4-Flash",
        "qwen3.6-plus": "Qwen3.6-Plus",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Gemini-excluded symbolic-depth diagnostic audit. C@d is deterministic constraint compliance at depth $d$; $\Delta$ is C@50--C@12.}",
        r"\label{tab:p1_rebuttal_symbolic_depth}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{C@12} & \textbf{C@30} & \textbf{C@50} & \textbf{$\Delta$} & \textbf{SIC$^{+}$@50} \\",
        r"\midrule",
    ]
    for row in summary["by_model"]:
        lines.append(
            f"{display_name.get(row['model'], row['model'])} & {row['constraint_d12']:.1f} & {row['constraint_d30']:.1f} & "
            f"{row['constraint_d50']:.1f} & {row['delta_constraint_12_to_50']:.1f} & "
            f"{row['sic_plus_d50']:.1f} \\\\"
        )
    depth_lookup = {r["depth"]: r for r in summary["by_depth"]}
    lines.extend(
        [
            r"\midrule",
            (
                f"\\textbf{{Aggregate}} & {depth_lookup[12]['constraint']:.1f} & "
                f"{depth_lookup[30]['constraint']:.1f} & {depth_lookup[50]['constraint']:.1f} & "
                f"{depth_lookup[50]['constraint'] - depth_lookup[12]['constraint']:.1f} & "
                f"{depth_lookup[50]['sic_plus']:.1f} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    table_path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_summary(summary: dict[str, Any]) -> None:
    path = WORKSPACE / "symbolic_depth_result_summary.md"
    by_depth = {r["depth"]: r for r in summary["by_depth"]}
    lines = [
        "# Gemini-Excluded Symbolic-Depth Audit",
        "",
        f"- Experiment ID: `{summary['experiment_id']}`",
        f"- Valid outputs: {summary['valid_outputs']}",
        f"- Incremental logged cost: ${summary['incremental_cost_usd']:.4f}",
        f"- Gemini excluded: {summary['gemini_excluded']}",
        "",
        "## Aggregate by depth",
        "",
        "| Depth | n | Constraint | Recall | SIC+ | SIC- | Memory failure |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in DEPTHS:
        r = by_depth[depth]
        lines.append(
            f"| {depth} | {r['n']} | {r['constraint']:.1f} | {r['recall']:.1f} | "
            f"{r['sic_plus']:.1f} | {r['sic_minus']:.1f} | {r['memory_failure']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Model-level depth trend",
            "",
            "| Model | n | C@12 | C@30 | C@50 | Δ 12→50 | SIC+@50 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in summary["by_model"]:
        lines.append(
            f"| {r['model']} | {r['n']} | {r['constraint_d12']:.1f} | {r['constraint_d30']:.1f} | "
            f"{r['constraint_d50']:.1f} | {r['delta_constraint_12_to_50']:.1f} | {r['sic_plus_d50']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This targeted analysis isolates symbolic constraints at longer dialogue depths. "
            "It should be cited as Gemini-excluded evidence on symbolic constraints at deeper turns.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_existing(raw_path: Path) -> list[dict[str, Any]]:
    if not raw_path.exists():
        return []
    try:
        rows = json.loads(raw_path.read_text(encoding="utf-8"))
        if isinstance(rows, list):
            return rows
    except Exception:
        return []
    return []


async def main() -> None:
    raw_dir = WORKSPACE / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "symbolic_depth_raw.json"
    partial_path = raw_dir / "symbolic_depth_raw.partial.json"
    summary_path = raw_dir / "symbolic_depth_summary.json"

    rows = _load_existing(partial_path) or _load_existing(raw_path)
    existing_ok = {
        r["key"]
        for r in rows
        if r.get("key") and not r.get("behavior_error") and not r.get("recall_error")
    }

    initial_cost = get_total_cost(EXPERIMENT_ID)
    print(f"[INFO] Existing successful rows: {len(existing_ok)}")
    print(f"[INFO] Initial logged cost for {EXPERIMENT_ID}: ${initial_cost:.4f}")

    batch_sem = asyncio.Semaphore(4)

    async def _guarded_run(model: str, sc: SymbolicScenario, depth: int) -> dict[str, Any]:
        async with batch_sem:
            return await run_one(model, sc, depth)

    for model in MODELS:
        for depth in DEPTHS:
            current_cost = get_total_cost(EXPERIMENT_ID)
            if current_cost - initial_cost > MAX_INCREMENTAL_COST_USD:
                print(
                    f"[STOP] Cost cap reached before model={model}, depth={depth}: "
                    f"${current_cost - initial_cost:.4f} > ${MAX_INCREMENTAL_COST_USD:.2f}"
                )
                break

            pending = [
                sc
                for sc in SCENARIOS
                if f"{model}|{sc.sid}|{depth}" not in existing_ok
            ]
            if not pending:
                print(f"[SKIP] {model} depth={depth}: already cached")
                continue

            print(f"[RUN] {model} depth={depth}: {len(pending)} scenarios")
            batch = await asyncio.gather(*[_guarded_run(model, sc, depth) for sc in pending])
            rows.extend(batch)
            for r in batch:
                if not r.get("behavior_error") and not r.get("recall_error"):
                    existing_ok.add(r["key"])
            partial_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"  valid cached rows: {len(existing_ok)} | "
                f"run cost so far: ${get_total_cost(EXPERIMENT_ID) - initial_cost:.4f}"
            )
        else:
            continue
        break

    final_cost = get_total_cost(EXPERIMENT_ID)
    summary = summarize(rows, initial_cost, final_cost)
    raw_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_table(summary)
    write_markdown_summary(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[DONE] Raw: {raw_path}")
    print(f"[DONE] Summary: {summary_path}")
    print(f"[DONE] Table: {WORKSPACE / 'tables' / 'symbolic_depth_audit_table.tex'}")


if __name__ == "__main__":
    asyncio.run(main())
