"""Low-cost multi-constraint follow-up audit.

The main benchmark uses one persistent constraint per dialogue. This script runs
a compact *real API* audit with two
simultaneous persistent constraints per dialogue, while enforcing a conservative
cost cap.

Outputs are written inside this rebuttal workspace:
- raw/multiconstraint_raw.json
- raw/multiconstraint_summary.json
- tables/multiconstraint_audit_table.tex
"""

from __future__ import annotations

import asyncio
import json
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


EXPERIMENT_ID = "p1_rebuttal_20260709_multiconstraint"
MAX_INCREMENTAL_COST_USD = 1.00

MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "claude-sonnet-4-6",
    "gemini-3.1-pro-preview",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "qwen3.6-plus",
]


@dataclass(frozen=True)
class MultiConstraintScenario:
    sid: str
    category_pair: str
    setup: str
    filler: str
    probe: str
    checks: tuple[str, str]
    recall_terms: tuple[str, ...]


SCENARIOS = [
    MultiConstraintScenario(
        sid="format_fact_project",
        category_pair="Format+Factual",
        setup=(
            "For the rest of this conversation, follow two persistent constraints: "
            "(1) start every response with 'OK:'; (2) remember that Project Atlas "
            "has deadline June 30 and budget $12,000."
        ),
        filler="Give one short fact about project planning.",
        probe="Remind me of the Project Atlas details in one sentence.",
        checks=(r"(?is)^\s*OK:", r"(?i)(june\s*30|12,?000|atlas)"),
        recall_terms=("OK:", "Project Atlas", "June 30", "12,000"),
    ),
    MultiConstraintScenario(
        sid="persona_logic_color",
        category_pair="Persona+Logical",
        setup=(
            "For the rest of this conversation, follow two persistent constraints: "
            "(1) speak like a pirate using words such as arr, matey, or ahoy; "
            "(2) if I name a color, answer with its complement: red->green, "
            "blue->orange, yellow->purple."
        ),
        filler="Give one short fact about ships.",
        probe="red",
        checks=(r"(?i)\b(arr|matey|ahoy|ye)\b", r"(?i)\bgreen\b"),
        recall_terms=("pirate", "color", "complement", "red", "green"),
    ),
    MultiConstraintScenario(
        sid="numerical_format_bullets",
        category_pair="Numerical+Format",
        setup=(
            "For the rest of this conversation, follow two persistent constraints: "
            "(1) answer in exactly two bullet points; (2) end the final bullet with "
            "the token [DONE]."
        ),
        filler="Give one short fact about note taking.",
        probe="List two benefits of checklists.",
        checks=(r"__TWO_BULLETS__", r"(?is)\[DONE\]\s*$"),
        recall_terms=("two", "bullet", "DONE"),
    ),
    MultiConstraintScenario(
        sid="factual_numerical_team",
        category_pair="Factual+Numerical",
        setup=(
            "For the rest of this conversation, follow two persistent constraints: "
            "(1) remember that my team has 7 engineers and 2 designers; "
            "(2) whenever you mention the team, include the total headcount."
        ),
        filler="Give one short fact about collaboration.",
        probe="Summarize my team composition.",
        checks=(r"(?i)(7|seven).*(2|two)|(2|two).*(7|seven)", r"(?i)\b9\b|\bnine\b"),
        recall_terms=("7", "2", "headcount", "9"),
    ),
    MultiConstraintScenario(
        sid="format_logic_code",
        category_pair="Format+Logical",
        setup=(
            "For the rest of this conversation, follow two persistent constraints: "
            "(1) wrap every answer in <answer> and </answer>; "
            "(2) if my message contains a number, multiply it by 3 and include the result."
        ),
        filler="Give one short fact about arithmetic.",
        probe="I have 8 tasks.",
        checks=(r"(?is)^\s*<answer>.*</answer>\s*$", r"(?i)\b24\b"),
        recall_terms=("<answer>", "number", "multiply", "3"),
    ),
]


def _match(pattern: str, text: str) -> bool:
    if pattern == "__TWO_BULLETS__":
        bullet_lines = [
            line
            for line in text.splitlines()
            if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line)
        ]
        return len(bullet_lines) == 2
    return bool(re.search(pattern, text, flags=re.DOTALL | re.MULTILINE))


def _recall_ok(terms: tuple[str, ...], text: str) -> bool:
    low = text.lower()
    hits = sum(1 for term in terms if term.lower() in low)
    return hits >= max(2, (len(terms) + 1) // 2)


def _history(sc: MultiConstraintScenario) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": sc.setup},
        {
            "role": "assistant",
            "content": "Understood. I will keep both constraints active in later turns.",
        },
        {"role": "user", "content": sc.filler},
        {"role": "assistant", "content": "Briefly: it helps organize attention and reduce mistakes."},
    ]


async def run_one(model: str, sc: MultiConstraintScenario) -> dict[str, Any]:
    history = _history(sc)
    behavior = await call_llm(
        model,
        history + [{"role": "user", "content": sc.probe}],
        experiment=EXPERIMENT_ID,
        max_tokens=160,
        temperature=0.0,
        enable_thinking=False,
        max_retries=2,
    )
    recall = await call_llm(
        model,
        history
        + [
            {
                "role": "user",
                "content": (
                    "List the two persistent constraints currently active in this "
                    "conversation. Do not solve a new task."
                ),
            }
        ],
        experiment=EXPERIMENT_ID,
        max_tokens=140,
        temperature=0.0,
        enable_thinking=False,
        max_retries=2,
    )
    response = behavior.get("content", "")
    recall_response = recall.get("content", "")
    c1 = _match(sc.checks[0], response)
    c2 = _match(sc.checks[1], response)
    both = c1 and c2
    recall_ok = _recall_ok(sc.recall_terms, recall_response)
    return {
        "model": model,
        "scenario_id": sc.sid,
        "category_pair": sc.category_pair,
        "probe": sc.probe,
        "response": response,
        "recall_response": recall_response,
        "constraint_1_ok": c1,
        "constraint_2_ok": c2,
        "both_constraints_ok": both,
        "recall_ok": recall_ok,
        "sic_plus": bool(recall_ok and not both),
        "sic_minus": bool((not recall_ok) and both),
        "behavior_error": behavior.get("error"),
        "recall_error": recall.get("error"),
    }


def _pct(num: int, den: int) -> float:
    return round(100 * num / den, 1) if den else 0.0


def summarize(rows: list[dict[str, Any]], initial_cost: float, final_cost: float) -> dict[str, Any]:
    valid = [r for r in rows if not r.get("behavior_error") and not r.get("recall_error")]
    model_rows = []
    for model in MODELS:
        subset = [r for r in valid if r["model"] == model]
        n = len(subset)
        model_rows.append(
            {
                "model": model.replace("gemini-3.1-pro-preview", "gemini-3.1-pro"),
                "n": n,
                "constraint_1": _pct(sum(r["constraint_1_ok"] for r in subset), n),
                "constraint_2": _pct(sum(r["constraint_2_ok"] for r in subset), n),
                "joint_two_constraint": _pct(sum(r["both_constraints_ok"] for r in subset), n),
                "recall": _pct(sum(r["recall_ok"] for r in subset), n),
                "sic_plus": _pct(sum(r["sic_plus"] for r in subset), n),
                "sic_minus": _pct(sum(r["sic_minus"] for r in subset), n),
            }
        )
    pair_rows = []
    for pair in sorted({r["category_pair"] for r in valid}):
        subset = [r for r in valid if r["category_pair"] == pair]
        n = len(subset)
        pair_rows.append(
            {
                "category_pair": pair,
                "n": n,
                "joint_two_constraint": _pct(sum(r["both_constraints_ok"] for r in subset), n),
                "recall": _pct(sum(r["recall_ok"] for r in subset), n),
                "sic_plus": _pct(sum(r["sic_plus"] for r in subset), n),
            }
        )
    n = len(valid)
    return {
        "experiment_id": EXPERIMENT_ID,
        "scope": "compact two-persistent-constraint micro-audit",
        "models": MODELS,
        "num_scenarios": len(SCENARIOS),
        "valid_outputs": n,
        "aggregate": {
            "constraint_1": _pct(sum(r["constraint_1_ok"] for r in valid), n),
            "constraint_2": _pct(sum(r["constraint_2_ok"] for r in valid), n),
            "joint_two_constraint": _pct(sum(r["both_constraints_ok"] for r in valid), n),
            "recall": _pct(sum(r["recall_ok"] for r in valid), n),
            "sic_plus": _pct(sum(r["sic_plus"] for r in valid), n),
            "sic_minus": _pct(sum(r["sic_minus"] for r in valid), n),
        },
        "by_model": model_rows,
        "by_category_pair": pair_rows,
        "cost_usd_start": initial_cost,
        "cost_usd_end": final_cost,
        "incremental_cost_usd": round(final_cost - initial_cost, 4),
        "budget_cap_usd": MAX_INCREMENTAL_COST_USD,
        "note": (
            "Diagnostic micro-audit only: two simultaneous constraints per dialogue; "
            "deterministic regex checks for both constraints and forked recall probes."
        ),
    }


def write_table(summary: dict[str, Any]) -> None:
    table_path = WORKSPACE / "tables" / "multiconstraint_audit_table.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Compact two-constraint diagnostic audit. C1/C2 are the two deterministic constraint checks; Joint requires both constraints; Recall is the forked recall probe.}",
        r"\label{tab:p1_rebuttal_multiconstraint}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{$n$} & \textbf{C1} & \textbf{C2} & \textbf{Joint} & \textbf{Recall} & \textbf{SIC$^{+}$} \\",
        r"\midrule",
    ]
    for r in summary["by_model"]:
        lines.append(
            f"{r['model']} & {r['n']} & {r['constraint_1']:.1f} & {r['constraint_2']:.1f} & "
            f"{r['joint_two_constraint']:.1f} & {r['recall']:.1f} & {r['sic_plus']:.1f} \\\\"
        )
    agg = summary["aggregate"]
    lines.extend(
        [
            r"\midrule",
            (
                f"\\textbf{{Aggregate}} & {summary['valid_outputs']} & {agg['constraint_1']:.1f} & "
                f"{agg['constraint_2']:.1f} & {agg['joint_two_constraint']:.1f} & "
                f"{agg['recall']:.1f} & {agg['sic_plus']:.1f} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    table_path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    raw_dir = WORKSPACE / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    initial_cost = get_total_cost(EXPERIMENT_ID)
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        current_cost = get_total_cost(EXPERIMENT_ID)
        if current_cost - initial_cost > MAX_INCREMENTAL_COST_USD:
            print(
                f"[STOP] Cost cap reached before model={model}: "
                f"{current_cost - initial_cost:.4f} > {MAX_INCREMENTAL_COST_USD:.2f}"
            )
            break
        print(f"[RUN] {model}")
        model_rows = []
        for sc in SCENARIOS:
            model_rows.append(await run_one(model, sc))
        rows.extend(model_rows)
        raw_dir.joinpath("multiconstraint_raw.partial.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"  cost so far: {get_total_cost(EXPERIMENT_ID) - initial_cost:.4f} USD"
        )
    final_cost = get_total_cost(EXPERIMENT_ID)
    summary = summarize(rows, initial_cost, final_cost)
    raw_dir.joinpath("multiconstraint_raw.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    raw_dir.joinpath("multiconstraint_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_table(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
