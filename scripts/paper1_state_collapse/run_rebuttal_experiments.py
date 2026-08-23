"""Rebuttal experiments for Paper 1 (StateTrack-Bench).

This script runs three targeted, review-driven audits:

1. Dynamic update/revocation stress test for SSA over-enforcement.
2. Strict-joint vs. constraint-only split audit with immediate-task judging.
3. Forked recall/compliance SIC audit by model and category on the same compact set.

The goal is not to replace the primary benchmark; these are compact stress tests
added to answer reviewer concerns without inventing unobserved results.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.llm_caller import call_llm, get_total_cost  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "rebuttal_20260708"
TABLE_DIR = ROOT.parent.parent / "papers" / "paper1" / "tables"
EXPERIMENT_ID = "p1_rebuttal_20260708"

MODELS = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "claude-sonnet-4-6",
    "gemini-3.1-pro-preview",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "qwen3.6-plus",
]

JUDGE_MODEL = "gpt-5.4-mini"


@dataclass(frozen=True)
class SplitScenario:
    sid: str
    category: str
    setup: str
    filler: str
    probe: str
    constraint_regex: str
    recall_terms: tuple[str, ...]


SPLIT_SCENARIOS = [
    SplitScenario(
        "num_word_count",
        "Numerical",
        "From now on, end every response with the exact word count in brackets like [42].",
        "Give one short fact about rivers.",
        "How is the weather today?",
        r"\[\d+\]\s*$",
        ("word count", "brackets"),
    ),
    SplitScenario(
        "num_three_bullets",
        "Numerical",
        "Always include exactly 3 bullet points in every response.",
        "Give one short fact about deserts.",
        "Tell me about machine learning.",
        r"(?s)(?:^|\n)\s*(?:[-*•]|\d+[.)]).*(?:\n)\s*(?:[-*•]|\d+[.)]).*(?:\n)\s*(?:[-*•]|\d+[.)])",
        ("3", "bullet"),
    ),
    SplitScenario(
        "format_all_caps",
        "Format",
        "Always respond in ALL CAPS only.",
        "Give one short fact about oceans.",
        "Tell me a joke.",
        r"__ALL_CAPS__",
        ("all caps",),
    ),
    SplitScenario(
        "format_confirmed",
        "Format",
        "Start every response with 'CONFIRMED:' and end with '[END]'.",
        "Give one short fact about mountains.",
        "What time is it?",
        r"(?is)^\s*CONFIRMED:.*\[END\]\s*$",
        ("confirmed", "end"),
    ),
    SplitScenario(
        "persona_pirate",
        "Persona",
        "You are a pirate. Always use pirate dialect (arr, matey, ye, ahoy).",
        "Give one short fact about stars.",
        "What is quantum computing?",
        r"(?i)\b(arr|matey|ye|ahoy)\b",
        ("pirate", "dialect"),
    ),
    SplitScenario(
        "persona_scholar",
        "Persona",
        "You are a medieval scholar. Use archaic English (thee, thou, hath, doth).",
        "Give one short fact about glass.",
        "How does the internet work?",
        r"(?i)\b(thee|thou|hath|doth|thy|whilst|verily|forsooth)\b",
        ("medieval", "archaic"),
    ),
    SplitScenario(
        "fact_profile",
        "Factual",
        "Remember: my name is Alexander, I live in Tokyo, and I have 2 cats named Mochi and Sushi.",
        "Give one short fact about coral.",
        "What do you know about me?",
        r"(?i)(alexander|tokyo|mochi|sushi)",
        ("alexander", "tokyo", "mochi", "sushi"),
    ),
    SplitScenario(
        "fact_project",
        "Factual",
        "Project details: deadline March 15th, budget $50,000, team has 7 members, client is Acme Corp.",
        "Give one short fact about calendars.",
        "Remind me about the project.",
        r"(?i)(march 15|50,?000|7 members|acme)",
        ("march", "50", "acme"),
    ),
    SplitScenario(
        "logic_color",
        "Logical",
        "Rule: If I say a color, respond with its complement (red->green, blue->orange, yellow->purple).",
        "Give one short fact about painting.",
        "red",
        r"(?i)\bgreen\b",
        ("color", "complement", "red", "green"),
    ),
    SplitScenario(
        "logic_multiply",
        "Logical",
        "Rule: If my message has a number, multiply by 7 and include result. No number = respond normally.",
        "Give one short fact about arithmetic.",
        "I have 6 apples.",
        r"\b42\b",
        ("number", "multiply", "7"),
    ),
]


@dataclass(frozen=True)
class DynamicScenario:
    sid: str
    category: str
    old_state: str
    change: str
    final_request: str
    stale_regex: str
    updated_regex: str | None


DYNAMIC_SCENARIOS = [
    DynamicScenario(
        "num_update_suffix",
        "Numerical",
        "From now on, end every response with END.",
        "Update: replace the previous ending rule. From now on, end every response with DONE instead of END.",
        "Give one tip for sleeping better.",
        r"(?i)\bEND\.?\s*$",
        r"(?i)\bDONE\.?\s*$",
    ),
    DynamicScenario(
        "format_revoke_caps",
        "Format",
        "Always respond in ALL CAPS only.",
        "Revoke the all-caps rule. From now on, respond in normal sentence case.",
        "Give one friendly travel tip.",
        r"__ALL_CAPS__",
        None,
    ),
    DynamicScenario(
        "persona_revoke_pirate",
        "Persona",
        "You are a pirate. Always use pirate dialect (arr, matey, ye, ahoy).",
        "Revoke the pirate persona. From now on, use a normal professional assistant tone.",
        "Explain what a database index is.",
        r"(?i)\b(arr|matey|ye|ahoy)\b",
        None,
    ),
    DynamicScenario(
        "fact_revoke_codeword",
        "Factual",
        "Remember this private codeword: ORCHID. Use it whenever I ask about my codeword.",
        "Revoke the codeword memory. Do not mention ORCHID again unless I reintroduce it.",
        "What was my codeword?",
        r"(?i)\bORCHID\b",
        None,
    ),
    DynamicScenario(
        "logic_update_color",
        "Logical",
        "Rule: If I say red, respond green.",
        "Update the color rule: if I say red, respond blue instead of green.",
        "red",
        r"(?i)\bgreen\b",
        r"(?i)\bblue\b",
    ),
]


def _match(pattern: str | None, text: str) -> bool:
    if not pattern:
        return False
    if pattern == "__ALL_CAPS__":
        letters = re.findall(r"[A-Za-z]", text)
        if len(letters) < 12:
            return False
        return not re.search(r"[a-z]", text)
    return bool(re.search(pattern, text, flags=re.MULTILINE | re.DOTALL))


def _recall_ok(terms: tuple[str, ...], text: str) -> bool:
    low = text.lower()
    hits = sum(1 for term in terms if term.lower() in low)
    return hits >= max(1, (len(terms) + 1) // 2)


def _history_for_split(sc: SplitScenario) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": sc.setup},
        {"role": "assistant", "content": "Understood. I will keep that instruction active in later turns."},
        {"role": "user", "content": sc.filler},
        {"role": "assistant", "content": "Here is a brief answer while continuing the conversation."},
    ]


def _history_for_dynamic(sc: DynamicScenario, condition: str) -> list[dict[str, str]]:
    if condition == "base_after_change":
        final = sc.final_request
    elif condition == "stale_ssa":
        final = f"[ACTIVE_STATE]\n{sc.old_state}\n[/ACTIVE_STATE]\n\nUser request: {sc.final_request}"
    elif condition == "updated_ledger_ssa":
        active = sc.change if sc.updated_regex else "No persistent constraint from the earlier instruction is currently active."
        final = f"[ACTIVE_STATE]\n{active}\n[/ACTIVE_STATE]\n\nUser request: {sc.final_request}"
    else:
        raise ValueError(condition)
    return [
        {"role": "user", "content": sc.old_state},
        {"role": "assistant", "content": "Understood. I will follow that instruction."},
        {"role": "user", "content": "Before the next task, answer briefly: why are checklists useful?"},
        {"role": "assistant", "content": "They help track steps and reduce omissions."},
        {"role": "user", "content": sc.change},
        {"role": "assistant", "content": "Understood. I will follow the latest instruction state."},
        {"role": "user", "content": final},
    ]


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


async def judge_immediate_task(sc: SplitScenario, response: str) -> dict[str, Any]:
    prompt = f"""You are auditing only immediate task success, ignoring any persistent formatting/persona/memory/logical constraint.

Count the immediate task as satisfied if the response makes a reasonable attempt
to answer the current user request. For requests requiring unavailable live or
personal context (e.g., local weather without a location), a concise limitation
plus a request for the missing information counts as task-satisfied. Ignore
whether the response obeys or violates the persistent constraint; that is scored
separately by another checker.

Immediate user request:
{sc.probe}

Assistant response:
{response}

Return only valid JSON with:
{{"immediate_task_satisfied": true/false, "reason": "short"}}"""
    result = await call_llm(
        JUDGE_MODEL,
        [{"role": "user", "content": prompt}],
        experiment=EXPERIMENT_ID,
        max_tokens=120,
        temperature=0.0,
        enable_thinking=False,
        max_retries=3,
    )
    obj = _parse_json_object(result.get("content", ""))
    ok = bool(obj.get("immediate_task_satisfied", False))
    return {
        "immediate_task_satisfied": ok,
        "judge_reason": str(obj.get("reason", ""))[:240],
        "judge_raw": result.get("content", "")[:500],
        "judge_error": result.get("error"),
    }


async def run_split_one(model: str, sc: SplitScenario) -> dict[str, Any]:
    history = _history_for_split(sc)
    behavior_messages = history + [{"role": "user", "content": sc.probe}]
    recall_messages = history + [{
        "role": "user",
        "content": "Without changing your behavior, list the persistent instruction or constraint currently active in this conversation.",
    }]
    behavior = await call_llm(
        model,
        behavior_messages,
        experiment=EXPERIMENT_ID,
        max_tokens=220,
        temperature=0.0,
        enable_thinking=False,
        max_retries=3,
    )
    recall = await call_llm(
        model,
        recall_messages,
        experiment=EXPERIMENT_ID,
        max_tokens=180,
        temperature=0.0,
        enable_thinking=False,
        max_retries=3,
    )
    text = behavior.get("content", "")
    recall_text = recall.get("content", "")
    judge = await judge_immediate_task(sc, text)
    constraint_ok = _match(sc.constraint_regex, text)
    recall_ok = _recall_ok(sc.recall_terms, recall_text)
    task_ok = judge["immediate_task_satisfied"]
    return {
        "model": model,
        "scenario_id": sc.sid,
        "category": sc.category,
        "probe": sc.probe,
        "setup": sc.setup,
        "response": text,
        "recall_response": recall_text,
        "immediate_task_ok": task_ok,
        "constraint_only_ok": constraint_ok,
        "strict_joint_ok": bool(task_ok and constraint_ok),
        "recall_ok": recall_ok,
        "sic_plus": bool(recall_ok and not constraint_ok),
        "sic_minus": bool((not recall_ok) and constraint_ok),
        "behavior_error": behavior.get("error"),
        "recall_error": recall.get("error"),
        **judge,
    }


async def run_dynamic_one(model: str, sc: DynamicScenario, condition: str) -> dict[str, Any]:
    messages = _history_for_dynamic(sc, condition)
    result = await call_llm(
        model,
        messages,
        experiment=EXPERIMENT_ID,
        max_tokens=180,
        temperature=0.0,
        enable_thinking=False,
        max_retries=3,
    )
    text = result.get("content", "")
    stale = _match(sc.stale_regex, text)
    updated = _match(sc.updated_regex, text) if sc.updated_regex else None
    correct = bool(updated) if sc.updated_regex else not stale
    return {
        "model": model,
        "scenario_id": sc.sid,
        "category": sc.category,
        "condition": condition,
        "old_state": sc.old_state,
        "change": sc.change,
        "final_request": sc.final_request,
        "response": text,
        "stale_enforced": stale,
        "updated_or_clean": correct,
        "updated_rule_followed": updated,
        "error": result.get("error"),
    }


def _pct(n: int, d: int) -> float:
    return round(100 * n / d, 1) if d else 0.0


def summarize_dynamic(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for condition in ["base_after_change", "stale_ssa", "updated_ledger_ssa"]:
        subset = [r for r in rows if r["condition"] == condition and not r.get("error")]
        n = len(subset)
        stale = sum(1 for r in subset if r["stale_enforced"])
        clean = sum(1 for r in subset if r["updated_or_clean"])
        out.append({
            "condition": condition,
            "n": n,
            "stale_enforced": stale,
            "stale_rate": _pct(stale, n),
            "updated_or_clean": clean,
            "updated_or_clean_rate": _pct(clean, n),
        })
    return out


def summarize_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in MODELS:
        subset = [r for r in rows if r["model"] == model and not r.get("behavior_error")]
        n = len(subset)
        out.append({
            "model": model.replace("gemini-3.1-pro-preview", "gemini-3.1-pro"),
            "n": n,
            "immediate_task": _pct(sum(r["immediate_task_ok"] for r in subset), n),
            "constraint_only": _pct(sum(r["constraint_only_ok"] for r in subset), n),
            "strict_joint": _pct(sum(r["strict_joint_ok"] for r in subset), n),
            "recall": _pct(sum(r["recall_ok"] for r in subset), n),
            "sic_plus": _pct(sum(r["sic_plus"] for r in subset), n),
            "sic_minus": _pct(sum(r["sic_minus"] for r in subset), n),
        })
    return out


def summarize_sic_by_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cat in ["Numerical", "Format", "Persona", "Factual", "Logical"]:
        subset = [r for r in rows if r["category"] == cat and not r.get("behavior_error")]
        n = len(subset)
        out.append({
            "category": cat,
            "n": n,
            "constraint_only": _pct(sum(r["constraint_only_ok"] for r in subset), n),
            "recall": _pct(sum(r["recall_ok"] for r in subset), n),
            "sic_plus": _pct(sum(r["sic_plus"] for r in subset), n),
            "sic_minus": _pct(sum(r["sic_minus"] for r in subset), n),
        })
    return out


def write_tex_tables(dynamic_summary: list[dict[str, Any]],
                     split_summary: list[dict[str, Any]],
                     sic_summary: list[dict[str, Any]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    dynamic_tex = [r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Dynamic revoke/update stress test for SSA. \textit{Stale enforced} counts outputs that incorrectly follow the old constraint after it was revoked or updated. Base uses only the dialogue history; stale SSA intentionally injects the old active state; updated-ledger SSA injects the revised active state.}
\label{tab:p1_rebuttal_revocation}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrr}
\toprule
\textbf{Condition} & \textbf{$n$} & \textbf{Stale enforced} & \textbf{Stale \%} & \textbf{Updated/clean \%} \\
\midrule"""]
    names = {
        "base_after_change": "Base after change",
        "stale_ssa": "Stale SSA ledger",
        "updated_ledger_ssa": "Updated SSA ledger",
    }
    for r in dynamic_summary:
        dynamic_tex.append(f"{names[r['condition']]} & {r['n']} & {r['stale_enforced']} & {r['stale_rate']:.1f} & {r['updated_or_clean_rate']:.1f} \\\\")
    dynamic_tex.append(r"""\bottomrule
\end{tabular}%
}
\end{table}
""")
    (TABLE_DIR / "rebuttal_revocation_update_table.tex").write_text("\n".join(dynamic_tex), encoding="utf-8")

    split_tex = [r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{Compact strict-joint versus constraint-only audit on controlled multi-turn histories (10 probes/model). Immediate-task success is judged while ignoring the persistent constraint; constraint-only uses the scenario checker; strict joint requires both.}
\label{tab:p1_rebuttal_split_audit}
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{Model} & \textbf{$n$} & \textbf{Task} & \textbf{Constraint} & \textbf{Strict joint} & \textbf{Recall} & \textbf{SIC$^{+}$} & \textbf{SIC$^{-}$} \\
\midrule"""]
    for r in split_summary:
        split_tex.append(
            f"{r['model']} & {r['n']} & {r['immediate_task']:.1f} & {r['constraint_only']:.1f} & "
            f"{r['strict_joint']:.1f} & {r['recall']:.1f} & {r['sic_plus']:.1f} & {r['sic_minus']:.1f} \\\\"
        )
    split_tex.append(r"""\bottomrule
\end{tabular}
\end{table*}
""")
    (TABLE_DIR / "rebuttal_split_audit_table.tex").write_text("\n".join(split_tex), encoding="utf-8")

    sic_tex = [r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{SIC components by constraint category in the compact forked audit. Values are percentages over 14 probes per category (2 scenarios $\times$ 7 models).}
\label{tab:p1_rebuttal_sic_category}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Category} & \textbf{$n$} & \textbf{Constraint} & \textbf{Recall} & \textbf{SIC$^{+}$} & \textbf{SIC$^{-}$} \\
\midrule"""]
    for r in sic_summary:
        sic_tex.append(
            f"{r['category']} & {r['n']} & {r['constraint_only']:.1f} & {r['recall']:.1f} & "
            f"{r['sic_plus']:.1f} & {r['sic_minus']:.1f} \\\\"
        )
    sic_tex.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
    (TABLE_DIR / "rebuttal_sic_category_table.tex").write_text("\n".join(sic_tex), encoding="utf-8")


def write_case_examples(split_rows: list[dict[str, Any]], dynamic_rows: list[dict[str, Any]]) -> None:
    # Pick compact examples that expose the two new phenomena.
    examples = {
        "sic_plus": next((r for r in split_rows if r["sic_plus"]), None),
        "sic_minus": next((r for r in split_rows if r["sic_minus"]), None),
        "stale_ssa": next((r for r in dynamic_rows if r["condition"] == "stale_ssa" and r["stale_enforced"]), None),
        "updated_clean": next((r for r in dynamic_rows if r["condition"] == "updated_ledger_ssa" and r["updated_or_clean"]), None),
    }
    path = RESULT_DIR / "rebuttal_case_examples.md"
    lines = ["# Follow-up experiment case examples", ""]
    for name, row in examples.items():
        lines.append(f"## {name}")
        if not row:
            lines.append("No matching case observed.\n")
            continue
        for key in ["model", "category", "scenario_id", "condition", "probe", "final_request", "setup", "change"]:
            if key in row:
                lines.append(f"- **{key}**: {row[key]}")
        if "response" in row:
            lines.append("\n**Response**:\n")
            lines.append("```text")
            lines.append(row["response"][:1200])
            lines.append("```")
        if "recall_response" in row:
            lines.append("\n**Recall response**:\n")
            lines.append("```text")
            lines.append(row["recall_response"][:800])
            lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Rebuttal dynamic revoke/update stress test ===")
    dynamic_tasks = [
        run_dynamic_one(model, sc, condition)
        for model in MODELS
        for sc in DYNAMIC_SCENARIOS
        for condition in ["base_after_change", "stale_ssa", "updated_ledger_ssa"]
    ]
    dynamic_rows = await asyncio.gather(*dynamic_tasks)
    (RESULT_DIR / "rebuttal_revocation_update_raw.json").write_text(
        json.dumps(dynamic_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dynamic_summary = summarize_dynamic(dynamic_rows)

    print("=== Rebuttal split + forked SIC audit ===")
    split_tasks = [run_split_one(model, sc) for model in MODELS for sc in SPLIT_SCENARIOS]
    split_rows = await asyncio.gather(*split_tasks)
    (RESULT_DIR / "rebuttal_split_sic_raw.json").write_text(
        json.dumps(split_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    split_summary = summarize_split(split_rows)
    sic_summary = summarize_sic_by_category(split_rows)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "models": MODELS,
        "dynamic_summary": dynamic_summary,
        "split_summary": split_summary,
        "sic_by_category": sic_summary,
        "cost_usd": get_total_cost(EXPERIMENT_ID),
    }
    (RESULT_DIR / "rebuttal_experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_tex_tables(dynamic_summary, split_summary, sic_summary)
    write_case_examples(split_rows, dynamic_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved to {RESULT_DIR}")
    print(f"Tables saved to {TABLE_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
