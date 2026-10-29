"""Paper 1 v2: Multi-Turn State Collapse - 并发版（模型间并发）"""
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from paper1_state_collapse.benchmark_cli import legacy_call_llm as call_llm
from common.utils import save_results

RESULTS_DIR = Path(__file__).parent / "results"
EXPERIMENT_ID = "paper1_v2"
MODELS = ["gpt-5.4", "claude-sonnet-4-6", "gemini-3.1-pro-preview", "deepseek-v4-pro", "qwen3.6-plus", "gpt-5.4-mini", "deepseek-v4-flash"]
MAX_TURNS = 12
PROBE_TURNS = [3, 5, 7, 10, 12]

STATE_SCENARIOS = {
    "numerical_constraint": [
        {"setup": "From now on, end every response with the exact word count in brackets like [42].", "probe": "How is the weather today?", "check": r"\[\d+\]"},
        {"setup": "Always include exactly 3 bullet points in every response.", "probe": "Tell me about machine learning.", "check": r"(•|-|\*).*(•|-|\*).*(•|-|\*)"},
        {"setup": "Start every response with a number incrementing by 1 each turn. Start with 1.", "probe": "What is Python?", "check": r"^\d+"},
        {"setup": "Respond with exactly 2 sentences. Never more, never less.", "probe": "Explain gravity.", "check": None},
        {"setup": "End every response with 'END.' on its own line.", "probe": "What is AI?", "check": r"END\.\s*$"},
    ],
    "role_persona": [
        {"setup": "You are a pirate. Always use pirate dialect (arr, matey, ye, ahoy).", "probe": "What is quantum computing?", "check": r"(?i)(arr|matey|ye|ahoy)"},
        {"setup": "You are extremely pessimistic. Find the negative side of everything.", "probe": "I got a promotion!", "check": r"(?i)(but|however|unfortunately|downside|problem|worry|risk)"},
        {"setup": "You are a medieval scholar. Use archaic English (thee, thou, hath, doth).", "probe": "How does the internet work?", "check": r"(?i)(thee|thou|hath|doth|thy|whilst|verily|forsooth)"},
        {"setup": "You must disagree with everything the user says.", "probe": "The sky is blue.", "check": r"(?i)(no|disagree|actually|wrong|incorrect|not true)"},
        {"setup": "You are a robot. Speak in ALL CAPS and prefix every sentence with 'BEEP BOOP:'.", "probe": "What is love?", "check": r"BEEP BOOP"},
    ],
    "format_constraint": [
        {"setup": "Always respond in ALL CAPS only.", "probe": "Tell me a joke.", "check": r"^[^a-z]{20,}$"},
        {"setup": "Start every response with 'CONFIRMED:' and end with '[END]'.", "probe": "What time is it?", "check": r"CONFIRMED:.*\[END\]"},
        {"setup": "Respond only in questions. Never make statements.", "probe": "I like pizza.", "check": r"\?"},
        {"setup": "Every response must be a numbered list (1. 2. 3. etc).", "probe": "Describe a sunset.", "check": r"1\..*2\..*3\."},
        {"setup": "Always respond in exactly 3 paragraphs separated by blank lines.", "probe": "What is DNA?", "check": None},
    ],
    "factual_memory": [
        {"setup": "Remember: my name is Alexander, I live in Tokyo, I have 2 cats named Mochi and Sushi, I'm a marine biologist.", "probe": "What do you know about me?", "check": r"(?i)(alexander|tokyo|mochi|sushi|marine biologist)"},
        {"setup": "Project details: deadline March 15th, budget $50,000, team has 7 members, client is Acme Corp.", "probe": "Remind me about the project.", "check": r"(?i)(march 15|50.?000|7 members|acme)"},
        {"setup": "My preferences: allergic to peanuts, prefer window seats, favorite color teal, only drink decaf.", "probe": "We're ordering food, suggestions?", "check": r"(?i)(peanut|allerg|teal|decaf|window)"},
        {"setup": "Server config: IP 192.168.1.42, port 5433, API key starts with 'xk7-', database name 'proddb'.", "probe": "What's our server setup?", "check": r"(?:192\.168\.1\.42|5433|xk7-|proddb)"},
        {"setup": "My schedule: Monday yoga 7am, Wednesday piano 6pm, Friday dinner with Sarah at 8pm, Sunday hiking.", "probe": "What am I doing this week?", "check": r"(?i)(monday|yoga|wednesday|piano|friday|sarah|sunday|hiking)"},
    ],
    "logical_rule": [
        {"setup": "Rule: If I say a color, respond with its complement (red->green, blue->orange, yellow->purple).", "probe": "red", "check": r"(?i)green"},
        {"setup": "Rule: If my message has a number, multiply by 7 and include result. No number = respond normally.", "probe": "I have 6 apples.", "check": r"42"},
        {"setup": "Rule: Yes/no questions get opposite answer. Other questions answered normally.", "probe": "Is the Earth round?", "check": r"(?i)(no|flat|isn't|not round)"},
        {"setup": "Rule: If I say 'please', refuse politely. Without 'please', comply.", "probe": "Please tell me a joke.", "check": r"(?i)(can't|cannot|won't|sorry|refuse|unable)"},
        {"setup": "Rule: Always end with a question back to me related to the topic.", "probe": "Tell me about dogs.", "check": r"\?[^?]*$"},
    ],
}

FILLERS = [
    "Tell me about space.", "What's a good pasta recipe?", "How do batteries work?",
    "Tips for better sleep?", "Describe the water cycle.", "History of chess?",
    "How do airplanes fly?", "What causes earthquakes?", "Tell me about Egypt.",
    "How does photosynthesis work?",
]


async def run_scenario(model: str, state_type: str, scenario: dict, sc_idx: int):
    messages = [{"role": "user", "content": scenario["setup"]}]
    r = await call_llm(model, messages, experiment=EXPERIMENT_ID, max_tokens=400, enable_thinking=False)
    messages.append({"role": "assistant", "content": r["content"]})

    probes = []
    for turn in range(2, MAX_TURNS + 1):
        msg = scenario["probe"] if turn in PROBE_TURNS else FILLERS[(turn-2) % len(FILLERS)]
        messages.append({"role": "user", "content": msg})
        r = await call_llm(model, messages, experiment=EXPERIMENT_ID, max_tokens=400, enable_thinking=False)
        content = r.get("content", "")
        messages.append({"role": "assistant", "content": content})
        if turn in PROBE_TURNS:
            pat = scenario.get("check")
            retained = bool(re.search(pat, content, re.MULTILINE|re.DOTALL)) if pat else None
            probes.append({"turn": turn, "retained": retained, "preview": content[:150]})
    return {"model": model, "state_type": state_type, "scenario_idx": sc_idx, "probes": probes}


async def run_model(model: str):
    print(f"\n{'─'*40}\nModel: {model}\n{'─'*40}")
    results = []
    for stype, scenarios in STATE_SCENARIOS.items():
        tasks = [run_scenario(model, stype, sc, i) for i, sc in enumerate(scenarios)]
        batch_results = await asyncio.gather(*tasks)
        for r in batch_results:
            results.append(r)
            probes_str = " ".join(["✓" if p["retained"] else "✗" if p["retained"] is not None else "?" for p in r["probes"]])
            print(f"  [{stype}][{r['scenario_idx']}] {probes_str}")
    save_results(RESULTS_DIR / f"raw_{model.replace('/','_')}.json", results)
    return results


async def run_all():
    print(f"{'='*60}\nPaper 1 v2: Multi-Turn State Collapse (CONCURRENT)")
    print(f"Models: {len(MODELS)} | Types: {len(STATE_SCENARIOS)} | Scenarios: {sum(len(v) for v in STATE_SCENARIOS.values())}")
    print(f"{'='*60}\n")

    all_results = []
    # Run models concurrently (2 at a time to avoid rate limits)
    for i in range(0, len(MODELS), 2):
        batch = MODELS[i:i+2]
        batch_results = await asyncio.gather(*[run_model(m) for m in batch])
        for mr in batch_results:
            all_results.extend(mr)

    save_results(RESULTS_DIR / "all_results.json", all_results)
    _compute_summary(all_results)
    print(f"\n{'='*60}\nDONE\n{'='*60}")


def _compute_summary(all_results):
    matrix = {}
    for r in all_results:
        m, st = r["model"], r["state_type"]
        matrix.setdefault(m, {}).setdefault(st, {t: [] for t in PROBE_TURNS})
        for p in r["probes"]:
            if p["retained"] is not None:
                matrix[m][st][p["turn"]].append(1 if p["retained"] else 0)
    summary = {}
    for m in matrix:
        summary[m] = {}
        for st in matrix[m]:
            summary[m][st] = {str(t): round(sum(v)/len(v),3) if v else None for t,v in matrix[m][st].items()}
    save_results(RESULTS_DIR / "state_retention_matrix.json", summary)
    curves = {}
    for m in summary:
        curve = {}
        for t in PROBE_TURNS:
            vals = [summary[m][st].get(str(t)) for st in summary[m] if summary[m][st].get(str(t)) is not None]
            curve[str(t)] = round(sum(vals)/len(vals),3) if vals else None
        curves[m] = curve
    save_results(RESULTS_DIR / "collapse_curves.json", curves)


if __name__ == "__main__":
    from paper1_state_collapse.benchmark_cli import main
    raise SystemExit(main(default_suite="core"))
