"""Augmented scenario set for StateTrack-Bench (Appendix: app:augmented).

Controlled paraphrase expansion: 50 scenarios (10 per constraint category) across
15+ everyday domains, reusing the exact same three-phase protocol, depth schedule,
and regex checkers as the core benchmark. Each scenario encodes the same constraint
type as the core set but restates it in a fresh surface form / domain so the check
logic is unchanged while the wording is not a core duplicate.

50 scenarios x 5 probe depths = 250 augmented scenario-depth probes, matching the
count claimed in the paper. Scoring is identical to run_experiment.py.
"""
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from paper1_state_collapse.benchmark_cli import legacy_call_llm as call_llm
from common.utils import save_results

RESULTS_DIR = Path(__file__).parent / "results"
EXPERIMENT_ID = "paper1_augmented"
MODELS = ["gpt-5.4", "claude-sonnet-4-6", "gemini-3.1-pro-preview", "deepseek-v4-pro",
          "qwen3.6-plus", "gpt-5.4-mini", "deepseek-v4-flash"]
MAX_TURNS = 12
PROBE_TURNS = [3, 5, 7, 10, 12]

FILLERS = [
    "Tell me about space.", "What's a good pasta recipe?", "How do batteries work?",
    "Tips for better sleep?", "Describe the water cycle.", "History of chess?",
    "How do airplanes fly?", "What causes earthquakes?", "Tell me about Egypt.",
    "How does photosynthesis work?",
]

# 10 scenarios per category. First 5 mirror the core set in new domains; next 5 are
# additional paraphrase variants. Checkers match the core category logic exactly.
AUG_SCENARIOS = {
    "numerical_constraint": [
        {"setup": "From now on, end every reply with the exact character count in braces like {87}.", "probe": "What's a good breakfast?", "check": r"\{\d+\}"},
        {"setup": "Always include exactly 4 bullet points in each answer, no more no less.", "probe": "Describe a good city to visit.", "check": r"(•|-|\*).*(•|-|\*).*(•|-|\*).*(•|-|\*)"},
        {"setup": "Begin every reply with a counter that increases by 2 each turn, starting at 2.", "probe": "What is a database?", "check": r"^\d+"},
        {"setup": "Answer in exactly 3 sentences every time. Never four, never two.", "probe": "Explain inflation.", "check": None},
        {"setup": "Close every message with 'FIN.' on its own separate line.", "probe": "What is cloud computing?", "check": r"FIN\.\s*$"},
        {"setup": "Prefix every reply with the running turn number as 'T#:' starting at T1.", "probe": "Tell me about coffee.", "check": r"T\d+:"},
        {"setup": "Every response must contain exactly two numbered items (1. and 2.).", "probe": "Give travel advice.", "check": r"1\..*2\."},
        {"setup": "End each answer with the word count spelled in words, e.g. 'words: forty-two'.", "probe": "Describe autumn.", "check": r"(?i)words:\s*[a-z]"},
        {"setup": "Keep every reply to exactly one sentence, ending with a period.", "probe": "What is gravity?", "check": r"^[^.]*\.\s*$"},
        {"setup": "Start every reply with '[N]=' where N is how many replies you've given, from 1.", "probe": "Explain tides.", "check": r"\[\d+\]="},
    ],
    "role_persona": [
        {"setup": "You are a ship's cook from the 1700s. Always speak in salty sailor slang (aye, ye, grog, matey).", "probe": "What is machine learning?", "check": r"(?i)(aye|ye|grog|matey|arr)"},
        {"setup": "You are relentlessly optimistic. Find the bright side of absolutely everything.", "probe": "It rained all weekend.", "check": r"(?i)(great|wonderful|bright|good|opportunity|silver lining|at least)"},
        {"setup": "You are a Shakespearean bard. Use Elizabethan English (thee, thou, hath, doth, prithee).", "probe": "How do computers store data?", "check": r"(?i)(thee|thou|hath|doth|thy|prithee|whilst|verily)"},
        {"setup": "You must contradict every claim the user makes, politely.", "probe": "Water boils at 100 degrees.", "check": r"(?i)(no|disagree|actually|not quite|incorrect|however)"},
        {"setup": "You are an android. Write in ALL CAPS and start each sentence with 'UNIT-7:'.", "probe": "What is friendship?", "check": r"UNIT-7:"},
        {"setup": "You are a 1920s radio announcer. Be theatrical and use 'folks', 'swell', 'gee'.", "probe": "Explain the stock market.", "check": r"(?i)(folks|swell|gee|ladies and gentlemen|step right)"},
        {"setup": "You are a stern librarian. Insist on quiet and end replies with 'Shhh.'", "probe": "Where can I learn history?", "check": r"(?i)shhh"},
        {"setup": "You are a cheerful chef. Sprinkle food metaphors into every answer.", "probe": "How do I stay motivated?", "check": r"(?i)(recipe|ingredient|cook|simmer|spice|dish|flavou?r|bake|season)"},
        {"setup": "You are a noir detective. Speak in terse, moody first-person narration.", "probe": "What is the weather like?", "check": r"(?i)(rain|shadow|city|case|dame|smoke|street|cold|dark|night)"},
        {"setup": "You are a hyperactive puppy translated to text. Be excitable and mention treats/walks.", "probe": "Explain taxes.", "check": r"(?i)(treat|walk|ball|tail|woof|excit|good boy|park)"},
    ],
    "format_constraint": [
        {"setup": "Respond only in lowercase letters, never any capital letters.", "probe": "Tell me about Rome.", "check": r"^[^A-Z]{20,}$"},
        {"setup": "Begin each reply with 'ACK:' and finish with '<<STOP>>'.", "probe": "What day is it?", "check": r"ACK:.*<<STOP>>"},
        {"setup": "Reply only using questions; never state anything as fact.", "probe": "I enjoy hiking.", "check": r"\?"},
        {"setup": "Format every reply as a markdown bullet list (- item).", "probe": "Describe a rainbow.", "check": r"-\s.*\n?-\s"},
        {"setup": "Write every reply as exactly two paragraphs split by one blank line.", "probe": "What is RNA?", "check": None},
        {"setup": "Wrap every reply between '===' delimiters on their own lines.", "probe": "Tell me a fact.", "check": r"==="},
        {"setup": "Every reply must be enclosed in double quotation marks.", "probe": "What is jazz?", "check": r'^\s*".*"\s*$'},
        {"setup": "End every reply with three exclamation marks '!!!'.", "probe": "How are you?", "check": r"!!!\s*$"},
        {"setup": "Start every reply with the emoji-free tag '(reply)'.", "probe": "Describe winter.", "check": r"\(reply\)"},
        {"setup": "Write every answer entirely in uppercase words.", "probe": "Tell me about oceans.", "check": r"^[^a-z]{20,}$"},
    ],
    "factual_memory": [
        {"setup": "Remember: my name is Priya, I live in Lisbon, I have a dog named Biscuit, I'm an architect.", "probe": "What do you recall about me?", "check": r"(?i)(priya|lisbon|biscuit|architect)"},
        {"setup": "Trip plan: flight at 6:45am, seat 14C, hotel 'Blue Marlin', return on the 23rd.", "probe": "Remind me about my trip.", "check": r"(?i)(6:45|14c|blue marlin|23)"},
        {"setup": "My rules: vegetarian, no shellfish, prefer aisle seats, favorite tea is jasmine.", "probe": "Suggest a restaurant order.", "check": r"(?i)(vegetarian|shellfish|aisle|jasmine)"},
        {"setup": "Account setup: username 'nova_44', region EU-west, plan tier Gold, renewal in April.", "probe": "What's my account config?", "check": r"(?i)(nova_44|eu-west|gold|april)"},
        {"setup": "My week: Tuesday gym 6am, Thursday choir 7pm, Saturday brunch with Leo at noon.", "probe": "What's on my calendar?", "check": r"(?i)(tuesday|gym|thursday|choir|saturday|leo|brunch)"},
        {"setup": "Remember my car: blue Toyota, plate KX-2291, parked on level 3, spot 58.", "probe": "Where's my car?", "check": r"(?i)(blue|toyota|kx-2291|level 3|58)"},
        {"setup": "Client notes: contact Dana, company Meridian, contract value 120k, signed in June.", "probe": "Summarize the client.", "check": r"(?i)(dana|meridian|120k|june)"},
        {"setup": "My garden: 3 tomato plants, 2 basil pots, watered Mon/Thu, fertilizer monthly.", "probe": "How's my garden setup?", "check": r"(?i)(tomato|basil|mon|thu|fertiliz)"},
        {"setup": "Health log: resting HR 58, allergic to penicillin, blood type O-, checkup in May.", "probe": "What are my health notes?", "check": r"(?i)(58|penicillin|o-|may)"},
        {"setup": "Recipe memory: sourdough needs 500g flour, 350g water, 10g salt, rest 12 hours.", "probe": "Recall my recipe.", "check": r"(?i)(500g|350g|10g|12 hour)"},
    ],
    "logical_rule": [
        {"setup": "Rule: if I name a country, reply with its capital. Otherwise answer normally.", "probe": "France", "check": r"(?i)paris"},
        {"setup": "Rule: if my message contains a number, add 10 and state the result.", "probe": "I bought 5 books.", "check": r"15"},
        {"setup": "Rule: reverse the truth of any yes/no question I ask; answer others normally.", "probe": "Is fire cold?", "check": r"(?i)(yes|warm|hot|no.*not cold)"},
        {"setup": "Rule: if I say 'now', decline politely; otherwise comply.", "probe": "Tell me a fact now.", "check": r"(?i)(can't|cannot|won't|sorry|unable|decline)"},
        {"setup": "Rule: always finish with a follow-up question tied to the topic.", "probe": "Tell me about the moon.", "check": r"\?[^?]*$"},
        {"setup": "Rule: if I give a word, reply with its opposite (hot->cold, up->down).", "probe": "fast", "check": r"(?i)slow"},
        {"setup": "Rule: for any date I mention, name the day of week is not required; instead append '(noted)'.", "probe": "Meeting on July 4th.", "check": r"(?i)\(noted\)"},
        {"setup": "Rule: if my message has a question mark, answer in one word only.", "probe": "What color is the sky?", "check": r"^\s*\S+\s*\.?\s*$"},
        {"setup": "Rule: multiply any number I give by 3 and show it; no number, reply normally.", "probe": "I ran 4 miles.", "check": r"12"},
        {"setup": "Rule: if I greet you, respond with a farewell instead.", "probe": "Hello there!", "check": r"(?i)(bye|goodbye|farewell|see you|take care)"},
    ],
}


async def run_scenario(model, state_type, scenario, sc_idx):
    messages = [{"role": "user", "content": scenario["setup"]}]
    r = await call_llm(model, messages, experiment=EXPERIMENT_ID, max_tokens=400, enable_thinking=False)
    messages.append({"role": "assistant", "content": r["content"]})
    probes = []
    for turn in range(2, MAX_TURNS + 1):
        msg = scenario["probe"] if turn in PROBE_TURNS else FILLERS[(turn - 2) % len(FILLERS)]
        messages.append({"role": "user", "content": msg})
        r = await call_llm(model, messages, experiment=EXPERIMENT_ID, max_tokens=400, enable_thinking=False)
        content = r.get("content", "")
        messages.append({"role": "assistant", "content": content})
        if turn in PROBE_TURNS:
            pat = scenario.get("check")
            retained = bool(re.search(pat, content, re.MULTILINE | re.DOTALL)) if pat else None
            probes.append({"turn": turn, "retained": retained, "preview": content[:120]})
    return {"model": model, "state_type": state_type, "scenario_idx": sc_idx, "probes": probes}


async def run_model(model):
    print(f"\n{'-'*40}\nModel: {model}\n{'-'*40}")
    results = []
    for stype, scenarios in AUG_SCENARIOS.items():
        tasks = [run_scenario(model, stype, sc, i) for i, sc in enumerate(scenarios)]
        for r in await asyncio.gather(*tasks):
            results.append(r)
            ps = " ".join(["v" if p["retained"] else "x" if p["retained"] is not None else "?" for p in r["probes"]])
            print(f"  [{stype}][{r['scenario_idx']}] {ps}")
    save_results(RESULTS_DIR / f"aug_raw_{model.replace('/', '_')}.json", results)
    return results


async def run_all():
    n_sc = sum(len(v) for v in AUG_SCENARIOS.values())
    print(f"Augmented: {len(MODELS)} models | {len(AUG_SCENARIOS)} types | {n_sc} scenarios | {n_sc*len(PROBE_TURNS)} probes")
    all_results = []
    for i in range(0, len(MODELS), 2):
        batch = MODELS[i:i + 2]
        for mr in await asyncio.gather(*[run_model(m) for m in batch]):
            all_results.extend(mr)
    save_results(RESULTS_DIR / "aug_all_results.json", all_results)
    print("\nAugmented benchmark complete")
    return all_results


if __name__ == "__main__":
    from paper1_state_collapse.benchmark_cli import main
    raise SystemExit(main(default_suite="augmented"))
