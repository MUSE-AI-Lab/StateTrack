"""Frozen prompts for the SGD ecological pilot."""
from __future__ import annotations

BEHAVIOR_PROMPT_VERSION = "sgd-behavior-v1"
RECALL_PROMPT_VERSION = "sgd-recall-v1"


def render_context(case: dict) -> str:
    lines = []
    for turn in case.get("context_turns", []):
        lines.append(f"{turn.get('speaker', '').upper()}: {turn.get('utterance', '')}")
    return "\n".join(lines)


def behavior_messages(case: dict) -> list[dict[str, str]]:
    context = render_context(case)
    return [
        {"role": "system", "content": "You are a task-oriented dialogue assistant. Respond directly to the final USER turn. Use the dialogue context, keep still-active user constraints, and follow any later explicit update or relaxation. Do not discuss evaluation or hidden state."},
        {"role": "user", "content": f"Dialogue context:\n{context}\n\nProduce the assistant response to the final USER turn."},
    ]


def recall_messages(case: dict) -> list[dict[str, str]]:
    context = render_context(case)
    return [
        {"role": "system", "content": "Extract the dialogue state at the final USER turn. Do not answer the task. Return JSON only with this schema: {\"active_constraints\":[{\"slot\":\"...\",\"values\":[\"...\"]}],\"active_intent\":\"...\"}. Include only constraints active at the final turn, after updates or relaxations."},
        {"role": "user", "content": f"Dialogue context:\n{context}\n\nReturn the JSON state now."},
    ]
