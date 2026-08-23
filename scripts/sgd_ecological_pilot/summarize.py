"""Aggregate judged pilot metrics with Wilson confidence intervals."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = ("task_success", "constraint_compliance", "strict_joint", "recall", "sic_plus", "sic_minus", "old_rule_enforcement")


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float]:
    if n <= 0: return [0.0, 0.0]
    p = k / n; denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows); counts = {m: sum(bool(row.get(m)) for row in rows) for m in METRICS}
    return {"n": n, "counts": counts, "rates": {m: (counts[m] / n if n else 0.0) for m in METRICS}, "wilson_95": {m: wilson_interval(counts[m], n) for m in METRICS}}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list); by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["case_type"]].append(row); by_model[row["model"]].append(row)
    return {"n_records": len(rows), "metrics": list(METRICS), "overall": _group(rows), "by_case_type": {k: _group(v) for k, v in sorted(by_type.items())}, "by_model": {k: _group(v) for k, v in sorted(by_model.items())}}


def write_summary(results: Path) -> dict[str, Any]:
    rows = json.loads((results / "judged_outputs.json").read_text(encoding="utf-8"))
    summary = aggregate(rows)
    (results / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# SGD ecological pilot summary",
        "",
        f"Records: {summary['n_records']}",
        "",
        "| Group | N | Compliance | Recall | SIC+ | SIC- | Strict joint |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def row(label, group):
        return (
            f"| {label} | {group['n']} | "
            f"{group['rates']['constraint_compliance']:.1%} | "
            f"{group['rates']['recall']:.1%} | {group['rates']['sic_plus']:.1%} | "
            f"{group['rates']['sic_minus']:.1%} | {group['rates']['strict_joint']:.1%} |"
        )

    lines.append(row("Overall", summary["overall"]))
    for key, group in summary["by_case_type"].items():
        lines.append(row(key, group))
    lines += [
        "",
        "Wilson 95% intervals are stored in `summary.json`; all SIC labels are derived in code from recall and compliance.",
    ]
    (results / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parents[2] / "results" / "ecological_pilot"
    print(json.dumps(write_summary(output_dir), ensure_ascii=False, indent=2))
