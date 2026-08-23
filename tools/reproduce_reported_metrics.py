from __future__ import annotations

import json
import statistics
from pathlib import Path


def load_json(root: Path, relative: str):
    return json.loads((root / relative).read_text(encoding="utf-8-sig"))


def build_report(root: Path) -> dict:
    curves = load_json(root, "results/core/collapse_curves.json")
    primary = {
        model: round(100.0 * statistics.mean(depth_rows.values()), 1)
        for model, depth_rows in curves.items()
    }

    rebuttal = load_json(root, "results/rebuttal/rebuttal_experiment_summary.json")
    dynamic = {row["condition"]: row["stale_rate"] for row in rebuttal["dynamic_summary"]}

    symbolic = load_json(root, "results/rebuttal/symbolic_depth_summary.json")
    depth50 = next(row for row in symbolic["by_depth"] if row["depth"] == 50)

    mechanism = load_json(root, "results/rebuttal/symbolic_execution_bottleneck_audit.json")
    numerical = next(row for row in mechanism["category_rows"] if row["category"] == "Numerical")

    multi = load_json(root, "results/rebuttal/multiconstraint_summary.json")
    calibration = load_json(root, "results/checker_calibration/summary.json")

    return {
        "primary_strict_joint": primary,
        "dynamic_old_rule": dynamic,
        "symbolic_depth50": {
            "n": depth50["n"],
            "compliance": depth50["constraint"],
            "recall": depth50["recall"],
            "sic_plus": depth50["sic_plus"],
        },
        "execution_bottleneck": {
            "numerical_n": numerical["n"],
            "numerical_recall": numerical["recall_rate"],
            "numerical_compliance": numerical["constraint_rate"],
            "numerical_sic_plus": numerical["sic_plus_rate"],
            "models_with_sic_plus": mechanism["models_with_at_least_one_sic_plus"],
        },
        "two_constraint": {
            "valid_outputs": multi["valid_outputs"],
            "joint_success": multi["aggregate"]["joint_two_constraint"],
            "recall": multi["aggregate"]["recall"],
            "sic_plus": multi["aggregate"]["sic_plus"],
        },
        "checker_calibration": {
            "items": calibration["items"],
            "exact_pass_agreement": round(100.0 * calibration["pass_exact_agreement"], 1),
            "adjudicated_items": calibration["adjudicated_items"],
            "task_agreement": round(100.0 * calibration["task_agreement"]["agreement"], 1),
            "task_kappa": round(calibration["task_agreement"]["cohen_kappa"], 3),
            "constraint_agreement": round(100.0 * calibration["constraint_agreement"]["agreement"], 1),
            "constraint_kappa": round(calibration["constraint_agreement"]["cohen_kappa"], 3),
            "recall_agreement": round(100.0 * calibration["recall_agreement"]["agreement"], 1),
            "recall_kappa": round(calibration["recall_agreement"]["cohen_kappa"], 3),
        },
    }


def render_report(report: dict) -> str:
    lines = ["# StateTrack-Bench cached metric reproduction", "", "## Primary strict-joint retention"]
    for model, value in report["primary_strict_joint"].items():
        lines.append(f"{model} | {value:.1f}%")

    dynamic = report["dynamic_old_rule"]
    lines += [
        "",
        "## Dynamic update/revocation",
        f"Base dialogue | {dynamic['base_after_change']:.1f}%",
        f"Stale-ledger SSA | {dynamic['stale_ssa']:.1f}%",
        f"Updated-ledger SSA | {dynamic['updated_ledger_ssa']:.1f}%",
    ]

    depth = report["symbolic_depth50"]
    mech = report["execution_bottleneck"]
    lines += [
        "",
        "## Symbolic depth-50 audit",
        f"Valid paired probes | {depth['n']}",
        f"Depth-50 compliance | {depth['compliance']:.1f}%",
        f"Depth-50 recall | {depth['recall']:.1f}%",
        f"Depth-50 SIC+ | {depth['sic_plus']:.1f}%",
        f"Numerical depth-50 recall | {mech['numerical_recall']:.1f}%",
        f"Numerical depth-50 compliance | {mech['numerical_compliance']:.1f}%",
        f"Numerical depth-50 SIC+ | {mech['numerical_sic_plus']:.1f}%",
        f"Non-Gemini models with at least one SIC+ | {mech['models_with_sic_plus']}/6",
    ]

    multi = report["two_constraint"]
    lines += [
        "",
        "## Two-constraint audit",
        f"Valid outputs | {multi['valid_outputs']}",
        f"Two-constraint joint success | {multi['joint_success']:.1f}%",
        f"Forked recall | {multi['recall']:.1f}%",
        f"SIC+ | {multi['sic_plus']:.1f}%",
    ]

    calibration = report["checker_calibration"]
    lines += [
        "",
        "## Blind checker calibration",
        f"Checker calibration rows | {calibration['items']}",
        f"Exact A/B agreement | {calibration['exact_pass_agreement']:.1f}%",
        f"Disagreement adjudications | {calibration['adjudicated_items']}",
        f"Immediate-task agreement / kappa | {calibration['task_agreement']:.1f}% / {calibration['task_kappa']:.3f}",
        f"Constraint agreement / kappa | {calibration['constraint_agreement']:.1f}% / {calibration['constraint_kappa']:.3f}",
        f"Recall agreement / kappa | {calibration['recall_agreement']:.1f}% / {calibration['recall_kappa']:.3f}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(render_report(build_report(root)), end="")


if __name__ == "__main__":
    main()
