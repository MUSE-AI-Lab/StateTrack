from __future__ import annotations

import json
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


class StateTrackEvidenceTests(unittest.TestCase):
    def test_submitted_retention_table(self):
        curves = load_json("results/core/collapse_curves.json")
        self.assertEqual(len(curves), 7)
        overall = {
            model: round(100.0 * statistics.mean(depths.values()), 1)
            for model, depths in curves.items()
        }
        self.assertEqual(
            overall,
            {
                "gpt-5.4": 94.4,
                "claude-sonnet-4-6": 90.4,
                "gemini-3.1-pro-preview": 62.4,
                "deepseek-v4-pro": 96.0,
                "qwen3.6-plus": 95.2,
                "gpt-5.4-mini": 98.4,
                "deepseek-v4-flash": 100.0,
            },
        )
        self.assertEqual(curves["gemini-3.1-pro-preview"]["12"], 0.54)

    def test_dynamic_and_split_analyses(self):
        split_rows = load_json("results/rebuttal/rebuttal_split_sic_raw.json")
        dynamic_rows = load_json("results/rebuttal/rebuttal_revocation_update_raw.json")
        summary = load_json("results/rebuttal/rebuttal_experiment_summary.json")
        self.assertEqual(len(split_rows), 70)
        self.assertEqual(len(dynamic_rows), 105)
        rates = {row["condition"]: row["stale_rate"] for row in summary["dynamic_summary"]}
        self.assertEqual(rates, {"base_after_change": 2.9, "stale_ssa": 74.3, "updated_ledger_ssa": 0.0})

    def test_symbolic_depth_and_two_constraint_analyses(self):
        symbolic_raw = load_json("results/rebuttal/symbolic_depth_raw.json")
        symbolic = load_json("results/rebuttal/symbolic_depth_summary.json")
        mechanism = load_json("results/rebuttal/symbolic_execution_bottleneck_audit.json")
        multi = load_json("results/rebuttal/multiconstraint_summary.json")
        self.assertEqual(len(symbolic_raw), 216)
        depth50 = next(row for row in symbolic["by_depth"] if row["depth"] == 50)
        self.assertEqual((depth50["n"], depth50["sic_plus"]), (72, 9.7))
        numerical = next(row for row in mechanism["category_rows"] if row["category"] == "Numerical")
        self.assertEqual((numerical["recall_rate"], numerical["constraint_rate"], numerical["sic_plus_rate"]), (100.0, 79.2, 20.8))
        self.assertEqual((multi["valid_outputs"], multi["aggregate"]["joint_two_constraint"]), (34, 76.5))

    def test_ecological_pilot_records(self):
        cases = load_json("results/ecological_pilot/selected_cases.json")
        raw = load_json("results/ecological_pilot/raw_subject_outputs.json")
        judged = load_json("results/ecological_pilot/judged_outputs.json")
        summary = load_json("results/ecological_pilot/summary.json")
        self.assertEqual((len(cases), len(raw), len(judged)), (30, 210, 210))
        self.assertEqual(summary["overall"]["counts"]["constraint_compliance"], 187)
        self.assertEqual(summary["overall"]["counts"]["strict_joint"], 140)
        self.assertEqual(summary["overall"]["counts"]["sic_plus"], 9)
        revocation = summary["by_case_type"]["revocation"]["counts"]
        self.assertEqual((revocation["constraint_compliance"], revocation["recall"]), (34, 34))

    def test_snapshot_manifest_lists_seven_models(self):
        lines = (ROOT / "metadata/snapshot_manifest.csv").read_text(encoding="utf-8-sig").strip().splitlines()
        self.assertEqual(len(lines), 8)


if __name__ == "__main__":
    unittest.main()
