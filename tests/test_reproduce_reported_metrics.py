from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "reproduce_reported_metrics.py"


class ReproduceReportedMetricsTests(unittest.TestCase):
    def load_module(self):
        spec = importlib.util.spec_from_file_location("reproduce_reported_metrics", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    def test_build_report_reproduces_primary_and_rebuttal_values(self):
        module = self.load_module()
        report = module.build_report(ROOT)
        self.assertEqual(report["primary_strict_joint"]["gemini-3.1-pro-preview"], 62.4)
        self.assertEqual(report["primary_strict_joint"]["deepseek-v4-flash"], 100.0)
        self.assertEqual(report["dynamic_old_rule"]["stale_ssa"], 74.3)
        self.assertEqual(report["symbolic_depth50"]["sic_plus"], 9.7)
        self.assertEqual(report["execution_bottleneck"]["numerical_recall"], 100.0)
        self.assertEqual(report["execution_bottleneck"]["numerical_compliance"], 79.2)
        self.assertEqual(report["two_constraint"]["joint_success"], 76.5)
        self.assertEqual(report["checker_calibration"]["items"], 70)
        self.assertEqual(report["checker_calibration"]["exact_pass_agreement"], 98.6)
        self.assertEqual(report["checker_calibration"]["constraint_kappa"], 0.870)
        self.assertEqual(report["checker_calibration"]["recall_kappa"], 0.900)

    def test_render_report_contains_copyable_tables(self):
        module = self.load_module()
        text = module.render_report(module.build_report(ROOT))
        self.assertIn("Primary strict-joint retention", text)
        self.assertIn("gemini-3.1-pro-preview | 62.4%", text)
        self.assertIn("Updated-ledger SSA | 0.0%", text)
        self.assertIn("Numerical depth-50 recall | 100.0%", text)
        self.assertIn("Two-constraint joint success | 76.5%", text)
        self.assertIn("Checker calibration rows | 70", text)
        self.assertIn("Constraint agreement / kappa | 95.7% / 0.870", text)
        self.assertIn("Recall agreement / kappa | 95.7% / 0.900", text)


if __name__ == "__main__":
    unittest.main()
