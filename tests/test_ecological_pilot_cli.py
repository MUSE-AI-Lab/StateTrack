from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class EcologicalPilotCliTests(unittest.TestCase):
    def test_reproduce_pilot_metrics_command(self):
        completed = subprocess.run(
            [sys.executable, "scripts/sgd_ecological_pilot/reproduce_pilot_metrics.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"n_records": 210', completed.stdout)


if __name__ == "__main__":
    unittest.main()
