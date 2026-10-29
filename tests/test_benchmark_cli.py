import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "run_benchmark.py"


class BenchmarkCLI(unittest.TestCase):
    def call(self, *args, env=None):
        return subprocess.run([sys.executable, str(CLI), *map(str, args)], cwd=ROOT,
                              capture_output=True, text=True, env=env, timeout=60)

    def test_banks_match_original_scenario_constants(self):
        for suite, name, constant in (("core", "run_experiment.py", "STATE_SCENARIOS"),
                                      ("augmented", "run_augmented.py", "AUG_SCENARIOS")):
            bank = json.loads((ROOT / "benchmarks" / f"{suite}.json").read_text(encoding="utf-8"))
            tree = ast.parse((ROOT / "scripts" / "paper1_state_collapse" / name).read_text(encoding="utf-8-sig"))
            original = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == constant for t in n.targets))
            self.assertEqual(bank["scenarios"], original)
            self.assertEqual(sum(map(len, original.values())), 25 if suite == "core" else 50)
            self.assertEqual(bank["probe_turns"], [3, 5, 7, 10, 12])

    def test_dry_run_has_no_provider_dependency(self):
        with tempfile.TemporaryDirectory() as output:
            result = self.call("--suite", "augmented", "--dry-run", "--output", output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((Path(output) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["scenario_count"], 50)
            self.assertEqual(manifest["probe_count_per_model"], 250)

    def test_replay_recomputes_every_core_curve(self):
        with tempfile.TemporaryDirectory() as output:
            result = self.call("--replay", ROOT / "results/core/all_results.json", "--output", output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            curves = json.loads((Path(output) / "collapse_curves.json").read_text(encoding="utf-8"))
            expected = json.loads((ROOT / "results/core/collapse_curves.json").read_text(encoding="utf-8-sig"))
            self.assertEqual(curves, expected)
            matrix = json.loads((Path(output) / "state_retention_matrix.json").read_text(encoding="utf-8"))
            self.assertEqual(matrix, json.loads((ROOT / "results/core/state_retention_matrix.json").read_text(encoding="utf-8-sig")))

    def test_online_protocol_and_resume_with_local_stub(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(data)
                body = json.dumps({"choices": [{"message": {"content": "Test answer.\nEND."}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as output:
                args = ["--model", "test-model", "--state-type", "numerical_constraint", "--scenario-index", "4",
                        "--output", output, "--base-url", f"http://127.0.0.1:{server.server_port}/v1"]
                env = {**os.environ, "OPENAI_API_KEY": "test-only-key"}
                result = self.call(*args, env=env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(len(requests), 12)
                self.assertEqual([len(r["messages"]) for r in requests], list(range(1, 24, 2)))
                rows = [json.loads(line) for line in (Path(output) / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual([p["turn"] for p in rows[0]["probes"]], [3, 5, 7, 10, 12])
                self.assertTrue(all(p["retained"] for p in rows[0]["probes"]))
                self.assertEqual(len(rows[0]["messages"]), 24)
                result = self.call(*args, "--resume", env=env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(len(requests), 12, "resume must not make duplicate calls")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_empty_and_incomplete_replays_do_not_report_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty.json"
            empty.write_text("[]", encoding="utf-8")
            result = self.call("--replay", empty, "--output", root / "empty-output")
            self.assertNotEqual(result.returncode, 0)
            rows = json.loads((ROOT / "results/core/all_results.json").read_text(encoding="utf-8-sig"))
            partial = root / "partial.json"
            partial.write_text(json.dumps(rows[:1]), encoding="utf-8")
            result = self.call("--replay", partial, "--output", root / "partial-output")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            summary = json.loads((root / "partial-output/run_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["completed_dialogues"], 1)

    def test_online_requires_explicit_credentials(self):
        with tempfile.TemporaryDirectory() as output:
            env = {key: value for key, value in os.environ.items() if key != "OPENAI_API_KEY"}
            result = self.call("--model", "test-model", "--output", output, env=env)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((Path(output) / "raw_results.jsonl").exists())

    def test_package_membership_excludes_runtime_but_detects_unknown_files(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("package_verifier", ROOT / "verify_package.py")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            (root / "outputs/runtime.json").write_text("{}")
            (root / "__pycache__").mkdir()
            (root / "__pycache__/test.pyc").write_bytes(b"cache")
            (root / "unexpected.py").write_text("pass")
            self.assertEqual(verifier.release_files(root), {"unexpected.py"})


if __name__ == "__main__":
    unittest.main()
