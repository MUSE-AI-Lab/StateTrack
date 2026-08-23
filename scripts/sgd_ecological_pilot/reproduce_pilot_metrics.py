from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results" / "ecological_pilot"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from summarize import aggregate  # noqa: E402

rows = json.loads((RESULTS / "judged_outputs.json").read_text(encoding="utf-8"))
recomputed = aggregate(rows)
stored = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))

if recomputed != stored:
    raise SystemExit("summary.json does not match judged_outputs.json")

print(json.dumps(recomputed, ensure_ascii=False, indent=2))
