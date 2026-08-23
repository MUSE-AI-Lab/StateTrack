from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results" / "ecological_pilot"
MANIFEST = RESULTS / "checksums.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
errors = []
for row in manifest["files"]:
    path = RESULTS / row["path"]
    if not path.is_file():
        errors.append(f"missing: {row['path']}")
        continue
    if path.stat().st_size != row["size"] or sha256(path) != row["sha256"]:
        errors.append(f"checksum mismatch: {row['path']}")

if errors:
    raise SystemExit("\n".join(errors))

print(f"Verified {len(manifest['files'])} ecological-pilot files.")
