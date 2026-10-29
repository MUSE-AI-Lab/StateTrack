#!/usr/bin/env python3
# Verify exact archive membership, byte counts, and SHA-256 digests.
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "PACKAGE_MANIFEST.json"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files(root: Path) -> set[str]:
    """Exclude only documented runtime state, keeping unknown extras detectable."""
    ignored_dirs = {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}
    result = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or path == root / "PACKAGE_MANIFEST.json":
            continue
        if set(relative.parts) & ignored_dirs or relative.parts[0] == "outputs":
            continue
        if path.suffix in {".pyc", ".pyo", ".log"} or relative.as_posix() in {".env", "experiment_config.json", ".DS_Store"}:
            continue
        result.add(relative.as_posix())
    return result

def main() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in payload["members"]}
    actual = release_files(ROOT)
    errors: list[str] = []
    if set(expected) != actual:
        errors.append(
            f"member set mismatch: missing={sorted(set(expected)-actual)}, "
            f"extra={sorted(actual-set(expected))}"
        )
    for name in sorted(set(expected) & actual):
        path = ROOT / Path(name)
        item = expected[name]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            errors.append(f"digest mismatch: {name}")
    if errors:
        print("PACKAGE INVALID")
        print("\n".join(errors))
        return 1
    print(f"PACKAGE OK: {len(expected)} verified members")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
