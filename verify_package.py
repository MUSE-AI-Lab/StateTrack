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

def main() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in payload["members"]}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST and ".git" not in path.relative_to(ROOT).parts
    }
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
