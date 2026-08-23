from pathlib import Path
import hashlib
import sys

root = Path(__file__).resolve().parents[1]
manifest = root / "SHA256SUMS.txt"
errors = []
for line in manifest.read_text(encoding="utf-8-sig").splitlines():
    if not line.strip():
        continue
    expected, relative = line.split(maxsplit=1)
    path = root / relative.lstrip("*").replace("\\", "/")
    if not path.exists():
        errors.append(f"missing: {relative}")
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual.lower() != expected.lower():
        errors.append(f"checksum mismatch: {relative}")
if errors:
    print("\n".join(errors))
    sys.exit(1)
print("All SHA256 checks passed.")
