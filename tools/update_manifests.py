"""Refresh release checksums after intentionally changing public files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_package import release_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-date", help="Optional ISO release date; otherwise keep the existing date")
    args = parser.parse_args()
    names = sorted(release_files(ROOT) - {"SHA256SUMS.txt"})
    lines = [f'{hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}  {name}' for name in names]
    (ROOT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest_path = ROOT / "PACKAGE_MANIFEST.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.release_date:
        metadata["release_date"] = args.release_date
    metadata["members"] = [{"path": name, "bytes": (ROOT / name).stat().st_size,
                            "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
                           for name in sorted(release_files(ROOT))]
    manifest_path.write_text(json.dumps(metadata, indent=4, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f'RELEASE_MANIFESTS: updated {len(metadata["members"])} members')


if __name__ == "__main__":
    main()
