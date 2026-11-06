"""Verify packaged measurements without CUDA or private router inputs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_file(path, expected):
    require(path.is_file(), f"Missing file: {path}")
    require(digest(path.read_bytes()) == expected, f"SHA-256 mismatch: {path}")


def unpack(archive, destination):
    # Accept only ordinary files/directories below the evidence/ directory.
    with tarfile.open(archive, "r:gz") as tar:
        for item in tar.getmembers():
            path = Path(item.name)
            require(not path.is_absolute() and ".." not in path.parts,
                    f"Unsafe archive member: {item.name}")
            require(path.parts and path.parts[0] == "evidence",
                    f"Unexpected archive root: {item.name}")
            require(item.isdir() or item.isfile(),
                    f"Unsupported archive member: {item.name}")
            target = destination / path
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(item) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sources", action="store_true",
                        help="Also verify this checkout's measured source files")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "delivery_manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        verify_file(ROOT / name, expected)
    if args.check_sources:
        for name, expected in manifest["runtime_and_tests"].items():
            verify_file(ROOT.parent.parent / name, expected)
    receipt = json.loads((ROOT / "validation_receipt.json").read_text())
    summary_bytes = gzip.decompress((ROOT / "summary.json.gz").read_bytes())
    for name, expected in receipt["sha256"].items():
        if name == "summary.json":
            require(digest(summary_bytes) == expected, "Original summary hash mismatch")
        else:
            verify_file(ROOT / name, expected)
    with tempfile.TemporaryDirectory(prefix="moonep-sm128-verify-") as work:
        work = Path(work)
        for name in ("analysis.py", "run_context.json", "source_manifest.json"):
            shutil.copy2(ROOT / name, work / name)
        shutil.copytree(ROOT / "refs", work / "refs")
        unpack(ROOT / "evidence.tar.gz", work)
        require((work / "evidence" / "complete").exists(), "Incomplete evidence")
        for name in ("selective32_before", "bounded128", "selective128",
                     "profile128", "selective32_after"):
            require((work / "evidence" / f"{name}.exit").read_text().strip() == "0",
                    f"GPU run failed: {name}")
        subprocess.run([sys.executable, str(work / "analysis.py")], cwd=work,
                       check=True)
        require((work / "summary.json").read_bytes() == summary_bytes,
                "Recomputed summary differs from delivery")
        require((work / "report.md").read_bytes() == (ROOT / "report.md").read_bytes(),
                "Recomputed report differs from delivery")
    print(f"PASS: {len(manifest['artifacts'])} artifact hashes; "
          "raw evidence independently recomputed; summary/report exact.")
    if args.check_sources:
        print(f"PASS: {len(manifest['runtime_and_tests'])} measured source hashes.")


if __name__ == "__main__":
    main()
