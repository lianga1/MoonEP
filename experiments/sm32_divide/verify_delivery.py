"""Recompute the delivered evidence without CUDA, PyTorch, or private fixtures.

Run from a clean checkout: python3 experiments/sm32_divide/verify_delivery.py --check-sources
Omit --check-sources when inspecting evidence in a checkout with unrelated edits.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as handle:
        for member in handle.getmembers():
            p = Path(member.name)
            assert not p.is_absolute() and '..' not in p.parts
            assert member.isdir() or member.isfile(), member.name
        handle.extractall(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sources', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'delivery_manifest.json').read_text())
    for name, expected in manifest['artifacts'].items():
        assert digest(HERE / name) == expected, ('artifact changed', name)
    if args.check_sources:
        repo = HERE.parent.parent
        for name, expected in manifest['runtime_and_tests'].items():
            assert digest(repo / name) == expected, ('source changed', name)
    with tempfile.TemporaryDirectory(prefix='moonep-sm32-audit-') as tmp:
        root = Path(tmp)
        unpack(HERE / 'baseline_profiles_frozen03.tar.gz', root)
        unpack(HERE / 'final03.tar.gz', root)
        unpack(HERE / 'components_diagnostics_02.tar.gz', root / 'components_diagnostics_02')
        for name in ('candidate_manifest_03.json', 'candidate_payload_03.tar.gz',
                     'recompute_baseline32.py', 'recompute_final32.py', 'recompute_peer_grids.py'):
            shutil.copy2(HERE / name, root / name)
        for script in ('recompute_baseline32.py', 'recompute_final32.py', 'recompute_peer_grids.py'):
            subprocess.run([sys.executable, str(root / script)], check=True,
                           stdout=subprocess.DEVNULL)
        for stem in ('analysis_baseline32', 'analysis_final32'):
            expected = json.loads(gzip.decompress((HERE / (stem + '.json.gz')).read_bytes()))
            actual = json.loads((root / (stem + '.json')).read_text())
            assert expected == actual, ('recomputed analysis differs', stem)
        expected_peer = json.loads((HERE / 'peer_grids_independent_check.json').read_text())
        actual_peer = json.loads((root / 'peer_grids_independent_check.json').read_text())
        assert expected_peer == actual_peer, 'peer-grid independent recomputation differs'
        evidence = root / 'final03/evidence'
        for name in ('probe_selective03', 'config03', 'components03', 'tiles03', 'public03', 'compatibility03'):
            assert (evidence / (name + '.exit')).read_text().strip() == '0', name
        cases = dict(config03=24, components03=9, tiles03=5, public03=17, compatibility03=5)
        for name, expected_count in cases.items():
            observed = {int(n) for n in re.findall(r'(\d+) passed', (evidence / (name + '.log')).read_text())}
            assert observed == {expected_count}, ('pytest case count', name, observed)
        assert (evidence / 'final_complete03').is_file()
    print(json.dumps(dict(status='passed', artifacts=len(manifest['artifacts']),
        source_files=len(manifest['runtime_and_tests']) if args.check_sources else 'not requested',
        baseline_and_final_analysis_exact=True, peer_grids_exact=True,
        unique_pytest_cases=60,
        private_fixture_or_GPU_required=False), indent=2))


if __name__ == '__main__':
    main()
