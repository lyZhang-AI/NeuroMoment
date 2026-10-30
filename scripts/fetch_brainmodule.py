"""Fetch optional CC-BY-NC-4.0 Brainmagick sources at pinned hashes.

These files are not part of the MIT-licensed core distribution.
See THIRD_PARTY_NOTICES.md and the downloaded original LICENSE.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]


def fetch(manifest, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for filename, meta in manifest.items():
        if Path(filename).name != filename or filename in {'.', '..'}:
            raise ValueError('Source filename must be a basename')
        target = out / filename
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != meta['sha256']:
                raise ValueError(f'Existing source has an unexpected hash: {filename}')
        else:
            request = urllib.request.Request(meta['url'], headers={'User-Agent': 'NeuroMoment/0.2.1'})
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != meta['sha256']:
                raise ValueError(f'Download hash mismatch: {filename}')
            temporary = target.with_suffix(target.suffix + '.part')
            temporary.write_bytes(content)
            temporary.replace(target)
        target.with_suffix(target.suffix + '.json').write_text(json.dumps(meta, indent=2) + '\n')
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=PROJECT / 'third_party/baseline_sources')
    args = parser.parse_args()
    manifest = json.loads((PROJECT / 'third_party/brainmagick_sources.json').read_text())
    count = fetch(manifest, args.out)
    print(f'Verified {count} optional files. Their upstream CC-BY-NC-4.0 license applies.')


if __name__ == '__main__':
    main()
