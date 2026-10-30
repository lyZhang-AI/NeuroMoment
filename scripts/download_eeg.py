"""Hydrate pinned official dataset metadata, resume transfers, verify SHA256 annex keys.

Clone metadata first. Only fills files listed by the checked-out official repository.
Existing Alice files may be copied only after an exact size AND SHA256 match.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import quote

import requests

KEY = re.compile(r'SHA256E-s(\d+)--([a-f0-9]{64})')
LOCAL = threading.local()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def get_session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    return LOCAL.session


def manifest(root, base):
    out = root / 'download_manifest.json'
    if out.exists():
        data = json.loads(out.read_text())
        if data['commit'] != subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip():
            raise ValueError('Repository revision differs from recorded manifest')
        return data
    records = []
    tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
    for relative in filter(None, tracked):
        p = root / relative
        if p.is_symlink():
            text = str(p.readlink())
            target = p.resolve()
        elif p.is_file() and p.stat().st_size < 300:
            text = p.read_text(errors='replace')
            if not text.startswith('/annex/objects/'):
                continue
            target = p
        else:
            continue
        m = KEY.search(text)
        if m:
            if not target.is_relative_to(root):
                raise ValueError(f'Annex target outside dataset: {p}')
            records.append({'path': relative, 'target': str(target.relative_to(root)),
                            'size': int(m[1]), 'sha256': m[2],
                            'url': base + quote(relative, safe='/')})
        elif p.is_symlink():
            raise ValueError(f'Unsupported annex key: {p}')
    data = {'dataset': root.name, 'commit': subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_base': base, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'files': records, 'total_bytes': sum(r['size'] for r in records)}
    out.write_text(json.dumps(data, indent=2))
    return data


def fetch(root, rec, reuse):
    target = root / rec['target']
    if target.is_file() and target.stat().st_size == rec['size'] and digest(target) == rec['sha256']:
        return 'verified_existing', rec['size']
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    if reuse:
        old = reuse / rec['path']
        if old.is_file() and old.stat().st_size == rec['size'] and digest(old) == rec['sha256']:
            shutil.copyfile(old, part)
            if digest(part) != rec["sha256"]:
                raise ValueError("Copied file SHA256 mismatch")
            os.replace(part, target)
            return 'verified_local_copy', rec['size']
    last = None
    for attempt in range(6):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > rec['size']:
                part.unlink(); offset = 0
            # Bound each HTTP response: avoids long-lived proxy stream failures.
            while offset < rec['size']:
                end = min(offset + 8*1024*1024, rec['size']) - 1
                headers = {'Range': f'bytes={offset}-{end}'}
                with get_session().get(rec['url'], headers=headers, stream=True,
                                       timeout=(25, 60)) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                            raise ValueError('Incorrect resumed content range')
                        mode = 'ab' if offset else 'wb'
                    elif response.status_code == 200:
                        mode = 'wb'
                    else:
                        raise ValueError('Unexpected HTTP response')
                    with part.open(mode) as f:
                        for block in response.iter_content(1024*1024):
                            f.write(block)
                new_offset = part.stat().st_size
                if new_offset <= offset or new_offset > rec['size']:
                    raise ValueError('Invalid range transfer progress')
                offset = new_offset
            if part.stat().st_size != rec['size']:
                raise ValueError(f'Size mismatch {part.stat().st_size} != {rec["size"]}')
            if digest(part) != rec['sha256']:
                part.unlink()
                raise ValueError('SHA256 mismatch')
            os.replace(part, target)
            return 'downloaded_sha256_verified', rec['size']
        except Exception as e:
            last = str(e)
            if attempt < 5:
                time.sleep(min(2**attempt, 16))
    raise RuntimeError(f'{rec["path"]}: {last}')


def run(root, base, workers, reuse=None):
    data = manifest(root, base)
    total = len(data['files']); done = 0; checked_bytes = 0; errors = []; methods = {}
    started = time.monotonic()
    print(json.dumps({'dataset': root.name, 'files': total, 'GiB': data['total_bytes']/2**30,
                      'commit': data['commit']}), flush=True)
    report_path = root / 'download_status.json'
    def report(complete=False):
        status = {'dataset': root.name, 'commit': data['commit'], 'complete': complete,
                  'files_verified': done, 'files_total': total, 'bytes_verified': checked_bytes,
                  'bytes_total': data['total_bytes'], 'elapsed_seconds': round(time.monotonic()-started, 1),
                  'methods': methods, 'errors': errors, 'updated_utc': datetime.now(timezone.utc).isoformat()}
        temp = report_path.with_suffix('.tmp'); temp.write_text(json.dumps(status, indent=2)); temp.replace(report_path)
        print(json.dumps({**status, "errors": errors[:3], "error_count": len(errors)}), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(fetch, root, rec, reuse): rec for rec in data['files']}
        last = time.monotonic()
        while pending:
            completed, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            for future in completed:
                rec = pending.pop(future)
                try:
                    method, size = future.result()
                    done += 1; checked_bytes += size; methods[method] = methods.get(method, 0)+1
                except Exception as e:
                    errors.append(str(e)); print('ERROR '+str(e), flush=True)
            if time.monotonic()-last >= 20:
                report(); last = time.monotonic()
    report(complete=not errors and done == total)
    return not errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='datasets')
    p.add_argument('--dataset', choices=['ds004408', 'Brennan2019'], required=True)
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args(); root = Path(a.root).resolve()
    base = ('https://s3.amazonaws.com/openneuro.org/ds004408/' if a.dataset == 'ds004408'
            else 'https://data.nemar.org/nm000180/v1.1.3/')
    ok = run(root/a.dataset, base, a.workers, root/'Alice' if a.dataset == 'Brennan2019' else None)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
