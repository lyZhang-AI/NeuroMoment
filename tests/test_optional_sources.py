"""Verify optional downloads without requesting network access."""
import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('fetch_optional', Path(__file__).parents[1]/'scripts/fetch_brainmodule.py')
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


def test_hash_mismatch_never_installs_source(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'corrupted'))
    metadata = {'source.py': {'url': 'https://example.test/source.py', 'sha256': hashlib.sha256(b'expected').hexdigest()}}
    with pytest.raises(ValueError, match='hash mismatch'):
        fetcher.fetch(metadata, tmp_path)
    assert not list(tmp_path.iterdir())


def test_verified_sources_can_be_reused_offline(tmp_path, monkeypatch):
    content = b'# pinned source\n'
    metadata = {'source.py': {'url': 'https://example.test/source.py', 'sha256': hashlib.sha256(content).hexdigest()}}
    monkeypatch.setattr(fetcher.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(content))
    assert fetcher.fetch(metadata, tmp_path) == 1
    monkeypatch.setattr(fetcher.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('Unexpected network request'))
    assert fetcher.fetch(metadata, tmp_path) == 1
    (tmp_path/'source.py').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='unexpected hash'):
        fetcher.fetch(metadata, tmp_path)
