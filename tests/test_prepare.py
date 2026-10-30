import csv
import json

import numpy as np
import pytest

from neuromoment.prepare import REQUIRED, prepare


def fixture(tmp_path, overlap=False):
    rng = np.random.default_rng(1)
    rows = []
    for subject in ['a', 'b']:
        np.save(tmp_path/f'{subject}.npy', rng.normal(size=(4, 6000)))
        for segment in range(10):
            start = segment*6.
            for j in range(4):
                rows.append(dict(recording=f'{subject}.npy', subject=subject, stimulus='story',
                    segment=str(segment), segment_start=start,
                    segment_end=start+ (12 if overlap else 6),
                    event_id=f'{segment}_{j}', onset=start+1+j*.7, word=['a', 'b'][j%2]))
    manifest = tmp_path/'manifest.tsv'
    with manifest.open('w') as f:
        w = csv.DictWriter(f, fieldnames=sorted(REQUIRED), delimiter='\t')
        w.writeheader(); w.writerows(rows)
    emb = tmp_path/'text.npz'
    np.savez(emb, words=np.array(['a', 'b']), embeddings=np.eye(2))
    return manifest, emb


def test_shared_stimulus_split_and_train_only_statistics(tmp_path):
    manifest, emb = fixture(tmp_path)
    out = tmp_path/'events.npz'
    report = prepare(manifest, emb, out, 100, candidates=250, min_train=1)
    assert report['actual_candidates'] == 2
    from neuromoment.data import EventArchive
    archive=EventArchive(out,'train',max_shift=.08)
    assert archive.start==-.2 and archive.end==1. and archive.n==120
    # Independent event counts are not doubled by listening subjects.
    assert sum(report['independent_event_counts']['train'].values())*2 == sum(report['counts']['train'].values())
    with np.load(out) as f:
        for event in np.unique(f['event_id']):
            assert len(set(f['split'][f['event_id'] == event])) == 1
    # Change only held-out continuous samples; normalization must remain unchanged.
    spans = report['split_groups']
    for subject in ['a', 'b']:
        x = np.load(tmp_path/f'{subject}.npy')
        for s in spans:
            if s['split'] != 'train':
                i = int(s['segment']); x[:, i*600:(i+1)*600] += 100
        np.save(tmp_path/f'{subject}.npy', x)
    revised = prepare(manifest, emb, tmp_path/'revised.npz', 100, min_train=1)
    assert report['normalization'] == revised['normalization']


def test_reject_overlapping_cross_split_segments(tmp_path):
    manifest, emb = fixture(tmp_path, overlap=True)
    with pytest.raises(ValueError, match='overlap'):
        prepare(manifest, emb, tmp_path/'bad.npz', 100, min_train=1)


def test_reject_cross_subject_event_group_mismatch(tmp_path):
    manifest, emb = fixture(tmp_path)
    with manifest.open() as f:
        reader = csv.DictReader(f, delimiter='\t'); names = reader.fieldnames; rows = list(reader)
    rows[-1]['segment'] = 'inconsistent_segment'
    with manifest.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=names, delimiter='\t'); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match='Shared event_id'):
        prepare(manifest, emb, tmp_path/'bad.npz', 100, min_train=1)
