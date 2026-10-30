import numpy as np
import pytest

from neuromoment.statistics import paired_bootstrap


def test_paired_cluster_bootstrap_and_identity_check(tmp_path):
    labels = np.tile(np.arange(12), 4)
    groups = np.repeat(np.array(['a', 'b', 'c', 'd']), 12)
    common = dict(groups=groups, labels=labels, event_ids=np.arange(48).astype(str),
                  subjects=np.repeat('s', 48), words=np.arange(12).astype(str))
    scores = np.eye(12)[labels]
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    np.savez(a, scores=scores, **common); np.savez(b, scores=scores, **common)
    report = paired_bootstrap(a, b, draws=20)
    assert report['difference'] == 0 and report['ci95_percentile'] == [0, 0]
    common['words'] = common['words'][::-1]
    np.savez(b, scores=scores, **common)
    with pytest.raises(ValueError, match='words'):
        paired_bootstrap(a, b, draws=20)
