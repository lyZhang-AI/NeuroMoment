"""Paired stimulus-segment bootstrap; does not treat subjects hearing the same event as independent."""
import argparse
import json

import numpy as np


def paired_bootstrap(first, second, draws=2000, seed=7):
    if draws < 1:
        raise ValueError("draws must be positive")
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        for field in ['groups', 'labels', 'event_ids', 'subjects', 'words']:
            if field not in a or field not in b or not np.array_equal(a[field], b[field]):
                raise ValueError(f"Missing or unmatched paired metadata: {field}")
        groups, labels = a['groups'].copy(), a['labels'].copy()
        sa, sb = a['scores'].copy(), b['scores'].copy()
    if sa.shape != sb.shape:
        raise ValueError("Candidate score shapes differ")
    clusters = np.unique(groups)
    if len(clusters) < 2:
        raise ValueError("Need at least two stimulus segments; preferably many more")
    # Per-event hits followed by per-class aggregation in each cluster resample.
    def hits(scores):
        order = np.argsort(-scores, axis=1, kind='stable')
        return (order[:, :min(10, scores.shape[1])] == labels[:, None]).any(1)
    ha, hb = hits(sa), hits(sb)
    classes = np.unique(labels)
    indices = {g: np.flatnonzero(groups == g) for g in clusters}
    def macro_difference(idx):
        # Skip replicates missing any originally evaluated class; report acceptance.
        if len(np.unique(labels[idx])) != len(classes):
            return None
        return np.mean([ha[idx][labels[idx] == c].mean()-hb[idx][labels[idx] == c].mean()
                        for c in classes])
    observed = macro_difference(np.arange(len(labels)))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        idx = np.concatenate([indices[g] for g in rng.choice(clusters, len(clusters), replace=True)])
        value = macro_difference(idx)
        if value is not None:
            estimates.append(value)
    if len(estimates) < max(1, draws//2):
        raise ValueError("Sparse class coverage makes macro bootstrap unreliable; change task design")
    return {"metric": "balanced_top10_difference_first_minus_second", "difference": observed,
            "ci95_percentile": np.quantile(estimates, [.025, .975]).tolist(),
            "clusters": len(clusters), "draws_requested": draws, "draws_accepted": len(estimates),
            "seed": seed, "warning": "Stimulus-cluster uncertainty only; report training seeds separately"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("first"); p.add_argument("second")
    p.add_argument("--draws", type=int, default=2000); p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    print(json.dumps(paired_bootstrap(a.first, a.second, a.draws, a.seed), indent=2))


if __name__ == '__main__':
    main()
