"""Prepare subject-normalized events from continuous .npy recordings and a TSV manifest."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REQUIRED = {"recording", "subject", "stimulus", "segment", "segment_start", "segment_end",
            "event_id", "onset", "word"}


def prepare(manifest, embedding_file, out, sample_rate, candidates=250, min_train=1,
            context_start=-0.28, context_end=1.08, guard=0.0, seed=7, epoch_start=-.2, epoch_end=1.):
    if sample_rate <= 0 or context_end <= context_start or guard < 0 or candidates < 2 or min_train < 1:
        raise ValueError("Invalid preparation settings")
    if not context_start <= epoch_start < epoch_end <= context_end:
        raise ValueError("Epoch must lie inside stored context")
    manifest = Path(manifest)
    with manifest.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires {sorted(REQUIRED)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    # Shared stimulus segments, not subject rows, determine the split.
    group = lambda r: (r["stimulus"], r["segment"])
    groups = sorted({group(r) for r in rows}, key=lambda k: hashlib.sha256(
        f"{seed}:{k}".encode()).hexdigest())
    if len(groups) < 3:
        raise ValueError("Need at least three independent stimulus segments")
    n_test = max(1, round(len(groups) * 0.10))
    n_val = max(1, round(len(groups) * 0.10))
    mapping = {key: "test" if i < n_test else "val" if i < n_test+n_val else "train"
               for i, key in enumerate(groups)}
    recordings, spans, seen_events = {}, defaultdict(dict), set()
    shared_identity = {}
    retained, dropped = [], Counter()
    for r in rows:
        path = str((manifest.parent / r["recording"]).resolve())
        if path not in recordings:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if arr.ndim != 2:
                raise ValueError("Continuous recording must have shape [channels, samples]")
            recordings[path] = arr
        onset, left, right = map(float, [r["onset"], r["segment_start"], r["segment_end"]])
        if not np.isfinite([onset, left, right]).all() or left >= right or not left <= onset < right:
            raise ValueError("Invalid segment/event times")
        old = spans[path].get(group(r))
        if old is not None and old != (left, right):
            raise ValueError("Inconsistent segment boundaries in one recording")
        spans[path][group(r)] = (left, right)
        event_key = (r["subject"], r["stimulus"], r["event_id"])
        if event_key in seen_events:
            raise ValueError("Duplicate subject/stimulus/event_id")
        seen_events.add(event_key)
        identity = (r["stimulus"], r["event_id"])
        signature = (group(r), r["word"])
        if identity in shared_identity and shared_identity[identity] != signature:
            raise ValueError("Shared event_id has inconsistent segment or word across subjects")
        shared_identity[identity] = signature
        if onset + context_start < left + guard or onset + context_end > right - guard:
            dropped["segment_boundary"] += 1
            continue
        a = round((onset + context_start) * sample_rate)
        n = round((context_end - context_start) * sample_rate)
        if a < 0 or a+n > recordings[path].shape[-1]:
            dropped["recording_boundary"] += 1
            continue
        r = dict(r, path=path, a=a, n=n, split=mapping[group(r)])
        retained.append(r)
    # No overlapping source spans assigned to different splits.
    for path, intervals in spans.items():
        items = list(intervals.items())
        for i, (g1, (a, b)) in enumerate(items):
            for g2, (c, d) in items[i+1:]:
                if mapping[g1] != mapping[g2] and max(a, c) < min(b, d):
                    raise ValueError(f"Cross-split recording overlap in {path}")
    occurrences = defaultdict(set)
    event_words = {}
    for r in retained:
        key = (r["stimulus"], r["event_id"])
        if key in event_words and event_words[key] != r["word"]:
            raise ValueError("Shared event_id has inconsistent word identity")
        event_words[key] = r["word"]
        if r["split"] == "train":
            occurrences[r["word"]].add(key)
    # Vocabulary selected on training events ONLY, before looking at held-out coverage.
    words = sorted([w for w, events in occurrences.items() if len(events) >= min_train],
                   key=lambda w: (-len(occurrences[w]), w))[:candidates]
    if len(words) < 2:
        raise ValueError("Insufficient repeated training words")
    with np.load(embedding_file, allow_pickle=False) as f:
        ew, em = f["words"].tolist(), f["embeddings"]
        if len(set(ew)) != len(ew) or em.ndim != 2 or len(ew) != len(em):
            raise ValueError("Invalid embedding table")
        lookup = dict(zip(ew, em))
    missing = set(words) - lookup.keys()
    if missing:
        raise ValueError(f"Missing frozen text embeddings: {sorted(missing)}")
    labels = {w: i for i, w in enumerate(words)}
    selected = [r for r in retained if r["word"] in labels]
    # Unique continuous training spans avoid overweighting samples shared by event windows.
    subject_spans = defaultdict(set)
    for r in retained:
        if r["split"] == "train":
            lo, hi = spans[r["path"]][group(r)]
            subject_spans[r["subject"]].add((r["path"], round((lo+guard)*sample_rate),
                                           round((hi-guard)*sample_rate)))
    stats = {}
    for subject, intervals in subject_spans.items():
        total = sq = None
        count = 0
        for path, a, b in sorted(intervals):
            arr = np.asarray(recordings[path][:, a:b], dtype=np.float64)
            if not np.isfinite(arr).all() or not arr.shape[-1]:
                raise ValueError("Invalid training normalization span")
            s, s2 = arr.sum(1), (arr * arr).sum(1)
            total = s if total is None else total+s
            sq = s2 if sq is None else sq+s2
            count += arr.shape[1]
        mean = total/count
        sd = np.maximum(np.sqrt(np.maximum(sq/count - mean**2, 0)), 1e-12)
        stats[subject] = (mean, sd)
    xs, ys, splits, subjects, stimuli, event_ids = [], [], [], [], [], []
    coverage = {s: Counter() for s in ["train", "val", "test"]}
    unique_coverage = {s: defaultdict(set) for s in coverage}
    for r in selected:
        if r["subject"] not in stats:
            raise ValueError("No training normalization for subject; held-out-subject needs separate policy")
        mean, sd = stats[r["subject"]]
        x = np.asarray(recordings[r["path"]][:, r["a"]:r["a"]+r["n"]], dtype=np.float64)
        xs.append(((x-mean[:, None])/sd[:, None]).astype("float32"))
        ys.append(labels[r["word"]]); splits.append(r["split"])
        subjects.append(r["subject"]); stimuli.append(r["stimulus"]); event_ids.append(r["event_id"])
        coverage[r["split"]][r["word"]] += 1
        unique_coverage[r["split"]][r["word"]].add((r["stimulus"], r["event_id"]))
    if any(not coverage[s] for s in coverage):
        raise ValueError("Empty candidate evaluation split; revise preregistered task, not test-selected vocabulary")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=np.stack(xs), y=np.array(ys), split=np.array(splits),
        words=np.array(words), embeddings=np.stack([lookup[w] for w in words]),
        sample_rate=sample_rate, context_start=context_start, subject=np.array(subjects),
        stimulus=np.array(stimuli), event_id=np.array(event_ids),
        segment=np.array([r["segment"] for r in selected]))
    report = {"synthetic": False, "requested_candidates": candidates, "actual_candidates": len(words),
        "sample_rate": sample_rate, "context_start": context_start, "context_end": context_end,
        "epoch_start": epoch_start, "epoch_end": epoch_end,
        "guard": guard, "seed": seed, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "embedding_sha256": hashlib.sha256(Path(embedding_file).read_bytes()).hexdigest(),
        "split_groups": [{"stimulus": g[0], "segment": g[1], "split": mapping[g]} for g in groups],
        "dropped": dict(dropped), "counts": {s: dict(v) for s, v in coverage.items()},
        "independent_event_counts": {s: {w: len(v.get(w, set())) for w in words}
                                     for s, v in unique_coverage.items()},
        "full_candidate_coverage": {s: len(v) == len(words) for s, v in coverage.items()},
        "normalization": {s: {"mean": m.tolist(), "std": d.tolist()} for s, (m, d) in stats.items()}}
    Path(str(out)+".json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--embeddings", required=True)
    p.add_argument("--out", required=True); p.add_argument("--sample-rate", type=float, required=True)
    p.add_argument("--candidates", type=int, default=250); p.add_argument("--min-train", type=int, default=1)
    p.add_argument("--guard", type=float, default=0); p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epoch-start", type=float, default=-.2);p.add_argument("--epoch-end", type=float, default=1.)
    a = p.parse_args()
    print(json.dumps(prepare(a.manifest, a.embeddings, a.out, a.sample_rate,
                            a.candidates, a.min_train, guard=a.guard, seed=a.seed,epoch_start=a.epoch_start,epoch_end=a.epoch_end), indent=2))


if __name__ == "__main__":
    main()
