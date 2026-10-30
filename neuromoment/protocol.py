"""Shared, deterministic stimulus-level protocol. No split or vocabulary selection on test labels."""
import hashlib
import re
import unicodedata
from collections import defaultdict


def normalize_word(text):
    text = unicodedata.normalize('NFKC', text).replace('’', "'").replace('‘', "'").lower().strip()
    text = text.strip('.,!?;:\"()[]{}“”„…')
    if text in {'sil', 'sp', '<unk>', '<sil>', ''}:
        return None
    return text if re.fullmatch(r"[a-z]+(?:['-][a-z]+)*", text) else None


def assign_splits(groups, seed=7, split_counts=None):
    groups = sorted(set(groups), key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    if len(groups) < 3:
        raise ValueError('Need at least three stimulus segments')
    if split_counts is not None:
        if len(split_counts) != 3 or min(split_counts) < 1 or sum(split_counts) != len(groups):
            raise ValueError("Split counts must be positive train/val/test counts summing to all run groups")
        _, n_val, n_test = split_counts
    else:
        n_test = max(1, round(len(groups)*.10)); n_val = max(1, round(len(groups)*.10))
    return {g: 'test' if i < n_test else 'val' if i < n_test+n_val else 'train'
            for i, g in enumerate(groups)}


def vocabulary_report(events, mapping, candidates=250, min_train=1):
    counts = {s: defaultdict(set) for s in ['train', 'val', 'test']}
    observations = {s: defaultdict(int) for s in counts}
    for e in events:
        s = mapping[(e['stimulus'], e['segment'])]
        counts[s][e['word']].add((e['stimulus'], e['event_id']))
        observations[s][e['word']] += 1
    eligible = [w for w, ids in counts['train'].items() if len(ids) >= min_train]
    words = sorted(eligible, key=lambda w: (-len(counts['train'][w]), w))[:candidates]
    return {'words': words, 'requested_candidates': candidates, 'actual_candidates': len(words),
            'eligible_training_words': len(eligible), 'min_train_independent_events': min_train,
            'counts': {s: {w: observations[s][w] for w in words} for s in counts},
            'independent_event_counts': {s: {w: len(counts[s][w]) for w in words} for s in counts},
            'full_candidate_coverage': {s: all(counts[s][w] for w in words) for s in counts},
            'missing_words': {s: [w for w in words if not counts[s][w]] for s in counts},
            'coverage': {s: sum(bool(counts[s][w]) for w in words)/len(words) if words else 0 for s in counts}}
