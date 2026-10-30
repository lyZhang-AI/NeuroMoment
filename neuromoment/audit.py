"""Audit real annotations and training-selected candidate pools before EEG preprocessing."""
import argparse
import json
from pathlib import Path

from .bids import catalog, usable_events
from .protocol import assign_splits, vocabulary_report


def audit(root, dataset, out, seed=7, guard=2., subjects=None, include_excluded=False, min_train=1, split_counts=None, context_start=-.28, context_end=1.08):
    records, metadata = catalog(root, dataset, subjects, include_excluded)
    events, dropped = usable_events(records, guard,context_start,context_end)
    mapping = assign_splits([(r['stimulus'],r['segment']) for r in records], seed,split_counts)
    pools = {str(v): vocabulary_report(events, mapping, v, min_train) for v in [50,100,250]}
    report = {**metadata, 'synthetic': False, 'seed': seed, 'guard': guard,
        'source_spans':len(records), 'event_observations':len(events),
        'independent_events':len({(e['stimulus'],e['event_id']) for e in events}),
        'boundary_dropped':dropped, 'candidate_pools':pools,
        'split_groups':[{'stimulus':g[0],'segment':g[1],'split':s} for g,s in mapping.items()],
        'clock_residual_max_seconds':max(r.get('clock_residual_max_seconds',0) for r in records)}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in report.items() if k not in {'candidate_pools','split_groups'}},indent=2))
    for v,pool in pools.items():
        print(v, json.dumps({k:pool[k] for k in ['actual_candidates','eligible_training_words','coverage','full_candidate_coverage']}))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--dataset',choices=['ds004408','brennan'],required=True)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--guard',type=float,default=2.);p.add_argument('--subjects',nargs='+')
    p.add_argument('--min-train',type=int,default=1)
    p.add_argument('--include-excluded',action='store_true')
    p.add_argument('--split-counts',type=int,nargs=3)
    p.add_argument('--context-start',type=float,default=-.28);p.add_argument('--context-end',type=float,default=1.08)
    a=p.parse_args();audit(a.root,a.dataset,a.out,a.seed,a.guard,a.subjects,a.include_excluded,a.min_train,a.split_counts,a.context_start,a.context_end)


if __name__=='__main__':main()
