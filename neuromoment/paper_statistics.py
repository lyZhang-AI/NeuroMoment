"""Five-split retrieval summaries and subject-paired permutation tests with Holm correction."""
import argparse
import json
from pathlib import Path
import numpy as np
from .text import sha256

METRICS=('top10','top5','top1')


def ranks(scores, labels):
    order=np.argsort(-scores,axis=1,kind='stable')
    return np.argmax(order==labels[:,None],axis=1)+1


def macro_hits(rank, labels, subject=None):
    mask=np.ones(len(labels),dtype=bool) if subject is None else subject
    classes=np.unique(labels[mask])
    return np.array([np.mean([np.mean(rank[mask & (labels==c)]<=k) for c in classes]) for k in [10,5,1]])


def holm(pvalues):
    values=np.asarray(pvalues,dtype=float)
    order=np.argsort(values)
    adjusted=np.minimum(1,np.maximum.accumulate(values[order]*np.arange(len(values),0,-1)))
    result=np.empty_like(values);result[order]=adjusted
    return result


def paired_permutation(differences, draws=100000, seed=7):
    """Subjects x metrics, after averaging each subject over matched splits."""
    differences=np.asarray(differences,dtype=float)
    if differences.ndim!=2 or not np.isfinite(differences).all():raise ValueError('Invalid paired differences')
    n=len(differences)
    if n<2:raise ValueError('At least two paired subjects required')
    observed=np.abs(differences.mean(0));counts=np.zeros(differences.shape[1],dtype='int64')
    exact=n<=20;total=2**n if exact else draws
    if total<1:raise ValueError('Permutation count must be positive')
    rng=np.random.default_rng(seed)
    for begin in range(0,total,4096):
        size=min(4096,total-begin)
        if exact:
            signs=2*((np.arange(begin,begin+size,dtype='uint64')[:,None]>>np.arange(n,dtype='uint64')[None])&1).astype('int8')-1
        else:signs=rng.choice([-1,1],(size,n))
        permuted=np.abs(signs@differences/n)
        counts+=(permuted>=observed[None]-1e-12).sum(0)
    p=counts/total if exact else (counts+1)/(total+1)
    return p,{'exact':exact,'permutations':total,'subjects':n,'two_sided':True}


def compare(first, second, expected_subjects=19, expected_splits=5):
    if len(first)!=expected_splits or len(second)!=expected_splits:raise ValueError('Need five matched split artifacts per method')
    subject_set=None;differences=[];sources=[]
    for pa,pb in zip(first,second):
        with np.load(pa,allow_pickle=False) as a,np.load(pb,allow_pickle=False) as b:
            for key in ['labels','subjects','event_ids','groups','words']:
                if key not in a or key not in b or not np.array_equal(a[key],b[key]):raise ValueError('Unmatched paired event identities or candidates')
            subjects=np.unique(a['subjects'])
            if len(subjects)!=expected_subjects:raise ValueError('Unexpected subject count')
            if subject_set is None:subject_set=subjects
            elif not np.array_equal(subjects,subject_set):raise ValueError('Subject sets differ across splits')
            ra=ranks(a['scores'],a['labels']);rb=ranks(b['scores'],b['labels'])
            differences.append(np.stack([macro_hits(ra,a['labels'],a['subjects']==s)-macro_hits(rb,a['labels'],a['subjects']==s) for s in subjects]))
        sources.append({'first_sha256':sha256(pa),'second_sha256':sha256(pb)})
    mean_subject_difference=np.mean(differences,axis=0)
    p,details=paired_permutation(mean_subject_difference)
    return {'metrics':list(METRICS),'subject_mean_difference':mean_subject_difference.mean(0).tolist(),
            'raw_p':p.tolist(),'holm_adjusted_p':holm(p).tolist(),
            'subject_differences_after_split_average':mean_subject_difference.tolist(),
            'subject_ids':subject_set.tolist(),'matched_splits':expected_splits,'test':details,'sources':sources}


def summarize(project, seeds, models):
    project=Path(project);rows=[];missing=[]
    for model in models:
        values=[]
        for seed in seeds:
            root=project/f'runs/paper/split_{seed}/{model}'
            path=root/'test/metrics.json'
            if not path.exists():missing.append(str(path.relative_to(project)));continue
            report=json.loads(path.read_text())
            if report['checkpoint_sha256']!=sha256(root/'best.pt'):raise ValueError('Stale checkpoint metrics')
            m=report['results']['0.0']
            values.append([m[f'balanced_{key}']*100 for key in METRICS])
        if len(values)==len(seeds):
            values=np.asarray(values)
            rows.append({'model':model,'splits':len(values),'mean_percent':dict(zip(METRICS,values.mean(0).tolist())),
                         'std_percent':dict(zip(METRICS,values.std(0,ddof=1).tolist()))})
    controls=[]
    for condition in ['shuffled_prior','random_masking','selected_masking']:
        values=[]
        for seed in seeds:
            if condition=='shuffled_prior':
                path=project/f'runs/paper/split_{seed}/neuromoment/shuffled_prior/metrics.json'
            else:path=project/f'reports/paper/split_{seed}/occlusion/metrics.json'
            if not path.exists():continue
            report=json.loads(path.read_text())
            checkpoint=project/f'runs/paper/split_{seed}/neuromoment/best.pt'
            if report['checkpoint_sha256']!=sha256(checkpoint):raise ValueError('Stale control metrics')
            key='0.0' if condition=='shuffled_prior' else 'random' if condition=='random_masking' else 'selected'
            m=report['results'][key]
            values.append([m[f'balanced_{k}']*100 for k in METRICS])
        if len(values)==len(seeds):
            v=np.asarray(values)
            controls.append({'condition':condition,'splits':len(values),
                             'mean_percent':dict(zip(METRICS,v.mean(0).tolist())),
                             'std_percent':dict(zip(METRICS,v.std(0,ddof=1).tolist()))})
    out=project/'reports/paper';out.mkdir(parents=True,exist_ok=True)
    result={'scope':'Computed only from complete sets of actual five-split evaluations; no manuscript numbers inserted.',
            'split_seeds':seeds,'rows':rows,'controls':controls,'missing_artifacts':missing}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# Five-split retrieval summary','','| Method | Top-10 (%) | Top-5 (%) | Top-1 (%) |','|---|---:|---:|---:|']
    for row in rows:
        text.append('| '+row['model']+' | '+' | '.join(f"{row['mean_percent'][k]:.2f} ± {row['std_percent'][k]:.2f}" for k in METRICS)+' |')
    if controls:
        text+=['','## Frozen-weight controls','','| Condition | Top-10 (%) | Top-5 (%) | Top-1 (%) |','|---|---:|---:|---:|']
        for row in controls:
            text.append('| '+row['condition']+' | '+' | '.join(f"{row['mean_percent'][k]:.2f} ± {row['std_percent'][k]:.2f}" for k in METRICS)+' |')
    if missing:text+=['',f'Incomplete evaluation: {len(missing)} artifacts are missing. Incomplete model rows are not averaged.']
    (out/'summary.md').write_text('\n'.join(text)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first',nargs='+',required=True);p.add_argument('--second',nargs='+',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args();result=compare(a.first,a.second)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
