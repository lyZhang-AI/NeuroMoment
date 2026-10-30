"""Matched frozen-text Ridge and frequency baselines; all tuning uses validation only."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import EventArchive, archive_fingerprint
from .metrics import retrieval_metrics


def features(ds, center, width=.3, delta=0., batch_size=256):
    lo=round((center-width/2-ds.start)*ds.sample_rate)
    hi=round((center+width/2-ds.start)*ds.sample_rate)
    if lo<0 or hi>ds.n or hi<=lo:raise ValueError('Lag window outside input')
    values=[]
    for context,_ in DataLoader(ds,batch_size=batch_size):
        x=ds.view(context,delta)
        values.append(F.adaptive_avg_pool1d(x[:,:,lo:hi],8).flatten(1))
    return torch.cat(values).double()


def ridge_statistics(x,target):
    mean=x.mean(0);sd=x.std(0,unbiased=False).clamp_min(1e-6)
    x=(x-mean)/sd;bias=target.mean(0)
    return x.T@x,x.T@(target-bias),mean,sd,bias


def from_statistics(stats,alpha):
    gram,rhs,mean,sd,bias=stats
    if alpha<=0:raise ValueError('Ridge alpha must be positive')
    weights=torch.linalg.solve(gram+alpha*torch.eye(gram.shape[0],dtype=gram.dtype),rhs)
    return weights,mean,sd,bias


def fit_ridge(x,target,alpha):
    return from_statistics(ridge_statistics(x,target),alpha)


def frequency_metrics(train,ds):
    counts=torch.bincount(train.y,minlength=len(train.words)).float()
    return retrieval_metrics(counts[None].expand(len(ds),-1),ds.y)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--lags',nargs='+',type=float,default=[0.,.15,.3,.45,.6,.75])
    p.add_argument('--alphas',nargs='+',type=float,default=[.1,1.,10.,100.])
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--shifts',nargs='+',type=float,default=[0.])
    a=p.parse_args();torch.set_num_threads(a.threads)
    if min(a.alphas)<=0:raise ValueError('Ridge alphas must be positive')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'ridge.pt').exists():raise FileExistsError('Choose a fresh output directory')
    train,val=[EventArchive(a.data,s,max_shift=0) for s in ['train','val']]
    embeddings=F.normalize(train.embeddings.double(),dim=-1)
    best=-float('inf');chosen=None;validation=[]
    for lag in a.lags:
        x=features(train,lag);xv=features(val,lag)
        stats=ridge_statistics(x,embeddings[train.y]);del x
        for alpha in a.alphas:
            fit=from_statistics(stats,alpha);w,mean,sd,bias=fit
            scores=F.normalize(((xv-mean)/sd)@w+bias,dim=-1)@embeddings.T
            metrics=retrieval_metrics(scores,val.y)
            validation.append({'lag':lag,'alpha':alpha,'metrics':metrics})
            selection=metrics['balanced_top10']+1e-6*metrics['balanced_mrr']
            if selection>best:best,chosen=selection,(lag,alpha,fit)
        print(json.dumps({'validated_lag':lag,'best_selection':best}),flush=True)
    lag,alpha,(w,mean,sd,bias)=chosen
    # Test is loaded only AFTER hyperparameters have been locked on validation.
    test=EventArchive(a.data,'test',max_shift=0)
    report={'lag':lag,'alpha':alpha,'width':.3,'data_sha256':archive_fingerprint(a.data),
        'synthetic':train.metadata.get('synthetic','unknown'),'selection_split':'val',
        'validation_grid':validation,'metrics':{},'frequency_baseline':frequency_metrics(train,test)}
    for delta in a.shifts:
        xt=features(test,lag,delta=delta)
        scores=F.normalize(((xt-mean)/sd)@w+bias,dim=-1)@embeddings.T
        report['metrics'][str(delta)]=retrieval_metrics(scores,test.y)
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    torch.save({'weights':w,'mean':mean,'sd':sd,'bias':bias,'lag':lag,'alpha':alpha,
                'words':train.words,'embeddings':embeddings,'data_sha256':report['data_sha256']},out/'ridge.pt')
    print(json.dumps({'chosen_lag':lag,'alpha':alpha,'test_clean':report['metrics']['0.0']}),flush=True)


if __name__=='__main__':main()
