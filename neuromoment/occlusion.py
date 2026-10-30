"""Selected-versus-random Top-K masking with exactly matched temporal union coverage."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import EventArchive, archive_fingerprint
from .model import ModelConfig, NeuroMoment
from .metrics import retrieval_metrics
from .text import sha256


def coverage_catalog(model, samples):
    times=model.coordinates(samples,torch.device('cpu'),torch.float32)
    width=max(1,round(model.config.local_width*model.config.sample_rate))
    centers=torch.round((times-model.config.start)*model.config.sample_rate).long().numpy()
    legal=[i for i,c in enumerate(centers) if c-width//2>=0 and c-width//2+width<=samples]
    masks=[];by_coverage={};by_indices={}
    if model.config.moments>4:raise ValueError('Exact matched coverage enumeration is intended for K<=4')
    for indices in itertools.combinations(legal,model.config.moments):
        if any(abs(float(times[a]-times[b]))<model.config.nms_interval for a,b in itertools.combinations(indices,2)):continue
        mask=np.zeros(samples,dtype=bool)
        for index in indices:
            lo=centers[index]-width//2;mask[lo:lo+width]=True
        key=len(masks);masks.append(mask);by_indices[indices]=key
        by_coverage.setdefault(int(mask.sum()),[]).append(key)
    if not masks:raise ValueError('No legal Top-K mask configurations')
    return np.stack(masks),by_coverage,by_indices


def matched_masks(indices,catalog,rng):
    masks,by_coverage,by_indices=catalog
    selected=[];random=[]
    for row in indices:
        key=by_indices[tuple(sorted(map(int,row)))]
        choices=by_coverage[int(masks[key].sum())]
        alternatives=[i for i in choices if not np.array_equal(masks[i],masks[key])]
        if not alternatives:raise ValueError('No distinct random mask with equal coverage')
        selected.append(masks[key]);random.append(masks[int(rng.choice(alternatives))])
    selected=np.stack(selected);random=np.stack(random)
    assert np.array_equal(selected.sum(-1),random.sum(-1))
    return selected,random


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=64);p.add_argument('--seed',type=int,default=107)
    a=p.parse_args();torch.set_num_threads(2)
    ckpt=torch.load(a.checkpoint,map_location=a.device,weights_only=True)
    if archive_fingerprint(a.data)!=ckpt['data_sha256']:raise ValueError('Checkpoint/data mismatch')
    config=ModelConfig(**ckpt['config']);model=NeuroMoment(config).to(a.device).eval();model.load_state_dict(ckpt['model'])
    ds=EventArchive(a.data,'test',start=config.start,end=config.end,max_shift=0)
    catalog=coverage_catalog(model,ds.n);rng=np.random.default_rng(a.seed)
    embeds=F.normalize(ds.embeddings.to(a.device),dim=-1)
    scores={k:[] for k in ['clean','selected','random']};coverage=[]
    with torch.inference_mode():
        for context,_ in DataLoader(ds,batch_size=a.batch_size):
            x=ds.view(context.to(a.device));base=model(x)
            indices,_,_=model.select(base['alpha'],base['times'],ds.n)
            selected,random=matched_masks(indices.cpu().numpy(),catalog,rng)
            coverage.append(selected.sum(-1))
            scores['clean'].append((base['embedding']@embeds.T).cpu())
            for key,mask in [('selected',selected),('random',random)]:
                masked=x.masked_fill(torch.as_tensor(mask,device=a.device)[:,None],0.)
                scores[key].append((model(masked)['embedding']@embeds.T).cpu())
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);results={}
    for name,batches in scores.items():
        values=torch.cat(batches);results[name]=retrieval_metrics(values,ds.y)
        meta={'scores':values.numpy(),'labels':ds.y.numpy(),'words':np.array(ds.words),
              'subjects':ds.subjects,'groups':ds.groups,'event_ids':ds.event_ids,'masked_samples':np.zeros(len(ds),dtype=int) if name=='clean' else np.concatenate(coverage)}
        np.savez_compressed(out/f'{name}.npz',**{k:v for k,v in meta.items() if v is not None})
    report={'scope':'Fixed-weight input sensitivity, not physiological localization. No results are inserted from the manuscript.',
            'protocol':{'moments':config.moments,'window_seconds':config.local_width,'nms_interval':config.nms_interval,
                        'replacement':'zero in standardized EEG','matched_union_samples_per_trial':True,'random_seed':a.seed,
                        'random_rule':'Uniform distinct legal NMS configuration conditional on the selected union coverage'},
            'data_sha256':ckpt['data_sha256'],'checkpoint_sha256':sha256(a.checkpoint),'results':results}
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
