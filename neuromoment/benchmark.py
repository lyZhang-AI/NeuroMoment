"""Measure warm inference latency for a locked EEG checkpoint; excludes disk IO/text encoding."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import EventArchive, archive_fingerprint
from .model import NeuroMoment, ModelConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--repeats',type=int,default=30)
    a=p.parse_args()
    if min(a.batch_size,a.repeats,a.threads)<1:raise ValueError('Benchmark counts must be positive')
    torch.set_num_threads(a.threads)
    checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=True)
    if checkpoint['data_sha256']!=archive_fingerprint(a.data):raise ValueError('Checkpoint/dataset mismatch')
    c=ModelConfig(**checkpoint['config']);model=NeuroMoment(c).to(a.device).eval()
    model.load_state_dict(checkpoint['model'])
    ds=EventArchive(a.data,'test',start=c.start,end=c.end,max_shift=0)
    x=torch.stack([ds[i%len(ds)][0] for i in range(a.batch_size)]).to(a.device)
    x=ds.view(x);words=F.normalize(ds.embeddings.to(a.device),dim=-1)
    def sync():
        if str(a.device).startswith('cuda'):torch.cuda.synchronize()
    times=[]
    with torch.inference_mode():
        for _ in range(5):model(x)['embedding']@words.T
        sync()
        for _ in range(a.repeats):
            start=time.perf_counter();model(x)['embedding']@words.T;sync()
            times.append((time.perf_counter()-start)*1000)
    report={'checkpoint':a.checkpoint,'device':a.device,'machine':platform.machine(),
        'torch_version':str(torch.__version__),'threads':a.threads,'batch_size':a.batch_size,
        'input_shape':list(x.shape),'repeats':a.repeats,'registered_parameters':sum(p.numel() for p in model.parameters()),
        'median_batch_ms':float(np.median(times)),'p95_batch_ms':float(np.quantile(times,.95)),
        'median_ms_per_event':float(np.median(times)/a.batch_size),
        'scope':'warm model + candidate cosine scores; excludes disk IO, preprocessing and frozen text encoding',
        'flops_measured':False}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
