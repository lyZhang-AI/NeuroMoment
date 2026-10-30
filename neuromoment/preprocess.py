"""Split-isolated finite-support filtering and memory-mapped real EEG dataset construction."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
from scipy.signal import firwin, resample_poly

from .bids import catalog, usable_events
from .protocol import assign_splits, vocabulary_report
from .text import sha256


def filter_spec(sfreq, target_rate, low=1., high=30.):
    if not 80 <= target_rate < sfreq:
        raise ValueError('Target sample rate must be >=80 Hz and below source rate')
    upper= min(high,target_rate/2-10.)
    if not 0 < low < upper:raise ValueError("Invalid bandpass")
    kwargs=dict(l_freq=low,h_freq=upper,l_trans_bandwidth=1.,h_trans_bandwidth=10.,
                filter_length='auto',method='fir',phase='zero',fir_window='hamming',fir_design='firwin')
    taps=mne.filter.create_filter(None,sfreq,verbose=False,**kwargs)
    fraction=Fraction(float(target_rate)/float(sfreq)).limit_denominator(10000)
    if abs(float(fraction)*sfreq-target_rate)>1e-6:
        raise ValueError('Cannot represent resampling ratio accurately with a bounded FIR')
    up,down=fraction.numerator,fraction.denominator
    half=10*max(up,down)
    antialias=firwin(2*half+1,1/max(up,down),window=('kaiser',5.))
    support=(len(taps)-1)/(2*sfreq)+half/(up*sfreq)
    return kwargs,up,down,antialias,float(support),len(taps)


def transform_record(record, channels, out, dataset, target_rate, guard, low=1., high=30.):
    key=f"{record['subject']}_{record['segment']}"
    relative=f'recordings/{key}.npy';path=out/relative;meta_path=out/'recordings'/f'{key}.json'
    source_binary=Path(record['header']).with_suffix('.eeg')
    source_stat=source_binary.stat()
    signature=hashlib.sha256(json.dumps({'record':record,'channels':channels,'rate':target_rate,
        'source_binary_size':source_stat.st_size,'source_binary_mtime_ns':source_stat.st_mtime_ns,
        'source_header_sha256':sha256(record['header']),
        'guard':guard,'bandpass_hz':[low,high],'pipeline':'finite_fir_bandpass_standard_montage_reference_v2'},sort_keys=True).encode()).hexdigest()
    if path.exists() and meta_path.exists():
        cached=json.loads(meta_path.read_text())
        if cached['signature'] != signature or sha256(path) != cached['sha256']:
            raise ValueError('Stale or corrupted preprocessing cache; use a fresh output directory')
        return cached
    raw=mne.io.read_raw_brainvision(record['header'],preload=False,verbose='ERROR')
    sfreq=float(raw.info['sfreq'])
    kwargs,up,down,antialias,support,ntaps=filter_spec(sfreq,target_rate,low,high)
    if guard < support:
        raise ValueError(f'guard={guard} is below finite preprocessing half-support={support}')
    lo=round(record['start']*sfreq);hi=min(round(record['end']*sfreq),raw.n_times)
    data=raw.get_data(picks=channels,start=lo,stop=hi)
    raw.close()
    bad=record['bad_channels'];bad_indices=[channels.index(ch) for ch in bad]
    good=np.array([i for i in range(len(channels)) if i not in bad_indices])
    if len(bad) > len(channels)*.2:
        raise ValueError(f'Too many BIDS bad channels ({len(bad)}); explicit QC decision needed')
    if not np.isfinite(data[good]).all():
        raise ValueError('Non-finite EEG in channels not marked bad')
    raw_std=float(np.median(data[good].std(1)))
    raw_abs=float(np.median(np.abs(data[good])))
    # Deterministic local DC removal for numeric stability; fitted subject scaling is train-only below.
    data[bad_indices]=0
    data -= data.mean(1,keepdims=True)
    data=mne.filter.filter_data(data,sfreq,verbose=False,**kwargs)
    if bad:
        montage='biosemi128' if dataset=='ds004408' else 'easycap-M10'
        interpolated=mne.io.RawArray(data,mne.create_info(channels,sfreq,'eeg'),verbose='ERROR')
        interpolated.set_montage(mne.channels.make_standard_montage(montage),on_missing='raise',verbose=False)
        interpolated.info['bads']=bad
        interpolated.interpolate_bads(reset_bads=True,origin=(0.,0.,.04),verbose=False)
        data=interpolated.get_data();interpolated.close()
    data -= data.mean(0,keepdims=True)
    data=resample_poly(data,up,down,axis=-1,window=antialias).astype('float32')
    if not np.isfinite(data).all():
        raise ValueError('Non-finite filtered EEG')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:np.save(f,data)
    temporary.replace(path)
    info={'path':relative,'subject':record['subject'],'stimulus':record['stimulus'],'segment':record['segment'],
        'source_header':record['header'],'source_start_sample':lo,'source_end_sample':hi,
        'source_start_seconds':lo/sfreq,'source_sample_rate':sfreq,'sample_rate':target_rate,
        'channels':channels,'bad_channels_interpolated':bad,'shape':list(data.shape),
        'raw_median_channel_std_as_mne_volts':raw_std,'raw_median_absolute_as_mne_volts':raw_abs,
        'amplitude_unit_flag':raw_std>.01,'filtered_channel_std':data.std(1).tolist(),
        'fir_taps':ntaps,'preprocessing_half_support_seconds':support,'signature':signature,
        'sha256':sha256(path),'size_bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
    meta_path.write_text(json.dumps(info,indent=2))
    return info


def build(root,dataset,embeddings,out,seed=7,candidates=250,min_train=1,sample_rate=128.,
          guard=2.,subjects=None,workers=4,include_excluded=False,
          split_counts=None, low=1., high=30., context_start=-.28, context_end=1.08,
          epoch_start=-.2, epoch_end=1.):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'dataset.json').exists():
        raise FileExistsError('Dataset already finalized; choose a fresh output directory')
    records,metadata=catalog(root,dataset,subjects,include_excluded)
    mapping=assign_splits([(r['stimulus'],r['segment']) for r in records],seed,split_counts)
    events,dropped=usable_events(records,guard,context_start,context_end)
    report=vocabulary_report(events,mapping,candidates,min_train)
    words=report['words']
    if len(words)<2:
        raise ValueError('Insufficient candidate words')
    with np.load(embeddings,allow_pickle=False) as f:
        lookup=dict(zip(f['words'].tolist(),f['embeddings']))
    if set(words)-lookup.keys():
        raise ValueError(f'Missing frozen embeddings for {set(words)-lookup.keys()}')
    table=np.stack([lookup[w] for w in words]).astype('float32')
    if not np.isfinite(table).all() or (np.linalg.norm(table,axis=-1)==0).any():
        raise ValueError('Invalid frozen embeddings')
    print(json.dumps({'dataset':dataset,'records':len(records),'subjects':len(metadata['subjects']),
        'candidates':len(words),'coverage':report['coverage'],'stage':'preprocess'}),flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator=pool.map(lambda r:transform_record(r,metadata['channels'],out,dataset,sample_rate,guard,low,high),records)
        processed=[]
        for i,result in enumerate(iterator):
            result['split']=mapping[(result['stimulus'],result['segment'])]
            processed.append(result)
            if (i+1)%20==0 or i+1==len(records):
                print(json.dumps({'records_preprocessed':i+1,'records_total':len(records)}),flush=True)
    # Fit per-subject channel moments using disjoint TRAIN source spans, never event-overlap counts.
    accum={}
    for rec in processed:
        if rec['split']!='train':continue
        data=np.load(out/rec['path'],mmap_mode='r')
        margin=round(guard*sample_rate)
        x=np.asarray(data[:,margin:-margin],dtype='float64') if margin else np.asarray(data,dtype='float64')
        if x.shape[1]==0:raise ValueError('Empty training normalization span')
        subject=rec['subject'];s,q,n=accum.get(subject,(np.zeros(len(metadata['channels'])),np.zeros(len(metadata['channels'])),0))
        accum[subject]=(s+x.sum(1),q+(x*x).sum(1),n+x.shape[1])
    stats={}
    for subject,(s,q,n) in accum.items():
        mean=s/n;std=np.sqrt(np.maximum(q/n-mean**2,0))
        # A relative floor is independent of exported voltage scaling.
        floor=max(float(np.median(std))*1e-6,np.finfo('float32').tiny)
        stats[subject]={'mean':mean.tolist(),'std':np.maximum(std,floor).tolist(),'samples':n,
                        'near_flat_channels':[metadata['channels'][i] for i in np.flatnonzero(std<floor)]}
    labels={w:i for i,w in enumerate(words)};selected=[];length=round((context_end-context_start)*sample_rate)
    for e in events:
        if e['word'] not in labels:continue
        rec=processed[e['record']]
        start=round((e['onset']-rec['source_start_seconds']+context_start)*sample_rate)
        if start<0 or start+length>rec['shape'][1]:
            raise ValueError('Rounded crop exceeds preprocessed segment')
        if e['subject'] not in stats:raise ValueError('Subject lacks training normalization')
        selected.append(dict(e,y=labels[e['word']],start=start,split=rec['split']))
    if any(not any(e['split']==s for e in selected) for s in ['train','val','test']):
        raise ValueError('Empty split')
    np.savez_compressed(out/'events.npz',**{field:np.array([e[field] for e in selected])
        for field in ['record','start','y','split','subject','stimulus','segment','event_id','onset','duration']})
    np.savez_compressed(out/'embeddings.npz',words=np.array(words),embeddings=table)
    specification={'format':'neuromoment-continuous-v2','synthetic':False,'sample_rate':sample_rate,
        'context_start':context_start,'context_end':context_end,'epoch_start':epoch_start,'epoch_end':epoch_end,'split_counts':split_counts,'guard':guard,'seed':seed,'normalization_clip_z':10.,'channels':metadata['channels'],
        'source':metadata,'recordings':processed,'normalization':stats,'vocabulary':report,
        'coverage_complete':len(words)==candidates and all(report['full_candidate_coverage'].values()),
        'experiment_scope':'run_disjoint_word_retrieval',
        'boundary_dropped':dropped,'events_sha256':sha256(out/'events.npz'),
        'embeddings_sha256':sha256(out/'embeddings.npz'),'embedding_source_sha256':sha256(embeddings),
        'text_provenance':json.loads(Path(str(embeddings)+'.json').read_text()) if Path(str(embeddings)+'.json').exists() else None,
        'preprocessing':{'bandpass_hz':[low,min(high,sample_rate/2-10)],'reference':'scalp average',
          'bad_channels':'BIDS-marked only; standard-montage spherical interpolation; no fitted ICA',
          'filtering':'separate source stimulus segment, finite-support zero-phase FIR + polyphase resampling',
          'amplitude':'preserve MNE-calibrated values; train-only unitless z-score; no absolute microvolt rejection',
          'unit_flag_records':sum(r['amplitude_unit_flag'] for r in processed)},
        'versions':{'numpy':np.__version__,'mne':mne.__version__}}
    # Descriptor is the finalization marker; interrupted preprocessing remains resumable but not trainable.
    temp=out/'dataset.tmp';temp.write_text(json.dumps(specification,indent=2));temp.replace(out/'dataset.json')
    print(json.dumps({'dataset':dataset,'out':str(out),'events':len(selected),
        'candidates':len(words),'coverage_complete':specification['coverage_complete'],'complete':True}),flush=True)
    return specification


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--dataset',choices=['ds004408','brennan'],required=True)
    p.add_argument('--embeddings',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--candidates',type=int,default=250)
    p.add_argument('--min-train',type=int,default=1);p.add_argument('--sample-rate',type=float,default=128.)
    p.add_argument('--guard',type=float,default=2.);p.add_argument('--subjects',nargs='+')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--include-excluded',action='store_true')
    p.add_argument('--split-counts',type=int,nargs=3)
    p.add_argument('--low',type=float,default=1.);p.add_argument('--high',type=float,default=30.)
    p.add_argument('--context-start',type=float,default=-.28);p.add_argument('--context-end',type=float,default=1.08)
    p.add_argument('--epoch-start',type=float,default=-.2);p.add_argument('--epoch-end',type=float,default=1.)
    a=p.parse_args();build(a.root,a.dataset,a.embeddings,a.out,a.seed,a.candidates,a.min_train,
                           a.sample_rate,a.guard,a.subjects,a.workers,a.include_excluded,
                           a.split_counts,a.low,a.high,a.context_start,a.context_end,a.epoch_start,a.epoch_end)


if __name__=='__main__':main()
