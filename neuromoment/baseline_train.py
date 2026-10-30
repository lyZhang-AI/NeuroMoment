"""Train external word-retrieval baselines without changing the frozen core models."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .baseline_models import BaselineConfig, SOURCE, build_model
from .data import EventArchive, archive_fingerprint
from .metrics import retrieval_metrics
from .train import output_lock
from .text import sha256


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def source_identity():
    paths = [Path(__file__).with_name(n) for n in ['baseline_models.py','baseline_train.py','data.py','metrics.py']]
    paths += sorted(SOURCE.glob('*'))
    return {(str(p.relative_to(Path(__file__).parents[1])) if p.is_relative_to(Path(__file__).parents[1]) else 'optional_sources/' + p.name): sha256(p) for p in paths if p.is_file()}


class SubjectEvents(Dataset):
    def __init__(self, archive, subjects, shuffle_seed=None):
        self.archive = archive
        names = archive.subjects if archive.subjects is not None else np.repeat('synthetic', len(archive))
        mapping = {s: i for i, s in enumerate(subjects)}
        if any(str(s) not in mapping for s in names):
            raise ValueError('Evaluation subject absent from training')
        self.subject_index = torch.tensor([mapping[str(s)] for s in names])
        self.source_index = np.arange(len(archive))
        if shuffle_seed is not None:
            order = np.random.default_rng(shuffle_seed).permutation(len(archive))
            self.source_index[order] = np.roll(order, 1)

    def __len__(self): return len(self.archive)

    def __getitem__(self, i):
        source = int(self.source_index[i])
        x, _ = self.archive[source]
        # Subject identity travels with EEG; true target labels never enter the encoder.
        return x, self.archive.y[i], self.subject_index[source]


def make_config(archive, arch, width, depth):
    subjects = sorted(set(map(str, archive.subjects))) if archive.subjects is not None else ['synthetic']
    positions = []
    if arch == 'brainmodule':
        import mne
        if not archive.path.is_dir():
            raise ValueError('BrainModule requires real montage metadata')
        name = archive.metadata['source']['dataset']
        info = mne.create_info(archive.metadata['channels'], archive.sample_rate, 'eeg')
        info.set_montage(mne.channels.make_standard_montage('biosemi128' if name == 'ds004408' else 'easycap-M10'))
        layout = mne.find_layout(info)
        indices = [layout.names.index(n) for n in info.ch_names]
        coordinates = layout.pos[indices, :2]
        coordinates = (coordinates - coordinates.min(0)) / np.ptp(coordinates, axis=0)
        positions = coordinates.tolist()
    return BaselineConfig(arch, archive.channels, archive.embeddings.shape[1], archive.n,
                          archive.sample_rate, subjects, positions, width, depth)


@torch.no_grad()
def evaluate(model, archive, device, batch_size, shift=0., shuffle_seed=None):
    model.eval(); wrapped = SubjectEvents(archive, model.config.subjects, shuffle_seed)
    scores = []; embeddings = F.normalize(archive.embeddings.to(device), dim=-1)
    for context, _, subjects in DataLoader(wrapped, batch_size=batch_size):
        prediction = model(archive.view(context.to(device), shift), subjects.to(device))
        scores.append((prediction @ embeddings.T).cpu())
    scores = torch.cat(scores)
    metrics = retrieval_metrics(scores, archive.y)
    metrics['metric_scope'] = 'full_pool_macro' if metrics['coverage'] == 1 else 'observed_classes_macro_only'
    artifacts = {'scores': scores.numpy(), 'labels': archive.y.numpy(), 'words': np.array(archive.words),
                 'groups': archive.groups, 'event_ids': archive.event_ids, 'subjects': archive.subjects}
    if shuffle_seed is not None: artifacts['eeg_source_index'] = wrapped.source_index
    return metrics, {k:v for k,v in artifacts.items() if v is not None}


def evaluate_locked(model, args, data_hash):
    root = Path(args.out); ckpt = torch.load(root/'best.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt['model']); checkpoint_hash = sha256(root/'best.pt')
    test = EventArchive(args.data, 'test', max_shift=args.max_shift)
    for variant, shifts, shuffle in [('test', args.shifts, None), ('shuffled_eeg', [0.], 107)]:
        dest = root/variant; dest.mkdir(exist_ok=True); metrics_file = dest/'metrics.json'
        if metrics_file.exists():
            previous = json.loads(metrics_file.read_text())
            if previous['checkpoint_sha256'] == checkpoint_hash and previous['data_sha256'] == data_hash:
                continue
        results = {}
        for shift in shifts:
            m, artifacts = evaluate(model, test, args.device, args.batch_size, shift, shuffle)
            results[str(shift)] = m
            np.savez_compressed(dest/f'predictions_{shift:+.3f}.npz', **artifacts)
        atomic_json(metrics_file, {'evaluation_revision':1, 'architecture':args.arch,
                                  'split':'test', 'epoch':ckpt['epoch'], 'data_sha256':data_hash,
                                  'checkpoint_sha256':checkpoint_hash, 'eeg_shuffle_seed':shuffle,
                                  'localization':'not predicted; no localization metric claimed', 'results':results})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    p.add_argument('--arch', choices=['eegnet','brainmodule'], required=True)
    p.add_argument('--epochs', type=int, default=100); p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.0003); p.add_argument('--seed', type=int, default=7)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--threads', type=int, default=2)
    p.add_argument('--workers', type=int, default=2); p.add_argument('--temperature', type=float, default=.07)
    p.add_argument('--max-shift', type=float, default=.08)
    p.add_argument('--brain-width', type=int, default=320); p.add_argument('--brain-depth', type=int, default=10)
    p.add_argument('--shifts', type=float, nargs='+', default=[-.08,-.04,0.,.04,.08])
    p.add_argument('--resume', action='store_true'); p.add_argument('--skip-evaluation', action='store_true')
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--sampling',choices=['natural','balanced'],default='natural')
    args = p.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.max_shift<max(map(abs,args.shifts)) or args.temperature<=0:
        raise ValueError('Invalid training/evaluation budget or crop margins')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root=Path(args.out); root.mkdir(parents=True,exist_ok=True)
    with output_lock(root):
        train=EventArchive(args.data,'train',max_shift=args.max_shift)
        val=EventArchive(args.data,'val',max_shift=args.max_shift)
        config=make_config(train,args.arch,args.brain_width,args.brain_depth)
        model=build_model(config).to(args.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        sampler_rng=torch.Generator().manual_seed(args.seed)
        augmentation_rng=torch.Generator(device=args.device).manual_seed(args.seed+100)
        worker_rng=torch.Generator().manual_seed(args.seed+200)
        counts=torch.bincount(train.y,minlength=len(train.words)).double().clamp_min(1)
        sampler=(WeightedRandomSampler(1/counts[train.y],len(train),replacement=True,generator=sampler_rng)
                 if args.sampling=='balanced' else torch.utils.data.RandomSampler(train,generator=sampler_rng))
        loader=DataLoader(SubjectEvents(train,config.subjects),batch_size=args.batch_size,sampler=sampler,
                          num_workers=args.workers,persistent_workers=args.workers>0,generator=worker_rng)
        data_hash=archive_fingerprint(args.data); sources=source_identity()
        history=[];best=-float('inf');start=0;bad_epochs=0;early_stopped=False
        if args.resume:
            old=torch.load(root/'last.pt',map_location=args.device,weights_only=True)
            if old['config']!=asdict(config) or old['data_sha256']!=data_hash or old['source_sha256']!=sources:
                raise ValueError('Resume model/data/source mismatch')
            for key in ['arch','batch_size','lr','seed','temperature','max_shift','brain_width','brain_depth','patience','weight_decay','sampling']:
                if old['args'][key]!=vars(args)[key]:raise ValueError(f'Resume mismatch: {key}')
            model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
            start=old['epoch'];history=old['history'];best=old['best'];bad_epochs=old.get('bad_epochs',0)
            if args.patience and bad_epochs>=args.patience:early_stopped=True;start=args.epochs
            torch.set_rng_state(old['torch_rng'].cpu());random.setstate(old['python_rng'])
            n=old['numpy_rng'];np.random.set_state((n[0],np.array(n[1],dtype='uint32'),n[2],n[3],n[4]))
            if old['cuda_rng'] and torch.cuda.is_available():torch.cuda.set_rng_state_all([s.cpu() for s in old['cuda_rng']])
            sampler_rng.set_state(old['sampler_rng'].cpu());augmentation_rng.set_state(old['augmentation_rng'].cpu())
            worker_rng.set_state(old['worker_rng'].cpu())
        elif (root/'best.pt').exists():raise FileExistsError('Existing baseline output; use --resume')
        atomic_json(root/'run.json',{'args':vars(args),'config':asdict(config),'data_sha256':data_hash,
                                    'source_sha256':sources,'torch_version':str(torch.__version__),
                                    'parameters':sum(p.numel() for p in model.parameters()),
                                    'subject_metadata_used':args.arch=='brainmodule'})
        def save(value,path):
            tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
        embeds=F.normalize(train.embeddings.to(args.device),dim=-1)
        for epoch in range(start,args.epochs):
            began=time.monotonic();model.train();total=0.
            for step,(context,y,subjects) in enumerate(loader,1):
                context=context.to(args.device);y=y.to(args.device);subjects=subjects.to(args.device)
                bound=round(args.max_shift*train.sample_rate)
                delta=torch.randint(-bound,bound+1,(len(y),),device=args.device,generator=augmentation_rng)/train.sample_rate
                first=model(train.view(context),subjects);second=model(train.view(context,-delta),subjects)
                loss=(F.cross_entropy(first@embeds.T/args.temperature,y)+F.cross_entropy(second@embeds.T/args.temperature,y))/2
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite baseline loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                if hasattr(model,'constrain'):model.constrain()
                total+=loss.item()*len(y)
                if step%100==0:print(json.dumps({'epoch':epoch+1,'step':step,'steps':len(loader),'loss':loss.item(),'seconds':round(time.monotonic()-began,1)}),flush=True)
            metrics,_=evaluate(model,val,args.device,args.batch_size)
            history.append({'epoch':epoch+1,'loss':total/len(train),'val':metrics,'epoch_seconds':round(time.monotonic()-began,2)})
            print(json.dumps(history[-1]),flush=True)
            base={'epoch':epoch+1,'model':model.state_dict(),'config':asdict(config),'args':vars(args),
                  'data_sha256':data_hash,'source_sha256':sources}
            selection=metrics['balanced_top10']+1e-6*metrics['balanced_mrr']
            if selection>best:
                best=selection;bad_epochs=0;save(base,root/'best.pt')
            else:bad_epochs+=1
            early_stopped=bool(args.patience and bad_epochs>=args.patience)
            n=np.random.get_state()
            save({**base,'optimizer':optimizer.state_dict(),'history':history,'best':best,'bad_epochs':bad_epochs,
                  'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
                  'numpy_rng':(n[0],n[1].tolist(),n[2],n[3],n[4]),
                  'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                  'sampler_rng':sampler_rng.get_state(),'augmentation_rng':augmentation_rng.get_state(),
                  'worker_rng':worker_rng.get_state()},root/'last.pt')
            atomic_json(root/'history.json',history)
            if early_stopped:break
        atomic_json(root/'training_complete.json',{'epochs_requested':args.epochs,'epochs_completed':len(history),'early_stopped':early_stopped})
        if not args.skip_evaluation:evaluate_locked(model,args,data_hash)


if __name__=='__main__':main()
