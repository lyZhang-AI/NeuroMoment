from contextlib import contextmanager
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import EventArchive, synthetic_archive, archive_fingerprint
from .losses import equivariance_ms, objective, transport_anchor
from .metrics import retrieval_metrics
from .diagnostics import absolute_equivariance_ms, full_support_js
from .model import ModelConfig, NeuroMoment


def fingerprint(path):
    return archive_fingerprint(path)

@torch.no_grad()
def evaluate(model, dataset, device, batch_size=64, delta=0., intervention=None, intervention_seed=7):
    model.eval()
    scores, labels, errors = [], [], []
    absolute_errors = []
    overlap_valid = []
    reference, shifted_alpha, priors, content, windows = [], [], [], [], []
    retained_mass = []
    embeds = F.normalize(dataset.embeddings.to(device), dim=-1)
    loader = DataLoader(dataset, batch_size=batch_size)
    permutation = current_bank = base_bank = None
    if intervention is not None:
        if intervention != 'shuffled' or model.config.mode != 'adaptive' or len(dataset) < 2:
            raise ValueError('Shuffled intervention requires adaptive model and >=2 events')
        generator = torch.Generator().manual_seed(intervention_seed)
        order = torch.randperm(len(dataset), generator=generator)
        permutation = torch.empty_like(order)
        permutation[order] = order.roll(1)  # A seeded global derangement, independent of minibatch size.
        def bank(shift):
            values = []
            for context, _ in loader:
                values.append(model(dataset.view(context.to(device), shift))['prior'].cpu())
            return torch.cat(values)
        current_bank = bank(delta)
        base_bank = bank(0.) if delta else current_bank
    offset = 0
    for context, y in loader:
        context = context.to(device)
        override = current_bank[permutation[offset:offset+len(y)]].to(device) if permutation is not None else None
        out = model(dataset.view(context, delta), prior_override=override)
        scores.append((out["embedding"] @ embeds.T).cpu()); labels.append(y)
        reference.append(out["alpha"].cpu()); priors.append(out["prior"].cpu())
        content.append(out["score"].cpu())
        if out["indices"] is not None:
            windows.append(out["times"][out["indices"]].cpu())
        if delta:
            base_override = base_bank[permutation[offset:offset+len(y)]].to(device) if permutation is not None else None
            base = model(dataset.view(context), prior_override=base_override)
            transported, mask = transport_anchor(base["alpha"], out["times"], delta)
            retained_mass.append(torch.stack([transported.sum(-1), (out["alpha"]*mask).sum(-1)], -1).cpu())
            absolute_errors.append(absolute_equivariance_ms(base["alpha"], out["alpha"], out["times"], delta).cpu())
            valid = (transported.sum(-1)>1e-12) & ((out["alpha"]*mask).sum(-1)>1e-12)
            overlap_valid.append(valid.cpu())
            if valid.any():
                errors.append(equivariance_ms(base["alpha"][valid], out["alpha"][valid], out["times"], delta).cpu())
        shifted_alpha.append(out["log_prior"].cpu())
        offset += len(y)
    metrics = retrieval_metrics(torch.cat(scores), torch.cat(labels))
    if absolute_errors:
        metrics["equivariance_wasserstein_ms"] = torch.cat(absolute_errors).mean().item()
        metrics["overlap_conditioned_wasserstein_ms"] = torch.cat(errors).mean().item() if errors else None
        metrics["overlap_valid_fraction"] = torch.cat(overlap_valid).float().mean().item()
        metrics["overlap_retained_mass_target"] = torch.cat(retained_mass)[:, 0].mean().item()
        metrics["overlap_retained_mass_prediction"] = torch.cat(retained_mass)[:, 1].mean().item()
    p = torch.cat(priors)
    metrics["prior_between_event_variance"] = p.var(0, unbiased=False).mean().item()
    metrics["prior_entropy"] = (-(p * p.clamp_min(1e-12).log()).sum(-1)).mean().item()
    artifacts = {"alpha": torch.cat(reference).numpy(), "prior": p.numpy(),
                 "log_prior": torch.cat(shifted_alpha).numpy(), "score": torch.cat(content).numpy(),
                 "times": out["times"].cpu().numpy(), "labels": torch.cat(labels).numpy(),
                 "scores": torch.cat(scores).numpy(), "words": np.array(dataset.words)}
    if permutation is not None:
        artifacts["prior_source_index"] = permutation.numpy()
    if windows:
        artifacts["selected_times"] = torch.cat(windows).numpy()
        artifacts["local_width"] = model.config.local_width
    if dataset.groups is not None:
        artifacts["groups"] = dataset.groups
        artifacts["event_ids"] = dataset.event_ids
        artifacts["subjects"] = dataset.subjects
    metrics["metric_scope"] = "full_pool_macro" if metrics["coverage"] == 1 else "observed_classes_macro_only"
    metrics["synthetic"] = dataset.metadata.get("synthetic", "unknown")
    return metrics, artifacts


def training_objective(first, second, labels, embeddings, delta, temperature=.07, lambda_shift=.2, lambda_inv=.1, shift_support="overlap", alignment_views="original"):
    if shift_support not in {"overlap","full"}:raise ValueError("Unknown shift support")
    if lambda_shift and shift_support=="overlap":
        return objective(first, second, labels, embeddings, delta, temperature, lambda_shift, lambda_inv, alignment_views)
    # A disabled JS term must not fail merely because the unregularized model lost overlap.
    embeddings = F.normalize(embeddings, dim=-1)
    align = (F.cross_entropy(first['embedding']@embeddings.T/temperature, labels)
             + F.cross_entropy(second['embedding']@embeddings.T/temperature, labels))/2
    if alignment_views=='original':align=F.cross_entropy(first['embedding']@embeddings.T/temperature,labels)
    elif alignment_views!='both':raise ValueError('Unknown alignment view protocol')
    inv = (1-(first['embedding']*second['embedding']).sum(-1)).mean()
    shift = full_support_js(first['alpha'],second['alpha'],first['times'],delta) if lambda_shift else (first['alpha'].sum()+second['alpha'].sum())*0
    return align+lambda_shift*shift+lambda_inv*inv, {'align':align.detach(),'shift':shift.detach(),'inv':inv.detach()}


@contextmanager
def output_lock(directory):
    import fcntl
    import os
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.train.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Training already running in {directory}') from exc
        handle.seek(0); handle.truncate(); handle.write(str(os.getpid())); handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    import os
    import time
    from torch.utils.data import RandomSampler, WeightedRandomSampler
    from .text import sha256
    parser = argparse.ArgumentParser(description="Train event-anchored EEG word retrieval")
    parser.add_argument('--data', default='data/synthetic.npz')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--out', default='runs/smoke')
    parser.add_argument('--mode', choices=['adaptive','attention','fixed_prior','global_offset','fixed_window','mil','fixed_gaussian','temporal_pool'],default='adaptive')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--lambda-shift', type=float, default=.2)
    parser.add_argument('--lambda-inv', type=float, default=.1)
    parser.add_argument('--shift-support', choices=['overlap','full'], default='overlap')
    parser.add_argument('--max-shift', type=float, default=.08)
    parser.add_argument('--temperature', type=float, default=.07)
    parser.add_argument('--fine', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fusion', choices=['soft','all','global','local'])
    parser.add_argument('--fixed-center', type=float, default=.3)
    parser.add_argument('--prior-sigma', type=float, default=.08)
    parser.add_argument('--prior-beta', type=float, default=1.)
    parser.add_argument('--latency-min', type=float, default=0.)
    parser.add_argument('--latency-max', type=float, default=1.)
    parser.add_argument('--latency-bins', type=int, default=26)
    parser.add_argument('--local-width', type=float, default=.16)
    parser.add_argument('--moments', type=int, default=3)
    parser.add_argument('--init', help='Stage I checkpoint for Stage II')
    parser.add_argument('--resume', action='store_true', help='Resume last.pt at an epoch boundary; --epochs is the total target')
    parser.add_argument('--sampling', choices=['natural','balanced'], default='natural')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--alignment-views',choices=['original','both'],default='original')
    parser.add_argument('--nms-interval',type=float,default=.12)
    parser.add_argument('--content-temperature',type=float,default=1.)
    parser.add_argument('--selection-temperature',type=float,default=1.)
    parser.add_argument('--selection',choices=['straight_through','hard'],default='straight_through')
    parser.add_argument('--fine-layers',type=int,default=1)
    parser.add_argument('--weight-decay',type=float,default=.01)
    parser.add_argument('--patience',type=int,default=10,help='Validation checks without improvement; 0 disables early stopping')
    parser.add_argument('--min-delta',type=float,default=0.)
    parser.add_argument('--shift-convention',choices=['signal','anchor'],default='signal')
    parser.add_argument('--start',type=float)
    parser.add_argument('--end',type=float)
    args = parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.max_shift<0 or args.workers<0 or args.threads<1:
        raise ValueError('Invalid training settings')
    if args.temperature<=0 or args.lambda_shift<0 or args.lambda_inv<0:
        raise ValueError('Temperature must be positive and loss weights nonnegative')
    if args.prior_sigma<=0 or args.prior_beta<0 or args.latency_bins<2 or args.latency_min>=args.latency_max or args.local_width<=0 or args.moments<1:
        raise ValueError('Invalid grounding configuration')
    if args.patience < 0 or args.min_delta < 0 or args.weight_decay < 0:raise ValueError('Invalid optimizer/early-stopping settings')
    if args.init and args.resume:raise ValueError('--init and --resume are mutually exclusive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if args.synthetic:
        if Path(args.data).exists():raise FileExistsError('Synthetic archive already exists')
        synthetic_archive(args.data,seed=args.seed)
    out_dir=Path(args.out);out_dir.mkdir(parents=True,exist_ok=True)
    with output_lock(out_dir):
        if (out_dir/'best.pt').exists() and not args.resume:
            raise FileExistsError('Output contains a checkpoint; choose a fresh directory or --resume')
        train=EventArchive(args.data,'train',start=args.start,end=args.end,max_shift=args.max_shift)
        val=EventArchive(args.data,'val',start=args.start,end=args.end,max_shift=args.max_shift)
        config=ModelConfig(channels=train.channels,text_dim=train.embeddings.shape[1],dim=args.dim,
                          sample_rate=train.sample_rate,start=train.start,end=train.end,mode=args.mode,fine=args.fine,fixed_center=args.fixed_center,
                          fusion=args.fusion or ('all' if args.fine else 'soft'),
                          sigma=args.prior_sigma,beta=args.prior_beta,latency_min=args.latency_min,latency_max=args.latency_max,
                          latency_bins=args.latency_bins,local_width=args.local_width,moments=args.moments,
                          nms_interval=args.nms_interval,content_temperature=args.content_temperature,
                          selection=args.selection,selection_temperature=args.selection_temperature,fine_layers=args.fine_layers)
        model=NeuroMoment(config).to(args.device)
        data_hash=fingerprint(args.data)
        source_hash={name:sha256(Path(__file__).with_name(name)) for name in ['model.py','losses.py','data.py','train.py','diagnostics.py','metrics.py']}
        if args.init:
            checkpoint=torch.load(args.init,map_location=args.device,weights_only=True)
            if checkpoint['data_sha256']!=data_hash:raise ValueError('Stage I belongs to another dataset')
            for key in ['channels','text_dim','dim','sample_rate','start','stride','mode']:
                if checkpoint['config'][key]!=config.__dict__[key]:raise ValueError(f'Stage I mismatch: {key}')
            loaded=model.load_state_dict({k:v for k,v in checkpoint['model'].items()
                if not k.startswith('fusion.') and k in model.state_dict()},strict=False)
            print(json.dumps({'initialized_from':args.init,'new_parameters':loaded.missing_keys}),flush=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        sampler_generator=torch.Generator().manual_seed(args.seed)
        augmentation_generator=torch.Generator(device=args.device).manual_seed(args.seed+100)
        worker_generator=torch.Generator().manual_seed(args.seed+200)
        if args.sampling=='balanced':
            counts=torch.bincount(train.y,minlength=len(train.words)).double().clamp_min(1)
            sampler=WeightedRandomSampler(1/counts[train.y],len(train),replacement=True,generator=sampler_generator)
        else:
            sampler=RandomSampler(train,generator=sampler_generator)
        loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,
                          persistent_workers=args.workers>0,generator=worker_generator)
        embeddings=train.embeddings.to(args.device)
        best=-float('inf');history=[];start_epoch=0;bad_epochs=0;early_stopped=False
        if args.resume:
            checkpoint=torch.load(out_dir/'last.pt',map_location=args.device,weights_only=True)
            if checkpoint['data_sha256']!=data_hash or checkpoint['config']!=model.config_dict():
                raise ValueError('Resume dataset/model configuration mismatch')
            for key in ['sampling','max_shift','lambda_shift','lambda_inv','temperature','batch_size','seed','lr','weight_decay','patience','min_delta','shift_convention','alignment_views']:
                if checkpoint['args'][key]!=vars(args)[key]:raise ValueError(f'Resume training configuration mismatch: {key}')
            if checkpoint['args'].get('shift_support','overlap')!=args.shift_support:
                raise ValueError('Resume shift-objective mismatch')
            for key in source_hash:
                if checkpoint['source_sha256'][key]!=source_hash[key]:raise ValueError(f'Resume code changed: {key}')
            model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch=checkpoint['epoch'];best=checkpoint['best'];history=checkpoint['history']
            bad_epochs=checkpoint.get('bad_epochs',0)
            torch.set_rng_state(checkpoint['torch_rng'].cpu());random.setstate(checkpoint['python_rng'])
            n=checkpoint['numpy_rng'];np.random.set_state((n[0],np.array(n[1],dtype='uint32'),n[2],n[3],n[4]))
            if 'cuda_rng' in checkpoint and torch.cuda.is_available():torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
            sampler_generator.set_state(checkpoint['sampler_rng'].cpu())
            augmentation_generator.set_state(checkpoint['augmentation_rng'].cpu())
            worker_generator.set_state(checkpoint['worker_rng'].cpu())
            if checkpoint.get('early_stopped',False):
                print(json.dumps({'already_early_stopped':True,'epoch':start_epoch}),flush=True)
                return
            print(json.dumps({'resumed_after_epoch':start_epoch,'target_epochs':args.epochs}),flush=True)
        run={'args':vars(args),'config':model.config_dict(),'data_sha256':data_hash,'source_sha256':source_hash,
             'torch_version':str(torch.__version__),'parameters':sum(p.numel() for p in model.parameters()),
             'synthetic':train.metadata.get('synthetic','unknown'),
             'coverage_complete':train.metadata.get('coverage_complete',False),
             'experiment_scope':train.metadata.get('experiment_scope','unspecified'),
             'events':{'train':len(train),'val':len(val)}}
        (out_dir/'run.json').write_text(json.dumps(run,indent=2))
        def save(value,path):
            temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)
        def base_checkpoint(epoch):
            return {'config':model.config_dict(),'model':model.state_dict(),'epoch':epoch,
                    'data_sha256':data_hash,'args':vars(args),'source_sha256':source_hash}
        for epoch in range(start_epoch,args.epochs):
            started=time.monotonic();model.train()
            sums={'loss':0.,'align':0.,'shift':0.,'inv':0.}
            for step,(context,y) in enumerate(loader,1):
                context,y=context.to(args.device),y.to(args.device)
                bound=round(args.max_shift*train.sample_rate)
                delta=torch.randint(-bound,bound+1,(len(y),),device=args.device,
                                    generator=augmentation_generator)/train.sample_rate
                anchor_delta = -delta if args.shift_convention == 'signal' else delta
                first=model(train.view(context));second=model(train.view(context,anchor_delta))
                loss,parts=training_objective(first,second,y,embeddings,anchor_delta,args.temperature,args.lambda_shift,args.lambda_inv,args.shift_support,args.alignment_views)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();sums['loss']+=loss.item()*len(y)
                for k,v in parts.items():sums[k]+=v.item()*len(y)
                if args.log_every>0 and step%args.log_every==0:
                    print(json.dumps({'epoch':epoch+1,'step':step,'steps':len(loader),
                                      'loss':loss.item(),'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
            metrics,_=evaluate(model,val,args.device,args.batch_size)
            record={'epoch':epoch+1,**{k:v/len(train) for k,v in sums.items()},'val':metrics,
                    'epoch_seconds':round(time.monotonic()-started,2)}
            history.append(record);print(json.dumps(record),flush=True)
            selection=metrics['balanced_top10']+1e-6*metrics['balanced_mrr']
            if selection>best+args.min_delta:
                bad_epochs=0
                best=selection;save(base_checkpoint(epoch+1),out_dir/'best.pt')
            else:
                bad_epochs+=1
            early_stopped=bool(args.patience and bad_epochs>=args.patience)
            n=np.random.get_state()
            checkpoint=base_checkpoint(epoch+1)
            checkpoint.update({'optimizer':optimizer.state_dict(),'best':best,'history':history,'bad_epochs':bad_epochs,'early_stopped':early_stopped,
                'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
                'numpy_rng':(n[0],n[1].tolist(),n[2],n[3],n[4]),
                'sampler_rng':sampler_generator.get_state(),'augmentation_rng':augmentation_generator.get_state(),
                'worker_rng':worker_generator.get_state()})
            if torch.cuda.is_available():checkpoint['cuda_rng']=torch.cuda.get_rng_state_all()
            save(checkpoint,out_dir/'last.pt')
            temporary=out_dir/'history.tmp';temporary.write_text(json.dumps(history,indent=2));temporary.replace(out_dir/'history.json')
            if early_stopped:
                print(json.dumps({'early_stopping_epoch':epoch+1,'patience':args.patience}),flush=True)
                break
        marker={'epochs_requested':args.epochs,'epochs_completed':len(history),'early_stopped':early_stopped}
        temporary=out_dir/'training_complete.tmp';temporary.write_text(json.dumps(marker,indent=2));temporary.replace(out_dir/'training_complete.json')


if __name__=='__main__':main()
