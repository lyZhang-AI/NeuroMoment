"""Configuration-driven, provenance-checked data preparation and training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .data import archive_fingerprint
from .text import sha256

PROJECT=Path(__file__).resolve().parents[1]


def finished(root, epochs):
    root=Path(root)
    if not (root/'best.pt').exists():return False
    history=json.loads((root/'history.json').read_text()) if (root/'history.json').exists() else []
    if len(history)>epochs:raise ValueError('Existing training exceeds configured epoch budget')
    if len(history)==epochs:return True
    marker=root/'training_complete.json'
    if marker.exists():
        value=json.loads(marker.read_text())
        return value['epochs_requested']==epochs and value['epochs_completed']==len(history) and value['early_stopped']
    return False


def run(config,stage='all',device=None):
    cfg=json.loads(Path(config).resolve().read_text())
    def path(value):return PROJECT/value
    def call(module,args):
        command=[sys.executable,'-m',f'neuromoment.{module}',*map(str,args)]
        print('RUN '+repr(command),flush=True)
        environment=os.environ.copy()
        if module=='preprocess':
            environment.update({k:'1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
        subprocess.run(command,cwd=PROJECT,env=environment,check=True)
    def flags(values):
        args=[]
        for key,value in values.items():
            if value is None:continue
            flag='--'+key.replace('_','-')
            if isinstance(value,bool):args.append(flag if value else '--no-'+key.replace('_','-'))
            elif isinstance(value,list):args.extend([flag,*value])
            else:args.extend([flag,value])
        return args
    shared={'root':cfg['root'],'dataset':cfg['dataset'],'seed':cfg['seed'],'guard':cfg['guard']}
    for key in ['split_counts','context_start','context_end']:
        if key in cfg:shared[key]=cfg[key]
    audit_path=path(cfg['audit']);prepared=path(cfg['prepared']);embedding=path(cfg['embeddings'])
    if stage in {'all','audit'}:
        call('audit',flags({**shared,'out':audit_path,'min_train':cfg['min_train']}))
        if cfg.get('strict_candidates'):
            audit=json.loads(audit_path.read_text())
            if audit['candidate_pools'][str(cfg['candidates'])]['actual_candidates']!=cfg['candidates']:
                raise ValueError('Training split cannot supply the configured candidate vocabulary')
    if stage in {'all','text'}:
        if embedding.exists():
            meta=json.loads(Path(str(embedding)+'.json').read_text())
            if meta['revision']!=cfg['text_revision'] or meta['output_sha256']!=sha256(embedding):
                raise ValueError('Embedding cache differs from pinned configuration')
            if 'text_model' in cfg and meta['model']!=cfg['text_model']:raise ValueError('Text model changed')
        else:
            values={'audit':audit_path,'out':embedding,'revision':cfg['text_revision'],'pool':cfg['candidates']}
            if 'text_model' in cfg:values['model']=cfg['text_model']
            call('text',flags(values))
    if stage in {'all','prepare'}:
        if (prepared/'dataset.json').exists():
            spec=json.loads((prepared/'dataset.json').read_text())
            for k in ['seed','guard','sample_rate','split_counts','context_start','context_end','epoch_start','epoch_end']:
                if k in cfg and spec.get(k)!=cfg[k]:raise ValueError(f'Prepared dataset configuration changed: {k}')
            if cfg.get('upstream_commit') and spec['source'].get('upstream_commit')!=cfg['upstream_commit']:
                raise ValueError('Prepared upstream revision differs from configuration')
            if Path(spec['source']['root']).resolve()!=Path(cfg['root']).resolve() or spec['source']['dataset']!=cfg['dataset']:
                raise ValueError('Prepared source dataset differs from configuration')
            if spec['vocabulary']['min_train_independent_events']!=cfg['min_train'] or spec['embedding_source_sha256']!=sha256(embedding):
                raise ValueError('Prepared vocabulary/text provenance differs from configuration')
            if spec['vocabulary']['requested_candidates']!=cfg['candidates']:raise ValueError('Prepared candidate pool differs')
            if 'low' in cfg and spec['preprocessing']['bandpass_hz']!=[cfg['low'],cfg['high']]:raise ValueError('Bandpass changed')
        else:
            values={**shared,'embeddings':embedding,'out':prepared,'candidates':cfg['candidates'],
                    'min_train':cfg['min_train'],'sample_rate':cfg['sample_rate'],'workers':cfg['preprocess_workers']}
            for key in ['low','high','epoch_start','epoch_end']:
                if key in cfg:values[key]=cfg[key]
            call('preprocess',flags(values))
    if stage in {'all','verify'}:call('verify',flags({'data':prepared,'out':path(cfg['verification'])}))
    if stage not in {'all','train','ridge','evaluate'}:return
    training=cfg['training'];actual_device=device or training.get('device','cpu');runs=path(cfg['runs'])
    if stage in {'all','train'}:
        for model in cfg['models']:
            out=runs/model['name']
            values={k:v for k,v in training.items() if k!='device'}
            values.update({k:v for k,v in model.items() if k not in {'name','init_model'}})
            values.setdefault('seed',cfg.get('training_seed',cfg['seed']))
            values.setdefault('fine',False)
            values.update(data=prepared,out=out,device=actual_device)
            if (out/'run.json').exists():
                prior=json.loads((out/'run.json').read_text())
                if any(sha256(PROJECT/'neuromoment'/name)!=value for name,value in prior['source_sha256'].items()):raise ValueError('Training source code identity changed')
                if prior['data_sha256']!=archive_fingerprint(prepared):raise ValueError('Training data identity changed')
                for key,value in values.items():
                    if key not in {'data','out','epochs','device','threads','workers','log_every'} and prior['args'].get(key)!=value:
                        raise ValueError(f'Existing training configuration changed: {key}')
            if finished(out,values['epochs']):
                print('REUSE completed training '+str(out),flush=True);continue
            if (out/'last.pt').exists():values['resume']=True
            elif model.get('init_model'):values['init']=runs/model['init_model']/'best.pt'
            call('train',flags(values))
    if stage in {'all','ridge'} and cfg.get('ridge'):
        if not (runs/'ridge/ridge.pt').exists():
            call('ridge',flags({'data':prepared,'out':runs/'ridge',**cfg['ridge']}))
    if stage in {'all','evaluate'}:
        for model in cfg['models']:
            checkpoint=runs/model['name']/'best.pt';out=runs/model['name']/'test'
            shifts=cfg.get('evaluation_shifts',[-.08,-.04,0.,.04,.08])
            existing=out/'metrics.json'
            if existing.exists():
                prior=json.loads(existing.read_text())
                if (prior.get('evaluation_revision')==2 and prior.get('checkpoint_sha256')==sha256(checkpoint)
                    and prior['data_sha256']==archive_fingerprint(prepared) and prior.get('split')=='test'
                    and prior.get('intervention') is None and prior.get('eeg_shuffle_seed') is None
                    and set(prior['results'])==set(map(str,shifts))):continue
            call('evaluate',flags({'data':prepared,'checkpoint':checkpoint,'out':out,'device':actual_device,
                                  'batch_size':cfg.get('evaluation_batch_size',64),'shifts':shifts}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['all','audit','text','prepare','verify','train','ridge','evaluate'],default='all')
    p.add_argument('--device')
    a=p.parse_args();run(a.config,a.stage,a.device)


if __name__=='__main__':main()
