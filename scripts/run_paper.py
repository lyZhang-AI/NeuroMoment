"""Execute the five matched 16/2/2 stimulus-run splits in the supplied paper protocol."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from neuromoment.pipeline import run
from neuromoment.text import sha256


def split_config(base, seed, training_seed):
    cfg=json.loads(json.dumps(base))
    cfg['seed']=seed;cfg['training_seed']=training_seed
    cfg.update(audit=f'reports/paper/split_{seed}/audit.json',
               embeddings=f'data/paper/split_{seed}/t5_large_words.npz',
               prepared=f'data/paper/split_{seed}/eeg',
               verification=f'reports/paper/split_{seed}/verification.json',runs=f'runs/paper/split_{seed}')
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/paper.json')
    p.add_argument('--stage',choices=['prepare','train','evaluate','occlusion','report','all'],default='all')
    p.add_argument('--device',default='cpu')
    p.add_argument('--models',nargs='+',help='Explicit subset of named model arms')
    p.add_argument('--baselines',nargs='+',choices=['eegnet','brainmodule'],help='Optional implemented external adapters')
    a=p.parse_args();spec=json.loads((PROJECT/a.config).read_text())
    base=json.loads((PROJECT/spec['base_config']).read_text())
    for seed in spec['split_seeds']:
        cfg=split_config(base,seed,spec['training_seed'])
        if a.models:
            names={m['name'] for m in cfg['models']}
            if set(a.models)-names:raise ValueError('Unknown model names')
            cfg['models']=[m for m in cfg['models'] if m['name'] in a.models]
        root=PROJECT/f'reports/paper/split_{seed}';root.mkdir(parents=True,exist_ok=True)
        config=root/'config.json'
        if config.exists() and ({k:v for k,v in json.loads(config.read_text()).items() if k!='models'}!={k:v for k,v in cfg.items() if k!='models'}):
            raise ValueError('Existing split configuration differs; use a fresh output directory')
        config.write_text(json.dumps(cfg,indent=2)+'\n')
        if a.stage in {'all','prepare'}:
            for stage in ['audit','text','prepare','verify']:run(config,stage,a.device)
        if a.stage in {'all','train'}:run(config,'train',a.device)
        if a.stage in {'all','evaluate'}:
            run(config,'evaluate',a.device)
            for m in cfg['models']:
                if m['mode']!='adaptive':continue
                checkpoint=PROJECT/cfg['runs']/m['name']/'best.pt'
                dest=checkpoint.parent/'shuffled_prior';existing=dest/'metrics.json'
                if existing.exists():
                    old=json.loads(existing.read_text())
                    if old['checkpoint_sha256']==sha256(checkpoint) and old.get('intervention')=='shuffled':continue
                subprocess.run([sys.executable,'-m','neuromoment.evaluate','--data',cfg['prepared'],
                    '--checkpoint',str(checkpoint),'--out',str(dest),'--device',a.device,
                    '--shifts','0','--shuffled-prior'],cwd=PROJECT,check=True)
        if a.stage in {'all','occlusion'} and any(m['name']=='neuromoment' for m in cfg['models']):
            checkpoint=PROJECT/cfg['runs']/'neuromoment/best.pt'
            dest=root/'occlusion';existing=dest/'metrics.json'
            complete=False
            if existing.exists():
                prior=json.loads(existing.read_text())
                complete=prior['checkpoint_sha256']==sha256(checkpoint)
            if not complete:
                subprocess.run([sys.executable,'-m','neuromoment.occlusion','--data',cfg['prepared'],
                    '--checkpoint',str(checkpoint),'--out',str(dest),'--device',a.device],cwd=PROJECT,check=True)
        for arch in a.baselines or []:
            out=PROJECT/cfg['runs']/arch
            if a.stage in {'all','train','evaluate'}:
                if a.stage=='evaluate' and not (out/'last.pt').exists():raise FileNotFoundError('Train the baseline first')
                args=[sys.executable,'-m','neuromoment.baseline_train','--data',cfg['prepared'],'--out',str(out),
                      '--arch',arch,'--device',a.device,'--epochs','100','--batch-size','64','--lr','.0003',
                      '--patience','10','--weight-decay','.01','--max-shift','.08','--sampling','natural',
                      '--seed',str(spec['training_seed'])]
                if (out/'last.pt').exists():args.append('--resume')
                subprocess.run(args,cwd=PROJECT,check=True)
    if a.stage in {'all','report'}:
        from neuromoment.paper_statistics import summarize
        summarize(PROJECT,spec['split_seeds'],[m['name'] for m in base['models']]+(a.baselines or []))


if __name__=='__main__':main()
