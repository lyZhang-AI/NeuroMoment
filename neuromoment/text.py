"""Export frozen isolated-word language embeddings from a pinned local HF snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


DEFAULT_MODEL = 'google-t5/t5-large'
DEFAULT_REVISION = '150ebc2c4b72291e770f58e6057481c8d2ed331a'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def export(words, out, model_id=DEFAULT_MODEL, revision=DEFAULT_REVISION, batch_size=64):
    from transformers import AutoModel, AutoTokenizer, T5EncoderModel
    from huggingface_hub import snapshot_download
    out=Path(out)
    if out.exists():
        raise FileExistsError(out)
    words=sorted(set(words))
    if not words:
        raise ValueError('Empty vocabulary')
    local=Path(model_id)
    if not local.is_dir():
        local=Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    tokenizer=AutoTokenizer.from_pretrained(local, local_files_only=True)
    model_class = T5EncoderModel if json.loads((local/'config.json').read_text()).get('model_type') == 't5' else AutoModel
    model=model_class.from_pretrained(local, local_files_only=True).eval().requires_grad_(False)
    vectors=[]
    torch.set_num_threads(4)
    with torch.inference_mode():
        for i in range(0,len(words),batch_size):
            inputs=tokenizer(words[i:i+batch_size],padding=True,truncation=True,max_length=32,return_tensors='pt')
            hidden=model(**inputs).last_hidden_state
            mask=inputs['attention_mask'][...,None]
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            vectors.append(torch.nn.functional.normalize(pooled,dim=-1).cpu().numpy())
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,words=np.array(words),embeddings=np.concatenate(vectors).astype('float32'))
    source_files={p.name:sha256(p) for p in local.iterdir() if p.is_file() and p.suffix in {'.json','.safetensors','.txt'}}
    metadata={'model':model_id,'revision':revision,'snapshot':str(local.resolve()),
              'input':'isolated normalized word only; no sentence context',
              'pooling':'attention-masked mean including special tokens, then L2 normalization',
              'frozen':True,'synthetic':False,'source_sha256':source_files,'output_sha256':sha256(out)}
    Path(str(out)+'.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'words':len(words),'dimension':vectors[0].shape[-1],'out':str(out),'revision':revision}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',required=True);p.add_argument('--pool',default='250');p.add_argument('--out',required=True)
    p.add_argument('--model',default=DEFAULT_MODEL);p.add_argument('--revision',default=DEFAULT_REVISION)
    a=p.parse_args();report=json.loads(Path(a.audit).read_text())
    export(report['candidate_pools'][a.pool]['words'],a.out,a.model,a.revision)


if __name__=='__main__':main()
