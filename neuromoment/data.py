"""Portable archive contract. Raw adapters must preserve real recording timestamps."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def archive_fingerprint(path):
    from .text import sha256
    path = Path(path)
    if path.is_file():
        return sha256(path)
    h = __import__('hashlib').sha256()
    for name in ['dataset.json', 'events.npz', 'embeddings.npz']:
        h.update(name.encode()); h.update(sha256(path/name).encode())
    return h.hexdigest()


class EventArchive(Dataset):
    """Legacy NPZ or v2 continuous recordings with lazy, train-normalized crops."""
    def __init__(self, path, split, start=None, end=None, max_shift=0.08):
        from collections import OrderedDict
        self.path = Path(path)
        self._directory = self.path.is_dir()
        self._cache = OrderedDict()
        if self._directory:
            self._load_directory(split)
        else:
            self._load_legacy(split)
        if not len(self.y):
            raise ValueError(f'Empty {split} split')
        if self.embeddings.ndim != 2 or self.y.ndim != 1:
            raise ValueError('Expected embeddings[V,D] and y[B]')
        if not torch.isfinite(self.embeddings).all():
            raise ValueError('Non-finite text embeddings')
        if len(self.words) != len(self.embeddings) or len(set(self.words)) != len(self.words):
            raise ValueError('Candidate words must be unique and match embeddings')
        if self.y.min() < 0 or self.y.max() >= len(self.words):
            raise ValueError('Labels outside candidate vocabulary')
        if (self.embeddings.norm(dim=-1) <= 0).any():
            raise ValueError('Zero candidate embedding')
        start = self.metadata.get('epoch_start', -.4) if start is None else start
        end = self.metadata.get('epoch_end', 1.2) if end is None else end
        self.start, self.end = start, end
        self.n = round((end-start)*self.sample_rate)
        if self.n <= 0 or max_shift < 0:
            raise ValueError('Invalid input duration or max shift')
        for delta in [-max_shift, max_shift]:
            offset = round((start+delta-self.context_start)*self.sample_rate)
            if offset < 0 or offset+self.n > self.context_shape[-1]:
                raise ValueError('Archive lacks continuous margins for requested shifts')

    def _load_legacy(self, split):
        with np.load(self.path, allow_pickle=False) as f:
            required = {'x','y','split','sample_rate','context_start','embeddings','words'}
            if not required.issubset(f.files):
                raise ValueError(f'Missing archive fields: {required-set(f.files)}')
            indices = np.flatnonzero(f['split']==split)
            self._x = torch.tensor(f['x'][indices],dtype=torch.float32)
            if self._x.ndim != 3 or not torch.isfinite(self._x).all():
                raise ValueError('Expected finite x[B,C,T]')
            self.y = torch.tensor(f['y'][indices],dtype=torch.long)
            self.embeddings = torch.tensor(f['embeddings'],dtype=torch.float32)
            self.words = f['words'].tolist()
            self.sample_rate = float(f['sample_rate']); self.context_start = float(f['context_start'])
            self.groups = (np.char.add(np.char.add(f['stimulus'][indices], '::'),f['segment'][indices])
                           if {'stimulus','segment'}.issubset(f.files) else None)
            self.event_ids = f['event_id'][indices] if 'event_id' in f.files else None
            self.subjects = f['subject'][indices] if 'subject' in f.files else None
        self.channels = self._x.shape[1]; self.context_shape = tuple(self._x.shape[1:])
        sidecar = Path(str(self.path)+'.json')
        self.metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}

    def _load_directory(self, split):
        from .text import sha256
        spec = self.metadata = json.loads((self.path/'dataset.json').read_text())
        if spec['format'] != 'neuromoment-continuous-v2':
            raise ValueError('Unsupported dataset format')
        for name in ['events','embeddings']:
            if sha256(self.path/f'{name}.npz') != spec[f'{name}_sha256']:
                raise ValueError(f'Corrupted {name} index')
        with np.load(self.path/'events.npz',allow_pickle=False) as f:
            indices=np.flatnonzero(f['split']==split)
            self.y=torch.tensor(f['y'][indices],dtype=torch.long)
            self.record_ids=f['record'][indices];self.offsets=f['start'][indices]
            self.groups=np.char.add(np.char.add(f['stimulus'][indices], '::'),f['segment'][indices])
            self.event_ids=f['event_id'][indices];self.subjects=f['subject'][indices]
        with np.load(self.path/'embeddings.npz',allow_pickle=False) as f:
            self.words=f['words'].tolist();self.embeddings=torch.tensor(f['embeddings'],dtype=torch.float32)
        self.sample_rate=float(spec['sample_rate']);self.context_start=float(spec['context_start'])
        self.channels=len(spec['channels']);self.context_shape=(self.channels,round((spec['context_end']-spec['context_start'])*self.sample_rate))
        self._stats={s:(np.asarray(v['mean'],dtype='float32')[:,None],np.asarray(v['std'],dtype='float32')[:,None])
                     for s,v in spec['normalization'].items()}
        self._records=spec['recordings']
        self._clip_z=float(spec.get('normalization_clip_z',10.))
        if self._clip_z<=0:raise ValueError('Invalid clipping threshold')
        # Cheap unchanged-file checks; if timestamps differ (e.g. copied dataset), verify actual bytes.
        for rec in self._records:
            p=self.path/rec['path'];st=p.stat()
            if st.st_size!=rec['size_bytes']:
                raise ValueError(f'Corrupted recording size: {p}')
            if st.st_mtime_ns!=rec['mtime_ns'] and sha256(p)!=rec['sha256']:
                raise ValueError(f'Corrupted recording hash: {p}')
        if len(indices) and (self.record_ids.min()<0 or self.record_ids.max()>=len(self._records)):
            raise ValueError('Invalid recording index')

    @property
    def x(self):
        if self._directory:
            raise RuntimeError('Continuous v2 data are lazy; use DataLoader instead of materializing .x')
        return self._x

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if not self._directory:
            return self._x[index],self.y[index]
        record_id=int(self.record_ids[index]);rec=self._records[record_id]
        if record_id not in self._cache:
            self._cache[record_id]=np.load(self.path/rec['path'],mmap_mode='r',allow_pickle=False)
            if len(self._cache)>32:
                self._cache.popitem(last=False)
        data=self._cache[record_id];self._cache.move_to_end(record_id)
        a=int(self.offsets[index]);b=a+self.context_shape[-1]
        if a<0 or b>data.shape[1] or data.shape[0]!=self.channels:
            raise ValueError('Event context outside recording or incorrect montage')
        mean,std=self._stats[rec['subject']]
        x=(np.array(data[:,a:b],dtype='float32')-mean)/std
        np.clip(x,-self._clip_z,self._clip_z,out=x)  # Fixed unitless clipping; same training-calibrated scale in all splits.
        if not np.isfinite(x).all():
            raise ValueError('Non-finite normalized EEG crop')
        return torch.from_numpy(x),self.y[index]

    def __getstate__(self):
        state=self.__dict__.copy();state['_cache']={}
        from collections import OrderedDict
        state['_cache']=OrderedDict()
        return state

    def view(self, context, delta=0.0):
        delta=torch.as_tensor(delta,device=context.device)
        if delta.ndim==0:delta=delta.expand(context.shape[0])
        offset=torch.round((self.start+delta-self.context_start)*self.sample_rate).long()
        index=offset[:,None]+torch.arange(self.n,device=context.device)
        if index.min()<0 or index.max()>=context.shape[-1]:
            raise ValueError('Requested crop exceeds recorded context')
        return context.gather(2,index[:,None].expand(-1,context.shape[1],-1))


def synthetic_archive(path, seed=7, classes=16, channels=8, sample_rate=100):
    """Smoke fixture only. Planted random spatial patterns are NOT language embeddings."""
    rng = np.random.default_rng(seed)
    words = np.array([f"synthetic_{i}" for i in range(classes)])
    embeddings = rng.normal(size=(classes, 32)).astype("float32")
    patterns = rng.normal(size=(classes, channels))
    times = np.arange(round(2 * sample_rate)) / sample_rate - 0.6
    xs, ys, splits = [], [], []
    for split, repeats in [("train", 10), ("val", 3), ("test", 3)]:
        for cls in range(classes):
            for _ in range(repeats):
                latency = rng.uniform(0.05, 0.7)
                pulse = np.exp(-((times - latency) / 0.08) ** 2)
                x = rng.normal(scale=0.5, size=(channels, len(times)))
                x += 2 * patterns[cls, :, None] * pulse
                xs.append(x)
                ys.append(cls)
                splits.append(split)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=np.asarray(xs, dtype="float32"), y=np.asarray(ys),
                        split=np.array(splits), embeddings=embeddings, words=words,
                        sample_rate=sample_rate, context_start=-0.6)
    Path(str(path) + ".json").write_text(json.dumps({"synthetic": True, "seed": seed,
        "warning": "Integration fixture; no scientific performance claim"}, indent=2))
