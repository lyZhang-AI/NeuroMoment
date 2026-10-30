"""Independently verify prepared-array hashes, event identities and source split isolation."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import EventArchive, archive_fingerprint
from .text import sha256


def verify(path):
    path=Path(path)
    if not path.is_dir():
        splits={s:len(EventArchive(path,s,max_shift=0)) for s in ['train','val','test']}
        return {'format':'legacy_npz','data_sha256':archive_fingerprint(path),'splits':splits}
    spec=json.loads((path/'dataset.json').read_text())
    sources=defaultdict(list)
    for rec in spec['recordings']:
        if sha256(path/rec['path'])!=rec['sha256']:raise ValueError(f'Recording hash mismatch: {rec["path"]}')
        data=np.load(path/rec['path'],mmap_mode='r')
        if list(data.shape)!=rec['shape'] or not np.isfinite(data).all():raise ValueError('Bad EEG array')
        sources[rec['source_header']].append(rec)
    for records in sources.values():
        ordered=sorted(records,key=lambda r:r['source_start_sample'])
        for first,second in zip(ordered,ordered[1:]):
            if first['source_end_sample']>second['source_start_sample']:
                raise ValueError('Overlapping preprocessed source spans')
    with np.load(path/'events.npz',allow_pickle=False) as archive:
        index={name:archive[name] for name in archive.files}
        identity={}
        for i in range(len(index['y'])):
            record=spec['recordings'][int(index['record'][i])]
            if record['split']!=index['split'][i] or record['subject']!=index['subject'][i]:
                raise ValueError('Event/source mismatch')
            key=(str(index['stimulus'][i]),str(index['event_id'][i]))
            value=(str(index['split'][i]),int(index['y'][i]))
            if key in identity and identity[key]!=value:raise ValueError('Cross-subject stimulus identity/split mismatch')
            identity[key]=value
    splits={s:len(EventArchive(path,s,max_shift=0)) for s in ['train','val','test']}
    return {'format':spec['format'],'data_sha256':archive_fingerprint(path),'splits':splits,
            'recordings_sha256_verified':len(spec['recordings']),'independent_events':len(identity),
            'subjects':len(spec['source']['subjects']),'channels':len(spec['channels']),
            'coverage_complete':spec.get('coverage_complete',False),'synthetic':spec['synthetic']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--out')
    a=p.parse_args();report=verify(a.data)
    if a.out:Path(a.out).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
