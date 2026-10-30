"""Check complete transfer status and read every BrainVision header and edge sample."""
import argparse
import csv
import json
from pathlib import Path

import mne


def audit(root):
    status=json.loads((root/'download_status.json').read_text())
    if not status['complete']:
        raise ValueError('Download has not finished')
    checks=[]
    for p in sorted(root.glob('sub-*/eeg/*_eeg.vhdr')):
        raw=mne.io.read_raw_brainvision(str(p),preload=False,verbose='ERROR')
        raw.get_data(start=0,stop=1)
        raw.get_data(start=raw.n_times-1,stop=raw.n_times)
        event=p.with_name(p.name.replace('_eeg.vhdr','_events.tsv'))
        count=None
        if event.is_file():
            with event.open() as f:
                rows=list(csv.DictReader(f,delimiter='\t'))
            count=len(rows)
        checks.append({'subject':p.parts[-3],'header':str(p.relative_to(root)),
            'channels':int(raw.info['nchan']),'sample_rate':float(raw.info['sfreq']),
            'samples':int(raw.n_times),'word_events_present':event.is_file(),'word_event_rows':count})
        raw.close()
    report={'dataset':root.name,'commit':status['commit'],
        'sha256_verified_files':status['files_verified'],'verified_bytes':status['bytes_verified'],
        'all_eeg_headers_and_edge_samples_readable':True,'recordings':len(checks),
        'word_event_files':sum(c['word_events_present'] for c in checks),
        'subjects_without_word_events':sorted({c['subject'] for c in checks if not c['word_events_present']}),
        'textgrid_files':len(list(root.glob('stimuli/*.TextGrid'))),
        'audio_files':len(list(root.glob('stimuli/*.wav'))),'checks':checks,
        'note':'Annotation coverage reflects the pinned upstream repository; ds004408 word times are in TextGrid, not per-recording word events.'}
    (root/'download_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    audit(p.parse_args().root)
