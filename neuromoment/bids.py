"""Dataset-specific annotation adapters. EEG timestamps remain in the source recording clock."""
from collections import Counter, defaultdict
import csv
import re
from pathlib import Path
import wave

import numpy as np

from .protocol import normalize_word


def read_table(path, delimiter='\t'):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def textgrid_words(path):
    """Parse this dataset's Praat long-format IntervalTier, including escaped double quotes."""
    text = Path(path).read_text(encoding='utf-8-sig')
    if 'Object class = "TextGrid"' not in text:
        raise ValueError('Expected long-format TextGrid')
    duration = float(re.search(r'^\s*xmax\s*=\s*([\d.eE+-]+)', text, re.M)[1])
    blocks = re.split(r'^\s*item \[\d+\]:\s*$', text, flags=re.M)
    tiers = [b for b in blocks[1:] if re.search(r'name\s*=\s*"words"', b)]
    if len(tiers) != 1:
        raise ValueError('Expected exactly one words tier')
    pattern = r'intervals \[(\d+)\]:\s*xmin\s*=\s*([\d.eE+-]+)\s*xmax\s*=\s*([\d.eE+-]+)\s*text\s*=\s*"((?:""|[^"])*)"'
    intervals = re.findall(pattern, tiers[0])
    declared = int(re.search(r'intervals:\s*size\s*=\s*(\d+)', tiers[0])[1])
    if len(intervals) != declared:
        raise ValueError('TextGrid interval parser count mismatch')
    words = []; dropped = Counter(); previous = 0.
    for index, start, end, label in intervals:
        start, end = float(start), float(end)
        if start < previous-1e-6 or not 0 <= start <= end <= duration+1e-6:
            raise ValueError('Invalid TextGrid interval order')
        previous = end
        word = normalize_word(label.replace('""', '"'))
        if word and end > start:
            words.append({'event_id': index, 'onset': start, 'duration': end-start, 'word': word})
        else:
            dropped[label] += 1
    return duration, words, dict(dropped)


def wav_duration(path):
    with wave.open(str(path)) as f:
        return f.getnframes()/f.getframerate()


def eeg_channels(header):
    rows = read_table(header.with_name(header.name.replace('_eeg.vhdr', '_channels.tsv')))
    eeg = [r['name'] for r in rows if r['type'].upper() == 'EEG']
    bad = [r['name'] for r in rows if r['type'].upper() == 'EEG' and r.get('status') == 'bad']
    if not eeg:
        raise ValueError('Empty EEG montage')
    return eeg, bad


def catalog(root, dataset, subjects=None, include_excluded=False):
    """Return disjoint source spans and events, without loading continuous EEG samples."""
    import mne
    root = Path(root).resolve(); records = []; skipped = []; channels = None
    subject_filter = set(subjects) if subjects else None
    common = {'normalization': "NFKC lowercase, strip boundary punctuation; preserve internal apostrophe/hyphen"}
    if dataset == 'ds004408':
        annotations = {}
        for p in sorted((root/'stimuli').glob('*.TextGrid')):
            duration, events, dropped = textgrid_words(p)
            audio_duration = wav_duration(p.with_suffix('.wav'))
            if abs(duration-audio_duration) > .1:
                raise ValueError(f'TextGrid/audio duration disagreement: {p}')
            annotations[p.stem] = (min(duration, audio_duration), events, dropped)
        for header in sorted(root.glob('sub-*/eeg/*_eeg.vhdr')):
            subject = header.parts[-3]
            if subject_filter and subject not in subject_filter:
                continue
            run = re.search(r'_run-(\d+)_', header.name)[1]
            segment = f'audio{int(run):02d}'
            duration, template, dropped = annotations[segment]
            names, bad = eeg_channels(header)
            if len(bad) > .2*len(names):
                skipped.append({'subject':subject,'header':str(header),'reason':'excessive_bids_bad_channels','count':len(bad)}); continue
            channels = names if channels is None else channels
            if set(names) != set(channels):
                raise ValueError('Inconsistent EEG channels between recordings')
            raw = mne.io.read_raw_brainvision(header, preload=False, verbose='ERROR')
            if raw.n_times/raw.info['sfreq'] < duration:
                raise ValueError('EEG shorter than stimulus')
            records.append({'header': str(header), 'subject': subject, 'stimulus': 'old_man_and_the_sea',
                'segment': segment, 'bad_channels': bad, 'start': 0., 'end': duration, 'sfreq': float(raw.info['sfreq']),
                'clock': 'Official README: EEG and audio starts aligned; no per-run word triggers available',
                'events': [dict(e, event_id=f"{segment}:{e['event_id']}") for e in template],
                'annotation_dropped': dropped})
            raw.close()
        common['clock_limit'] = 'Uses publisher start alignment; independent audio-channel verification unavailable.'
    elif dataset == 'brennan':
        participants = {r['participant_id']: r for r in read_table(root/'participants.tsv')}
        story = {int(r['Order']): r for r in read_table(root/'sourcedata/AliceChapterOne-EEG.csv', ',')}
        for header in sorted(root.glob('sub-*/eeg/*_eeg.vhdr')):
            subject = header.parts[-3]
            if subject_filter and subject not in subject_filter:
                continue
            participant = participants[subject]
            if not include_excluded and participant['analyzed'] != 'yes':
                skipped.append({'subject': subject, 'reason': participant['exclusion_reason']}); continue
            events_file = header.with_name(header.name.replace('_eeg.vhdr', '_events.tsv'))
            if not events_file.exists():
                skipped.append({'subject': subject, 'reason': 'missing_word_events'}); continue
            rows = read_table(events_file)
            names, bad = eeg_channels(header)
            if len(bad) > .2*len(names):
                skipped.append({'subject':subject,'header':str(header),'reason':'excessive_bids_bad_channels','count':len(bad)}); continue
            channels = names if channels is None else channels
            if set(names) != set(channels):
                raise ValueError('Inconsistent EEG channels between recordings')
            raw = mne.io.read_raw_brainvision(header, preload=False, verbose='ERROR')
            sfreq = float(raw.info['sfreq']); raw_end = raw.n_times/sfreq
            grouped = defaultdict(list)
            for e in rows:
                ref = story[int(e['word_index'])]
                if normalize_word(e['word']) != normalize_word(ref['Word']) or int(e['segment']) != int(ref['Segment']):
                    raise ValueError('Event identity disagrees with original source CSV')
                grouped[int(e['segment'])].append(e)
            spans = []; clock_errors = []; subject_begin = len(records)
            for segment, group in sorted(grouped.items()):
                # Original CSV onsets are within each audio segment. TSV onsets are EEG-clock times.
                offsets = np.array([float(e['onset'])-float(story[int(e['word_index'])]['onset']) for e in group])
                anchor = float(np.median(offsets)); residual = float(np.max(np.abs(offsets-anchor)))
                if residual > .01:
                    clock_errors.append({'segment':segment,'reason':'clock_residual','residual_seconds':residual}); continue
                duration = wav_duration(root/'stimuli'/group[0]['stim_file'])
                # Snap the segment cut to a true EEG sample; preserve word times without a second conversion.
                start = round(anchor*sfreq)/sfreq
                end = min(start+duration, raw_end)
                if start < 0 or end <= start or start+duration > raw_end+.01:
                    clock_errors.append({'segment':segment,'reason':'stimulus_outside_raw','start':start,'duration':duration,'raw_end':raw_end}); continue
                ev = []
                for e in group:
                    word = normalize_word(e['word'])
                    if word:
                        ev.append({'event_id': f"word{int(e['word_index']):05d}", 'word': word,
                                   'onset': float(e['onset']), 'duration': float(e['duration'])})
                records.append({'header': str(header), 'subject': subject, 'stimulus': 'alice_chapter1',
                    'segment': f'segment{segment:02d}', 'bad_channels': bad, 'start': start, 'end': end, 'sfreq': sfreq,
                    'clock': 'TSV onset; source CSV segment onset offsets + WAV duration define cuts',
                    'clock_residual_max_seconds': residual, 'events': ev})
                spans.append((start,end))
            if any(b > c+.002 for (_,b),(c,_) in zip(spans,spans[1:])):
                clock_errors.append({'reason':'overlapping_source_segments'})
            if clock_errors:
                records = records[:subject_begin]
                skipped.append({'subject':subject,'reason':'incompatible_source_timeline','details':clock_errors})
            raw.close()
    else:
        raise ValueError('dataset must be ds004408 or brennan')
    if not records:
        raise ValueError('No eligible recordings')
    if subject_filter:
        missing = subject_filter - {r['subject'] for r in records}
        if missing:
            raise ValueError(f'Requested subjects unavailable or excluded: {sorted(missing)}')
    # Disjoint source segments must agree on event identity across subjects.
    identity = {}
    for record in records:
        for e in record['events']:
            key = (record['stimulus'], e['event_id']); value = (record['segment'],e['word'])
            if key in identity and identity[key] != value:
                raise ValueError('Cross-subject event identity mismatch')
            identity[key] = value
    return records, {'dataset': dataset, 'root': str(root), 'channels': channels,
        'subjects': sorted({r['subject'] for r in records}), 'excluded': skipped,
        'include_excluded': include_excluded, 'bad_channel_limit_fraction':.2,
        'upstream_version':__import__('json').loads((root/'dataset_description.json').read_text()).get('Version'),
        'upstream_commit':__import__('json').loads((root/'download_status.json').read_text()).get('commit') if (root/'download_status.json').exists() else None, **common}


def usable_events(records, guard=2., context_start=-.28, context_end=1.08):
    events = []; dropped = Counter()
    for i, r in enumerate(records):
        for e in r['events']:
            if e['onset']+context_start < r['start']+guard or e['onset']+context_end > r['end']-guard:
                dropped['boundary_context_guard'] += 1; continue
            events.append(dict(e, record=i, subject=r['subject'], stimulus=r['stimulus'], segment=r['segment']))
    return events, dict(dropped)
