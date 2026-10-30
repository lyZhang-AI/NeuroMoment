# Data preparation

The training configuration uses OpenNeuro **ds004408 1.0.8**, with 19 retained participants and 128 EEG channels. Obtain the recordings under the dataset's original terms.

## Download ds004408

```bash
mkdir -p datasets
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OpenNeuroDatasets/ds004408 datasets/ds004408
git -C datasets/ds004408 checkout b2d91784a90c18e1d76bd77dcb2b29b0ce27a0be
python scripts/download_eeg.py --root datasets --dataset ds004408 --workers 8
python scripts/audit_eeg_download.py datasets/ds004408
```

The downloader checks each payload's size and SHA-256 and supports resuming transfers. Metadata cloning alone does not download the EEG recordings. Allow approximately 20.1 GB for source payloads plus space for processed data. Keep `.git/annex/objects` when working-tree files are symlinks.

Set `root` in [configs/ds004408.json](../configs/ds004408.json) to the downloaded dataset directory. Cache T5-large as shown in the [README](../README.md), then run:

```bash
python scripts/run_paper.py --stage prepare
```

## Preprocessing and event windows

- Use TextGrid word onsets and aligned recording/stimulus runs.
- Filter individual stimulus segments at **1–30 Hz** and resample at **128 Hz**. Apply BIDS bad-channel interpolation, average reference and training-only subject standardization; clip standardized samples at ±10.
- The model input is **[−0.2, 1.0) seconds** relative to the target word onset.
- Stored EEG context is [−0.28, 1.08) seconds to supply margins for ±80-ms signal shifts. Stored context and model input are different quantities.
- Exclude events crossing stimulus boundaries, including a two-second preprocessing guard. Generated audit records document retained recordings, participants and exclusions.

## Five run-level splits

[configs/paper.json](../configs/paper.json) specifies split seeds **7, 17, 27, 37, 47**. Each partition assigns **16 training, 2 validation and 2 test runs**. All participants hearing the same stimulus run share its assignment. The training initialization seed is separately set to **7**.

Each split selects 250 candidate words by independent training word occurrences, not by repeated participant rows or test frequencies. Candidate vectors come from the frozen T5-large encoder applied to isolated words. Neither target identity nor sentence context enters the EEG encoder at inference.

Vocabulary coverage varies across splits. Report the number of observed test classes separately from the 250-word retrieval pool. All normalization statistics are fitted on training EEG only.

## Custom recordings

For continuous `[channels, samples]` NPY recordings, see `python -m neuromoment.prepare --help`. A TSV manifest supplies recording, subject, stimulus, segment, segment_start, segment_end, event_id, onset and word. Paths are relative to the manifest; times are recording-clock seconds. Preserve cross-participant stimulus/event identities. Supply candidate `words[V]` and `embeddings[V,D]` without fitting on test context.
