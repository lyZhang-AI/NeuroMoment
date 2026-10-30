# NeuroMoment

**Latency-Adaptive Neural Moment Grounding for EEG-Based Word Decoding**

NeuroMoment combines an EEG-conditioned latency prior with content relevance to select neural moments, then fuses global EEG context and local features for word retrieval. Training uses straight-through moment selection, temporal shift equivariance and embedding consistency.

## Experimental settings

The training workflow uses [configs/ds004408.json](configs/ds004408.json) and [configs/paper.json](configs/paper.json).

| Component | Setting |
|---|---|
| Dataset | ds004408 1.0.8; 19 participants, 128 EEG channels |
| EEG preprocessing | 1–30 Hz bandpass; 128 Hz sampling; training-only standardization |
| Model input | Event-relative [−0.2, 1.0) seconds |
| Evaluation protocol | Five run-level splits; 16/2/2 train/validation/test runs; shared run assignment across participants |
| Text representations | Frozen T5-large; 250 training-selected candidate words |
| Latency prior | 26 offsets from 0 to 1000 ms; Gaussian width 80 ms |
| Moment selection | Top-3; 120-ms NMS exclusion; 160-ms local windows; straight-through gradients during training |
| Shift consistency | Signal shifts within ±80 ms; JS after normalization on common valid temporal support |
| Optimization | AdamW; lr 3e−4; batch 64; weight decay 0.01; up to 100 epochs with early stopping |

The five split seeds are `[7, 17, 27, 37, 47]`; the training initialization seed is `7`. These are separate settings: the five repetitions change the data partition. See [implementation details](docs/paper_settings.md) for configuration keys and sampling discretization.

## Installation

Use Python ≥3.10 and a suitable CPU or CUDA build of PyTorch.

```bash
git clone https://github.com/lyZhang-AI/NeuroMoment.git
cd NeuroMoment
python -m pip install -e '.[eeg,text,download,test]'
```

## Prepare data

Follow [data preparation](docs/data.md), set the dataset `root` in `configs/ds004408.json`, and cache the text encoder:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('google-t5/t5-large', revision='150ebc2c4b72291e770f58e6057481c8d2ed331a')"
python scripts/run_paper.py --stage prepare
```

Each split selects its vocabulary from training events and prepares its own embeddings and EEG archive.

## Train and evaluate

Run the full model and configured component ablations over the five splits:

```bash
python scripts/run_paper.py --stage train --device cuda:0
python scripts/run_paper.py --stage evaluate --device cuda:0
python scripts/run_paper.py --stage occlusion --device cuda:0
python scripts/run_paper.py --stage report
```

Use `--models neuromoment` during training/evaluation to run the full model alone. Commands, output paths and subject-level statistics are described in [the experiment guide](docs/reproducibility.md).

## Tests

```bash
python -m pytest -q
```

The offline tests use generated fixtures. BrainModule integration additionally requires its optional upstream source files.

## Code and data

The repository contains model, preprocessing, training, evaluation and statistics code with experiment configurations. Raw datasets, trained EEG checkpoints and experiment outputs are not distributed.

Project-authored code is [MIT licensed](LICENSE). External data, text models and optional upstream code retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
