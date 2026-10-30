# Third-party components

The root MIT license covers project-authored code, not separately downloaded datasets, text-model weights, or optional upstream source code. This release contains no EEG recordings or pretrained weights.

## Optional BrainModule baseline

The adapter in `neuromoment/baseline_models.py` loads selected original classes from **facebookresearch/brainmagick**, revision `1aa77d2dbd801b4901aeadfd8606d26a89ee7e3e`. Upstream source is licensed under **Creative Commons Attribution–NonCommercial 4.0 International (CC-BY-NC-4.0)**. It is not an unrestricted open-source component and is not included in this distribution.

The optional downloader retrieves the pinned source files and original LICENSE, checks SHA-256, and stores provenance alongside them:

```bash
python scripts/fetch_brainmodule.py
python scripts/run_paper.py --stage train --models neuromoment --baselines brainmodule --device cuda:0
```

Review the downloaded upstream LICENSE before use. The files remain under upstream terms; the root MIT license does not relicense them. `third_party/baseline_sources/` is ignored by Git and excluded from the release archive. For a separate cache, set `NEUROMOMENT_BASELINE_SOURCES` and pass the same directory to the downloader's `--out`.

Source and attribution: https://github.com/facebookresearch/brainmagick/tree/1aa77d2dbd801b4901aeadfd8606d26a89ee7e3e . The adapter executes selected class bodies without changing them, replaces recording/layout access with supplied nominal positions, and adapts the temporal output to frozen word-vector retrieval. It is not a reproduction of the upstream speech-reconstruction protocol.

## EEGNet

The native PyTorch EEGNet-8,2 implementation follows the EEGNet architecture by Lawhern et al. The reference is **vlawhern/arl-eegmodels**, revision `4a512e503198db2010848813ead9afbf8cd54c97` (https://github.com/vlawhern/arl-eegmodels/tree/4a512e503198db2010848813ead9afbf8cd54c97). Its upstream Keras source is not bundled or executed. The original reference license text is retained at `third_party/EEGModels_LICENSE.txt` (CC0 and Apache-2.0 portions). The local implementation uses sampling-rate-dependent temporal kernels and adds a word-vector projection.

## Data and text model

- ds004408 1.0.8: obtain independently from OpenNeuro, retaining its CC0 dataset notice.
- Brennan2019 / NEMAR nm000180 1.1.3: obtain independently, retaining its CC-BY-4.0 attribution and original dataset citation.
- `google-t5/t5-large`: the primary text encoder; obtain its pinned snapshot separately and retain its Apache-2.0 license/model card.

Dependency packages retain their individual licenses. This repository does not grant rights in upstream materials.
