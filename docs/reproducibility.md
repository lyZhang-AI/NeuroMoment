# Training and evaluation

Use [configs/ds004408.json](../configs/ds004408.json) for dataset/model settings and [configs/paper.json](../configs/paper.json) for five run-level data splits. The protocol uses frozen T5-large, 1–30 Hz EEG at 128 Hz, [−0.2,1.0) second inputs, and common-valid-support normalized JS.

## Prepare the text encoder

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('google-t5/t5-large', revision='150ebc2c4b72291e770f58e6057481c8d2ed331a')"
```

Set the dataset `root` in `configs/ds004408.json`, then execute:

```bash
python scripts/run_paper.py --stage prepare
python scripts/run_paper.py --stage train --device cuda:0
python scripts/run_paper.py --stage evaluate --device cuda:0
python scripts/run_paper.py --stage occlusion --device cuda:0
python scripts/run_paper.py --stage report
```

Splits are processed sequentially and each gets a separate training-selected vocabulary, frozen word embedding table and EEG archive. `--models neuromoment` restricts the experiment to the full method. The default model list also contains the matched component ablations.

Outputs live under `data/paper/split_SEED`, `runs/paper/split_SEED`, and `reports/paper/split_SEED`. Training stops at the validation-based criterion or 100 epochs. The completion marker distinguishes early stopping from an interrupted run. Existing incompatible settings are rejected. Prior shuffling is a frozen-weight intervention on the adaptive models.

Implemented external adapters can run on the same prepared splits with `--baselines eegnet`, or `--baselines eegnet brainmodule` after obtaining BrainModule's optional upstream sources. Their architecture and adaptation details are described in the code and third-party notices.

## Selected-moment masking

For each trained split:

```bash
python -m neuromoment.occlusion --data data/paper/split_7/eeg --checkpoint runs/paper/split_7/neuromoment/best.pt --out reports/paper/split_7/occlusion --device cuda:0
```

The selected and random masks use three equal-width windows, respect the same NMS constraint and have exactly equal sample coverage after unioning overlaps. Both replace the same number of all-channel standardized samples by zero. This is input sensitivity, not physiological localization; input-distribution and temporal-position confounds remain.

## Subject-paired significance

```bash
python -m neuromoment.paper_statistics \
  --first runs/paper/split_{7,17,27,37,47}/neuromoment/test/moments_+0.000.npz \
  --second runs/paper/split_{7,17,27,37,47}/content_attention/test/moments_+0.000.npz \
  --out reports/paper/paired_attention.json
```

The brace expansion above assumes Bash. Replace the second path list with matched prediction artifacts for the comparison of interest. Within each split, candidate vocabulary and subject/event identities must match. Subject scores are averaged across five splits before a two-sided paired permutation test. Holm correction is applied across Top-10, Top-5 and Top-1. The permutation units are subjects, not model initializations.

## Experiment records

Save generated run assignments, configurations, checkpoint identities and per-subject outputs with each experiment. The repository distributes code and configurations; trained EEG checkpoints and evaluation outputs are not included.
