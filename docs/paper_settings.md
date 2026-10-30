# Implementation details

This page explains the settings in [configs/ds004408.json](../configs/ds004408.json) and [configs/paper.json](../configs/paper.json), used by `scripts/run_paper.py`.

| Setting in manuscript | Code/configuration |
|---|---|
| Broderick / ds004408 1.0.8, 19 participants, 128 EEG channels | `configs/ds004408.json`; `bids.catalog` records actual source exclusions |
| Bandpass 1–30 Hz; resample 128 Hz | `low=1`, `high=30`, `sample_rate=128`; actual FIR and resampling parameters are used in preprocessing and its cache identity |
| Event-relative epoch [−0.2, 1.0] seconds | `epoch_start=-0.2`, `epoch_end=1.0`; training, evaluation and benchmarking read those coordinates |
| Subject-seen, run-disjoint 16/2/2 | `split_counts=[16,2,2]`; one shared assignment per audio run across subjects |
| Five repeated run-level splits | `configs/paper.json` and `scripts/run_paper.py`; a separate vocabulary and prepared archive per split |
| 250 most frequent training words | `candidates=250`, `min_train=1`, `strict_candidates=true`; rank independent training word occurrences, not repeated subject rows |
| Frozen T5-large | `google-t5/t5-large`, immutable revision `150ebc2c4b72291e770f58e6057481c8d2ed331a`; encoder-only word representations, 1024 dimensions |
| AdamW, lr 3e−4, batch 64, weight decay 0.01 | Dataset configuration and training defaults; optimizer receives configured weight decay |
| Up to 100 epochs with early stopping | `epochs=100`, validation-based stopping and reusable completion marker |
| 26 latency offsets, 0–1000 ms in 40-ms increments | `latency_min=0`, `latency_max=1`, `latency_bins=26` |
| σ=80 ms, β=1, τs=1 | `prior_sigma=.08`, `prior_beta=1`, `content_temperature=1` |
| K=3, 120-ms NMS exclusion, 160-ms crops | `moments=3`, `nms_interval=.12`, `local_width=.16`; exclusion distance is independent of crop width |
| Hard forward selection, differentiable soft backward selection | Straight-through candidate-window weights during training; deterministic NMS at inference |
| Fine encoder uses Transformer and temporal refinement | Shared `LocalTransformer` between temporal convolutions; fusion includes global, coarse-weighted and local representations |
| Signal-coordinate shifts, anchor fixed, Δ∈[−80,80] ms | `shift_convention=signal`, `max_shift=.08`; crop earlier for positive signal shift and invert the offset passed to anchor-coordinate transport |
| JS on normalized common valid temporal support | `shift_support=overlap`; normalize both distributions on their common valid support before JS |
| Vocabulary CE in Eq. (10) | `alignment_views=original` |
| Balanced Top-10/5/1, mean ± sample SD over five splits | `paper_statistics.summarize`; incomplete sets are not averaged or filled with manuscript values |
| Paired, two-sided permutation over 19 subjects, Holm across three metrics | `paper_statistics.compare`; average each subject's paired scores over five splits before exact 2^19 sign-flip enumeration |
| Selected versus random moment occlusion with equal coverage | `neuromoment.occlusion`; match the actual union of all three windows per trial, including overlaps |

## Additional implementation choices

The configuration explicitly records the following implementation defaults; these should not be confused with experimentally recovered hyperparameters:

- Split seeds: 7, 17, 27, 37, 47; training initialization seed: 7. The split seeds control run assignments; the training seed controls model initialization. The manuscript does not specify the original split seeds.
- Early-stopping patience: 10 validation epochs, min delta 0; monitor balanced Top-10 with balanced MRR as a tie-breaker.
- Hidden dimension 32, one coarse Transformer layer, one local Transformer layer, four attention heads, fixed-Gaussian center 300 ms, straight-through temperature 1, λs=0.2 and λi=0.1.
- T5 isolated-word pooling: attention-mask mean of encoder hidden states, then L2 normalization. No sentence context or decoder prompt is used. The paper did not specify pooling.
- 160 ms at 128 Hz rounds to 20 samples (156.25 ms); the 1.2-second epoch rounds to 154 samples. NMS uses actual token-coordinate distances, so effective minimum spacing is quantized by the token stride. The ±80-ms shift bound uses ±10 samples (±78.125 ms) on the 128-Hz grid. These discretization effects are recorded rather than claiming exact noninteger sample lengths.
- A [−0.28,1.08] second stored context supplies shift margins. A two-second edge guard is used with finite-support segmentwise filtering; this is stricter than simply retaining epochs within runs. Bad-channel interpolation, average referencing and train-only standardization are retained and recorded.
- Overlap JS can be undefined for distributions with no common probability support. The code raises an error instead of silently dropping such pairs. The configured workflow uses common-support normalization.

## Coverage audit on the downloaded dataset

The five run assignments were audited without training. Every split has 16 training, 2 validation and 2 test runs, 19 retained subjects, and a 250-word training vocabulary. Observed test-label classes are **161, 142, 157, 144 and 162 / 250**, respectively, for the seeds above. Balanced metrics therefore average over observed classes; uniform Top-10 chance for the candidate pool is still 4%. These counts are not experimental retrieval results.

## Available evaluation workflows

The repository provides five-split retrieval summaries, subject-paired permutation tests, component ablations, prior shuffling, and selected-versus-random moment masking. Ridge, EEGNet and a BrainModule adaptation are available as comparison implementations. Commands are provided in [reproducibility.md](reproducibility.md).
