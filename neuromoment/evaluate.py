import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import EventArchive
from .model import ModelConfig, NeuroMoment
from .train import evaluate, fingerprint
from .text import sha256
from .controls import ShuffledEEG


def main():
    p = argparse.ArgumentParser(description="Evaluate a locked checkpoint, including anchor perturbations")
    p.add_argument("--data", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True); p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--shifts", nargs="+", type=float, help="Event-anchor offsets in seconds", default=[-0.08,-0.04,0.,0.04,0.08])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--intervention-seed", type=int, default=7)
    p.add_argument("--eeg-shuffle-seed", type=int)
    p.add_argument("--device", default="cpu"); p.add_argument("--shuffled-prior", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(2)
    ckpt = torch.load(a.checkpoint, map_location=a.device, weights_only=True)
    if ckpt["data_sha256"] != fingerprint(a.data):
        raise ValueError("Data archive differs from checkpoint training archive")
    c = ModelConfig(**ckpt["config"])
    ds = EventArchive(a.data, a.split, start=c.start, end=c.end, max_shift=max(abs(d) for d in a.shifts))
    if a.eeg_shuffle_seed is not None:
        ds = ShuffledEEG(ds, a.eeg_shuffle_seed)
    model = NeuroMoment(c).to(a.device); model.load_state_dict(ckpt["model"])
    if a.shuffled_prior and c.mode != "adaptive":
        raise ValueError("Shuffled prior intervention applies to adaptive model only")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    for delta in a.shifts:
        metrics, artifacts = evaluate(model, ds, a.device, batch_size=a.batch_size, delta=delta,
                                     intervention="shuffled" if a.shuffled_prior else None,
                                     intervention_seed=a.intervention_seed)
        results[str(delta)] = metrics
        if a.eeg_shuffle_seed is not None:
            artifacts["eeg_source_index"] = ds.eeg_source_index
        np.savez_compressed(out / f"moments_{delta:+.3f}.npz", **artifacts)
    report = {"evaluation_revision": 2, "equivariance_definition": "absolute-time union-support W1; conditional overlap diagnostic reported separately", "split": a.split, "checkpoint": a.checkpoint, "epoch": ckpt["epoch"],
              "data_sha256": ckpt["data_sha256"], "eeg_shuffle_seed": a.eeg_shuffle_seed, "checkpoint_sha256": sha256(a.checkpoint), "intervention_seed": a.intervention_seed, "intervention": "shuffled" if a.shuffled_prior else None,
              "results": results}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
