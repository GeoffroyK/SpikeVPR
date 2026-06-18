"""
Recompute the SpikeVPR (SNN) vs NetVLAD (ANN) energy comparison.

Everything is measured from the actual models and real data — no values are
hardcoded. The SNN spike rate is measured on the dataset's eval frames; the ANN
ReLU sparsity is measured on event tensors fed to the NetVLAD baseline.

CLI:
    python -m spikevpr.energy.compare --dataset nsavp --config configs/nsavp.yaml \\
        --encoder sew_resnet34 --checkpoint weights/sew_resnet34_nsavp.pth \\
        --netvlad weights/netvlad_weights.pth --wpca weights/wpca_weights.pth \\
        --out results/energy_comparison.json
"""
import argparse
import json

import numpy as np
import torch

from ..models import build_spikevpr
from ..data.loaders import build_datasets, make_loader
from ..baselines.netvlad import RetrievalModel
from .estimate import (extract_layers, measure_relu_sparsity,
                       assign_relu_sparsity_to_layers, estimate_model,
                       estimate_snn_energy, print_comparison_table)


def _synthetic_events(n_events=5000, height=260, width=346, device="cpu"):
    """A single-item raw event tensor (N, 5): [x, y, t, p, batch_index]."""
    t = np.sort(np.random.rand(n_events)).astype(np.float32)
    x = np.random.randint(0, width, n_events).astype(np.float32)
    y = np.random.randint(0, height, n_events).astype(np.float32)
    p = np.random.choice([-1.0, 1.0], n_events).astype(np.float32)
    b = np.zeros(n_events, dtype=np.float32)
    return torch.from_numpy(np.stack([x, y, t, p, b], axis=1)).to(device)


class _EventBatches:
    """Iterable of synthetic event tensors for ReLU-sparsity measurement."""

    def __init__(self, n_batches, device):
        self.n_batches = n_batches
        self.device = device

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self._tuple()

    def _tuple(self):
        return (_synthetic_events(device=self.device),)


def _load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Recompute the SNN vs ANN energy comparison.")
    parser.add_argument("--dataset", required=True, choices=["brisbane", "nsavp", "nyc"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="SpikeVPR (SNN) checkpoint.")
    parser.add_argument("--encoder", default="sew_resnet34", choices=["sew_resnet18", "sew_resnet34"])
    parser.add_argument("--out_channels", type=int, default=512)
    parser.add_argument("--out_rows", type=int, default=8)
    parser.add_argument("--netvlad", default=None, help="NetVLAD weights (ANN baseline).")
    parser.add_argument("--wpca", default=None, help="WPCA weights (ANN baseline).")
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--out", default=None, help="Optional path to write JSON results.")
    args = parser.parse_args()

    config = _load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    # ── SpikeVPR (SNN) ───────────────────────────────────────────────────────────
    neuron = config["model"].get("aggregator_neuron", "LIFNode")
    model = build_spikevpr(args.encoder, out_channels=args.out_channels, out_rows=args.out_rows,
                           neuron_type=neuron, checkpoint=args.checkpoint, device=device, eval_mode=True)
    datasets = build_datasets(args.dataset, config)
    eval_ds = datasets.get("eval", datasets.get("eval_query"))
    loader = make_loader(eval_ds, batch_size=16, shuffle=False, num_workers=0)
    snn_result = estimate_snn_energy(model, loader, device, num_batches=args.n_batches)
    snn_result["name"] = f"SpikeVPR ({args.encoder})"
    results.append(snn_result)

    # ── NetVLAD + ResNet-34 (ANN) ────────────────────────────────────────────────
    if args.netvlad and args.wpca:
        netvlad = RetrievalModel.from_weights(args.netvlad, args.wpca, device=device,
                                              voxel_dimension=(9, 260, 346))
        netvlad.eval()
        dummy = _synthetic_events(device=device)
        ann_layers = extract_layers(netvlad, dummy)
        sparsity = measure_relu_sparsity(netvlad, _EventBatches(args.n_batches, device),
                                         device, num_batches=args.n_batches)
        ann_layers = assign_relu_sparsity_to_layers(ann_layers, sparsity)
        results.append(estimate_model("NetVLAD+ResNet34", ann_layers, mode="ann"))
    else:
        print("(skipping ANN baseline: pass --netvlad and --wpca to include it)")

    print_comparison_table(results)

    if args.out:
        with open(args.out, "w") as f:
            json.dump([{**r, "methods": {k: round(v, 2) for k, v in r["methods"].items()}}
                       for r in results], f, indent=2)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
