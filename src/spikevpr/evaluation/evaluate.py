"""
Unified evaluation entry point.

``evaluate(dataset, config, model, device)`` builds the dataset's eval loaders,
extracts descriptors and returns a recall@N dict. The same call works for all
three datasets; only the retrieval bookkeeping differs (Brisbane/NSAVP pair the
two traverses, NYC uses separate database/query archives with cross-session
recall).

CLI:
    python -m spikevpr.evaluation.evaluate --dataset brisbane \\
        --config configs/brisbane.yaml --encoder sew_resnet34 \\
        --checkpoint weights/sew_resnet34_brisbane.pth
"""
import argparse

import torch

from ..models import build_spikevpr
from ..data.loaders import build_datasets, make_loader
from ..data.nyc import extract_embeddings_nyc
from .metrics import (extract_pair_embeddings, similarity_matrix,
                      recall_at_n, recall_at_n_nyc)

N_VALUES = (1, 5, 10, 15, 20, 25)


@torch.no_grad()
def evaluate(dataset, config, model, device, batch_size=16, num_workers=4):
    dataset = dataset.lower()
    model.eval()
    datasets = build_datasets(dataset, config)
    threshold = config["evaluation"]["recall_threshold"]

    if dataset in ("brisbane", "nsavp"):
        loader = make_loader(datasets["eval"], batch_size, shuffle=False,
                             num_workers=num_workers)
        q_emb, q_gps, r_emb, r_gps = extract_pair_embeddings(model, loader, device)
        sim = similarity_matrix(q_emb, r_emb)
        return recall_at_n(sim, q_gps, r_gps, threshold=threshold, n_values=N_VALUES)

    # NYC: separate database / query archives, cross-session strict recall.
    ref_loader = make_loader(datasets["eval_ref"], batch_size, shuffle=False,
                             num_workers=num_workers)
    q_loader = make_loader(datasets["eval_query"], batch_size, shuffle=False,
                           num_workers=num_workers)
    r_emb, r_gps, r_sess = extract_embeddings_nyc(model, ref_loader, device)
    q_emb, q_gps, q_sess = extract_embeddings_nyc(model, q_loader, device)
    return recall_at_n_nyc(q_emb, q_gps, r_emb, r_gps, n_values=(1, 5, 10),
                           threshold_m=threshold, query_sessions=q_sess,
                           ref_sessions=r_sess)


def _load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a SpikeVPR checkpoint.")
    parser.add_argument("--dataset", required=True, choices=["brisbane", "nsavp", "nyc"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="sew_resnet34", choices=["sew_resnet18", "sew_resnet34"])
    parser.add_argument("--out_channels", type=int, default=512)
    parser.add_argument("--out_rows", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    config = _load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neuron = config["model"].get("aggregator_neuron", "LIFNode")
    model = build_spikevpr(args.encoder, out_channels=args.out_channels,
                           out_rows=args.out_rows, neuron_type=neuron,
                           checkpoint=args.checkpoint, device=device, eval_mode=True)

    recalls = evaluate(args.dataset, config, model, device,
                       batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"\n{args.dataset} recall@N ({args.encoder}):")
    for k, v in recalls.items():
        print(f"  {k:18s}: {v:.4f}")


if __name__ == "__main__":
    main()
