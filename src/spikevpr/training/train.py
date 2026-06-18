"""
Unified SpikeVPR training.

The same loop trains on any of the three datasets: build the model, build the
dataset's train loader, optimise the InfoNCE (NT-Xent) loss, and validate each
epoch with recall@1. The best checkpoint (by val recall@1) is kept via early
stopping.

CLI:
    python -m spikevpr.training.train --dataset brisbane \\
        --config configs/brisbane.yaml --encoder sew_resnet34 \\
        --output_folder runs/brisbane_r34
"""
import argparse
import os

import numpy as np
import torch

from ..models import build_spikevpr, count_parameters
from ..data.loaders import build_datasets, make_loader
from ..evaluation.evaluate import evaluate
from .losses import build_infonce_loss
from .early_stopping import EarlyStopping


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running = 0.0
    for batch in loader:
        optimizer.zero_grad()
        anchor = batch["anchor"].to(device).float()
        positive = batch["positive"].to(device).float()
        label = batch["label"].to(device)

        # Embed anchors and positives together; NT-Xent treats every other sample
        # in the batch as a negative. The backbone resets its state internally.
        embeddings = model(torch.cat([anchor, positive], dim=0))
        labels = torch.cat([label, label], dim=0)
        loss = criterion(embeddings, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        running += loss.item()
    return running / max(1, len(loader))


def train(dataset, config, args, device, mlflow_run=None):
    os.makedirs(args.output_folder, exist_ok=True)
    best_path = os.path.join(args.output_folder, "best_model.pth")

    neuron = config["model"].get("aggregator_neuron", "LIFNode")
    model = build_spikevpr(args.encoder, out_channels=args.out_channels,
                           out_rows=args.out_rows, neuron_type=neuron,
                           checkpoint=args.checkpoint, device=device)
    desc_dim = args.out_channels * args.out_rows
    print(f"Model: {args.encoder} | {count_parameters(model) / 1e6:.1f}M params | {desc_dim}-D descriptor")

    datasets = build_datasets(dataset, config)
    tr = config["training"]
    train_loader = make_loader(datasets["train"], tr["batch_size"], shuffle=True,
                               num_workers=args.num_workers, drop_last=True)

    criterion = build_infonce_loss(temperature=tr["temperature"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"],
                                  weight_decay=tr["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=tr["learning_rate"], steps_per_epoch=len(train_loader),
        epochs=tr["epochs"], pct_start=0.1, anneal_strategy="cos")

    early = EarlyStopping(patience=tr["patience"], verbose=True, path=best_path,
                          metric_name="val_R@1")

    for epoch in range(tr["epochs"]):
        loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        recalls = evaluate(dataset, config, model, device,
                           batch_size=args.eval_batch_size, num_workers=args.num_workers)
        r1 = recalls.get("strict_recall_1", recalls.get("recall_1", 0.0))
        print(f"[{epoch + 1:03d}/{tr['epochs']}] loss={loss:.4f}  val_R@1={r1:.4f}")

        if mlflow_run is not None:
            mlflow_run.log_metrics({"train_loss": loss, "val_recall_1": r1,
                                    "lr": optimizer.param_groups[0]["lr"]}, step=epoch)

        early(-r1, model)
        if early.early_stop:
            print("Early stopping.")
            break

    torch.save(model.state_dict(), os.path.join(args.output_folder, "last_model.pth"))
    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"Best model -> {best_path}")
    return model


def _load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train a SpikeVPR model with triplet loss.")
    parser.add_argument("--dataset", required=True, choices=["brisbane", "nsavp", "nyc"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--encoder", default="sew_resnet34", choices=["sew_resnet18", "sew_resnet34"])
    parser.add_argument("--out_channels", type=int, default=512)
    parser.add_argument("--out_rows", type=int, default=8)
    parser.add_argument("--checkpoint", default=None, help="Resume from a state_dict.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--mlflow", action="store_true", help="Log to MLflow.")
    parser.add_argument("--mlflow_experiment", default="spikevpr")
    args = parser.parse_args()

    set_seed(args.seed)
    config = _load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.mlflow:
        import mlflow
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=f"{args.dataset}_{args.encoder}"):
            mlflow.log_params({"dataset": args.dataset, "encoder": args.encoder,
                               "out_channels": args.out_channels, "out_rows": args.out_rows,
                               "loss": "infonce", "seed": args.seed})
            train(args.dataset, config, args, device, mlflow_run=mlflow)
    else:
        train(args.dataset, config, args, device)


if __name__ == "__main__":
    main()
