"""Early stopping on a monitored score (higher is better after sign flip)."""
import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False,
                 path="checkpoint.pth", metric_name="metric"):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.metric_name = metric_name
        self.counter = 0
        self.best_score = None
        self.best_value = np.inf
        self.early_stop = False

    def __call__(self, value, model):
        """``value`` is the *minimised* quantity (e.g. negative recall@1)."""
        score = -value
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self._save(value, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, value, model):
        value = abs(value)
        if self.verbose:
            print(f"  improved {self.metric_name}: {self.best_value:.6f} -> {value:.6f}")
        torch.save(model.state_dict(), self.path)
        self.best_value = value
