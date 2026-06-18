"""
Dataset and DataLoader construction.

One place that maps a dataset name to its transform pipeline and train/eval
datasets, so training and evaluation share exactly the same data definitions.
Every pipeline ends in a (2, 260, 346) float frame.

``build_datasets(name, config, split_transforms=True)`` returns a dict:

    brisbane / nsavp -> {'train': ds, 'eval': pair_ds}
    nyc              -> {'train': ds, 'eval_ref': ds, 'eval_query': ds}

Pair-style eval datasets ('eval') expose anchor=query, positive=reference.
"""
import tonic
from torch.utils.data import DataLoader

from .brisbane import BrisbaneProcessing, BrisbanePairDataset
from .nsavp import NSAVPDataset
from .nyc import NYCVoxelGridDataset, NYCVoxelGridEvalDataset
from .transforms import (EventDilation, FirstSingleFrame, RandomTimewindowToFrame,
                         CollapseVoxelGrid, VoxelGridTemporalDilation,
                         VoxelGridEventDrop, VoxelGridFlipLR, ComposeVoxelGrid)

SENSOR_SIZE = (346, 260, 2)


# ── transform pipelines ──────────────────────────────────────────────────────────

def brisbane_transforms(denoise=0.1, event_count=15_000, dilation_window=(0.018, 0.048)):
    # A Brisbane slice is one place. ToFrame(event_count=...) accumulates the
    # first `event_count` events into a frame; taking frame[0] with FirstSingleFrame
    # always yields a single (2, H, W) frame. (The old SqueezeTransform produced a
    # (k, 2, H, W) tensor and broke collation whenever a slice held >= 2*event_count
    # events; brisbane slices reach ~29k.)
    #
    # EventDilation (the paper's augmentation) crops each slice to a random
    # temporal window before framing; pass dilation_window=None to disable.
    train_steps = [tonic.transforms.Denoise(filter_time=denoise)]
    if dilation_window is not None:
        train_steps.append(EventDilation(*dilation_window))
    train_steps += [
        tonic.transforms.EventDrop(sensor_size=SENSOR_SIZE),
        tonic.transforms.RandomFlipLR(p=0.3, sensor_size=SENSOR_SIZE),
        tonic.transforms.ToFrame(sensor_size=SENSOR_SIZE, event_count=event_count),
        FirstSingleFrame(),
    ]
    train = tonic.transforms.Compose(train_steps)
    eval_ = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=denoise),
        tonic.transforms.ToFrame(sensor_size=SENSOR_SIZE, event_count=event_count),
        FirstSingleFrame(),
    ])
    return train, eval_


def nsavp_transforms(min_time=18e6, max_time=44e6, eval_window=33e6):
    # Training: random temporal window (data augmentation).
    # Eval: a FIXED window then the first frame -> deterministic, reproducible.
    # (The old eval used a random window length, so the same place produced a
    # different frame on every run.)
    train = tonic.transforms.Compose([
        tonic.transforms.EventDrop(sensor_size=SENSOR_SIZE),
        tonic.transforms.RandomFlipLR(p=0.3, sensor_size=SENSOR_SIZE),
        RandomTimewindowToFrame(sensor_size=SENSOR_SIZE, min_time=min_time, max_time=max_time),
    ])
    eval_ = tonic.transforms.Compose([
        tonic.transforms.ToFrame(sensor_size=SENSOR_SIZE, time_window=eval_window),
        FirstSingleFrame(),
    ])
    return train, eval_


def nyc_transforms(dilation_t_min=5):
    # VoxelGridTemporalDilation is the voxel-grid form of EventDilation: it keeps a
    # random contiguous run of >= dilation_t_min temporal bins before collapsing.
    train = ComposeVoxelGrid([
        VoxelGridTemporalDilation(t_min=dilation_t_min, p=0.5),
        VoxelGridEventDrop(h_frac_range=(0.05, 0.3), w_frac_range=(0.05, 0.3), p=0.5),
        VoxelGridFlipLR(p=0.3),
        CollapseVoxelGrid(),
    ])
    eval_ = ComposeVoxelGrid([CollapseVoxelGrid()])
    return train, eval_


# ── dataset builders ─────────────────────────────────────────────────────────────

def build_datasets(name, config):
    name = name.lower()
    if name == "brisbane":
        return _build_brisbane(config)
    if name == "nsavp":
        return _build_nsavp(config)
    if name == "nyc":
        return _build_nyc(config)
    raise ValueError(f"Unknown dataset '{name}'. Choose from brisbane, nsavp, nyc.")


def _build_brisbane(config):
    # EventDilation augmentation: enabled by default (the paper uses it); set
    # data.event_dilation: false to turn it off, or data.dilation_window to retune.
    dilation = (tuple(config["data"].get("dilation_window", (0.018, 0.048)))
                if config["data"].get("event_dilation", True) else None)
    train_tf, eval_tf = brisbane_transforms(denoise=config["data"].get("denoise", 0.1),
                                            dilation_window=dilation)
    radius = config["data"].get("brisbane_gps_radius", 75)

    train_proc = BrisbaneProcessing.from_config(config, config["data"]["training_traverse"])
    eval_proc = BrisbaneProcessing.from_config(config, config["data"]["val_traverse"])
    train_data, _, train_gps = train_proc.get_split()
    eval_data, _, eval_gps = eval_proc.get_split()

    return {
        "train": BrisbanePairDataset(train_data, train_gps, gps_radius=radius, transform=train_tf),
        "eval": BrisbanePairDataset(eval_data, eval_gps, transform=eval_tf, eval_mode=True),
    }


def _build_nsavp(config):
    # NSAVP EventDilation lives in the time-window-to-frame step: a random window
    # length in [min, max] (ns) for training, a fixed window for deterministic eval.
    win = config["data"].get("dilation_window", (18e6, 44e6))
    eval_window = config["data"].get("eval_window", 33e6)
    train_tf, eval_tf = nsavp_transforms(min_time=win[0], max_time=win[1],
                                         eval_window=eval_window)
    return {
        "train": NSAVPDataset.from_config(config, "train", transform=train_tf),
        "eval": NSAVPDataset.from_config(config, "val", transform=eval_tf),
    }


def _build_nyc(config):
    train_tf, eval_tf = nyc_transforms(dilation_t_min=config["data"].get("dilation_t_min", 5))
    root = config["dataset_paths"]["nyc"]["root"]
    radius = config["data"].get("nyc_geo_threshold", 25)
    return {
        "train": NYCVoxelGridDataset(f"{root}/images/train/database.zip",
                                     geo_threshold=radius, transform=train_tf),
        "eval_ref": NYCVoxelGridEvalDataset(f"{root}/images/val/database.zip", transform=eval_tf),
        "eval_query": NYCVoxelGridEvalDataset(f"{root}/images/val/queries.zip", transform=eval_tf),
    }


def make_loader(dataset, batch_size, shuffle, num_workers=4, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=drop_last)
