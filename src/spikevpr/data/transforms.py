"""
Event and voxel-grid transforms.

Two representations feed the same 2-channel (ON/OFF) event frame to the model:

  * Raw event streams (Brisbane, NSAVP) are turned into a single frame with
    tonic's ``ToFrame`` plus the helpers below.
  * Voxel grids (NYC) are collapsed from (T, 1, H, W) to (2, H, W) here.

All datasets ultimately produce a (2, 260, 346) tensor.
"""
import random

import numpy as np
import torch
import tonic.transforms as T


# ── raw event-stream helpers ────────────────────────────────────────────────────

class EventDilation:
    """
    EventDilation augmentation (Keime et al., 2026).

    Crops the raw event stream to a temporal window of *random* length, placed at
    a random start. This simulates different accumulation durations / traversal
    speeds, making the learned descriptor robust to speed and temporal variation.
    It is applied before frame creation.

    ``min_window`` / ``max_window`` are in the event array's native time units
    (seconds for Brisbane; for NSAVP the equivalent is bundled into
    ``RandomTimewindowToFrame``).
    """

    def __init__(self, min_window, max_window):
        self.min_window = min_window
        self.max_window = max_window

    def __call__(self, events):
        t = events["t"]
        if len(t) == 0:
            return events
        span = random.uniform(self.min_window, self.max_window)
        t0, t_max = float(t.min()), float(t.max())
        latest_start = max(t0, t_max - span)
        start = random.uniform(t0, latest_start) if latest_start > t0 else t0
        return events[(t >= start) & (t < start + span)]


class SqueezeTransform:
    """Drop the singleton time axis left by tonic ``ToFrame`` (-> (2, H, W) tensor)."""

    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, events):
        if isinstance(events, np.ndarray):
            events = torch.from_numpy(events)
        return torch.squeeze(events, self.dim) if self.dim is not None else torch.squeeze(events)


class RandomTimewindowToFrame:
    """
    Accumulate events into frames using a *random* time window, then return one
    randomly chosen frame. Acts as a temporal-dilation augmentation for training.
    """

    def __init__(self, sensor_size, min_time, max_time):
        self.sensor_size = sensor_size
        self.min_time = min_time
        self.max_time = max_time

    def __call__(self, events):
        window = random.uniform(self.min_time, self.max_time)
        frames = T.ToFrame(sensor_size=self.sensor_size, time_window=window)(events)
        return frames[np.random.randint(0, frames.shape[0])]


class FirstTimewindowToFrame:
    """Deterministic counterpart of ``RandomTimewindowToFrame`` (returns frame 0)."""

    def __init__(self, sensor_size, min_time, max_time):
        self.sensor_size = sensor_size
        self.min_time = min_time
        self.max_time = max_time

    def __call__(self, events):
        window = random.uniform(self.min_time, self.max_time)
        frames = T.ToFrame(sensor_size=self.sensor_size, time_window=window)(events)
        return frames[0]


class FirstSingleFrame:
    """Take the first frame of an already-formed (T, 2, H, W) sequence."""

    def __call__(self, sequence):
        return sequence[0, :, :, :]


def to_tonic_format(events):
    """Convert a plain (N, 4) [x, y, t, p] array to a tonic structured array."""
    structured = np.zeros(len(events), dtype=[("t", np.int64), ("x", np.int16),
                                              ("y", np.int16), ("p", np.int8)])
    structured["t"] = events[:, 2].astype(np.int64)
    structured["x"] = events[:, 0].astype(np.int16)
    structured["y"] = events[:, 1].astype(np.int16)
    structured["p"] = events[:, 3].astype(np.int8)
    return structured


# ── voxel-grid helpers (NYC) ─────────────────────────────────────────────────────

class CollapseVoxelGrid:
    """
    Collapse a (T, 1, H, W) voxel grid into a 2-channel event frame (2, H, W).

    ON channel  = sum over T of the positive bin contributions.
    OFF channel = sum over T of the negative bin contributions.
    """

    def __call__(self, vg):
        v = vg[:, 0, :, :]
        on = np.sum(np.maximum(0.0, v), axis=0)
        off = np.sum(np.maximum(0.0, -v), axis=0)
        return np.stack([on, off], axis=0).astype(np.float32)


class VoxelGridTemporalDilation:
    """Keep a random contiguous run of >= ``t_min`` temporal bins (train-time aug)."""

    def __init__(self, t_min=5, p=0.5):
        self.t_min = t_min
        self.p = p

    def __call__(self, vg):
        if np.random.random() >= self.p:
            return vg
        T_ = vg.shape[0]
        t_len = np.random.randint(self.t_min, T_ + 1)
        t_start = np.random.randint(0, T_ - t_len + 1)
        return vg[t_start:t_start + t_len]


class VoxelGridEventDrop:
    """Zero out a random rectangle across all temporal bins (train-time aug)."""

    def __init__(self, h_frac_range=(0.05, 0.3), w_frac_range=(0.05, 0.3), p=0.5):
        self.h_frac_range = h_frac_range
        self.w_frac_range = w_frac_range
        self.p = p

    def __call__(self, vg):
        if np.random.random() >= self.p:
            return vg
        H, W = vg.shape[2], vg.shape[3]
        h_size = max(1, int(H * np.random.uniform(*self.h_frac_range)))
        w_size = max(1, int(W * np.random.uniform(*self.w_frac_range)))
        h_start = np.random.randint(0, H - h_size + 1)
        w_start = np.random.randint(0, W - w_size + 1)
        vg = vg.copy()
        vg[:, :, h_start:h_start + h_size, w_start:w_start + w_size] = 0.0
        return vg


class VoxelGridFlipLR:
    """Left-right flip of the voxel grid (train-time aug)."""

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, vg):
        if np.random.random() >= self.p:
            return vg
        return np.ascontiguousarray(np.flip(vg, axis=-1))


class ComposeVoxelGrid:
    """Sequential composition of voxel-grid transforms (numpy in, numpy out)."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, vg):
        for t in self.transforms:
            vg = t(vg)
        return vg
