"""
NYC-Event-VPR (VoxelGrid) dataset.

Large-scale event VPR across many recording days in New York City. Frames are
stored as voxel grids (T=15, 1, H=260, W=346) inside per-split zip archives:

    <root>/images/{train,val,test}/database.zip   – reference frames
    <root>/images/{train,val,test}/queries.zip    – query frames

GPS is encoded in each filename as UTM east/north; distances are Euclidean
metres. ``NYCVoxelGridDataset`` yields the standard anchor/positive training
schema (positive = frame within ``geo_threshold`` metres); ``NYCVoxelGridEvalDataset``
yields single frames for embedding extraction.
"""
import io
import os
import random
import threading
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset


# ── Python 3.12 zip-compatibility shim ──────────────────────────────────────────
# Python 3.12.4+ added "overlapped entry" and "bad CRC" checks that false-positive
# on the ZIP_STORED archives written by zipfile.writestr(). np.load only reads the
# npy header + payload, so disabling both guards is safe here.
class _PermissiveZipFile(zipfile.ZipFile):
    def _RealGetContents(self):
        super()._RealGetContents()
        for info in self.infolist():
            info._end_offset = None

    def open(self, name, mode="r", pwd=None, *, force_zip64=False):
        f = super().open(name, mode=mode, pwd=pwd, force_zip64=force_zip64)
        f._expected_crc = None
        return f


# ── per-process zip handle cache ────────────────────────────────────────────────
# A forked DataLoader worker inherits the parent's open fd, whose file offset is
# shared across processes; concurrent seek()+read() then corrupts reads. Keying
# the cache on PID forces each worker to open its own private handle.
_zip_cache = threading.local()


def _open_zip(path):
    pid = os.getpid()
    if getattr(_zip_cache, "pid", None) != pid:
        _zip_cache.handles = {}
        _zip_cache.pid = pid
    if path not in _zip_cache.handles:
        _zip_cache.handles[path] = _PermissiveZipFile(path, "r")
    return _zip_cache.handles[path]


# ── filename parsing ─────────────────────────────────────────────────────────────

def _parse_gps(name):
    """Return (utm_e, utm_n, lat, lon) from a VG zip entry name."""
    tokens = name.split("/")[-1].split("@")
    return float(tokens[1]), float(tokens[2]), float(tokens[5]), float(tokens[6])


def _parse_session(name):
    """Return the recording-day 'YYYY-MM-DD' from a VG zip entry name."""
    return name.split("/")[-1].split("@")[-3][:10]


class NYCVoxelGridDataset(Dataset):
    """Anchor/positive pairs from one database.zip (training/validation loss)."""

    def __init__(self, zip_path, geo_threshold=25.0, transform=None):
        self.zip_path = os.path.abspath(zip_path)
        self.geo_threshold = geo_threshold
        self.transform = transform

        zf = _open_zip(self.zip_path)
        self.names = [n for n in zf.namelist() if n.endswith(".npy")]
        self.utm_e = np.empty(len(self.names))
        self.utm_n = np.empty(len(self.names))
        for i, name in enumerate(self.names):
            e, n, _, _ = _parse_gps(name)
            self.utm_e[i], self.utm_n[i] = e, n

        self._build_label_bins()
        self._build_positive_index()

    def _build_label_bins(self):
        n = len(self.names)
        labels = np.full(n, -1, dtype=np.int64)
        assigned = np.zeros(n, dtype=bool)
        bin_id = 0
        for i in range(n):
            if assigned[i]:
                continue
            mask = np.sqrt((self.utm_e - self.utm_e[i]) ** 2
                           + (self.utm_n - self.utm_n[i]) ** 2) <= self.geo_threshold
            labels[mask] = bin_id
            assigned[mask] = True
            bin_id += 1
        self.labels = labels

    def _build_positive_index(self):
        self.positive_indices = []
        for i in range(len(self.names)):
            d = np.sqrt((self.utm_e - self.utm_e[i]) ** 2 + (self.utm_n - self.utm_n[i]) ** 2)
            self.positive_indices.append(np.where((d > 0) & (d <= self.geo_threshold))[0])

    def _load(self, name):
        return np.load(io.BytesIO(_open_zip(self.zip_path).read(name)))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        anchor = self._load(self.names[idx])
        candidates = self.positive_indices[idx]
        if len(candidates) == 0:
            d = np.sqrt((self.utm_e - self.utm_e[idx]) ** 2 + (self.utm_n - self.utm_n[idx]) ** 2)
            d[idx] = np.inf
            pos_idx = int(np.argmin(d))
        else:
            pos_idx = int(random.choice(candidates))
        positive = self._load(self.names[pos_idx])

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)

        return {
            "anchor": torch.from_numpy(anchor).float(),
            "positive": torch.from_numpy(positive).float(),
            "label": int(self.labels[idx]),
            "gps_anchor": torch.tensor([self.utm_e[idx], self.utm_n[idx]], dtype=torch.float64),
            "gps_positive": torch.tensor([self.utm_e[pos_idx], self.utm_n[pos_idx]], dtype=torch.float64),
        }


class NYCVoxelGridEvalDataset(Dataset):
    """Single-frame loader for one zip (database or queries) used for retrieval."""

    def __init__(self, zip_path, transform=None):
        self.zip_path = os.path.abspath(zip_path)
        self.transform = transform
        zf = _open_zip(self.zip_path)
        self.names = [n for n in zf.namelist() if n.endswith(".npy")]
        self.utm_e = np.empty(len(self.names))
        self.utm_n = np.empty(len(self.names))
        self.session = []
        for i, name in enumerate(self.names):
            e, n, _, _ = _parse_gps(name)
            self.utm_e[i], self.utm_n[i] = e, n
            self.session.append(_parse_session(name))

    def _load(self, name):
        return np.load(io.BytesIO(_open_zip(self.zip_path).read(name)))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        vg = self._load(self.names[idx])
        if self.transform is not None:
            vg = self.transform(vg)
        return {
            "frame": torch.from_numpy(vg).float(),
            "gps": torch.tensor([self.utm_e[idx], self.utm_n[idx]], dtype=torch.float64),
            "session": self.session[idx],
        }


@torch.no_grad()
def extract_embeddings_nyc(model, loader, device):
    """Return (embeddings [N, D], gps [N, 2], sessions [N]) for an eval loader."""
    model.eval()
    embeddings, gps, sessions = [], [], []
    for batch in loader:
        embeddings.append(model(batch["frame"].to(device).float()).cpu())
        gps.append(batch["gps"].cpu())
        sessions.extend(batch["session"])
    return torch.cat(embeddings), torch.cat(gps), sessions
