"""
NSAVP dataset.

The Novel Sensors for Autonomous Vehicle Perception dataset (Carmichael et al.,
2024) provides repeated forward/reverse routes under varied lighting. Each
traverse is stored as per-frame event ``.npy`` files; ground-truth place
correspondences and EDEF (metric) positions are provided as ``.npy`` / ``.npz``.

The first traverse path is the reference/anchor; the rest are query traverses.
Places are binned every ``geo_threshold`` metres of travelled distance so that
nearby frames share a label, and stationary segments are filtered out. The
sample schema matches the rest of SpikeVPR:

    {'anchor', 'positive', 'label', 'gps_anchor', 'gps_positive'}
"""
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import to_tonic_format


def _edef_distance(p1, p2):
    return np.linalg.norm(p2[:2] - p1[:2])


class NSAVPDataset(Dataset):
    def __init__(self, traverses_paths, gps_positions, ground_truth_matrices,
                 geo_threshold=10, start_offset=0, end_offset=None, transform=None):
        assert len(traverses_paths) - 1 == len(gps_positions), \
            "Need one GPS-positions entry per query traverse"
        assert len(traverses_paths) - 1 == len(ground_truth_matrices), \
            "Need one ground-truth matrix per query traverse"

        self.traverses_paths = traverses_paths
        self.geo_threshold = geo_threshold
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.transform = transform
        self.filtered_index_mapping = None
        self.pairs_count = len(gps_positions)

        self.gps_positions = [dict(np.load(f)) if isinstance(f, str) else f
                              for f in gps_positions]
        self.ground_truth_matrices = [np.load(f, allow_pickle=True) if isinstance(f, str) else f
                                      for f in ground_truth_matrices]

        self._remove_offsets()
        self._cumulative_distance()
        self._filter_stops(min_movement_threshold=1.5)
        if geo_threshold > 0:
            self._index_to_bin_label()

        self.distance_matrices = [
            self._distance_matrix(self.ground_truth_matrices[t],
                                  self.gps_positions[t]["ref_pos"],
                                  self.gps_positions[t]["qry_pos"])
            for t in range(self.pairs_count)
        ]
        self.positive_indices = [self._local_positives(dm, geo_threshold)
                                 for dm in self.distance_matrices]

    # ── geometry helpers ─────────────────────────────────────────────────────────

    def _distance_matrix(self, gt, ref_pos, qry_pos):
        rows, cols = gt.shape
        dm = np.full((rows, cols), np.nan)
        for i in range(rows):
            for j in range(cols):
                dm[i][j] = _edef_distance(ref_pos[i], qry_pos[j])
        return dm

    def _cumulative_distance(self):
        self.cumulative_distances = []
        for gps in self.gps_positions:
            ref = gps["ref_pos"]
            cum, seg = [0], 0.0
            for n in range(len(ref) - 1):
                seg += _edef_distance(ref[n], ref[n + 1])
                cum.append(seg)
            self.cumulative_distances.append(np.array(cum))

    def _create_bins(self, cumulative, bin_size):
        bins = []
        if len(cumulative) == 0:
            return bins
        start = 0
        while start <= cumulative[-1]:
            end = start + bin_size
            members = [i for i, d in enumerate(cumulative) if start <= d < end]
            if members:
                bins.append(members)
            start = end
        return bins

    def _index_to_bin_label(self):
        self.index_to_label = []
        for cum in self.cumulative_distances:
            bins = self._create_bins(cum, self.geo_threshold)
            mapping = np.full(len(cum), -1, dtype=int)
            for bin_id, members in enumerate(bins):
                mapping[members] = bin_id
            self.index_to_label.append(mapping)

    def _filter_stops(self, min_movement_threshold=1.0):
        num_ref = self.ground_truth_matrices[0].shape[0]
        cum = self.cumulative_distances[0][:num_ref]
        kept = [0]
        for i in range(1, len(cum)):
            if cum[i] - cum[kept[-1]] >= min_movement_threshold:
                kept.append(i)
        self.filtered_index_mapping = np.array(kept)

        self.cumulative_distances[0] = cum[kept]
        self.gps_positions[0]["ref_pos"] = self.gps_positions[0]["ref_pos"][kept]
        for t in range(self.pairs_count):
            self.ground_truth_matrices[t] = self.ground_truth_matrices[t][kept, :]
            if t > 0:
                self.cumulative_distances[t] = self.cumulative_distances[t][kept]
                self.gps_positions[t]["ref_pos"] = self.gps_positions[t]["ref_pos"][kept]

    def _remove_offsets(self):
        # NOTE: end_offset defaults to None (keep all frames). The original code
        # defaulted to -1, so the default build silently dropped the last
        # reference frame via [start:-1]. See CHANGES.md.
        for t in range(len(self.ground_truth_matrices)):
            self.ground_truth_matrices[t] = self.ground_truth_matrices[t][self.start_offset:self.end_offset]
            self.gps_positions[t]["ref_pos"] = self.gps_positions[t]["ref_pos"][self.start_offset:self.end_offset]

    def _local_positives(self, distance_matrix, geo_threshold):
        return [np.where(row <= geo_threshold)[0] for row in distance_matrix]

    # ── dataset interface ────────────────────────────────────────────────────────

    def __len__(self):
        return self.ground_truth_matrices[0].shape[0]

    def __getitem__(self, idx):
        traverse = np.random.randint(0, self.pairs_count)
        original_idx = (self.filtered_index_mapping[idx]
                        if self.filtered_index_mapping is not None else idx)

        anchor = np.load(f"{self.traverses_paths[0]}/frame_{original_idx + self.start_offset:06d}.npy",
                         allow_pickle=True)

        if len(self.positive_indices[traverse][idx]) == 0:
            pos_idx = int(np.argmin(self.distance_matrices[traverse][idx]))
        else:
            pos_idx = int(random.choice(self.positive_indices[traverse][idx]))
        positive = np.load(f"{self.traverses_paths[traverse + 1]}/frame_{pos_idx:06d}.npy",
                           allow_pickle=True)
        if len(positive) < 100:  # near-empty frame: fall back to ground-truth match
            pos_idx = int(np.argmin(self.distance_matrices[traverse][idx]))
            positive = np.load(f"{self.traverses_paths[traverse + 1]}/frame_{pos_idx:06d}.npy",
                               allow_pickle=True)

        if anchor.dtype.names is None or positive.dtype.names is None:
            anchor, positive = to_tonic_format(anchor), to_tonic_format(positive)
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)

        if self.geo_threshold > 0:
            label = int(self.index_to_label[traverse][idx]) + traverse * 1000
        else:
            label = idx

        return {
            "anchor": anchor,
            "positive": positive,
            "label": label,
            "gps_anchor": torch.tensor(self.gps_positions[traverse]["ref_pos"][idx]),
            "gps_positive": torch.tensor(self.gps_positions[traverse]["qry_pos"][pos_idx]),
        }

    @classmethod
    def from_config(cls, config, split, transform=None):
        """Build a train/val NSAVP dataset from the config's ``dataset_paths.nsavp``."""
        cfg = config["dataset_paths"]["nsavp"]
        geo = config["data"].get("nsavp_geo_threshold", 10)
        return cls(cfg[f"data_{split}_folders"], cfg[f"gps_{split}_files"],
                   cfg[f"gt_{split}_files"], geo_threshold=geo, transform=transform)
