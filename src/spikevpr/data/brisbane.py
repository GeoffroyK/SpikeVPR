"""
Brisbane-Event-VPR dataset.

The Brisbane dataset is a set of day/night traverses of the same route recorded
with an event camera (Fischer & Milford, 2020). ``BrisbaneProcessing`` loads the
per-traverse event ``.npy`` files, removes hot pixels and aligns every traverse
to the reference traverse by GPS so that frame *i* of every traverse is the same
physical place. ``BrisbanePairDataset`` then exposes the standard SpikeVPR
sample schema:

    {'anchor', 'positive', 'label', 'gps_anchor', 'gps_positive'}

  * train mode: positive = another frame within ``gps_radius`` metres of the anchor.
  * eval  mode: anchor = reference-traverse frame, positive = the aligned frame
    of the query traverse (deterministic, one pair per place).
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .gps import (get_gps, match_x1_to_x2, get_positions, get_raw_gps_timestamps,
                  get_timestamp_matches, haversine_matrix)

# ── traverse metadata ───────────────────────────────────────────────────────────

TRAVERSE_TO_NAME = {
    "dvs_vpr_2020-04-21-17-03-03": "sunset1",
    "dvs_vpr_2020-04-22-17-24-21": "sunset2",
    "dvs_vpr_2020-04-24-15-12-03": "daytime",
    "dvs_vpr_2020-04-27-18-13-29": "night",
    "dvs_vpr_2020-04-28-09-14-11": "morning",
    "dvs_vpr_2020-04-29-06-20-23": "sunrise",
}
NAME_TO_TRAVERSE = {v: k for k, v in TRAVERSE_TO_NAME.items()}

NMEA_FILES = {
    "sunset1": "20200421_170039-sunset1_concat.nmea",
    "sunset2": "20200422_172431-sunset2_concat.nmea",
    "morning": "20200428_091154-morning_concat.nmea",
    "daytime": "20200424_151015-daytime_concat.nmea",
    "sunrise": "20200429_061912-sunrise_concat.nmea",
    "night":   "20200427_181204-night_concat.nmea",
}

# Offset (s) to align event timestamps with the GPS/video clock per traverse.
VIDEO_BEGINNING = {
    "sunset1": 1587452582.35,
    "sunset2": 1587540271.65,
    "daytime": 1587705130.80,
    "morning": 1588029265.73,
    "sunrise": 1588105232.91,
    "night":   1587975221.10,
}

HOT_PIXEL_FILES = {
    "sunset1": "dvs_vpr_2020-04-21-17-03-03_hot_pixels.txt",
    "sunset2": "dvs_vpr_2020-04-22-17-24-21_hot_pixels.txt",
    "sunrise": "dvs_vpr_2020-04-29-06-20-23_hot_pixels.txt",
    "daytime": "dvs_vpr_2020-04-24-15-12-03_hot_pixels.txt",
    "morning": "dvs_vpr_2020-04-28-09-14-11_hot_pixels.txt",
    "night":   "dvs_vpr_2020-04-27-18-13-29_hot_pixels.txt",
}


class BrisbaneProcessing:
    """
    Load and GPS-align a set of Brisbane traverses.

    Args:
        events_root: directory holding one sub-folder per traverse (named by the
            raw ``dvs_vpr_*`` id) of per-frame event ``.npy`` files.
        gps_root: directory holding the ``.nmea`` GPS files.
        hot_pixels_root: directory holding the per-traverse hot-pixel files.
        traverses: traverse names to load; the first is the reference/anchor.
        offset, sampling_rate, n_places: place subsampling controls.
    """

    def __init__(self, events_root, gps_root, hot_pixels_root, traverses,
                 offset=0, sampling_rate=1, n_places="all"):
        self.events_root = events_root
        self.gps_root = gps_root
        self.hot_pixels_root = hot_pixels_root
        self.traverses = traverses
        self.offset = offset
        self.sampling_rate = sampling_rate
        self.n_places = n_places

        self.hot_pixels = {t: np.loadtxt(os.path.join(hot_pixels_root, HOT_PIXEL_FILES[t]),
                                         delimiter=",", dtype=int) for t in traverses}
        self.data = {t: [] for t in traverses}
        self.gps_positions = {}
        self.traverses_paths = {}

        self._match_traverses_gps()
        self._load_events()

    @classmethod
    def from_config(cls, config, traverses):
        paths = config["dataset_paths"]["brisbane"]
        data = config["data"]
        return cls(events_root=paths["events_root"], gps_root=paths["gps_root"],
                   hot_pixels_root=paths["hot_pixels_root"], traverses=traverses,
                   offset=data.get("offset", 0), sampling_rate=data.get("sampling_rate", 1),
                   n_places=data.get("n_places", "all"))

    def _list_files(self, traverse):
        path = os.path.join(self.events_root, NAME_TO_TRAVERSE[traverse])
        return sorted(f for f in os.listdir(path) if f.endswith(".npy"))

    def _match_traverses_gps(self):
        anchor = self.traverses[0]
        for other in (t for t in self.traverses if t != anchor):
            self._match_pair(anchor, other)

    def _match_pair(self, t1, t2):
        x1 = get_gps(os.path.join(self.gps_root, NMEA_FILES[t1]))
        x2 = get_gps(os.path.join(self.gps_root, NMEA_FILES[t2]))
        matched = match_x1_to_x2(x1, x2)

        x1 = x1[self.offset::self.sampling_rate]
        x2 = x2[matched][self.offset::self.sampling_rate]
        if self.n_places != "all":
            x1, x2 = x1[:self.n_places], x2[:self.n_places]

        t_gps1 = np.array([t + VIDEO_BEGINNING[t1] for t in get_raw_gps_timestamps(x1)])
        t_gps2 = np.array([t + VIDEO_BEGINNING[t2] for t in get_raw_gps_timestamps(x2)])
        files1 = np.array(self._list_files(t1))
        files2 = np.array(self._list_files(t2))
        places1 = get_timestamp_matches([np.float64(f[:-4]) for f in files1], t_gps1)
        places2 = get_timestamp_matches([np.float64(f[:-4]) for f in files2], t_gps2)

        self.traverses_paths[t1] = [os.path.join(self.events_root, NAME_TO_TRAVERSE[t1], f)
                                    for f in files1[places1]]
        self.traverses_paths[t2] = [os.path.join(self.events_root, NAME_TO_TRAVERSE[t2], f)
                                    for f in files2[places2]]
        self.gps_positions[t1] = get_positions(x1)
        self.gps_positions[t2] = get_positions(x2)

    def _load_events(self):
        for traverse in self.traverses:
            self.data[traverse] = [self._load_filtered(p, traverse)
                                   for p in self.traverses_paths[traverse]]

    def _load_filtered(self, path, traverse):
        events = np.load(path, allow_pickle=True)
        events = self._filter_hot_pixels(events, traverse)
        structured = np.zeros(events.shape[0], dtype=[("t", "float64"), ("x", "i4"),
                                                      ("y", "i4"), ("p", "i4")])
        structured["t"] = events[:, 0]
        structured["x"] = events[:, 1]
        structured["y"] = events[:, 2]
        structured["p"] = events[:, 3]
        return structured

    def _filter_hot_pixels(self, events, traverse):
        coords = self.hot_pixels[traverse]
        keep = ~np.any(np.all(events[:, [1, 2]] == coords[:, None], axis=2), axis=0)
        return events[keep]

    def get_split(self):
        return (self.data, {t: list(range(len(self.data[t]))) for t in self.traverses},
                self.gps_positions)


class BrisbanePairDataset(Dataset):
    """
    Anchor/positive pairs over GPS-aligned Brisbane traverses.

    Train mode flattens every traverse into a single pool and samples a positive
    within ``gps_radius`` metres of the anchor. Eval mode pairs the reference
    traverse (queries) with the aligned query traverse (references), one pair per
    place, for deterministic recall@N.
    """

    def __init__(self, data, gps_positions, gps_radius=75.0, transform=None, eval_mode=False):
        self.transform = transform
        self.gps_radius = gps_radius
        self.eval_mode = eval_mode
        self.traverses = list(data.keys())

        # Flatten frames and aligned metadata. ``place`` is the within-traverse
        # index: identical places across traverses share it (-> shared label).
        self.events, self.lats, self.lons, self.place = [], [], [], []
        for traverse in self.traverses:
            for i, ev in enumerate(data[traverse]):
                self.events.append(ev)
                self.lats.append(float(gps_positions[traverse][i][0]))
                self.lons.append(float(gps_positions[traverse][i][1]))
                self.place.append(i)
        self.lats = np.asarray(self.lats)
        self.lons = np.asarray(self.lons)
        self.place = np.asarray(self.place)

        if eval_mode:
            self._build_eval_pairs(data, gps_positions)
        else:
            self._build_positive_index()

    def _build_positive_index(self):
        dist = haversine_matrix(self.lats, self.lons)
        np.fill_diagonal(dist, np.inf)
        self.positive_indices = [np.where(row <= self.gps_radius)[0] for row in dist]

    def _build_eval_pairs(self, data, gps_positions):
        ref, qry = self.traverses[0], self.traverses[1]
        n = min(len(data[ref]), len(data[qry]))
        self.eval_ref = data[ref][:n]
        self.eval_qry = data[qry][:n]
        self.eval_ref_gps = np.asarray(gps_positions[ref][:n], dtype=np.float64)
        self.eval_qry_gps = np.asarray(gps_positions[qry][:n], dtype=np.float64)

    def __len__(self):
        return len(self.eval_ref) if self.eval_mode else len(self.events)

    def _frame(self, events):
        return self.transform(events) if self.transform is not None else events

    def __getitem__(self, idx):
        if self.eval_mode:
            return {
                "anchor": self._frame(self.eval_ref[idx]),
                "positive": self._frame(self.eval_qry[idx]),
                "label": idx,
                "gps_anchor": torch.tensor(self.eval_ref_gps[idx]),
                "gps_positive": torch.tensor(self.eval_qry_gps[idx]),
            }

        candidates = self.positive_indices[idx]
        pos_idx = int(np.random.choice(candidates)) if len(candidates) else idx
        return {
            "anchor": self._frame(self.events[idx]),
            "positive": self._frame(self.events[pos_idx]),
            "label": int(self.place[idx]),
            "gps_anchor": torch.tensor([self.lats[idx], self.lons[idx]]),
            "gps_positive": torch.tensor([self.lats[pos_idx], self.lons[pos_idx]]),
        }
