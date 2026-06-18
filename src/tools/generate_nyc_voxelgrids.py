"""
Generate the NYC-Event-VPR VoxelGrid dataset from raw Prophesee EVT3 streams.

Two phases:
  1. Per session: stream 33 ms windows with expelliarmus, interpolate GPS at each
     window midpoint, drop near-empty / stationary windows, spatially downsample
     1280x720 -> 346x260, encode each kept window as a voxel grid (T, 1, 260, 346),
     and stage it with a metadata CSV.
  2. Pool all sessions, random 40/30/30 train/val/test split with 10 % queries per
     split, and write the VG-named entries into the split zips.

Output tree (what the loader reads):
  <out_dir>/images/{train,val,test}/{database,queries}.zip

Each entry name encodes UTM/GPS:
  {role}/@{utm_e}@{utm_n}@{zone}@{band}@{lat}@{lon}@@@{heading}@@@@{timestamp}@{md5}@.npy

Extra dependencies (data-prep only): expelliarmus, utm, scipy, pillow.
Ported from NYC-Event-VPR/generate_vg_33ms.py; reuses the spatial downsampler
in tools.downsample_nsavp.

Usage:
    python -m tools.generate_nyc_voxelgrids \
        --raw_dir  NYC-Event-VPR_raw_data \
        --out_dir  NYC-Event-VPR_VoxelGrid \
        --work_dir raw_work --voxel_bins 15
"""
import argparse
import csv
import hashlib
import io
import os
import random
import shutil
import zipfile
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

import numpy as np
import tonic.transforms as tonic_transforms

from .downsample_nsavp import spatial_downsample, remove_duplicates

WINDOW_US = 33_000          # 33 ms accumulation window
ENCODING = "evt3"           # Prophesee EVK4 on all NYC sessions
SENSOR_IN = (1280, 720)     # native resolution (W, H)
SENSOR_OUT = (346, 260)     # target (W, H)
W, H = SENSOR_OUT

STAGING_CSV_HEADER = ["session", "timestamp", "lat", "lon", "heading",
                      "utm_e", "utm_n", "zone_num", "zone_band", "frame_key"]


# ── timestamp helpers ────────────────────────────────────────────────────────────

def _ts_to_us(ts):
    return int(datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S_%f")
               .replace(tzinfo=timezone.utc).timestamp() * 1_000_000)


def _us_to_ts(abs_us):
    dt = datetime.fromtimestamp(abs_us / 1_000_000, tz=timezone.utc)
    return dt.strftime(f"%Y-%m-%d_%H-%M-%S_{abs_us % 1_000_000:06d}")


def _raw_start_us(zip_path):
    """'data_YYYY-MM-DD_HH-MM-SS.zip' -> absolute microseconds since epoch."""
    ts = zip_path.stem[len("data_"):]
    return int(datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S")
               .replace(tzinfo=timezone.utc).timestamp() * 1_000_000)


def load_gps_interpolators(csv_path):
    from scipy.interpolate import interp1d
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append((_ts_to_us(r["Timestamp"]), float(r["Latitude"]),
                         float(r["Longitude"]), float(r["HeadMotion"])))
    rows.sort(key=lambda r: r[0])
    t, lat, lon, hdg = (np.array(c, dtype=np.float64) for c in zip(*rows))
    kw = dict(bounds_error=False, fill_value=np.nan)
    return interp1d(t, lat, **kw), interp1d(t, lon, **kw), interp1d(t, hdg, **kw)


# ── voxel grid ───────────────────────────────────────────────────────────────────

def make_voxel_grid(events, n_bins):
    """Tonic ToVoxelGrid -> (n_bins, 1, H, W) float64. Polarity becomes a signed weight."""
    signed = np.empty(len(events), dtype=[("t", events.dtype["t"]), ("x", events.dtype["x"]),
                                          ("y", events.dtype["y"]), ("p", "<i2")])
    for k in ("t", "x", "y"):
        signed[k] = events[k]
    signed["p"] = events["p"].astype(np.int16) * 2 - 1   # {0,1} -> {-1,+1}
    return tonic_transforms.ToVoxelGrid(sensor_size=(W, H, 2), n_time_bins=n_bins)(signed)


def make_vg_name(role, utm_e, utm_n, zone_num, zone_band, lat, lon, heading, timestamp):
    core = f"{utm_e}@{utm_n}@{zone_num}@{zone_band}@{lat}@{lon}@@@{heading}@@@@{timestamp}"
    return f"{role}/@{core}@{hashlib.md5(core.encode()).hexdigest()}@.npy"


# ── phase 1: per-session accumulation ────────────────────────────────────────────

def process_session(session_dir, work_dir, staging_dir, min_events, min_dist_m, voxel_bins):
    import expelliarmus
    import utm

    name = session_dir.name
    staging_zip = staging_dir / f"{name}.zip"
    staging_csv = staging_dir / f"{name}.csv"
    if staging_csv.exists():
        print(f"  [skip] {name} already staged")
        return

    raw_zips = list(session_dir.glob("data_*.zip"))
    gps_csvs = list(session_dir.glob("GPS_*.csv"))
    if not raw_zips or not gps_csvs:
        print(f"  [warn] missing files in {name}")
        return
    raw_zip_path, gps_csv_path = raw_zips[0], gps_csvs[0]

    raw_dir = work_dir / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip_path) as zf:
        raw_name = next(n for n in zf.namelist() if n.endswith(".raw"))
        raw_path = raw_dir / raw_name
        if not raw_path.exists():
            zf.extract(raw_name, raw_dir)

    lat_f, lon_f, hdg_f = load_gps_interpolators(gps_csv_path)
    start_us = _raw_start_us(raw_zip_path)
    wizard = expelliarmus.Wizard(encoding=ENCODING, fpath=str(raw_path), time_window=WINDOW_US)

    last_utm = None
    frame_idx = n_emitted = 0
    rows = []
    with zipfile.ZipFile(staging_zip, "w", compression=zipfile.ZIP_STORED) as sz:
        for events in wizard.read_time_window():
            if len(events) < min_events:
                frame_idx += 1
                continue
            mid_abs_us = start_us + frame_idx * WINDOW_US + WINDOW_US // 2
            lat, lon, hdg = float(lat_f(mid_abs_us)), float(lon_f(mid_abs_us)), float(hdg_f(mid_abs_us))
            if np.isnan(lat) or np.isnan(lon):
                frame_idx += 1
                continue
            utm_e, utm_n, zone_num, zone_band = utm.from_latlon(lat, lon)
            if last_utm is not None and sqrt((utm_e - last_utm[0]) ** 2 + (utm_n - last_utm[1]) ** 2) < min_dist_m:
                frame_idx += 1
                continue

            events = remove_duplicates(spatial_downsample(events, SENSOR_IN, SENSOR_OUT))
            buf = io.BytesIO()
            np.save(buf, make_voxel_grid(events, voxel_bins))
            frame_key = f"{frame_idx:08d}.npy"
            sz.writestr(frame_key, buf.getvalue())

            rows.append(dict(session=name, timestamp=_us_to_ts(mid_abs_us), lat=lat, lon=lon,
                             heading=hdg, utm_e=utm_e, utm_n=utm_n, zone_num=zone_num,
                             zone_band=zone_band, frame_key=frame_key))
            last_utm = (utm_e, utm_n)
            n_emitted += 1
            frame_idx += 1

    with open(staging_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STAGING_CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    shutil.rmtree(raw_dir, ignore_errors=True)
    print(f"  {name}: {n_emitted} frames emitted")


# ── phase 2: split + write final zips ────────────────────────────────────────────

def write_splits(staging_dir, out_dir, seed):
    rng = random.Random(seed)
    all_rows = []
    for csv_path in sorted(staging_dir.glob("*.csv")):
        staging_zip = staging_dir / (csv_path.stem + ".zip")
        if not staging_zip.exists():
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                row["_zip"] = str(staging_zip)
                all_rows.append(row)
    print(f"Total frames: {len(all_rows)}")
    rng.shuffle(all_rows)

    n = len(all_rows)
    splits = {"train": all_rows[:int(0.4 * n)],
              "val": all_rows[int(0.4 * n):int(0.7 * n)],
              "test": all_rows[int(0.7 * n):]}

    images_dir = out_dir / "images"
    out_zips, staging_cache = {}, {}
    for split in ("train", "val", "test"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        for role in ("database", "queries"):
            out_zips[(split, role)] = zipfile.ZipFile(
                images_dir / split / f"{role}.zip", "w", compression=zipfile.ZIP_STORED)
    try:
        for split_name, rows in splits.items():
            rng.shuffle(rows)
            n_q = max(1, int(0.1 * len(rows)))
            for role, role_rows in (("queries", rows[:n_q]), ("database", rows[n_q:])):
                zf_out = out_zips[(split_name, role)]
                for row in role_rows:
                    if row["_zip"] not in staging_cache:
                        staging_cache[row["_zip"]] = zipfile.ZipFile(row["_zip"], "r")
                    data = staging_cache[row["_zip"]].read(row["frame_key"])
                    name = make_vg_name(role, row["utm_e"], row["utm_n"], row["zone_num"],
                                        row["zone_band"], row["lat"], row["lon"],
                                        row["heading"], row["timestamp"])
                    zf_out.writestr(name, data)
            print(f"  {split_name}: {len(rows) - n_q} database + {n_q} queries")
    finally:
        for zf in (*out_zips.values(), *staging_cache.values()):
            zf.close()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Generate NYC-Event-VPR voxel-grid dataset.")
    parser.add_argument("--raw_dir", required=True, help="Root with sensor_data_* session folders.")
    parser.add_argument("--out_dir", required=True, help="Output root (writes images/{split}/{role}.zip).")
    parser.add_argument("--work_dir", required=True, help="Scratch space for extracted .raw files.")
    parser.add_argument("--voxel_bins", type=int, default=15, help="Temporal bins T (released VG used 15).")
    parser.add_argument("--min_events", type=int, default=1000)
    parser.add_argument("--min_dist_m", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=["all", "accumulate", "split"], default="all")
    args = parser.parse_args()

    raw_dir, out_dir, work_dir = Path(args.raw_dir), Path(args.out_dir), Path(args.work_dir)
    staging_dir = work_dir / f"staging_voxelgrid_T{args.voxel_bins}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    sessions = sorted(d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("sensor_data_"))
    print(f"Found {len(sessions)} sessions; staging -> {staging_dir}")

    if args.phase in ("all", "accumulate"):
        for session_dir in sessions:
            process_session(session_dir, work_dir, staging_dir,
                            args.min_events, args.min_dist_m, args.voxel_bins)
    if args.phase in ("all", "split"):
        print("\n-- Phase 2: split + write VG zips --")
        write_splits(staging_dir, out_dir, args.seed)


if __name__ == "__main__":
    main()
