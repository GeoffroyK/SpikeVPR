"""
Spatially downsample NSAVP event frames from DVXplorer (640x480) to DAVIS346
(346x260) resolution — the `downsampled/` folders the loaders read.

Each per-frame `.npy` (tonic structured array with fields x, y, t, p) is mapped
to the lower resolution by scaling x/y and snapping to the target grid. Optional
duplicate removal and temporal density filtering follow the EvDownsampling idea
(Ghosh et al., 2024); both are off by default because spatial downsampling alone
is what the released checkpoints were trained on.

Usage:
    # one traverse folder of frame_*.npy  ->  <folder>/downsampled/
    python -m tools.downsample_nsavp nsavp/R0_FS0-frames-1000

    # all traverses under a base directory
    python -m tools.downsample_nsavp nsavp --batch
"""
import argparse
import glob
import os

import numpy as np


def spatial_downsample(events, in_wh=(640, 480), out_wh=(346, 260)):
    """Scale x/y to the target resolution, keeping t and p unchanged."""
    sx = out_wh[0] / in_wh[0]
    sy = out_wh[1] / in_wh[1]
    out = np.empty(len(events), dtype=events.dtype)
    out["x"] = np.clip(np.floor(events["x"] * sx), 0, out_wh[0] - 1).astype(events["x"].dtype)
    out["y"] = np.clip(np.floor(events["y"] * sy), 0, out_wh[1] - 1).astype(events["y"].dtype)
    out["t"] = events["t"]
    out["p"] = events["p"]
    return out


def remove_duplicates(events, time_tolerance=1.0):
    """Drop events at the same (x, y, p) within ``time_tolerance`` of each other."""
    if len(events) == 0:
        return events
    order = np.lexsort((events["t"], events["p"], events["y"], events["x"]))
    e = events[order]
    dup = ((e["x"][1:] == e["x"][:-1]) & (e["y"][1:] == e["y"][:-1])
           & (e["p"][1:] == e["p"][:-1])
           & (np.abs(e["t"][1:].astype(np.float64) - e["t"][:-1].astype(np.float64)) < time_tolerance))
    keep = np.concatenate([[True], ~dup])
    e = e[keep]
    return e[np.argsort(e["t"])]


def _as_structured(events):
    """Accept either a tonic structured array or a plain (N, 4) [x, y, t, p] array."""
    if events.dtype.names is not None:
        return events
    out = np.empty(len(events), dtype=[("x", np.int16), ("y", np.int16),
                                       ("t", np.int64), ("p", np.int16)])
    out["x"] = events[:, 0]
    out["y"] = events[:, 1]
    out["t"] = events[:, 2]
    out["p"] = events[:, 3]
    return out


def downsample_frame(events, in_wh=(640, 480), out_wh=(346, 260), dedup=False):
    events = _as_structured(events)
    out = spatial_downsample(events, in_wh, out_wh)
    return remove_duplicates(out) if dedup else out


def downsample_traverse(traverse_dir, in_wh=(640, 480), out_wh=(346, 260), dedup=False):
    """Write `<traverse_dir>/downsampled/frame_*.npy` for every frame in the traverse."""
    frames = sorted(glob.glob(os.path.join(traverse_dir, "frame_*.npy")))
    if not frames:
        print(f"[skip] no frame_*.npy in {traverse_dir}")
        return
    out_dir = os.path.join(traverse_dir, "downsampled")
    os.makedirs(out_dir, exist_ok=True)
    for path in frames:
        ev = np.load(path, allow_pickle=True)
        ds = downsample_frame(ev, in_wh, out_wh, dedup)
        np.save(os.path.join(out_dir, os.path.basename(path)), ds)
    print(f"[done] {len(frames)} frames -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Downsample NSAVP event frames to 346x260.")
    parser.add_argument("path", help="A traverse folder, or a base dir with --batch.")
    parser.add_argument("--batch", action="store_true",
                        help="Treat path as a base dir and process every *-frames-* subfolder.")
    parser.add_argument("--in_wh", type=int, nargs=2, default=(640, 480))
    parser.add_argument("--out_wh", type=int, nargs=2, default=(346, 260))
    parser.add_argument("--dedup", action="store_true", help="Also remove duplicate events.")
    args = parser.parse_args()

    if args.batch:
        traverses = sorted(d for d in glob.glob(os.path.join(args.path, "*-frames-*"))
                           if os.path.isdir(d))
        if not traverses:
            print(f"No *-frames-* folders under {args.path}")
        for t in traverses:
            downsample_traverse(t, tuple(args.in_wh), tuple(args.out_wh), args.dedup)
    else:
        downsample_traverse(args.path, tuple(args.in_wh), tuple(args.out_wh), args.dedup)


if __name__ == "__main__":
    main()
