"""
Export NSAVP traverses to the ensemble-event-vpr text format.

The Ensemble-Event-VPR baseline (Fischer & Milford, 2020,
https://github.com/Tobias-Fischer/ensemble-event-vpr) reconstructs images with
E2VID over several temporal windows and matches them with image VPR. Its
reconstruction reader expects one text file per traverse:

    line 1 : <width> <height>
    line k : <t_seconds> <x> <y> <polarity>

This script reads the NSAVP `downsampled/frame_*.npy` files (tonic structured
arrays with t in nanoseconds) and writes the equivalent per-traverse text files.

Usage:
    python -m tools.nsavp_to_ensemble --nsavp_base nsavp --out_dir ensemble_nsavp
    python -m tools.nsavp_to_ensemble --traverses R0_FS0 R0_FA0 --compress
"""
import argparse
import glob
import os
import zipfile

import numpy as np

SENSOR_W, SENSOR_H = 346, 260
DEFAULT_TRAVERSES = ["R0_FS0", "R0_FA0", "R0_FN0", "R0_RS0", "R0_RA0", "R0_RN0"]

# E2VID windows used by the original ensemble (event-count and fixed-duration).
EVENTS_PER_PIXEL = [0.1, 0.3, 0.6, 0.8]
DURATIONS_MS = [44, 66, 88, 120, 140]


def _traverse_dir(nsavp_base, name):
    return os.path.join(nsavp_base, f"{name}-frames-1000", "downsampled")


def convert_traverse(name, traverse_dir, out_dir, start_frame=0, end_frame=None, compress=False):
    frames = sorted(glob.glob(os.path.join(traverse_dir, "frame_*.npy")))[start_frame:end_frame]
    if not frames:
        print(f"[skip] no frames for {name} in {traverse_dir}")
        return
    txt_path = os.path.join(out_dir, f"{name}.txt")
    with open(txt_path, "w", buffering=8 * 1024 * 1024) as f:
        f.write(f"{SENSOR_W} {SENSOR_H}\n")
        for frame_path in frames:
            ev = np.load(frame_path, allow_pickle=True)
            if len(ev) == 0:
                continue
            # t stored in nanoseconds -> seconds for the ensemble reader.
            rows = np.column_stack([ev["t"].astype(np.float64) / 1e9,
                                    ev["x"].astype(np.int32),
                                    ev["y"].astype(np.int32),
                                    ev["p"].astype(np.int32)])
            np.savetxt(f, rows, fmt=["%.9f", "%d", "%d", "%d"])
    print(f"[done] {name}: {len(frames)} frames -> {txt_path} ({os.path.getsize(txt_path)/1e9:.2f} GB)")

    if compress:
        zip_path = os.path.join(out_dir, f"{name}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write(txt_path, os.path.basename(txt_path))
        print(f"       compressed -> {zip_path}")


def print_e2vid_commands(out_dir, example="R0_FS0"):
    print("\nNext, run E2VID reconstructions (in the ensemble-event-vpr repo):")
    for nep in EVENTS_PER_PIXEL:
        print(f"  run_reconstruction.py -i {out_dir}/{example}.txt "
              f"--num_events_per_pixel {nep} --output_folder N_{nep}/{example} --dataset_name {example}")
    for dur in DURATIONS_MS:
        print(f"  run_reconstruction.py -i {out_dir}/{example}.txt "
              f"--fixed_duration --window_duration {dur} --output_folder t_{dur}/{example} --dataset_name {example}")


def main():
    parser = argparse.ArgumentParser(description="Export NSAVP to ensemble-event-vpr text format.")
    parser.add_argument("--nsavp_base", default="nsavp", help="Base dir holding *-frames-1000 folders.")
    parser.add_argument("--out_dir", default="ensemble_nsavp")
    parser.add_argument("--traverses", nargs="+", default=DEFAULT_TRAVERSES)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--compress", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.traverses:
        tdir = _traverse_dir(args.nsavp_base, name)
        if not os.path.isdir(tdir):
            print(f"[skip] {tdir} not found")
            continue
        convert_traverse(name, tdir, args.out_dir, args.start_frame, args.end_frame, args.compress)
    print_e2vid_commands(args.out_dir)


if __name__ == "__main__":
    main()
