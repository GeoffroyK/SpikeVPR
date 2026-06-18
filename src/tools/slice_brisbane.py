"""
Slice raw Brisbane-Event-VPR traverses into per-place event `.npy` files.

Each traverse is distributed as a text/zip event file:

    line 1 : <width> <height>
    line k : <t> <x> <y> <polarity>

This cuts the stream into non-overlapping windows of a fixed number of events
(``width * height * num_events_per_pixel``) and writes each window as
``<events_root>/<traverse_id>/<last_timestamp>.npy`` — a plain `(N, 4)`
`[t, x, y, p]` array, the format ``BrisbaneProcessing`` reads. File names are the
window's last timestamp (matched to GPS by the loader).

Ported from the original ``SlicedBrisbane/slice_dataset.py`` (FixedSizeEventReader
from rpg_e2vid), made self-contained.

Usage:
    python -m tools.slice_brisbane --input_dir <raw_zips> --out_dir SlicedBrisbane
    python -m tools.slice_brisbane --input_dir . --files dvs_vpr_2020-04-21-17-03-03.zip
"""
import argparse
import os

import numpy as np
import pandas as pd

# The six Brisbane traverse recordings (raw event files, .zip or .txt).
DEFAULT_FILES = [
    "dvs_vpr_2020-04-21-17-03-03.zip",
    "dvs_vpr_2020-04-22-17-24-21.zip",
    "dvs_vpr_2020-04-24-15-12-03.zip",
    "dvs_vpr_2020-04-27-18-13-29.zip",
    "dvs_vpr_2020-04-28-09-14-11.zip",
    "dvs_vpr_2020-04-29-06-20-23.zip",
]


def read_sensor_size(path):
    """First line of the event file is '<width> <height>'."""
    head = pd.read_csv(path, sep=r"\s+", header=None, names=["width", "height"],
                       dtype={"width": np.int32, "height": np.int32}, nrows=1)
    return int(head.values[0][0]), int(head.values[0][1])


def fixed_size_windows(path, num_events, start_index=0):
    """Yield successive (num_events, 4) [t, x, y, p] windows from the event file."""
    reader = pd.read_csv(path, sep=r"\s+", header=None, names=["t", "x", "y", "p"],
                         dtype={"t": np.float64, "x": np.int16, "y": np.int16, "p": np.int16},
                         engine="c", skiprows=start_index + 1, chunksize=num_events,
                         memory_map=True)
    for chunk in reader:
        yield chunk.values


def slice_traverse(path, out_dir, num_events_per_pixel=0.35):
    width, height = read_sensor_size(path)
    n = int(width * height * num_events_per_pixel)
    traverse_id = os.path.splitext(os.path.basename(path))[0]
    dest = os.path.join(out_dir, traverse_id)
    os.makedirs(dest, exist_ok=True)
    print(f"{traverse_id}: sensor {width}x{height}, {n} events/window -> {dest}")

    count = 0
    for window in fixed_size_windows(path, num_events=n):
        last_timestamp = window[-1, 0]
        np.save(os.path.join(dest, f"{last_timestamp}.npy"), window)
        count += 1
    print(f"  wrote {count} windows")


def main():
    parser = argparse.ArgumentParser(description="Slice raw Brisbane traverses into per-place .npy files.")
    parser.add_argument("--input_dir", default=".", help="Directory holding the raw traverse files.")
    parser.add_argument("--out_dir", default="SlicedBrisbane", help="Output events_root.")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                        help="Traverse file names within input_dir.")
    parser.add_argument("--events_per_pixel", type=float, default=0.35,
                        help="Window size = width*height*events_per_pixel (default: 0.35).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.files:
        path = os.path.join(args.input_dir, name)
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        slice_traverse(path, args.out_dir, args.events_per_pixel)


if __name__ == "__main__":
    main()
