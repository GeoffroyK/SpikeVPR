# Datasets

SpikeVPR is trained and evaluated on three event-camera VPR datasets. Every
pipeline produces a **(2, 260, 346)** ON/OFF event frame for the model. Set the
paths in the corresponding `configs/*.yaml` to point at your local copy.

The layouts below are the ones the loaders expect (and were verified against the
local copies used for the release).

---

## Brisbane-Event-VPR

Source: Fischer & Milford, *Event-Based Visual Place Recognition With Ensembles
of Temporal Windows* (RA-L 2020).
Download: https://open.qcr.ai/dataset/brisbane_event_vpr_dataset/ ·
ensemble baseline: https://github.com/Tobias-Fischer/ensemble-event-vpr

Six traverses of one route under different lighting: `sunset1, sunset2, daytime,
morning, sunrise, night`.

Expected layout (`configs/brisbane.yaml → dataset_paths.brisbane`):

```
events_root/                       # e.g. SlicedBrisbane/
  dvs_vpr_2020-04-21-17-03-03/     # one folder per traverse (raw dvs_vpr id)
    1587452584.644198.npy          # one file per event slice, named <timestamp>.npy
    ...                            #   each .npy is an (N, 4) array of [t, x, y, p]
  dvs_vpr_2020-04-22-17-24-21/
  ...
gps_root/                          # e.g. brisbane_dataset/
  20200421_170039-sunset1_concat.nmea   # one NMEA GPS file per traverse
  ...
hot_pixels_root/                   # e.g. brisbane_hot_pixels/
  dvs_vpr_2020-04-21-17-03-03_hot_pixels.txt   # "x,y" per line, removed at load
  ...
```

Reconstruction: download the Brisbane-Event-VPR rosbags, slice each continuous
event stream into per-window `.npy` files named by their absolute timestamp, and
export the per-traverse NMEA GPS and hot-pixel lists. `BrisbaneProcessing` then
GPS-aligns the traverses (frame *i* of every traverse = same place) using the
`VIDEO_BEGINNING` offsets in `spikevpr/data/brisbane.py`.

## NSAVP

Source: Carmichael et al., *NSAVP: Novel Sensors for Autonomous Vehicle
Perception* (2024). Repeated forward (`F*`) and reverse (`R*`) routes; suffixes
`S/A/N` = sunset/afternoon/night.
Download: https://umautobots.github.io/nsavp ·
ground-truth tooling (Event-LAB): https://github.com/EventLAB-Team/Event-LAB

Expected layout (`configs/nsavp.yaml → dataset_paths.nsavp`):

```
nsavp/
  R0_FS0-frames-1000/
    downsampled/                   # use this (346x260); parent holds full-res 640x480
      frame_000000.npy             # one file per frame, (N, 4) events
      ...
  R0_FA0-frames-1000/downsampled/
  ...
  ground_truth/
    R0_FS0_R0_FA0_GT.npy           # ground-truth place-correspondence matrix
    R0_FS0_R0_FA0_GT_positions.npz # EDEF metric positions: {'ref_pos', 'qry_pos'}
    ...
```

Each config lists `data_train_folders` / `data_val_folders` (first = reference,
rest = query), plus one `gt_*` matrix and one `gps_*` positions file per query
traverse. Places are binned every `nsavp_geo_threshold` metres and stationary
segments are filtered out (`NSAVPDataset`).

Frame formats: the full-res `frame_*.npy` files are plain `(N, 4)` `[x, y, t, p]`
arrays; the `downsampled/` copies are tonic **structured** arrays
(fields `x, y, t, p`) with `t` in **nanoseconds** (≈1 s of events per file). The
ground-truth matrix rows = reference frames, columns = query frames, and
`ref_pos` / `qry_pos` align with those (verified: e.g. R0_FS0↔R0_FA0 = 910×1047).

Reconstruction: download NSAVP, convert each traverse to per-frame event `.npy`
files, then build the 346×260 `downsampled/` folders the loaders read:

```bash
python -m tools.downsample_nsavp nsavp --batch          # 640x480 -> 346x260
```

Export the GT correspondence matrices and EDEF positions per traverse pair (these
ship with NSAVP). For the Ensemble-Event-VPR baseline, convert to its text format:

```bash
python -m tools.nsavp_to_ensemble --nsavp_base nsavp --out_dir ensemble_nsavp
```

## NYC-Event-VPR (VoxelGrid)

Source: NYC-Event-VPR — large-scale event VPR across many recording days.
Frames are stored as voxel grids inside per-split zip archives.
Download: https://ai4ce.github.io/NYC-Event-VPR/

Expected layout (`configs/nyc.yaml → dataset_paths.nyc.root`):

```
NYC-Event-VoxelGrid/
  images/
    train/   database.zip   queries.zip
    val/     database.zip   queries.zip
    test/    database.zip   queries.zip
```

Each zip holds `.npy` voxel grids of shape **(15, 1, 260, 346)** float64. GPS is
encoded in each entry name as UTM east/north (and lat/lon, heading, timestamp):

```
{role}/@{utm_e}@{utm_n}@{zone}@{band}@{lat}@{lon}@@@{heading}@@@@{YYYY-MM-DD_hh-mm-ss_xxx}@{hash}@.npy
```

The loader collapses each voxel grid to a (2, 260, 346) ON/OFF frame.
`recall_at_n_nyc` reports both standard recall and **strict** (cross-session)
recall, which masks same recording-day database frames — the honest metric for a
random frame-level split.

> The `val/database.zip` archive is large (~150 GB); listing it and a full recall
> sweep take time. The tutorial uses small subsets.

Reconstruction: download NYC-Event-VPR and build the voxel-grid representation
(15 temporal bins) per frame, packing each split's references/queries into
`database.zip` / `queries.zip` with the `@`-delimited UTM filename convention above.
