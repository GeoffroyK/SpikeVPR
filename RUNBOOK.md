# End-to-end runbook

From raw data to a trained/evaluated model, for each dataset. All commands run
from `src/`. Replace `<dataset>` with `brisbane`, `nsavp` or `nyc`.

## 0. Install

```bash
cd src
pip install -e .              # or: pip install -r requirements.txt
```

## 1. Get the data and point the config at it

Full layouts and sources are in [DATASETS.md](DATASETS.md). Each
`configs/<dataset>.yaml` has a `dataset_paths` block to edit.

### Brisbane
1. Download Brisbane-Event-VPR (per-traverse event files + `.nmea` GPS +
   hot-pixel lists).
2. Slice each traverse into per-place `.npy` windows under `events_root/<id>/`:
   ```bash
   python -m tools.slice_brisbane --input_dir <raw_zips_dir> --out_dir SlicedBrisbane
   ```
3. Edit `configs/brisbane.yaml → dataset_paths.brisbane` (`events_root`,
   `gps_root`, `hot_pixels_root`).

### NSAVP
1. Download NSAVP; convert each traverse to per-frame full-res `.npy`
   (`R0_XX-frames-1000/frame_000000.npy`, plain `(N,4)` `[x,y,t,p]`).
2. Downsample 640x480 → 346x260 (creates the `downsampled/` folders the loader
   reads):
   ```bash
   python -m tools.downsample_nsavp nsavp --batch
   ```
3. Place the ground-truth `*_GT.npy` + `*_GT_positions.npz` under
   `nsavp/ground_truth/` (these ship with NSAVP).
4. Edit `configs/nsavp.yaml → dataset_paths.nsavp`.

### NYC
1. Download NYC-Event-VPR raw data (`sensor_data_*` sessions with `data_*.zip`
   EVT3 streams + `GPS_*.csv`).
2. Build the voxel-grid dataset (`(15,1,260,346)` grids packed into
   `images/{train,val,test}/{database,queries}.zip`):
   ```bash
   python -m tools.generate_nyc_voxelgrids \
       --raw_dir NYC-Event-VPR_raw_data --out_dir NYC-Event-VPR_VoxelGrid \
       --work_dir raw_work --voxel_bins 15
   ```
   (needs the data-prep extras: `pip install -e ".[dataprep]"`.)
3. Edit `configs/nyc.yaml → dataset_paths.nyc.root`.

## 2. Get a model — either download or train

### Option A: use the released checkpoints (no training)
```bash
SPIKEVPR_WEIGHTS_URL=https://your-host/spikevpr/weights ./weights/download_weights.sh
```

### Option B: train from scratch (InfoNCE)
```bash
python -m spikevpr.training.train \
    --dataset <dataset> --config configs/<dataset>.yaml \
    --encoder sew_resnet34 \
    --output_folder runs/<dataset>_r34
# add --encoder sew_resnet18 for the smaller backbone
# add --mlflow to log metrics
```
Best checkpoint (by val recall@1) → `runs/<dataset>_r34/best_model.pth`.
The MixVPR neuron type comes from the config (`model.aggregator_neuron`:
`LIFNode` for Brisbane/NSAVP, `IFNode` for NYC) — keep it the same for training
and evaluation.

## 3. Evaluate (recall@N)

```bash
python -m spikevpr.evaluation.evaluate \
    --dataset <dataset> --config configs/<dataset>.yaml \
    --encoder sew_resnet34 \
    --checkpoint runs/<dataset>_r34/best_model.pth   # or weights/sew_resnet34_<dataset>.pth
```

## 4. Energy comparison

```bash
python -m spikevpr.energy.compare \
    --dataset <dataset> --config configs/<dataset>.yaml \
    --encoder sew_resnet34 --checkpoint weights/sew_resnet34_<dataset>.pth \
    --netvlad weights/netvlad_weights.pth --wpca weights/wpca_weights.pth \
    --out results/energy_comparison.json
```

## 5. Ensemble-Event-VPR baseline on NSAVP

```bash
python -m tools.nsavp_to_ensemble --nsavp_base nsavp --out_dir ensemble_nsavp
# then run E2VID reconstruction (printed commands) in the ensemble-event-vpr repo
```

## Tutorial

See `notebooks/tutorial.ipynb` — it loads a model, evaluates a small subset and
estimates energy, end to end, in a few minutes on CPU.
