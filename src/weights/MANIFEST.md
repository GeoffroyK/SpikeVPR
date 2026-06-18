# Model weights

These files are **not tracked in git** (see `.gitignore`). Fetch them with
`./download_weights.sh`, or copy them manually to this folder. SHA-256 checksums
are in `SHA256SUMS.txt` (verify with `sha256sum -c SHA256SUMS.txt`).

All SpikeVPR descriptors are 4096-D (`out_channels=512 × out_rows=8`). Load a
model with the matching `aggregator_neuron` (the MixVPR head neuron type the
checkpoint was trained with):

```python
from spikevpr.models import build_spikevpr
model = build_spikevpr("sew_resnet34", checkpoint="weights/sew_resnet34_nsavp.pth",
                       neuron_type="LIFNode", eval_mode=True)
```

| File                          | Model                         | Dataset  | MixVPR neuron | Size  |
|-------------------------------|-------------------------------|----------|---------------|-------|
| `sew_resnet18_brisbane.pth`   | SEW-ResNet18 + MixVPR (4096-D)| Brisbane | `LIFNode`     | 6.8 MB|
| `sew_resnet34_brisbane.pth`   | SEW-ResNet34 + MixVPR (4096-D)| Brisbane | `LIFNode`     | 12 MB |
| `sew_resnet18_nsavp.pth`      | SEW-ResNet18 + MixVPR (4096-D)| NSAVP    | `LIFNode`     | 6.8 MB|
| `sew_resnet34_nsavp.pth`      | SEW-ResNet34 + MixVPR (4096-D)| NSAVP    | `LIFNode`     | 12 MB |
| `sew_resnet34_nyc.pth`        | SEW-ResNet34 + MixVPR (4096-D)| NYC      | `IFNode`      | 12 MB |
| `netvlad_weights.pth`         | NetVLAD layer (ANN baseline)  | —        | —             | 260 KB|
| `wpca_weights.pth`            | NetVLAD WPCA projection (ANN) | —        | —             | 513 MB|

Notes:
- There is no SEW-ResNet18 checkpoint for NYC (only ResNet34 was trained on NYC).
- `netvlad_weights.pth` + `wpca_weights.pth` load into
  `spikevpr.baselines.RetrievalModel` (`.netvlad` and `.wpca`) for the energy /
  recall comparison against SpikeVPR.
