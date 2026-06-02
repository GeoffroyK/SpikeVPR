# SpikeVPR

```text
  ______             __  __                 __     __  _______   _______  
 /      \           |  \|  \               |  \   |  \|       \ |       \ 
|  $$$$$$\  ______   \$$| $$   __   ______ | $$   | $$| $$$$$$$\| $$$$$$$\
| $$___\$$ /      \ |  \| $$  /  \ /      \| $$   | $$| $$__/ $$| $$__| $$
 \$$    \ |  $$$$$$\| $$| $$_/  $$|  $$$$$$\\$$\ /  $$| $$    $$| $$    $$
 _\$$$$$$\| $$  | $$| $$| $$   $$ | $$    $$ \$$\  $$ | $$$$$$$ | $$$$$$$\
|  \__| $$| $$__/ $$| $$| $$$$$$\ | $$$$$$$$  \$$ $$  | $$      | $$  | $$
 \$$    $$| $$    $$| $$| $$  \$$\ \$$     \   \$$$   | $$      | $$  | $$
  \$$$$$$ | $$$$$$$  \$$ \$$   \$$  \$$$$$$$    \$     \$$       \$$   \$$
          | $$                                                            
          | $$                                                            
           \$$                                                                                                      
```

**Event-Driven Neuromorphic Vision Enables Energy-Efficient Visual Place Recognition**

Geoffroy Keime, Nicolas Cuperlier, and Benoit R. Cottereau

IPAL (CNRS IRL 2955) · CerCo (CNRS UMR 5549) · ETIS (CNRS UMR 8051)

---

SpikeVPR is a bio-inspired visual place recognition system that combines event-based cameras with spiking neural networks to generate compact, invariant place descriptors. It achieves performance comparable to state-of-the-art deep networks while using **50× fewer parameters** and consuming **30–250× less energy**.

> **Paper:** Under review.
>
> Code will be released upon acceptance.

## Overview

Visual place recognition (VPR) aims to identify previously visited locations from visual input alone. SpikeVPR addresses this task using a fully neuromorphic pipeline:

- **Event camera input** — asynchronous, sparse binary signals encoding illumination changes, robust to lighting and motion blur.
- **Spiking neural network** — a SEW ResNet encoder with depthwise separable convolutions, followed by a spiking MixVPR aggregator, producing 512-dimensional binary descriptors.
- **Contrastive learning** — trained end-to-end with surrogate gradient learning using the NT-Xent loss.
- **EventDilation** — a novel data augmentation strategy that varies the temporal integration window to improve robustness to speed and temporal variations.

## Key Results

| Method | Recall@1 (%) | Parameters (M) | Energy per inference (mJ) |
|---|---|---|---|
| Ensemble | 58.3 | 149.0 | 4,600 |
| EventVPR | 62.5 | 155.6 | 633.9 |
| **SpikeVPR** | **60.8** | **2.9** | **18.0** |

Results on the Brisbane-Event-VPR dataset. SpikeVPR processes a single input in ~9.5 ms, enabling real-time deployment at over 100 fps.

## Datasets

- [Brisbane-Event-VPR](https://open.qcr.ai/dataset/brisbane_event_vpr_dataset/) — 6 traverses of an 8 km peri-urban route under varying conditions.
- [NSAVP](https://umautobots.github.io/nsavp) — Urban traffic dataset with forward and reverse traversals.

## Requirements

- Python 3.8+
- PyTorch
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly)

A full environment specification will be provided with the code release.

## Citation

```
@article{keime2025spikevpr,
  title={Event-Driven Neuromorphic Vision Enables Energy-Efficient Visual Place Recognition},
  author={Keime, Geoffroy and Cuperlier, Nicolas and Cottereau, Benoit R.},
  year={2025},
  note={Preprint}
}
```

## Acknowledgments

This work was supported by the French Defense Innovation Agency (AID) under grant 2023 65 0082.

## License

Coming soon.
