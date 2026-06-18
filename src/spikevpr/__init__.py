"""
SpikeVPR — spiking neural networks for event-based Visual Place Recognition.

A SEW-ResNet (separable convolutions) backbone with a spiking MixVPR head,
trained with InfoNCE on three event datasets (Brisbane, NSAVP, NYC).

Typical use:

    from spikevpr.models import build_spikevpr
    model = build_spikevpr("sew_resnet34", checkpoint="weights/sew_resnet34_nsavp.pth",
                           neuron_type="LIFNode", eval_mode=True)
    descriptor = model(event_frame)        # (B, 2, 260, 346) -> (B, 4096), L2-normalised
"""
from .models import build_spikevpr, build_aggregator, count_parameters

__version__ = "1.0.0"

__all__ = ["build_spikevpr", "build_aggregator", "count_parameters", "__version__"]
