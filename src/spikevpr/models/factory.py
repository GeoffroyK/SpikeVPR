"""
Model factory for SpikeVPR.

A SpikeVPR model is a SEW-ResNet backbone (separable convolutions, 2-channel
event-frame input, T=1) plus an aggregation head that produces a single
L2-normalised descriptor. ``build_spikevpr`` is the one entry point used by
training, evaluation and the tutorial.

All three datasets feed the backbone a (2, 260, 346) frame, which produces a
(512, 9, 11) feature map; MixVPR is configured for that resolution.
"""
import torch

from .sew_resnet import sew_resnet18, sew_resnet34
from .aggregation import MixVPR, GEMPool, MLPAggregation

# Feature-map resolution after the backbone for a (2, 260, 346) input frame.
# MixVPR takes the spatial dims as (in_h, in_w) = (11, 9).
_FEATURE_H, _FEATURE_W = 11, 9

ENCODERS = {
    "sew_resnet18": sew_resnet18,
    "sew_resnet34": sew_resnet34,
}


def build_aggregator(name="mixvpr", out_channels=512, out_rows=8,
                     neuron_type="LIFNode", **kwargs):
    """
    Build an aggregation head.

    Args:
        name: 'mixvpr' (default), 'gem' or 'mlp'.
        out_channels, out_rows: MixVPR projection sizes; descriptor dimension is
            their product (512 * 8 = 4096 for the shipped models).
        neuron_type: spiking neuron used inside MixVPR/MLP. The shipped Brisbane
            and NSAVP checkpoints were trained with 'LIFNode'; NYC with 'IFNode'.
    """
    name = name.lower()
    if name == "mixvpr":
        return MixVPR(in_channels=512, in_h=_FEATURE_H, in_w=_FEATURE_W,
                      out_channels=out_channels, mix_depth=3, mlp_ratio=1,
                      out_rows=out_rows, neuron_type=neuron_type, **kwargs)
    if name == "gem":
        return GEMPool()
    if name == "mlp":
        return MLPAggregation(input_dim=512, output_dim=out_channels,
                              neuron_type=neuron_type, **kwargs)
    raise ValueError(f"Unknown aggregator '{name}'. Choose from mixvpr, gem, mlp.")


def build_spikevpr(encoder="sew_resnet34", aggregator="mixvpr",
                   out_channels=512, out_rows=8, neuron_type="LIFNode",
                   checkpoint=None, device=None, eval_mode=False):
    """
    Build a complete SpikeVPR model (backbone + aggregator).

    Args:
        encoder: 'sew_resnet18' or 'sew_resnet34'.
        aggregator: aggregation head name (see ``build_aggregator``).
        out_channels, out_rows: descriptor configuration.
        neuron_type: MixVPR neuron type (must match the checkpoint to reproduce it).
        checkpoint: optional path to a state_dict to load.
        device: optional torch device; defaults to cuda if available.
        eval_mode: if True, call ``model.eval()`` before returning.

    Returns:
        The model placed on ``device``.
    """
    if encoder not in ENCODERS:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose from {list(ENCODERS)}.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head = build_aggregator(aggregator, out_channels=out_channels,
                            out_rows=out_rows, neuron_type=neuron_type)
    model = ENCODERS[encoder](
        zero_init_residual=True, T=1, connect_f="ADD", in_channels=2,
        aggregator=head, only_encoder=True,
    )

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)

    model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
