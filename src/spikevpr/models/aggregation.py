"""
Feature aggregation heads.

SpikeVPR uses a spiking MixVPR head on top of the SEW-ResNet backbone to turn
the (512, H, W) feature map into a single L2-normalised descriptor. GEM and a
simple MLP head are kept as lightweight alternatives, but MixVPR is the default
and the one all shipped checkpoints use.

The MixVPR / FeatureMixerLayer attribute names match the shipped checkpoints so
``load_state_dict`` matches exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron


def _make_neuron(neuron_type, threshold, tau):
    if neuron_type == "IFNode":
        return neuron.IFNode(v_threshold=threshold)
    if neuron_type == "LIFNode":
        return neuron.LIFNode(v_threshold=threshold, tau=tau)
    raise ValueError("Unsupported neuron type. Use 'IFNode' or 'LIFNode'.")


class FeatureMixerLayer(nn.Module):
    """Spiking feature-mixer block, adapted from MixVPR (Ali-bey et al., 2023)."""

    def __init__(self, in_dim, mlp_ratio=1, neuron_type="IFNode", threshold=1.0, tau=2.0):
        super().__init__()
        self.sn = _make_neuron(neuron_type, threshold, tau)
        self.ln = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, int(in_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(in_dim * mlp_ratio), in_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        y = self.ln(x)
        y = self.fc(y)
        y = self.sn(y)
        y = self.fc2(y)
        return x + y


class MixVPR(nn.Module):
    """
    Spiking MixVPR aggregation.

    Output descriptor dimension is ``out_channels * out_rows``. The shipped
    SpikeVPR models use out_channels=512, out_rows=8 (4096-D).
    """

    def __init__(self, in_channels=1024, in_h=20, in_w=20, out_channels=512,
                 mix_depth=1, mlp_ratio=1, out_rows=4,
                 neuron_type="IFNode", threshold=1.0, tau=2.0):
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_rows = out_rows
        self.mix_depth = mix_depth
        self.mlp_ratio = mlp_ratio

        hw = in_h * in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(hw, mlp_ratio, neuron_type, threshold, tau)
            for _ in range(mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        return F.normalize(x.flatten(1), p=2, dim=-1)

    @property
    def descriptor_dim(self):
        return self.out_channels * self.out_rows


class GEMPool(nn.Module):
    """Generalised-mean pooling (Radenovic et al., 2018)."""

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x, norm=True):
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                         (x.size(-2), x.size(-1))).pow(1.0 / self.p)
        x = x.flatten(1)
        if norm:
            x = F.normalize(x, p=2, dim=1)
        return x


class MLPAggregation(nn.Module):
    """Minimal spiking MLP head (global pool + two spiking FC layers)."""

    def __init__(self, input_dim, output_dim, hidden_dim=512,
                 neuron_type="IFNode", threshold=1.0, tau=2.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.sn1 = _make_neuron(neuron_type, threshold, tau)
        self.sn2 = _make_neuron(neuron_type, float("inf"), tau)

    def forward(self, x):
        x = self.gap(x).flatten(1)
        x = self.sn1(self.fc1(x))
        x = self.sn2(self.fc2(x))
        return F.normalize(self.sn2.v, p=2, dim=1)
