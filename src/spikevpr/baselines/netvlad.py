"""
NetVLAD event-VPR baseline (ANN), for comparison against SpikeVPR.

Pipeline: an EST event-representation layer (learned trilinear voxel grid,
Gehrig et al. 2019) feeds a ResNet-34 backbone, a NetVLAD aggregation
(Arandjelovic et al. 2016) and a 1x1-conv WPCA projection to a 4096-D
L2-normalised descriptor — the same descriptor size as SpikeVPR.

The module layout matches the shipped ``netvlad_weights.pth`` / ``wpca_weights.pth``
so they load into ``RetrievalModel.netvlad`` / ``.wpca`` respectively.

Input convention: ``forward`` takes a raw event tensor of shape (N, 5) with
columns [x, y, t, p, batch_index] (events from all batch items concatenated).
"""
from os.path import dirname, isfile, join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34


class ValueLayer(nn.Module):
    """Per-event MLP that realises the trilinear voting kernel of the EST grid."""

    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[0] == 1 and mlp_layers[-1] == 1, \
            "ValueLayer MLP must start and end with 1 channel"
        super().__init__()
        self.mlp = nn.ModuleList()
        self.activation = activation
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            self.load_state_dict(torch.load(path))
        else:
            self._init_kernel(num_channels)

    def forward(self, x):
        x = x[None, ..., None]
        for layer in self.mlp[:-1]:
            x = self.activation(layer(x))
        return self.mlp[-1](x).squeeze()

    def _trilinear_kernel(self, ts, num_channels):
        gt = torch.zeros_like(ts)
        gt[ts > 0] = (1 - (num_channels - 1) * ts)[ts > 0]
        gt[ts < 0] = ((num_channels - 1) * ts + 1)[ts < 0]
        gt[ts < -1.0 / (num_channels - 1)] = 0
        gt[ts > 1.0 / (num_channels - 1)] = 0
        return gt

    def _init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)
        torch.manual_seed(1)
        for _ in range(1000):
            optim.zero_grad()
            ts.uniform_(-1, 1)
            loss = (self.forward(ts) - self._trilinear_kernel(ts, num_channels)).pow(2).sum()
            loss.backward()
            optim.step()


class QuantizationLayer(nn.Module):
    """Build a (B, 2C, H, W) EST voxel grid from a raw (N, 5) event tensor."""

    def __init__(self, dim, mlp_layers=(1, 100, 100, 1),
                 activation=nn.LeakyReLU(negative_slope=0.1)):
        super().__init__()
        self.value_layer = ValueLayer(list(mlp_layers), activation=activation, num_channels=dim[0])
        self.dim = dim

    def forward(self, events):
        B = int((1 + events[-1, -1]).item())
        C, H, W = self.dim
        vox = events[0].new_full([2 * int(np.prod(self.dim)) * B], fill_value=0)
        x, y, t, p, b = events.t()
        for bi in range(B):
            t[events[:, -1] == bi] /= t[events[:, -1] == bi].max()
        p = (p + 1) / 2
        idx_base = x + W * y + W * H * C * p + W * H * C * 2 * b
        for i_bin in range(C):
            values = t * self.value_layer.forward(t - i_bin / (C - 1))
            vox.put_((idx_base + W * H * i_bin).long(), values, accumulate=True)
        vox = vox.view(-1, 2, C, H, W)
        return torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)


class NetVLAD(nn.Module):
    """NetVLAD aggregation layer (MatConvNet-style intra/post normalisation)."""

    def __init__(self, num_clusters, dim, skip_postnorm=False):
        super().__init__()
        self.K = num_clusters
        self.D = dim
        self.skip_postnorm = skip_postnorm
        self.assignment = nn.Conv2d(dim, num_clusters, kernel_size=1, bias=False)
        self.clusters = nn.Parameter(torch.zeros(1, 1, 1, dim, num_clusters))

    @staticmethod
    def _normalize(inputs, eps=1e-12):
        return inputs / torch.sqrt(torch.sum(inputs ** 2, dim=-1, keepdim=True) + eps)

    def forward(self, x):
        N, D, H, W = x.shape
        a = F.softmax(self.assignment(x), dim=1).permute(0, 2, 3, 1).unsqueeze(-2)
        v = x.permute(0, 2, 3, 1).unsqueeze(-1) + self.clusters
        v = (a * v).sum(dim=(1, 2)).permute(0, 2, 1)        # (N, K, D)
        if not self.skip_postnorm:
            v = self._normalize(v, 1e-12).permute(0, 2, 1).reshape(N, -1)
            v = self._normalize(v, 1e-12)
        return v


class RetrievalModel(nn.Module):
    """EST + ResNet-34 + NetVLAD + WPCA -> 4096-D descriptor."""

    def __init__(self, voxel_dimension=(9, 260, 346), crop_dimension=(224, 224),
                 dim=512, num_clusters=64, mlp_layers=(1, 30, 30, 1),
                 activation=nn.LeakyReLU(negative_slope=0.1), pretrained=False):
        super().__init__()
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation)
        self.crop_dimension = crop_dimension

        backbone = resnet34(pretrained=pretrained)
        backbone.conv1 = nn.Conv2d(2 * voxel_dimension[0], 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
        self.classifier = nn.Sequential(*list(backbone.children())[:-2])  # conv feature maps

        self.netvlad = NetVLAD(num_clusters=num_clusters, dim=dim)
        self.wpca = nn.Conv2d(num_clusters * dim, 4096, kernel_size=1)

    def _crop_resize(self, x, resolution):
        _, _, H, W = x.shape
        if H > W:
            h = H // 2
            x = x[:, :, h - W // 2:h + W // 2, :]
        else:
            h = W // 2
            x = x[:, :, :, h - H // 2:h + H // 2]
        return F.interpolate(x, size=resolution)

    def forward(self, events):
        vox = self.quantization_layer(events)
        feats = self.classifier(self._crop_resize(vox, self.crop_dimension))
        vlad = self.netvlad(feats)                       # (N, K*D)
        vlad = vlad.view(vlad.size(0), -1, 1, 1)
        vlad = self.wpca(vlad).view(vlad.size(0), -1)    # (N, 4096)
        return F.normalize(vlad, p=2, dim=1), vox

    @classmethod
    def from_weights(cls, netvlad_path, wpca_path, device=None, **kwargs):
        """Load a RetrievalModel with the shipped NetVLAD + WPCA weights."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(**kwargs)
        model.netvlad.load_state_dict(torch.load(netvlad_path, map_location=device))
        model.wpca.load_state_dict(torch.load(wpca_path, map_location=device))
        return model.to(device)
