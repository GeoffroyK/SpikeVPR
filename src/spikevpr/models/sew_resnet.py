"""
SEW-ResNet with depthwise-separable convolutions.

Spiking-Element-Wise (SEW) residual network (Fang et al., 2021) where every
3x3 convolution is replaced by a depthwise + pointwise pair. This is the only
backbone family used by SpikeVPR; the release ships ResNet-18 and ResNet-34
variants. Inputs are 2-channel (ON/OFF) event frames; the network runs a single
time step (T=1) and is stateless by default (membrane potentials are reset at
the start of every forward).

The module layout (attribute names, block order) is kept identical to the
checkpoints shipped with the release so ``load_state_dict`` matches exactly.
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, layer

__all__ = [
    "SEWResNet", "sew_resnet18", "sew_resnet34",
]


def conv3x3_separated(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 depthwise convolution followed by a 1x1 pointwise convolution."""
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride,
                  padding=dilation, groups=in_planes, bias=False, dilation=dilation),
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def zero_init_blocks(net):
    for m in net.modules():
        if isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2[1].weight, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super().__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = layer.SeqToANNContainer(
            conv3x3_separated(inplanes, planes, stride), norm_layer(planes))
        self.sn1 = neuron.IFNode(detach_reset=True)
        self.conv2 = layer.SeqToANNContainer(
            conv3x3_separated(planes, planes), norm_layer(planes))
        self.sn2 = neuron.IFNode(detach_reset=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.connect_f == "ADD":
            out = out + identity
        elif self.connect_f == "AND":
            out = out * identity
        elif self.connect_f == "IAND":
            out = identity * (1.0 - out)
        else:
            raise NotImplementedError(self.connect_f)
        return out


class SEWResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 in_channels=3, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, T=4,
                 connect_f=None, only_encoder=True, aggregator=None, stateful=False):
        super().__init__()
        self.T = T
        self.connect_f = connect_f
        self.only_encoder = only_encoder
        self.aggregator = aggregator
        self.stateful = stateful
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=2, padding=3,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, self.inplanes, kernel_size=1, stride=1, bias=False),
        )
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = neuron.IFNode(detach_reset=True)
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ),
                neuron.IFNode(detach_reset=True),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer, connect_f)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))
        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.stateful:
            functional.reset_net(self)
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = self.sn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.only_encoder:
            x = x.mean(dim=0)
            if self.aggregator is not None:
                return self.aggregator(x)
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        return x.mean(dim=0)


def sew_resnet18(**kwargs):
    return SEWResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34(**kwargs):
    return SEWResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
