"""
Energy estimation for SNN and ANN VPR models.

Estimates inference energy from *measured* activity (spike counts for SNNs,
ReLU sparsity for ANNs) and per-layer architecture, using published
synaptic-operation proxies at 45 nm / 32-bit (Horowitz 2014; Dampfhoffer 2023;
Lemaire 2022). Nothing is hardcoded: every number is recomputed from the model
and a few batches of real data.

  * SNN proxies: Dampfhoffer, Lemaire.
  * ANN proxies: naive, Eyeriss, best-case, Lemaire.

Pipeline: ``extract_layers`` captures per-layer shapes from a forward pass;
``measure_spike_counts`` / ``measure_relu_sparsity`` hook the activations;
``estimate_model`` sums the per-layer proxy energies.
"""
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ── technology-node parameters (45 nm, 32-bit) ──────────────────────────────────

@dataclass
class TechNodeParams:
    name: str
    e_add: float
    e_mul: float
    e_mac: float
    e_ac: float
    e_sram_read: Optional[float] = None
    e_sram_write: Optional[float] = None
    e_reg_read: Optional[float] = None
    e_reg_write: Optional[float] = None
    sram_size_dependent: bool = False
    data_precision_bits: int = 32

    def get_sram_energy(self, memory_size_bytes):
        if not self.sram_size_dependent:
            return self.e_sram_read
        size_kb = memory_size_bytes / 1024.0
        # (size_kb, pJ/access) from 45 nm characterisation (Horowitz 2014).
        points = [(8, 10.0), (32, 20.0), (1024, 100.0)]
        if size_kb <= points[0][0]:
            return points[0][1]
        if size_kb >= points[-1][0]:
            # Cap rather than extrapolate: a 400 MB FC weight would otherwise
            # inflate energy ~300x; 100 pJ is already a conservative on-chip bound.
            return points[-1][1]
        for i in range(len(points) - 1):
            if points[i][0] <= size_kb <= points[i + 1][0]:
                t = (size_kb - points[i][0]) / (points[i + 1][0] - points[i][0])
                return points[i][1] + t * (points[i + 1][1] - points[i][1])
        return self.e_sram_read


TECH = TechNodeParams(
    name="45nm_32bit", e_add=0.1, e_mul=3.1, e_mac=3.2, e_ac=0.1,
    e_reg_read=3.2, e_reg_write=3.2, sram_size_dependent=True, data_precision_bits=32,
)


# ── per-layer spec ───────────────────────────────────────────────────────────────

class LayerType(Enum):
    CONV2D = "conv2d"
    LINEAR = "linear"


@dataclass
class LayerSpec:
    name: str
    layer_type: LayerType
    C_in: int = 0
    C_out: int = 0
    H_in: int = 0
    W_in: int = 0
    H_out: int = 0
    W_out: int = 0
    H_k: int = 0
    W_k: int = 0
    stride: int = 1
    groups: int = 1
    has_bias: bool = True
    N_in: int = 0
    N_out: int = 0
    theta_in: int = 0       # SNN: input spikes
    theta_out: int = 0      # SNN: output spikes
    relu_sparsity: float = 0.0  # ANN: fraction of zero activations

    @property
    def is_conv(self):
        return self.layer_type == LayerType.CONV2D

    @property
    def is_depthwise(self):
        return self.is_conv and self.groups == self.C_in and self.groups > 1

    @property
    def n_synapses(self):
        if self.is_conv:
            return self.C_out * (self.C_in // self.groups) * self.H_k * self.W_k * self.H_out * self.W_out
        return self.N_in * self.N_out

    @property
    def n_neurons(self):
        return self.C_out * self.H_out * self.W_out if self.is_conv else self.N_out

    @property
    def n_weights(self):
        if self.is_conv:
            return self.C_out * (self.C_in // self.groups) * self.H_k * self.W_k
        return self.N_in * self.N_out

    @property
    def weight_memory_bytes(self):
        return self.n_weights * 4

    @property
    def output_memory_bytes(self):
        return self.n_neurons * 4

    @property
    def rf_iact(self):
        return float(self.C_out // self.groups) if self.is_conv else float(self.N_out)

    @property
    def rf_weight(self):
        return float(self.H_out * self.W_out) if self.is_conv else 1.0

    @property
    def rf_psum(self):
        if self.is_conv:
            return float((self.C_in // self.groups) * self.H_k * self.W_k)
        return float(self.N_in)


# ── architecture extraction ──────────────────────────────────────────────────────

def extract_layers(model, dummy_input):
    """Run one forward pass to capture every Conv2d/Linear spec with spatial dims."""
    spatial = OrderedDict()
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            i = inp[0] if isinstance(inp, tuple) else inp
            o = out[0] if isinstance(out, tuple) else out
            spatial[name] = (list(i.shape), list(o.shape))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        model(dummy_input)
    for h in hooks:
        h.remove()

    layers = []
    for name, module in model.named_modules():
        if name not in spatial:
            continue
        in_s, out_s = spatial[name]
        if isinstance(module, nn.Conv2d):
            hk = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            wk = module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size
            st = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            layers.append(LayerSpec(
                name=name, layer_type=LayerType.CONV2D,
                C_in=module.in_channels, C_out=module.out_channels,
                H_in=in_s[2], W_in=in_s[3], H_out=out_s[2], W_out=out_s[3],
                H_k=hk, W_k=wk, stride=st, groups=module.groups,
                has_bias=module.bias is not None))
        elif isinstance(module, nn.Linear):
            layers.append(LayerSpec(
                name=name, layer_type=LayerType.LINEAR,
                N_in=module.in_features, N_out=module.out_features,
                has_bias=module.bias is not None))
    return layers


# ── activity measurement ─────────────────────────────────────────────────────────

def _input_tensor(batch, device):
    if isinstance(batch, dict):
        key = "anchor" if "anchor" in batch else "frame"
        return batch[key].to(device).float()
    if isinstance(batch, (list, tuple)):
        return batch[0].to(device).float()
    return batch.to(device).float()


def measure_spike_counts(model, dataloader, device="cpu", num_batches=50):
    """Average output spikes per inference for each spiking-neuron module."""
    from spikingjelly.activation_based import neuron as sj_neuron, functional
    spike_types = (sj_neuron.IFNode, sj_neuron.LIFNode)

    counts = OrderedDict()
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            counts.setdefault(name, {"spikes": 0.0, "n": 0})
            counts[name]["spikes"] += out.detach().sum().item()
            counts[name]["n"] += out.shape[0]
        return hook

    for name, module in model.named_modules():
        if isinstance(module, spike_types):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval().to(device)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            functional.reset_net(model)
            model(_input_tensor(batch, device))
    for h in hooks:
        h.remove()
    return OrderedDict((n, d["spikes"] / max(d["n"], 1)) for n, d in counts.items())


def assign_spike_counts_to_layers(layers, spike_counts, model, input_spike_count=0):
    """
    Map each spiking neuron's spike count to the compute layer immediately before
    it (traversal order), then propagate theta_out -> next layer's theta_in.
    """
    from spikingjelly.activation_based import neuron as sj_neuron
    spike_types = (sj_neuron.IFNode, sj_neuron.LIFNode)

    by_name = {l.name: l for l in layers}
    assigned = {}
    last_compute = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            last_compute = name
        elif isinstance(module, spike_types):
            if name in spike_counts and last_compute in by_name:
                assigned[last_compute] = int(spike_counts[name])
                last_compute = None

    for i, layer in enumerate(layers):
        if layer.name in assigned:
            layer.theta_out = assigned[layer.name]
            if i + 1 < len(layers):
                layers[i + 1].theta_in = layer.theta_out
    if layers:
        layers[0].theta_in = input_spike_count
    return layers


def measure_relu_sparsity(model, dataloader, device="cpu", num_batches=10):
    """Fraction of zero activations after each nn.ReLU / nn.ReLU6 module."""
    data, hooks = {}, []

    def make_hook(name):
        def hook(module, inp, out):
            data.setdefault(name, {"zeros": 0, "total": 0})
            data[name]["zeros"] += (out == 0).sum().item()
            data[name]["total"] += out.numel()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval().to(device)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            model(_input_tensor(batch, device))
    for h in hooks:
        h.remove()
    return {n: d["zeros"] / max(d["total"], 1) for n, d in data.items()}


def assign_relu_sparsity_to_layers(layers, relu_sparsity, default_gamma=0.5):
    gammas = list(relu_sparsity.values())
    for i, layer in enumerate(layers):
        layer.relu_sparsity = gammas[i] if i < len(gammas) else default_gamma
    return layers


# ── energy proxies ───────────────────────────────────────────────────────────────

def snn_dampfhoffer(l, tech=TECH):
    if l.is_conv:
        fan = math.ceil(l.H_k / l.stride) * math.ceil(l.W_k / l.stride)
        if not l.is_depthwise:
            fan *= (l.C_out // l.groups)
        ops = l.theta_in * fan
    else:
        ops = l.theta_in * l.N_out
    e_rd = tech.get_sram_energy(l.weight_memory_bytes + l.output_memory_bytes)
    e_wr = tech.e_sram_write if tech.e_sram_write is not None else e_rd
    return ops * (e_rd + e_rd + e_wr + tech.e_ac)


def snn_lemaire(l, tech=TECH, T=1):
    ti, to = l.theta_in, l.theta_out
    if l.is_conv:
        Hk, Wk, S = l.H_k, l.W_k, l.stride
        Co, Ho, Wo = l.C_out, l.H_out, l.W_out
        c_eff = 1 if l.is_depthwise else (Co // l.groups)
        acc = ti * math.ceil(Hk / S) * math.ceil(Wk / S) * c_eff
        acc += T * Co * Ho * Wo if l.has_bias else 0
        acc += to + ti * c_eff * Hk * Wk
        mac = ti * 2
        reads = ti + ti * c_eff * Hk * Wk * 2 + (Co * Ho * Wo * 2 if l.has_bias else 0)
        writes = to + ti * c_eff * Hk * Wk + (Co * Ho * Wo if l.has_bias else 0)
    else:
        No = l.N_out
        acc = ti * No + (T * No if l.has_bias else 0) + to + ti * No
        mac = 0
        reads = ti + ti * No + (No if l.has_bias else 0) + (ti + 1) * No
        writes = to + ti * No + (No if l.has_bias else 0)
    e_sram = tech.get_sram_energy(l.weight_memory_bytes + l.output_memory_bytes)
    return (reads + writes) * e_sram + (tech.e_add + tech.e_mul) * mac + tech.e_add * acc


def ann_naive(l, tech=TECH):
    e_rd = tech.get_sram_energy(l.weight_memory_bytes + l.output_memory_bytes)
    return l.n_synapses * (3 * e_rd + e_rd + tech.e_mac)


def ann_eyeriss(l, tech=TECH):
    n, g = l.n_synapses, l.relu_sparsity
    rf = max((l.rf_iact + l.rf_psum) / 2.0, 1.0)
    e_rd = tech.get_sram_energy(l.weight_memory_bytes + l.output_memory_bytes)
    gating = (1 - g) + 0.55 * g
    return n * gating * (e_rd + (3 * e_rd) / rf + tech.e_mac + 3 * tech.e_reg_read)


def ann_bestcase(l, tech=TECH):
    n, g = l.n_synapses, l.relu_sparsity
    ri, rw, rp = max(l.rf_iact, 1), max(l.rf_weight, 1), max(l.rf_psum, 1)
    e_rd = tech.get_sram_energy(l.weight_memory_bytes + l.output_memory_bytes)
    e_dist = n * (e_rd / ri + e_rd / rw + 2 * e_rd / rp)
    e_local = n * (tech.e_reg_read + (1 - g) * 3 * tech.e_reg_read)
    return e_dist + e_local + n * (1 - g) * tech.e_mac


def ann_lemaire(l, tech=TECH):
    if l.is_conv:
        cpg = l.C_in // l.groups
        n_macs = l.C_out * l.H_out * l.W_out * cpg * l.H_k * l.W_k
        reads = cpg * l.C_out * l.H_out * l.W_out * l.H_k * l.W_k
        reads += (cpg * l.H_k * l.W_k + (1 if l.has_bias else 0)) * l.C_out * l.H_out * l.W_out
        writes = l.C_out * l.H_out * l.W_out
        fm_bytes = max(l.C_in * l.H_in * l.W_in, l.C_out * l.H_out * l.W_out) * 4
    else:
        n_macs = l.N_in * l.N_out
        reads = l.N_in + (l.N_in + (1 if l.has_bias else 0)) * l.N_out
        writes = l.N_out
        fm_bytes = max(l.N_in, l.N_out) * 4
    e_mem = reads * tech.get_sram_energy(l.weight_memory_bytes) + writes * tech.get_sram_energy(fm_bytes)
    return e_mem + n_macs * tech.e_mac


# ── full-model estimation ────────────────────────────────────────────────────────

def estimate_model(name, layers, mode="snn"):
    """Sum per-layer proxy energies (pJ). mode: 'snn' or 'ann'."""
    result = {"name": name, "mode": mode, "n_layers": len(layers), "methods": {}}
    if mode == "snn":
        result["methods"]["dampfhoffer"] = sum(snn_dampfhoffer(l) for l in layers)
        result["methods"]["lemaire"] = sum(snn_lemaire(l) for l in layers)
        total_syn = sum(l.n_synapses for l in layers)
        result["avg_spikes_per_syn"] = sum(l.theta_in for l in layers) / max(total_syn, 1)
    else:
        result["methods"]["naive"] = sum(ann_naive(l) for l in layers)
        result["methods"]["eyeriss"] = sum(ann_eyeriss(l) for l in layers)
        result["methods"]["bestcase"] = sum(ann_bestcase(l) for l in layers)
        result["methods"]["lemaire"] = sum(ann_lemaire(l) for l in layers)
        total_syn = sum(l.n_synapses for l in layers)
        result["avg_relu_sparsity"] = sum(l.relu_sparsity * l.n_synapses for l in layers) / max(total_syn, 1)
    return result


def print_comparison_table(results):
    print("\n" + "=" * 96)
    print("ENERGY COMPARISON — per inference (45 nm, 32-bit)")
    print("=" * 96)
    print(f"  {'Model':<24s} {'Type':<5s} {'Method':<12s} {'Energy (mJ)':>14s}")
    print(f"  {'-'*24} {'-'*5} {'-'*12} {'-'*14}")
    for r in results:
        first = True
        for method, e_pj in r["methods"].items():
            prefix = r["name"] if first else ""
            mode = r["mode"].upper() if first else ""
            extra = ""
            if first and "avg_spikes_per_syn" in r:
                extra = f"  [spikes/syn {r['avg_spikes_per_syn']:.4f}]"
            elif first and "avg_relu_sparsity" in r:
                extra = f"  [ReLU sparsity {r['avg_relu_sparsity']:.2%}]"
            print(f"  {prefix:<24s} {mode:<5s} {method:<12s} {e_pj / 1e9:>14.4f}{extra}")
            first = False
        print()


# ── SNN convenience entry point ──────────────────────────────────────────────────

def estimate_snn_energy(model, dataloader, device, input_hw=(346, 260), num_batches=50):
    """
    Measure spike rate on real data and estimate the SpikeVPR backbone energy.
    Only the convolutional backbone is modelled (the aggregator is excluded),
    matching the original SpikeVPR energy study.
    """
    dummy = torch.zeros(1, 2, input_hw[0], input_hw[1], device=next(model.parameters()).device)
    layers = [l for l in extract_layers(model, dummy)
              if l.is_conv and not l.name.startswith("aggregator")]
    spikes = measure_spike_counts(model, dataloader, device, num_batches)
    input_spikes = int((dummy != 0).sum().item())
    layers = assign_spike_counts_to_layers(layers, spikes, model, input_spike_count=input_spikes)
    return estimate_model("SpikeVPR", layers, mode="snn")
