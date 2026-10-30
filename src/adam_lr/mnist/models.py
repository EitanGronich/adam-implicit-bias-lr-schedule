"""Scalar-output ReLU models and paired, scaled initialization."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class ReLUNetwork(nn.Module):
    """A configurable ReLU network with a scalar output."""
    def __init__(self, *, input_dim=784, hidden_widths=(512, 512), bias=False):
        super().__init__()
        if (type(input_dim) is not int or input_dim < 1
                or any(type(width) is not int or width < 1 for width in hidden_widths)
                or not isinstance(bias, bool)):
            raise ValueError("use positive integer widths and a boolean bias flag")
        self.input_dim, self.hidden_widths, self.bias = input_dim, tuple(hidden_widths), bias
        self.layer_names = tuple(f"hidden{i + 1}" for i in range(len(hidden_widths))) + ("output",)
        sizes = (input_dim, *hidden_widths, 1)
        for name, left, right in zip(self.layer_names, sizes[:-1], sizes[1:]):
            self.add_module(name, nn.Linear(left, right, bias=bias))
        self.affine_layer_count = len(self.layer_names)
        self.homogeneity_degree = float(self.affine_layer_count)
        self.exact_homogeneous = not bias

    def forward(self, x):
        value = x.flatten(1)
        for name in self.layer_names[:-1]:
            value = F.relu(getattr(self, name)(value))
        return self.output(value).reshape(-1)


def make_scaled_initialization(model_factory, example_input, *, layer_scale, seed,
                               initialization_dtype=None, kind="torch_default", std=1.):
    """Construct weights on CPU so schedules and epsilon values share a seed."""
    if not math.isfinite(layer_scale) or layer_scale <= 0 or not math.isfinite(std) or std <= 0:
        raise ValueError("initialization scale and standard deviation must be positive")
    dtype = initialization_dtype or example_input.dtype
    torch.manual_seed(seed)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = model_factory().to(device="cpu", dtype=dtype)
    finally:
        torch.set_default_dtype(previous_dtype)
    with torch.no_grad():
        if kind == "normal":
            for parameter in model.parameters():
                parameter.normal_(0., std)
        elif kind != "torch_default":
            raise ValueError("initialization must be torch_default or normal")
        for parameter in model.parameters():
            parameter.mul_(layer_scale)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    degree = model.homogeneity_degree
    metadata = dict(seed=int(seed), kind=kind, per_parameter_multiplier=float(layer_scale),
                    homogeneity_degree=degree if model.exact_homogeneous else None,
                    normalization_degree=degree, exact_homogeneous=model.exact_homogeneous,
                    parameter_count=sum(value.numel() for value in state.values()))
    return state, metadata
