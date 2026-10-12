"""Bias-free ReLU networks and reproducible scaled initialization."""
from __future__ import annotations
import copy
from collections.abc import Callable
import torch
from torch import nn
from torch.nn import functional as F

class HomogeneousReLU(nn.Module):
    """A scalar-output ReLU network; depth is len(hidden_widths) + 1.

    Parameters retain hidden1/hidden2/output names for the historical depth-3
    preset. An empty hidden_widths list gives a linear scalar predictor.
    """
    def __init__(self, *, input_dim: int = 784, hidden_widths=(1024, 1024)):
        super().__init__()
        if input_dim <= 0 or any(int(w) != w or w <= 0 for w in hidden_widths):
            raise ValueError("input dimension and hidden widths must be positive integers")
        self.input_dim = int(input_dim)
        self.hidden_widths = tuple(int(w) for w in hidden_widths)
        self.layer_names = tuple(f"hidden{i+1}" for i in range(len(self.hidden_widths))) + ("output",)
        sizes = (self.input_dim, *self.hidden_widths, 1)
        for name, left, right in zip(self.layer_names, sizes[:-1], sizes[1:]):
            self.add_module(name, nn.Linear(left, right, bias=False))
        self.affine_layer_count = len(self.layer_names)
        self.homogeneity_degree = float(self.affine_layer_count)

    def forward(self, x):
        value = x.flatten(1)
        for name in self.layer_names[:-1]:
            value = F.relu(getattr(self, name)(value))
        return self.output(value).reshape(-1)

class HomogeneousThreeLayerReLU(HomogeneousReLU):
    """Historical architecture retained for explicit regression tests."""
    def __init__(self, *, input_dim=784, hidden_width=1024):
        super().__init__(input_dim=input_dim, hidden_widths=(hidden_width, hidden_width))

def make_scaled_initialization(
    model_factory: Callable[[], nn.Module],
    example_input: torch.Tensor,
    *,
    layer_scale: float,
    seed: int,
    initialization_dtype: torch.dtype | None = None,
    kind: str = "torch_default", std: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Scale every parameter and numerically verify the output power law."""
    if layer_scale <= 0:
        raise ValueError("layer_scale must be positive")
    torch.manual_seed(seed)
    initialization_device = (
        torch.device("cpu")
        if initialization_dtype is not None else example_input.device
    )
    initialization_dtype = (
        example_input.dtype
        if initialization_dtype is None else initialization_dtype
    )
    initialization_input = example_input.to(
        device=initialization_device, dtype=initialization_dtype,
    )
    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(initialization_dtype)
    try:
        reference = model_factory().to(
            device=initialization_device, dtype=initialization_dtype,
        )
    finally:
        torch.set_default_dtype(previous_default_dtype)
    if kind == "normal":
        with torch.no_grad():
            for parameter in reference.parameters():
                nn.init.normal_(parameter, mean=0.0, std=std)
    elif kind != "torch_default":
        raise ValueError(f"unknown initialization kind: {kind}")
    scaled = copy.deepcopy(reference)
    degree_value = scaled.homogeneity_degree
    degree = float(degree_value)
    with torch.no_grad():
        for parameter in scaled.parameters():
            parameter.mul_(layer_scale)
        reference_logits = reference(initialization_input)
        scaled_logits = scaled(initialization_input)
        expected = reference_logits * layer_scale ** degree
        absolute_error = float((scaled_logits - expected).abs().max().item())
        denominator = float(expected.abs().max().clamp_min(1e-300).item())
        relative_error = absolute_error / denominator
    state = {
        name: tensor.detach().clone()
        for name, tensor in scaled.state_dict().items()
    }
    metadata = {
        "seed": int(seed),
        "kind": kind, "normal_std_before_scale": std,
        "per_parameter_multiplier": float(layer_scale),
        "homogeneity_degree": degree,
        "predicted_output_multiplier": float(layer_scale ** degree),
        "max_output_scaling_absolute_error": absolute_error,
        "max_output_scaling_relative_error": relative_error,
        "parameter_count": sum(p.numel() for p in scaled.parameters()),
        "parameter_standard_deviations": {
            name: float(tensor.std().item())
            for name, tensor in state.items()
            if tensor.numel() > 1
        },
    }
    return state, metadata
