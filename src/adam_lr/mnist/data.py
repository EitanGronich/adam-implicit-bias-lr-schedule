"""Deterministic balanced odd/even MNIST and small synthetic smoke data."""
from pathlib import Path
import torch

def balanced_mnist(root: str | Path, train: bool, per_class: int, *, download=False):
    """First n odd and n even images in official order; uint8/255 pixels.

    Return labels 0 (even) and 1 (odd). Loading uses torchvision only for the
    MNIST dataset reader, so synthetic runs do not require torchvision.
    """
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    from torchvision.datasets import MNIST
    dataset = MNIST(str(root), train=train, download=download)
    counts = [0, 0]
    indices, labels = [], []
    for index, digit in enumerate(dataset.targets):
        label = int(digit) % 2
        if counts[label] >= per_class:
            continue
        counts[label] += 1
        indices.append(index)
        labels.append(label)
        if min(counts) == per_class:
            break
    if counts != [per_class, per_class]:
        raise RuntimeError(f"insufficient balanced MNIST data: {counts}")
    # Pixel conversion is independent of torch's global default dtype, so
    # float32 -> float64 continuation preserves the dataset fingerprint.
    images = dataset.data[indices].to(torch.float64).div(255).unsqueeze(1)
    return images, torch.tensor(labels, dtype=torch.float64)


def synthetic_data(per_class: int, input_dim: int, seed: int):
    """Paired linearly separable Gaussian examples, for execution tests only."""
    if per_class <= 0 or input_dim <= 0:
        raise ValueError("synthetic counts and dimension must be positive")
    generator = torch.Generator().manual_seed(seed)
    positive = torch.randn(per_class, input_dim, generator=generator, dtype=torch.float64)
    positive[:, 0] = positive[:, 0].abs() + 0.25
    return torch.cat((-positive, positive)), torch.cat((torch.zeros(per_class), torch.ones(per_class))).double()
