"""Standalone approximate nuclear norm for optional matrix diagnostics."""
import math
import torch

JIACHENG_COEFFICIENTS: tuple[tuple[float, float, float], ...] = (
    (3955 / 1024, -8306 / 1024, 5008 / 1024),
    (3735 / 1024, -6681 / 1024, 3463 / 1024),
    (3799 / 1024, -6499 / 1024, 3211 / 1024),
    (4019 / 1024, -6385 / 1024, 2906 / 1024),
    (2677 / 1024, -3029 / 1024, 1162 / 1024),
    (2172 / 1024, -1833 / 1024, 682 / 1024),
)


@torch.no_grad()
def polar_factor_jiacheng(
    gradient: torch.Tensor,
    *,
    steps: int = 12,
    epsilon: float = 1e-20,
) -> torch.Tensor:
    """Approximate the rectangular polar factor with Jiacheng's six-step cycle.

    This is the ``jiacheng`` branch of ``polar_factor_iterative`` in
    ``experiments/utils.py`` in the user's original WEXAC repository.  As in
    that source, ``steps`` is rounded upward to a multiple of six and the
    input is Frobenius-normalized before iteration.
    """

    if gradient.ndim != 2:
        raise ValueError("Jiacheng polar iteration requires a matrix")
    if steps <= 0:
        raise ValueError("steps must be a positive integer")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    # The polar factor of a row or column vector is available exactly.  Apart
    # from avoiding needless polynomial approximation error, this makes a
    # single-output layer precisely L2-normalized momentum descent.
    if min(gradient.shape) == 1:
        return gradient / (torch.linalg.norm(gradient) + epsilon)
    actual_steps = 6 * int(math.ceil(steps / 6))
    transpose = gradient.shape[1] > gradient.shape[0]
    iterate = gradient.T if transpose else gradient
    iterate = iterate / (torch.linalg.norm(iterate) + epsilon)
    identity = torch.eye(
        iterate.shape[1], dtype=iterate.dtype, device=iterate.device,
    )
    for index in range(actual_steps):
        gram = iterate.T @ iterate
        a, b, c = JIACHENG_COEFFICIENTS[index % 6]
        # Preserve the source expression ``c*A @ A`` exactly, including its
        # floating-point operation order: Python evaluates it as ``(c*A) @ A``.
        iterate = iterate @ (a * identity + b * gram + (c * gram) @ gram)
    return iterate.T if transpose else iterate


@torch.no_grad()
def nuclear_norm_iterative(
    matrix: torch.Tensor,
    *,
    polar_steps: int = 12,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Approximate the nuclear norm using a scale-invariant polar trace.

    This ports ``nuclear_norm_iterative`` from ``experiments/utils.py`` in
    the user's original WEXAC repository: if ``P`` is the Jiacheng polar
    approximation, the estimate is ``trace(P.T @ matrix)``. Normalize the
    input before that approximation and restore its scale afterward; an
    absolute epsilon on the raw gradient biases small-gradient diagnostics.
    Row and column vectors have an exact nuclear norm and bypass iteration.
    """

    if matrix.ndim != 2:
        raise ValueError("iterative nuclear norm requires a matrix")
    if not matrix.is_floating_point() or not bool(torch.isfinite(matrix).all()):
        raise ValueError("iterative nuclear norm requires a finite float matrix")
    if bool(torch.count_nonzero(matrix) == 0):
        return torch.zeros((), dtype=matrix.dtype, device=matrix.device)
    scale = matrix.abs().max()
    normalized = matrix / scale
    if min(matrix.shape) == 1:
        return scale * torch.linalg.vector_norm(normalized)
    polar = polar_factor_jiacheng(
        normalized,
        steps=polar_steps,
        epsilon=epsilon,
    )
    # Preserve the original implementation rather than replacing the trace
    # by an elementwise Frobenius pairing.
    return scale * (polar.T @ normalized).trace()
