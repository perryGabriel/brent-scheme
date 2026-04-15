from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch


def block_diag(matrix, n):
    m, k = matrix.shape
    result = np.zeros((m * n, k * n))
    for i in range(n):
        result[i * m:(i + 1) * m, i * k:(i + 1) * k] = matrix
    return result


def permutation_matrix(
    indices: Sequence[int],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return permutation matrix ``P`` such that ``P @ x == x[indices]``."""
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    n = idx.numel()
    return torch.eye(n, dtype=dtype, device=device).index_select(dim=1, index=idx)


def random_unitary(
    n: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a random orthogonal/unitary-like matrix using QR decomposition."""
    A = torch.randn((n, n), dtype=dtype, device=device, generator=generator)
    Q, R = torch.linalg.qr(A, mode="reduced")
    diag = torch.diagonal(R)
    phase = torch.sign(diag)
    phase = torch.where(phase == 0, torch.ones_like(phase), phase)
    Q = Q * phase
    return Q


def rand_square(
    n: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return an ``(n, n)`` standard normal matrix."""
    return torch.randn((n, n), dtype=dtype, device=device, generator=generator)


def random_right_invertible(
    l: int,
    *,
    r: Optional[int] = None,
    s: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Construct random right-invertible matrix of shape ``(l, r)`` with ``r >= l``."""
    if r is None or r < l:
        r = l

    if s is None:
        s_vec = torch.ones((l,), dtype=dtype, device=device)
    else:
        s_vec = torch.as_tensor(s, dtype=dtype, device=device)
        if s_vec.shape != (l,):
            raise ValueError(f"s must have shape ({l},), got {tuple(s_vec.shape)}")

    U_l = random_unitary(l, dtype=dtype, device=device, generator=generator)
    U_r = random_unitary(r, dtype=dtype, device=device, generator=generator)

    S = torch.zeros((l, r), dtype=dtype, device=device)
    S[:, :l] = torch.diag(s_vec)

    return U_l @ S @ U_r


def mode_n_product(tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
    """Apply an n-mode product of ``tensor`` by ``matrix`` along ``mode``.

    If tensor has shape ``(I0, ..., In, ..., Ik)`` and matrix has shape ``(J, In)``,
    result has shape ``(I0, ..., J, ..., Ik)``.
    """
    if mode < 0 or mode >= tensor.ndim:
        raise ValueError(f"mode must be in [0, {tensor.ndim - 1}], got {mode}")
    if matrix.ndim != 2:
        raise ValueError("matrix must be rank-2")
    if matrix.shape[1] != tensor.shape[mode]:
        raise ValueError(
            f"matrix second dim ({matrix.shape[1]}) must equal tensor mode dim ({tensor.shape[mode]})"
        )

    moved = torch.movedim(tensor, mode, 0)
    out = torch.tensordot(matrix, moved, dims=([1], [0]))
    return torch.movedim(out, 0, mode)


def hosvd(
    tensor: torch.Tensor,
    *,
    ranks: Sequence[int] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute (optionally truncated) HOSVD.

    Returns ``(core, factors)`` where:
    - ``factors[i]`` has shape ``(tensor.shape[i], rank_i)``
    - ``core`` has shape ``tuple(rank_i)``

    Set ``ranks=None`` for full HOSVD ranks.
    """
    if tensor.ndim < 2:
        raise ValueError("hosvd expects tensor with ndim >= 2")

    shape = tensor.shape
    if ranks is None:
        ranks = shape
    if len(ranks) != tensor.ndim:
        raise ValueError(f"ranks length ({len(ranks)}) must match ndim ({tensor.ndim})")

    factors: list[torch.Tensor] = []
    for mode, (dim, rank) in enumerate(zip(shape, ranks)):
        if rank <= 0 or rank > dim:
            raise ValueError(f"rank for mode {mode} must be in [1, {dim}], got {rank}")

        unfolding = torch.movedim(tensor, mode, 0).reshape(dim, -1)
        U, _, _ = torch.linalg.svd(unfolding, full_matrices=False)
        factors.append(U[:, :rank])

    core = tensor
    for mode, U in enumerate(factors):
        core = mode_n_product(core, U.mT.conj(), mode)

    return core, factors
