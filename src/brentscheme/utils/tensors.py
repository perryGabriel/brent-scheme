from __future__ import annotations

from typing import Optional, Sequence

import torch


def permutation_matrix(
    indices: Sequence[int],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Return the permutation matrix P such that P @ x == x[indices].

    Parameters
    ----------
    indices:
        A permutation (or general reindexing) of 0..n-1.
    dtype, device:
        Tensor dtype and device.

    Returns
    -------
    (n, n) tensor with exactly one 1 in each row (and typically each column).
    """
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    n = idx.numel()
    # Row i picks column idx[i]
    return torch.eye(n, dtype=dtype, device=device).index_select(dim=1, index=idx)


def random_unitary(
    n: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample a random orthogonal matrix Q (a "random unitary" in the real case)
    using QR decomposition of a standard normal matrix.

    Notes
    -----
    - For real-valued tensors, Q is orthogonal: Q.T @ Q = I.
    - The QR sign ambiguity is fixed to make diag(R) nonnegative.

    This avoids SciPy and stays on the chosen device.

    Returns
    -------
    Q : (n, n) tensor
    """
    A = torch.randn((n, n), dtype=dtype, device=device, generator=generator)
    Q, R = torch.linalg.qr(A, mode="reduced")
    # Fix sign ambiguity so it's more stable/reproducible in distribution.
    diag = torch.diagonal(R)
    phase = torch.sign(diag)
    phase = torch.where(phase == 0, torch.ones_like(phase), phase)
    Q = Q * phase  # broadcasts over columns
    return Q


def rand_square(
    n: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return an (n, n) standard normal matrix."""
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
    """
    Construct a random (l, r) matrix with full row rank (right-invertible when r >= l).

    Construction
    -----------
    A = U_l @ [diag(s) | 0] @ U_r
    where U_l (l×l) and U_r (r×r) are random orthogonal matrices.

    Parameters
    ----------
    l:
        Number of rows.
    r:
        Number of columns. If None or < l, set r = l.
    s:
        Singular values for the l rows (shape (l,)). If None, uses all ones.
    dtype, device, generator:
        Tensor options.

    Returns
    -------
    A : (l, r) tensor
    """
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

    # Build [diag(s) | 0] as (l, r)
    S = torch.zeros((l, r), dtype=dtype, device=device)
    S[:, :l] = torch.diag(s_vec)

    return U_l @ S @ U_r
