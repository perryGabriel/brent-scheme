import torch

from brentscheme.utils.tensors import hosvd, mode_n_product


def reconstruct(core: torch.Tensor, factors: list[torch.Tensor]) -> torch.Tensor:
    out = core
    for mode, U in enumerate(factors):
        out = mode_n_product(out, U, mode)
    return out


def test_hosvd_full_reconstruction_matches_input():
    torch.manual_seed(0)
    X = torch.randn(4, 3, 2, dtype=torch.float64)

    core, factors = hosvd(X)
    X_hat = reconstruct(core, factors)

    assert X_hat.shape == X.shape
    assert torch.allclose(X_hat, X, atol=1e-10, rtol=1e-10)


def test_hosvd_truncated_reduces_rank_and_has_nonzero_error():
    torch.manual_seed(0)
    X = torch.randn(6, 5, 4, dtype=torch.float64)

    core, factors = hosvd(X, ranks=(3, 3, 2))
    X_hat = reconstruct(core, factors)

    assert core.shape == (3, 3, 2)
    assert factors[0].shape == (6, 3)
    assert factors[1].shape == (5, 3)
    assert factors[2].shape == (4, 2)

    rel = torch.linalg.norm(X - X_hat) / torch.linalg.norm(X)
    assert rel > 0
    assert rel < 1
