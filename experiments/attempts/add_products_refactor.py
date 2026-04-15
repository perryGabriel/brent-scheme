"""Attempt script: adding products in a controlled way.

Inspired by:
- expiriments/Experiment with Refactoring_Adding Products (NUMPY).ipynb

This script demonstrates a conservative, explicit baseline:
start from a known exact scheme and append zero-contributing products,
then verify the represented bilinear map is unchanged.
"""

from __future__ import annotations

import torch

from brentscheme import BrentScheme, SchemeDisplay


def append_zero_product(scheme: BrentScheme) -> None:
    """Append one all-zero product slice, preserving represented map."""
    alpha_pad = torch.zeros((1, scheme.n, scheme.d), dtype=scheme.alpha_pnd.dtype)
    beta_pad = torch.zeros((1, scheme.d, scheme.m), dtype=scheme.beta__pdm.dtype)
    gamma_pad = torch.zeros((scheme.n, scheme.m, 1), dtype=scheme.gamma_nmp.dtype)

    scheme.alpha_pnd = torch.cat([scheme.alpha_pnd, alpha_pad], dim=0)
    scheme.beta__pdm = torch.cat([scheme.beta__pdm, beta_pad], dim=0)
    scheme.gamma_nmp = torch.cat([scheme.gamma_nmp, gamma_pad], dim=2)
    scheme.p += 1


def run() -> float:
    scheme = BrentScheme(n=2, d=2, m=2, preset="naive")
    append_zero_product(scheme)
    return SchemeDisplay().error(scheme)


if __name__ == "__main__":
    err = run()
    print(f"log10 error after append-zero-product: {err}")
