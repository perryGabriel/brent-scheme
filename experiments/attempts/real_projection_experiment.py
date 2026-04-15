"""Attempt script: project complex schemes onto reals at different stages.

This script mirrors ideas from:
- expiriments/Brent Schemes Taking the Real Part at Various Points NOT READY.ipynb

It is intentionally lightweight and reproducible. The goal is to compare:
1. Real part taken directly on tensors (alpha, beta, gamma).
2. Real part taken on final bilinear tensor map.
"""

from __future__ import annotations

import torch

from brentscheme import BrentScheme, SchemaFactory, SchemeDisplay


def run(seed: int = 0) -> dict[str, float]:
    torch.manual_seed(seed)
    scheme = BrentScheme(n=2, d=2, m=2, preset="complex")

    # Baseline error in complex field
    baseline = SchemeDisplay().error(scheme)

    # Projection at coefficient level
    projected = scheme.clone()
    projected.alpha_pnd = projected.alpha_pnd.real.to(torch.float64)
    projected.beta__pdm = projected.beta__pdm.real.to(torch.float64)
    projected.gamma_nmp = projected.gamma_nmp.real.to(torch.float64)
    SchemaFactory().set_triple_delta(projected)
    coeff_projection_error = SchemeDisplay().error(projected)

    # Projection at map level
    map_projection = (scheme.forward()).real
    target = scheme.TRIPLE_DELTA_nmnddm
    map_projection_error = torch.log10(torch.mean(torch.abs(map_projection - target))).item()

    return {
        "complex_baseline_log10": baseline,
        "coeff_projection_log10": coeff_projection_error,
        "map_projection_log10": map_projection_error,
    }


if __name__ == "__main__":
    results = run()
    for k, v in results.items():
        print(f"{k}: {v:.6f}")
