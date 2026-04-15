"""Attempt script: momentum/geodesic-inspired training baseline.

Inspired by:
- expiriments/Momentum and Geodesics.ipynb

Current implementation keeps a practical baseline using Adam from Stepper/Trainer.
Future work can replace this with manifold-aware updates.
"""

from __future__ import annotations

from brentscheme import BrentScheme, SchemeDisplay, Trainer


def run(epochs: int = 40, lr: float = 1e-3) -> tuple[float, float]:
    scheme = BrentScheme(n=2, d=2, m=2, p=7, preset="random")
    display = SchemeDisplay()
    before = display.metrics(scheme).log10_L1

    Trainer().train(scheme, epochs=epochs, batch_size=1, lr=lr, verbose=0)

    after = display.metrics(scheme).log10_L1
    return before, after


if __name__ == "__main__":
    before, after = run()
    print(f"L1 log10 before: {before:.6f}")
    print(f"L1 log10 after : {after:.6f}")
