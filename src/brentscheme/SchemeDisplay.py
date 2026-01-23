from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass(slots=True)
class SchemeMetrics:
    log10_L1: float
    log10_L2: float
    log10_Linf: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.log10_L1, self.log10_L2, self.log10_Linf)


@dataclass(slots=True)
class SchemeDisplay:
    """
    Reporting / evaluation / persistence utilities for a BrentScheme-like object.

    Expected scheme interface:
      - n, d, m, p
      - complexity() -> float
      - L_norm, field
      - forward() -> Tensor  (either (n,m,n,d,d,m) or already comparable to TRIPLE_DELTA)
      - TRIPLE_DELTA_nmnddm : Tensor broadcast-compatible with forward()
      - measure(tensor) -> Tensor (scalar)

      - alpha_pnd, beta__pdm, gamma_nmp tensors for verbose algebra display
    """

    directory: Path = Path(".")
    precision: int = 3
    missing_ok: bool = True

    # ---------- public API ----------

    def summary(self, scheme) -> str:
        """Return a one-paragraph human readable summary."""
        return (
            f"Scheme for ({scheme.n}×{scheme.d}) @ ({scheme.d}×{scheme.m}) using p={scheme.p} "
            f"out of {scheme.n * scheme.d * scheme.m} products; "
            f"complexity n^{scheme.complexity():.3f}\n"
            f"Norm: L{scheme.L_norm} over field {scheme.field}"
        )

    def error(self, scheme) -> float:
        """
        Return log10(measure(error)) like your old test(verbose=0).
        """
        error = self._error_tensor(scheme)
        return float(torch.log10(scheme.measure(error)).item())

    def metrics(self, scheme) -> SchemeMetrics:
        """
        Return (log10 L1, log10 L2, log10 Linf) averaged over entries,
        matching your old test(verbose=1).
        """
        error = self._error_tensor(scheme)
        mags = error.abs()
        err_size = mags.numel()

        L1 = mags.sum() / err_size
        L2 = (mags.square().sum() / err_size).sqrt()
        Linf = mags.max()

        return SchemeMetrics(
            log10_L1=float(torch.log10(L1).item()),
            log10_L2=float(torch.log10(L2).item()),
            log10_Linf=float(torch.log10(Linf).item()),
        )

    def report(self, scheme, *, verbose: int = 0) -> None:
        """
        Print a report. (Side-effecting convenience wrapper.)
        verbose:
          0: summary only
          1: summary + metrics line
          2: + prints algebraic decomposition (large)
          3: + shows triple-delta plots (very large)
        """
        print(self.summary(scheme))

        if verbose >= 1:
            mets = self.metrics(scheme)
            print(
                f"Avg L1 error: 10^{mets.log10_L1:.4f}, "
                f"Avg L2 error: 10^{mets.log10_L2:.4f}, "
                f"Max error: 10^{mets.log10_Linf:.4f}"
            )

        if verbose >= 2:
            print()
            print(self.algebra(scheme))

        if verbose >= 3:
            self.plot_triple_deltas(scheme)

    def algebra(self, scheme) -> str:
        """
        Return a (potentially long) string describing the bilinear form:
          P_i = (alpha_i ⋅ A) * (beta_i ⋅ B)
          AB = gamma ⋅ P
        """
        lines: list[str] = []
        fmt = f" .{self.precision}f"

        # Products
        lines.append("Products P_i = (alpha_i · A) * (beta_i · B)")
        for pi in range(scheme.p):
            a_terms = []
            for aj in range(scheme.n):
                for ak in range(scheme.d):
                    c = scheme.alpha_pnd[pi, aj, ak].item()
                    if c != 0.0:
                        a_terms.append(f"{c:{fmt}}*A[{aj+1},{ak+1}]")
            b_terms = []
            for bj in range(scheme.d):
                for bk in range(scheme.m):
                    c = scheme.beta__pdm[pi, bj, bk].item()
                    if c != 0.0:
                        b_terms.append(f"{c:{fmt}}*B[{bj+1},{bk+1}]")

            a_sum = " + ".join(a_terms) if a_terms else "0"
            b_sum = " + ".join(b_terms) if b_terms else "0"
            lines.append(f"P_{pi+1} = ({a_sum}) * ({b_sum})")

        lines.append("")
        lines.append("Outputs AB = gamma · P")
        for i in range(scheme.n):
            for j in range(scheme.m):
                terms = []
                for k in range(scheme.p):
                    c = scheme.gamma_nmp[i, j, k].item()
                    if c != 0.0:
                        terms.append(f"{c:{fmt}}*P_{k+1}")
                rhs = " + ".join(terms) if terms else "0"
                lines.append(f"AB[{i+1},{j+1}] = {rhs}")

        return "\n".join(lines)

    def plot_triple_deltas(self, scheme, *, output: Optional[torch.Tensor] = None) -> None:
        """
        Plot exact vs approximation vs error. Requires matplotlib.
        """
        import matplotlib.pyplot as plt

        if output is None:
            output = self._flatten(scheme.forward(), scheme)
        else:
            output = self._flatten(output, scheme)

        target = self._flatten(scheme.TRIPLE_DELTA_nmnddm, scheme)
        error = output - target

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(target.detach().cpu(), cmap="seismic", interpolation="nearest")
        plt.title("Exact")

        plt.subplot(1, 3, 2)
        plt.imshow(output.detach().cpu(), cmap="seismic", interpolation="nearest")
        plt.title("Approximation")

        plt.subplot(1, 3, 3)
        plt.imshow(error.detach().cpu(), cmap="seismic", interpolation="nearest")
        plt.title(f"Max Error: {error.abs().max().item():.5f}")

        plt.show()

    def dump_tensors(self, scheme, *, score: Optional[float] = None) -> float:
        """
        Save alpha/beta/gamma tensors to pickle files in `directory`.

        If score is None, computes it via `error()` and uses that in the filename.
        Returns the rounded score used in filenames.
        """
        import pickle

        used = round(self.error(scheme) if score is None else score, 3)
        prefix = self._prefix(scheme, used)

        self.directory.mkdir(parents=True, exist_ok=True)

        for suffix, tensor in [
            ("alpha_pnd.pkl", scheme.alpha_pnd),
            ("beta__pdm.pkl", scheme.beta__pdm),
            ("gamma_nmp.pkl", scheme.gamma_nmp),
        ]:
            path = self.directory / f"{prefix}{suffix}"
            with path.open("wb") as f:
                pickle.dump(tensor, f)

        return used

    def delete_tensors(self, scheme, *, score: float) -> None:
        """Delete the tensor pickle files for this scheme/score."""
        used = round(score, 3)
        prefix = self._prefix(scheme, used)
        for suffix in ("alpha_pnd.pkl", "beta__pdm.pkl", "gamma_nmp.pkl"):
            (self.directory / f"{prefix}{suffix}").unlink(missing_ok=self.missing_ok)

    # ---------- internals ----------

    def _prefix(self, scheme, score: float) -> str:
        return f"{scheme.n}_{scheme.d}_{scheme.m}_{scheme.p}_e{score:.3f}_"

    def _flatten(self, T: torch.Tensor, scheme) -> torch.Tensor:
        n = scheme.n * scheme.d * scheme.m
        return T.reshape((n, n))

    def _error_tensor(self, scheme) -> torch.Tensor:
        return scheme.forward() - scheme.TRIPLE_DELTA_nmnddm
