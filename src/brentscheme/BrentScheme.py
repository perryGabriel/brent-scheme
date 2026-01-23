from __future__ import annotations
import math
from typing import Any, Optional
import torch
import torch.nn as nn


class BrentScheme(nn.Module):
    """
    A bilinear matrix multiplication scheme for shapes:

        (n x d) @ (d x m) -> (n x m)

    The scheme is parameterized by three tensors (typically learned or preset):

        alpha:  shape (p, n, d)   (named alpha_pnd)
        beta:   shape (p, d, m)   (named beta__pdm)   # note: your code uses beta__pdm
        gamma:  shape (n, m, p)   (named gamma_nmp)

    If called with A and B, computes the (n x m) product via the bilinear form.
    If called with A=None or B=None, returns the 6D tensor representing the bilinear map.
    """

    def __init__(
        self,
        *,
        n: int = 2,
        d: Optional[int] = None,
        m: Optional[int] = None,
        p: Optional[int] = None,
        preset: str = "random",
        verbose: int = 0,
        device: Optional[torch.device | str] = None,
        **extra: Any,
    ) -> None:
        super().__init__()

        self.n = int(n)
        self.d = int(self.n if d is None else d)
        self.m = int(self.n if m is None else m)

        size = self.n * self.d * self.m
        self.p = int(size if p is None else p)

        # Optional: store device preference (but don't force-move parameters here).
        self._device_hint = device

        # If you truly want to accept arbitrary extra config, store it safely:
        self.extra: dict[str, Any] = dict(extra)

        if verbose > 0:
            print(
                f"A scheme for ({self.n} x {self.d}) @ ({self.d} x {self.m}) "
                f"using {self.p} products: complexity is n^{self.complexity():.3f}"
            )

        # NOTE: this imports inside __init__ to avoid import cycles at module import time.
        from brentscheme.SchemaFactory import SchemaFactory

        factory = SchemaFactory()
        factory.set_scheme(self, preset=preset)
        factory.set_triple_delta(self)

    # ---------- Python protocols / dunders ----------

    def __repr__(self) -> str:
        # nn.Module's repr can get huge; a compact repr is often nicer.
        return (
            f"{self.__class__.__name__}(n={self.n}, d={self.d}, m={self.m}, p={self.p}, "
            f"complexity={self.complexity():.3f})"
            # f"error={}"
        )

    def __iter__(self):
        """
        Iterate over the core tensors (gamma, alpha, beta) in a consistent order.
        Useful for quick unpacking: gamma, alpha, beta = scheme
        """
        yield self.gamma_nmp
        yield self.alpha_pnd
        yield self.beta__pdm

    # ---------- Core methods ----------

    def clone(self) -> "BrentScheme":
        """
        Return a deep-ish copy of the scheme parameters into a new BrentScheme instance
        with the same shapes.

        Note: this does NOT preserve optimizer state; it's for copying parameters only.
        """
        new = BrentScheme(n=self.n, d=self.d, m=self.m, p=self.p, preset="random", device=self._device_hint)

        # Avoid assumptions about attribute existence until after SchemaFactory has set them.
        from brentscheme.SchemeManipulator import SchemeManipulator

        manipulator = SchemeManipulator()
        manipulator.set(
            new,
            self.alpha_pnd.detach().clone(),
            self.beta__pdm.detach().clone(),
            self.gamma_nmp.detach().clone(),
        )
        return new

    def complexity(self) -> float:
        """Return the exponent ω-like proxy: 3 * log_{n*d*m}(p)."""
        size = self.n * self.d * self.m
        if size == 1:
            return float(self.p)
        return 3.0 * math.log(self.p, size)

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        """
        Measure a tensor with a user-defined norm + inverse-normalization.

        Expects the instance to define:
            - self.L_norm: a norm order or torch.inf
            - self.norm: callable Tensor -> Tensor
            - self.inv_norm: callable Tensor -> Tensor

        If L_norm == inf, returns inv_norm(norm(x)).
        Otherwise returns inv_norm(norm(x) / numel(x)).
        """
        # This method is intentionally explicit about its dependencies.
        L_norm = getattr(self, "L_norm", None)
        if L_norm is None:
            raise AttributeError("BrentScheme.measure expects self.L_norm to be set.")
        if not hasattr(self, "norm") or not hasattr(self, "inv_norm"):
            raise AttributeError("BrentScheme.measure expects self.norm and self.inv_norm to be set.")

        if L_norm == torch.inf:
            return self.inv_norm(self.norm(x))
        return self.inv_norm(self.norm(x) / x.numel())

    def forward(
        self,
        A_nd: Optional[torch.Tensor] = None,
        B_dm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        If A_nd and B_dm are provided, compute A @ B using the scheme.

        If either is None, return the 6D tensor representing the bilinear map.
        """
        if A_nd is None or B_dm is None:
            # Produces shape (n, m, n, d, d, m) with your original einsum indices.
            return torch.einsum(
                "cCi,iaA,ibB->cCaAbB",
                self.gamma_nmp,
                self.alpha_pnd,
                self.beta__pdm,
            )

        # Compute output shape (n, m) according to your einsum.
        return torch.einsum(
            "cCi,iaA,ibB,aA,bB->cC",
            self.gamma_nmp,
            self.alpha_pnd,
            self.beta__pdm,
            A_nd,
            B_dm,
        )
