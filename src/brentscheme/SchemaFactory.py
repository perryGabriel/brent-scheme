from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch


Preset = Literal["random", "complex", "naive", "fourier", "strassen", "winograd", "laderman"]


@dataclass(slots=True)
class SchemaFactory:
    """
    Factory for initializing Brent-style bilinear matrix multiplication schemes.

    A scheme is expected to be a mutable object with attributes:
      n, d, m, p : int
      alpha_pnd : Tensor (p, n, d)
      beta__pdm : Tensor (p, d, m)
      gamma_nmp : Tensor (n, m, p)
      TRIPLE_DELTA_nmnddm : Tensor (n, m, n, d, d, m) (set by set_triple_delta)

    The factory also relies on SchemeManipulator for:
      - set_norm(scheme, norm=..., field=...)
      - set(scheme, alpha, beta, gamma)
      - change_basis(scheme, L=..., M=..., R=...)
    """

    dtype_real: torch.dtype = torch.float64
    dtype_complex: torch.dtype = torch.complex128

    # ---------------- core helpers ----------------

    def set_triple_delta(self, scheme) -> None:
        """Set the exact bilinear tensor for standard matrix multiplication."""
        scheme.TRIPLE_DELTA_nmnddm = torch.einsum(
            "ac,Ab,BC->cCaAbB",
            torch.eye(scheme.n, dtype=self.dtype_real),
            torch.eye(scheme.d, dtype=self.dtype_real),
            torch.eye(scheme.m, dtype=self.dtype_real),
        )

    def _set_norm(self, scheme, *, norm, field: str) -> None:
        from brentscheme.SchemeManipulator import SchemeManipulator
        SchemeManipulator().set_norm(scheme, norm=norm, field=field)

    # ---------------- public API ----------------

    def set_scheme(
        self,
        scheme,
        preset: Preset = "random",
        *,
        n: Optional[int] = None,
        d: Optional[int] = None,
        m: Optional[int] = None,
        p: Optional[int] = None,
        fourier: Optional[int] = None,
    ) -> None:
        """
        Configure `scheme` with a preset.

        Parameters
        ----------
        preset:
          - "random": random real tensors
          - "complex": random complex tensors
          - "naive": exact multiplication with p = n*d*m
          - "fourier": complex naive + Fourier basis change
          - "strassen", "winograd", "laderman": hard-coded classics
        n, d, m, p:
          Override scheme dimensions.
        fourier:
          If provided, forces preset="fourier" and chooses Fourier level.
        """
        # Determine sizes
        if n is not None:
            scheme.n = int(n)
        elif getattr(scheme, "n", None) is None:
            scheme.n = 2

        if d is not None:
            scheme.d = int(d)
        elif getattr(scheme, "d", None) is None:
            scheme.d = scheme.n

        if m is not None:
            scheme.m = int(m)
        elif getattr(scheme, "m", None) is None:
            scheme.m = scheme.n

        if p is not None:
            scheme.p = int(p)
        elif getattr(scheme, "p", None) is None:
            scheme.p = scheme.n * scheme.d * scheme.m

        # Fourier overrides preset
        if fourier is not None:
            preset = "fourier"

        match preset:
            case "random":
                self.set_random(scheme, field="R", norm=2)
            case "complex":
                self.set_random(scheme, field="C", norm=1)
            case "naive":
                self.set_naive(scheme, field="R", norm=1)
            case "fourier":
                level = int(0 if fourier is None else fourier)
                # level 0 is "complex naive"
                self.set_naive(scheme, field="C", norm=1)
                self.apply_fourier_basis(scheme, level=level)
            case "strassen":
                self.set_strassen(scheme)
            case "winograd":
                self.set_winograd(scheme)
            case "laderman":
                self.set_laderman(scheme)
            case _:
                raise ValueError(f"Unknown preset: {preset!r}")

    def set_random(
        self,
        scheme,
        *,
        re_std: float = 1.0,
        im_std: float = 1.0,
        norm=2,
        field: Literal["R", "C"] = "R",
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Random scheme tensors; real or complex."""
        p, n, d, m = scheme.p, scheme.n, scheme.d, scheme.m

        alpha = torch.randn((p, n, d), dtype=self.dtype_real, generator=generator) * re_std
        beta = torch.randn((p, d, m), dtype=self.dtype_real, generator=generator) * re_std
        gamma = torch.randn((n, m, p), dtype=self.dtype_real, generator=generator) * re_std

        if field == "C":
            alpha = alpha.to(self.dtype_complex) + 1j * (
                torch.randn((p, n, d), dtype=self.dtype_real, generator=generator) * im_std
            ).to(self.dtype_complex)
            beta = beta.to(self.dtype_complex) + 1j * (
                torch.randn((p, d, m), dtype=self.dtype_real, generator=generator) * im_std
            ).to(self.dtype_complex)
            gamma = gamma.to(self.dtype_complex) + 1j * (
                torch.randn((n, m, p), dtype=self.dtype_real, generator=generator) * im_std
            ).to(self.dtype_complex)

        scheme.alpha_pnd = alpha
        scheme.beta__pdm = beta
        scheme.gamma_nmp = gamma

        self._set_norm(scheme, norm=norm, field=field)
        self.set_triple_delta(scheme)

    def set_naive(self, scheme, *, norm=1, field: Literal["R", "C"] = "R") -> None:
        """
        Exact scheme with p = n*d*m.
        Produces real tensors; if field="C", promotes to complex dtype.
        """
        scheme.p = scheme.n * scheme.d * scheme.m

        # Identities (real)
        L = torch.eye(scheme.n, dtype=self.dtype_real)
        M = torch.eye(scheme.d, dtype=self.dtype_real)
        R = torch.eye(scheme.m, dtype=self.dtype_real)
        b = torch.ones(scheme.n, dtype=self.dtype_real)
        c = torch.ones(scheme.d, dtype=self.dtype_real)
        a = torch.ones(scheme.m, dtype=self.dtype_real)

        alpha = torch.einsum("ia,i,j,Ak,k->ijkaA", L, b, a, M, c).reshape((scheme.p, scheme.n, scheme.d))
        beta = torch.einsum("i,jB,j,kb,k->ijkbB", b, R, a, M, c).reshape((scheme.p, scheme.d, scheme.m))
        gamma = torch.einsum("ci,i,Cj,j,k->cCijk", L, b, R, a, c).reshape((scheme.n, scheme.m, scheme.p))

        if field == "C":
            alpha = alpha.to(self.dtype_complex)
            beta = beta.to(self.dtype_complex)
            gamma = gamma.to(self.dtype_complex)
        else:
            alpha = alpha.to(self.dtype_real)
            beta = beta.to(self.dtype_real)
            gamma = gamma.to(self.dtype_real)

        scheme.alpha_pnd = alpha
        scheme.beta__pdm = beta
        scheme.gamma_nmp = gamma

        self._set_norm(scheme, norm=norm, field=field)
        self.set_triple_delta(scheme)

    def apply_fourier_basis(self, scheme, *, level: int = 2) -> None:
        """
        Apply Fourier/Vandermonde basis changes.

        level:
          0: no-op (already complex naive if called from set_scheme)
          1: apply basis change on M only
          2: apply basis change on L, M, R
          3+: currently same as level 2 (product-axis Fourier not implemented)
        """
        if level <= 0:
            return

        # Ensure complex
        scheme.alpha_pnd = scheme.alpha_pnd.to(self.dtype_complex)
        scheme.beta__pdm = scheme.beta__pdm.to(self.dtype_complex)
        scheme.gamma_nmp = scheme.gamma_nmp.to(self.dtype_complex)

        vander_n = self._fourier_vandermonde(scheme.n, dtype=self.dtype_complex)
        vander_d = self._fourier_vandermonde(scheme.d, dtype=self.dtype_complex)
        vander_m = self._fourier_vandermonde(scheme.m, dtype=self.dtype_complex)

        from brentscheme.SchemeManipulator import SchemeManipulator
        manip = SchemeManipulator()

        if level == 1:
            manip.change_basis(scheme, M=vander_d)
        else:
            manip.change_basis(scheme, L=vander_n, M=vander_d, R=vander_m)

        # Recompute exact tensor (still real-valued identity structure, but stored on scheme)
        self.set_triple_delta(scheme)

    def _fourier_vandermonde(self, n: int, *, dtype: torch.dtype) -> torch.Tensor:
        """
        Vandermonde matrix built from nth roots of unity:
          w = exp(2πi/n), vec = [w^0, w^1, ..., w^{n-1}],
          V[k, j] = vec[k]^j  (increasing powers).
        """
        k = torch.arange(n, dtype=torch.float64)
        w = torch.exp(2j * torch.pi * k / n).to(dtype)  # w_k = exp(2πi k/n)
        return torch.vander(w, increasing=True).to(dtype)

    # ---------------- named classic presets ----------------

    def set_strassen(self, scheme) -> None:
        scheme.n = scheme.d = scheme.m = 2
        scheme.p = 7
        scheme.alpha_pnd = torch.Tensor([[[ 1,0],[0, 1]],[[ 0,0],[1, 1]],[[ 1,0],[0, 0]],[[ 0,0],[0, 1]],[[ 1,1],[0, 0]],[[-1,0],[1, 0]],[[ 0,1],[0,-1]]]).type(torch.float64)
        scheme.beta__pdm  = torch.Tensor([[[ 1,0],[0, 1]],[[ 1,0],[0, 0]],[[ 0,1],[0,-1]],[[-1,0],[1, 0]],[[ 0,0],[0, 1]],[[ 1,1],[0, 0]],[[ 0,0],[1, 1]]]).type(torch.float64)
        scheme.gamma_nmp = torch.Tensor([[[1, 0,0,1,-1,0,1], [0, 0,1,0, 1,0,0]],[[0, 1,0,1, 0,0,0], [1,-1,1,0, 0,1,0]]]).type(torch.float64)
        self._set_norm(scheme, norm=torch.inf, field="R")
        self.set_triple_delta(scheme)

    def set_winograd(self, scheme) -> None:
        scheme.n = scheme.d = scheme.m = 2
        scheme.p = 7
        scheme.alpha_pnd = torch.Tensor([[[ 1,0],[ 0, 0]],[[ 0,1],[ 0, 0]],[[ 1,1],[-1,-1]],[[ 0,0],[ 0, 1]],[[-1,0],[ 1, 0]],[[ 0,0],[ 1, 1]],[[-1,0],[ 1, 1]]]).type(torch.float64)
        scheme.beta__pdm  = torch.Tensor([[[ 1, 0],[0, 0]],[[ 0, 0],[1, 0]],[[ 0, 0],[0, 1]],[[-1, 1],[1,-1]],[[ 0, 1],[0,-1]],[[-1, 1],[0, 0]],[[ 1,-1],[0, 1]]]).type(torch.float64)
        scheme.gamma_nmp = torch.Tensor([[[1,1,0,0,0,0,0], [1,0,1,0,0,1,1]],[[1,0,0,1,1,0,1], [1,0,0,0,1,1,1]]]).type(torch.float64)
        self._set_norm(scheme, norm=torch.inf, field="R")
        self.set_triple_delta(scheme)

    def set_laderman(self, scheme) -> None:
        # Keeping your hard-coded tensors exactly; only modernizing tensor construction.
        scheme.n = scheme.d = scheme.m = 3
        scheme.p = 23
        scheme.alpha_pnd = torch.Tensor([[[ 1,1, 1],[-1,-1, 0],[ 0,-1,-1]],[[ 1,0, 0],[-1, 0, 0],[ 0, 0, 0]],[[ 0,0, 0],[ 0, 1, 0],[ 0, 0, 0]],[[-1,0, 0],[ 1, 1, 0],[ 0, 0, 0]],[[ 0,0, 0],[ 1, 1, 0],[ 0, 0, 0]], # 5
                               [[ 1,0, 0],[ 0, 0, 0],[ 0, 0, 0]],[[-1,0, 0],[ 0, 0, 0],[ 1, 1, 0]],[[-1,0, 0],[ 0, 0, 0],[ 1, 0, 0]],[[ 0,0, 0],[ 0, 0, 0],[ 1, 1, 0]],[[ 1,1, 1],[ 0,-1,-1],[-1,-1, 0]], # 10
                               [[ 0,0, 0],[ 0, 0, 0],[ 0, 1, 0]],[[ 0,0,-1],[ 0, 0, 0],[ 0, 1, 1]],[[ 0,0, 1],[ 0, 0, 0],[ 0, 0,-1]],[[ 0,0, 1],[ 0, 0, 0],[ 0, 0, 0]],[[ 0,0, 0],[ 0, 0, 0],[ 0, 1, 1]], # 15
                               [[ 0,0,-1],[ 0, 1, 1],[ 0, 0, 0]],[[ 0,0, 1],[ 0, 0,-1],[ 0, 0, 0]],[[ 0,0, 0],[ 0, 1, 1],[ 0, 0, 0]],[[ 0,1, 0],[ 0, 0, 0],[ 0, 0, 0]],[[ 0,0, 0],[ 0, 0, 1],[ 0, 0, 0]], # 20
                               [[ 0,0, 0],[ 1, 0, 0],[ 0, 0, 0]],[[ 0,0, 0],[ 0, 0, 0],[ 1, 0, 0]],[[ 0,0, 0],[ 0, 0, 0],[ 0, 0, 1]]]).type(torch.float64)
        scheme.beta__pdm  = torch.Tensor([[[ 0, 0, 0],[0, 1, 0],[ 0, 0, 0]],[[ 0,-1, 0],[0, 1, 0],[ 0, 0, 0]],[[-1, 1, 0],[1,-1,-1],[-1, 0, 1]],[[ 1,-1, 0],[0, 1, 0],[ 0, 0, 0]],[[-1, 1, 0],[0, 0, 0],[ 0, 0, 0]], # 5
                               [[ 1, 0, 0],[0, 0, 0],[ 0, 0, 0]],[[ 1, 0,-1],[0, 0, 1],[ 0, 0, 0]],[[ 0, 0, 1],[0, 0,-1],[ 0, 0, 0]],[[-1, 0, 1],[0, 0, 0],[ 0, 0, 0]],[[ 0, 0, 0],[0, 0, 1],[ 0, 0, 0]], # 10
                               [[-1, 0, 1],[1,-1,-1],[-1, 1, 0]],[[ 0, 0, 0],[0, 1, 0],[ 1,-1, 0]],[[ 0, 0, 0],[0, 1, 0],[ 0,-1, 0]],[[ 0, 0, 0],[0, 0, 0],[ 1, 0, 0]],[[ 0, 0, 0],[0, 0, 0],[-1, 1, 0]], # 15
                               [[ 0, 0, 0],[0, 0, 1],[ 1, 0,-1]],[[ 0, 0, 0],[0, 0, 1],[ 0, 0,-1]],[[ 0, 0, 0],[0, 0, 0],[-1, 0, 1]],[[ 0, 0, 0],[1, 0, 0],[ 0, 0, 0]],[[ 0, 0, 0],[0, 0, 0],[ 0, 1, 0]], # 20
                               [[ 0, 0, 1],[0, 0, 0],[ 0, 0, 0]],[[ 0, 1, 0],[0, 0, 0],[ 0, 0, 0]],[[ 0, 0, 0],[0, 0, 0],[ 0, 0, 1]]]).type(torch.float64)
        scheme.gamma_nmp = torch.Tensor([[[0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0], [1,0,0,1,1,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,1,0,1,1,0,0,0,1,0,1,0,1,0,0,0,0,0]],
                               [[0,1,1,1,0,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0], [0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0]],
                               [[0,0,0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0], [0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]]).type(torch.float64)
        self._set_norm(scheme, norm=torch.inf, field="R")
        self.set_triple_delta(scheme)

    # ---------------- persistence ----------------

    def read_from_files(
        self,
        scheme,
        *,
        filename: Optional[str] = None,
        n: Optional[int] = None,
        d: Optional[int] = None,
        m: Optional[int] = None,
        p: Optional[int] = None,
        number: Optional[float] = None,
        directory: Path | str = ".",
        verbose: int = 0,
    ) -> None:
        """
        Load alpha/beta/gamma from pickle files and set scheme tensors.

        You may provide either:
          - filename="2_2_2_8_e10.000"
          - or (n,d,m,p,number)
          - or number alone (interpreted as filename header = "{number:.3f}")
        """
        base = Path(directory)

        if filename is None:
            if number is None:
                raise ValueError("Must provide filename=... or number=... (and optionally sizes).")
            if n is None and d is None and m is None and p is None:
                filename = f"{float(number):.3f}"
            elif None not in (n, d, m, p):
                filename = f"{n}_{d}_{m}_{p}_e{float(number):.3f}"
            else:
                raise ValueError("If providing sizes, must provide n,d,m,p all together.")

        import pickle

        def load(name: str):
            with (base / f"{filename}_{name}.pkl").open("rb") as f:
                return pickle.load(f)

        alpha = load("alpha_pnd")
        beta = load("beta__pdm")
        gamma = load("gamma_nmp")

        from brentscheme.SchemeManipulator import SchemeManipulator
        SchemeManipulator().set(scheme, alpha, beta, gamma)

        self.set_triple_delta(scheme)

        if verbose > 0:
            from brentscheme.SchemeDisplay import SchemeDisplay
            SchemeDisplay().report(scheme, verbose=verbose)

    # ---------------- scheme algebra ----------------

    def compose_schemes(self, outer, inner):
        """
        Kronecker-like composition of two schemes, producing a larger scheme.

        Result shapes:
          n = n_out * n_in,  d = d_out * d_in,  m = m_out * m_in,  p = p_out * p_in
        """
        from brentscheme.BrentScheme import BrentScheme

        result = BrentScheme(n=outer.n * inner.n, d=outer.d * inner.d, m=outer.m * inner.m, p=outer.p * inner.p)

        self.set_triple_delta(result)

        result.gamma_nmp = torch.einsum("cCi,zZj->czCZij", outer.gamma_nmp, inner.gamma_nmp).reshape(
            (result.n, result.m, result.p)
        )
        result.alpha_pnd = torch.einsum("iaA,jxX->ijaxAX", outer.alpha_pnd, inner.alpha_pnd).reshape(
            (result.p, result.n, result.d)
        )
        result.beta__pdm = torch.einsum("ibB,jyY->ijbyBY", outer.beta__pdm, inner.beta__pdm).reshape(
            (result.p, result.d, result.m)
        )
        return result

    def degenerate_scheme(
        self,
        scheme,
        *,
        alpha_pnd: Optional[torch.Tensor] = None,
        beta__pdm: Optional[torch.Tensor] = None,
        gamma_nmp: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Fill in exactly one missing tensor among (alpha, beta, gamma) by pseudoinverse.

        Exactly two of the three must be provided.

        Notes
        -----
        This is algebraically fragile; it will be numerically unstable if the implied
        linear system is ill-conditioned.
        """
        missing = (alpha_pnd is None) + (beta__pdm is None) + (gamma_nmp is None)
        if missing != 1:
            raise ValueError("Exactly two of alpha_pnd, beta__pdm, gamma_nmp must be provided.")

        # Infer dimensions from provided tensors
        if alpha_pnd is not None:
            p_, n_, d_ = alpha_pnd.shape
        else:
            p_, n_, d_ = None, None, None

        if beta__pdm is not None:
            p2, d2, m_ = beta__pdm.shape
        else:
            p2, d2, m_ = None, None, None

        if gamma_nmp is not None:
            n2, m2, p3 = gamma_nmp.shape
        else:
            n2, m2, p3 = None, None, None

        # Set scheme sizes consistently
        scheme.n = n_ if n_ is not None else n2
        scheme.d = d_ if d_ is not None else d2
        scheme.m = m_ if m_ is not None else m2
        scheme.p = p_ if p_ is not None else (p2 if p2 is not None else p3)

        # Assign provided tensors
        if alpha_pnd is not None:
            scheme.alpha_pnd = alpha_pnd
        if beta__pdm is not None:
            scheme.beta__pdm = beta__pdm
        if gamma_nmp is not None:
            scheme.gamma_nmp = gamma_nmp

        # Solve for missing tensor
        if alpha_pnd is None:
            if scheme.beta__pdm.shape[2] != scheme.gamma_nmp.shape[1]:
                raise ValueError("Incompatible beta/gamma dimensions (m mismatch).")
            # Solve alpha from contraction over (d,m) and (n,m)
            X = torch.einsum("pdm,nmp->ndp", scheme.beta__pdm, scheme.gamma_nmp).reshape(-1, scheme.p)
            scheme.alpha_pnd = torch.linalg.pinv(X).reshape(scheme.p, scheme.n, scheme.d)

        elif beta__pdm is None:
            if scheme.alpha_pnd.shape[1] != scheme.gamma_nmp.shape[0]:
                raise ValueError("Incompatible alpha/gamma dimensions (n mismatch).")
            X = torch.einsum("pnd,nmp->dmp", scheme.alpha_pnd, scheme.gamma_nmp).reshape(-1, scheme.p)
            scheme.beta__pdm = torch.linalg.pinv(X).reshape(scheme.p, scheme.d, scheme.m)

        else:  # gamma_nmp is None
            if scheme.alpha_pnd.shape[2] != scheme.beta__pdm.shape[1]:
                raise ValueError("Incompatible alpha/beta dimensions (d mismatch).")
            X = torch.einsum("pnd,pdm->pnm", scheme.alpha_pnd, scheme.beta__pdm).reshape(scheme.p, -1)
            scheme.gamma_nmp = torch.linalg.pinv(X).reshape(scheme.n, scheme.m, scheme.p)

        # (Optional) sanity checks for known degenerate families; raise rather than print
        self._validate_degenerate_family(scheme)

        # Keep TRIPLE_DELTA consistent with inferred n,d,m
        self.set_triple_delta(scheme)

    def _validate_degenerate_family(self, scheme) -> None:
        """
        Optional: enforce requirements for degenerate problem classes.
        """
        if scheme.n == 1 and scheme.m == 1:
            # inner product family (fine)
            return
        if scheme.d == 1:
            # outer product family (fine)
            return
        if scheme.n == 1:
            # vector-matrix product family
            return
        if scheme.m == 1:
            # matrix-vector product family
            return
