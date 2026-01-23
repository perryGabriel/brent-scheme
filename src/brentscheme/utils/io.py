from __future__ import annotations

from pathlib import Path
from typing import Literal


def delete_scheme_files(
    n: int,
    d: int,
    m: int,
    p: int,
    error: float,
    *,
    directory: str | Path = ".",
    missing_ok: bool = True,
) -> None:
    """
    Delete saved scheme tensor files for a given parameterization.

    Expected filenames:
      {n}_{d}_{m}_{p}_e{error:.3f}_alpha_pnd.pkl
      {n}_{d}_{m}_{p}_e{error:.3f}_beta__pdm.pkl
      {n}_{d}_{m}_{p}_e{error:.3f}_gamma_nmp.pkl
    """
    base = Path(directory)
    prefix = f"{n}_{d}_{m}_{p}_e{error:.3f}_"
    for name in ("alpha_pnd.pkl", "beta__pdm.pkl", "gamma_nmp.pkl"):
        (base / (prefix + name)).unlink(missing_ok=missing_ok)


def delete_diagram_file(
    n: int,
    d: int,
    m: int,
    p: int,
    error: float,
    *,
    directory: str | Path = ".",
    missing_ok: bool = True,
) -> None:
    """
    Delete a saved diagram image.

    Expected filename:
      {n}_{d}_{m}_scheme_{p}_prod_{error:.3f}_best.png
    """
    base = Path(directory)
    filename = f"{n}_{d}_{m}_scheme_{p}_prod_{error:.3f}_best.png"
    (base / filename).unlink(missing_ok=missing_ok)

def delete_file(n, d, m, p, number, scheme_or_diagram):
  if scheme_or_diagram == 'scheme':
     delete_scheme_files(n, d, m, p, number)
  if scheme_or_diagram == 'diagram':
    delete_diagram_file(n, d, m, p, number)