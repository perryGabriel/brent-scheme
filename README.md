# BrentScheme: Tensor Schemes for Fast Matrix Multiplication

`brentscheme` is a research-oriented Python package for constructing, evaluating, and training bilinear tensor decompositions of matrix multiplication maps.

At a high level, the package studies schemes of the form:

\[
(n \times d) @ (d \times m) \rightarrow (n \times m)
\]

using `p` scalar products, represented by three tensors \(\alpha, \beta, \gamma\).

## What this repository now includes

- A package-style source layout (`src/brentscheme`) with importable modules.
- Preset schemes (naive, Strassen, Winograd, Laderman, Fourier variants).
- Training and manipulation utilities for experimenting with low-rank or approximate decompositions.
- Expanded unit tests to guard numerical correctness and optimization behavior.
- Experiment documentation under [`experiments/`](experiments/README.md) with a notebook-by-notebook status log and migration notes.
- A new HOSVD utility in `brentscheme.utils.tensors` for decomposition experiments.

## Installation

### Editable install (recommended for development)

```bash
pip install -e .
```

### Standard install from GitHub

```bash
pip install "git+https://github.com/perryGabriel/brent-scheme.git"
```

## Quickstart

```python
import torch
from brentscheme import BrentScheme, SchemaFactory, SchemeDisplay

scheme = BrentScheme(n=2, d=2, m=2, preset="naive")
A = torch.randn(2, 2, dtype=torch.float64)
B = torch.randn(2, 2, dtype=torch.float64)

C_scheme = scheme(A, B)
C_exact = A @ B

print(torch.max(torch.abs(C_scheme - C_exact)))  # ~0 for exact schemes
print(SchemeDisplay().summary(scheme))
```

## Testing

Run the complete test suite:

```bash
pytest -q
```

## Repository structure

```text
brent-scheme/
├─ src/brentscheme/            # library source
├─ tests/                      # pytest-based unit tests
├─ expiriments/                # original exploratory notebooks (legacy spelling kept)
├─ experiments/                # cleaned documentation + scripted notebook attempts
├─ models/                     # saved tensor checkpoints
└─ notebooks/                  # additional interactive demos
```

## Notes on API style (static-utility classes)

The current API uses utility-style classes (`SchemaFactory`, `SchemeManipulator`, `Stepper`, `Trainer`) that are instantiated but mostly stateless. This is still supported; see the forward-looking migration plan in:

- [`docs/API_REFACTOR_PLAN.md`](docs/API_REFACTOR_PLAN.md)

## Citation

```bibtex
@misc{perry2026_brent_scheme,
  author = {Perry, Gabriel M.},
  title = {Brent Scheme},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/perryGabriel/brent-scheme}},
  note = {Accessed: 2026-01-06}
}
```
