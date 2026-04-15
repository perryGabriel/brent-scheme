# Experiments catalog (cleaned documentation)

The repository historically used the folder name `expiriments/` (typo preserved for compatibility). This `experiments/` folder documents those notebooks and tracks migration into testable Python code.

## Notebook inventory and status

| Notebook (legacy path) | Main idea | Current assessment | Migration status |
|---|---|---|---|
| `expiriments/HOSVD.ipynb` | Higher-order SVD factorization of tensors. | Useful and general-purpose. | Migrated to `brentscheme.utils.tensors.hosvd`. |
| `expiriments/Timing Brent Scheme Optimizers.ipynb` | Compare optimizer wall-time/performance. | Useful; needs reproducible benchmark harness. | Partially documented; scripted benchmark TODO. |
| `expiriments/Cataloging Quality and Sizes of Error Catchments.ipynb` | Empirical basin/catchment mapping by initialization quality. | Promising but exploratory. | Documented; needs reproducible seed sweep script. |
| `expiriments/Dimension of Solution Manifold Investigation.ipynb` | Tangent-space and dimension counting for exact manifolds. | Valuable theory exploration; mostly derivational. | Documented as research notes; not moved to runtime package. |
| `expiriments/Logging Accuracy of Scheme Size for Various p.ipynb` | Error-vs-rank (p) tradeoff curves. | Useful for model selection. | Candidate for CLI/report script in future. |
| `expiriments/Momentum and Geodesics.ipynb` | Optimizer geometry and geodesic-inspired steps. | Interesting, unclear convergence gains yet. | Scripted starter added: `experiments/attempts/momentum_and_geodesics.py`. |
| `expiriments/Experiment with Refactoring_Adding Products (NUMPY).ipynb` | Product insertion/refactoring heuristics in NumPy. | Mixed; some ideas need clearer objective constraints. | Scripted baseline added: `experiments/attempts/add_products_refactor.py`. |
| `expiriments/Brent Interesting Scheme Derivations NOT READY.ipynb` | Mixed notebook: 2x2x2 rank-6 search, 3x3x3 p=22 training, basis playground. | Multiple partially independent ideas. | Split into documented attempts (see below). |
| `expiriments/Brent Schemes Taking the Real Part at Various Points NOT READY.ipynb` | Project complex schemes to real part at different stages. | Potentially useful but numerically delicate. | Scripted baseline added: `experiments/attempts/real_projection_experiment.py`. |
| `expiriments/alpha_evolve matmult new schemes.ipynb` | Integration and inspection of AlphaEvolve-discovered decompositions. | Useful external reference; licensing/context required. | Kept as reference notebook; not vendored into core code. |

## Split-out attempts from mixed notebooks

The notebook **"Brent Interesting Scheme Derivations NOT READY"** contained multiple independent threads. These are now split conceptually into:

1. Rank-6 attempt for `<2,2,2>` search (documented as unresolved/experimental).
2. `3x3x3` with `p=22` approximate training workflow.
3. Basis-playground transformations and diagnostics.

Where possible, these are represented as standalone scripts under `experiments/attempts/` with explicit TODOs and reproducible seeds.

## Migration policy for notebooks

- If an idea is reusable and stable, migrate it into `src/brentscheme` + tests.
- If an idea is promising but not production-ready, keep a script in `experiments/attempts` with assumptions and limitations.
- If an idea is unclear or failed, preserve it as a documented negative result.
