# Brent-Type Tensor Schemes for Fast Matrix Multiplication


## Motivation



## Key Findings (from the paper)



## Repository Structure

```text
brent-scheme/
├─ notebooks/
│  ├─ NUMPY Mat_Mult.ipynb           # original Colab/interactive exploration
│  └─ TORCH_Mat_Mult.ipynb           # updated Colab/interactive exploration
├─ scripts/
│  ├─ 1_init_test.py                 # test the BrentScheme object initialization
│  ├─ 2_display_tests.py             # test the SchemeDisplay object methods
│  ├─ 3_factory_tests.py             # test the SchemaFactory object methods
│  ├─ 4_manipulator_tests.py         # test the SchemeManipulator object methods
│  ├─ 5_stepper_tests.py             # test the Stepper object methods
│  └─ 6_trainer_tests.py             # test the Trainer object methods
├─ src/
│  └─ brentscheme/
│     ├─ __init__.py                 # public API re-exports (see below)
│     ├─ BrentScheme.py              # A tensor Scheme for multiplying matrices
│     ├─ misc.py                     # Misc helper Functions
│     ├─ SchemaFactory.py            # A Factory for setting preset schema
│     ├─ SchemeDisplay.py            # A Display object for printing and saving data about a scheme, including accuracy tests
│     ├─ SchemeManipulatior.py       # A Manipulation object for an existing scheme
│     ├─ Stepper.py                  # A Single-Step trainer for schema
│     └─ Trainer.py                  # A Multi-step trainer for schema
├─ pyproject.toml                    # packaging config (editable install support)
├─ README.md
└─ .gitignore
```
## Usage

To use this package locally, install it directly from this github repository:

`pip install "git+https://github.com/perryGabriel/brent-scheme.git"
from brentscheme import *`

The unit tests are avalable as `scripts/*.py` provided in this repository is a minimal working example to replicate the results from the paper, included as a PDF in this repository.

## Citation

The Bibtex for citing this repository is:

```
@misc{perry2026_brent_scheme,
  author = {Perry, Gabriel M.},
  title = {Brent Scheme},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\\\url{https://github.com/perryGabriel/brent-scheme}},
  note = {Accessed: 2026-01-06}
}
```