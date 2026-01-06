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
│  └─ run_experiments.py             # command-line entrypoint to reproduce figures/results
├─ src/
│  └─ brentscheme/
│     ├─ __init__.py                 # public API re-exports (see below)
│     ├─ BrentScheme.py              # 
│     ├─ misc.py                     # 
│     ├─ SchemaFactory.py            # 
│     ├─ SchemeDisplay.py            # 
│     ├─ SchemeManipulatior.py       # 
│     ├─ Stepper.py                  # 
│     └─ Trainer.py                  # 
├─ pyproject.toml                    # packaging config (editable install support)
├─ README.md
└─ .gitignore
```
## Usage

To use this package locally, install it directly from this github repository:

`pip install "git+https://github.com/perryGabriel/brent-scheme.git"`

The script `scripts/run_expiriments.py` provided in this repository is a minimal working example to replicate the results from the paper, included as a PDF in this repository.
