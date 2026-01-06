# Evacuation Traffic Queuing Simulation


## Motivation



## Key Findings (from the paper)



## Repository Structure

```text
evacuation-traffic-games/
├─ notebooks/
│  └─ evac_simulation.ipynb          # original Colab/interactive exploration
├─ scripts/
│  └─ run_experiments.py             # command-line entrypoint to reproduce figures/results
├─ src/
│  └─ evacsim/
│     ├─ __init__.py                 # public API re-exports (see below)
│     ├─ sim.py                      # network generation + simulation loop(s)
│     ├─ policies.py                 # signal-control policies
│     └─ plotting.py                 # plotting utilities for paper-style figures
├─ CS_501R_Final_Project_Traffic_Queuing__IEEE_.pdf  # paper/report
├─ pyproject.toml                    # packaging config (editable install support)
├─ README.md
└─ .gitignore
```
## Usage

To use this package locally, install it directly from this github repository:

`pip install "git+https://github.com/perryGabriel/evacuation-traffic-games.git"`

The script `scripts/run_expiriments.py` provided in this repository is a minimal working example to replicate the results from the paper, included as a PDF in this repository.
