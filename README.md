# :zap:sting:zap:

Welcome! This repository contains sting—**S**pecialized **T**ool for **IN**verter-based **G**rids. STING is an open-source software that is able to run:

- AC Power Flow
- Stochastic capacity expansion
- Kron reduction
- Large-scale model reduction
- Small-signal modeling
- Electromagnetic simulation


## Installation 

1. **Download STING**: Make sure you have [python3.13](https://www.python.org/downloads/release/python-31311/) installed on your computer. Using [`pyenv`](https://github.com/pyenv/pyenv) can be helpful for managing multiple versions of python on your PC. Start by cloning this repository and navigating into the STING directory.
    ```
    git clone https://github.com/REAM-lab/sting
    cd sting
    ```
    Next, create a virtual environment and download all required packages.
    ```
    python3.13 -m venv .venv 
    source .venv/bin/activate
    pip install -e .  
    ```
To install all optional dependencies, run  `pip install -e ".[all]"`. This will install extra packages necessary for optimization, specifically solvers.

2. **Run sting**: To ensure that sting was installed correctly navigate to the examples folder. You will see examples for different modules. Find the file `run.py` and execute it.

### Solvers

Most of modules additionally require commercial or open-source solvers to run various optimization models. For example, solving optimal power flow is needed to find an equilibrium point for small-signal modeling. We currently support and use the following libraries

| Solver | How to install in your python environment     | Use                |
|--------|-----------------------------------------------|--------------------|
| IPOPT  | `brew install ipopt`                          | ACOPF              |
| Gurobi | `pip install gurobipy`                        | Capacity expansion |
| MOSEK  | `pip install mosek`                           | Capacity expansion |

### Installing STING for a Project
For research projects we recommend creating a separate directory for your source files and scripts---isolating them from the source files for STING. To install and configure STING in this manner perform the following steps.
0.  **Download STING**: If you haven't already, install [python3.13](https://www.python.org/downloads/release/python-31311/) and clone STING
    ```
    git clone https://github.com/REAM-lab/sting
    ```
1. **Project Installation**: In the *same* parent directory of STING, create a new directory for your project---for instance `my_project`. Then install all packages from STING in a virtual environment
    ```
    mkdir my_project
    cd my_project
    python3.13 -m venv .venv 
    source .venv/bin/activate
    pip install -e ../sting/.
    ```
2. **Visual Studio Code (Optional)**: If you are using Visual Studio Code as a text editor you can add STING as an "extra path". To do so go to `Preferences > Settings` and search for extra paths. Under `Python > Analysis: Extra Paths` add the global path of STING on your machine---for instance `Users/your_name/python/sting`.

## Citing
```
@misc{STING,
    author = {{Renewable Energy + Advanced Mathematics Lab (REAM)}},
    title = {Specialized tool for inverter-based grids},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    url = {https://github.com/REAM-lab/sting}
}
```