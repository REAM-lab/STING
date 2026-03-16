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
    $ git clone https://github.com/REAM-lab/sting
    $ cd sting
    ```
    Next, create a virtual environment and download all required packages.
    ```
    $ python3.13 -m venv .venv 
    $ source .venv/bin/activate
    (.venv)$ pip install -e .  
    ```

2. **Run sting**: To ensure that sting was installed correctly navigate to the examples folder. You will see examples for different modules. Find the file `run.py` and execute it.

### Solvers

Most of modules require commercial or open-source solvers to run various optimization models. Even small-signal modeling requires a solver to run optimal power flow. Make sure you have a solver installed in your computer. We are currently offering support for the following solvers:

| Solver | How to install      | Use                |
|--------|---------------------|--------------------|
| IPOPT  | `pip install ipopt` | AC OPF             |
| Gurobi | `pip install ipopt` | Capacity expansion |
| MOSEK  | `pip install mosek` | Capacity expansion |

### SLICOT

Some of our model reduction algorithms use the `slycot` python wrapper for FORTRAN SLICOT routines. Please refer to the `slycot` documentation for full installation instructions
#### OSX 
For Mac users your can build `slycot` with support from brew. Run `brew install gcc` and then add `slycot` in your virtual environment with pip, `pip install slycot`.

### EMT simulation with SPS (Deprecated)

Currently, we are offering a library of EMT models in Simulink using Specialized Power Systems (SPS) models. Unfortunately [MATLAB has dropped support for the SPS library](https://www.mathworks.com/matlabcentral/answers/2180147-unable-to-find-the-specialized-power-systems-group-in-simscape-electrical-in-newer-version-r2025b) in versions after 2025a. As such, we are actively working to replace these EMT models with pure Python scripts for EMT simulation.

1. **Open SPS library**: Make sure that you have MATLAB R2025a. Go to the folder `sps_library`. Open the library, and make sure that it is open while you are running EMT simulation with our testcases.