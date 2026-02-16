# :zap:sting:zap:

Welcome! This repository contains sting—**S**pecialized **T**ool for **IN**verter-based **G**rids. STING is an open-source software that is able to run:

- AC Power Flow
- Stochastic capacity expansion
- Kron Reduction
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

2. **Run sting**: To ensure that sting was installed correctly navigate to the examples folder. You will see examples for differents modules. Find the file run.py and execute it.

## Requirements

Most of modules require solvers to solve optimization models, even small-signal modeling requires to solve optimal power flow. Make sure you have a solver installed in your
computer. Here, we show a list of solvers we have used:

- Ipopt
- Gurobi
- Mosek 

## EMT simulation with SPS (Deprecated)

Currently, we are offering a library of EMT models in Simulink using Specialized Power Systems (SPS) models. The idea is to replace these EMT models with pure Python scripts for EMT simulation.
We are working on it. Make sure that you have MATLAB R2025a.

1. **Open SPS library**: Go to the folder sps_library. Open the library, and make sure that it is open
while you are running EMT simulation with our testcases.