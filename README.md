# :zap:sting:zap:

Welcome! This repository contains sting—**S**pecialized **T**ool for **IN**verter-based **G**rids. It is a Python package for small-signal modeling, electromagnetic transient (EMT) simulation, and large-scale reduction methods for power systems 

## Installation 

1. **Download STING**: Make sure you have [python3.13](https://www.python.org/downloads/release/python-31311/) installed on your computer. Using [`pyenv`](https://github.com/pyenv/pyenv) can be helpful for managing multiple versions of python on your PC. Start by cloning this repository and navigating into the STING directory.
    ```
    $ git clone https://github.com/REAM-lab/sting
    $ cd sting
    ```
    Next, create a virtual environment and download all required packages.
    ```
    $ python3.12 -m venv .venv 
    $ source .venv/bin/activate
    (.venv)$ pip install -e .  
    ```

2. **Install gamspy**: First register for a [gams account](https://www.gams.com/academics/) using academic email if you do not already have one. Then download a gamspy Local License from their website. A license is a either a 36 character access code or an ASCII file of six lines. In order to install your license, all you need to do is to run:
    ```
    (.venv)$ gamspy install license <access code or path_to_ascii_file>
    ```
    You can run:
    ```
    (.venv)$ gamspy show license
    ```
    to verify the installation of the license. You need to install some solvers to run power flow or other optimization models in STING. Install ipopt in GAMSPy, you can execute this command in your terminal having your python environment activated.
    ```
    (.venv)$ gamspy install solver ipopt
     ```
    
4. **Run sting**: To ensure that sting was installed correctly navigate to the examples folder. You will see testcases. Execute the file run.py.

## EMT simulation (Optional)

Currently, we are offering a library of EMT models in Simulink using Specialized Power Systems (SPS) models. The idea is to replace these EMT models with pure Python scripts for EMT simulation.
We are working on it. Make sure that you have MATLAB R2025a.

1. **Open SPS library**: Go to the folder sps_library. Open the library, and make sure that it is open
while you are running EMT simulation with our testcases.