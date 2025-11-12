# :zap:STING:zap:

Welcome! This repository contains the STING—**S**pecialized **T**ool for **IN**verter-based **G**rids.

## Installation 


1. **Download STING**: Make sure you have [python3.14](https://www.python.org/downloads/) (or greater) installed on your computer. Start by cloning this repository and navigating into the STING directory.
    ```
    $ git clone https://github.com/REAM-lab/STING
    $ cd STING
    ```
    Next, create a virtual environment and download all required packages.
    ```
    $ python3.14 -m venv .venv 
    $ source .venv/bin/activate
    (.venv)$ pip install -e .  
    ```

2. **Install GAMSPy**: First register for a [GAMS account](https://www.gams.com/academics/) using academic email if you do not already have one. Then download a GAMSPy Local License from their website. A license is a either a 36 character access code or an ASCII file of six lines. In order to install your license, all you need to do is to run:
    ```
    (.venv)$ gamspy install license <access code or path_to_ascii_file>
    ```
    You can run:
    ```
    (.venv)$ gamspy show license
    ```
    to verify the installation of the license.

3. **Run STING**: To ensure that STING was installed correctly navigate to the examples folder. Then, within your python virtual environment, launch python3.14 and execute `run_ssm()`.
    ```
    (.venv)$ cd ./examples/testcase1/
    (.venv)$ python3.14
    >>> from sting.main import run_ssm
    >>> run_ssm()
    ```

## TODOs
1. Model reduction module
2. Additional docs