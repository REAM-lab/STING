import pytest
import os

from sting.system.core import System
from sting import main

@pytest.mark.parametrize("dataset", ["2-bus_2-src", "3-bus_src-gfm", "5-bus_1-src-3-gfm", "9-bus_3-src"])
def test_emt_ssm(dataset):
    
    case_dir=os.path.join(os.getcwd(), "tests", "temp_outputs")
    os.makedirs(case_dir, exist_ok=True)

    # Step function inputs to simulate
    step1 = lambda t: 0.1 if t >= 0.5 else 0.0
    inputs = {'infinite_sources_0': {'v_ref_d': step1}}
    t_max = 1.0 # Simulation length in seconds
    # Load dataset
    system = System.from_dataset(dataset=dataset)
    system.case_directory = case_dir
    # Construct system and small-signal model
    _, ssm = main.run_ssm(system=system, case_directory=case_dir)
    ssm.simulate_ssm(t_max=t_max, inputs=inputs)
    # Run EMT simulation
    main.run_emt(inputs=inputs, t_max=t_max, system=system, case_directory=case_dir)

    emt_dir = os.path.join(case_dir, "outputs", "simulation_emt")
    ssm_dir = os.path.join(case_dir, "outputs", "small_signal_model")

    # Difference between EMT and SSM for each state
    delta = dict()
    for component in system:
        if hasattr(component, "compare_ssm_emt"):
            delta |= getattr(component, "compare_ssm_emt")(emt_dir, ssm_dir)

    assert max([max(abs(d)) for d in delta.values()]) < 0.05