import pytest
import os

from sting.system.core import System
from sting import main
import numpy as np

# Collection of step functions to test with
step1 = lambda t: 0.1 if t >= 0.5 else 0.0
step2 = lambda t: -0.05 if t >= 0.5 else 0.0
step3 = lambda t: 0.075 if t >= 0.25 else 0.0
step4 = lambda t: -0.1 if t >= 0.25 else 0.0

def evaluate_dataset(dataset, inputs, tol=0.05):
    """
    Helper function to evaluate the time domain response of EMT and SSM.
    """
    # Set up a temporary directory used by all tests
    case_dir=os.path.join(os.getcwd(), "tests", "tmpdir")
    os.makedirs(case_dir, exist_ok=True)

    t_max = 1.0 # Simulation length in seconds
    # Load dataset
    dataset_dir = os.path.join(os.getcwd(), "tests", "test_datasets", dataset)
    system = System.from_csv(dataset_dir)
    system.case_directory = case_dir
    # Construct system and small-signal model
    _, ssm = main.run_ssm(system=system, case_directory=case_dir)
    ssm.simulate_ssm(t_max=t_max, inputs=inputs)
    # Run EMT simulation
    main.run_emt(inputs=inputs, t_max=t_max, system=system, case_directory=case_dir)

    emt_dir = os.path.join(case_dir, "outputs", "simulation_emt")
    ssm_dir = os.path.join(case_dir, "outputs", "small_signal_model")

    # Difference between EMT and SSM for each state
    responses = dict()
    for component in system:
        if hasattr(component, "compare_ssm_emt"):
            responses |= getattr(component, "compare_ssm_emt")(emt_dir, ssm_dir)

    for component_state, (emt, ssm) in responses.items():
        # Check that EMT simulations have a flat start
        start_tol = 5e-4
        n = 50
        start_error = max(abs(np.mean(emt[:n]) - emt[:n]))
        assert start_error < start_tol, f"The initial EMT response of state {component_state} "\
            f"has a maximum deviation of {start_error} in the first {n} time points. This exceeds "\
            f" the {start_tol} threshold. Please check the your simulation has a flat start."

        # Check that the time domain responses are close
        max_error = max(abs(emt - ssm))
        assert max_error < tol, f"The state {component_state} has an maximum absolute error of {max_error} "\
            f"between the EMT and SSM time domain response. This exceeds the expected {tol} threshold."

# Test 2- and 9-bus systems
@pytest.mark.parametrize("dataset", ["2-bus_2-src", "9-bus_3-src"])
@pytest.mark.parametrize("inputs", [
    {'infinite_sources_0': {'v_ref_d': step1}, 'infinite_sources_1': {'v_ref_q': step2}},
    {'infinite_sources_0': {'v_ref_q': step2}, 'infinite_sources_1': {'v_ref_d': step4}},
])
def test_infinite_source(dataset, inputs):
    evaluate_dataset(dataset=dataset, inputs=inputs)

# Test 3- and 5-bus systems
@pytest.mark.parametrize("dataset", ["3-bus_src-gfm", "5-bus_1-src-3-gfm"])
@pytest.mark.parametrize("inputs", [
    {'gfmi_c_0': {'p_ref': step2}},
    {'infinite_sources_0': {'v_ref_q': step2}, 'gfmi_c_0': {'v_ref': step1}},
])
def test_gfmi_c(dataset, inputs):
    evaluate_dataset(dataset=dataset, inputs=inputs)