import numpy as np

def tune_pll(w_n: float,
                        dr: float) -> dict:
    """
    Inputs:
    - w_n [rad/s]:  natural frequency of the response of the angle estimated by the PLL when the grid angle is perturbed.
                    A value like 50 rad/s is for strong grid, whereas 5 rad/s is for weak grid. 
    - dr [pu]:  damping ratio of the response of the angle estimated by the PLL when the grid angle is perturbed. 
                Value of 1 is for overdamped response, whereas 0.707 for fast but still overshoot.
    
    Outputs:
    - kp [rad/s]: proportional gain of the PLL controller
    - ki [rad^2/s^2]: integral gain of the PLL controller

    Note: The units of kp and ki are suitable for a PLL that receives the voltages in per unit
    and outputs frequency in Hz, angular frequency in rad/s, and angle in rad.
    """

    # Calculate proportional gain
    kp = 2 * dr * w_n

    # Calculate integral gain
    ki = w_n**2

    print(f"PLL controller Gains: kp = {kp:.4f}, ki = {ki:.4f}")

    return {
        "kp": kp,
        "ki": ki
    }