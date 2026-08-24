import numpy as np


def tune_lcl_filter(s_base: float, 
                    v_base: float, 
                    f_base: float, 
                    v_dc: float, 
                    lf2_pu,
                    f_sw: float = 5*10**3,
                    reactive_limit: float = 0.05,
                    ripple_ratio: float = 0.20) -> dict:
    """
    Inputs:
    - s_base: three-phase base power in VA
    - v_base: base line-to-line voltage in V
    - f_base: base frequency in Hz
    - v_dc: DC voltage in V
    - lf2_pu: per unit value of the second inductance (it is normally a transformer leakage inductance)
    - f_sw: switching frequency in Hz (default is 5 kHz)
    - reactive_limit: maximum reactive power limit in per unit (default is 0.05 pu)
    - ripple_ratio: maximum ripple ratio (lower than 0.40 is recommended, default is 0.20)
    
    - Outputs:
    - lf1_pu: per unit value of the first inductance
    - lf2_pu: per unit value of the second inductance
    - cf_pu: per unit value of the capacitance
    - w_res: resonant frequency in rad/s
    """

    # Calculate the switching angular frequency
    w_sw = 2 * np.pi * f_sw

    # Calculate the base impedance, base capacitance, and base angular frequency
    z_base = v_base**2 / s_base
    i_base = s_base / (np.sqrt(3) * v_base)
    c_base = 1 / (2 * np.pi * f_base * z_base)
    w_base = 2 * np.pi * f_base

    # Calculate the capacitance in Farads based on the reactive power limit
    cf = reactive_limit * c_base

    # Calculate the maximum current ripple in Amperes 
    delta_i_max = ripple_ratio * np.sqrt(2) * i_base

    # Use formula to calculate the first inductance in Henry and resonant frequency
    lf1 = v_dc / (6 * f_sw * delta_i_max)

    # Calculate the second inductance on Henry
    lf2 = lf2_pu * z_base / w_base

    # Compute the resonant frequency of the LCL filter
    w_res = np.sqrt((lf1 + lf2) / (lf1 * lf2 * cf))

    print(f"w_base = {w_base:.2f}, w_sw = {w_sw:.2f}, w_res = {w_res:.2f}")

    if (10 * w_base < w_res) & (w_res < 0.5 * w_sw): 
        print("Resonant frequency is within the acceptable range.")
    else:
        raise ValueError(f"Resonant frequency {w_res:.2f} is not within the acceptable range")

    # Compute series resistance 
    rd = 1/3 * 1/(w_res * cf)
    print(f"rd = {rd:.6f} Ohm")

    lf1_pu = lf1 * w_base / z_base
    cf_pu = cf / c_base
    rd_pu = rd / z_base
    
    print(f"lf1_pu = {lf1_pu:.6f}, lf2_pu = {lf2_pu:.6f}, cf_pu = {cf_pu:.6f}, rd_pu = {rd_pu:.6f}")

    return {'lf1_pu': lf1_pu, 'lf2_pu': lf2_pu, 'cf_pu': cf_pu, 'w_res': w_res}

fs = 5*10**3
udc = 700
s_base = 6000
v_base = 380 * np.sqrt(3)
f_base = 60
lf2 = 0.08
ripple_ratio = 0.30
reactive_limit = 0.03
filter_params = tune_lcl_filter(s_base, v_base, f_base, udc, lf2, fs, reactive_limit=reactive_limit, ripple_ratio=ripple_ratio)



