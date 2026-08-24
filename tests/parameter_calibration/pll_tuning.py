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

    print(f"Tuned gains of the PLL: kp [rad/s] = {kp:.4f}, ki [rad^2/s^2] = {ki:.4f}")

    return {
        "kp": kp,
        "ki": ki
    }

def tune_inner_current_controller(wcc: float, rf: float, xf: float, wnom: float) -> tuple:
	"""
	It tunes the inner current control loop based on the parameters of the filter.
	
	Inputs:
	- wcc [rad/s]: bandwidth of the inner current controller. It is the inverse of the time constant 
                   of the inner current controller. Values like 10000 rad/s up to 20000 rad/s can be used.
                   The idea is that the current controller should be faster.
                   This means that if the reference current is a step, the controlled current in the filter 
                   should almost match the reference.
	- rf [pu]: resistance of the filter [pu]. For example, the current controller
               is used in a GFL where the filter is an LCL filter, then rf is the resistance 
               of the first branch and second branch rf = rf1 + rf2
               However, if the current controller is used in a GFM that has current controller
               and voltage controller in cascade, then rf = rf1 because the current controller
               is only controlling the current in the first branch of the LCL filter.
	- xf [pu]: reactance of the filter [pu]. As the resistance, it is the 
               sum of the reactances of the first and second branches of the LCL filter xf = xf1 + xf2.
               In the case of a GFM, xf = xf1 because the current controller is only controlling 
               the current in the first branch of the LCL filter.
	- wnom [rad/s]: nominal frequency [rad/s]. It is 2 * pi * f_base, where f_base is the base frequency of the system in Hz.

	Outputs:
	- kp_pu [pu]: proportional gain of the inner current controller
	- ki_puHz [pu/s]: integral gain of the inner current controller
	"""
     
	# Compute kp and ki of the inner current control loop based on tuning formulas
	tau = 1 / wcc # time constant of the inner current controller [s]
	kp_cc_pu = xf * wcc / wnom # proportional gain of the inner current controller [pu]
	ki_cc_puHz = rf * wcc # integral gain of the inner current controller [pu/s]

	print(f'Tuned gains of the inner current controller: kp [pu]: {kp_cc_pu}, ki [pu/s]: {ki_cc_puHz}, tau [s]: {tau}')
     
	return { "kp": kp_cc_pu, "ki": ki_cc_puHz }

def tune_inner_voltage_controller(wcc: float, phase_margin: float, cf: float, wnom: float) -> tuple:
    """
    It tunes the inner voltage control loop based on the parameters of the filter.

    Inputs:
    - wcc [rad/s]: bandwidth of the inner voltage controller. Similar to the bandwith of the inner current controller.
    - phase_margin [rad]: phase margin of the inner voltage controller. It should be between 30 and 75 degrees.
    - cf [pu]: capacitance of the filter's sunt in per unit.
    - wnom [rad/s]: nominal frequency [rad/s]. It is 2 * pi * f_base, where f_base is the base frequency of the system in Hz.

    Outputs:
    - kp [pu]: proportional gain of the inner voltage controller
    - ki [pu/s]: integral gain of the inner voltage controller
    """
     
    z = wcc * (1 - np.sin(phase_margin)) / (1 + np.sin(phase_margin))
    wm = np.sqrt(wcc * z)

    kp = (cf * wm) / wnom
    ki = (cf * wm**3) / (wnom * wcc)

    print(f'Tuned parameters for voltage controller - kp [pu]: {kp}, ki [pu/s]: {ki}')

    return { "kp": kp, "ki": ki }