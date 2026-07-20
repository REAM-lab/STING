# ------------------------------------------------------------------------------
# Import libraries
# ------------------------------------------------------------------------------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables

# ---------------------------------------
# Subclasses
# ---------------------------------------
class InitialConditionsEMT(NamedTuple):
    # Bus-side voltage and currents
    v_bus_d: float
    v_bus_q: float
    v_bus_D: float
    v_bus_Q: float
    i_bus_d: float
    i_bus_q: float
    i_bus_D: float
    i_bus_Q: float
    # Capacitor voltage
    v_cap_d: float
    v_cap_q: float
    # Converter-side voltage and current
    v_vsc_d: float
    v_vsc_q: float
    i_vsc_d: float
    i_vsc_q: float

# ---------------------------------------
# Main class
# ---------------------------------------
@dataclass(slots=True)
class LCLFilter6B:
    """
    The LCL filter connects the VSC to the grid. It has three branches: the first branch (RL) connects
    the VSC to the series shunt element (RC), and the third branch (RL) connects the series shunt element to the grid.

    Parameters:
    - rf1_pu: resistance [pu] of first branch of filter
    - xf1_pu: inductance [pu] of first branch of filter
    - rf2_pu: resistance [pu] of second branch of filter
    - xf2_pu: inductance [pu] of second branch of filter
    - rsh_pu: resistance [pu] of series RC shunt
    - csh_pu: capacitance [pu] of series RC shunt
    - wbase: nominal frequency [rad/s] of the system

    Graphical representation of the LCL filter:

    converter |----rf1----xf1-------+------rf2----xf2----| pcc or grid bus
                                    |
                                    rsh
                                    |
                                    csh
                                    |
                                    neutral
    """
    rf1_pu: float
    xf1_pu: float
    rsh_pu: float
    csh_pu: float
    rf2_pu: float
    xf2_pu: float
    wbase: float

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, v_bus_mag: float, v_bus_angle: float, p_bus: float, q_bus: float) -> InitialConditionsEMT:
        """
        Calculate the steady-state values of the LCL filter given the bus voltage magnitude, relative phase, 
        and the active and reactive power at the bus.
        Consider that: 
        DQ: reference frame of the grid
        dq: reference frame of the inverter

        Inputs:
        - v_bus_mag: bus voltage magnitude [pu]
        - v_bus_angle: bus voltage angle [degrees]
        - p_bus: active power at the bus [pu]
        - q_bus: reactive power at the bus [pu]

        Outputs:
        - InitialConditionsEMT: a named tuple containing the steady-state values.
        """
        
        # Voltage at the point of common coupling (PCC) in xy reference frame
        v_bus_DQ = v_bus_mag * np.exp(v_bus_angle * np.pi / 180 * 1j)

        # Current sent at the PCC in DQ reference frame
        i_bus_DQ = np.conj( (p_bus + 1j * q_bus) / v_bus_DQ)

        # Voltage across the shunt element in the LCL filter
        v_sh_DQ = v_bus_DQ + (self.rf2_pu + self.xf2_pu * 1j) * i_bus_DQ

        # Current flowing through shunt element of LCL filter
        i_sh_DQ = v_sh_DQ / (1/self.csh_pu * -1j + self.rsh_pu)

        # Voltage across capacitor in the LCL filter
        v_cap_DQ = v_sh_DQ - self.rsh_pu * i_sh_DQ

        # Current sent from the beginning of the LCL filter (VSC terminal)
        i_vsc_DQ = i_bus_DQ + i_sh_DQ
        v_vsc_DQ = v_sh_DQ + (self.rf1_pu + self.xf1_pu * 1j) * i_vsc_DQ

        # We refer the voltage and currents to the reference frame of the inverter
        angle_ref = np.angle(v_vsc_DQ) * 180 / np.pi
        v_vsc_dq = v_vsc_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)
        i_vsc_dq = i_vsc_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)
        v_bus_dq = v_bus_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)
        i_bus_dq = i_bus_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)
        v_cap_dq = v_cap_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        self.emt_init = InitialConditionsEMT(
            # Bus 
            v_bus_d=v_bus_dq.real,
            v_bus_q=v_bus_dq.imag,

            v_bus_D=v_bus_DQ.real,
            v_bus_Q=v_bus_DQ.imag,

            i_bus_d=i_bus_dq.real,
            i_bus_q=i_bus_dq.imag,

            i_bus_D=i_bus_DQ.real,
            i_bus_Q=i_bus_DQ.imag,

            # Capacitor
            v_cap_d=v_cap_dq.real,
            v_cap_q=v_cap_dq.imag,

            # Converter
            v_vsc_d=v_vsc_dq.real,
            v_vsc_q=v_vsc_dq.imag,

            i_vsc_d=i_vsc_dq.real,
            i_vsc_q=i_vsc_dq.imag,
        )

        return self.emt_init
    
    def get_algebraics_step_emt_abc( self, 
                                i_vsc_a: float, i_vsc_b: float, i_vsc_c: float, 
                                v_cap_a: float, v_cap_b: float, v_cap_c: float, 
                                i_pcc_a: float, i_pcc_b: float, i_pcc_c: float, 
                                ) -> list:
        """
        Compute the abc voltage across the shunt of the LCL filter given the abc currents and capacitor voltages
        for the next time step in the EMT simulation in abc frame.
        Inputs:
        - i_vsc_a, i_vsc_b, i_vsc_c: currents from the VSC in abc frame [pu]. They are states of the LCL filter model.
        - v_cap_a, v_cap_b, v_cap_c: voltages across the capacitor in abc frame [pu]. They are states of the LCL filter model.
        - i_pcc_a, i_pcc_b, i_pcc_c: currents at the PCC in abc frame [pu]. They are states of the LCL filter model.
        Outputs:
        - v_sh_a, v_sh_b, v_sh_c: voltages across the shunt of the LCL filter in abc frame [pu]. They are algebraic variables of the LCL filter model.
        """

        # Compute voltage across the shunt of the LCL filter
        v_sh_a = self.rsh_pu * (i_vsc_a - i_pcc_a) + v_cap_a
        v_sh_b = self.rsh_pu * (i_vsc_b - i_pcc_b) + v_cap_b
        v_sh_c = self.rsh_pu * (i_vsc_c - i_pcc_c) + v_cap_c

        return [v_sh_a, v_sh_b, v_sh_c]
    
    def get_derivatives_step_emt_abc( self,
                                i_vsc_a: float, i_vsc_b: float, i_vsc_c: float, 
                                v_cap_a: float, v_cap_b: float, v_cap_c: float, 
                                i_pcc_a: float, i_pcc_b: float, i_pcc_c: float, 
                                v_vsc_a: float, v_vsc_b: float, v_vsc_c: float, 
                                v_pcc_a: float, v_pcc_b: float, v_pcc_c: float, 
                                ) -> list:
        """
        Compute the derivates with respect to time of the states of the LCL filter model in abc frame.
        for the next time step in the EMT simulation in abc frame.
        Inputs:
        - i_vsc_a, i_vsc_b, i_vsc_c: currents from the VSC in abc frame [pu]. They are states of the LCL filter model.
        - v_cap_a, v_cap_b, v_cap_c: voltages across the capacitor in abc frame [pu]. They are states of the LCL filter model.
        - i_pcc_a, i_pcc_b, i_pcc_c: currents at the PCC in abc frame [pu]. They are states of the LCL filter model.
        - v_vsc_a, v_vsc_b, v_vsc_c: voltages at the VSC in abc frame [pu]. They are inputs to the LCL filter model.
        - v_pcc_a, v_pcc_b, v_pcc_c: voltages at the PCC in abc frame [pu]. They are inputs to the LCL filter model.
        Outputs:
        - d_i_vsc_a, d_i_vsc_b, d_i_vsc_c: derivatives of the VSC currents in abc frame [pu/s].
        - d_v_cap_a, d_v_cap_b, d_v_cap_c: derivatives of the capacitor voltages in abc frame [pu/s].
        - d_i_pcc_a, d_i_pcc_b, d_i_pcc_c: derivatives of the PCC currents in abc frame [pu/s].
        """
    
        # Get parameters
        rf1 = self.rf1_pu
        xf1 = self.xf1_pu
        rf2 = self.rf2_pu
        xf2 = self.xf2_pu
        csh = self.csh_pu
        wnom = self.wbase

        # Compute voltage across the shunt of the LCL filter
        v_sh_a, v_sh_b, v_sh_c = self.get_algebraic_step_emt_abc( i_vsc_a, i_vsc_b, i_vsc_c, 
                                                                v_cap_a, v_cap_b, v_cap_c, 
                                                                i_pcc_a, i_pcc_b, i_pcc_c)

        # Voltage drop across the first branch of the LCL filter
        d_i_vsc_a = wnom / xf1 * (v_vsc_a - v_sh_a - rf1 * i_vsc_a)
        d_i_vsc_b = wnom / xf1 * (v_vsc_b - v_sh_b - rf1 * i_vsc_b)
        d_i_vsc_c = wnom / xf1 * (v_vsc_c - v_sh_c - rf1 * i_vsc_c)

        # Voltage drop across the capacitor of the LCL filter
        d_v_cap_a = wnom / csh * (i_vsc_a - i_pcc_a)
        d_v_cap_b = wnom / csh * (i_vsc_b - i_pcc_b)
        d_v_cap_c = wnom / csh * (i_vsc_c - i_pcc_c)

        # Voltage drop across the second branch of the LCL filter
        d_i_pcc_a = wnom / xf2 * (v_sh_a - v_pcc_a - rf2 * i_pcc_a)
        d_i_pcc_b = wnom / xf2 * (v_sh_b - v_pcc_b - rf2 * i_pcc_b)
        d_i_pcc_c = wnom / xf2 * (v_sh_c - v_pcc_c - rf2 * i_pcc_c)

        return [d_i_vsc_a, d_i_vsc_b, d_i_vsc_c, 
                d_v_cap_a, d_v_cap_b, d_v_cap_c,
                d_i_pcc_a, d_i_pcc_b, d_i_pcc_c]
