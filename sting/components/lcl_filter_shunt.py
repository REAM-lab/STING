import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple

from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables

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
    # Shunt voltage
    v_sh_d: float
    v_sh_q: float
    # Converter-side voltage and current
    v_vsc_d: float
    v_vsc_q: float
    i_vsc_d: float
    i_vsc_q: float


@dataclass(slots=True)
class LCLFilterShunt:
    """
    The LCL filter connects the VSC to the grid. It has three branches: the first branch (RL) connects
    the VSC to the shunt element, the second branch is the shunt element (RC), and the third branch (RL)
    connects the shunt element to the grid.

    Parameters
    rf1_pu: resistance [pu] of first branch of filter
    xf1_pu: inductance [pu] of first branch of filter
    rf2_pu: resistance [pu] of second branch of filter
    xf2_pu: inductance [pu] of second branch of filter
    rsh_pu: resistance [pu] of series RC shunt
    csh_pu: capacitance [pu] of series RC shunt
    wbase: nominal frequency of the system
    """
    rf1_pu: float
    xf1_pu: float
    rsh_pu: float
    csh_pu: float
    rf2_pu: float
    xf2_pu: float
    wbase: float

    emt_init: InitialConditionsEMT = field(init=False)

    def get_steady_state(self, v_bus_mag, relative_phase_deg, p_bus, q_bus):
        # Convert degrees to radians
        phase_rad = relative_phase_deg * np.pi / 180

        # Voltage in the end of the LCL filter
        v_bus_DQ = v_bus_mag * np.exp(phase_rad * 1j)
        # Current sent from the end of the LCL filter
        i_bus_DQ = (p_bus - q_bus * 1j) / np.conjugate(v_bus_DQ)
        # Voltage across the shunt element in the LCL filter
        v_lcl_sh_DQ = v_bus_DQ + (self.rf2_pu + self.xf2_pu * 1j) * i_bus_DQ
        # Current flowing through shunt element of LCL filter
        i_lcl_sh_DQ = v_lcl_sh_DQ * (self.csh_pu * 1j) + v_lcl_sh_DQ / self.rsh_pu
        # Current sent from the beginning of the LCL filter
        i_vsc_DQ = i_bus_DQ + i_lcl_sh_DQ
        v_vsc_DQ = v_lcl_sh_DQ + (self.rf1_pu + self.xf1_pu * 1j) * i_vsc_DQ

        # We refer the voltage and currents to the synchronous frames of the inverter
        v_vsc_dq = v_vsc_DQ * np.exp(-phase_rad * 1j)
        i_vsc_dq = i_vsc_DQ * np.exp(-phase_rad * 1j)
        v_bus_dq = v_bus_DQ * np.exp(-phase_rad * 1j)
        i_bus_dq = i_bus_DQ * np.exp(-phase_rad * 1j)
        v_sh_dq = v_lcl_sh_DQ * np.exp(-phase_rad * 1j)

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
            # Shunt
            v_sh_d=v_sh_dq.real,
            v_sh_q=v_sh_dq.imag,

            # Converter
            v_vsc_d=v_vsc_dq.real,
            v_vsc_q=v_vsc_dq.imag,

            i_vsc_d=i_vsc_dq.real,
            i_vsc_q=i_vsc_dq.imag,
        )

        return self.emt_init

    

    def get_small_signal_model(self, i_vsc_d, i_vsc_q, i_bus_d, i_bus_q, v_sh_d, v_sh_q):
        rf1, xf1, rf2, xf2, rsh, csh = self.rf1_pu, self.xf1_pu, self.rf2_pu, self.xf2_pu, self.rsh_pu, self.csh_pu
        wb = self.wbase


        A = wb*np.array([
            [-rf1/xf1  ,   1       ,  0        ,   0       ,       -1/xf1      ,  0          ], # id_vsc
            [-1        ,   -rf1/xf1,  0        ,   0       ,       0           ,  -1/xf1     ], # iq_vsc
            [0         ,   0       ,  -rf2/xf2 ,   1       ,       1/xf2       ,  0          ], # id_bus
            [0         ,   0       ,  -1       ,   -rf2/xf2,       0           ,  1/xf2      ], # iq_bus
            [1/csh     ,   0       ,  -1/csh   ,   0       ,       -1/(rsh*csh),  1          ], # vd_sh
            [0         ,   1/csh   ,  0        ,   -1/csh  ,       -1          ,  -1/(rsh*csh)] # vq_sh
        ])
        B = wb*np.array([
            # v_vsc_d,  v_vsc_q,    v_bus_d,   v_bus_q,        w
            [1/xf1 ,    0      ,   0       ,   0      ,      i_vsc_q ], 
            [0     ,    1/xf1  ,   0       ,   0      ,     -i_vsc_d ], 
            [0     ,    0      ,   -1/xf2  ,   0      ,      i_bus_q ], 
            [0     ,    0      ,   0       ,   -1/xf2 ,     -i_bus_d ],
            [0     ,    0      ,   0       ,   0      ,      v_sh_q  ],
            [0     ,    0      ,   0       ,   0      ,     -v_sh_d  ]
        ])

        ssm = StateSpaceModel(
            A = A,
            B = B,
            C = np.eye(6),
            D = np.zeros((6,5)),
            x = DynamicalVariables(
                name=["i_vsc_d", "i_vsc_q", "i_bus_d", "i_bus_q", "v_lcl_sh_d", "v_lcl_sh_q"],
                init=[i_vsc_d, i_vsc_q, i_bus_d, i_bus_q, v_sh_d, v_sh_q]
            ),
            u = DynamicalVariables(name=['v_vsc_d', 'v_vsc_q', 'v_bus_d', 'v_bus_q', 'w']),
            y = DynamicalVariables(name=["i_vsc_d", "i_vsc_q", "i_bus_d", "i_bus_q", "v_lcl_sh_d", "v_lcl_sh_q"]))
        
        return ssm

    def differential_step_emt_abc(
            self, 
            i_vsc_a , i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, # states
            v_vsc_a, v_vsc_b, v_vsc_c, v_bus_a, v_bus_b, v_bus_c # inputs
            ):
        """
        Returns a step of differential equations that describe the EMT dynamics
        of the LCL filter with abc inputs.
        """
        rf1, xf1, rf2, xf2, rsh, csh = self.rf1_pu, self.xf1_pu, self.rf2_pu, self.xf2_pu, self.rsh_pu, self.csh_pu
        wb = self.wbase

        # Define ODEs that describe the dynamics of the LCL filter
        di_vsc_a = wb/xf1 *(v_vsc_a - v_sh_a - rf1 * i_vsc_a)
        di_vsc_b = wb/xf1 *(v_vsc_b - v_sh_b - rf1 * i_vsc_b)
        di_vsc_c = wb/xf1 *(v_vsc_c - v_sh_c - rf1 * i_vsc_c)

        dv_sh_a = wb/csh * (-v_sh_a/rsh + i_vsc_a - i_bus_a)
        dv_sh_b = wb/csh * (-v_sh_b/rsh + i_vsc_b - i_bus_b)
        dv_sh_c = wb/csh * (-v_sh_c/rsh + i_vsc_c - i_bus_c)

        di_bus_a = wb/xf2 *(v_sh_a - v_bus_a - rf2 * i_bus_a)
        di_bus_b = wb/xf2 *(v_sh_b - v_bus_b - rf2 * i_bus_b)
        di_bus_c = wb/xf2 *(v_sh_c - v_bus_c - rf2 * i_bus_c)

        return [di_vsc_a, di_vsc_b, di_vsc_c, dv_sh_a, dv_sh_b, dv_sh_c, di_bus_a, di_bus_b, di_bus_c]