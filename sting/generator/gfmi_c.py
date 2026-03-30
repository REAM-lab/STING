"""
This module contains the GFMI generator that includes:
- Virtual inertia control
- Droop control for reactive power
- LCL filter
- Voltage magnitude controller

There is no DC-side dynamics modeled for the GFMI.
"""
# ----------------------
# Import python packages
# ----------------------
import numpy as np
from typing import NamedTuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import polars as pl
# ------------------
# Import sting code
# ------------------
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables
from sting.generator.core import Generator
from sting.utils.transformations import dq02abc, abc2dq0
from sting.modules.simulation_emt.utils import VariablesEMT

# -----------
# Sub-classes
# -----------
class InitialConditionsEMT(NamedTuple):
    vmag_bus: float
    vphase_bus: float
    p_bus: float
    q_bus: float
    p_ref: float
    q_ref: float
    v_ref: float
    angle_ref: float
    v_vsc_d: float
    i_vsc_d: float
    i_vsc_q: float
    i_bus_d: float
    i_bus_q: float
    v_lcl_sh_d: float
    v_lcl_sh_q: float
    i_bus_D: float
    i_bus_Q: float
    v_bus_D: float
    v_bus_Q: float
    v_vsc_mag: float
    v_vsc_DQ_phase: float

# -----------
# Main class
# -----------
@dataclass(slots=True, kw_only=True, eq=False)
class GFMIc(Generator):
    rf1_pu: float
    xf1_pu: float
    rsh_pu: float
    csh_pu: float
    txr_power_MVA: float
    txr_voltage1_kV: float
    txr_voltage2_kV: float
    txr_r1_pu: float
    txr_x1_pu: float
    txr_r2_pu: float
    txr_x2_pu: float
    h_s: float
    kd_pu: float
    droop_q_pu: float
    tau_pc_s: float
    kp_vc_pu: float
    ki_vc_puHz: float
    v_dc_pu: float
    emt_init: Optional[InitialConditionsEMT] = None

    @property
    def rf2_pu(self):
        return (self.txr_r1_pu + self.txr_r2_pu) * self.base_power_MVA / self.txr_power_MVA

    @property
    def xf2_pu(self):
        return (self.txr_x1_pu + self.txr_x2_pu) * self.base_power_MVA / self.txr_power_MVA
    
    @property
    def wbase(self):
        return 2 * np.pi * self.base_frequency_Hz
    
    def _calculate_emt_initial_conditions(self):
        vmag_bus = self.power_flow_variables.vmag_bus
        vphase_bus = self.power_flow_variables.vphase_bus
        p_bus = self.power_flow_variables.p_bus
        q_bus = self.power_flow_variables.q_bus

        # Voltage in the end of the LCL filter
        v_bus_DQ = vmag_bus * np.exp(vphase_bus * np.pi / 180 * 1j)

        # Current sent from the end of the LCL filter
        i_bus_DQ = (p_bus - q_bus * 1j) / np.conjugate(v_bus_DQ)

        # Voltage across the shunt element in the LCL filter
        v_lcl_sh_DQ = v_bus_DQ + (self.rf2_pu + self.xf2_pu * 1j) * i_bus_DQ

        # Voltage and power references
        v_ref = abs(v_lcl_sh_DQ)
        s_ref = v_lcl_sh_DQ * np.conjugate(i_bus_DQ)
        p_ref = s_ref.real
        q_ref = s_ref.imag

        # Current flowing through shunt element of LCL filter
        i_lcl_sh_DQ = v_lcl_sh_DQ * (self.csh_pu * 1j) + v_lcl_sh_DQ / self.rsh_pu

        # Current sent from the beginning of the LCL filter
        i_vsc_DQ = i_bus_DQ + i_lcl_sh_DQ
        v_vsc_DQ = v_lcl_sh_DQ + (self.rf1_pu + self.xf1_pu * 1j) * i_vsc_DQ
        
        # Angle reference
        angle_ref = np.angle(v_vsc_DQ, deg=True)

        # We refer the voltage and currents to the synchronous frames of the
        # inverter
        v_vsc_dq = v_vsc_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)
        i_vsc_dq = i_vsc_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        v_bus_dq = v_bus_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)
        i_bus_dq = i_bus_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        v_lcl_sh_dq = v_lcl_sh_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        self.emt_init = InitialConditionsEMT(
            vmag_bus=vmag_bus,
            vphase_bus=vphase_bus,
            p_bus=p_bus,
            q_bus=q_bus,
            p_ref=p_ref,
            q_ref=q_ref,
            v_ref=v_ref,
            angle_ref=angle_ref,
            v_vsc_d=v_vsc_dq.real,
            i_vsc_d=i_vsc_dq.real,
            i_vsc_q=i_vsc_dq.imag,
            i_bus_d=i_bus_dq.real,
            i_bus_q=i_bus_dq.imag,
            v_lcl_sh_d=v_lcl_sh_dq.real,
            v_lcl_sh_q=v_lcl_sh_dq.imag,
            i_bus_D=i_bus_DQ.real,
            i_bus_Q=i_bus_DQ.imag,
            v_bus_D=v_bus_DQ.real,
            v_bus_Q=v_bus_DQ.imag,
            v_vsc_mag = abs(v_vsc_DQ),
            v_vsc_DQ_phase = np.angle(v_vsc_DQ, deg=True)
        )

    def _build_small_signal_model(self):
        
        # Power controller (Virtual inertia and droop control for reactive power)
        tau_pc = self.tau_pc_s
        wb = self.wbase
        h = self.h_s
        kd = self.kd_pu
        droop_q = self.droop_q_pu
        i_bus_d, i_bus_q = self.emt_init.i_bus_d, self.emt_init.i_bus_q
        v_lcl_sh_d, v_lcl_sh_q = self.emt_init.v_lcl_sh_d, self.emt_init.v_lcl_sh_q
        p_ref, q_ref = self.emt_init.p_ref, self.emt_init.q_ref
        
        pc_controller = StateSpaceModel( 
                                        A = np.array([  [0, wb,           0,          0],
                                                        [0, -kd/(2*h),    -1/(2*h),   0],
                                                        [0, 0,            -1/tau_pc,  0],
                                                        [0, 0,            0,          -1/tau_pc]]),
                                        B = np.vstack(( [0, 0, 0, 0, 0,       0, 0],
                                                        [0, 0, 0, 0, 1/(2*h), 0, 0],
                                                        1/tau_pc*np.array([i_bus_d, i_bus_q, v_lcl_sh_d, v_lcl_sh_q,   0, 0, 0]),
                                                        1/tau_pc*np.array([-i_bus_q, i_bus_d, v_lcl_sh_q, -v_lcl_sh_d, 0, 0, 0]))),
                                        C = np.array([  [1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 0, -droop_q]]),
                                        D = np.vstack(( np.zeros((2,7)),
                                                        np.hstack((np.zeros((5, )), [droop_q, 1])))),
                                        x = DynamicalVariables(name=['angle_pc', 'w_pc', 'p_pc', 'q_pc'],
                                                               init = [0, 0, p_ref, q_ref]),
                                        u = DynamicalVariables(name=['v_lcl_sh_d', 'v_lcl_sh_q', 'i_bus_d', 'i_bus_q', 'p_ref', 'q_ref', 'v_ref']),
                                        y = DynamicalVariables(name=['phi_pc', 'w_pc', 'v_lcl_sh_ref'])
                                        )


        # Voltage magnitude controller
        kp_vc, ki_vc = self.kp_vc_pu, self.ki_vc_puHz
        v_vsc_d = self.emt_init.v_vsc_d
        
        voltage_mag_controller = StateSpaceModel(  
                                                A = np.array([ [0] ]),
                                                B = ki_vc*np.array([[1, -1]]),
                                                C = np.array([[1], [0]]),
                                                D = kp_vc*np.array([[1, -1],
                                                                    [0, 0 ]]),
                                                x = DynamicalVariables(name = ['pi_vc'],
                                                                       init = [v_vsc_d]),
                                                u = DynamicalVariables(name=['v_sh_mag_ref', 'v_sh_mag']),
                                                y =  DynamicalVariables(name=['v_vsc_d', 'v_vsc_q'])
                                                )


        # LCL filter
        rf1, xf1, rf2, xf2, rsh, csh = self.rf1_pu, self.xf1_pu, self.rf2_pu, self.xf2_pu, self.rsh_pu, self.csh_pu
        wb = self.wbase
        i_vsc_d, i_vsc_q = self.emt_init.i_vsc_d, self.emt_init.i_vsc_q
        i_bus_d, i_bus_q = self.emt_init.i_bus_d, self.emt_init.i_bus_q
        v_lcl_sh_d, v_lcl_sh_q = self.emt_init.v_lcl_sh_d, self.emt_init.v_lcl_sh_q

        lcl_filter = StateSpaceModel(
                        A = wb*np.array([[-rf1/xf1  ,   1       ,  0        ,   0       ,       -1/xf1      ,  0],
                                         [-1        ,   -rf1/xf1,  0        ,   0       ,       0           ,  -1/xf1],
                                         [0         ,   0       ,  -rf2/xf2 ,   1       ,       1/xf2       ,  0],
                                         [0         ,   0       ,  -1       ,   -rf2/xf2,       0           ,  1/xf2],
                                         [1/csh     ,   0       ,  -1/csh   ,   0       ,       -1/(rsh*csh),  1],
                                         [0         ,   1/csh   ,  0        ,   -1/csh  ,       -1          ,  -1/(rsh*csh)]]),
                        B = wb*np.array([[1/xf1 ,    0      ,   0       ,   0      ,      i_vsc_q],
                                         [0     ,    1/xf1  ,   0       ,   0      ,      -i_vsc_d],
                                         [0     ,    0      ,   -1/xf2  ,   0      ,      i_bus_q],
                                         [0     ,    0      ,   0       ,   -1/xf2 ,      -i_bus_d],
                                         [0     ,    0      ,   0       ,   0      ,      v_lcl_sh_q],
                                         [0     ,    0      ,   0       ,   0      ,      -v_lcl_sh_d]]),
                        C = np.eye(6),
                        D = np.zeros((6,5)),
                        x = DynamicalVariables(name=["i_vsc_d", "i_vsc_q", "i_bus_d", "i_bus_q", "v_lcl_sh_d", "v_lcl_sh_q"],
                                               init=[i_vsc_d, i_vsc_q, i_bus_d, i_bus_q, v_lcl_sh_d, v_lcl_sh_q]),
                        u = DynamicalVariables(name=['v_vsc_d', 'v_vsc_q', 'v_bus_d', 'v_bus_q', 'w']),
                        y = DynamicalVariables(name=["i_vsc_d", "i_vsc_q", "i_bus_d", "i_bus_q", "v_lcl_sh_d", "v_lcl_sh_q"]))
        
        # Construccion of CCM matrices
        v_ref = self.emt_init.v_ref
        angle_ref = self.emt_init.angle_ref
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        sinphi = np.sin(angle_ref*np.pi/180)
        cosphi = np.cos(angle_ref*np.pi/180)
        a = v_lcl_sh_d/v_ref
        b = v_lcl_sh_q/v_ref
        c = -sinphi*v_bus_D+cosphi*v_bus_Q
        d = -cosphi*v_bus_D-sinphi*v_bus_Q
        e = -sinphi*i_bus_d- cosphi*i_bus_q
        f = cosphi*i_bus_d - sinphi*i_bus_q
        
        Fccm = np.vstack((  np.hstack((np.zeros((2,9)), np.eye(2))), # v_lcl_sh_dq
                            np.hstack((np.zeros((2,7)), np.eye(2), np.zeros((2,2)))), # i_bus_dq
                            np.zeros((3,11)), # p_ref, q_ref, v_ref
                            np.hstack( ( [0, 0, 1], np.zeros((8, )) )), # v_lcl_sh_ref
                            np.hstack( ( np.zeros((9, )),  [a, b])), # v_lcl_sh_mag
                            np.hstack( ( np.zeros((2,3)), np.eye(2), np.zeros((2,6))) ), # v_vsc_dq
                            np.hstack( ( [c], np.zeros((10, )) )), # v_bus_d
                            np.hstack( ( [d], np.zeros((10, )) )), # v_bus_q
                            np.hstack( ( [0, 1], np.zeros((9, )))) # w
                            ))
        Gccm = np.vstack(( np.zeros((4,5)) ,
                           np.hstack( (np.eye(3), np.zeros((3,2)))),
                           np.zeros((4,5)),
                           np.hstack( (np.zeros((3,)), [cosphi, sinphi])),
                           np.hstack( (np.zeros((3,)), [-sinphi, cosphi])),
                           np.zeros((5, ))))
        Hccm = np.vstack(( np.hstack(( [e], np.zeros((6,)), [cosphi, -sinphi], [0, 0] )), 
                           np.hstack(( [f], np.zeros((6,)), [sinphi, cosphi], [0, 0] )) ))
        Lccm = np.zeros((2,5))
    
        components = [pc_controller, voltage_mag_controller, lcl_filter]
        connections = [Fccm, Gccm, Hccm, Lccm]

        # Inputs and outputs
        u = DynamicalVariables(
                                    name=["p_ref", "q_ref", "v_ref", "v_bus_D", "v_bus_Q"],
                                    type=["device", "device", "device", "grid", "grid"],
                                    init=[p_ref, q_ref, v_ref, v_bus_D, v_bus_Q]
                                    )

        i_bus_D, i_bus_Q = self.emt_init.i_bus_D, self.emt_init.i_bus_Q
        y = DynamicalVariables(
                                    name=["i_bus_D", "i_bus_Q"],
                                    init=[i_bus_D, i_bus_Q]
                                    )

        # Generate small-signal model
        self.ssm = StateSpaceModel.from_interconnected(components, connections, u, y, component_label=f"{self.type_}_{self.id}")

    def define_variables_emt(self):
        # States 
        # ------ 
        
        # Initial conditions 
        angle_ref = self.emt_init.v_vsc_DQ_phase
        p_ref, q_ref, v_ref = self.emt_init.p_ref, self.emt_init.q_ref, self.emt_init.v_ref
        
        # these quantities are already in the converter ref frame (defined by angle_ref)  
        i_bus_d, i_bus_q = self.emt_init.i_bus_d, self.emt_init.i_bus_q 
        i_vsc_d, i_vsc_q = self.emt_init.i_vsc_d, self.emt_init.i_vsc_q  
        v_sh_d, v_sh_q = self.emt_init.v_lcl_sh_d, self.emt_init.v_lcl_sh_q  
        
        # convert to abc 
        i_bus_a, i_bus_b, i_bus_c = dq02abc(i_bus_d, i_bus_q, 0, angle_ref*np.pi/180)
        i_vsc_a, i_vsc_b, i_vsc_c = dq02abc(i_vsc_d, i_vsc_q, 0, angle_ref*np.pi/180)
        v_sh_a, v_sh_b, v_sh_c = dq02abc(v_sh_d, v_sh_q, 0, angle_ref*np.pi/180)

        v_vsc_d = self.emt_init.v_vsc_d

        x = DynamicalVariables(
            name = ['angle_pc', 'w_pc', 'p_pc', 'q_pc', 'gamma',"i_vsc_a", "i_vsc_b","i_vsc_c", "v_sh_a", "v_sh_b","v_sh_c", "i_bus_a", "i_bus_b", "i_bus_c"],
            component = f"{self.type_}_{self.id}",
            init=[angle_ref*np.pi/180, 1.0, p_ref, q_ref, v_vsc_d, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c]
        )

        # Inputs 
        # ------
        
        # Initial conditions 
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        v_bus_a, v_bus_b, v_bus_c = dq02abc(v_bus_D, v_bus_Q, 0, 0)

        u = DynamicalVariables(
                            name=["p_ref", "q_ref", "v_ref", "v_bus_a", "v_bus_b", "v_bus_c"],
                            component=f"{self.type_}_{self.id}",
                            type=["device", "device", "device", "grid", "grid", "grid"],
                            init=[p_ref, q_ref, v_ref, v_bus_a, v_bus_b, v_bus_c]
                            )

        # Outputs
        # -------

        y = DynamicalVariables(
                                    name=["i_bus_a", "i_bus_b", "i_bus_c"],
                                    component=f"{self.type_}_{self.id}",
                                    init=[i_bus_a, i_bus_b, i_bus_c]
                                    )
        
        self.variables_emt = VariablesEMT(x=x,u=u,y=y)

    def get_derivative_state_emt(self):
        """
        It returns a vector with the differential equations that describe the dynamics of the GFMI-VSM.
        This model includes: power controller, voltage controller, and LCL filter.
        """    
        # Get state values 
        angle_pc, w_pc, p_pc, q_pc, gamma, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value 
        
        # Get input values (external inputs)
        p_ref, q_ref, v_ref, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value

        # convert relevant quantities to dq 
        v_sh_d, v_sh_q, _ = abc2dq0(v_sh_a, v_sh_b, v_sh_c, angle_pc) # for power controller 
        i_bus_d, i_bus_q, _ = abc2dq0(i_bus_a, i_bus_b, i_bus_c, angle_pc) # for power controller 

        # Calculate DC-side current using current vsc voltage and current  
        # need to calculate v_vsc_dq, because it is an algebraic state so is not available in our state vector 
        i_vsc_d, i_vsc_q, _ = abc2dq0(i_vsc_a, i_vsc_b, i_vsc_c, angle_pc)
        i_vsc_dq = i_vsc_d + 1j*i_vsc_q
        v_sh_dq = v_sh_d + 1j*v_sh_q 
        v_vsc_dq = v_sh_dq + (self.rf1_pu + self.xf1_pu * 1j) * i_vsc_dq
        
        # Do Q-V droop 
        v_sh_mag = (v_sh_d**2+v_sh_q**2)**0.5 # current voltage mag 
        v_sh_mag_ref = v_ref - self.droop_q_pu*(q_pc - q_ref) # droop on error from ref 
        
        # NB updating algebraic states!
        v_vsc_d = gamma + self.kp_vc_pu*(v_sh_mag_ref - v_sh_mag) # update 
        v_vsc_q = 0.0 # update 

        # convert to abc to feed into filter dynamics 
        v_vsc_a, v_vsc_b, v_vsc_c = dq02abc(v_vsc_d, v_vsc_q, 0, angle_pc) # correct to use this angle?

        def power_controller_dynamics(y, internal_inputs):
            """
            It returns the differential equations that describe the dynamics of the power controller.
            The power controller has: virtual inertia, filter for active and reactive power.
            """

            # Definition of states for the ODEs of the power controller
            angle_pc, w_pc, p_pc, q_pc  = y[0], y[1], y[2], y[3]  

            # Extract the list of parameters
            tau_pc = self.tau_pc_s  # proportional gain of active power controller
            droop_q = self.droop_q_pu  # integral gain of active power controller
            kd = self.kd_pu  # proportional gain of reactive power controller
            h = self.h_s  # integral gain of reactive power controller
            wb = self.wbase  # nominal frequency of the system

            # Define measured active and reactive power at timepoint "t"
            v_sh_d, v_sh_q, i_bus_d, i_bus_q, p_ref = internal_inputs

            # Define ODEs that describe the dynamics of the power controller
            d_angle_pc = wb * w_pc
            d_w_pc = 1/(2*h) * (p_ref - p_pc - kd * (w_pc - 1))
            d_p_pc = 1/tau_pc * (- p_pc + v_sh_d * i_bus_d + v_sh_q * i_bus_q)
            d_q_pc = 1/tau_pc * (- q_pc - v_sh_d * i_bus_q + v_sh_q * i_bus_d)
            
            return [d_angle_pc, d_w_pc, d_p_pc, d_q_pc]

        def voltage_controller_dynamics(y, internal_inputs):
            """
            It returns the differential equations that describe the dynamics of the voltage controller.
            The voltage controller has a PI structure that regulates the magnitude of the voltage at the
            middle of the LCL filter. 
            """    
            # Definition of states for the ODEs of the voltage controller
            gamma = y[0]

            # Extract the list of parameters
            kp_vc = self.kp_vc_pu  # proportional gain of voltage controller
            ki_vc = self.ki_vc_puHz  # integral gain of voltage controller

            # Define measured capacitor voltages at timepoint "t"
            v_sh_mag_ref, v_sh_d, v_sh_q = internal_inputs

            # Define ODEs that describe the dynamics of the voltage controller
            d_gamma = ki_vc * v_sh_mag_ref - ki_vc * (v_sh_d**2 + v_sh_q**2)**0.5

            return d_gamma

        def lcl_filter_dynamics(y, internal_inputs):
            """
            It returns the differential equations that describe the dynamics of the LCL filter.
            The LCL filter connects the VSC to the grid. It has three branches: the first branch (RL) connects
            the VSC to the shunt element, the second branch is the shunt element (RC), and the third branch (RL)
            connects the shunt element to the grid.
            """

            # Definition of states for the ODEs of the LCL filter
            i_vsc_a , i_vsc_b, i_vsc_c = y[0], y[1], y[2] # currents flowing out of VSC
            v_sh_a, v_sh_b, v_sh_c = y[3], y[4], y[5] # currents flowing through paralell RC shunt
            i_bus_a, i_bus_b, i_bus_c = y[6], y[7], y[8] # currents flowing to bus

            # Extract the list of parameters
            r1 = self.rf1_pu # resistance [p.u.] of first branch of filter
            l1 = self.xf1_pu # inductance [p.u.] of first branch of filter
            r2 = self.rf2_pu # resistance [p.u.] of second branch of filter
            l2 = self.xf2_pu # inductance [p.u.] of second branch of filter
            rsh = self.rsh_pu # resistance [p.u.] of series RC shunt
            csh = self.csh_pu # capacitance [p.u.] of series RC shunt
            wb = self.wbase # nominal frequency of the system

            # Define voltage at vsc at timepoint "t"
            v_vsc_a, v_vsc_b, v_vsc_c, v_bus_a, v_bus_b, v_bus_c = internal_inputs

            # Define ODEs that describe the dynamics of the LCL filter
            di_vsc_a = wb/l1 *(v_vsc_a - v_sh_a - r1 * i_vsc_a)
            di_vsc_b = wb/l1 *(v_vsc_b - v_sh_b - r1 * i_vsc_b)
            di_vsc_c = wb/l1 *(v_vsc_c - v_sh_c - r1 * i_vsc_c)

            dv_sh_a = wb/csh * (-v_sh_a/rsh + i_vsc_a - i_bus_a)
            dv_sh_b = wb/csh * (-v_sh_b/rsh + i_vsc_b - i_bus_b)
            dv_sh_c = wb/csh * (-v_sh_c/rsh + i_vsc_c - i_bus_c)

            di_bus_a = wb/l2 *(v_sh_a - v_bus_a - r2 * i_bus_a)
            di_bus_b = wb/l2 *(v_sh_b - v_bus_b - r2 * i_bus_b)
            di_bus_c = wb/l2 *(v_sh_c - v_bus_c - r2 * i_bus_c)

            return [di_vsc_a, di_vsc_b, di_vsc_c, dv_sh_a, dv_sh_b, dv_sh_c, di_bus_a, di_bus_b, di_bus_c]
          
        d_angle_pc, d_w_pc, d_p_pc, d_q_pc = power_controller_dynamics([angle_pc, w_pc, p_pc, q_pc], [v_sh_d, v_sh_q, i_bus_d, i_bus_q, p_ref])

        d_gamma = voltage_controller_dynamics([gamma], [v_sh_mag_ref, v_sh_d, v_sh_q])
            
        di_vsc_a, di_vsc_b, di_vsc_c, dv_sh_a, dv_sh_b, dv_sh_c, di_bus_a, di_bus_b, di_bus_c = lcl_filter_dynamics([i_vsc_a , i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c], [v_vsc_a, v_vsc_b, v_vsc_c, v_bus_a, v_bus_b, v_bus_c])

        return [d_angle_pc, d_w_pc, d_p_pc, d_q_pc, d_gamma, di_vsc_a, di_vsc_b, di_vsc_c, dv_sh_a, dv_sh_b, dv_sh_c, di_bus_a, di_bus_b, di_bus_c]
    
    def get_output_emt(self):
        
        angle_pc, w_pc, p_pc, q_pc, gamma, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value

        return [i_bus_a, i_bus_b, i_bus_c]

    def plot_results_emt(self, output_dir):
        """
        Plot EMT simulation results
        """

        angle_pc, w_pc, p_pc, q_pc, gamma, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value 
        
        tps = self.variables_emt.x.time
        p_ref, q_ref, v_ref, v_bus_a, v_bus_b, v_bus_c, = self.variables_emt.u.value
        
        # Transform abc to dq0
        i_vsc_d, i_vsc_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_vsc_a, i_vsc_b, i_vsc_c, angle_pc)])
        v_sh_d, v_sh_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(v_sh_a, v_sh_b, v_sh_c, angle_pc)])
        i_bus_d, i_bus_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_bus_a, i_bus_b, i_bus_c, angle_pc)])
        
        fig = make_subplots(
            rows=7, cols=2,
            horizontal_spacing=0.15,
            vertical_spacing=0.05,
        )

        fig.add_trace(go.Scatter(x=tps, y=w_pc, mode='lines', line=dict(color='red', dash='solid')),
                    row=1, col=1)
        fig.update_xaxes(title_text='Time [s]', row=1, col=1)
        fig.update_yaxes(title_text='Frequency pc [p.u.]', row=1, col=1)

        fig.add_trace(go.Scatter(x=tps, y=angle_pc * 180 / np.pi, mode='lines', line=dict(color='red', dash='solid')),
                    row=1, col=2)
        fig.update_xaxes(title_text='Time [s]', row=1, col=2)
        fig.update_yaxes(title_text='Angle pc [deg]', row=1, col=2)

        fig.add_trace(go.Scatter(x=tps, y=p_pc, name="p_pc", mode='lines', line=dict(color='red', dash='solid')),
                    row=2, col=1)
        fig.update_xaxes(title_text='Time [s]', row=2, col=1)
        fig.update_yaxes(title_text='Active Power pc [p.u.]', row=2, col=1)

        fig.add_trace(go.Scatter(x=tps, y=q_pc, mode='lines', line=dict(color='red', dash='solid')),
                    row=2, col=2)
        fig.update_xaxes(title_text='Time [s]', row=2, col=2)
        fig.update_yaxes(title_text='Reactive Power pc [p.u.]', row=2, col=2)

        fig.add_trace(go.Scatter(x=tps, y=gamma, mode='lines', line=dict(color='red', dash='solid')),
                    row=3, col=1)
        fig.update_xaxes(title_text='Time [s]', row=3, col=1)
        fig.update_yaxes(title_text='Gamma [p.u.]', row=3, col=1)   

        fig.add_trace(go.Scatter(x=tps, y=i_vsc_d, mode='lines', line=dict(color='red', dash='solid')),
                    row=4, col=1)
        fig.update_xaxes(title_text='Time [s]', row=4, col=1)
        fig.update_yaxes(title_text='i_vsc_d [p.u.]', row=4, col=1) 

        fig.add_trace(go.Scatter(x=tps, y=i_vsc_q, mode='lines', line=dict(color='red', dash='solid')),
                    row=4, col=2)
        fig.update_xaxes(title_text='Time [s]', row=4, col=2)
        fig.update_yaxes(title_text='i_vsc_q [p.u.]', row=4, col=2)

        fig.add_trace(go.Scatter(x=tps, y=v_sh_d, mode='lines', line=dict(color='red', dash='solid')),
                    row=5, col=1)
        fig.update_xaxes(title_text='Time [s]', row=5, col=1)
        fig.update_yaxes(title_text='v_sh_d [p.u.]', row=5, col=1)

        fig.add_trace(go.Scatter(x=tps, y=v_sh_q, mode='lines', line=dict(color='red', dash='solid')),
                    row=5, col=2)
        fig.update_xaxes(title_text='Time [s]', row=5, col=2)
        fig.update_yaxes(title_text='v_sh_q [p.u.]', row=5, col=2)

        fig.add_trace(go.Scatter(x=tps, y=i_bus_d, mode='lines', line=dict(color='red', dash='solid')),
                    row=6, col=1)
        fig.update_xaxes(title_text='Time [s]', row=6, col=1)
        fig.update_yaxes(title_text='i_bus_d [p.u.]', row=6, col=1)

        fig.add_trace(go.Scatter(x=tps, y=i_bus_q, mode='lines', line=dict(color='red', dash='solid')),
                    row=6, col=2)
        fig.update_xaxes(title_text='Time [s]', row=6, col=2)
        fig.update_yaxes(title_text='i_bus_q [p.u.]', row=6, col=2)

        name = f"{self.type_}_{self.id}"
        fig.update_layout(  title_text = name,
                            title_x=0.5,
                            showlegend = False,
                            )

        fig.update_layout(height=1200*2, 
                        width=800*2, 
                        showlegend=False,
                        margin={'t': 0, 'l': 0, 'b': 0, 'r': 0})
        
        fig.write_html(os.path.join(output_dir, name + ".html"))


    def compare_ssm_emt(self, emt_directory, ssm_directory):
        """
        TODO: "angle_pc" and "w_pc" are not compared as it.
        """
         # Read the SSM and EMT states
        emt = pl.read_csv(os.path.join(emt_directory, f"{self.type_}_{self.id}_states.csv"))
        ssm = pl.read_csv(os.path.join(ssm_directory, f"{self.type_}_{self.id}_states.csv"))

        # Create mapping of errors using columns/states that can be compared directly
        delta = {
            f"({self.type_}_{self.id}, {col})": (emt[col].to_numpy(), ssm[col].to_numpy())
            for col in ["p_pc","q_pc"]
        }
        # Name of EMT (abc) variables to compare with SSM (dq) variables
        outputs = [
            ("i_vsc_a", "i_vsc_b", "i_vsc_c","i_vsc_d","i_vsc_q"),
            ("v_sh_a","v_sh_b","v_sh_c","v_lcl_sh_d","v_lcl_sh_q"),
            ("i_bus_a","i_bus_b","i_bus_c","i_bus_d","i_bus_q")
            ]
        
        for (a,b,c,d,q) in outputs:
            # Transform EMT abc states to dq0 states
            a, b, c, angle_ref = [c.to_numpy() for c in emt.select(a,b,c,"angle_pc")]
            d_emt, q_emt, _ = zip(*map(abc2dq0, a, b, c, angle_ref))
            # Unpack the SSM dq states
            d_ssm, q_ssm = [c.to_numpy() for c in ssm.select(d,q)]
            # Append deltas
            delta[f"({self.type_}_{self.id}, {d})"] = (d_emt, d_ssm)
            delta[f"({self.type_}_{self.id}, {q})"] = (q_emt, q_ssm)
            
        return delta