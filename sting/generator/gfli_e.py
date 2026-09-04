"""
This module implements a GFLI that incorporates: 
- LCL filter: Two Series RL branches (one branch is the transformer) and one Parallel RC shunt. 
- Outer loop DC voltage PI controller
- DC-side capacitor dynamics with resistor and current source representing a load 
- Current controller: A dq-based frame PI controller
- PLL: A basic implementation

"""
# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

# ------------------
# Import sting code
# ------------------
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables
from sting.utils.transformations import dq02abc, abc2dq0
from sting.generator.core import Generator

# -----------
# Sub-classes
# -----------
class PowerFlowVariables(NamedTuple):
    p_bus: float
    q_bus: float
    vmag_bus: float
    vphase_bus: float


class InitialConditionsEMT(NamedTuple):
    vmag_bus: float
    vphase_bus: float
    p_bus: float
    q_bus: float
    angle_ref: float
    pi_cc_d: float
    pi_cc_q: float
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
    v_vsc_d: float 
    v_vsc_q: float 
    p_vsc: float 
    v_dc: float 
    int_vdc: float 
    i_dc: float 
    i_load_ref: float 


class VariablesEMT(NamedTuple):
    x: DynamicalVariables
    u: DynamicalVariables
    y: DynamicalVariables
    
# -----------
# Main class
# -----------
@dataclass(slots=True, kw_only=True, eq=False)
class GFLIe(Generator):
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
    beta: float
    kp_pll_pu: float
    ki_pll_puHz: float
    kp_cc_pu: float
    ki_cc_puHz: float
    v_dc_ref: float # added 
    c_dc: float # added 
    kp_oc_pu: float # added 
    ki_oc_puHz: float # added 
    Tload: float # added 
    r_dc: float # added 
    x_pll_rescale: np.ndarray = field(default_factory=lambda: np.array([[100, 0], [0, 1]])) 
    name: str = field(default_factory=str)
    emt_init: Optional[InitialConditionsEMT] = None
    ssm: Optional[StateSpaceModel] = None

    @property
    def rf2_pu(self):
        return (self.txr_r1_pu + self.txr_r2_pu) * self.base_power_MVA / self.txr_power_MVA

    @property
    def xf2_pu(self):
        return (self.txr_x1_pu + self.txr_x2_pu) * self.base_power_MVA / self.txr_power_MVA

    @property
    def wbase(self):
        return 2 * np.pi * self.base_frequency_Hz
    
    def _build_small_signal_model(self):

        # Current PI controller
        kp_cc, ki_cc = self.kp_cc_pu, self.ki_cc_puHz
        pi_cc_d, pi_cc_q = self.emt_init.pi_cc_d, self.emt_init.pi_cc_q

        pi_controller = StateSpaceModel(  A = np.zeros((2,2)), 
                                          B = ki_cc*np.hstack((np.eye(2), -np.eye(2))),
                                          C = np.eye(2),
                                          D = kp_cc*np.hstack((np.eye(2), -np.eye(2))),
                                          u = DynamicalVariables(name=['i_bus_d_ref', 'i_bus_q_ref', 'i_bus_d', 'i_bus_q']), 
                                          y = DynamicalVariables(name=['e_d', 'e_q']),
                                          x = DynamicalVariables(   name=['pi_cc_d', 'pi_cc_q'],
                                                                    init= [pi_cc_d, pi_cc_q]) )

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
                        C = np.hstack((np.eye(4,4), np.zeros((4,2)))),
                        D = np.zeros((4,5)),
                        x = DynamicalVariables(name=["i_vsc_d", "i_vsc_q", "i_bus_d", "i_bus_q", "v_lcl_sh_d", "v_lcl_sh_q"],
                                               init=[i_vsc_d, i_vsc_q, i_bus_d, i_bus_q, v_lcl_sh_d, v_lcl_sh_q]),
                        u = DynamicalVariables(name=['v_vsc_d', 'v_vsc_q', 'v_bus_d', 'v_bus_q', 'w']),
                        y = DynamicalVariables(name=["i_vsc_d", "i_vsc_q", "i_bus_d", "i_bus_q"]))

        # Phase-locked loop
        kp_pll, ki_pll = self.kp_pll_pu, self.ki_pll_puHz
        beta = self.beta
        vmag_bus = self.emt_init.vmag_bus
        sinphi = np.sin(self.emt_init.angle_ref*np.pi/180)
        cosphi = np.cos(self.emt_init.angle_ref*np.pi/180)
        int_pll = 0
        phase_pll =  self.emt_init.angle_ref*np.pi/180

        pll = StateSpaceModel(  A = np.array([  [  0         ,  -vmag_bus*ki_pll],
                                                [1          , -1*vmag_bus*kp_pll]]),
                                B = np.array([  [-sinphi*ki_pll   ,        +cosphi*ki_pll],
                                                [-1*kp_pll*sinphi,  1*kp_pll*cosphi]]),
                                C = np.array([  [0  , 1],
                                                [1/wb  , -1/wb * vmag_bus*kp_pll]]),
                                D = np.array([  [0                ,           0],
                                                [-1/wb * kp_pll * sinphi ,  1/wb * kp_pll * cosphi]]),
                                u = DynamicalVariables(name=['v_bus_D', 'v_bus_Q']),
                                y = DynamicalVariables(name=['phase', 'w']),
                                x = DynamicalVariables(name=["int_pll", "phase_pll"], 
                                                       init=[int_pll, phase_pll] ) )

        # Re-scale the states so that they are not very small numbers compared to 
        # other states. It was tested in EMT simulation.
        #pll.A = self.x_pll_rescale @ pll.A @ scipy.linalg.inv(self.x_pll_rescale)
        #pll.B = self.x_pll_rescale @ pll.B
        #pll.C = pll.C @ scipy.linalg.inv(self.x_pll_rescale)
        
        # Outer control + DC capacitor dynamics
        Kp, Ki, Cdc, Tload, r_dc = self.kp_oc_pu, self.ki_oc_puHz, self.c_dc, self.Tload, self.r_dc
        outer_control = StateSpaceModel(A = np.array([[0, Ki, 0], 
                                                     [0, -1/r_dc,  -wb/Cdc],
                                                     [0, 0, -1/Tload]]),
                                        B = np.array([[-Ki, 0, 0], 
                                                      [0, 0, -wb/Cdc],
                                                      [0, 1/Tload, 0]]),
                                        C = np.array([[1, Kp, 0],
                                                      [0, 1, 0]]),
                                        D = np.array([[-Kp, 0,  0], 
                                                      [0, 0, 0]]),
                                        u = DynamicalVariables(name=["v_dc_ref", "i_load_ref", "idc"]),
                                        y = DynamicalVariables(name=["i_bus_d_ref", "v_dc"]),
                                        x = DynamicalVariables(name=["int_vdc", "v_dc", "i_load"], 
                                                               init=[self.emt_init.int_vdc, self.emt_init.v_dc, self.emt_init.i_load_ref]))

        # Construction of CCM matrices
        # ustack = F*ystack + G*u 
        
        # ustack = i2dq_ref, i2dq_c, v1dq_c, v2dq_c, w, v2dq, vdcref, iload_ref, idc (14)
        # y_stack = e1d, e1q, i1d_c, i1q_c, i2d_c, i2q_c, theta, w, i2d_ref, vdc (12) 
        # u = vdc_ref, iload_ref, i2q_ref, v2d, v2q
        # y = i2dq 
        
        # dc power balance linearization 
        v_dc = self.emt_init.v_dc 
        
        b1 = self.emt_init.v_vsc_d/v_dc # i1d
        b2 = self.emt_init.i_vsc_d/v_dc #e1d
        b3 = - (self.emt_init.i_vsc_d/v_dc)*(self.xf1_pu+self.xf2_pu) #i2q
        b4 = self.emt_init.v_vsc_q/v_dc # i1q 
        b5 = self.emt_init.i_vsc_q/v_dc # e1q
        b6 = (self.emt_init.i_vsc_q/v_dc)*(self.xf1_pu+self.xf2_pu) #i2d 
        b7 = - self.emt_init.i_dc/v_dc #vdc 
        b8 = -(self.emt_init.i_vsc_q/v_dc)*self.beta*vmag_bus # theta 

        # Construction of CCM matrices
        Fccm = np.vstack( ( np.hstack((np.zeros((8, )), 1, 0)) ,# i2d_ref
                            np.zeros((10,)), # i2q_ref 
                            np.hstack((np.zeros((2,4)), np.eye(2) ,np.zeros((2,4)))), # i2dq_c
                            [1, 0, 0, 0, 0, -(xf1+xf2), 0, 0, 0, 0], # v1d_c
                            [0, 1, 0, 0, (xf1+xf2), 0, -beta*vmag_bus, 0, 0, 0], # v1q_c
                            np.zeros((10, )) , # v2d_c
                            np.append( np.zeros((6,)) , [-vmag_bus,  0, 0, 0] ), # v2q_c
                            np.append( np.zeros((7,)) , [1, 0, 0] ), # w
                            np.zeros((2, 10)), # v2_dq
                            np.zeros((2, 10)), #vdc_ref, iload_ref 
                            [b2, b5, b1, b4, b6, b3, b8, 0, 0, b7] #idc
                         )
        ) 


        Gccm = np.vstack((      np.zeros(5,), # i2ref_d, 
                                [0, 0, 1, 0, 0], #i2q_ref
                                np.zeros((2,5)), #i2dq_c, 
                                [0, 0, 0, beta*cosphi ,    beta*sinphi],  # v1d_c
                                [0, 0, 0, -beta*sinphi,    beta*cosphi], # v1q_c
                                [0, 0, 0, cosphi   ,sinphi], # v2d_c
                                [0, 0,  0, -sinphi ,cosphi], # v2q_c
                                np.zeros((5, )), # w
                                np.hstack((np.zeros((2,3)), np.eye(2) ) ),  # v2dq 
                                [1, 0, 0, 0, 0], #vdc_ref 
                                [0, 1, 0, 0, 0], #iload_ref
                                np.zeros((1,5)) # idc 
                                ) 
                         ) 
  
        Hccm = np.vstack(( [ 0, 0 , 0, 0, cosphi , -sinphi, -sinphi*i_bus_d-cosphi*i_bus_q, 0, 0, 0],
                            [0, 0, 0, 0, sinphi , cosphi , cosphi*i_bus_d-sinphi*i_bus_q, 0, 0, 0] ))
        
        Lccm = np.zeros((2, 5))

        components = [pi_controller, lcl_filter, pll, outer_control]
        connections = [Fccm, Gccm, Hccm, Lccm]

        # Inputs and outputs
        v_bus_D, v_bus_Q= self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        u = DynamicalVariables(
                                name=["v_dc_ref", "i_load_ref", "i_bus_q_ref", "v_bus_D", "v_bus_Q"],
                                type=["device", "device", "device", "grid", "grid"],
                                init=[self.emt_init.v_dc, self.emt_init.i_load_ref, self.emt_init.i_bus_q, v_bus_D, v_bus_Q])

        i_bus_D, i_bus_Q= self.emt_init.i_bus_D, self.emt_init.i_bus_Q
        y = DynamicalVariables(
                                name=['i_bus_D', 'i_bus_Q'],
                                init=[i_bus_D, i_bus_Q])

        # Generate small-signal model
        ssm = StateSpaceModel.from_interconnected(components, connections, u, y, component_label=f"{self.type_}_{self.id}")

        self.ssm = ssm  

    def _calculate_emt_initial_conditions(self):
        
        
        # Extract power flow solution
        vmag_bus = self.power_flow_variables.vmag_bus
        vphase_bus = self.power_flow_variables.vphase_bus
        p_bus = self.power_flow_variables.p_bus
        q_bus = self.power_flow_variables.q_bus

        # Voltage in the end of the LCL filter
        v_bus_DQ = vmag_bus * np.exp(vphase_bus * np.pi / 180 * 1j)
        angle_ref = np.angle(v_bus_DQ, deg=True)

        # Current sent from the end of the LCL filter
        i_bus_DQ = (p_bus - q_bus * 1j) / np.conjugate(v_bus_DQ)

        # Voltage across the shunt element in the LCL filter
        v_lcl_sh_DQ = v_bus_DQ + (self.rf2_pu + self.xf2_pu * 1j) * i_bus_DQ

        # Current flowing through shunt element of LCL filter
        i_lcl_sh_DQ = v_lcl_sh_DQ * (self.csh_pu * 1j) + v_lcl_sh_DQ / self.rsh_pu

        # Current sent from the beginning of the LCL filter
        i_vsc_DQ = i_bus_DQ + i_lcl_sh_DQ
        v_vsc_DQ = v_lcl_sh_DQ + (self.rf1_pu + self.xf1_pu * 1j) * i_vsc_DQ

        # We refer the voltage and currents to the synchronous frames of the
        # inverter
        v_vsc_dq = v_vsc_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)
        i_vsc_dq = i_vsc_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        v_bus_dq = v_bus_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)
        i_bus_dq = i_bus_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        v_lcl_sh_dq = v_lcl_sh_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        # Initial conditions for the current controller
        pi_cc_dq = (
            v_vsc_dq - self.beta * v_bus_dq - 1j * (self.xf1_pu + self.xf2_pu) * i_bus_dq
        )
        
        # DC-side initial conditions 
        v_dc = self.v_dc_ref
        p_vsc = (v_vsc_dq*np.conjugate(i_vsc_dq)).real # power at converter terminals 
        i_dc = p_vsc/v_dc 
        i_load = -v_dc/self.r_dc - i_dc 

        self.emt_init = InitialConditionsEMT(
            vmag_bus=vmag_bus,
            vphase_bus=vphase_bus,
            p_bus=p_bus,
            q_bus=q_bus,
            angle_ref=angle_ref,
            pi_cc_d=pi_cc_dq.real,
            pi_cc_q=pi_cc_dq.imag,
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
            v_vsc_DQ_phase = np.angle(v_vsc_DQ, deg=True),
            v_dc=v_dc,
            p_vsc=p_vsc,
            i_dc=i_dc,
            int_vdc=i_bus_dq.real,
            i_load_ref=i_load,
            v_vsc_d=v_vsc_dq.real,
            v_vsc_q=v_vsc_dq.imag
        )
        
    def define_variables_emt(self):
        # States 
        # ------ 
        
        # Initial conditions 
        angle_ref = self.emt_init.angle_ref
        pi_cc_d, pi_cc_q = self.emt_init.pi_cc_d, self.emt_init.pi_cc_q
        
        # these quantities are already in the converter ref frame (defined by angle_ref)  
        i_bus_d, i_bus_q = self.emt_init.i_bus_d, self.emt_init.i_bus_q 
        i_vsc_d, i_vsc_q = self.emt_init.i_vsc_d, self.emt_init.i_vsc_q  
        v_sh_d, v_sh_q = self.emt_init.v_lcl_sh_d, self.emt_init.v_lcl_sh_q  
        
        # convert to abc 
        i_bus_a, i_bus_b, i_bus_c = dq02abc(i_bus_d, i_bus_q, 0, angle_ref*np.pi/180)
        i_vsc_a, i_vsc_b, i_vsc_c = dq02abc(i_vsc_d, i_vsc_q, 0, angle_ref*np.pi/180)
        v_sh_a, v_sh_b, v_sh_c = dq02abc(v_sh_d, v_sh_q, 0, angle_ref*np.pi/180)

        x = DynamicalVariables(
            name = ['pi_cc_d', 'pi_cc_q', 'theta_pll', 'gamma_pll', "i_vsc_a", "i_vsc_b", "i_vsc_c", "v_sh_a", "v_sh_b","v_sh_c", "i_bus_a", "i_bus_b", "i_bus_c", "int_vdc", "v_dc", "i_load"],
            component = f"{self.type_}_{self.id}",
            init = [pi_cc_d, pi_cc_q, angle_ref * np.pi/180, 0, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, self.emt_init.int_vdc, self.emt_init.v_dc, self.emt_init.i_load_ref]
        )

        # Inputs 
        # ------
        
        # Initial conditions 
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        v_bus_a, v_bus_b, v_bus_c = dq02abc(v_bus_D, v_bus_Q, 0, 0)

        u = DynamicalVariables(
                            name=["v_dc_ref", "i_load_ref", "i_bus_q_ref", "v_bus_a", "v_bus_b", "v_bus_c"],
                            component=f"{self.type_}_{self.id}",
                            type=["device", "device", "device", "grid", "grid", "grid"],
                            init=[self.emt_init.v_dc, self.emt_init.i_load_ref, self.emt_init.i_bus_q, v_bus_a, v_bus_b, v_bus_c]
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
        It returns a vector with the differential equations that describe the dynamics of the GFLI.
        This model includes: pi controller, pll, and LCL filter.
        """    
        # Get state values # here in progress
        pi_cc_d, pi_cc_q, theta_pll, gamma_pll, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, int_vdc, v_dc, i_load = self.variables_emt.x.value 
        
        # Get input values (external inputs)
        v_dc_ref, i_load_ref, i_bus_q_ref, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value

        # convert relevant quantities to dq 
        v_bus_d, v_bus_q, _ = abc2dq0(v_bus_a, v_bus_b, v_bus_c, theta_pll) 
        i_bus_d, i_bus_q, _ = abc2dq0(i_bus_a, i_bus_b, i_bus_c, theta_pll) 
      
        # Update algebraic states
        
        # outer loop Vdc control 
        i_bus_d_ref = self.kp_oc_pu*(-v_dc_ref + v_dc) + int_vdc 
        
        # current controller output 
        e_d = pi_cc_d + self.kp_cc_pu * (i_bus_d_ref - i_bus_d) 
        e_q = pi_cc_q + self.kp_cc_pu * (i_bus_q_ref - i_bus_q)
        
        v_vsc_d = e_d + self.beta * v_bus_d - (self.xf1_pu + self.xf2_pu) * i_bus_q 
        v_vsc_q = e_q + self.beta * v_bus_q + (self.xf1_pu + self.xf2_pu) * i_bus_d  
        
        # calculate idc 
        i_vsc_d, i_vsc_q, _ = abc2dq0(i_vsc_a, i_vsc_b, i_vsc_c, theta_pll)
        i_dc = (v_vsc_d*i_vsc_d + v_vsc_q*i_vsc_q)/v_dc

        # convert to abc to feed into filter dynamics 
        v_vsc_a, v_vsc_b, v_vsc_c = dq02abc(v_vsc_d, v_vsc_q, 0, theta_pll) 

        def current_controller_dynamics(y, internal_inputs):
            """
            It returns the differential equations that describe the dynamics of the current controller.
            The current controller has: virtual inertia, filter for active and reactive power.
            """

            # Definition of states for the ODEs of the current controller
            pi_cc_d, pi_cc_q  = y[0], y[1]  

            # Extract the list of parameters
            kp_cc = self.kp_cc_pu  # proportional gain of current controller
            ki_cc = self.ki_cc_puHz  # integral gain of current controller

            # Define internal inputs for the current controller at timepoint "t"
            i_bus_d_ref, i_bus_q_ref, i_bus_d, i_bus_q = internal_inputs

            # Define ODEs that describe the dynamics of the current controller
            d_pi_cc_d = ki_cc * (i_bus_d_ref - i_bus_d)
            d_pi_cc_q = ki_cc * (i_bus_q_ref - i_bus_q)
            
            return [d_pi_cc_d, d_pi_cc_q]

        def pll_dynamics(y, internal_inputs):
            """
            It returns the differential equations that describe the dynamics of the PLL.
            The PLL tracks the phase of the grid voltage.
            """    
            # Definition of states for the ODEs of the pll
            theta_pll, gamma_pll = y[0], y[1]

            # Extract the list of parameters
            kp_pll = self.kp_pll_pu  # proportional gain of PLL
            ki_pll = self.ki_pll_puHz  # integral gain of PLL
            w_base = self.wbase # base frequency of the system

            # Define voltage at bus
            v_bus_q = internal_inputs

            # Define ODEs that describe the dynamics of the PLL
            d_theta_pll = kp_pll * v_bus_q + gamma_pll + w_base
            d_gamma_pll = ki_pll * v_bus_q

            return [d_theta_pll, d_gamma_pll]

        def lcl_filter_dynamics(y, internal_inputs):
            """
            It returns the differential equations that describe the dynamics of the LCL filter.
            The LCL filter connects the VSC to the grid. It has three branches: the first branch (RL) connects the VSC to the shunt element, the second branch is the shunt element (RC), and the third branch (RL) connects the shunt element to the grid.
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
          
        def outer_loop_and_dc_side(y, internal_inputs):
            """
            Dynamics of PI controller (of Vdc), Vdc (capacitor dynamics based on current balance), and the load (current source). 
            """
            int_vdc, v_dc, i_load = y[0], y[1], y[2]
            
            v_dc_ref, i_load_ref, i_dc = internal_inputs 
            
            d_int_vdc = self.ki_oc_puHz*(-v_dc_ref + v_dc)
            d_vdc = (self.wbase/self.c_dc)*(-i_load - v_dc/self.r_dc - i_dc)
            d_iload = 1/(self.Tload)*(i_load_ref - i_load)
            
            return [d_int_vdc, d_vdc, d_iload]
        
        d_pi_cc_d, d_pi_cc_q = current_controller_dynamics([pi_cc_d, pi_cc_q], [i_bus_d_ref, i_bus_q_ref, i_bus_d, i_bus_q])

        d_theta_pll, d_gamma_pll = pll_dynamics([theta_pll, gamma_pll], v_bus_q)

        di_vsc_a, di_vsc_b, di_vsc_c, dv_sh_a, dv_sh_b, dv_sh_c, di_bus_a, di_bus_b, di_bus_c = lcl_filter_dynamics([i_vsc_a , i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c], [v_vsc_a, v_vsc_b, v_vsc_c, v_bus_a, v_bus_b, v_bus_c])
        
        d_int_vdc, d_vdc, d_iload = outer_loop_and_dc_side([int_vdc, v_dc, i_load], [v_dc_ref, i_load_ref, i_dc])

        return [d_pi_cc_d, d_pi_cc_q, d_theta_pll, d_gamma_pll, di_vsc_a, di_vsc_b, di_vsc_c, dv_sh_a, dv_sh_b, dv_sh_c, di_bus_a, di_bus_b, di_bus_c, d_int_vdc, d_vdc, d_iload]
    
    def get_output_emt(self):
        
        pi_cc_d, pi_cc_q, theta_pll, gamma_pll, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, int_vdc, v_dc, i_load = self.variables_emt.x.value
        return [i_bus_a, i_bus_b, i_bus_c]

    def plot_results_emt(self) -> DynamicalVariables:
        """
        Plot EMT simulation results
        """

        pi_cc_d, pi_cc_q, theta_pll, gamma_pll, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, int_vdc, v_dc, i_load = self.variables_emt.x.value
        
        tps = self.variables_emt.x.time
        
        # Transform abc to dq0
        i_vsc_d, i_vsc_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_vsc_a, i_vsc_b, i_vsc_c, theta_pll)])
        v_sh_d, v_sh_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(v_sh_a, v_sh_b, v_sh_c, theta_pll)])
        i_bus_d, i_bus_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_bus_a, i_bus_b, i_bus_c, theta_pll)])
        
        # Additional quantities to plot 
        v_sh_dq = v_sh_d + np.multiply(v_sh_q, 1j)
        i_vsc_dq = i_vsc_d + np.multiply(i_vsc_q, 1j)
        i_bus_dq = i_bus_d + np.multiply(i_bus_q, 1j)
        v_vsc_dq = v_sh_dq + np.multiply((self.rf1_pu + self.xf1_pu * 1j), i_vsc_dq)
        p_vsc = (v_vsc_dq*np.conjugate(i_vsc_dq)).real 
        
        p_sh = (v_sh_dq*np.conjugate(i_bus_dq)).real 
        q_sh = (v_sh_dq*np.conjugate(i_bus_dq)).imag 
        
        p_load = v_dc*i_load
        
        results = DynamicalVariables(
            name=['pi_cc_d', 'pi_cc_q', 'theta_pll', 'gamma_pll', 'i_vsc_d', 'i_vsc_q', 'v_sh_d', 'v_sh_q', 'i_bus_d', 'i_bus_q', 'int_vdc', 'v_dc', 'i_load', 'p_vsc', 'p_sh', 'pload', 'q_sh'],
            component=f"{self.type_}_{self.id}",
            value=[pi_cc_d, pi_cc_q, theta_pll, gamma_pll, i_vsc_d, i_vsc_q, v_sh_d, v_sh_q, i_bus_d, i_bus_q, int_vdc, v_dc, i_load, p_vsc, p_sh, p_load, q_sh],
            time=tps
        )
        return results