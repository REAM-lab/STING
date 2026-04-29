"""
This module implements a GFLI that incorporates: 
- LCL filter: Two Series RL branches (one branch is the transformer) and one Parallel RC shunt. 
- Outer Vdc controller
- Current controller: A dq-based frame PI controller
- PLL: A basic implementation

in progress - Ruth (April 16)
"""
# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple, Optional, ClassVar
import scipy.linalg 

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
    int_q: float 
    i_dc: float 
    i_load: float 


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
    lf1_pu: float
    rsh_pu: float
    csh_pu: float
    txr_power_MVA: float
    txr_voltage1_kV: float
    txr_voltage2_kV: float
    txr_r1_pu: float
    txr_l1_pu: float
    txr_r2_pu: float
    txr_l2_pu: float
    beta: float
    kp_pll: float
    ki_pll: float
    kp_cc: float
    ki_cc: float
    vdc_ref: float # added 
    i_load_ref: float # added 
    c_dc: float # added 
    kp_oc: float # added 
    ki_oc: float # added 
    x_pll_rescale: np.ndarray = field(default_factory=lambda: np.array([[100, 0], [0, 1]])) 
    name: str = field(default_factory=str)
    emt_init: Optional[InitialConditionsEMT] = None
    ssm: Optional[StateSpaceModel] = None

    @property
    def rf2_pu(self):
        return (self.txr_r1_pu + self.txr_r2_pu) * self.base_power_MVA / self.txr_power_MVA

    @property
    def lf2_pu(self):
        return (self.txr_l1_pu + self.txr_l2_pu) * self.base_power_MVA / self.txr_power_MVA

    @property
    def wbase(self):
        return 2 * np.pi * self.base_frequency_Hz
    
    def _build_small_signal_model(self):

        # Current PI controller
        kp_cc, ki_cc = self.kp_cc, self.ki_cc
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
        rf1, lf1, rf2, lf2, rsh, csh = self.rf1_pu, self.lf1_pu, self.rf2_pu, self.lf2_pu, self.rsh_pu, self.csh_pu
        wb = self.wbase
        i_vsc_d, i_vsc_q = self.emt_init.i_vsc_d, self.emt_init.i_vsc_q
        i_bus_d, i_bus_q = self.emt_init.i_bus_d, self.emt_init.i_bus_q
        v_lcl_sh_d, v_lcl_sh_q = self.emt_init.v_lcl_sh_d, self.emt_init.v_lcl_sh_q

        lcl_filter = StateSpaceModel(
                        A = wb*np.array([[-rf1/lf1  ,   1       ,  0        ,   0       ,       -1/lf1      ,  0],
                                         [-1        ,   -rf1/lf1,  0        ,   0       ,       0           ,  -1/lf1],
                                         [0         ,   0       ,  -rf2/lf2 ,   1       ,       1/lf2       ,  0],
                                         [0         ,   0       ,  -1       ,   -rf2/lf2,       0           ,  1/lf2],
                                         [1/csh     ,   0       ,  -1/csh   ,   0       ,       -1/(rsh*csh),  1],
                                         [0         ,   1/csh   ,  0        ,   -1/csh  ,       -1          ,  -1/(rsh*csh)]]),
                        B = wb*np.array([[1/lf1 ,    0      ,   0       ,   0      ,      i_vsc_q],
                                         [0     ,    1/lf1  ,   0       ,   0      ,      -i_vsc_d],
                                         [0     ,    0      ,   -1/lf2  ,   0      ,      i_bus_q],
                                         [0     ,    0      ,   0       ,   -1/lf2 ,      -i_bus_d],
                                         [0     ,    0      ,   0       ,   0      ,      v_lcl_sh_q],
                                         [0     ,    0      ,   0       ,   0      ,      -v_lcl_sh_d]]),
                        C = np.eye(6),
                        D = np.zeros((6,5)),
                        x = DynamicalVariables(name=["i_vsc_d", "i_vsc_q", "i_bus_d", "i_bus_q", "v_lcl_sh_d", "v_lcl_sh_q"],
                                               init=[i_vsc_d, i_vsc_q, i_bus_d, i_bus_q, v_lcl_sh_d, v_lcl_sh_q]),
                        u = DynamicalVariables(name=['v_vsc_d', 'v_vsc_q', 'v_bus_d', 'v_bus_q', 'w']),
                        y = DynamicalVariables(name=["i_vsc_d", "i_vsc_q", "i_bus_d", "i_bus_q", "v_lcl_sh_d", "v_lcl_sh_q"]))

        # Phase-locked loop
        kp_pll, ki_pll = self.kp_pll, self.ki_pll
        beta = self.beta
        vmag_bus = self.emt_init.vmag_bus
        sinphi = np.sin(self.emt_init.angle_ref*np.pi/180)
        cosphi = np.cos(self.emt_init.angle_ref*np.pi/180)
        int_pll = 0
        phase_pll =  self.emt_init.angle_ref*np.pi/180

        pll = StateSpaceModel(  A = np.array([  [  0         ,  -vmag_bus*ki_pll],
                                                [wb          , -wb*vmag_bus*kp_pll]]),
                                B = np.array([  [-sinphi*ki_pll   ,        +cosphi*ki_pll],
                                                [-wb*kp_pll*sinphi,  wb*kp_pll*cosphi]]),
                                C = np.array([  [0  , 1],
                                                [1  , -1*vmag_bus*kp_pll]]),
                                D = np.array([  [0                ,           0],
                                                [-1*kp_pll*sinphi ,  1*kp_pll*cosphi]]),
                                u = DynamicalVariables(name=['v_bus_D', 'v_bus_Q']),
                                y = DynamicalVariables(name=['phase', 'w']),
                                x = DynamicalVariables(name=["int_pll", "phase_pll"], 
                                                       init=[int_pll, phase_pll] ) )
        
        # Re-scale the states so that they are not very small numbers compared to 
        # other states. It was tested in EMT simulation.
        pll.A = self.x_pll_rescale @ pll.A @ scipy.linalg.inv(self.x_pll_rescale)
        pll.B = self.x_pll_rescale @ pll.B
        pll.C = pll.C @ scipy.linalg.inv(self.x_pll_rescale)
        
        # Outer control + DC capacitor dynamics
        Kp, Ki, Cdc = self.kp_oc, self.ki_oc, self.c_dc 
        r_dc = 100
        outer_control = StateSpaceModel(A = np.array([[0, 0, Ki], 
                                                     [0, 0, 0],
                                                     [0, 0, -1/r_dc]]),
                                        B = np.array([[-Ki, 0, 0, 0, 0], 
                                                      [0, 0, -Ki, Ki, 0], 
                                                      [0, wb/Cdc, 0, 0, -wb/Cdc]]),
                                        C = np.array([[1, 0, Kp], 
                                                      [0, 1, 0],
                                                      [0, 0, 1]]),
                                        D = np.array([[-Kp, 0, 0, 0, 0],
                                                      [0, 0, -Kp, Kp, 0], 
                                                      [0, 0, 0, 0, 0]]),
                                        u = DynamicalVariables(name=["vdc_ref", "i_load_ref", "q_ref", "q", "idc"]),
                                        y = DynamicalVariables(name=["i_bus_d_ref", "i_bus_q_ref", "v_dc"]),
                                        x = DynamicalVariables(name=["int_vdc", "int_q", "v_dc"], 
                                                               init=[self.emt_init.int_vdc, self.emt_init.int_q, self.emt_init.v_dc]))

        # Construction of CCM matrices
        # ustack = F*ystack + G*u 
        
        # ustack = i2cdq, v1cdq, v2cdq, w, v2cdq, vdcref, iload, qref, q, idc (14)
        # y_stack = vmd, vmq, i2d, i2q, i1d, i1q, v3d, v3q, theta, w, idref, iqref, vdc (13) - extra 4 + 3 = 7 
        # u = vdcref, iload, qref, v2d, v2q
        # y = i2dq 
        
        # reactive power linearization coefficients         
        a1 = -self.emt_init.v_lcl_sh_d
        a2 = -self.emt_init.i_bus_q
        a3 = self.emt_init.v_lcl_sh_q
        a4 = self.emt_init.i_bus_d
        
        # dc power balance linearization 
        b1 = self.emt_init.v_vsc_d/self.emt_init.v_dc
        b2 = self.emt_init.i_vsc_d/self.emt_init.v_dc 
        b3 = self.emt_init.v_vsc_q/self.emt_init.v_dc # is this just 0 ? 
        b4 = self.emt_init.i_vsc_q/self.emt_init.v_dc 
        b5 = - self.emt_init.i_dc/self.emt_init.v_dc 

        # Construction of CCM matrices
        Fccm = np.vstack( ( np.hstack((np.zeros((10, )), 1, 0, 0)) ,# i2ref_d
                            np.hstack((np.zeros((10, )), 0, 1, 0)) , # i2ref_q 
                            np.hstack((np.zeros((2,4)), np.eye(2) ,np.zeros((2,7)))), # i2c_dq
                            [1, 0, 0, 0, 0, -(lf1+lf2),  0, 0, 0, 0, 0, 0, 0], # v1c_d
                            [0, 1, 0, 0, (lf1+lf2), 0, 0, 0, -beta*vmag_bus, 0, 0, 0, 0], # v1c_q
                            np.zeros((13, )) , # v2c_d
                            np.append( np.zeros((8,)) , [-vmag_bus,  0, 0, 0, 0] ), # v2c_q
                            np.append( np.zeros((9,)) , [1, 0, 0, 0] ), # w
                            np.zeros((2, 13)), # v2c_dq
                            np.zeros((3, 13)), #vdcref, iload, qref 
                            np.hstack((np.zeros(4,), [a3, a1, a2, a4], np.zeros(5,))), # q 
                            np.hstack(([b2, b4, b1, b3], np.zeros(8,), [b5])) #idc
                         )
        ) 


        Gccm = np.vstack((      np.zeros((4, 5)), # i2ref_dq, i2c_dq, 
                                [0, 0, 0, beta*cosphi ,    beta*sinphi],  # v1c_d
                                [0, 0, 0, -beta*sinphi,    beta*cosphi], # v1c_q
                                [0, 0, 0, cosphi   ,sinphi], # v2c_d
                                [0, 0,  0, -sinphi ,cosphi], # v2c_q
                                np.zeros((5, )), # w
                                np.hstack((np.zeros((2,3)), np.eye(2) ) ),  # v2dq 
                                np.hstack((np.eye(3), np.zeros((3,2)))), # vdc_ref, iload, q_ref 
                                np.zeros((2,5)) # q, idc 
                                ) 
                         ) 
  
        Hccm = np.vstack(( [ 0, 0 , 0, 0, cosphi , -sinphi, 0, 0, -sinphi*i_bus_d-cosphi*i_bus_q, 0, 0, 0, 0],
                            [0, 0, 0, 0, sinphi , cosphi , 0, 0, cosphi*i_bus_d-sinphi*i_bus_q, 0, 0, 0, 0] ))
        
        Lccm = np.zeros((2, 5))

        components = [pi_controller, lcl_filter, pll, outer_control]
        connections = [Fccm, Gccm, Hccm, Lccm]

        # Inputs and outputs
        v_bus_D, v_bus_Q= self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        u = DynamicalVariables(
                                name=["vdc_ref", "i_load_ref", "q_ref", "v_bus_D", "v_bus_Q"],
                                type=["device", "device", "device", "grid", "grid"],
                                init=[self.vdc_ref, self.emt_init.i_dc, self.emt_init.q_bus, v_bus_D, v_bus_Q])

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
        v_lcl_sh_DQ = v_bus_DQ + (self.rf2_pu + self.lf2_pu * 1j) * i_bus_DQ

        # Current flowing through shunt element of LCL filter
        i_lcl_sh_DQ = v_lcl_sh_DQ * (self.csh_pu * 1j) + v_lcl_sh_DQ / self.rsh_pu

        # Current sent from the beginning of the LCL filter
        i_vsc_DQ = i_bus_DQ + i_lcl_sh_DQ
        v_vsc_DQ = v_lcl_sh_DQ + (self.rf1_pu + self.lf1_pu * 1j) * i_vsc_DQ

        # We refer the voltage and currents to the synchronous frames of the
        # inverter
        v_vsc_dq = v_vsc_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)
        i_vsc_dq = i_vsc_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        v_bus_dq = v_bus_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)
        i_bus_dq = i_bus_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        v_lcl_sh_dq = v_lcl_sh_DQ * np.exp(-angle_ref * np.pi / 180 * 1j)

        # Initial conditions for the integral controller
        pi_cc_dq = (
            v_vsc_dq - self.beta * v_bus_dq - 1j * (self.lf1_pu + self.lf2_pu) * i_bus_dq
        )
        
        
        # Initial conditions for outer control / dc 
        
        # DC-side initial conditions 
        v_dc = self.vdc_ref
        p_vsc = (v_vsc_dq*np.conjugate(i_vsc_dq)).real # power at converter terminals 
        i_dc = p_vsc/v_dc 

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
            int_q=i_bus_dq.imag,
            i_load=i_dc,
            v_vsc_d=v_vsc_dq.real,
            v_vsc_q=v_vsc_dq.imag
            
        )
        
    def define_variables_emt(self):
        # States 
        # ------ 
        

        
        x = DynamicalVariables(
            name = [],
            component = f"{self.type_}_{self.id}",
            init=[]
        )
        
        # Inputs 
        # ------
        
        # Initial conditions 

        u = DynamicalVariables(
                            name=["vdc_ref", "iload", "q_ref", "v_bus_a", "v_bus_b", "v_bus_c"],
                            component=f"{self.type_}_{self.id}",
                            type=["device", "device", "device", "grid", "grid"],
                            init=[]
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
        
        # Get state values 
        # outer control / current control / lcl / pll 
        angle_pc, w_pc, p_pc, q_pc, gamma, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, i_Lf, v_dcf, i_dcf, i_loadf, x1, x2, i_L, v_dc, i_load, soc, x3 = self.variables_emt.x.value 
        
        # Get input values (external inputs)
        vdc_ref, iload, qref, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value 

        # convert relevant quantities to dq (ibr frame)
        v_sh_d, v_sh_q, _ = abc2dq0(v_sh_a, v_sh_b, v_sh_c, angle_pc) # for power controller 
        i_bus_d, i_bus_q, _ = abc2dq0(i_bus_a, i_bus_b, i_bus_c, angle_pc) # for power controller 
        i_vsc_d, i_vsc_q, _ = abc2dq0(i_vsc_a, i_vsc_b, i_vsc_c, angle_pc)

        # outer control 
        id_ref = int_vdc + self.kp_oc*(vdc_ref - v_dc)
        iq_ref = int_q + self.kp_oc*(qref - q)
        
        # current controller 
        vmd = self.ki_cc*gamma_d + self.kp_cc*(id_ref - i_d)
        vmq = self.ki_cc*gamma_q + self.kp_cc*(id_ref - i_d)
        
        # PLL 
        
        
        # Do Q-V droop 
        v_sh_mag_ref = v_ref - self.droop_q_pu*(q_pc - q_ref) # adjusting voltage reference based on reactive power error (measured at capacitor)
        
        # NB updating algebraic states!
        # Updating converter terminal voltages 
        v_vsc_d = gamma + self.kp_vc_pu*(v_sh_mag_ref - (v_sh_d**2 + v_sh_q**2)**0.5) # update 
        v_vsc_q = 0.0 # update 
        
        # # check magnitude wrt v_dc and limit if necessary 
        # if v_vsc_d > v_dc: # can check only d component because v_vsc_q= 0 
        #     v_vsc_d = v_dc 
        
        # convert to abc to feed into filter dynamics 
        v_vsc_a, v_vsc_b, v_vsc_c = dq02abc(v_vsc_d, v_vsc_q, 0, angle_pc) # correct to use this angle?
        
        # DC/AC power balance 
        i_dc = (v_vsc_d*i_vsc_d + v_vsc_q*i_vsc_q)/v_dc
        
        
        
        # Differential equations 
        # ----------------------
        
        # Power controller: 
        def outer_loop_dynamics(y, internal_inputs):
            """
            PI control of Vdc error to generate id_ref
            PI control of Q error to generate iq_ref 
            """
            int_vdc, int_q, v_dc = y[0], y[1], y[2]
            
            vdc_ref, iload, qref, idc, q = internal_inputs
            
            d_int_vdc = self.Ki_oc*(v_dc_ref - v_dc)
            d_int_q = self.Ki_oc*(qref - q)
            d_v_dc = (-self.wbase/self.cdc)*(idc - iload)

            return [d_int_vdc, d_int_q, d_v_dc]
        
        
        def current_controller_dynamics(y, internal_inputs):
            """
            decoupled PI controllers 
            """    
            gamma = y[0]
            
            id_ref, iq_ref, id, iq = internal_inputs
            
            d_gamma_d = id_ref - id 
            d_gamma_q = iq_ref - iq 

            return [d_gamma_d, d_gamma_q]
        
        
        def lcl_filter_dynamics(y, internal_inputs):
            """
            The LCL filter connects the VSC to the grid. It has three branches: the first branch (RL) connects
            the VSC to the shunt element, the second branch is the shunt element (RC), and the third branch (RL)
            connects the shunt element to the grid.
            """

            # Definition of states for the ODEs of the LCL filter
            i_vsc_a , i_vsc_b, i_vsc_c = y[0], y[1], y[2] # currents flowing out of VSC
            v_sh_a, v_sh_b, v_sh_c = y[3], y[4], y[5] # currents flowing through paralell RC shunt
            i_bus_a, i_bus_b, i_bus_c = y[6], y[7], y[8] # currents flowing to bus

            # Extract the list of parameters
            rf1, xf1, rf2, xf2, rsh, csh = self.rf1_pu, self.xf1_pu, self.rf2_pu, self.xf2_pu, self.rsh_pu, self.csh_pu
            wb = self.wbase
            r1 = rf1 
            r2 = rf2 
            x1 = xf1 
            x2 = xf2 

            # Inputs 
            v_vsc_a, v_vsc_b, v_vsc_c, v_bus_a, v_bus_b, v_bus_c = internal_inputs

            # Define ODEs that describe the dynamics of the LCL filter
            di_vsc_a = wb/x1 *(v_vsc_a - v_sh_a - r1 * i_vsc_a)
            di_vsc_b = wb/x1 *(v_vsc_b - v_sh_b - r1 * i_vsc_b)
            di_vsc_c = wb/x1 *(v_vsc_c - v_sh_c - r1 * i_vsc_c)

            dv_sh_a = wb/csh * (-v_sh_a/rsh + i_vsc_a - i_bus_a)
            dv_sh_b = wb/csh * (-v_sh_b/rsh + i_vsc_b - i_bus_b)
            dv_sh_c = wb/csh * (-v_sh_c/rsh + i_vsc_c - i_bus_c)

            di_bus_a = wb/x2 *(v_sh_a - v_bus_a - r2 * i_bus_a)
            di_bus_b = wb/x2 *(v_sh_b - v_bus_b - r2 * i_bus_b)
            di_bus_c = wb/x2 *(v_sh_c - v_bus_c - r2 * i_bus_c)

            return [di_vsc_a, di_vsc_b, di_vsc_c, dv_sh_a, dv_sh_b, dv_sh_c, di_bus_a, di_bus_b, di_bus_c]
        

        def pll_dynamics(y, internal_inputs):
            """ 
            """

            int_pll, theta = y[0], y[1]
            
            v_bus_q = internal_inputs 
        
            d_int_pll = self.ki_pll*(v_bus_q)
            d_theta = self.wbase*self.ki_pll*int_pll - self.wbase*vm*self.kp_pll*theta 
            
            return  [d_int_pll, d_theta]
        
        
        dy_pc = outer_loop_dynamics([angle_pc, w_pc, p_pc, q_pc], [v_sh_d, v_sh_q, i_bus_d, i_bus_q, p_ref])
        dy_vc = current_controller_dynamics([gamma], [v_sh_mag_ref, v_sh_d, v_sh_q])
        dy_lcl = lcl_filter_dynamics([i_vsc_a , i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c], [v_vsc_a, v_vsc_b, v_vsc_c, v_bus_a, v_bus_b, v_bus_c])
        dy_pll = pll_dynamics([int_pll, theta], [v_sh_a, v_sh_b, v_sh_c])

        return np.hstack([dy_pc, dy_vc, dy_lcl])
        

    def get_output_emt(self):
        
        # Output is i_bus_abc 
        angle_pc, w_pc, p_pc, q_pc, gamma, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, i_Lf, v_dcf, i_dcf, i_loadf, x1, x2, i_L, v_dc, i_load, soc, x3 = self.variables_emt.x.value 
        
        return [i_bus_a, i_bus_b, i_bus_c]

    def plot_results_emt(self, output_dir):
        
        angle_pc, w_pc, p_pc, q_pc, gamma, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, i_Lf, v_dcf, i_dcf, i_loadf, x1, x2, i_L, v_dc, i_load, soc, x3 = self.variables_emt.x.value 
        
        #i_L = np.clip(i_L, -1000.0, 1.0)
        
        tps = self.variables_emt.x.time
        
        # Transform abc to dq0
        i_vsc_d, i_vsc_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_vsc_a, i_vsc_b, i_vsc_c, angle_pc)])
        v_sh_d, v_sh_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(v_sh_a, v_sh_b, v_sh_c, angle_pc)])
        i_bus_d, i_bus_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_bus_a, i_bus_b, i_bus_c, angle_pc)])
        
        # calculate v_vsc 
        v_sh_dq = v_sh_d + np.multiply(v_sh_q, 1j)
        i_vsc_dq = i_vsc_d + np.multiply(i_vsc_q, 1j)
        v_vsc_dq = v_sh_dq + np.multiply((self.rf1_pu + self.xf1_pu * 1j), i_vsc_dq)
        
        fig = make_subplots(
            rows=14, cols=2,
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
        
        fig.add_trace(go.Scatter(x=tps, y=i_loadf, mode='lines', line=dict(color='red', dash='solid')),
                    row=3, col=2)
        fig.update_xaxes(title_text='Time [s]', row=3, col=2)
        fig.update_yaxes(title_text='iload_f [p.u.]', row=3, col=2)   
        
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

        fig.add_trace(go.Scatter(x=tps, y=v_dc, mode='lines', line=dict(color='red', dash='solid')),
                    row=7, col=1)
        fig.update_xaxes(title_text='Time [s]', row=7, col=1)
        fig.update_yaxes(title_text='v_dc [p.u.]', row=7, col=1)

        fig.add_trace(go.Scatter(x=tps, y=i_L, mode='lines', line=dict(color='red', dash='solid')),
                    row=7, col=2)
        fig.update_xaxes(title_text='Time [s]', row=7, col=2)
        fig.update_yaxes(title_text='i_L [p.u.]', row=7, col=2)
        
        
        fig.add_trace(go.Scatter(x=tps, y=v_dcf, mode='lines', line=dict(color='red', dash='solid')),
                    row=8, col=1)
        fig.update_xaxes(title_text='Time [s]', row=8, col=1)
        fig.update_yaxes(title_text='v_dcf [p.u.]', row=8, col=1)

        fig.add_trace(go.Scatter(x=tps, y=i_Lf, mode='lines', line=dict(color='red', dash='solid')),
                    row=8, col=2)
        fig.update_xaxes(title_text='Time [s]', row=8, col=2)
        fig.update_yaxes(title_text='i_Lf [p.u.]', row=8, col=2)
        
        
        fig.add_trace(go.Scatter(x=tps, y=i_dcf, mode='lines', line=dict(color='red', dash='solid')),
                    row=9, col=1)
        fig.update_xaxes(title_text='Time [s]', row=9, col=1)
        fig.update_yaxes(title_text='i_dcf [p.u.]', row=9, col=1)

        fig.add_trace(go.Scatter(x=tps, y=x1, mode='lines', line=dict(color='red', dash='solid')),
                    row=9, col=2)
        fig.update_xaxes(title_text='Time [s]', row=9, col=2)
        fig.update_yaxes(title_text='x1 [p.u.]', row=9, col=2)

        fig.add_trace(go.Scatter(x=tps, y=x2, mode='lines', line=dict(color='red', dash='solid')),
                    row=10, col=1)
        fig.update_xaxes(title_text='Time [s]', row=10, col=1)
        fig.update_yaxes(title_text='x2 [p.u.]', row=10, col=1)

        fig.add_trace(go.Scatter(x=tps, y=i_load, mode='lines', line=dict(color='red', dash='solid')),
                    row=10, col=2)
        fig.update_xaxes(title_text='Time [s]', row=10, col=2)
        fig.update_yaxes(title_text='i_load [p.u.]', row=10, col=2)
        
        # power comparisons (calculated)
        p_vsc = (v_vsc_dq*np.conjugate(i_vsc_dq)).real 
        p_load = i_load*v_dc 
        p_ref, q_ref, v_ref, v_dc_ref, v_s, i_load_ref, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value 
        p_bat = i_L*v_s  
        p_capacitor = p_vsc + p_load - p_bat 
        fig.add_trace(go.Scatter(x=tps, y=p_vsc, name="p_vsc", mode='lines', line=dict(color='red', dash='solid')),
                    row=11, col=1)
        fig.add_trace(go.Scatter(x=tps, y=p_load, name="p_load", mode='lines', line=dict(color='blue', dash='solid')),
                    row=11, col=1)
        fig.add_trace(go.Scatter(x=tps, y=p_bat, name="p_bat", mode='lines', line=dict(color='green', dash='solid')),
                    row=11, col=1)
        fig.add_trace(go.Scatter(x=tps, y=p_capacitor, name="p_cap", mode='lines', line=dict(color='pink', dash='solid')),
                    row=11, col=1)
        
        fig.add_trace(go.Scatter(x=tps, y=np.ones_like(p_bat)*self.Pbat_max_pu, name="p_bat_lim", mode='lines', line=dict(color='green', dash='dot')),
                    row=11, col=1)
        
        fig.update_xaxes(title_text='Time [s]', row=11, col=1)
        fig.update_yaxes(title_text='p [p.u.]', row=11, col=1)
        
        # v_vsc (calculated)
        fig.add_trace(go.Scatter(x=tps, y=v_vsc_dq.real, mode='lines', line=dict(color='red', dash='solid')),
                    row=12, col=1)
        fig.update_xaxes(title_text='Time [s]', row=12, col=1)
        fig.update_yaxes(title_text='v_vsc_d [p.u.]', row=12, col=1)
        
        fig.add_trace(go.Scatter(x=tps, y=v_vsc_dq.imag, mode='lines', line=dict(color='red', dash='solid')),
                    row=12, col=2)
        fig.update_xaxes(title_text='Time [s]', row=12, col=2)
        fig.update_yaxes(title_text='v_vsc_q [p.u.]', row=12, col=2)
        
        e_cap = 0.5*self.c_dc*(v_dc**2)
        e_ind = 0.5*self.l_dc*(i_L**2)
        fig.add_trace(go.Scatter(x=tps, y=soc, name='battery soc', mode='lines', line=dict(color='red', dash='solid')),
                    row=13, col=1)
        fig.add_trace(go.Scatter(x=tps, y=e_cap, name='cap energy', mode='lines', line=dict(color='blue', dash='dot')),
                    row=13, col=1)
        fig.add_trace(go.Scatter(x=tps, y=e_ind, name='ind energy', mode='lines', line=dict(color='pink', dash='dot')),
                    row=13, col=1)
        
        fig.add_trace(go.Scatter(x=tps, y=np.ones_like(soc)*self.SOC_max_pu, name="soc lim", mode='lines', line=dict(color='green', dash='dot')),
                    row=13, col=1)
        fig.update_xaxes(title_text='Time [s]', row=13, col=1)
        fig.update_yaxes(title_text='energy [p.u.]', row=13, col=1)
        
        duty_cycle = self.kp_i_L*(self.kp_v_dc*(v_dc_ref - v_dcf) + x1 - i_Lf + self.Kff_idc*i_dcf + self.Kff_iload*i_loadf) + x2
        duty_cycle = np.clip(duty_cycle, 0.0, 1.0)
        fig.add_trace(go.Scatter(x=tps, y=duty_cycle, name="duty cycle", mode='lines', line=dict(color='red', dash='solid')),
                    row=13, col=2)
        fig.update_xaxes(title_text='Time [s]', row=13, col=2)
        fig.update_yaxes(title_text='duty cycle [p.u.]', row=13, col=2)
        
        fig.add_trace(go.Scatter(x=tps, y=x3, mode='lines', line=dict(color='red', dash='solid')),
                    row=14, col=1)
        fig.update_xaxes(title_text='Time [s]', row=14, col=1)
        fig.update_yaxes(title_text='x3 [p.u.]', row=14, col=1)
        

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
                
                
            