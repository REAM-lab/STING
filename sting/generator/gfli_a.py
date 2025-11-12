# Import standard python packages
import pandas as pd
import numpy as np
from scipy.linalg import block_diag
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

# Import src packages
from sting.utils import linear_systems_tools
from sting.utils.linear_systems_tools import State_space_model

class Power_flow_variables(NamedTuple):
    p_bus: float
    q_bus: float
    vmag_bus: float
    vphase_bus: float

class EMT_initial_conditions(NamedTuple):
    vmag_bus: float
    vphase_bus: float
    p_bus: float
    q_bus: float
    ref_angle: float
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

@dataclass(slots=True)
class GFLI_a:
    idx: str
    bus_idx: str
    p_min: float	
    p_max: float
    q_min: float
    q_max: float
    sbase: float
    vbase: float
    vbase: float
    fbase: float
    vdc: float
    rf1: float	
    lf1: float
    csh: float
    rsh: float
    txr_sbase: float
    txr_r1: float
    txr_l1: float
    txr_r2: float
    txr_l2: float	
    beta: float	
    kp_pll: float
    ki_pll: float
    kp_cc: float	
    ki_cc: float
    name: str = field(default_factory=str)
    type: str = 'generator'
    pf: Optional[Power_flow_variables] = None
    emt_init_cond: Optional[EMT_initial_conditions] = None
    ssm: Optional[State_space_model] = None

    @property
    def rf2(self):
        return (self.txr_r1 + self.txr_r2)*self.sbase/self.txr_sbase

    @property
    def lf2(self):
        return (self.txr_l1 + self.txr_l2)*self.sbase/self.txr_sbase
    
    @property
    def wbase(self):
        return 2*np.pi*self.fbase

    def _load_power_flow_solution(self, power_flow_instance):
        sol = power_flow_instance.generators.loc[self.idx]
        self.pf  = Power_flow_variables(p_bus = sol.p.item(),
                                        q_bus = sol.q.item(),
                                        vmag_bus = sol.bus_vmag.item(),
                                        vphase_bus = sol.bus_vphase.item())

    def _calculate_emt_initial_conditions(self):
        vmag_bus = self.pf.vmag_bus
        vphase_bus = self.pf.vphase_bus
        p_bus = self.pf.p_bus
        q_bus = self.pf.q_bus

        # Voltage in the end of the LCL filter
        v_bus_DQ = vmag_bus*np.exp(vphase_bus*np.pi/180*1j); 
        ref_angle = np.angle(v_bus_DQ, deg=True)

        # Current sent from the end of the LCL filter
        i_bus_DQ = (p_bus - q_bus*1j)/np.conjugate(v_bus_DQ)

        # Voltage across the shunt element in the LCL filter
        v_lcl_sh_DQ = v_bus_DQ + (self.rf2 + self.lf2*1j)*i_bus_DQ

        # Current flowing through shunt element of LCL filter
        i_lcl_sh_DQ =  v_lcl_sh_DQ*(self.csh*1j) + v_lcl_sh_DQ/self.rsh

        # Current sent from the beginning of the LCL filter
        i_vsc_DQ = i_bus_DQ + i_lcl_sh_DQ
        v_vsc_DQ = v_lcl_sh_DQ + (self.rf1 + self.lf1*1j)*i_vsc_DQ

        # We refer the voltage and currents to the synchronous frames of the
        # inverter 
        v_vsc_dq = v_vsc_DQ*np.exp(-ref_angle*np.pi/180*1j) 
        i_vsc_dq = i_vsc_DQ*np.exp(-ref_angle*np.pi/180*1j) 

        v_bus_dq = v_bus_DQ*np.exp(-ref_angle*np.pi/180*1j) 
        i_bus_dq = i_bus_DQ*np.exp(-ref_angle*np.pi/180*1j) 

        v_lcl_sh_dq = v_lcl_sh_DQ*np.exp(-ref_angle*np.pi/180*1j); 

        # Initial conditions for the integral controller
        pi_cc_dq = v_vsc_dq - self.beta*v_bus_dq - 1j*(self.lf1 + self.lf2)*i_bus_dq 

        self.emt_init_cond = EMT_initial_conditions(vmag_bus= vmag_bus,
                                                    vphase_bus= vphase_bus,
                                                    p_bus = p_bus,
                                                    q_bus = q_bus,
                                                    ref_angle = ref_angle,
                                                    pi_cc_d = pi_cc_dq.real, 
                                                    pi_cc_q = pi_cc_dq.imag, 
                                                    i_vsc_d = i_vsc_dq.real, 
                                                    i_vsc_q = i_vsc_dq.imag,
                                                    i_bus_d = i_bus_dq.real, 
                                                    i_bus_q = i_bus_dq.imag,                                                    
                                                    v_lcl_sh_d = v_lcl_sh_dq.real, 
                                                    v_lcl_sh_q = v_lcl_sh_dq.imag,
                                                    i_bus_D = i_bus_DQ.real,
                                                    i_bus_Q = i_bus_DQ.imag,
                                                    v_bus_D = v_bus_DQ.real,
                                                    v_bus_Q = v_bus_DQ.imag
                                                    )

    def _build_small_signal_model(self):
              
        # Current PI controller
        kp_cc, ki_cc = self.kp_cc, self.ki_cc
        pi_cc_d, pi_cc_q = self.emt_init_cond.pi_cc_d, self.emt_init_cond.pi_cc_q

        pi_controller = State_space_model( A = np.zeros((2,2)), 
                                          B = np.hstack((np.eye(2), -np.eye(2))),
                                          C = ki_cc*np.eye(2),
                                          D = kp_cc*np.hstack((np.eye(2), -np.eye(2))),
                                          inputs = ['i_bus_d_ref', 'i_bus_q_ref'], 
                                          states= ['pi_cc_d', 'pi_cc_q'],
                                          outputs = ['e_d', 'e_q'],
                                          initial_states= np.array([[pi_cc_d], [pi_cc_q]]))
        

        # LCL filter
        rf1, lf1, rf2, lf2, rsh, csh = self.rf1, self.lf1, self.rf2, self.lf2, self.rsh, self.csh
        wb = self.wbase
        i_vsc_d, i_vsc_q = self.emt_init_cond.i_vsc_d, self.emt_init_cond.i_vsc_q
        i_bus_d, i_bus_q = self.emt_init_cond.i_bus_d, self.emt_init_cond.i_bus_q
        v_lcl_sh_d, v_lcl_sh_q = self.emt_init_cond.v_lcl_sh_d, self.emt_init_cond.v_lcl_sh_q

        lcl_filter = State_space_model(
                        A = wb*np.array([[-rf1/lf1 ,   1     ,      0   ,        0      ,     -1/lf1 ,     0],
                                        [-1   ,       -rf1/lf1  ,  0    ,       0      ,     0   ,        -1/lf1],
                                        [0    ,       0     ,      -rf2/lf2  ,  1     ,      1/lf2   ,    0],
                                        [0    ,       0     ,      -1     ,     -rf2/lf2 ,   0   ,        1/lf2],
                                        [1/csh   ,       0    ,       -1/csh  ,     0      ,       -1/(rsh*csh)  ,      1],
                                        [0     ,      1/csh     ,      0    ,       -1/csh  ,      -1    ,  -1/(rsh*csh)]]),
                        B = wb*np.array([[   1/lf1   ,    0     ,      0    ,       0    ,      i_vsc_q],
                                         [0     ,      1/lf1    ,   0    ,       0      ,      -i_vsc_d],
                                         [0     ,      0    ,       -1/lf2  ,    0     ,          i_bus_q],
                                         [0     ,      0    ,       0    ,       -1/lf2    ,     -i_bus_d],
                                         [0     ,      0    ,       0     ,      0      ,          v_lcl_sh_q],
                                         [0     ,      0    ,       0     ,      0      ,        -v_lcl_sh_d]]),
                    C = np.array([[0     ,      0     ,      1     ,      0     ,      0   ,    0],
                                       [0     ,      0     ,      0     ,      1     ,      0   ,    0]]),
                    D = np.zeros((2,5)),
                    states=["i_vsc_d", "i_vsc_q", "i_bus_d", "i_bus_q", "v_lcl_sh_d", "v_lcl_sh_q"],
                    inputs=['v_vsc_d', 'v_vsc_q', 'v_bus_d', 'v_bus_q', 'w'],
                    outputs=["i_vsc_d", "i_vsc_q", "i_bus_d", "i_bus_q"],
                    initial_states=  np.array([[i_vsc_d], [i_vsc_q], [i_bus_d], [i_bus_q], [v_lcl_sh_d], [v_lcl_sh_q]]))

        # Phase-locked loop
        kp_pll, ki_pll = self.kp_pll, self.ki_pll
        beta = self.beta
        vmag_bus = self.emt_init_cond.vmag_bus
        sinphi = np.sin(self.emt_init_cond.ref_angle*np.pi/180)
        cosphi = np.cos(self.emt_init_cond.ref_angle*np.pi/180)
        int_pll = 0
        phase_pll =  self.emt_init_cond.ref_angle*np.pi/180

        pll = State_space_model(
                                A = np.array([  [  0         ,  -vmag_bus],
                                                [wb*ki_pll ,  -wb*vmag_bus*kp_pll]]),
                                B = np.array([[  -sinphi      ,        +cosphi],
                                              [-wb*kp_pll*sinphi,  wb*kp_pll*cosphi]]),
                                C = np.array([  [  0  , 1],
                                                [1*ki_pll , -1*vmag_bus*kp_pll]]),
                                D = np.array([  [0          ,           0],
                                                [-1*kp_pll*sinphi ,  1*kp_pll*cosphi]]),
                                inputs = ['v_bus_D', 'v_bus_Q'],
                                outputs= ['phase', 'w'],
                                states=["int_pll", "phase_pll"],
                                initial_states=np.array([[int_pll], [phase_pll]])) 

        # Interconnection matrices
        Fccm = np.vstack( (     np.zeros((1,6)) ,# i2ref_d
                                np.zeros((1,6)) , # i2ref_q 
                                np.hstack((np.zeros((2,2)), np.eye(2) ,np.zeros((2,2)))), # i2c_dq
                                [1, 0,  0  ,       -(lf1+lf2),  0     ,     0], # v1c_d
                                [0 , 1 , (lf1+lf2), 0    ,      -beta*vmag_bus , 0], # v1c_q
                                np.zeros((1,6)) , # v2c_d
                                np.append( np.zeros((1,4)) , [-vmag_bus,  0] ), # v2c_q
                                np.append( np.zeros((1,5)) , [1] ), # w
                                np.zeros((2,6)) )) # v2c_dq

        Gccm = np.vstack((      [ 1, 0,  0, 0], # i2ref_d
                                [0,  1, 0, 0], # i2ref_q
                                np.zeros((2,4)), # i2c_dq
                                [0, 0, beta*cosphi ,   beta*sinphi],  # v1c_d
                                [0, 0, -beta*sinphi,    beta*cosphi], # v1c_q
                                [0, 0, cosphi ,sinphi], # v2c_d
                                [0, 0,  -sinphi ,cosphi], # v2c_q
                                np.zeros((1,4)), # w
                                np.hstack( (np.zeros((2,2)), np.eye(2) ) ) ) ) # v2_dq ;  
  
        Hccm = np.vstack(( [ 0, 0 ,cosphi , -sinphi, -sinphi*i_bus_d-cosphi*i_bus_q, 0],
                            [0, 0, sinphi , cosphi  , cosphi*i_bus_d-sinphi*i_bus_q, 0] ))

        Lccm = np.zeros((2,4))

        # Generate small-signal model 
        ssm = linear_systems_tools.connect_models_via_CCM(Fccm, Gccm, Hccm, Lccm, [pi_controller, lcl_filter, pll])

        # Note that states and initial states do not need to be defined as they result from stacking states in ccm tool.

        # Inputs and outputs
        device_side_inputs = ['i_bus_d_ref', 'i_bus_q_ref']
        initial_device_side_inputs = np.array([[i_bus_d], [i_bus_q]])
        
        grid_side_inputs = ['v_bus_D', 'v_bus_Q']
        v_bus_D, v_bus_Q= self.emt_init_cond.v_bus_D, self.emt_init_cond.v_bus_Q
        initial_grid_side_inputs = np.array([[v_bus_D], [v_bus_Q]])
        
        outputs = ['i_bus_D', 'i_bus_Q']
        i_bus_D, i_bus_Q= self.emt_init_cond.i_bus_D, self.emt_init_cond.i_bus_Q
        initial_outputs = np.array([[i_bus_D], [i_bus_Q]])

        self.ssm = State_space_model(A = ssm.A,
                                     B = ssm.B,
                                     C = ssm.C,
                                     D = ssm.D,
                                     states= ssm.states,
                                     initial_states=ssm.initial_states,
                                     outputs= outputs,
                                     initial_outputs=initial_outputs,
                                     device_side_inputs=device_side_inputs,
                                     initial_device_side_inputs= initial_device_side_inputs,
                                     grid_side_inputs=grid_side_inputs,
                                     initial_grid_side_inputs= initial_grid_side_inputs
                                     )
        
