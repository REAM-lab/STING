# Import standard python packages
import numpy as np
from typing import NamedTuple, Optional
from dataclasses import dataclass

# Import sting packages
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
    v_vsc_d: float
    v_vsc_q: float
    i_bus_d: float
    i_bus_q: float
    v_dc: float
    i_out: float
    i_bus_D: float
    i_bus_Q: float
    v_bus_D: float
    v_bus_Q: float

@dataclass(slots=True)
class GFLI_b:
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
    rf: float	
    lf: float
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
    i_dc_src: float
    r_dc: float
    c_dc: float 
    kp_dc:float
    ki_dc:float
    name: Optional[str] = None
    pf: Optional[Power_flow_variables] = None
    emt_init_cond: Optional[EMT_initial_conditions] = None
    ssm: Optional[State_space_model] = None

    @property
    def txr_r(self):
        return (self.txr_r1 + self.txr_r2)*self.sbase/self.txr_sbase

    @property
    def txr_l(self):
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
        
        # L filter
        rf = self.rf + self.txr_r
        lf = self.lf + self.txr_l
        wb = self.wbase
        i_bus_d, i_bus_q = self.emt_init_cond.i_bus_d, self.emt_init_cond.i_bus_q

        l_filter = State_space_model( A = wb*np.array([[-rf/lf,  1], 
                                                       [-1    ,  -rf/lf]]),
                                      B = wb*np.array([[ 1/lf ,  0   ,-1/lf  ,0, -i_bus_q] ,
                                                       [0, 1/lf, 0, -1/lf, i_bus_d]]),
                                      C = np.eye(2),
                                      D = np.zeros((2,5)),
                                      states= ['i_bus_d', 'i_bus_q'],
                                      inputs=['v_vsc_d', 'v_vsc_q', 'v_bus_d', 'v_bus_q', 'w'],
                                      outputs=['i_bus_d', 'i_bus_q'],
                                      initial_states = np.array([[i_bus_d], [i_bus_q]]))
        
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

        # DC voltage PI controller
        kp_dc, ki_dc = self.kp_dc, self.ki_dc
        i_bus_d = self.emt_init_cond.i_bus_d
        dc_pi_controller = State_space_model( A = 0,
                                              B = ki_dc*np.array([-1, +1]),
                                              C = 1,
                                              D = kp_dc*np.array([-1, +1]),
                                              states=['pi_dc'],
                                              inputs=['v_dc_ref', 'v_dc'],
                                              outputs=['i_bus_d_ref'],
                                              initial_states= np.array([[i_bus_d]]))
        
        # DC circuit
        r_dc, c_dc = self.r_dc, self.c_dc
        v_dc = self.emt_init_cond.v_dc
        dc_circuit = State_space_model(A = -wb*2*1/(c_dc*r_dc),
                                       B = wb*2*1/c_dc*np.array([1, -1]),
                                       C = 1,
                                       D = np.array([0 , 0]),
                                       states = ['v_dc'],
                                       inputs=['i_dc_src', 'i_out'],
                                       outputs=['v_dc'],
                                       initial_states = np.array([[v_dc]]))
        
        # Construction of CCM matrices
        v_vsc_d = self.emt_init_cond.v_vsc_d
        v_vsc_q = self.emt_init_cond.v_vsc_q
        i_out = self.emt_init_cond.i_out

        a =  pi_cc_d + beta*v_vsc_d
        b =  pi_cc_q + beta*v_vsc_q
        c =  beta*i_bus_d*cosphi + beta*i_bus_q*(-sinphi)
        d =  beta*i_bus_d*sinphi + beta*i_bus_q*cosphi
        
        e = -sinphi*i_bus_d - cosphi*i_bus_q
        f = cosphi*i_bus_d - sinphi*i_bus_q

        Fccm = np.vstack( ( [0, 0, 0, 0, 0, 0, 1, 0],
                             np.zeros((1,8)),
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, -lf, 0, 0, 0, 0],
                             [0, 1, lf, 0, -beta*vmag_bus, 0, 0, 0],
                             np.zeros((1,8)),
                             [0, 0, 0, 0, -vmag_bus, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             np.zeros((3,8)),
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             np.zeros((1,8)),
                             1/v_dc*np.array([i_bus_d, i_bus_q, a, b, -beta*i_bus_q*vmag_bus, 0, 0, -i_out]) ) )
        
        Gccm = np.vstack(( [0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           np.zeros((2,5)),
                           [0, 0, 0, beta*cosphi, beta*sinphi],
                           [0, 0, 0, -beta*sinphi, beta*cosphi],
                           [0, 0, 0, cosphi, sinphi],
                           [0, 0, 0, -sinphi, cosphi],
                           np.zeros((1,5)),
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0],
                           np.zeros((1, 5)),
                           [0, 0, 1, 0, 0],
                           1/v_dc*np.array([0, 0, 0, c, d])))
        
        Hccm = np.array([[0, 0, cosphi, -sinphi, e, 0, 0, 0],
                         [0, 0,  sinphi, cosphi, f, 0, 0, 0]])
        
        Lccm = np.zeros((2,5))

        ssm = linear_systems_tools.connect_models_via_CCM(Fccm, Gccm, Hccm, Lccm, [pi_controller, 
                                                                                   l_filter, 
                                                                                   pll, 
                                                                                   dc_pi_controller, 
                                                                                   dc_circuit])
        
        # Inputs and outputs
        device_side_inputs = ['v_dc_ref', 'i_bus_q_ref', 'i_dc_src']
        i_dc_src = self.i_dc_src
        initial_device_side_inputs = np.array([[v_dc], [i_bus_q], [i_dc_src]])
        
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
                                     device_side_inputs=device_side_inputs,
                                     initial_device_side_inputs=initial_device_side_inputs,
                                     grid_side_inputs=grid_side_inputs,
                                     initial_grid_side_inputs=initial_grid_side_inputs,
                                     outputs=outputs,
                                     initial_outputs=initial_outputs)
        
        
        
    def _calculate_emt_initial_conditions(self):
        vmag_bus = self.pf.vmag_bus
        vphase_bus = self.pf.vphase_bus
        p_bus = self.pf.p_bus
        q_bus = self.pf.q_bus

        i_dc_src = self.i_dc_src
        r_dc = self.r_dc

        # Voltage in the end of the filter
        v_bus_DQ = vmag_bus*np.exp(vphase_bus*np.pi/180*1j)
        ref_angle = np.angle(v_bus_DQ, deg=True)

        # Current sent from the end of the filter
        i_bus_DQ = (p_bus - q_bus*1j)/np.conjugate(v_bus_DQ)

        # Voltage at the terminals of the VSC
        v_vsc_DQ = v_bus_DQ + (self.rf + self.txr_r + (self.lf + self.txr_l)*1j)*i_bus_DQ

        # We refer the voltage and currents to the synchronous frames of the
        # inverter 
        v_vsc_dq = v_vsc_DQ*np.exp(-ref_angle*np.pi/180*1j) 

        v_bus_dq = v_bus_DQ*np.exp(-ref_angle*np.pi/180*1j) 

        i_bus_dq = i_bus_DQ*np.exp(-ref_angle*np.pi/180*1j) 

        # Initial conditions for the integral controllers
        pi_cc_dq = v_vsc_dq - 1j*(self.lf + self.txr_l)*i_bus_dq - self.beta*v_bus_dq

        # Initial condition for DC-side circuit
        p_vsc = (v_vsc_dq*np.conjugate(i_bus_dq)).real
        v_dc = (i_dc_src + (i_dc_src**2 - 4*(1/r_dc)*p_vsc)**0.5)/(2/r_dc)
        i_out = p_vsc/v_dc

        
        self.emt_init_cond = EMT_initial_conditions(    vmag_bus = vmag_bus,
                                                        vphase_bus = vphase_bus,
                                                        p_bus = p_bus,
                                                        q_bus = q_bus,
                                                        ref_angle=ref_angle,
                                                        pi_cc_d= pi_cc_dq.real,
                                                        pi_cc_q= pi_cc_dq.imag,
                                                        v_vsc_d = v_bus_dq.real,
                                                        v_vsc_q = v_bus_dq.imag,
                                                        i_bus_d = i_bus_dq.real,
                                                        i_bus_q = i_bus_dq.imag, 
                                                        v_dc= v_dc,
                                                        i_out= i_out,
                                                        i_bus_D = i_bus_DQ.real,
                                                        i_bus_Q = i_bus_DQ.imag,
                                                        v_bus_D = v_bus_DQ.real,
                                                        v_bus_Q = v_bus_DQ.imag)
        


