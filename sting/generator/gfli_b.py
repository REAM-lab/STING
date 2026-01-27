"""
This module implements a GFLI that incorporates: 
- L filter: a series RL branch and a transformer
- current controller: A dq-based frame PI controller
- PLL: A basic implementation
- DC-side circuit: A resistance in parallel with two capacitors
- DC-side voltage controller: PI controller for the DC-side voltage.
"""
# ----------------------
# Import python packages
# ----------------------
import numpy as np
from typing import NamedTuple, Optional, ClassVar
from dataclasses import dataclass, field
import scipy.linalg 

# ------------------
# Import sting code
# ------------------
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables

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
    v_bus_d: float
    v_bus_q: float
    v_vsc_mag: float
    v_vsc_DQ_phase: float

# ----------------
# Main class
# ----------------
@dataclass(slots=True)
class GFLIb:
    """GFLI that has L filter, PLL, DC-side with voltage control."""
    id: int = field(default=-1, init=False)
    bus: str
    p_min: float	
    p_max: float
    q_min: float
    q_max: float
    base_power_VA: float
    base_voltage_V: float
    base_frequency_Hz: float
    rf_pu: float	
    xf_pu: float
    txr_power_VA: float
    txr_voltage1_V: float
    txr_voltage2_V: float
    txr_r1_pu: float
    txr_x1_pu: float
    txr_r2_pu: float
    txr_x2_pu: float	
    beta: float	
    kp_pll_pu: float
    ki_pll_puHz: float
    kp_cc_pu: float	
    ki_cc_puHz: float
    i_src_pu: float
    r_dc_pu: float
    c_dc_pu: float 
    kp_dc_pu: float
    ki_dc_puHz: float
    bus_id: int = None
    x_pll_rescale: np.ndarray = field(default_factory=lambda: np.array([[100, 0], [0, 1]])) 
    name: str = field(default_factory=str)
    type: str = "gfli_b"
    pf: Optional[PowerFlowVariables] = None
    emt_init: Optional[InitialConditionsEMT] = None
    ssm: Optional[StateSpaceModel] = None
    tags: ClassVar[list[str]] = ["generator"]

    @property
    def txr_r(self):
        return (self.txr_r1_pu + self.txr_r2_pu)*self.base_power_VA/self.txr_power_VA

    @property
    def txr_x(self):
        return (self.txr_x1_pu + self.txr_x2_pu)*self.base_power_VA/self.txr_power_VA
    
    @property
    def wbase(self):
        return 2*np.pi*self.base_frequency_Hz   
    
    def post_system_init(self, system):
        self.bus_id = next((n for n in system.bus if n.name == self.bus)).id
    
    def _load_power_flow_solution(self, power_flow_instance):
        sol = power_flow_instance.generators.loc[f"{self.type}_{self.id}"]
        self.pf  = PowerFlowVariables(p_bus = sol.p.item(),
                                        q_bus = sol.q.item(),
                                        vmag_bus = sol.bus_vmag.item(),
                                        vphase_bus = sol.bus_vphase.item())

    def _build_small_signal_model(self):
        
        # Current PI controller
        kp_cc, ki_cc = self.kp_cc_pu, self.ki_cc_puHz
        pi_cc_d, pi_cc_q = self.emt_init.pi_cc_d, self.emt_init.pi_cc_q

        pi_controller = StateSpaceModel( A = np.zeros((2,2)), 
                                          B = ki_cc*np.hstack((np.eye(2), -np.eye(2))),
                                          C = np.eye(2),
                                          D = kp_cc*np.hstack((np.eye(2), -np.eye(2))),
                                          u = DynamicalVariables(name=['i_bus_d_ref', 'i_bus_q_ref', 'i_bus_d', 'i_bus_q']), 
                                          y = DynamicalVariables(name=['e_d', 'e_q']),
                                          x = DynamicalVariables(   name=['pi_cc_d', 'pi_cc_q'],
                                                                    init= [pi_cc_d, pi_cc_q]) )
        
        # L filter
        rf_t = self.rf_pu + self.txr_r
        xf_t = self.xf_pu + self.txr_x
        wb = self.wbase
        i_bus_d, i_bus_q = self.emt_init.i_bus_d, self.emt_init.i_bus_q

        l_filter = StateSpaceModel( A = wb*np.array([[-rf_t/xf_t,  1], 
                                                       [-1    ,  -rf_t/xf_t]]),
                                      B = wb*np.array([[ 1/xf_t ,  0   ,  -1/xf_t  ,  0,    -i_bus_q] ,
                                                       [0,      1/xf_t,       0  , -1/xf_t,  i_bus_d]]),
                                      C = np.eye(2),
                                      D = np.zeros((2,5)),
                                      u = DynamicalVariables(name=['v_vsc_d', 'v_vsc_q', 'v_bus_d', 'v_bus_q', 'w']), 
                                      y = DynamicalVariables(name=['i_bus_d', 'i_bus_q']),
                                      x = DynamicalVariables( name=['i_bus_d', 'i_bus_q'],
                                                              init= [i_bus_d, i_bus_q]))
        
        # Phase-locked loop
        kp_pll, ki_pll = self.kp_pll_pu, self.ki_pll_puHz
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

        # DC voltage PI controller
        kp_dc, ki_dc = self.kp_dc_pu, self.ki_dc_puHz
        i_bus_d = self.emt_init.i_bus_d

        dc_pi_controller = StateSpaceModel(   A = np.array([ [0]]),
                                              B = ki_dc*np.array([ [-1, +1] ]),
                                              C = np.array([ [1] ]),
                                              D = kp_dc*np.array([ [-1, +1] ]),
                                              u = DynamicalVariables(name=['v_dc_ref', 'v_dc']),
                                              y = DynamicalVariables(name=['i_bus_d_ref']),
                                              x = DynamicalVariables(name=['pi_dc'],
                                                                     init = [i_bus_d] )  )
        
        # DC circuit
        r_dc, c_dc = self.r_dc_pu, self.c_dc_pu
        v_dc = self.emt_init.v_dc
        dc_circuit = StateSpaceModel(   A = -wb*2*1/(c_dc*r_dc)*np.eye(1),
                                        B = wb*2*1/c_dc*np.array([ [1, -1] ] ),
                                        C = np.eye(1),
                                        D = np.array([ [0 , 0] ] ),
                                        u = DynamicalVariables(name = ['i_dc_src', 'i_out']),
                                        y = DynamicalVariables(name = ['v_dc']),
                                        x = DynamicalVariables(name = ['v_dc'],
                                                               init = [v_dc] ) )

        
        # Construction of CCM matrices
        v_bus_d = self.emt_init.v_bus_d
        v_bus_q = self.emt_init.v_bus_q
        i_out = self.emt_init.i_out

        a =  pi_cc_d + beta*v_bus_d
        b =  pi_cc_q + beta*v_bus_q
        c =  beta*i_bus_d*cosphi + beta*i_bus_q*(-sinphi)
        d =  beta*i_bus_d*sinphi + beta*i_bus_q*cosphi
        
        e = -sinphi*i_bus_d - cosphi*i_bus_q
        f = cosphi*i_bus_d - sinphi*i_bus_q

        Fccm = np.vstack( ( [0, 0, 0, 0, 0, 0, 1, 0],
                             np.zeros((8,)),
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, -xf_t, 0, 0, 0, 0],
                             [0, 1, xf_t, 0, -beta*vmag_bus, 0, 0, 0],
                             np.zeros((8,)),
                             [0, 0, 0, 0, -vmag_bus, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             np.zeros((3,8)),
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             np.zeros((8,)),
                             1/v_dc*np.array([i_bus_d, i_bus_q, a, b, -beta*i_bus_q*vmag_bus, 0, 0, -i_out]) ) )
        
        Gccm = np.vstack(( [0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           np.zeros((2,5)),
                           [0, 0, 0, beta*cosphi, beta*sinphi],
                           [0, 0, 0, -beta*sinphi, beta*cosphi],
                           [0, 0, 0, cosphi, sinphi],
                           [0, 0, 0, -sinphi, cosphi],
                           np.zeros((5,)),
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0],
                           np.zeros((5, )),
                           [0, 0, 1, 0, 0],
                           1/v_dc*np.array([0, 0, 0, c, d])))
        
        Hccm = np.array([[0, 0, cosphi, -sinphi, e, 0, 0, 0],
                         [0, 0,  sinphi, cosphi, f, 0, 0, 0]])
        
        Lccm = np.zeros((2,5))

        components = [pi_controller, l_filter, pll, dc_pi_controller, dc_circuit]
        connections = [Fccm, Gccm, Hccm, Lccm]

        # Inputs and outputs
        i_src = self.i_src_pu
        v_bus_D, v_bus_Q= self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        u = DynamicalVariables(
                                    name=['v_dc_ref', 'i_bus_q_ref', 'i_src', 'v_bus_D', 'v_bus_Q'],
                                    type=["device", "device", "device", "grid", "grid"],
                                    init=[v_dc, i_bus_q, i_src, v_bus_D, v_bus_Q])
        
        i_bus_D, i_bus_Q= self.emt_init.i_bus_D, self.emt_init.i_bus_Q
        y = DynamicalVariables(
                                    name=['i_bus_D', 'i_bus_Q'],
                                    init=[i_bus_D, i_bus_Q])

        ssm = StateSpaceModel.from_interconnected(components, connections, u, y, component_label=f"{self.type}_{self.id}")

        self.ssm = ssm        
        
        
    def _calculate_emt_initial_conditions(self):
        
        # Extract power flow solution
        vmag_bus = self.pf.vmag_bus
        vphase_bus = self.pf.vphase_bus
        p_bus = self.pf.p_bus
        q_bus = self.pf.q_bus

        i_src = self.i_src_pu
        r_dc = self.r_dc_pu

        # Voltage in the end of the filter
        v_bus_DQ = vmag_bus*np.exp(vphase_bus*np.pi/180*1j)
        angle_ref = np.angle(v_bus_DQ, deg=True)

        # Current sent from the end of the filter
        i_bus_DQ = (p_bus - q_bus*1j)/np.conjugate(v_bus_DQ)

        # Voltage at the terminals of the VSC
        v_vsc_DQ = v_bus_DQ + (self.rf_pu + self.txr_r + (self.xf_pu + self.txr_x)*1j)*i_bus_DQ
    
        # We refer the voltage and currents to the synchronous frames of the
        # inverter 
        v_vsc_dq = v_vsc_DQ*np.exp(-angle_ref*np.pi/180*1j) 

        v_bus_dq = v_bus_DQ*np.exp(-angle_ref*np.pi/180*1j) 

        i_bus_dq = i_bus_DQ*np.exp(-angle_ref*np.pi/180*1j) 

        # Initial conditions for the integral controllers
        pi_cc_dq = v_vsc_dq - 1j*(self.xf_pu + self.txr_x)*i_bus_dq - self.beta*v_bus_dq

        # Initial condition for DC-side circuit
        p_vsc = (v_vsc_dq*np.conjugate(i_bus_dq)).real
        v_dc = (i_src + (i_src**2 - 4*(1/r_dc)*p_vsc)**0.5)/(2/r_dc)
        i_out = p_vsc/v_dc

        self.emt_init = InitialConditionsEMT(    vmag_bus = vmag_bus,
                                                        vphase_bus = vphase_bus,
                                                        p_bus = p_bus,
                                                        q_bus = q_bus,
                                                        angle_ref=angle_ref,
                                                        pi_cc_d= pi_cc_dq.real,
                                                        pi_cc_q= pi_cc_dq.imag,
                                                        v_vsc_d = v_vsc_dq.real,
                                                        v_vsc_q = v_vsc_dq.imag,
                                                        i_bus_d = i_bus_dq.real,
                                                        i_bus_q = i_bus_dq.imag, 
                                                        v_dc= v_dc,
                                                        i_out= i_out,
                                                        i_bus_D = i_bus_DQ.real,
                                                        i_bus_Q = i_bus_DQ.imag,
                                                        v_bus_D = v_bus_DQ.real,
                                                        v_bus_Q = v_bus_DQ.imag,
                                                        v_bus_d = v_bus_dq.real,
                                                        v_bus_q = v_bus_dq.imag,
                                                        v_vsc_mag = abs(v_vsc_DQ),
                                                        v_vsc_DQ_phase = np.angle(v_vsc_DQ, deg=True))
        
