"""
(In progress)

This module contains the GFMI generator that includes:
- Virtual inertia control
- Droop control for reactive power
- L filter
- Voltage magnitude controller
- DC voltage controller
- DC-side circuit

"""
# ----------------------
# Import python packages
# ----------------------
import numpy as np
from typing import NamedTuple, Optional, ClassVar
from dataclasses import dataclass, field

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
@dataclass(slots=True)
class GFMId:
    id: int = field(default=-1, init=False)
    bus: str
    p_min: float
    p_max: float
    q_min: float
    q_max: float
    base_power_VA: float
    base_voltage_V: float
    base_frequency_Hz: float
    rf1_pu: float
    xf1_pu: float
    txr_power_VA: float
    txr_voltage1_V: float
    txr_voltage2_V: float
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
    i_src_pu: float
    r_dc_pu: float
    c_dc_pu: float 
    kp_dc_pu: float
    ki_dc_puHz: float
    bus_id: int = None
    name: str = field(default_factory=str)
    type: str = "gfmi_d"
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
    
    def post_system_init(self, system):
        self.bus_id = next((n for n in system.bus if n.name == self.bus)).id

    def _load_power_flow_solution(self, power_flow_instance):
        sol = power_flow_instance.generators.loc[f"{self.type}_{self.id}"]
        self.pf = PowerFlowVariables(
            p_bus=sol.p.item(),
            q_bus=sol.q.item(),
            vmag_bus=sol.bus_vmag.item(),
            vphase_bus=sol.bus_vphase.item(),
        )
    
    def _build_small_signal_model(self):
    
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
