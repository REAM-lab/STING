import pandas as pd
import numpy as np
from scipy.linalg import block_diag

from dataclasses import dataclass, field
from sting.utils import linear_systems_tools
from typing import NamedTuple
from sting.utils.linear_systems_tools import State_space_model

class Power_flow_variables(NamedTuple):
    p_bus: float = 0
    q_bus: float = 0
    vmag_bus: float = 0
    vphase_bus: float = 0

class EMT_initial_conditions(NamedTuple):
    vmag_bus: float = 0
    vphase_bus: float = 0
    p_bus: float = 0
    q_bus: float = 0
    i_vsc_d: float = 0
    i_vsc_q: float = 0
    i_bus_d: float = 0
    i_bus_q: float = 0
    v_lcl_sh_d: float = 0
    v_lcl_sh_q: float = 0
    ref_angle: float = 0
    x: np.ndarray = 0
    u: np.ndarray = 0
    y: np.ndarray = 0

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
    cf: float
    rshf: float
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
    pf: Power_flow_variables = field(default_factory=Power_flow_variables)
    emt_init_cond: EMT_initial_conditions = field(default_factory=EMT_initial_conditions)
    ssm: State_space_model = field(default_factory=State_space_model)

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
        i_lcl_sh_DQ =  v_lcl_sh_DQ*(-self.cf*1j) + v_lcl_sh_DQ/self.rshf

        # Current sent from the beginning of the LCL filter
        i_vsc_DQ = i_bus_DQ + i_lcl_sh_DQ
        v_vsc_DQ = v_lcl_sh_DQ + (self.rf1 + self.lf1*1j)*i_vsc_DQ

        # We refer the voltage and currents to the synchronous frames of the
        # inverter 

        v_vsc_dq = v_vsc_DQ*np.exp(-ref_angle*np.pi/180*1j) 
        v_vsc_d, v_vsc_q = np.real(v_vsc_dq), np.imag(v_vsc_dq)

        i_vsc_dq = i_vsc_DQ*np.exp(-ref_angle*np.pi/180*1j) 
        i_vsc_d, i_vsc_q = np.real(i_vsc_dq), np.imag(i_vsc_dq)

        v_bus_dq = v_bus_DQ*np.exp(-ref_angle*np.pi/180*1j) 
        v_bus_d, v_bus_q  = np.real(v_bus_dq), np.imag(v_bus_dq)

        i_bus_dq = i_bus_DQ*np.exp(-ref_angle*np.pi/180*1j) 
        i_bus_d, i_bus_q = np.real(i_bus_dq), np.imag(i_bus_dq)

        v_lcl_sh_dq = v_lcl_sh_DQ*np.exp(-ref_angle*np.pi/180*1j); 
        v_lcl_sh_d, v_lcl_sh_q = np.real(v_lcl_sh_dq), np.imag(v_lcl_sh_dq)

        # Initial conditions for the integral controllers
        ki_cc_d = v_vsc_d + (self.lf1 + self.lf2)*i_bus_q - self.beta*v_bus_d
        ki_cc_q = v_vsc_q - (self.lf1 + self.lf2)*i_bus_d - self.beta*v_bus_q

        x = np.array([  ki_cc_d/self.ki_cc , 
                        ki_cc_q/self.ki_cc  ,
                        i_vsc_d ,
                        i_vsc_q ,
                        i_bus_d ,
                        i_bus_q ,
                        v_lcl_sh_d ,
                        v_lcl_sh_q ,
                        0 ,
                        ref_angle*np.pi/180 ])


        u = np.array([  i_bus_d , 
                        i_bus_q , 
                        v_bus_d  ,
                        v_bus_q , 
                                ])

        y = np.array([  i_bus_d , 
                        i_bus_q ])
        
        local_vars = locals()
        data = {field: local_vars[field] for field in EMT_initial_conditions._fields}

        self.emt_init_cond = EMT_initial_conditions(**data)

    def _build_small_signal_model(self):
              
        PIcontroller = State_space_model( A = np.zeros((2,2)), 
                                          B = np.hstack((np.eye(2), -np.eye(2))),
                                          C = self.ki_cc*np.eye(2),
                                          D = self.kp_cc*np.hstack((np.eye(2), -np.eye(2))), 
                                          states= ["IntCurrController_d", "IntCurrController_q"])



        rf1 = self.rf1
        lf1 = self.lf1
        rf2 = self.rf2
        lf2 = self.lf2
        rshf = self.rshf
        cf = self.cf
        wb = self.wbase
        I1c_q = self.emt_init_cond.i_vsc_q
        I1c_d = self.emt_init_cond.i_vsc_d
        I2c_q = self.emt_init_cond.i_bus_q
        I2c_d = self.emt_init_cond.i_bus_d
        V3c_q = self.emt_init_cond.v_lcl_sh_q
        V3c_d = self.emt_init_cond.v_lcl_sh_d
            
        LCLfilter = State_space_model(
                    A = wb*np.array([[-rf1/lf1 ,   1     ,      0   ,        0      ,     -1/lf1 ,     0],
                                        [-1   ,       -rf1/lf1  ,  0    ,       0      ,     0   ,        -1/lf1],
                                        [0    ,       0     ,      -rf2/lf2  ,  1     ,      1/lf2   ,    0],
                                        [0    ,       0     ,      -1     ,     -rf2/lf2 ,   0   ,        1/lf2],
                                        [1/cf   ,       0    ,       -1/cf  ,     0      ,       -1/(rshf*cf)  ,      1],
                                        [0     ,      1/cf     ,      0    ,       -1/cf  ,      -1    ,  -1/(rshf*cf)]]),
                    B = wb*np.array([[   1/lf1   ,    0     ,      0    ,       0    ,      I1c_q],
                                         [0     ,      1/lf1    ,   0    ,       0      ,      -I1c_d],
                                         [0     ,      0    ,       -1/lf2  ,    0     ,          I2c_q],
                                         [0     ,      0    ,       0    ,       -1/lf2    ,     -I2c_d],
                                         [0     ,      0    ,       0     ,      0      ,         V3c_q],
                                         [0     ,      0    ,       0     ,      0      ,        -V3c_d]]),
                    C = np.array([[0     ,      0     ,      1     ,      0     ,      0   ,    0],
                                       [0     ,      0     ,      0     ,      1     ,      0   ,    0]]),
                    D = np.zeros((2,5)),
                    states=["LCLcurrent1_d", "LCLcurrent1_q", "LCLcurrent2_d", "LCLcurrent2_q", "LCLvoltcapacitor_d", "LCLvoltcapacitor_q"]
        )

        # PLL small-signal state-space representation

        kp_pll = self.kp_pll
        ki_pll = self.ki_pll
        V2mag = self.emt_init_cond.vmag_bus
        beta = self.beta
        sinphase0 = np.sin(self.emt_init_cond.ref_angle*np.pi/180)
        cosphase0 = np.cos(self.emt_init_cond.ref_angle*np.pi/180)

        PLL = State_space_model(
            A = np.array([[  0         ,  -V2mag],
                                 [wb*ki_pll ,  -wb*V2mag*kp_pll]]),
            B = np.array([[  -sinphase0      ,        +cosphase0],
                                  [-wb*kp_pll*sinphase0  ,  wb*kp_pll*cosphase0]]),
            C = np.array([[  0  , 1],
                                 [1*ki_pll , -1*V2mag*kp_pll]]),
            D = np.array([[0          ,           0],
                                [-1*kp_pll*sinphase0 ,  1*kp_pll*cosphase0]]),
            states=["IntPLL", "PhasePLL"]) 


        Fccm = np.vstack( ( np.zeros((1,6)) ,# i2ref_d
                                np.zeros((1,6)) , # i2ref_q 
                                np.hstack((np.zeros((2,2)), np.eye(2) ,np.zeros((2,2)))), # i2c_dq
                                [1, 0,  0  ,       -(lf1+lf2),  0     ,     0], # v1c_d
                                [0 , 1 , (lf1+lf2), 0    ,      -beta*V2mag , 0], # v1c_q
                                np.zeros((1,6)) , # v2c_d
                                np.append( np.zeros((1,4)) , [-V2mag,  0] ), # v2c_q
                                np.append( np.zeros((1,5)) , [1] ), # w
                                np.zeros((2,6)) )) # v2c_dq

        Gccm = np.vstack(( [ 1, 0,  0, 0], # i2ref_d
                                  [0,  1, 0, 0], # i2ref_q
                                  np.zeros((2,4)), # i2c_dq
                                  [0, 0, beta*cosphase0 ,   beta*sinphase0],  # v1c_d
                                  [0, 0, -beta*sinphase0,    beta*cosphase0], # v1c_q
                                  [0, 0, cosphase0 ,sinphase0], # v2c_d
                                  [0, 0,  -sinphase0 ,cosphase0], # v2c_q
                                  np.zeros((1,4)), # w
                                  np.hstack( (np.zeros((2,2)), np.eye(2) ) ) ) ) # v2_dq ;  
  
        Hccm = np.vstack(( [ 0, 0 ,cosphase0 ,   -sinphase0, -sinphase0*I2c_d-cosphase0*I2c_q, 0],
                               [0, 0, sinphase0 , cosphase0  , cosphase0*I2c_d-sinphase0*I2c_q, 0] ))

        Lccm = np.zeros((2,4))

        ssm = linear_systems_tools.connect_models_via_CCM(Fccm, Gccm, Hccm, Lccm, [PIcontroller, LCLfilter, PLL])

        states = PIcontroller.states + LCLfilter.states + PLL.states

        inputs = ["i2d_ref", "i2q_ref", "v2_d", "v2_q"]

        device_side_inputs = [  "i2d_ref", "i2q_ref"]

        grid_side_inputs = ["v2_d", "v2_q"]

        outputs = [ "i2_d", "i2_q"]

        self.ssm = State_space_model(A = ssm.A,
                                     B = ssm.B,
                                     C = ssm.C,
                                     D = ssm.D,
                                     states= states,
                                     device_side_inputs=device_side_inputs,
                                     grid_side_inputs=grid_side_inputs,
                                     outputs=outputs)