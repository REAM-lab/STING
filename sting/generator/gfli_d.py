"""
This module implements a GFLI that incorporates: 
- LCL filter: Two Series RL branches (one branch is the transformer) and one Parallel RC shunt. 
- Current controller: A dq-based frame PI controller
- PLL: It that tracks the phase of the grid voltage. The PLL has a proportional and integral gain.
- DC-DC converter for controlling Vdc
- DC side circuit with load modeled as current source 
"""
# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, Optional
from sting.generator.core import Generator

# ------------------
# Import sting code
# ------------------
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables
from sting.modules.simulation_emt.utils import VariablesEMT
from sting.utils.transformations import dq02abc, abc2dq0

# -----------
# Sub-classes
# -----------
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
    v_dc: float # DC bus voltage 
    d: float # DC/DC converter duty cycle 
    i_dc: float # current into/out of inverter 
    i_L: float # DC/DC converter inductor current 
    x_1: float # DC voltage regulator integrator 
    x_2: float # DC current regulator integrator 
    i_load_ref: float 
    p_vsc: float 

# -----------
# Main class
# -----------
@dataclass(slots=True, kw_only=True, eq=False)
class GFLId(Generator):
    rf1_pu: float
    xf1_pu: float
    csh_pu: float
    rsh_pu: float
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
    kp_v_dc: float # DC voltage regulator P gain 
    ki_v_dc: float # DC voltage regulator I gain [1/s] 
    kp_i_L: float # DC current regulator P gain 
    ki_i_L: float # DC current regulator I gain [1/s]
    l_dc: float # DC/DC converter inductance, [pu]
    c_dc: float # DC link capacitance, [pu]
    v_dc_ref: float # DC bus reference voltage, [pu]
    v_s: float # DC voltage source voltage, [pu]
    Ti_L: float # measurement filter time constants [s]
    Tv_dc: float 
    Ti_dc: float 
    Kff_idc: float 
    Kff_iload: float 
    Ti_load: float # for DC/DC controller - measurement filter 
    Tload: float # time constant for actuation of load current change 
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
        rf1, lf1, rf2, lf2, rsh, csh = self.rf1_pu, self.xf1_pu, self.rf2_pu, self.xf2_pu, self.rsh_pu, self.csh_pu
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
                        C = np.hstack((np.eye(4), np.zeros((4,2)))), 
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

        # DC-DC controller 
        # Parameters
        l_dc, c_dc, TiL, Tvdc, Tidc, Kp_vdc, Ki_vdc, Kp_iL, Ki_iL, Kff_idc, Kff_iload, Ti_load, Tload = self.l_dc, self.c_dc, self.Ti_L, self.Tv_dc, self.Ti_dc, self.kp_v_dc, self.ki_v_dc, self.kp_i_L, self.ki_i_L, self.Kff_idc, self.Kff_iload, self.Ti_load, self.Tload 
        
        # Initial conditions
        v_dc, duty_cycle, i_dc, i_L, x1, x2, i_load = self.emt_init.v_dc, self.emt_init.d, self.emt_init.i_dc, self.emt_init.i_L, self.emt_init.x_1, self.emt_init.x_2, self.emt_init.i_load_ref
        
        
        # DC control 
        dc_dc_controller = StateSpaceModel(
            A = np.array([
                            [-1/TiL, 0, 0, 0, 0, 0],
                            [0,-1/Tvdc, 0, 0, 0, 0],
                            [0, 0, -1/Tidc, 0, 0, 0],
                            [0, 0, 0, -1/Ti_load, 0, 0],
                            [0, -Ki_vdc, 0, 0, 0, 0],
                            [-Ki_iL, -Kp_vdc*Ki_iL, Kff_idc*Ki_iL, Kff_iload*Ki_iL, Ki_iL, 0]]),
            B = np.array([
                            [1/TiL, 0, 0, 0, 0],
                            [0, 1/Tvdc, 0, 0, 0],
                            [0, 0, 1/Tidc, 0, 0],
                            [0, 0, 0, 1/Ti_load, 0],
                            [0, 0, 0, 0, Ki_vdc],
                            [0, 0, 0, 0, Ki_iL*Kp_vdc]]),
            
            C = np.array([[-Kp_iL, -Kp_iL*Kp_vdc, Kp_iL*Kff_idc, Kp_iL*Kff_iload, Kp_iL, 1]]),
            D = np.array([[0, 0, 0, 0, Kp_iL*Kp_vdc]]),
            u = DynamicalVariables(name=['i_L', 'v_dc', 'i_dc', 'i_load', 'v_dc_ref']),
            y = DynamicalVariables(name=['d']),
            x = DynamicalVariables(
                name = ['i_l_f', 'v_dc_f', 'i_dc_f', 'i_load_f','x_1', 'x_2'],
                init = [i_L, v_dc, i_dc, i_load, x1, x2]
            )
        )

        # DC circuit - includes capacitor, inductor, and load dynamics 
        dc_circuit = StateSpaceModel(
            A = wb*np.array([[0,         (duty_cycle-1)/l_dc, 0], 
                             [(1-duty_cycle)/c_dc,         0, -1/c_dc],
                             [0, 0, -1/(wb*Tload)]]),
            B = wb*np.array([[1/l_dc, v_dc/l_dc,       0,       0],
                             [0,      -i_L/c_dc, -1/c_dc, 0],
                             [0, 0, 0, 1/(wb*Tload)]]),
            C = np.eye(3), 
            D = np.zeros((3,4)),
            u = DynamicalVariables(name=['v_s', 'd', 'i_dc','i_load_ref']),
            y = DynamicalVariables(name=['i_L','v_dc', 'i_load']),
            x = DynamicalVariables(
                name = ['i_L', 'v_dc', 'iload'],
                init = [i_L, v_dc, i_load]
        )
        )
        
        # Construction of CCM matrices
        
        # dc power balance linearization coefficients 
        v_dc = self.emt_init.v_dc 
        
        b1 = self.emt_init.v_vsc_d/v_dc # i1d
        b2 = self.emt_init.i_vsc_d/v_dc #e1d
        b3 = - (self.emt_init.i_vsc_d/v_dc)*(self.xf1_pu+self.xf2_pu) #i2q
        b4 = self.emt_init.v_vsc_q/v_dc # i1q 
        b5 = self.emt_init.i_vsc_q/v_dc # e1q
        b6 = (self.emt_init.i_vsc_q/v_dc)*(self.xf1_pu+self.xf2_pu) #i2d 
        b7 = - self.emt_init.i_dc/v_dc #vdc 
        b8 = -(self.emt_init.i_vsc_q/v_dc)*self.beta*vmag_bus # phi 
        
        Fccm = np.vstack( (     np.zeros((12, )) ,# i2ref_d
                                np.zeros((12, )) , # i2ref_q 
                                np.hstack((np.zeros((2,4)), np.eye(2) ,np.zeros((2,6)))), # i2c_dq
                                [1, 0   ,  0, 0, 0        ,   -(lf1+lf2)  ,  0              , 0, 0, 0, 0, 0], # v1c_d
                                [0, 1   , 0, 0, (lf1+lf2) ,   0           ,  -beta*vmag_bus , 0, 0, 0, 0, 0], # v1c_q
                                np.zeros((12, )) , # v2c_d
                                np.append( np.zeros((1,6)) , [-vmag_bus,  0, 0, 0, 0, 0] ), # v2c_q
                                np.append( np.zeros((1,7)) , [1, 0, 0, 0, 0] ), # w
                                np.zeros((2,12)),# v2c_dq
                                np.hstack((np.zeros(9,), [1, 0, 0])), #iL
                                np.hstack((np.zeros(9,), [0, 1, 0])), #vdc
                                [b2, b5, b1, b4, b6, b3, b8, 0, 0, 0, b7, 0], #idc 
                                np.hstack((np.zeros(11,), [1])), #iload 
                                np.zeros((2,12)), #vdc_ref, v_s
                                np.hstack((np.zeros(8,), [1, 0, 0, 0])), #d
                                [b2, b5, b1, b4, b6, b3, b8, 0, 0, 0, b7, 0], #idc 
                                np.zeros(12,) # iload_ref 
                                )) 

        Gccm = np.vstack((      [1,  0,  0, 0, 0, 0, 0], # i2ref_d
                                [0,  1, 0, 0, 0, 0, 0], # i2ref_q
                                np.zeros((2,7)), # i2c_dq
                                [0, 0, 0, 0, 0, beta*cosphi ,    beta*sinphi],  # v1c_d
                                [0, 0, 0, 0, 0, -beta*sinphi,    beta*cosphi], # v1c_q
                                [0, 0, 0, 0, 0, cosphi   ,sinphi], # v2c_d
                                [0, 0, 0, 0, 0, -sinphi ,cosphi], # v2c_q
                                np.zeros((7, )), # w
                                np.hstack( (np.zeros((2,5)), np.eye(2) ) ),# v2_dq 
                                np.zeros((4,7)), #iL, vdc, idc, iload
                                [0, 0, 1, 0, 0, 0, 0], #vdc_ref
                                [0, 0, 0, 1, 0, 0, 0], #vs
                                np.zeros((2,7)), #d, idc 
                                [0, 0, 0, 0, 1, 0, 0] #i_load_ref 
                                ) ) 
  
        Hccm = np.vstack(( [ 0, 0 , 0, 0, cosphi , -sinphi, -sinphi*i_bus_d-cosphi*i_bus_q, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, sinphi , cosphi , cosphi*i_bus_d-sinphi*i_bus_q, 0, 0, 0, 0, 0] ))
        
        Lccm = np.zeros((2, 7))

        components = [pi_controller, lcl_filter, pll, dc_dc_controller, dc_circuit]
        connections = [Fccm, Gccm, Hccm, Lccm]

        # Inputs and outputs
        v_bus_D, v_bus_Q= self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        u = DynamicalVariables(
                                name=["i_bus_d_ref", "i_bus_q_ref", "v_dc_ref", "v_s", "i_load_ref", "v_bus_D", "v_bus_Q"],
                                type=["device", "device", "device", "device", "device", "grid", "grid"],
                                init=[i_bus_d, i_bus_q, v_dc, self.v_s, i_load, v_bus_D, v_bus_Q])

        i_bus_D, i_bus_Q= self.emt_init.i_bus_D, self.emt_init.i_bus_Q
        y = DynamicalVariables(
                                name=['i_bus_D', 'i_bus_Q'],
                                init=[i_bus_D, i_bus_Q])

        # Generate small-signal model
        self.ssm = StateSpaceModel.from_interconnected(components, connections, u, y, component_label=f"{self.type_}_{self.id}") 

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

        # Initial conditions for the integral controller
        pi_cc_dq = (
            v_vsc_dq - self.beta * v_bus_dq - 1j * (self.xf1_pu + self.xf2_pu) * i_bus_dq
        )
        
        # DC side 
        # DC-side initial conditions 
        v_dc = self.v_dc_ref 
        p_vsc = (v_vsc_dq*np.conjugate(i_vsc_dq)).real # power at converter terminals 
        i_dc = p_vsc/v_dc 
        i_load = -i_dc # negative sign because of how direction of idc is defined 
        duty_cycle = (v_dc - self.v_s)/v_dc 
        i_L = (i_load+i_dc)/(1-duty_cycle)
        x_1 = i_L - self.Kff_idc*i_dc - self.Kff_iload*i_load 
        x_2 = duty_cycle - self.kp_i_L*(x_1 - i_L + self.Kff_idc*i_dc + self.Kff_iload*i_load) 
        

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
            v_vsc_d=v_vsc_dq.real,
            v_vsc_q=v_vsc_dq.imag,
            v_dc=v_dc,
            p_vsc=p_vsc,
            i_dc=i_dc,
            i_load_ref=i_load,
            d = duty_cycle,
            i_L = i_L,
            x_1 = x_1,
            x_2 = x_2
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
        
        # DC side 
        v_dc, i_dc, i_L, x1, x2, i_load = self.emt_init.v_dc, self.emt_init.i_dc, self.emt_init.i_L, self.emt_init.x_1, self.emt_init.x_2, self.emt_init.i_load_ref
        

        x = DynamicalVariables(
            name = ['pi_cc_d', 'pi_cc_q', 'theta_pll', 'gamma_pll', "i_vsc_a", "i_vsc_b", "i_vsc_c", "v_sh_a", "v_sh_b","v_sh_c", "i_bus_a", "i_bus_b", "i_bus_c",'i_l_f', 'v_dc_f', 'i_dc_f', 'i_load_f','x_1', 'x_2', 'i_L', 'v_dc', 'i_load'],
            component = f"{self.type_}_{self.id}",
            init = [pi_cc_d, pi_cc_q, angle_ref * np.pi/180, 0, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c,i_L, v_dc, i_dc, i_load, x1, x2, i_L, v_dc, i_load]
        )

        # Inputs 
        # ------
        
        # Initial conditions 
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        v_bus_a, v_bus_b, v_bus_c = dq02abc(v_bus_D, v_bus_Q, 0, 0)

        u = DynamicalVariables(
                            name=["i_bus_d_ref", "i_bus_q_ref", "v_dc_ref", "v_s", "i_load_ref", "v_bus_a", "v_bus_b", "v_bus_c"],
                            component=f"{self.type_}_{self.id}",
                            type=["device", "device", "device", "device", "device", "grid", "grid", "grid"],
                            init=[i_bus_d, i_bus_q, v_dc, self.v_s, i_load, v_bus_a, v_bus_b, v_bus_c]
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
        pi_cc_d, pi_cc_q, theta_pll, gamma_pll, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, i_Lf, v_dcf, i_dcf, i_loadf, x1, x2, i_L, v_dc, i_load = self.variables_emt.x.value 
        
        # Get input values (external inputs)
        i_bus_d_ref, i_bus_q_ref, v_dc_ref, v_s, i_load_ref, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value

        # convert relevant quantities to dq 
        v_bus_d, v_bus_q, _ = abc2dq0(v_bus_a, v_bus_b, v_bus_c, theta_pll) 
        i_bus_d, i_bus_q, _ = abc2dq0(i_bus_a, i_bus_b, i_bus_c, theta_pll) 
      
        # Update algebraic states
        e_d = pi_cc_d + self.kp_cc_pu * (i_bus_d_ref - i_bus_d) 
        e_q = pi_cc_q + self.kp_cc_pu * (i_bus_q_ref - i_bus_q)
        
        v_vsc_d = e_d + self.beta * v_bus_d - (self.xf1_pu + self.xf2_pu) * i_bus_q 
        v_vsc_q = e_q + self.beta * v_bus_q + (self.xf1_pu + self.xf2_pu) * i_bus_d  

        # convert to abc to feed into filter dynamics 
        v_vsc_a, v_vsc_b, v_vsc_c = dq02abc(v_vsc_d, v_vsc_q, 0, theta_pll) 
        
        # DC/AC power balance 
        i_vsc_d, i_vsc_q, _ = abc2dq0(i_vsc_a, i_vsc_b, i_vsc_c, theta_pll)
        i_dc = (v_vsc_d*i_vsc_d + v_vsc_q*i_vsc_q)/v_dc

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
          
        def dc_side(y, internal_inputs):
            """
            DC-DC controller + circuit + load control 
            """
            # Define states
            i_Lf, v_dcf, i_dcf, i_loadf, x_1, x_2, i_L, v_dc, i_load = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8]
            
            # Inputs 
            i_dc, i_load_ref, v_dc_ref, v_s = internal_inputs
            
            # Parameters 
            l_dc, c_dc, TiL, Tvdc, Tidc, Kp_vdc, Ki_vdc, Kp_iL, Ki_iL, Kff_idc, Kff_iload, Ti_load, Tload = self.l_dc, self.c_dc, self.Ti_L, self.Tv_dc, self.Ti_dc, self.kp_v_dc, self.ki_v_dc, self.kp_i_L, self.ki_i_L, self.Kff_idc, self.Kff_iload, self.Ti_load, self.Tload 
            wb = self.wbase
            
            # ODEs 
            
            # DC-DC controller 
            d_i_Lf = (1/TiL)*(i_L - i_Lf)
            d_v_dcf = (1/Tvdc)*(v_dc - v_dcf)
            d_i_dcf = (1/Tidc)*(i_dc - i_dcf)
            d_i_load_f = (1/Ti_load)*(i_load - i_loadf)
            d_x_1 = Ki_vdc*(v_dc_ref - v_dcf)
            d_x_2 = Ki_iL*(Kp_vdc*(v_dc_ref - v_dcf) + x_1 - i_Lf + Kff_idc*i_dcf + Kff_iload*i_loadf)
            duty_cycle = Kp_iL*(Kp_vdc*(v_dc_ref - v_dcf) + x_1 - i_Lf + Kff_idc*i_dcf + Kff_iload*i_loadf) + x_2
            
            # Circuit equations  
            d_v_dc = (wb/c_dc)*(-i_dc - i_load + (1-duty_cycle)*i_L)
            d_i_L = (wb/l_dc)*(v_s - (1-duty_cycle)*v_dc)
            
            # Load control 
            d_i_load = (1/Tload)*(i_load_ref - i_load) 
            
            return [d_i_Lf, d_v_dcf, d_i_dcf, d_i_load_f, d_x_1, d_x_2, d_i_L, d_v_dc, d_i_load]

        d_cc= current_controller_dynamics([pi_cc_d, pi_cc_q], [i_bus_d_ref, i_bus_q_ref, i_bus_d, i_bus_q])

        d_pll = pll_dynamics([theta_pll, gamma_pll], v_bus_q)

        d_lcl = lcl_filter_dynamics([i_vsc_a , i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c], [v_vsc_a, v_vsc_b, v_vsc_c, v_bus_a, v_bus_b, v_bus_c])
        
        d_dc = dc_side([i_Lf, v_dcf, i_dcf, i_loadf, x1, x2, i_L, v_dc, i_load], [i_dc, i_load_ref, v_dc_ref, v_s])

        return np.hstack([d_cc, d_pll, d_lcl, d_dc])
    
    def get_output_emt(self):
        
        pi_cc_d, pi_cc_q, theta_pll, gamma_pll, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, i_Lf, v_dcf, i_dcf, i_loadf, x1, x2, i_L, v_dc, i_load = self.variables_emt.x.value 

        return [i_bus_a, i_bus_b, i_bus_c]
    
    def plot_results_emt(self):
        """
        Plot EMT simulation results
        """

        pi_cc_d, pi_cc_q, theta_pll, gamma_pll, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c, i_Lf, v_dcf, i_dcf, i_loadf, x1, x2, i_L, v_dc, i_load = self.variables_emt.x.value         
        tps = self.variables_emt.x.time

        # Transform abc to dq0
        i_vsc_d, i_vsc_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_vsc_a, i_vsc_b, i_vsc_c, theta_pll)])
        v_sh_d, v_sh_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(v_sh_a, v_sh_b, v_sh_c, theta_pll)])
        i_bus_d, i_bus_q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_bus_a, i_bus_b, i_bus_c, theta_pll)])
        
        results = DynamicalVariables(
            name=['pi_cc_d', 'pi_cc_q', 'theta_pll', 'gamma_pll', 'i_vsc_d', 'i_vsc_q', 'v_sh_d', 'v_sh_q', 'i_bus_d', 'i_bus_q', 'i_l_f', 'v_dc_f', 'i_dc_f', 'i_load_f','x_1', 'x_2', 'i_L', 'v_dc', 'i_load'],
            component=f"{self.type_}_{self.id}",
            value=[pi_cc_d, pi_cc_q, theta_pll, gamma_pll, i_vsc_d, i_vsc_q, v_sh_d, v_sh_q, i_bus_d, i_bus_q, i_Lf, v_dcf, i_dcf, i_loadf, x1, x2, i_L, v_dc, i_load],
            time=tps
        )
        return results