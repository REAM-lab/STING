"""
This module implements a GFLI that incorporates: 
- LCL filter: Two Series RL branches (one branch is the transformer) and one Parallel RC shunt. 
- Current controller: A dq-based frame PI controller
- PLL: It that tracks the phase of the grid voltage. The PLL has a proportional and integral gain.
"""
# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple, Optional
import scipy.linalg 
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

# -----------
# Main class
# -----------
@dataclass(slots=True, kw_only=True, eq=False)
class GFLIa(Generator):
    v_dc_pu: float
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
    x_pll_rescale: np.ndarray = field(default_factory=lambda: np.array([[100, 0], [0, 1]])) 
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
                        C = np.array(   [[0     ,      0     ,      1     ,      0     ,      0   ,    0],
                                         [0     ,      0     ,      0     ,      1     ,      0   ,    0]]),
                        D = np.zeros((2,5)),
                        x = DynamicalVariables(name=["i_vsc_d", "i_vsc_q", "i_bus_d", "i_bus_q", "v_lcl_sh_d", "v_lcl_sh_q"],
                                               init=[i_vsc_d, i_vsc_q, i_bus_d, i_bus_q, v_lcl_sh_d, v_lcl_sh_q]),
                        u = DynamicalVariables(name=['v_vsc_d', 'v_vsc_q', 'v_bus_d', 'v_bus_q', 'w']),
                        y = DynamicalVariables(name=["i_bus_d", "i_bus_q"]))

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

        # Construction of CCM matrices
        Fccm = np.vstack( (     np.zeros((6, )) ,# i2ref_d
                                np.zeros((6, )) , # i2ref_q 
                                np.hstack((np.zeros((2,2)), np.eye(2) ,np.zeros((2,2)))), # i2c_dq
                                [1, 0   ,  0        ,   -(lf1+lf2)  ,  0              , 0], # v1c_d
                                [0, 1   , (lf1+lf2) ,   0           ,  -beta*vmag_bus , 0], # v1c_q
                                np.zeros((6, )) , # v2c_d
                                np.append( np.zeros((1,4)) , [-vmag_bus,  0] ), # v2c_q
                                np.append( np.zeros((1,5)) , [1] ), # w
                                np.zeros((2,6)) )) # v2c_dq

        Gccm = np.vstack((      [1,  0,  0, 0], # i2ref_d
                                [0,  1, 0, 0], # i2ref_q
                                np.zeros((2,4)), # i2c_dq
                                [0, 0, beta*cosphi ,    beta*sinphi],  # v1c_d
                                [0, 0, -beta*sinphi,    beta*cosphi], # v1c_q
                                [0, 0, cosphi   ,sinphi], # v2c_d
                                [0, 0,  -sinphi ,cosphi], # v2c_q
                                np.zeros((4, )), # w
                                np.hstack( (np.zeros((2,2)), np.eye(2) ) ) ) ) # v2_dq ;  
  
        Hccm = np.vstack(( [ 0, 0 ,cosphi , -sinphi, -sinphi*i_bus_d-cosphi*i_bus_q, 0],
                            [0, 0, sinphi , cosphi , cosphi*i_bus_d-sinphi*i_bus_q, 0] ))
        
        Lccm = np.zeros((2, 4))

        components = [pi_controller, lcl_filter, pll]
        connections = [Fccm, Gccm, Hccm, Lccm]

        # Inputs and outputs
        v_bus_D, v_bus_Q= self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        u = DynamicalVariables(
                                name=["i_bus_d_ref", "i_bus_q_ref", "v_bus_D", "v_bus_Q"],
                                type=["device", "device", "grid", "grid"],
                                init=[i_bus_d, i_bus_q, v_bus_D, v_bus_Q])

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
            v_vsc_DQ_phase = np.angle(v_vsc_DQ, deg=True)
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

        v_vsc_d = self.emt_init.v_vsc_d

        x = DynamicalVariables(
            name = ['pi_cc_d', 'pi_cc_q', 'int_pll', 'phase_pll',"i_vsc_a", "i_vsc_b","i_vsc_c", "v_sh_a", "v_sh_b","v_sh_c", "i_bus_a", "i_bus_b", "i_bus_c"],
            component = f"{self.type_}_{self.id}",
            init = [pi_cc_d, pi_cc_q, 1.0, angle_ref * np.pi/180, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c]
        )

        # Inputs 
        # ------
        
        # Initial conditions 
        v_bus_D, v_bus_Q = self.emt_init.v_bus_D, self.emt_init.v_bus_Q
        v_bus_a, v_bus_b, v_bus_c = dq02abc(v_bus_D, v_bus_Q, 0, 0)

        u = DynamicalVariables(
                            name=["i_bus_d_ref", "i_bus_q_ref", "v_bus_a", "v_bus_b", "v_bus_c"],
                            component=f"{self.type_}_{self.id}",
                            type=["device", "device", "grid", "grid", "grid"],
                            init=[i_bus_d, i_bus_q, v_bus_a, v_bus_b, v_bus_c]
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
        pi_cc_d, pi_cc_q, theta_pll, gamma_pll, i_vsc_a, i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c = self.variables_emt.x.value 
        
        # Get input values (external inputs)
        i_bus_d_ref, i_bus_q_ref, v_bus_a, v_bus_b, v_bus_c = self.variables_emt.u.value

        # convert relevant quantities to dq 
        v_bus_d, v_bus_q, _ = abc2dq0(v_bus_a, v_bus_b, v_bus_c, theta_pll) # for power controller 
        i_bus_d, i_bus_q, _ = abc2dq0(i_bus_a, i_bus_b, i_bus_c, theta_pll) # for power controller 
      
        # Update algebraic states
        e_d = self.ki_cc_puHz * pi_cc_d + self.kp_cc_pu * (i_bus_d_ref - i_bus_d) 
        e_q = self.ki_cc_puHz * pi_cc_q + self.kp_cc_pu * (i_bus_q_ref - i_bus_q)
        
        v_vsc_d = e_d + self.beta * v_bus_d - (self.xf1_pu + self.xf2_pu) * i_bus_q 
        v_vsc_q = e_q + self.beta * v_bus_q + (self.xf1_pu + self.xf2_pu) * i_bus_q  

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
            d_pi_cc_d = i_bus_d_ref - i_bus_d
            d_pi_cc_q = i_bus_q_ref - i_bus_q
            
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

            # Define measured capacitor voltages at timepoint "t"
            v_bus_a, v_bus_b, v_bus_c = internal_inputs

            # Transform to dq frame using PLL angle
            v_q = 2/3*(-np.sin(theta_pll)*v_bus_a -np.sin(theta_pll- 2*np.pi/3)*v_bus_b - np.sin(theta_pll + 2*np.pi/3)*v_bus_c )

            # Define ODEs that describe the dynamics of the PLL
            d_theta_pll = kp_pll * v_q + gamma_pll + w_base
            d_gamma_pll = ki_pll * v_q

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
          
        d_pi_cc_d, d_pi_cc_q = current_controller_dynamics([pi_cc_d, pi_cc_q], [i_bus_d_ref, i_bus_q_ref, i_bus_d, i_bus_q])

        d_theta_pll, d_gamma_pll = pll_dynamics([theta_pll, gamma_pll], [v_bus_a, v_bus_b, v_bus_c])

        di_vsc_a, di_vsc_b, di_vsc_c, dv_sh_a, dv_sh_b, dv_sh_c, di_bus_a, di_bus_b, di_bus_c = lcl_filter_dynamics([i_vsc_a , i_vsc_b, i_vsc_c, v_sh_a, v_sh_b, v_sh_c, i_bus_a, i_bus_b, i_bus_c], [v_vsc_a, v_vsc_b, v_vsc_c, v_bus_a, v_bus_b, v_bus_c])

        return [d_pi_cc_d, d_pi_cc_q, d_theta_pll, d_gamma_pll, di_vsc_a, di_vsc_b, di_vsc_c, dv_sh_a, dv_sh_b, dv_sh_c, di_bus_a, di_bus_b, di_bus_c]