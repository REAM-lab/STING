from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from sting.utils.dynamical_systems import DynamicalVariables, QuadraticBilinearModel, StateSpaceModel


# ------------------------------------
# Sub-classes
# ------------------------------------
class InitialConditionsEMT(NamedTuple):
    x_l: float
    x_a: float
    x_e: float
    x_f: float
    v_ref: float

# ------------------------------------
# Main class
# ------------------------------------
@dataclass(slots=True)
class ExcitationSystem4A:
    """
    Models a 4th order excitation system consisting of a lead-lag compensator, 
    an amplifier, an exciter, and a stabilizer/damping sensor.
               ┌───┐    
    v_ref ────▶│ + │
    v_mag ────▶│ - │     ┌─────────────────┐     ┌─────────────┐     ┌─────────────┐
    v_stab ───▶│ + │────▶│ 1+s*tc / 1+s*tb │────▶│ ka / 1+s*ta │────▶│ 1 / ke+s*ta │───┬───▶ Δv_fd
         ┌────▶│ - │     └─────────────────┘     └─────────────┘     └─────────────┘   │
         │     └───┘                 ┌─────────────┐                                   │
         └───────────────────────────│ kf / 1+s*tf │───────────────────────────────────┘
                                     └─────────────┘
    
    This model is closely related to the DC1A/DC2A excitation systems without regulator output limits.
    See Kundur (page 363) or [1] for more details.
    
    [1] “Recommended Practice for Excitation System Models for Power System Stability Studies,” IEEE 
        Standard 421.5-1992, August, 1992.

    Parameters
    ----------

    Dynamics
    --------
    The dynamics governing the excitation system are given by the following equations:
             
             u_l = v_ref + v_stab - v_mag - v_f
        d/dt x_l = -(1/tb) * x_l + (tc/tb - 1) * u_l
             y_l = -(1/tb) * x_l + (tc/tb) * u_l
        d/dt x_a = (1/ta) * (ka * y_l - x_a)
        d/dt x_e = (1/te) * (x_a - ke * x_e)
        d/dt x_f = (1/tf) * (kf * x_e - x_f)
           Δv_fd = (1/tf) * (kf * x_e - x_f)
        
    """

    tb_s: float
    tc_s: float
    ka_pu: float
    ta_s: float
    te_s: float
    ke_pu: float
    tf_s: float
    kf_pu: float

    A: np.ndarray = None
    B: np.ndarray = None
    C: np.ndarray = None

    emt_init: InitialConditionsEMT = field(init=False)

    def __post_init__(self):
        # Construct each component model
        lead_lag = StateSpaceModel(
            A=np.array([[-1/self.tb_s]]),
            B=np.array([[self.tc_s/self.tb_s -1]]),
            C=np.array([[-1/self.tb_s]]),
            D=np.array([[self.tc_s/self.tb_s]])
        )
        amplifier = StateSpaceModel(
            A=np.array([[-1/self.ta_s]]),
            B=np.array([[self.ka_pu/self.ta_s]]),
            C=np.array([[1]]),
            D=np.array([[0]])
        )
        exciter = StateSpaceModel(
            A=np.array([[-self.ke_pu/self.te_s]]),
            B=np.array([[1/self.te_s]]),
            C=np.array([[1]]),
            D=np.array([[0]])
        )
        stabilizer = StateSpaceModel(
            A=np.array([[-1/self.tf_s]]),
            B=np.array([[self.kf_pu/self.tf_s]]),
            C=np.array([[-1/self.tf_s]]),
            D=np.array([[self.kf_pu/self.tf_s]]),
        )
        # Interconnection matrices
        F = np.array([
            [0, 0, 0,-1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ])
        G = np.array([
            [1,-1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        H = np.array([[0, 0, 1, 0]])
        L = np.zeros((1, 3))

        model = StateSpaceModel.from_interconnected(
            components=[lead_lag, amplifier, exciter, stabilizer],
            connections=[F,G,H,L],
            u=DynamicalVariables(name=['v_ref', 'v_mag', 'v_stab']),
            y=DynamicalVariables(name=['v_fd'])
        )
        # Save the state-space matrices
        self.A = model.A
        self.B = model.B
        self.C = model.C

    def get_steady_state(self, v_ref: float, v_mag: float, v_stab: float):
        """
        0 = A*x0 + B*u0
        x0 = -invA * B * u0
        """
        u0 = np.array([v_ref, v_mag, v_stab])
        x0 = np.linalg.solve(self.A, -self.B@u0)

        self.emt_init = InitialConditionsEMT(*x0, v_ref=v_ref)

    def get_small_signal_model(self, x_l: float, x_a: float, x_e: float, x_f: float, v_ref: float, v_mag: float, v_stab: float):
        ssm = StateSpaceModel(
            A=self.A,
            B=self.B,
            C=self.C,
            D=np.zeros((1, 3)),
            x=DynamicalVariables(
                name=['x_l', 'x_a', 'x_e', 'x_f'],
                init=[x_l, x_a, x_e, x_f]),
            u=DynamicalVariables(
                name=['v_ref', 'v_mag', 'v_stab'],
                init=[v_ref, v_mag, v_stab]),
            y=DynamicalVariables(
                name=['v_fd'],
                init=[0])
        )
        return ssm
        

    def get_quadratic_bilinear_model(self,  x_l: float, x_a: float, x_e: float, x_f: float, v_ref: float, v_mag: float, v_stab: float):
        ssm = self.get_small_signal_model(x_l, x_a, x_e, x_f, v_ref, v_mag, v_stab)
        return ssm.to_quadratic_bilinear()

    def get_derivatives_step_emt_dq0(self, x_l: float, x_a: float, x_e: float, x_f: float, v_ref: float, v_mag: float, v_stab: float) -> float:
        x = np.array([x_l, x_a, x_e, x_f])
        u = np.array([v_ref, v_mag, v_stab])

        return list(self.A@x + self.B@u)