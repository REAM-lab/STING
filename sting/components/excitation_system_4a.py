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
    v_c   ────▶│ - │     ┌───────────────────┐     ┌───────────────┐     ┌───────────────┐
    v_s   ────▶│ + │────▶│ 1+s*t_c / 1+s*t_b │────▶│ k_a / 1+s*t_a │────▶│ 1 / k_e+s*t_a │───┬───▶ Δv_fd
         ┌────▶│ - │     └───────────────────┘     └───────────────┘     └───────────────┘   │
         │     └───┘                 ┌───────────────┐                                       │
         └───────────────────────────│ k_f / 1+s*t_f │───────────────────────────────────────┘
                                     └───────────────┘
    
    This model is closely related to the DC1A/DC2A excitation systems without regulator output limits.
    See Kundur (page 363) or [1] for more details.
    
    [1] “Recommended Practice for Excitation System Models for Power System Stability Studies,” IEEE 
        Standard 421.5-1992, August, 1992.

    Parameters
    ----------

    Dynamics
    --------
    The dynamics governing the excitation system are given by the following equations:
             
             u_l = v_ref + v_s - v_c - v_f
        d/dt x_l = -(1/t_b) * x_l + (t_c/t_b - 1) * u_l
             y_l = -(1/t_b) * x_l + (t_c/t_b) * u_l
        d/dt x_a = (1/t_a) * (k_a * y_l - x_a)
        d/dt x_e = (1/t_e) * (x_a - k_e * x_e)
        d/dt x_f = (1/t_f) * (k_f * x_e - x_f)
           Δv_fd = (1/t_f) * (k_f * x_e - x_f)
        
    """

    t_b: float
    t_c: float
    k_a: float
    t_a: float
    t_e: float
    k_e: float
    t_f: float
    k_f: float

    A: np.ndarray = None
    B: np.ndarray = None
    C: np.ndarray = None

    emt_init: InitialConditionsEMT = field(init=False)

    def __post_init__(self):
        # Construct each component model
        lead_lag = StateSpaceModel(
            A=np.array([[-1/self.t_b]]),
            B=np.array([[self.t_c/self.t_b -1]]),
            C=np.array([[-1/self.t_b]]),
            D=np.array([[self.t_c/self.t_b]])
        )
        amplifier = StateSpaceModel(
            A=np.array([[-1/self.t_a]]),
            B=np.array([[self.k_a/self.t_a]]),
            C=np.array([[1]]),
            D=np.array([[0]])
        )
        exciter = StateSpaceModel(
            A=np.array([[-self.k_e/self.t_e]]),
            B=np.array([[1/self.t_e]]),
            C=np.array([[1]]),
            D=np.array([[0]])
        )
        stabilizer = StateSpaceModel(
            A=np.array([[-1/self.t_f]]),
            B=np.array([[self.k_f/self.t_f]]),
            C=np.array([[-1/self.t_f]]),
            D=np.array([[self.k_f/self.t_f]]),
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
            u=DynamicalVariables(name=['v_ref', 'v_c', 'v_s']),
            y=DynamicalVariables(name=['v_fd'])
        )
        # Save the state-space matrices
        self.A = model.A
        self.B = model.B
        self.C = model.C

    def get_steady_state(self, v_ref: float, v_c: float, v_s: float):
        """
        0 = A*x0 + B*u0
        x0 = -invA * B * u0
        """
        u0 = np.array([v_ref, v_c, v_s])
        x0 = np.linalg.solve(self.A, -self.B@u0)

        self.emt_init = InitialConditionsEMT(*x0, v_ref=v_ref)

    def get_small_signal_model(self, x_l: float, x_a: float, x_e: float, x_f: float, v_ref: float, v_c: float, v_s: float):
        ssm = StateSpaceModel(
            A=self.A,
            B=self.B,
            C=self.C,
            D=np.zeros((1, 3)),
            x=DynamicalVariables(
                name=['x_l', 'x_a', 'x_e', 'x_f'],
                init=[x_l, x_a, x_e, x_f]),
            u=DynamicalVariables(
                name=['v_ref', 'v_c', 'v_s'],
                init=[v_ref, v_c, v_s]),
            y=DynamicalVariables(
                name=['v_fd'],
                init=[0])
        )
        return ssm
        

    def get_quadratic_bilinear_model(self,  x_l: float, x_a: float, x_e: float, x_f: float, v_ref: float, v_c: float, v_s: float):
        ssm = self.get_small_signal_model(x_l, x_a, x_e, x_f, v_ref, v_c, v_s)
        return ssm.to_quadratic_bilinear()

    def get_derivatives_step_emt_dq0(self, x_l: float, x_a: float, x_e: float, x_f: float, v_ref: float, v_c: float, v_s: float) -> float:
        x = np.array([x_l, x_a, x_e, x_f])
        u = np.array([v_ref, v_c, v_s])

        return self.A@x + self.B@u

    def get_algebraics_step_emt_dq0(self, x_l: float, x_a: float, x_e: float, x_f: float):
        x = np.array([x_l, x_a, x_e, x_f])

        return self.C@x + self.emt_init.v_ref