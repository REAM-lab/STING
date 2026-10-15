from dataclasses import dataclass
import numpy as np
import polars as pl

def line_ieeerts79(base_voltage_kv: float, miles: float) -> dict:
    """
    Get the line parameters for a given base voltage and length in miles.
    The parameters are based on the IEEE RTS-79 test system.
    Base power = 100 MVA
    Base voltage = 138 kV or 230 kV
    """
    # Get the median values for the given base voltage
    median_by_base_voltage ={   138: {'r_pu_per_mile': 0.001, 'x_pu_per_mile': 0.003837, 'b_pu_per_mile': 0.001045},
                                230: {'r_pu_per_mile': 0.000182, 'x_pu_per_mile': 0.001447, 'b_pu_per_mile': 0.00303}
    }
    
    # Calculate the line parameters
    r_pu = median_by_base_voltage[base_voltage_kv]["r_pu_per_mile"] * miles
    x_pu = median_by_base_voltage[base_voltage_kv]["x_pu_per_mile"] * miles
    b_pu = median_by_base_voltage[base_voltage_kv]["b_pu_per_mile"] * miles
    
    return {"r_pu": r_pu, "x_pu": x_pu, "b_pu": b_pu}


@dataclass
class ParticipationFactors:
    """
    Class for computing participation factors of a linear system.
    """
    A: np.ndarray
    states: None # DynamicalVariables
    metadata: pl.DataFrame = None
    participation_factors: pl.DataFrame = None

    def __post_init__(self):
        # Compute eigenvalues and vectors
        d, V = np.linalg.eig(self.A)
        W = np.linalg.inv(V.T)
        # Participation factors matrix where each column sums to 1
        P = W * V

        # Matrix with modes and state names
        self.metadata = pl.from_dict({
            "id": list(range(len(d))),
            "name": self.states.name,
            "component": self.states.component,
            "mode_real": list(d.real),
            "mode_imag": list(d.imag),
            })

        self.participation_factors = pl.from_dict({"id": list(range(len(d)))}|{f"mode{i}":list(np.round(P[:,i].real, 5)) for i in range(len(d))})

    def get_mode_factors(self, mode):
        df_mode = (
            self.metadata
            .drop("mode_real", "mode_imag")
            .join(
                self.participation_factors.select("id",mode),
                on="id")
            .sort(descending=True, by=mode)
        )
        return df_mode