# ----------------------
# Import python packages
# ----------------------
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import pyomo.environ as pyo

# ------------------
# Import sting code
# ------------------
from sting.utils.dynamical_systems import DynamicalVariables


# ----------------
# Main classes
# ----------------
@dataclass(slots=True)
class Scenario:
    name: str
    probability: float
    id: int = None

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id

@dataclass(slots=True)
class Timepoint:
    name: str
    timeseries: str
    id: int = None
    timeseries_id: int = None
    weight: float = None
    duration_hr: float = None   
    prev_timepoint_id: int = None

    def post_system_init(self, system):
        timeseries = next((p for p in system.ts if p.name == self.timeseries))
        self.timeseries_id = timeseries.id
        self.duration_hr = timeseries.timepoint_duration_hr
        self.weight = self.duration_hr * timeseries.timeseries_scale_to_period

        tps_in_ts = next((ts for ts in system.ts if ts.id == self.timeseries_id)).timepoint_ids
        if self.id == tps_in_ts[0]:
            self.prev_timepoint_id = tps_in_ts[-1]
        else:
            self.prev_timepoint_id = self.id - 1

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id

@dataclass(slots=True)
class Timeseries:
    id: int = field(default=-1, init=False)
    name: str
    timepoint_duration_hr: float
    number_of_timepoints: int
    timeseries_scale_to_period: float
    timepoint_ids: Optional[list[int]] = field(default=None, init=False)
    start: str = None
    end: str = None
    period : str = None
    timepoint_selection_method: str = None
    
    def post_system_init(self, system):
        self.timepoint_ids = [t.id for t in system.tp if t.timeseries == self.name]

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id
