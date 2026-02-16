# ----------------------
# Import python packages
# ----------------------
from dataclasses import dataclass

# ------------------
# Import sting code
# ----------------
from sting.system.component import Component

# -----------------
# Main classes
# ----------------
@dataclass(slots=True)
class Scenario(Component):
    probability: float

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id

@dataclass(slots=True)
class Timepoint(Component):
    timeseries: str = None
    timeseries_id: int = None
    weight: float = None
    duration_hr: float = None   
    prev_timepoint_id: int = None

    def post_system_init(self, system):
        
        if self.timeseries is not None:
            timeseries = next((p for p in system.timeseries if p.name == self.timeseries))
            self.timeseries_id = timeseries.id
            self.duration_hr = timeseries.timepoint_duration_hr
            self.weight = self.duration_hr * timeseries.timeseries_scale_to_period

            tps_in_ts = next((ts for ts in system.timeseries if ts.id == self.timeseries_id)).timepoint_ids
            if self.id == tps_in_ts[0]:
                self.prev_timepoint_id = tps_in_ts[-1]
            else:
                self.prev_timepoint_id = self.id - 1
        else:
            self.weight = 1
            self.duration_hr = 1
            
    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id
    
    def __repr__(self):
        return f"Timepoint(id={self.id}, name='{self.name}', timeseries='{self.timeseries}')"

@dataclass(slots=True)
class Timeseries(Component):
    timepoint_duration_hr: float
    number_of_timepoints: int
    timeseries_scale_to_period: float
    timepoint_ids: list[int] = None
    start: str = None
    end: str = None
    period : str = None
    timepoint_selection_method: str = None
    
    def post_system_init(self, system):
        self.timepoint_ids = [t.id for t in system.timepoints if t.timeseries == self.name]

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id
    
    def __repr__(self):
        return f"Timeseries(id={self.id}, name='{self.name}')"
