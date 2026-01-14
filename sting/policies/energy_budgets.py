# -------------
# Import python packages
# --------------
from dataclasses import dataclass, field
import pyomo.environ as pyo
import os
import polars as pl

# -------------
# Import sting code
# --------------

# ----------------
# Main classes     
# ----------------
@dataclass(slots=True)
class EnergyBudgetTimeGroup:
    id: int = field(default=-1, init=False)
    name: str
    timepoint: str

@dataclass(slots=True)
class EnergyBudget:
    id: int = field(default=-1, init=False)
    generator: str
    time_group: str
    energy_budget_MWh: float
    generator_id: int = None
    timepoint_ids: list[str] = None
    scenario: str = None
    scenario_id: int = None

    def post_system_init(self, system):
         
        # Get timepoints within the time group
        time_groups = list(filter(lambda tg: tg.name == self.time_group, system.tg_bud))
        tps_name = list(map(lambda tg: tg.timepoint, time_groups))
        self.timepoint_ids = [ tp.id for tp in system.tp if tp.name in tps_name]

        # Get generator id
        self.generator_id = next((g.id for g in system.gen if g.name == self.generator))

        # Get scenario id
        if self.scenario is not None:
            self.scenario_id = next((sc for sc in system.sc if sc.name == self.scenario)).id

    def __hash__(self):
        """Hash based on id attribute, which must be unique for each instance."""
        return self.id

def construct_capacity_expansion_model(system, model, model_settings):

    def cEnergyBudget_rule(m, eb):
            tps = [system.tp[id] for id in eb.timepoint_ids]
            gen = system.gen[eb.generator_id]
            if eb.scenario_id is not None:
                sc = system.sc[eb.scenario_id]
                return  sum(m.vGENV[gen, sc, t] * t.weight for t in tps) <= eb.energy_budget_MWh
            else:
                return  sum(m.vGEN[gen, t] * t.weight for t in tps) <= eb.energy_budget_MWh
        
    model.cEnergyBudget = pyo.Constraint(system.e_bud, rule=cEnergyBudget_rule)

