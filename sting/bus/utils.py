# ---------------
# Import python packages
# ---------------
import polars as pl
import logging

# ---------------
# Import sting code
# ---------------
from sting.bus.core import Load

logger = logging.getLogger(__name__)

def load_as_dict(load: list[Load]) -> tuple[dict, dict]: 
    df = pl.DataFrame(
                        schema = ['id', 'bus', 'scenario', 'timepoint', 'load_MW', 'load_MVAR'],
                        data= map(lambda ld: (ld.id, ld.bus, ld.scenario, ld.timepoint, ld.load_MW, ld.load_MVAR), load)
                        )
    if len(df.select(['bus', 'scenario', 'timepoint']).unique()) != (df.height):
        logger.info("There are multiple load entries for the same bus, scenario, and timepoint. They will be grouped and summed.")
        df = df.group_by(['bus', 'scenario', 'timepoint']).agg(pl.col(['load_MW', 'load_MVAR']).sum())

    cols = ['bus', 'scenario', 'timepoint'] if load[0].scenario is not None else ['bus', 'timepoint']
    active_load = dict( zip(df.select(cols).iter_rows(), df['load_MW']) )
    reactive_load = dict( zip(df.select(cols).iter_rows(), df['load_MVAR'])) if load[0].load_MVAR is not None else None

    return active_load, reactive_load