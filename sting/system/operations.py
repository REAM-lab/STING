# -----------------------
# Import python packages
# -----------------------
import logging
from dataclasses import dataclass

# -----------------------
# Import sting code
# -----------------------
from sting.system.core_testing import System
from sting.branch.series_rl import BranchSeriesRL
from sting.shunt.parallel_rc import ShuntParallelRC

# Set up logger
logger = logging.getLogger(__name__)

@dataclass(slots=True)
class SystemModifier:
    system: System

    def decompose_lines(self, delete_lines: bool = False):
        logger.info("> Add branches and shunts from dissecting lines:")
        logger.info("  - Lines with no series compensation")

        for line in self.system.lines:
            if line.decomposed:
                continue  # Skip already decomposed lines

            branch = BranchSeriesRL(
                    name=f"from_line_{line.id}",
                    from_bus=line.from_bus,
                    from_bus_id=line.from_bus_id,
                    to_bus=line.to_bus,
                    to_bus_id=line.to_bus_id,
                    base_power_MVA=line.base_power_MVA,
                    base_voltage_kV=line.base_voltage_kV,
                    base_frequency_Hz=line.base_frequency_Hz,
                    r_pu=line.r_pu,
                    x_pu=line.x_pu,
                )

            from_shunt = ShuntParallelRC(
                    name=f"from_line_{line.id}",
                    bus=line.from_bus,
                    bus_id=line.from_bus_id,
                    base_power_MVA=line.base_power_MVA,
                    base_voltage_kV=line.base_voltage_kV,
                    base_frequency_Hz=line.base_frequency_Hz,
                    g_pu= line.g_pu,
                    b_pu= line.b_pu,
                )

            to_shunt = ShuntParallelRC(
                    name=f"to_line_{line.id}",
                    bus=line.to_bus,
                    bus_id=line.to_bus_id,
                    base_power_MVA=line.base_power_MVA,
                    base_voltage_kV=line.base_voltage_kV,
                    base_frequency_Hz=line.base_frequency_Hz,
                    g_pu= line.g_pu,
                    b_pu= line.b_pu,
                )

            # Add shunts and branch to system
            self.system.add(branch)
            self.system.add(from_shunt)
            self.system.add(to_shunt)

            # Mark line as decomposed, so it is not decomposed again
            line.decomposed = True

        # Delete all lines so they cannot be added to the system again
        if delete_lines:
            self.system.lines = []

        logger.info("... ok.\n")
        # TODO: Do the same for line with series compensation

    def combine_shunts(self):
        "untested"

        print("> Reduce shunts to have one shunt per bus:")

        shunt_df = (self.system.shunts
            .to_table("bus_id", "g", "b")
            .reset_index(drop=True)
            .pivot_table(index="bus_id", values=["g", "b"], aggfunc="sum")
        )

        shunt_df["r"] = 1 / shunt_df["g"]
        shunt_df["c"] = 1 / shunt_df["b"]
        shunt_df["id"] = range(len(shunt_df))
        shunt_df.drop(columns=["b", "g"], inplace=True)

        # Clear all existing parallel RC shunts
        self.system.pa_rc = []

        # Add each effective/combined parallel RC shunt to the pa_rc components
        for _, row in shunt_df.iterrows():
            shunt = ShuntParallelRC(**row.to_dict())
            self.system.add(shunt)

        print("\t- New list of parallel RC components created ... ok\n")
