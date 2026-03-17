# -----------------------
# Import python packages
# -----------------------
import logging
from dataclasses import dataclass
import copy
import polars as pl

# -----------------------
# Import sting code
# -----------------------
from sting.system.core import System
import sting.generator.shared.capacity_expansion as gen_capex
import sting.bus.shared.capacity_expansion as bus_capex
import sting.storage.shared.capacity_expansion as storage_capex
from sting.bus.core import Bus
from sting.branch.series_rl import BranchSeriesRL
from sting.shunt.parallel_rc import ShuntParallelRC
from sting.utils.runtime_tools import timeit

# Set up logger
logger = logging.getLogger(__name__)

# -----------------------
# Main classes
# -----------------------
@dataclass(slots=True)
class SystemModifier:
    """Class to perform operations on the system, such as grouping by zones or uploading built capacities from a previous capex solution.
    This class operates over all components of the system. The methods of this class could have been implemented as methods of the System class, 
    but we choose to implement them in a separate class to keep the System short"""

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
                    zone=line.zone
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
                    # Shunts inherit their zone from buses
                    zone=self.system.buses[line.from_bus_id].zone
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
                    zone=self.system.buses[line.to_bus_id].zone
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
        """
        Combine all shunts at a given bus into a single "effective" shunt
        by considering them as parallel circuits.

        ASSUMPTIONS
        1. There are only parallel RC shunts in the system
        2. All shunts at the same `bus_id` share the same
            - "base_power_MVA", "base_voltage_kV", "base_frequency_Hz", and "zone"
        """

        print("> Combining shunts into one 'effective' shunt per bus:")

        shared_columns = ["bus", "base_power_MVA", "base_voltage_kV", "base_frequency_Hz", "zone"]
        # DataFrame with effective shunt parameters
        shunt_df = (
            self.system.query(["shunt_parallel_rc"])
            .to_table("bus_id", "g_pu", "b_pu", *shared_columns)
            .group_by("bus_id")
            .agg(
                # Conductance and susceptance can be summed when in parallel
                pl.col("g_pu", "b_pu").sum(),
                # Take first value among parameters that are assumed to be shared
                pl.col(shared_columns).first()
            )
            .with_columns(
                name=pl.col("bus") + pl.lit("_shunt")
            )
        )

        # Total number of shunts to remove and create
        original_n = len(self.system.shunt_parallel_rc)
        reduced_n = shunt_df.height
        # Clear all existing parallel RC shunts
        self.system.shunt_parallel_rc.clear()

        # Add each effective/combined parallel RC shunt to the pa_rc components
        for row in shunt_df.iter_rows(named=True):
            shunt = ShuntParallelRC(**row)
            self.system.add(shunt)

        print(f"\t- Removed {original_n} shunts, created {reduced_n} effective shunts... ok\n")

    @timeit
    def group_by_zones(self, components_to_clone: list[str] = None) -> System:
        """
        Creation of a zonal system where buses are grouped by their zone attribute.

        Method created for a manual zonal reduction of the system, needed for the capacity expansion module.
        Warnings:
         - Only components that have bus, from_bus, to_bus attributes are re-assigned to the new zonal buses.
         - Buses without a zone attribute are ignored.
         - Other attributes are set to None or default values.
        """

        zonal_system = System(case_directory=self.system.case_directory)

        mapping_bus_to_zone = {n.name: n.zone for n in self.system.buses if n.zone is not None}
        zones = set(mapping_bus_to_zone.values())

        for zone in zones:
            zonal_system.add( Bus(
                name=zone,
                bus_type="zone_bus",
                zone=zone,
            ))
        logger.info(f" - System with new buses created: {zones}")

        for component in self.system:
            if (hasattr(component, 'bus')) and (component.bus in mapping_bus_to_zone):
                copied_component = copy.deepcopy(component)
                copied_component.bus = mapping_bus_to_zone[component.bus]
                zonal_system.add(copied_component)

            if ((hasattr(component, 'from_bus') and hasattr(component, 'to_bus')) and 
                (component.from_bus in mapping_bus_to_zone) and (component.to_bus in mapping_bus_to_zone)):
                if mapping_bus_to_zone[component.from_bus] != mapping_bus_to_zone[component.to_bus]:
                    copied_component = copy.deepcopy(component)
                    copied_component.from_bus = mapping_bus_to_zone[component.from_bus]
                    copied_component.to_bus = mapping_bus_to_zone[component.to_bus]
                    zonal_system.add(copied_component)
        
        logger.info(f" - Re-assigning bus, from_bus, to_bus attributes in system components completed.")

        if components_to_clone is not None:
            for attr in components_to_clone:
                setattr(zonal_system, attr, copy.deepcopy(getattr(self.system, attr)))
        logger.info(f" - Cloning components: {components_to_clone} completed.")

        logger.info(f" - New system has: ")
        for (type_, _ ) in zonal_system.components:
            logger.info(f"  - {len(getattr(zonal_system, type_))} '{type_}' components. ")

        zonal_system.apply("post_system_init", zonal_system)

        return zonal_system
    
    @timeit
    def upload_built_capacities_from_csv(self, built_capacity_directory: str,  make_non_expandable: bool = True, threshold_MW: float = 1e-1):
        """
        Upload built capacities from a previous capex solution. 
        
        ### Args:
        - built_capacity_directory: `str` 
                    Directory where the CSV files with built capacities are located.
        - make_non_expandable: `bool`, default True
                    If True, the generators, storage units and buses for which built capacities are uploaded will be made non-expandable, 
                    so that their capacities cannot be further expanded in the optimization. 
                    If False, we check the uploaded built capacity against the maximum capacity, and 
                    only make non-expandable those units for which the uploaded built capacity is greater or equal to the maximum capacity. 

        """
        gen_capex.upload_built_capacities_from_csv(self.system, built_capacity_directory, make_non_expandable, threshold_MW)
        storage_capex.upload_built_capacities_from_csv(self.system, built_capacity_directory, make_non_expandable, threshold_MW)
        bus_capex.upload_built_capacities_from_csv(self.system, built_capacity_directory, make_non_expandable, threshold_MW)