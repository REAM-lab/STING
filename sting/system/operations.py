# -----------------------
# Import python packages
# -----------------------
import logging
from dataclasses import dataclass
import copy
import polars as pl
from pyparsing import line

# -----------------------
# Import sting code
# -----------------------
from sting.load.impedance_load import ConstantImpedanceLoad
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

    def create_impedance_loads(self, delete_loads: bool = False):
        """
        Create constant impedance loads from generic loads in the system.
        This is used in Small Signal Modeling (SSM) and EMT simulations.
        The loads are given in MW and MVAR for Power Flow, then this info is passed to a constant impedance load, which is modeled as a series RL branch connected to ground.
        """

        logger.info("> Add constant impedance loads from generic loads:")
        counter = 0
        for load in self.system.loads:

            if load.modeled_as_other_load_type:
                continue  # Skip because it means that the generic load has already been modeled as another load type (e.g., constant impedance load, constant current load, etc.)

            # check if the load has zero MW or MVAR, if so, skip it, it means that there is no load.
            if (load.load_MW == 0) and (load.load_MVAR == 0):
                continue

            if (load.load_MW > 0) and (load.load_MVAR == 0):
                logger.warning(f"Load {load.name} has active power but no reactive power. Add a nonzero reactive power to make it a valid load.")
                raise ValueError(f"Load {load.name} has active power but no reactive power. Add a nonzero reactive power to make it a valid load.")

            z_load = ConstantImpedanceLoad(
                name=load.name,
                bus=load.bus,
                timepoint=load.timepoint,
                bus_id=load.bus_id,
                load_MW=load.load_MW,
                load_MVAR=load.load_MVAR,
                base_power_MVA=load.base_power_MVA,
                base_voltage_kV=load.base_voltage_kV,
                base_frequency_Hz=load.base_frequency_Hz,
            )
            counter += 1
            self.system.add(z_load)

            # Mark the load as modeled as a constant impedance load, so it is not modeled as a generic load anymore
            load.modeled_as_other_load_type = True

        # Delete all loads so they cannot be added to the system again
        if delete_loads:
            self.system.loads.clear()

        logger.info(f" - Created {counter} constant impedance loads... ok\n")

    def decompose_lines(self, delete_lines: bool = False):
        logger.info("> Add branches and shunts to system from dissecting pi-model lines:")
        logger.info(" - Lines with no series compensation")

        counter = 0
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
                    g_pu= line.g_pu/2,
                    b_pu= line.b_pu/2,
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
                    g_pu= line.g_pu/2,
                    b_pu= line.b_pu/2,
                    zone=self.system.buses[line.to_bus_id].zone
                )

            # Add shunts and branch to system
            self.system.add(branch)
            self.system.add(from_shunt)
            self.system.add(to_shunt)

            counter += 1
            # Mark line as decomposed, so it is not decomposed again
            line.decomposed = True

        # Delete all lines so they cannot be added to the system again
        if delete_lines:
            self.system.lines.clear()

        logger.info(f"  - Pi-model {counter} lines decomposed into {counter} branches and {2*counter} shunts... ok\n")
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

        logger.info("> Combining parallel RC shunts into one 'effective' parallel RC shunt per bus:")

        shared_columns = ["bus", "base_power_MVA", "base_voltage_kV", "base_frequency_Hz", "zone"]
        # DataFrame with effective shunt parameters
        shunt_df = (
            self.system.query(["shunt_parallel_rc"])
            .to_table("bus_id", "g_pu", "b_pu", *shared_columns)
        )

        # Check if the number of unique bus_id is equal to the number of rows in the shunt_df, if so, there is no need to combine shunts
        if shunt_df.select("bus_id").n_unique() == shunt_df.height:
            logger.info("  - System has already one shunt (g parallel b) per bus. No need to combine shunts... ok\n")
            return
        
        shunt_df = (
            shunt_df
            .group_by("bus_id")
            .agg(
                # Conductance and susceptance can be summed when in parallel
                pl.col("g_pu", "b_pu").sum(),
                # [!] WARNING [!]
                # Take first value among parameters that are assumed to be shared
                pl.col(shared_columns).first()
            )
            .with_columns(
                name=pl.col("bus") + pl.lit("_shunt")
            )
            # [!] CRITICAL [!]
            # We **must** sort the new shunts based on their bus_id in order for 
            # the CCM to be defined correctly. This is very important.
            # The CCM matrices are constructed assuming that shunts are ordered 
            # according to the buses. Thus we enforce (at this step) that 
            # the $N$ shunts in the set of all shunts $\mathcal{S}$ are assigned 
            # to the $N$ buses using the same index. The i-th bus id == the i-th 
            # shunt id.
            .sort("bus_id")
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

        logger.info(f"  - Removed {original_n} shunts, created {reduced_n} effective shunts... ok\n")

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

        logger.info(" - Grouping components by zones...")
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