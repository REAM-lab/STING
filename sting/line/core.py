from sting.branch.series_rl import BranchSeriesRL
from sting.shunt.parallel_rc import ShuntParallelRC


def decompose_lines(system):

    print("> Add branches and shunts from dissecting lines:")
    print("\t- Lines with no series compensation", end=" ")

    for line in system.line_pi:
        if not line.decomposed:
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
            system.add(branch)
            system.add(from_shunt)
            system.add(to_shunt)

            # Mark line as decomposed, so it is not decomposed again
            line.decomposed = True

    # Delete all lines so they cannot be added to the system again
    # system.line_pi = []

    print("... ok.\n")
    # TODO: Do the same for line with series compensation
