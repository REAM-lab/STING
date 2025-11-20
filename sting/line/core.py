from sting.branch.series_rl import BranchSeriesRL 
from sting.shunt.parallel_rc import ShuntParallelRC

def decompose_lines(system):

    print("> Add branches and shunts from dissecting lines:")
    print("\t- Lines with no series compensation", end=' ')
    
    # Get the next open index for parallel RC shunts
    shunt_idx = system.next_open("pa_rc")
    
    for line in system["line_pi"]:
        
        branch = BranchSeriesRL(idx = line.idx, from_bus = line.from_bus, 
            to_bus = line.to_bus, sbase = line.sbase, vbase = line.vbase, 
            fbase = line.fbase, r = line.r, l = line.l)
        
        from_shunt = ShuntParallelRC(idx = shunt_idx, bus_idx = line.from_bus,
            sbase = line.sbase, vbase = line.vbase, fbase = line.fbase,
            r = 1/line.g, c = 1/line.b)
        
        to_shunt = ShuntParallelRC(idx = shunt_idx+1, bus_idx = line.to_bus,
            sbase = line.sbase, vbase = line.vbase, fbase = line.fbase,
            r = 1/line.g, c = 1/line.b)
        
        # Add shunts and branch to system
        system.add(branch)
        system.add(from_shunt)
        system.add(to_shunt)
        # Increment the shunt index
        shunt_idx += 2
        
    # Delete all lines so they cannot be added to the system again
    system.clear("line_pi")
        
    print("... ok.\n")
    # TODO: Do the same for line with series compensation