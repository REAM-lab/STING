from dataclasses import dataclass, field

@dataclass
class Line_no_series_compensation:
    idx: str
    from_bus: str
    to_bus: str
    sbase: float	
    vbase: float
    fbase: float
    r: float
    l: float
    g: float
    b: float
    name: str = field(default_factory=str)
    type: str = 'line'


def decompose_lines(system):

    from sting.branch.series_rl import Series_rl_branch
    from sting.shunt.parallel_rc import Parallel_rc_shunt

    print("> Add branches and shunts from dissecting lines:")
    print("\t- Lines with no series compensation", end=' ')
    
    for line in system.components.line_ns:
        
        system.components.se_rl.append(
            Series_rl_branch(
                idx = 'from_' + line.idx, 
                type = 'branch', 
                from_bus = line.from_bus, 
                to_bus = line.to_bus,
                sbase = line.sbase,
                vbase = line.vbase,
                fbase = line.fbase,
                r = line.r,
                l = line.l ))
            
        system.components.pa_rc.append(
            Parallel_rc_shunt( 
                idx = line.idx + '_frombus',
                type = 'shunt', 
                bus_idx = line.from_bus,
                sbase = line.sbase,
                vbase = line.vbase,
                fbase = line.fbase,
                r = 1/line.g,
                c = 1/line.b))
        
        system.components.pa_rc.append(
            Parallel_rc_shunt(
                idx = line.idx + '_tobus',
                type = 'shunt', 
                bus_idx = line.to_bus,
                sbase = line.sbase,
                vbase = line.vbase,
                fbase = line.fbase,
                r = 1/line.g,
                c = 1/line.b))
        
    # Delete all lines, so they accidently get added to the system twice
    system.components.line_ns = []
        
    print("... ok.\n")
    # TODO: Do the same for line with series compensation