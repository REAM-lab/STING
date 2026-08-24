def line_ieeerts79(base_voltage_kv: float, miles: float) -> dict:
    """
    Get the line parameters for a given base voltage and length in miles.
    The parameters are based on the IEEE RTS-79 test system.
    Base power = 100 MVA
    Base voltage = 138 kV or 230 kV
    """
    # Get the median values for the given base voltage
    median_by_base_voltage ={   138: {'r_pu_per_mile': 0.001, 'x_pu_per_mile': 0.003837, 'b_pu_per_mile': 0.001045},
                                230: {'r_pu_per_mile': 0.000182, 'x_pu_per_mile': 0.001447, 'b_pu_per_mile': 0.00303}
    }
    
    # Calculate the line parameters
    r_pu = median_by_base_voltage[base_voltage_kv]["r_pu_per_mile"] * miles
    x_pu = median_by_base_voltage[base_voltage_kv]["x_pu_per_mile"] * miles
    b_pu = median_by_base_voltage[base_voltage_kv]["b_pu_per_mile"] * miles
    
    return {"r_pu": r_pu, "x_pu": x_pu, "b_pu": b_pu}