import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Read bus
bus = pl.read_csv("/Users/paul/Downloads/bus_ieee.csv")
branch = pl.read_csv("/Users/paul/Downloads/branch_ieee.csv")

# Create add voltage base column to branch
branch = branch.join(
    bus.select(["bus", "base_voltage_kv"]).rename({"bus": "from_bus", "base_voltage_kv": "from_base_kv"}),
    on="from_bus",
    how="left"
    ).join(
    bus.select(["bus", "base_voltage_kv"]).rename({"bus": "to_bus", "base_voltage_kv": "to_base_kv"}),
    on="to_bus",
    how="left"
    )

# Get lines if from base kv is equal to to base kv
lines = (branch.filter( (pl.col("from_base_kv") == pl.col("to_base_kv")) & (pl.col("miles") > 1) )
                .select(["id", "from_bus", "to_bus", "miles", "from_base_kv", "r_pu", "x_pu", "b_pu", "rating_mva"])
                .rename({"from_base_kv": "base_voltage_kv"}))  

# Get slopes of x_pu vs miles for lines with base voltage 230 kV
lines = (lines
                .with_columns((pl.col("x_pu") / pl.col("miles")).alias("x_pu_per_mile"))
                .with_columns((pl.col("r_pu") / pl.col("miles")).alias("r_pu_per_mile"))
                .with_columns((pl.col("b_pu") / pl.col("miles")).alias("b_pu_per_mile"))
)

# Get median
lines_median = lines.group_by("base_voltage_kv").agg(
    pl.median("x_pu_per_mile"),
    pl.median("r_pu_per_mile"),
    pl.median("b_pu_per_mile")
)

print("lines_median")
print(lines_median)

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

miles = 100 * 1/1.60934  # Convert km to miles
base_voltage_kv = 230
line_params = line_ieeerts79(base_voltage_kv, miles)
print(f"Line parameters for {miles:.2f} miles at {base_voltage_kv} kV: r_pu = {line_params['r_pu']:.6f}, x_pu = {line_params['x_pu']:.6f}, b_pu = {line_params['b_pu']:.6f}")

error_r = []
error_x = []
error_b = []
for row in lines.iter_rows(named=True):
    line_params = line_ieeerts79(row["base_voltage_kv"], row["miles"])
    error_r.append(abs(line_params["r_pu"] - row["r_pu"])/row["r_pu"] * 100)
    error_x.append(abs(line_params["x_pu"] - row["x_pu"])/row["x_pu"] * 100)
    error_b.append(abs(line_params["b_pu"] - row["b_pu"])/row["b_pu"] * 100)

# Report median and maximum errors
median_error_r = np.median(error_r)
median_error_x = np.median(error_x)
median_error_b = np.median(error_b)
print(f"Median relative error in r_pu: {median_error_r:.6f}")
print(f"Median relative error in x_pu: {median_error_x:.6f}")
print(f"Median relative error in b_pu: {median_error_b:.6f}")
max_error_r = max(error_r)
max_error_x = max(error_x)
max_error_b = max(error_b)
print(f"Maximum relative error in r_pu: {max_error_r:.6f}")
print(f"Maximum relative error in x_pu: {max_error_x:.6f}")
print(f"Maximum relative error in b_pu: {max_error_b:.6f}")
