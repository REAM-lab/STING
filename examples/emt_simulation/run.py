"""
Testcase1 simulates a two infinite sources connected via a transmission line.

First, we compute the system-wide small-signal model using STING. 
This small-signal model also contains EMT initial conditions.

Second (optional), we transfer the system data and initial conditions 
to Simulink. STING has a functionality to transfer this data. 
In EMT simulation is in the file sim_emt.slx. 

Comparison between the small-signal simulation and EMT simultion
shows a proximity of these domain responses. Dec 7, 2025.
"""
# %%
# Import Python standard and third-party packages
import os
import matlab.engine
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Import sting package
from sting import main
from sting.utils.transformations import dq02abc, abc2dq0

# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent
#case_dir = os.path.join(os.getcwd(), "examples", "testcase1")

# Construct system and small-signal model
def step1(t):
    return 0.1 if t >= 0.5 else 0.0

def step2(t):
    return 0.0

inputs = {'inf_src_1': {'v_ref_d': step1}, 
          'inf_src_2': {'v_ref_d': step2}}

t_max = 1.0

solution, sys = main.run_emt(t_max, inputs, case_dir =case_dir)

# Define timepoints that will be used to evaluate the solution of the ODEs
tps = np.linspace(0, t_max, 500)
n_tps = len(tps)

# Extract solution of the ODEs and evaluate at the timepoints
interp_sol = solution.sol(tps)
i_bus_a_1 = interp_sol[0]
i_bus_b_1 = interp_sol[1]
i_bus_c_1 = interp_sol[2]
i_bus_a_2 = interp_sol[3]
i_bus_b_2 = interp_sol[4]
i_bus_c_2 = interp_sol[5]
v_bus_a_1 = interp_sol[6]
v_bus_b_1 = interp_sol[7]
v_bus_c_1 = interp_sol[8]
v_bus_a_2 = interp_sol[9]
v_bus_b_2 = interp_sol[10]
v_bus_c_2 = interp_sol[11]
i_br_a = interp_sol[12]
i_br_b = interp_sol[13]
i_br_c = interp_sol[14]
angle_pc = interp_sol[15]

ang1 = sys.inf_src[0].emt_init.angle_ref * np.pi / 180
ang2 = sys.inf_src[1].emt_init.angle_ref * np.pi / 180

# Transform abc to dq0
i_bus_d_1, i_bus_q_1, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in 
                                zip(i_bus_a_1, i_bus_b_1, i_bus_c_1, angle_pc+ang1)])

i_bus_d_2, i_bus_q_2, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in 
                                zip(i_bus_a_2, i_bus_b_2, i_bus_c_2, angle_pc+ang2)])

v_bus_d_1, v_bus_q_1, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in 
                                zip(v_bus_a_1, v_bus_b_1, v_bus_c_1, angle_pc)])

v_bus_d_2, v_bus_q_2, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in 
                                zip(v_bus_a_2, v_bus_b_2, v_bus_c_2, angle_pc)])

i_br_D, i_br_Q, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in 
                                zip(i_br_a, i_br_b, i_br_c, angle_pc)])

# %% Plot results

fig = make_subplots(
    rows=5, cols=2,
    horizontal_spacing=0.15,
    vertical_spacing=0.05,
)
fig.add_trace(go.Scatter(x=tps, y=i_bus_d_1, name="i_bus_d_1"), row=1, col=1)
fig.update_xaxes(title_text='Time [s]', row=1, col=1)
fig.update_yaxes(title_text='i_bus_d_1', row=1, col=1)

fig.add_trace(go.Scatter(x=tps, y=i_bus_q_1, name="i_bus_q_1"), row=1, col=2)
fig.update_xaxes(title_text='Time [s]', row=1, col=2)
fig.update_yaxes(title_text='i_bus_q_1', row=1, col=2)

fig.add_trace(go.Scatter(x=tps, y=v_bus_d_1, name="v_bus_d_1"), row=2, col=1)
fig.update_xaxes(title_text='Time [s]', row=2, col=1)
fig.update_yaxes(title_text='v_bus_d_1', row=2, col=1)

fig.add_trace(go.Scatter(x=tps, y=v_bus_q_1, name="v_bus_q_1"), row=2, col=2)
fig.update_xaxes(title_text='Time [s]', row=2, col=2)
fig.update_yaxes(title_text='v_bus_q_1', row=2, col=2)

fig.add_trace(go.Scatter(x=tps, y=i_bus_d_2, name="i_bus_d_2"), row=3, col=1)
fig.update_xaxes(title_text='Time [s]', row=3, col=1)
fig.update_yaxes(title_text='i_bus_d_2', row=3, col=1)

fig.add_trace(go.Scatter(x=tps, y=i_bus_q_2, name="i_bus_q_2"), row=3, col=2)
fig.update_xaxes(title_text='Time [s]', row=3, col=2)
fig.update_yaxes(title_text='i_bus_q_2', row=3, col=2)

fig.add_trace(go.Scatter(x=tps, y=v_bus_d_2, name="v_bus_d_2"), row=4, col=1)
fig.update_xaxes(title_text='Time [s]', row=4, col=1)
fig.update_yaxes(title_text='v_bus_d_2', row=4, col=1)

fig.add_trace(go.Scatter(x=tps, y=v_bus_q_2, name="v_bus_q_2"), row=4, col=2)
fig.update_xaxes(title_text='Time [s]', row=4, col=2)
fig.update_yaxes(title_text='v_bus_q_2', row=4, col=2)

fig.add_trace(go.Scatter(x=tps, y=i_br_D, name="i_br_D"), row=5, col=1)
fig.update_xaxes(title_text='Time [s]', row=5, col=1)
fig.update_yaxes(title_text='i_br_D', row=5, col=1)

fig.add_trace(go.Scatter(x=tps, y=i_br_Q, name="i_br_Q"), row=5, col=2)
fig.update_xaxes(title_text='Time [s]', row=5, col=2)
fig.update_yaxes(title_text='i_br_Q', row=5, col=2)

fig.update_layout(height=1200, 
                  width=800, 
                  showlegend=False,
                  margin={'t': 0, 'l': 0, 'b': 0, 'r': 0})
fig.show()

print('ok')
# %%
