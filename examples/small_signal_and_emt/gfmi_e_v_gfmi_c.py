""" 
Compares the EMT and small signal response of the GFMI_C, which does not model the DC side of the converter,
the GFMI_E which models a battery and current-source load on the DC side.

In progress - March 31 - Ruth 
"""

# Import Python standard and third-party packages
from pathlib import Path
import os 
import numpy as np
import pandas as pd 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import eig, inv
import seaborn as sns 
import matplotlib.pyplot as plt
from sting.utils.transformations import dq02abc, abc2dq0
from scipy import signal


# Import sting package
from sting import main
from sting.system.core import System

# Specify path of the case study directory

case_dir_gfmc = os.path.join(Path(__file__).resolve().parent,"2-bus_src-gfm")
case_dir_gfme = os.path.join(Path(__file__).resolve().parent,"2-bus-src-gfmi_e")

# Define inputs 
def step1(t):
    return 0.3 if t >= 0.5 else 0.0

def step2(t):
    return 0.0

# Combine inputs for gfmi_c and gfmi_e 
inputs = {'infinite_sources_0': {'v_ref_d': step1}, 
          'gfmi_e_0': {'p_ref': step2, 
                       'q_ref': step2,
                       'v_ref': step2,
                       'v_dc_ref': step2,
                       'v_s': step2, 
                       'i_load_ref': step2},
          'gfmi_c_0': {'p_ref': step2,
                       'q_ref': step2,
                       'v_ref': step2}}

t_max = 2.0 

# Run GFMI_C  
_, ssm_c =  main.run_ssm(case_directory=case_dir_gfmc) # Construct system and small-signal model
ssm_c.simulate_ssm(t_max=t_max, inputs=inputs)
sys_c, emt_sc_c = main.run_emt(case_directory=case_dir_gfmc, inputs=inputs, t_max=t_max) # Run EMT simulation

# Run GFMI_E
_, ssm_e =  main.run_ssm(case_directory=case_dir_gfme) # Construct system and small-signal model
ssm_e.simulate_ssm(t_max=t_max, inputs=inputs)
sys_e, emt_sc_e = main.run_emt(case_directory=case_dir_gfme, inputs=inputs, t_max=t_max) # Run EMT simulation


# # Compare eigenvalues of A matrix 
# gfmic_A = pd.read_csv(case_dir_gfmc+"/outputs/small_signal_model/A.csv")
# gfmie_A = pd.read_csv(case_dir_gfme+"/outputs/small_signal_model/A.csv")

# gfmie_eigs = np.linalg.eigvals(gfmie_A.iloc[:,1:].to_numpy())
# gfmic_eigs = np.linalg.eigvals(gfmic_A.iloc[:,1:].to_numpy())

# # Plot eigenvalues 
# fig = make_subplots(
#     rows=1, cols=1,
#     horizontal_spacing=0.15,
#     vertical_spacing=0.05,
# )

# fig.add_trace(go.Scatter(x=gfmie_eigs.real, y=gfmie_eigs.imag, name="GFMI_E", mode="markers", marker=dict(size=12, opacity=1.0, symbol="circle", line=dict(color="black", width=1))))
# fig.add_trace(go.Scatter(x=gfmic_eigs.real, y=gfmic_eigs.imag, name="GFMI_C", mode="markers", marker=dict(size=12, opacity=0.5, symbol="square", line=dict(color="black", width=1))))
# fig.show()
# fig.write_html("examples/small_signal_and_emt/2-bus-scr-gfm_comparison/eig_comparison.html")

# Compare EMT traces 
gfmi_c = pd.read_csv(case_dir_gfmc+"/outputs/simulation_emt/gfmi_c_0_states.csv")
gfmi_e = pd.read_csv(case_dir_gfme+"/outputs/simulation_emt/gfmi_e_0_states.csv")

fig = make_subplots(
    rows=9, cols=2,
    horizontal_spacing=0.15,
    vertical_spacing=0.05,
)

tps = gfmi_c["time"]

# converting abc to dq 
# GFMI_E 
i_bus_de, i_bus_qe, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(gfmi_e["i_bus_a"], gfmi_e["i_bus_b"], gfmi_e["i_bus_c"], gfmi_e["angle_pc"])]) 
i_vsc_de, i_vsc_qe, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(gfmi_e["i_vsc_a"], gfmi_e["i_vsc_b"], gfmi_e["i_vsc_c"], gfmi_e["angle_pc"])])
v_sh_de, v_sh_qe, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(gfmi_e["v_sh_a"], gfmi_e["v_sh_b"], gfmi_e["v_sh_c"], gfmi_e["angle_pc"])])

# GFMI_c 
i_bus_dc, i_bus_qc, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(gfmi_c["i_bus_a"], gfmi_c["i_bus_b"], gfmi_c["i_bus_c"], gfmi_c["angle_pc"])]) 
i_vsc_dc, i_vsc_qc, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(gfmi_c["i_vsc_a"], gfmi_c["i_vsc_b"], gfmi_c["i_vsc_c"], gfmi_c["angle_pc"])])
v_sh_dc, v_sh_qc, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(gfmi_c["v_sh_a"], gfmi_c["v_sh_b"], gfmi_c["v_sh_c"], gfmi_c["angle_pc"])])

r, c = 1,1 
fig.add_trace(go.Scatter(x=tps, y=gfmi_c["angle_pc"], name="GFMI_C",  mode='lines', line=dict(color='red', dash='solid'), legendgroup='1'), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=gfmi_e["angle_pc"], name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='angle_pc', row=r, col=c)

r, c = 1,2
fig.add_trace(go.Scatter(x=tps, y=gfmi_c["w_pc"], name="GFMI_C",  mode='lines', line=dict(color='red', dash='solid'), legendgroup='1'), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=gfmi_e["w_pc"], name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='w_pc', row=r, col=c)

r, c = 2,1
fig.add_trace(go.Scatter(x=tps, y=gfmi_c["p_pc"], name="GFMI_C",  mode='lines', line=dict(color='red', dash='solid'), legendgroup='1'), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=gfmi_e["p_pc"], name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='p_pc', row=r, col=c)

r, c = 2,2
fig.add_trace(go.Scatter(x=tps, y=gfmi_c["q_pc"], name="GFMI_C",  mode='lines', line=dict(color='red', dash='solid'), legendgroup='1'), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=gfmi_e["q_pc"], name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='q_pc', row=r, col=c)

r, c = 3,1
fig.add_trace(go.Scatter(x=tps, y=gfmi_c["gamma"], name="GFMI_C",  mode='lines', line=dict(color='red', dash='solid'), legendgroup='1'), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=gfmi_e["gamma"], name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='gamma', row=r, col=c)

r, c = 4,1
fig.add_trace(go.Scatter(x=tps, y=i_vsc_dc, name="GFMI_C",  mode='lines', line=dict(color='red', dash='solid'), legendgroup='1'), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=i_vsc_de, name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='i_vsc_d', row=r, col=c)

r, c = 4,2
fig.add_trace(go.Scatter(x=tps, y=i_vsc_qc, name="GFMI_C",  mode='lines', line=dict(color='red', dash='solid'), legendgroup='1'), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=i_vsc_qe, name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='i_vsc_q', row=r, col=c)

r, c = 5,1
fig.add_trace(go.Scatter(x=tps, y=v_sh_dc, name="GFMI_C",  mode='lines', line=dict(color='red', dash='solid'), legendgroup='1'), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=v_sh_de, name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='v_sh_d', row=r, col=c)

r, c = 5,2
fig.add_trace(go.Scatter(x=tps, y=v_sh_qc, name="GFMI_C",  mode='lines', line=dict(color='red', dash='solid'), legendgroup='1'), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=v_sh_qe, name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='v_sh_q', row=r, col=c)

r, c = 6,1
fig.add_trace(go.Scatter(x=tps, y=i_bus_dc, name="GFMI_C",  mode='lines', line=dict(color='red', dash='solid'), legendgroup='1'), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=i_bus_de, name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='i_bus_d', row=r, col=c)

r, c = 6,2
fig.add_trace(go.Scatter(x=tps, y=i_bus_qc, name="GFMI_C",  mode='lines', line=dict(color='red', dash='solid'), legendgroup='1'), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=i_bus_qe, name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='i_bus_q', row=r, col=c)

r, c = 7,1
fig.add_trace(go.Scatter(x=tps, y=gfmi_e["i_L"], name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='i_L', row=r, col=c)

r, c = 7,2
fig.add_trace(go.Scatter(x=tps, y=gfmi_e["v_dc"], name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='v_dc', row=r, col=c)

r, c = 8,1
fig.add_trace(go.Scatter(x=tps, y=gfmi_e["i_load"], name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='i_load', row=r, col=c)

r, c = 8,2
fig.add_trace(go.Scatter(x=tps, y=gfmi_e["soc"], name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='soc', row=r, col=c)

r, c = 9,1
p_load = np.multiply(gfmi_e["i_load"],gfmi_e["v_dc"])
fig.add_trace(go.Scatter(x=tps, y=p_load, name="GFMI_E",  mode='lines', line=dict(color='blue', dash='dot'), legendgroup='1'), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='p_load', row=r, col=c)



# r, c = 9, 1
# duty_cycle = self.kp_i_L*(self.kp_v_dc*(v_dc_ref - v_dcf) + x1 - i_Lf + self.Kff_idc*i_dcf + self.Kff_iload*i_loadf) + x2
# duty_cycle = np.clip(duty_cycle, 0.0, 1.0)
# fig.add_trace(go.Scatter(x=tps, y=duty_cycle, name="duty cycle", mode='lines', line=dict(color='blue', dash='dot')),
#             row=r, col=c)
# fig.update_xaxes(title_text='Time [s]', row=r, col=c)
# fig.update_yaxes(title_text='duty cycle [p.u.]', row=r, col=c)



fig.update_layout(height=1200*2, 
                  width=800*2, 
                  legend_tracegroupgap=400,
                  showlegend=False,
                  margin={'t': 0, 'l': 0, 'b': 0, 'r': 0})



fig.show()
fig.write_html("examples/small_signal_and_emt/gfmi_e_vs_gfmi_c_emt_traces.html")


## Print nadir/rocof 

# Measure some performance metrics - frequency ROCOF and nadir and steady state error ? 
gfm = emt_sc_c.system.gfmi_c[0].variables_emt.x.value
w_pc = gfm[1,:]
# nadir 
nadir_c = np.min(w_pc)
tps = np.linspace(0, t_max, 500)
dt = tps[1] - tps[0]

# rocof - calculate as a moving average of duration 100ms (~25 timesteps for us)
df_dt = np.abs(np.diff(w_pc)/dt) # calculate raw rocof 
df_dt_ma = np.convolve(df_dt,np.ones(25)/25) # get moving averages over 25 time steps (~100ms)
rocof_c = np.max(df_dt_ma)

gfm = emt_sc_e.system.gfmi_e[0].variables_emt.x.value
w_pc = gfm[1,:]
# nadir 
nadir_e = np.min(w_pc)
tps = np.linspace(0, t_max, 500)
dt = tps[1] - tps[0]

# rocof - calculate as a moving average of duration 100ms (~25 timesteps for us)
df_dt = np.abs(np.diff(w_pc)/dt) # calculate raw rocof 
df_dt_ma = np.convolve(df_dt,np.ones(25)/25) # get moving averages over 25 time steps (~100ms)
rocof_e = np.max(df_dt_ma)



# participation factor analysis - can turn this into a function / method for SSM class 

# def participation_factor_plot(ssm):
#     w,vr = eig(ssm.model.A)
#     vl = inv(vr) # ensures normalization 

#     p = np.zeros_like(vl)
#     for i in range(len(w)):
#         for k in range(len(w)):
#             # use correct formula for complex eigenvalues 
#             p[k, i] = (vl[i,k].real)**2/(np.dot(vl[i,:].real,vl[i,:].real)) # participation of k-th state (row) in i-th mode (column)
            
#     pf_df = pd.DataFrame(p.real, index=ssm.model.x.name)
#     ax = sns.heatmap(pf_df, yticklabels=pf_df.index, xticklabels=w.real>0, linewidths=1, linecolor='white')
#     ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
#     ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
#     plt.show()
#     plt.savefig("ssm.png")
    

# participation_factor_plot(ssm_c)
# participation_factor_plot(ssm_e)



print('ok')
