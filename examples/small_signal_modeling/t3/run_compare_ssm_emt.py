"""
In progress (March 6, 2026 - Ruth) 

Simulates a a GFMIe connected to an infinite source via a transmission line.

1. Computes and simulates the small-signal model using STING. 

2. Runs EMT using sting. 

3. Compares corresponding SSM and EMT states in plots. 

"""

# Import Python standard and third-party packages
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Callable

# Import sting package
from sting import main
from sting.utils.transformations import dq02abc, abc2dq0
from sting.modules.small_signal_modeling.core import SmallSignalModel
from sting.system.core import System
from sting.system.component import Component
from sting.system.operations import SystemModifier
from sting.modules.power_flow.core import ACPowerFlow
from sting.modules.simulation_emt.core import SimulationEMT
 

# Define additional functions that return required data in order to compare SSM and EMT

def simulate_ssm_ruth(ssm: SmallSignalModel, t_max: float, inputs: dict[str, dict[str, Callable[[float], float]]] = None, settings={'dense_output': True, 'method': 'Radau', 'max_step': 0.001}):
    """Simulate the small-signal model under a given input profile."""
    
    x0 = np.zeros_like(ssm.model.x.init)
    tps, solution = ssm.model.simulate(t_max=t_max, inputs=inputs, x0=x0, settings=settings, output_directory=ssm.output_directory, plot=False)

    # Add the initial conditions back to the solution (for plotting purposes)
    for i in range(len(ssm.model.x.init)):
        solution[i] = solution[i] + ssm.model.x.init[i]
    
    components_to_plot = np.unique(ssm.model.x.component) # Get the components in the same order as solution vector
    i = 0 # Initialize counter 
    
    name_list = []
    component_list = []
    # Make a html file for each component. Each file plots the states corresponding to each component.
    for component in components_to_plot:
        number_of_states = sum(ssm.model.x.component == component)
        nrows = int(np.ceil(number_of_states / 2))
        ncols = 2 if number_of_states > 1 else 1
        #fig = make_subplots(rows=nrows, cols=ncols)
        for j in range(number_of_states):
            row = j // ncols + 1
            col = j % ncols + 1
            #fig.add_trace(go.Scatter(x=tps, y=solution[i]), row=row, col=col)
            #fig.update_xaxes(title_text='Time [s]', row=row, col=col)
            #fig.update_yaxes(title_text=ssm.model.x.name[i], row=row, col=col)
            name_list.append(ssm.model.x.name[i]) # added 
            component_list.append(component)
            i += 1
            

        #fig.update_layout(title_text = component, title_x=0.5, showlegend = False)
        #fig.write_html(os.path.join(ssm.output_directory, f"{component}.html"))
    
    return tps, solution, name_list, component_list 

def run_emt_ruth(t_max, inputs, case_directory=os.getcwd(), model_settings=None, solver_settings=None):
    """
    Routine to simulate the EMT dynamics of the system from a case study directory.
    """

    # Load system from CSV files
    sys = System.from_csv(case_directory=case_directory)

    # Run power flow
    pf = ACPowerFlow(system=sys, model_settings=model_settings, solver_settings=solver_settings)
    pf.solve()

    # Break down lines into branches and shunts for small-signal modeling
    sys_modifier = SystemModifier(system=sys)
    sys_modifier.decompose_lines()

    # Construct small-signal model
    ssm = SmallSignalModel(system=sys)
    ssm.construct_system_ssm()

    emt_sc = SimulationEMT(system=sys)
    emt_sc.sim(t_max, inputs)

    return sys, emt_sc


# Specify path of the case study directory
case_dir = Path(__file__).resolve().parent

# Construct system and small-signal model
def step1(t):
    return 0.1 if t >= 0.5 else 0.0

def step2(t):
    return 0.0

def step3(t):
    return 0.1 if t > 1.0 else 0.0 

# Specify inputs to excite - any constant input does not need to be specified 
# NB: input is a perturbation from the nominal value 
inputs = {'infinite_sources_0': {'v_ref_d': step3}, 
          'gfmi_e_0': {'p_ref': step2, 
                       'q_ref': step2,
                       'v_ref': step2,
                       'v_dc_ref': step2,
                       'v_s': step1, 
                       'Pload': step2}}

t_max = 4.0

# Build and perturb small-signal model 
_, ssm = main.run_ssm(case_directory=case_dir)
tps, sol, ssm_state_names, component_name = simulate_ssm_ruth(ssm, t_max=t_max, inputs=inputs)

# Retrieve state traces 
i_bus_d_inf = sol[0,:] # infinite source
i_bus_q_inf = sol[1,:] # infinite source 
angle_pc = sol[2,:]
w_pc = sol[3,:]
p_pc = sol[4,:]
q_pc = sol[5,:]
pi_vc = sol[6,:]
i_vsc_d = sol[7,:]
i_vsc_q = sol[8,:]
i_bus_d = sol[9,:] # gfmie
i_bus_q = sol[10,:] # gfmie 
v_sh_d = sol[11,:]
v_sh_q = sol[12,:]
iLf = sol[13,:]
vdcf = sol[14,:]
idcf = sol[15,:]
iloadf = sol[16,:]
x1 = sol[17,:]
x2 = sol[18,:]
iL = sol[19,:]
vdc = sol[20,:]
vdcfL = sol[21,:]
v_bus_D_rc0 = sol[22,:] # RC0 
v_bus_Q_rc0 = sol[23,:] # RC0 
v_bus_D_rc1 = sol[24,:] #RC1
v_bus_Q_rc1 = sol[25,:] #RC1
i_br_D_br0 = sol[26,:] #BR0
i_br_Q_br0 = sol[27,:] #BR0

# Run EMT 
sys, emt_sc = run_emt_ruth(case_directory=case_dir, inputs=inputs, t_max=t_max)
emt_state_names = emt_sc.variables.x.name 
# Define timepoints that will be used to evaluate the solution of the ODEs
tps = np.linspace(0, t_max, 500)
n_tps = len(tps)

# Extract solutions directly from each component 
gfm = emt_sc.system.gfmi_e[0].variables_emt.x.value
gfm_names = emt_sc.system.gfmi_e[0].variables_emt.x.name

angle_pc_emt = gfm[0,:]
w_pc_emt = gfm[1,:]
p_pc_emt = gfm[2,:]
q_pc_emt = gfm[3,:]
gamma_emt = gfm[4,:]
i_vsc_a = gfm[5,:]
i_vsc_b = gfm[6,:]
i_vsc_c = gfm[7,:]
v_sh_a = gfm[8,:]
v_sh_b = gfm[9,:]
v_sh_c = gfm[10,:]
i_bus_a = gfm[11,:]
i_bus_b = gfm[12,:]
i_bus_c = gfm[13,:]
iLf_emt = gfm[14,:]
vdcf_emt = gfm[15,:]
idcf_emt = gfm[16,:]
iloadf_emt = gfm[17,:]
x1_emt = gfm[18,:]
x2_emt = gfm[19,:]
iL_emt = gfm[20,:]
vdc_emt = gfm[21,:]
vdcfL_emt = gfm[22,:]

# convert abc to dq for plotting comparison 
angle_pc_emt_init = sys.gfmi_e[0].emt_init.angle_ref * np.pi / 180

i_bus_d_emt, i_bus_q_emt, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_bus_a, i_bus_b, i_bus_c, angle_pc_emt)]) #? what angle to use here

i_vsc_d_emt, i_vsc_q_emt, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(i_vsc_a, i_vsc_b, i_vsc_c, angle_pc_emt)])

v_sh_d_emt, v_sh_q_emt, _ = zip(*[abc2dq0(a, b, c, ang) for a, b, c, ang in zip(v_sh_a, v_sh_b, v_sh_c, angle_pc_emt)])

fig = make_subplots(
    rows=11, cols=2,
    horizontal_spacing=0.15,
    vertical_spacing=0.05,
)

r, c = 1,1 # define row, column for convenience 
# blank 

r, c = 1,2 
fig.add_trace(go.Scatter(x=tps, y=angle_pc_emt, name="angle_pc (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=angle_pc, name="angle_pc (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='angle_pc', row=r, col=c)


r, c = 2,1 
fig.add_trace(go.Scatter(x=tps, y=gamma_emt, name="gamma (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=pi_vc, name="gamma (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='gamma', row=r, col=c)

r, c = 2, 2
fig.add_trace(go.Scatter(x=tps, y=w_pc_emt, name="w_pc (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=w_pc, name="w_pc (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='w_pc', row=r, col=c)

r, c = 3,1
fig.add_trace(go.Scatter(x=tps, y=p_pc_emt, name="p_pc (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=p_pc, name="p_pc (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='p_pc', row=r, col=c)

r, c = 3,2 
fig.add_trace(go.Scatter(x=tps, y=q_pc_emt, name="q_pc (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=q_pc, name="q_pc (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='q_pc', row=r, col=c)

r, c = 4,1 
fig.add_trace(go.Scatter(x=tps, y=i_vsc_d_emt, name="i_vsc_d (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=i_vsc_d, name="i_vsc_d (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='i_vsc_d', row=r, col=c)

r, c = 4,2 
fig.add_trace(go.Scatter(x=tps, y=i_vsc_q_emt, name="i_vsc_q (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=i_vsc_q, name="i_vsc_q (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='i_vsc_q', row=r, col=c)


r, c = 5,1 
fig.add_trace(go.Scatter(x=tps, y=v_sh_d_emt, name="v_sh_d (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=v_sh_d, name="v_sh_d (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='v_sh_d', row=r, col=c)


r, c = 5,2 
fig.add_trace(go.Scatter(x=tps, y=v_sh_q_emt, name="v_sh_q (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=v_sh_q, name="v_sh_q (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='v_sh_q', row=r, col=c)

r, c = 6,1 
fig.add_trace(go.Scatter(x=tps, y=i_bus_d_emt, name="i_bus_d (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=i_bus_d, name="i_bus_d (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='i_bus_d', row=r, col=c)

r, c = 6,2 
fig.add_trace(go.Scatter(x=tps, y=i_bus_q_emt, name="i_bus_q (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=i_bus_q, name="i_bus_q (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='i_bus_q', row=r, col=c)

r, c = 7,1 
fig.add_trace(go.Scatter(x=tps, y=iLf_emt, name="iLf (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=iLf, name="iLf (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='iLf', row=r, col=c)


r, c = 7,2 
fig.add_trace(go.Scatter(x=tps, y=vdcf_emt, name="vdcf (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=vdcf, name="vdcf (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='vdcf', row=r, col=c)

r, c = 8,1 
fig.add_trace(go.Scatter(x=tps, y=idcf_emt, name="idcf (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=idcf, name="idcf (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='idcf', row=r, col=c)

r, c = 8,2 
fig.add_trace(go.Scatter(x=tps, y=iloadf_emt, name="iloadf (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=iloadf, name="iloadf (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='iloadf', row=r, col=c)

r, c = 9,1 
fig.add_trace(go.Scatter(x=tps, y=x1_emt, name="x1 (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=x1, name="x1 (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='x1', row=r, col=c)

r, c = 9,2 
fig.add_trace(go.Scatter(x=tps, y=x2_emt, name="x2 (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=x2, name="x2 (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='x2', row=r, col=c)


r, c = 10,1 
fig.add_trace(go.Scatter(x=tps, y=iL_emt, name="iL (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=iL, name="iL (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='iL', row=r, col=c)

r, c = 10,2 
fig.add_trace(go.Scatter(x=tps, y=vdc_emt, name="vdc (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=vdc, name="vdc (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='vdc', row=r, col=c)


r, c = 11,1 
fig.add_trace(go.Scatter(x=tps, y=vdcfL_emt, name="vdcfL (emt)",  mode='lines', line=dict(color='red', dash='solid')), row=r, col=c)
fig.add_trace(go.Scatter(x=tps, y=vdcfL, name="vdcfL (ssm)",  mode='lines', line=dict(color='blue', dash='dash')), row=r, col=c)
fig.update_xaxes(title_text='Time [s]', row=r, col=c)
fig.update_yaxes(title_text='vdcfL', row=r, col=c)

fig.update_layout(height=1200*4, 
                  width=800*2, 
                  showlegend=True,
                  margin={'t': 0, 'l': 0, 'b': 0, 'r': 0})
fig.show()
fig.write_html("examples/small_signal/t3/ssm_emt_comparison.html")
print('ok')
