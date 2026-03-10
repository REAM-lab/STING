
"""
In progress (March 9, 2026 - Ruth) 

Compares the small signal models for the GFMIc and GFMIe 

To do - compare EMT

"""
# Import Python standard and third-party packages
import numpy as np
import pandas as pd 
from pathlib import Path
import os
import plotly.graph_objects as go

# GFMIe test case folder - t3
# GFMIc test case folder - t4

# Compare initial conditions
gfmie_init = pd.read_csv(os.path.join(Path(__file__).resolve().parent,"t3/outputs/small_signal_model/x.csv"))
gfmic_init = pd.read_csv(os.path.join(Path(__file__).resolve().parent,"t4/outputs/small_signal_model/x.csv"))

# Compare eigenvalues of A matrix 
gfmie_A = pd.read_csv(os.path.join(Path(__file__).resolve().parent,"t3/outputs/small_signal_model/A.csv"))
gfmic_A = pd.read_csv(os.path.join(Path(__file__).resolve().parent,"t4/outputs/small_signal_model/A.csv"))

gfmie_eigs = np.linalg.eigvals(gfmie_A.iloc[:,1:].to_numpy())
gfmic_eigs = np.linalg.eigvals(gfmic_A.iloc[:,1:].to_numpy())

# Plot eigenvalues 
fig = go.Figure()

fig.add_trace(go.Scatter(x=gfmie_eigs.real, y=gfmie_eigs.imag, name="GFMI_E", mode="markers", marker=dict(size=12, opacity=1.0, symbol="circle", line=dict(color="black", width=1))))
fig.add_trace(go.Scatter(x=gfmic_eigs.real, y=gfmic_eigs.imag, name="GFMI_C", mode="markers", marker=dict(size=12, opacity=0.5, symbol="square", line=dict(color="black", width=1))))
fig.show()

print('ok')


