from plotly.subplots import make_subplots
import plotly.graph_objects as go
import polars as pl
import os

def compare_two_timeseries(df1: pl.DataFrame, 
                           df2: pl.DataFrame, 
                           compare: dict, 
                           df1_name: str, 
                           df2_name: str, 
                           figure_filepath: str = os.path.join(os.getcwd(), "comparison_plot.html"),
                           d1_color: str = 'blue', 
                           d2_color: str = 'red'):
    """
    Compare two timeseries dataframes and plot the results.
    
    Parameters:
    - df1: First dataframe containing timeseries data.
    - df2: Second dataframe containing timeseries data.
    - compare: Dictionary where keys are column names in df1 and values are corresponding column names in df2 to compare, e.g. {"p_sh": "p_sh", "q_sh": "q_sh"}.
    - df1_name: Name of the first dataframe (for labeling purposes), e.g., "EMT".
    - df2_name: Name of the second dataframe (for labeling purposes), e.g., "SSM".
    - figure_filepath: File path to save the resulting plot (e.g., "comparison_plot.html").
    - d1_color: Color for the first dataframe's traces (default is 'blue').
    - d2_color: Color for the second dataframe's traces (default is 'red').

    Returns:
    - None. The function saves the plot to the specified file path, defaulting to "comparison_plot.html" in the current working directory. Open it in a web browser to view the comparison.
    """

    # Number of subplots to create
    nplots = len(compare)

    # Create subplots
    ncols = 2
    nrows = nplots // ncols + int(nplots % ncols > 0)
        
    fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True)
    for i, (df1_col, df2_col) in enumerate(compare.items()):
        row = i // ncols + 1
        col = i % ncols + 1
        fig.add_trace(go.Scatter(x=df1['time'], y=df1[df1_col], name=f"{df1_name}: {df1_col}"), row=row, col=col, line=dict(color=d1_color))
        fig.add_trace(go.Scatter(x=df2['time'], y=df2[df2_col], name=f"{df2_name}: {df2_col}"), row=row, col=col, line=dict(color=d2_color))
        fig.update_xaxes(title_text='Time [s]',row=row, col=col)

    fig.write_html(figure_filepath)

