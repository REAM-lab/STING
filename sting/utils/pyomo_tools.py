import pyomo.environ as pyo
import polars as pl

def pyovariable_to_df(pyo_variable: pyo.Var, dfcol_to_field: dict, value_name: str, csv_filepath=None, given_dct = None ) -> pl.DataFrame:
    """
    Convert a Pyomo variable to a Polars DataFrame.

    #### Args:
        - pyo_variable: `pyo.Var` 
                    The Pyomo variable to convert.
        - dfcol_to_field: `dict[str]`
                    A dictionary mapping the fields of the variable's indices to the names of the columns in the resulting DataFrame.
        - value_name: `str`
                    The name of the column that will contain the variable's values. 
        - csv_filepath: `str`, optional
                    If provided, the DataFrame will be exported to a CSV file at this path.

    #### Returns:
        - df: `pl.DataFrame` 
                    A Polars DataFrame containing the variable's indices and values.
    """

    # Convert Pyomo variable to dictionary
    dct = pyo_variable.extract_values() if given_dct is None else given_dct

    def dct_to_tuple(dct_item):
        """
        Converts a dictionary item to a tuple.
        """
        k, v = dct_item
        
        if len(dfcol_to_field) == 1:
            k = [k]

        l = [None] * (len(dfcol_to_field) + 1)
        for i, (col_name, field) in enumerate(dfcol_to_field.items()):
            l[i] = getattr(k[i], field)
                
        l[-1] = v   
        return tuple(l)

    # Create schema for polars DataFrame
    schema = list(dfcol_to_field.keys()) + [value_name]

    # Create polars DataFrame
    df = pl.DataFrame(
        schema = schema,
        data= map(dct_to_tuple, dct.items())
    )

    # Export to CSV if filepath is provided
    if csv_filepath is not None:
        df.write_csv(csv_filepath)

    return df

def pyodual_to_df(model_dual: pyo.Suffix, pyo_constraint: pyo.Constraint, dfcol_to_field: dict, value_name: str, csv_filepath=None ) -> pl.DataFrame:
    """
    Convert Pyomo duals to a Polars DataFrame.
    """

    dims = list(pyo_constraint)

    dct = dict.fromkeys(dims, None)

    for i in dims:
        dct[i] = model_dual[pyo_constraint[i]]

    df = pyovariable_to_df(pyo_variable=None, dfcol_to_field=dfcol_to_field, value_name=value_name, csv_filepath=csv_filepath, given_dct = dct )

    return df

def pyoconstraint_to_df(pyo_constraint: pyo.Constraint, dfcol_to_field: dict, value_name: str, csv_filepath=None ) -> pl.DataFrame:
    """
    Convert Pyomo constraints to a Polars DataFrame.
    """

    dims = list(pyo_constraint)

    dct = dict.fromkeys(dims, None)

    for i in dims:
        dct[i] = pyo.value(pyo_constraint[i].body)

    df = pyovariable_to_df(pyo_variable=None, dfcol_to_field=dfcol_to_field, value_name=value_name, csv_filepath=csv_filepath, given_dct = dct )

    return df