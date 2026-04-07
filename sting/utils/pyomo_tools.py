# ----------------------
# Import python packages
# ----------------------
import pyomo.environ as pyo
import polars as pl
import math
import logging
from pyomo.repn import generate_standard_repn

# ------------------
# Import sting code
# ------------------
from sting.utils.runtime_tools import timeit

# Set up logging
logger = logging.getLogger(__name__)


@timeit
def inspect_coefficients(model: pyo.ConcreteModel):
    """
    Inspection of coefficients in the model.
    """

    min_coef = math.inf
    max_coef = 0

    min_coef_info = None
    max_coef_info = None

    min_rhs = math.inf
    max_rhs = 0

    min_rhs_info = None
    max_rhs_info = None

    for c in model.component_data_objects(pyo.Constraint, active=True):
        repn = generate_standard_repn(c.body, compute_values=False)

        for var, coef in zip(repn.linear_vars, repn.linear_coefs):
            val = abs(coef)

            if val == 0:
                continue

            if val < min_coef:
                min_coef = val
                min_coef_info = (coef, c.name, var.name)

            if val > max_coef:
                max_coef = val
                max_coef_info = (coef, c.name, var.name)

        if c.lower is not None:
            v = abs(pyo.value(c.lower))
            if v < min_rhs and v != 0:
                min_rhs = v
                min_rhs_info = (pyo.value(c.lower), c.name, "lower")
            if v > max_rhs:
                    max_rhs = v
                    max_rhs_info = (pyo.value(c.lower), c.name, "lower")


        if c.upper is not None:
            v = abs(pyo.value(c.upper))
            if v < min_rhs and v != 0:
                min_rhs = v
                min_rhs_info = (pyo.value(c.upper), c.name, "upper")
            if v > max_rhs:
                max_rhs = v
                max_rhs_info = (pyo.value(c.upper), c.name, "upper")

    obj = next(model.component_data_objects(pyo.Objective, active=True))

    repn = generate_standard_repn(obj.expr, compute_values=False)

    min_obj_coef = math.inf
    max_obj_coef = 0
    min_obj_info = None
    max_obj_info = None

    for var, coef in zip(repn.linear_vars, repn.linear_coefs):

        val = abs(coef)

        if val < min_obj_coef:
            min_obj_coef = coef
            min_obj_info = (coef, var.name)

        if val > max_obj_coef:
            max_obj_coef = coef               
        max_obj_info = (coef, var.name)

    logger.info(f"  - Matrix coefficient extremes:")
    logger.info(f"     - Minimum coefficient: {min_coef_info[0]}")
    logger.info(f"       Constraint={min_coef_info[1]}  ")
    logger.info(f"       Variable={min_coef_info[2]}")
    logger.info(f"     - Maximum coefficient: {max_coef_info[0]}")
    logger.info(f"       Constraint={max_coef_info[1]}  ")
    logger.info(f"       Variable={max_coef_info[2]} \n")
    logger.info(f"  - RHS extremes:")
    logger.info(f"     - Minimum RHS: {min_rhs_info[0]}")
    logger.info(f"       Constraint={min_rhs_info[1]}  ")
    logger.info(f"       Bound={min_rhs_info[2]}")
    logger.info(f"     - Maximum RHS: {max_rhs_info[0]}")
    logger.info(f"       Constraint={max_rhs_info[1]}  ")
    logger.info(f"       Bound={max_rhs_info[2]} \n")

    logger.info(f"  - Objective coefficient extremes:")
    logger.info(f"     - Minimum coefficient: {min_obj_info[0]}")
    logger.info(f"       Variable={min_obj_info[1]}")
    logger.info(f"     - Maximum coefficient: {max_obj_info[0]}")
    logger.info(f"       Variable={max_obj_info[1]}")

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