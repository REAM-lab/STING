import csv
import pandas as pd
import numpy as np
from dataclasses import is_dataclass, asdict
import pyomo.environ as pyo
import polars as pl
import logging
import os
import functools
import time
import logging
from typing import Callable
import pathlib as Path

def read_specific_csv_row(file_path, row_index):
    """
    Reads a specific row from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        row_index (int): The 0-indexed number of the row to read.

    Returns:
        list: A list of strings representing the elements of the specified row,
              or None if the row_index is out of bounds.
    """
    try:
        with open(file_path, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            for index, row in enumerate(csv_reader):
                if index == row_index:
                    return row
        return None  # If row_index is out of bounds
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None


def read_single_dataframe(file_path):
    """
    Reads a single dataframe from a csv file.

    Args:
        file_path (str): The path to the CSV file. The first row of the CSV must have the types.
    Returns:
        dataframe: A dataframe that fully represents the dataframe (except first row).

    """
    types = read_specific_csv_row(file_path, 0)
    attr = read_specific_csv_row(file_path, 1)
    dtype_mapping = dict(zip(attr, types))
    df = pd.read_csv(file_path, header=1, dtype=dtype_mapping)

    return df


def generate_unique_index_column_in_dataframe(
    dataframe: pd.DataFrame, column_name: str, preffix: str
):

    n = len(dataframe)
    list_of_numbers = list(range(1, n + 1))
    dataframe[column_name] = [preffix + str(item) for item in list_of_numbers]
    col_to_move = dataframe.pop(column_name)
    dataframe.insert(0, column_name, col_to_move)

    return dataframe


def convert_class_instance_to_dictionary(instance: object, excluded_attributes=None):

    if excluded_attributes is None:
        excluded_attributes = []

    # Create dictionary (attribute: value)
    if is_dataclass(instance):
        dicty = asdict(instance)
    elif isinstance(instance, tuple):
        dicty = instance._asdict()

    # Filter out some attributes inputted into the function.
    dicty = {
        key: value for key, value in dicty.items() if not key in excluded_attributes
    }

    # Filter out None values as matlab engine cannot handle None types
    dicty = {key: value for key, value in dicty.items() if value is not None}

    # If value is a NamedTuple (commonly used) then transform to dictionary
    dicty = {
        key: (value._asdict() if isinstance(value, tuple) else value)
        for key, value in dicty.items()
    }

    # If value is a dataclass (commonly used) then transform to dictionary
    dicty = {
        key: (asdict(value) if is_dataclass(value) else value)
        for key, value in dicty.items()
    }

    return dicty


def matrix_to_csv(filepath: str, matrix: np.ndarray, index: list, columns: list):
    df = pl.DataFrame(matrix, schema=columns)
    df = df.with_columns(
                        pl.Series(name="Index", values=index)).select(
                        # Reorder columns to place the 'Index' first
                        ["Index"] + columns
        )
    df.write_csv(filepath)


def csv_to_matrix(filepath):
    df = pd.read_csv(filepath)
    df.set_index(df.columns[0], inplace=True)
    # Unpack matrix, index, and columns
    matrix = df.to_numpy()
    index = df.index.tolist()
    columns = list(df.columns)
    return matrix, index, columns


def mat2cell(A: np.ndarray, m: list, n: list) -> np.ndarray:
    """Python clone of MATLAB mat2cell"""
    if m is None:
        m = [A.shape[0]]
    if n is None:
        n = [A.shape[1]]

    # Create a list of lists to hold the sub-arrays
    cell_array = np.empty((len(m), len(n)), dtype=object)

    r_start = 0
    for i, r_len in enumerate(m):
        c_start = 0
        for j, c_len in enumerate(n):
            cell_array[i,j] = A[
                r_start: r_start + r_len, c_start : c_start + c_len
            ]
            c_start += c_len
        r_start += r_len

    return cell_array


def cell2mat(C):
    """Python clone of MATLAB cell2mat"""
    if len(C) == 1:
        return np.vstack(C)
    return np.vstack([np.hstack(row) for row in C])


def block_permute(X, rows, cols, index):
    # Returns a matrix Y whose block matrix entries (given by rows and
    # cols) have been re-ordered according to an index.
    X = mat2cell(X, rows, cols)
    Y = np.empty(X.shape, dtype=object)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Y_ij = X_nm where the index maps n/m to i/j
            n = index[i]
            m = index[j]
            Y[i, j] = X[n, m]
    return cell2mat(Y)

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

def setup_logging_file(case_directory: str):
    """Setup file logging to the specified case directory."""

    file_path = os.path.join(case_directory, "sting_log.txt")
    Path.Path(file_path).touch(exist_ok=True) 
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    #file_handler.terminator = ''  # Remove automatic newline
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Also set terminator for console handler (StreamHandler)
    #for handler in root_logger.handlers:
    #    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
    #        handler.terminator = ''
    
    return file_handler

def timeit(func: Callable):
    """
    A decorator that measures the execution time of the decorated function.
    """
    first_line = func.__doc__.strip().splitlines()[0].strip()
    first_line = first_line.rstrip('.')
    first_line = first_line[0].lower() + first_line[1:]
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"> Initializing {first_line} ... ")
        start_time = time.perf_counter()  # Use perf_counter for more precise timing
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time)
        logging.info(f"> Completed in {elapsed_time:.2f} seconds. \n")
        return result
    return wrapper