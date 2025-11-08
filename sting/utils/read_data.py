import csv
import pandas as pd
import numpy as np

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
        with open(file_path, 'r', newline='') as csvfile:
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
    dtype_mapping = dict(zip(attr,types))
    df = pd.read_csv(file_path, header=1,dtype=dtype_mapping)
    
    return df

def generate_unique_index_column_in_dataframe(dataframe: pd.DataFrame,
                                              column_name: str,
                                              preffix: str):
    
    n = len(dataframe)
    list_of_numbers = list(range(1,n+1))
    dataframe[column_name] = [preffix + str(item) for item in list_of_numbers]
    col_to_move = dataframe.pop(column_name)
    dataframe.insert(0, column_name, col_to_move)
    
    return dataframe