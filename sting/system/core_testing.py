# -----------------------
# Import Python packages
# -----------------------
import os
import itertools
from typing import get_type_hints
from dataclasses import fields
import numpy as np 
from more_itertools import transpose
from scipy.linalg import block_diag
from scipy.integrate import solve_ivp
import polars as pl
import time
import logging
import datetime
import copy
from dataclasses import dataclass, fields, field

# -----------------------
# Import sting code
# -----------------------
from sting import __logo__
from sting import data_files
from sting.bus.bus import Bus
from sting.line.core import decompose_lines
from sting.utils.data_tools import timeit, convert_class_instance_to_dictionary
# from sting.shunt.core import combine_shunts
from sting.utils.graph_matrices import get_ccm_matrices, build_ccm_permutation
from sting.utils.dynamical_systems import StateSpaceModel, DynamicalVariables
import sting.system.selections as sl
import sting.bus.bus as bus
import sting.generator.generator as generator
import sting.generator.storage as storage
from sting.generator.infinite_source import InfiniteSource
from sting.generator.generator import Generator

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class System:
    case_directory: str = None
    components: list = None
    generators: list[Generator] = None
    infinite_sources: list[InfiniteSource] = None

    def __post_init__(self):

        logger.info(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") # Print datetime
        logger.info(__logo__) # print logo when a System instance is created
        logger.info("> Initializing system ...")

        if self.case_directory is None:
            self.case_directory = os.getcwd()
            logger.info(f"> No case directory provided. Set to current directory: {self.case_directory}")

        for component in fields(self):
            
        for component in fields(self):
            if getattr(self, component.name) is None:
                setattr(self, component.name, [])
        logger.info(f"System created with attributes: {[f.name for f in fields(self)]}")

    @classmethod
    def from_csv(cls, case_directory = None):

        self = cls(case_directory=case_directory) # Create instance of System.
        logger.info(f"> Loading system components from CSV files from {self.case_directory} ...") 

        for component in fields(self):
            component_name = component.name # For example, "generators", "infinite_sources", etc.
            csv_filename = f"{component_name}.csv"
            filepath = os.path.join(self.case_directory, "inputs", csv_filename) # For example, "./case_directory/inputs/generators.csv"
            logger.info(f" - '{csv_filename}' ... ")

            # Read only 1 row to check column names, do not treat the first row as headers (header=None)
            df = pl.read_csv(filepath, n_rows=1, has_header=False)
            columns_csv = df.row(0)

            # Get the type of the component list, for example Generator, and then get the type hints of that class
            class_attribute_types = get_type_hints(component.type.__args__[0]) 
            
            # Filter out the pairs (key, value) from class_attribute_types that are not in columns_csv
            attributes_to_type = {key: value for key, value in class_attribute_types.items() if key in columns_csv}

            # Read components csv
            df = pl.read_csv(filepath, dtypes=attributes_to_type)

            # Create a component for each row (i.e., component) in the csv
            for row in df.iter_rows(named=True):
                component_instance = component.type.__args__[0](**row) # For example, Generator(**row)
                # Add the component to the system
                self.add(component_instance)

            logger.info(f"   Added {df.height} '{component_name}' components. ")




