# -----------------------
# Import Python packages
# -----------------------
import os
import itertools
from typing import get_type_hints
from dataclasses import dataclass, fields
import polars as pl
import time
import logging
import datetime

# -----------------------
# Import sting code
# -----------------------
from sting import __logo__
import sting.system.selections as sl

# -----------------------
# Import sting components
# -----------------------
from sting.system.component import Component, SystemComponent
from sting.bus.core import Bus, Load
from sting.generator.core import Generator, CapacityFactor
from sting.storage.core import Storage
from sting.generator.infinite_source import InfiniteSource
from sting.generator.gfmi_c import GFMIc
from sting.reduced_order_model.linear_rom import LinearROM
from sting.line.pi_model import LinePiModel
from sting.branch.series_rl import BranchSeriesRL
from sting.shunt.parallel_rc import ShuntParallelRC
from sting.timescales.core import Scenario, Timepoint, Timeseries
from sting.policies.carbon_policies.core import CarbonPolicy
from sting.policies.energy_budgets.core import EnergyBudget

# Set up logging
logger = logging.getLogger(__name__)
logger.info(__logo__) # print logo when a System instance is created

@dataclass(slots=True)
class System:

    # Settings and metadata
    case_directory: str = None
    components: list[SystemComponent] = None 
    type_to_class: dict[str, type] = None
    class_to_type: dict[type, str] = None

    # Components
    generators: list[Generator] = None
    capacity_factors: list[CapacityFactor] = None
    storage: list[Storage] = None
    infinite_sources: list[InfiniteSource] = None
    gfmi_c: list[GFMIc] = None
    linear_roms: list[LinearROM] = None
    buses: list[Bus] = None
    loads: list[Load] = None
    lines: list[LinePiModel] = None
    branch_series_rl: list[BranchSeriesRL] = None
    shunt_parallel_rl: list[ShuntParallelRC] = None
    timeseries: list[Timeseries] = None
    timepoints: list[Timepoint] = None
    scenarios: list[Scenario] = None
    energy_budgets: list[EnergyBudget] = None
    carbon_policies: list[CarbonPolicy] = None

    def __post_init__(self):

        logger.info(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") # Print datetime
        logger.info(__logo__)
        logger.info("> Initializing system ...")

        if self.case_directory is None:
            self.case_directory = os.getcwd()
            logger.info(f"> No case directory provided. Set to current directory: {self.case_directory}")

        non_components = ['case_directory', 'components', 'type_to_class', 'class_to_type'] # define attributes that are not grid components
        
        # Store the names of the component attributes in self.components
        self.components = [SystemComponent(type_=f.name, class_=f.type.__args__[0]) for f in fields(self) if f.name not in non_components]

        # Build type_to_class and class_to_type dictionaries for mapping between component type names and their classes
        self.type_to_class = {c.type_: c.class_ for c in self.components}
        self.class_to_type = {c.class_: c.type_ for c in self.components}

        # Initialize component attributes to empty lists if they are None    
        for c in self.components:
            if getattr(self, c.type_) is None:
                setattr(self, c.type_, [])

        logger.info(f" System initialization completed.")

    @classmethod
    def from_csv(cls, case_directory = None) -> 'System':
        """
        Read of component data from csv files and post system initialization.

        Add components from csv files. Each csv file has components of the same type.
        For example: gfli_a.csv contains ten gflis, but from the same type gfli_a.
        Each row of gfli_a.csv is a gfli_a that will be added to the system attribute gfli_a. 
        
        ### Inputs:
        - cls: `System` 
        - case_directory: `str` 
                        Directory of the case study. 
                        This directory has a folder "inputs" that has the csv files. 
                        By default it is current directory where we execute sting.
        ### Outputs:
        - self: `System`
                    It contains the components that have data from csv files.
        """
        full_start_time = time.time()

        self = cls(case_directory=case_directory) # Create instance of System.
        logger.info(f"> Loading system components from CSV files from {self.case_directory} ...") 

        for c in self.components:# For example, "generators", "infinite_sources", etc.
            csv_filename = f"{c.type_}.csv"
            filepath = os.path.join(self.case_directory, "inputs", csv_filename) # For example, "./case_directory/inputs/generators.csv"

            if not os.path.exists(filepath):
                continue
            
            # Read only 1 row to check column names, do not treat the first row as headers (header=None)
            df = pl.read_csv(filepath, n_rows=1, has_header=False)
            columns_csv = df.row(0)

            # Get the type of the component list, for example Generator, and then get the type hints of that class
            class_attribute_types = get_type_hints(c.class_) # For example, get_type_hints(Generator)
            
            # Filter out the pairs (key, value) from class_attribute_types that are not in columns_csv
            attributes_to_type = {key: value for key, value in class_attribute_types.items() if key in columns_csv}

            # Read components csv
            df = pl.read_csv(filepath, schema_overrides=attributes_to_type)

            # Create a component for each row (i.e., component) in the csv
            for row in df.iter_rows(named=True):
                component_instance = c.class_(**row) # For example, Generator(**row)
                # Add the component to the system
                self.add(component_instance)

            logger.info(f" - '{csv_filename}' ... ")
            logger.info(f"   Added {df.height} '{c.type_}' ")

        start_time = time.time()
        self.apply("post_system_init", self)
        logger.info(f"> Post system initialization completed in {time.time() - start_time:.2f} seconds.")
        logger.info(f"> Completed in: {time.time() - full_start_time:.2f} seconds. \n")

        return self
    
    def write_csv(self, types = [int, float, str, bool], output_directory=None):
        """Export system components to csv files."""

        if output_directory is None:
            output_directory = os.path.join(self.case_directory, "outputs", "system_csv")
        
        os.makedirs(output_directory, exist_ok=True)

        for (type_, _) in self.components:
          lst = getattr(self, type_)
          csv_filename = f"{type_}.csv"
          if lst:
              # Assumes each component is a dataclass with fields
              cols = fields(lst[0])
              cols = [c.name for c in cols if c.type in types]
              df = self.query([type_]).to_table(*cols, index = 'id', index_name = 'id') # we need [type_] to be a list to query multiple types if needed, for example [type1, type2]
              df.to_csv(os.path.join(output_directory, csv_filename))

    # ------------------------------------------------------------
    # Component Management + Searching
    # ------------------------------------------------------------
    def add(self, component: Component):
        """Add a new component to the system."""
        # Get the component type (for example, "generators") 
        component_type = self.class_to_type[type(component)]
        # Get the list of components of that type, for example self.generators
        component_list: list[Component] = getattr(self, component_type)
        # Assign the component a 0-based index value
        component.id = len(component_list)
        # Assign type attribute to the component, for example "infinite_sources"
        component.type_ = component_type
        # Add the component to the list
        component_list.append(component)

        return component.id
        
    def _generator(self, names) -> itertools.chain:
        # Collect all lists of components in the component_types
        all_components = [getattr(self, name) for name in names]
         # Yield all components following the order in component_types
        return itertools.chain(*all_components)

    def query(self, *args):
        """
        Return a Stream over a set of component types. Analogous to FROM in 
        SQL, specifying which tables to access data from. For example, 
        "FROM gfmi_a, inf_src SELECT id" would be written as:
        >>> power_sys.query("gfmi_a", "inf_src").select("id")
        
        If no tables are provided runs a Stream over all components.
        """
        if not args:
            return sl.Stream(self, index_map=self.class_to_type)
        # Unpack all args calling on self if they are a function
        names = [arg(self) if callable(arg) else arg for arg in args]
        # Flatten the list of component types to query from
        names = itertools.chain(*names)

        return sl.Stream(self._generator(names), index_map=self.class_to_type)

    def __iter__(self):
        """Iterate over all components in the system."""
        return self._generator([c.type_ for c in self.components])
    
    def apply(self, method, *args):
        """Call a given method on all components with the method."""
        for component in self:
            if hasattr(component, method):
                getattr(component, method)(*args)

    @property
    def gens(self):
        """Return a lazy Stream (like list) of all components with the tag "generator"."""
        return self.query(self.find_tagged("generator"))

    @property
    def shunts(self):
        """Return a lazy Stream (like list) of all components with the tag "shunt"."""
        return self.query(self.find_tagged("shunt"))

    @property
    def branches(self):
        """Return a lazy Stream (like list) of all components with the tag "branch"."""
        return self.query(self.find_tagged("branch"))
    
    # ------------------------------------------------------------
    # Common selections
    # ------------------------------------------------------------
    def find_tagged(self, tag_name: str) -> list[str]:
        """
        Return a list of all components tagged with a specific tag name.
        """
        # List of all components with the given tag name
        tagged_components = []

        # Scan over all component types
        for (type_, _) in self.components:
            component_list = getattr(self, type_)
            # If the component is tagged with the current tag name, add it to the running list
            if  (len(component_list) > 0 and 
                hasattr(component_list[0], "tags") and 
                (tag_name in component_list[0].tags)):
                tagged_components.append(type_)

        return tagged_components

    def __repr__(self):
        """Return a string representation of the system, showing the number of components of each type."""
        lines_to_display = []

        lines_to_display.append(f"System: ")

        for (c_type, _) in self.components:
            if len(getattr(self, c_type)) > 0:
                lines_to_display.append(f"  - {c_type}: {len(getattr(self, c_type))}")

        return "\n".join(lines_to_display)
