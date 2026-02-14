# -----------------------
# Import Python packages
# -----------------------
import importlib
import os
import itertools
from typing import get_type_hints
from dataclasses import fields
import polars as pl
import time
import logging
import datetime
import copy

# -----------------------
# Import sting code
# -----------------------
from sting import __logo__
from sting import data_files
from sting.bus.bus import Bus
from sting.line.core import decompose_lines
from sting.utils.data_tools import timeit, convert_class_instance_to_dictionary
# from sting.shunt.core import combine_shunts
import sting.system.selections as sl
import sting.bus.bus as bus
import sting.generator.generator as generator
import sting.generator.storage as storage

logger = logging.getLogger(__name__)

class System:
    """
    A power system object comprised of multiple components. 

    A list components of all possible are components within the system
    can be supplied during initialization. If no such list is supplied, 
    the power system will be initialized to accommodate all components 
    in ./data_files/components_metadata.csv.
    """
    # ------------------------------------------------------------
    # Construction + Read/Write
    # ------------------------------------------------------------
    def __init__(self, components=None, case_directory=os.getcwd()):
        """
        Create attributes for the system that correspond to different types of components
        For example: if we type sys = System(), then sys will have the attributes
        sys.gfli_a, sys.gfmi_c, etc, and each of them initialized with empty lists []. 
        Each of these component types are by default given in the file components_metadata.csv.

        ### Inputs:
        - self (System instance)
        - components (list): Type of components, for example components=['gfli_a', 'gfmi_c'].

        ### Outputs:
        - self.components (dataframe): Stores the list of components, modules, classes and csv files.
        - self.class_to_str (dict): Maps class with type for each component. For example, InfiniteSource => inf_src
        """
        # Print datetime
        logger.info(f"{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

        # Print logo
        logo = __logo__.replace("\x1b[93m", "")  # For environments that do not support ANSI colors
        logo = logo.replace("\x1b[0m", "")  # For environments that do not support ANSI colors
        logger.info(logo) # print logo when a System instance is created

        logger.info("> Initializing system ...")

        # Get components_metadata.csv as a dataframe.
        # This file contains information of the lists of components that integrate the system
        data_dir = os.path.dirname(data_files.__file__) # get directory of data_files
        filepath = os.path.join(data_dir, "components_metadata.csv") # get directory 
        self.components = pl.read_csv(filepath) # get list of components as dataframe

        # If components are given, only use the relevant meta-data
        if components:
            active_components = self.components["type"].isin(components)
            self.components = self.components[active_components]

        # Mapping of a components class to its string representation
        self.class_to_str = dict(zip(self.components["class"], self.components["type"]))

        # Create a new attribute (an empty list) for each component type 
        for component_name in self.components["type"]:
            setattr(self, component_name, [])

        # Store case directory
        self.case_directory = case_directory

        logger.info("   Completed")

    @classmethod
    def from_csv(cls, components=None, case_directory=os.getcwd()):
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
        - components: `list`
                        Type of components, for example components=['gfli_a', 'gfmi_c'].
        
        ### Outputs:
        - self: `System`
                    It contains the components that have data from csv files.
        """
        full_start_time = time.time()

        # Get directory of the folder "inputs"
        inputs_dir = os.path.join(case_directory, "inputs")

        # Create instance System.
        self = cls(components=components, case_directory=case_directory)
        
        logger.info(f"> Loading system from CSV files from {inputs_dir} ...") 

        for c_name, c_class, c_module, filename in self.components.iter_rows():

            start_time = time.time()
            # Expected file with components, for example: gfli_a.csv, or inf_src.csv
            filepath = os.path.join(inputs_dir, filename)

            # If no such file exits, continue. For example, 
            # if there is no gfli_a.csv, then the sys.gfli_a = []
            if not os.path.exists(filepath):
                continue

            # Import module (.py file). For example import sting.generator.gfli_a
            class_module = importlib.import_module(c_module) 

            # Import class. For example, GFLI_a
            component_class = getattr(class_module, c_class)

            # Get a dictionary that maps fields of class with their corresponding types
            class_param_types = get_type_hints(component_class)
            #parameters_dtypes = {
            #    key: value
            #    for key, value in parameters_dtypes.items()
            #    if value.__module__ == "builtins"
            #}

            # Read only 1 row, do not treat the first row as headers (header=None)
            df = pl.read_csv(filepath, n_rows=1, has_header=False)
            csv_header = df.row(0)

            # Filter out the pairs (key, value) from class_param_types 
            # that are not in csv header
            param_types = {
                key: value
                for key, value in class_param_types.items()
                if key in csv_header
            }

            # Read components csv
            logger.info(f" - '{os.path.basename(filepath)}' ... ")
            df = pl.read_csv(filepath, dtypes=param_types)

            # Create a component for each row (i.e., component) in the csv
            for row in df.iter_rows(named=True):
                component = component_class(**row)
                # Add the component to the system
                self.add(component)

            logger.info(f"   Added {df.height} '{c_name}' components. ")
            logger.info(f"   Completed in {time.time() - start_time:.2f} seconds.")

        start_time = time.time()
        self.apply("post_system_init", self)
        logger.info(f"> Post system initialization completed in {time.time() - start_time:.2f} seconds.")

        logger.info(f"> Completed in: {time.time() - full_start_time:.2f} seconds. \n")

        return self

    def write_csv(self, types = [int, float, str, bool], output_directory=None):
        
        if output_directory is None:
            output_directory = os.path.join(self.case_directory, "outputs", "system_csv")
        
        os.makedirs(output_directory, exist_ok=True)

        for name in self.components["type"]:
          lst = getattr(self, name)
          csv_filename = self.components.filter(pl.col("type") == name).select("input_csv").item(0,0)
          if lst:
              # Assumes each component is a dataclass with fields
              cols = fields(lst[0])
              cols = [c.name for c in cols if c.type in types]
              df = self.query(name).to_table(*cols, index = 'id', index_name = 'id')
              df.to_csv(os.path.join(output_directory, csv_filename))

    def to_matlab(self, session_name=None, export=None, excluded_attributes=None):

        import matlab.engine

        if export is None:
            export = list(self.class_to_str.values())

        current_matlab_sessions = matlab.engine.find_matlab()

        if not session_name in current_matlab_sessions:
            logger.info("> Initiate Matlab session, as a session was not founded or entered.")
            eng = matlab.engine.start_matlab()
        else:
            eng = matlab.engine.connect_matlab(session_name)
            logger.info(f"> Connect to Matlab session: {session_name} ... ok.")
        for typ in export :
            components = getattr(self, typ)

            components_dict = [
                convert_class_instance_to_dictionary(i, excluded_attributes=excluded_attributes) for i in components
            ]
            eng.workspace[typ] = components_dict

        eng.quit()

    # TODO: This should be added to it's own module
    @timeit
    def group_by_zones(self, components_to_clone: list[str] = None):
        """
        Creation of a zonal system where buses are grouped by their zone attribute.

        Method created for a manual zonal reduction of the system, needed for the capacity expansion module.
        Warnings:
         - Only components that have bus, from_bus, to_bus attributes are re-assigned to the new zonal buses.
         - Buses without a zone attribute are ignored.
         - Other attributes are set to None or default values.
        """

        zonal_system = System(case_directory=self.case_directory)

        mapping_bus_to_zone = {n.name: n.zone for n in self.bus if n.zone is not None}
        zones = set(mapping_bus_to_zone.values())

        for zone in zones:
            zonal_system.add( Bus(
                name=zone,
                bus_type="zone_bus",
                zone=zone,
            ))
        logger.info(f" - System with new buses created: {zones}")

        for component in self:
            if (hasattr(component, 'bus')) and (component.bus in mapping_bus_to_zone):
                copied_component = copy.deepcopy(component)
                copied_component.bus = mapping_bus_to_zone[component.bus]
                zonal_system.add(copied_component)

            if ((hasattr(component, 'from_bus') and hasattr(component, 'to_bus')) and 
                (component.from_bus in mapping_bus_to_zone) and (component.to_bus in mapping_bus_to_zone)):
                if mapping_bus_to_zone[component.from_bus] != mapping_bus_to_zone[component.to_bus]:
                    copied_component = copy.deepcopy(component)
                    copied_component.from_bus = mapping_bus_to_zone[component.from_bus]
                    copied_component.to_bus = mapping_bus_to_zone[component.to_bus]
                    zonal_system.add(copied_component)
        
        logger.info(f" - Re-assigning bus, from_bus, to_bus attributes in system components completed.")

        if components_to_clone is not None:
            for attr in components_to_clone:
                setattr(zonal_system, attr, copy.deepcopy(getattr(self, attr)))
        logger.info(f" - Cloning components: {components_to_clone} completed.")

        logger.info(f" - New system has: ")
        for component_name in zonal_system.components["type"]:
            logger.info(f"  - {len(getattr(zonal_system, component_name))} '{component_name}' components. ")

        zonal_system.apply("post_system_init", zonal_system)

        return zonal_system
    
    @timeit
    def upload_built_capacities_from_csv(self, built_capacity_directory: str,  make_non_expandable: bool = True):
        """
        Upload built capacities from a previous capex solution. 
        
        ### Args:
        - built_capacity_directory: `str` 
                    Directory where the CSV files with built capacities are located.
        - make_non_expandable: `bool`, default True
                    If True, the generators, storage units and buses for which built capacities are uploaded will be made non-expandable, 
                    so that their capacities cannot be further expanded in the optimization. 
                    If False, we check the uploaded built capacity against the maximum capacity, and 
                    only make non-expandable those units for which the uploaded built capacity is greater or equal to the maximum capacity. 

        """
        generator.upload_built_capacities_from_csv(self, built_capacity_directory, make_non_expandable)
        storage.upload_built_capacities_from_csv(self, built_capacity_directory, make_non_expandable)
        bus.upload_built_capacities_from_csv(self, built_capacity_directory, make_non_expandable)

    # ------------------------------------------------------------
    # Component Management + Searching
    # ------------------------------------------------------------
    def add(self, component):
        """Add a new component to the system."""
        # Get the component string representation (InfiniteSource -> inf_src)
        component_attr = self.class_to_str[type(component).__name__]
        component_list = getattr(self, component_attr)
        # Assign the component a 0-based index value
        component.id = len(component_list)
        # Add the component to the list
        component_list.append(component)
        
    def _generator(self, names):
        # Collect all lists of components in the component_types
        all_components = [getattr(self, name) for name in names]
         # Yield all components following the order in component_types
        return itertools.chain(*all_components)

    def query(self, *args):
        """
        Return a Stream over a set of component types. Analogous to FROM in 
        SQL, specifying which tables to access data from. For example, 
        "FROM gfmi_a, inf_src SELECT idx" would be written as:
        >>> power_sys.query("gfmi_a", "inf_src").select("idx")
        
        If no tables are provided runs a Stream over all components.
        """
        if not args:
            return sl.Stream(self, index_map=self.class_to_str)
        # Unpack all args calling on self if they are a function
        names = [arg(self) if callable(arg) else [arg] for arg in args]
        # Flatten the list of component types to query from
        names = itertools.chain(*names)

        return sl.Stream(self._generator(names), index_map=self.class_to_str)

    def __iter__(self):
        return self._generator(self.components["type"])
    
    def apply(self, method, *args):
        """Call a given method on all components with the method."""
        for component in self:
            if hasattr(component, method):
                getattr(component, method)(*args)

    @property
    def generators(self):
        return self.query(sl.generators())

    @property
    def shunts(self):
        return self.query(sl.shunts())

    @property
    def branches(self):
        return self.query(sl.branches())
    

    def clean_up(self):
        """
        Apply any component clean up needed prior to methods like power flow.
        """
        decompose_lines(self)
        #TODO: I think combine_shunts(self) is untested
