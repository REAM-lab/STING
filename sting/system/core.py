# Import Python packages
import pandas as pd
import importlib
import os
import itertools
from typing import get_type_hints
from dataclasses import fields

# Import source packages
from sting import __logo__
from sting import data_files
from sting.line.core import decompose_lines
from sting.utils import data_tools

# from sting.shunt.core import combine_shunts
from sting.utils.graph_matrices import get_ccm_matrices
from sting.utils.dynamical_systems import StateSpaceModel
import sting.system.selections as sl


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
    def __init__(self, components=None):
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

        print(__logo__) # print logo when a System instance is created
        print("> System initialization", end=" ")

        # Get components_metadata.csv as a dataframe.
        # This file contains information of the lists of components that integrate the system
        data_dir = os.path.dirname(data_files.__file__) # get directory of data_files
        filepath = os.path.join(data_dir, "components_metadata.csv") # get directory 
        self.components = pd.read_csv(filepath) # get list of components as dataframe

        # If components are given, only use the relevant meta-data
        if components:
            active_components = self.components["type"].isin(components)
            self.components = self.components[active_components]

        # Mapping of a components class to its string representation
        self.class_to_str = dict(zip(self.components["class"], self.components["type"]))

        # Create a new attribute (an empty list) for each component type 
        for component_name in self.components["type"]:
            setattr(self, component_name, [])

        print("... ok.")

    @classmethod
    def from_csv(cls, case_dir=os.getcwd(), components=None):
        """
        Add components from csv files. Each csv file has components of the same type.
        For example: gfli_a.csv contains ten gflis, but from the same type gfli_a.
        Each row of gfli_a.csv is a gfli_a that will be added to the system attribute gfli_a. 
        
        ### Inputs:
        - cls (System class)
        - case_dir (str): Directory of the case study. 
                        This directory has a folder "inputs" that has the csv files. 
                        By default it is current directory where we execute sting.
        - components (list): Type of components, for example components=['gfli_a', 'gfmi_c'].
        
        ### Outputs:
        - self (System instance): it contains the components that have data from csv files.
        """

        # Get directory of the folder "inputs"
        inputs_dir = os.path.join(case_dir, "inputs") 

        # Create instance System.
        self = cls(components=components) 

        print("> Load components via CSV files from:")

        for _, c_name, c_class, c_module, filename in self.components.itertuples(
            name=None
        ):
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
            df = pd.read_csv(filepath, nrows=1, header=None)
            csv_header = df.iloc[0].tolist()

            # Filter out the pairs (key, value) from class_param_types 
            # that are not in csv header
            param_types = {
                key: value
                for key, value in class_param_types.items()
                if key in csv_header
            }

            # Read components csv
            print(f"\t- '{filepath}'", end=" ")
            df = pd.read_csv(filepath, dtype=param_types)

            # Create a component for each row (i.e., component) in the csv
            for row in df.itertuples(index=False):
                component = component_class(**row._asdict())
                # Add the component to the system
                self.add(component)

            print("... ok.")

        return self

    def to_csv(self, output_dir=None):
        # TODO: This is untested
        for name in self.components["type"]:
          lst = getattr(self, name)
          if lst:
              # Assumes each component is a dataclass with fields
              cols = fields(lst[0])
              df = self.query(name).to_table(cols)
              df.to_csv(os.path.join(output_dir, self.components["input_csv"]))

    def to_matlab(self, session_name=None, export=None, excluded_attributes=None):
        # TODO: Not sure if this has been tested
        import matlab.engine

        if export is None:
            export = list(self.class_to_str.values())

        current_matlab_sessions = matlab.engine.find_matlab()

        if not session_name in current_matlab_sessions:
            print("> Initiate Matlab session, as a session was not founded or entered.")
            eng = matlab.engine.start_matlab()
        else:
            eng = matlab.engine.connect_matlab(session_name)
            print(f"> Connect to Matlab session: {session_name} ... ok.")

        for typ in export :
            components = getattr(self, typ)

            components_dict = [
                data_tools.convert_class_instance_to_dictionary(i, excluded_attributes=excluded_attributes) for i in components
            ]
            eng.workspace[typ] = components_dict

        eng.quit()

    # ------------------------------------------------------------
    # Component Management + Searching
    # ------------------------------------------------------------
    def add(self, component):
        """Add a new component to the system."""
        # Get the component string representation (InfiniteSource -> inf_src)
        component_attr = self.class_to_str[type(component).__name__]
        component_list = getattr(self, component_attr)
        # Assign the component a 1-based index value
        component.idx = len(component_list) + 1
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

    # ------------------------------------------------------------
    # Small-Signal Modeling
    # ------------------------------------------------------------
    def clean_up(self):
        """
        Apply any component clean up needed prior to methods like power flow.
        """
        decompose_lines(self)
        #TODO: I think combine_shunts(self) is untested

    def construct_ssm(self, pf_instance):
        """
        Create each components SSM given a power flow solution
        """
        # Build each components SSM
        self.apply("_load_power_flow_solution", pf_instance)
        self.apply("_calculate_emt_initial_conditions")
        self.apply("_build_small_signal_model")

        # Construct the component connection matrices for the system model
        self.connections = get_ccm_matrices(self)

    def interconnect(self):
        """
        Return a state-space model of all interconnected components
        """
        # Get components in order of generators, then shunts, then branches
        generators, = self.generators.select("ssm")
        shunts, = self.shunts.select("ssm")
        branches, = self.branches.select("ssm")

        models = itertools.chain(generators, shunts, branches)

        # Then interconnect models
        return StateSpaceModel.from_interconnected(models, self.connections)
    
    # ------------------------------------------------------------
    # EMT simulation
    # ------------------------------------------------------------

    def sim_emt(self, t_stop: float, dt: float, solver: str = "RK45"):
        """
        Simulate the EMT dynamics of the system using scipy.integrate.solve_ivp
        """
        
        # define states
        
        def system_ode(t, x, u):

            # x

            ystack = []

            for component in self:
                if hasattr(component, "_EMT_output_equations"):
                    y = getattr(component, "_EMT_output_equations")(t, x, ud, ug)
                    ystack.append(y)

            # y_stack
            # u

            # ustack = F y_stack + G u_stack

            for component in self:
                if hasattr(component, "_EMT_state_dynamics"):            
                    dx = getattr(component, "_EMT_state_dynamics")(t, x, ud, ug)
                    

                    # collect by type

                    # 
            return dx
        
        

    # ------------------------------------------------------------
    # Model Reduction (TBD?)
    # ------------------------------------------------------------
    def create_zone(self, c_names):
        pass

    def permute(self, index):
        pass
