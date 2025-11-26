import pandas as pd
import importlib
import os
from typing import get_type_hints
from sting import __logo__
from sting import data_files
from sting.line.core import decompose_lines
from sting.utils import data_tools

# from sting.shunt.core import combine_shunts
from sting.utils.graph_matrices import get_ccm_matrices
from sting.utils.dynamical_systems import StateSpaceModel
import sting.system.selections as sl


class System:
    
    # ------------------------------------------------------------
    # Construction + Read/Write
    # ------------------------------------------------------------
    def __init__(self, components=None):

        print(__logo__)
        print("> System initialization", end=" ")

        data_dir = os.path.dirname(data_files.__file__)
        filepath = os.path.join(data_dir, "components_metadata.csv")
        meta_data = pd.read_csv(filepath)

        # If components are given, only use the relevant meta-data
        if components:
            self.meta_data = meta_data[meta_data["type"].isin(components)]

        # TODO: Set up attr lists

        print("... ok.")

    @classmethod
    def from_csv(cls, inputs_dir=None, components=None):

        # If no input directory is given, try using the working directory
        if not inputs_dir:
            inputs_dir = os.getcwd()

        inputs_dir = os.path.join(inputs_dir, "inputs")
        self = cls(components=components)

        print("> Load components via CSV files from:")

        for _, c_name, c_class, c_module, filename in self.meta_data.itertuples(
            name=None
        ):

            filepath = os.path.join(inputs_dir, filename)
            # If no such file exits, continue
            if not os.path.exists(filepath):
                continue

            # Import module, class, and expected data types
            class_module = importlib.import_module(c_module)
            component_class = getattr(class_module, c_class)
            parameters_dtypes = get_type_hints(component_class)
            parameters_dtypes = {
                key: value
                for key, value in parameters_dtypes.items()
                if value.__module__ == "builtins"
            }

            # Read components csv
            print(f"\t- '{filepath}'", end=" ")
            df = pd.read_csv(filepath, dtype=parameters_dtypes)

            # Create a component for each row (i.e., component) in the csv
            for row in df.itertuples(index=False):
                component = component_class(**row._asdict())
                # Add the component to the system
                self.add(component)

            print("... ok.")

        return self

    def to_csv(self, file_dir=None):
        pass

    def to_matlab(self, session_name=None):
        # TODO: Not sure if this has been tested
        import matlab.engine

        current_matlab_sessions = matlab.engine.find_matlab()

        if not session_name in current_matlab_sessions:
            print("> Initiate Matlab session, as a session was not founded or entered.")
            eng = matlab.engine.start_matlab()
        else:
            eng = matlab.engine.connect_matlab(session_name)
            print(f"> Connect to Matlab session: {session_name} ... ok.")

        components_types = self.component_types
        for typ in components_types:
            components = getattr(self, typ)

            components_dict = [
                data_tools.convert_class_instance_to_dictionary(i) for i in components
            ]

            eng.workspace[typ] = components_dict

        eng.quit()

    # ------------------------------------------------------------
    # Component Management + Searching
    # ------------------------------------------------------------
    def add(self, component):
        """Register a new component in the system."""
        # TODO: do this correctly
        c_type = type(component)

        c_list = getattr(self, c_type)
        component.idx = len(c_list) + 1
        c_list.append(component)
        
    def _generator(self, component_types):
        # Yield all components following the order in component_types
        for component_type in component_types:
            for component in getattr(self, component_type):
                yield component

    def query(self, *args):
        """
        Return a Stream over a set of component types. Analogous to FROM in 
        SQL, specifying which tables to access data from. For example, 
        "FROM gfmi_a, inf_src SELECT idx" would be written as:
        >>> system.query("gfmi_a", "inf_src").select("idx")
        
        If no tables are provided runs a Stream over all components.
        """
        if not args:
            return sl.Stream(self)
        
        return sl.Stream(self._generator(args))

    def __iter__(self):
        return self._generator(self.meta_data["type"])
    
    def apply(self, method, *args):
        """Call a given method on all components with the method."""
        def helper(component):
            if hasattr(component, method):
                getattr(component, method)(*args)
                
        return self.query().map(helper)

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
        # combine_shunts(self)

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
        # Get components in order of generators, then shunts, then branches
        generators = self.generators.select("ssm").to_list()
        shunts = self.shunts.select("ssm").to_list()
        branches = self.branches.select("ssm").to_list()

        models = generators + shunts + branches

        # Then interconnect models
        return StateSpaceModel.from_interconnected(models, self.connections)

    def stack(self):
        return StateSpaceModel.from_stacked(self.components.all())

    # ------------------------------------------------------------
    # Model Reduction (TBD?)
    # ------------------------------------------------------------
    def create_zone(self, c_names):
        pass

    def permute(self, index):
        pass
