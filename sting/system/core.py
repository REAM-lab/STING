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
from sting.utils.data_structures import ListMap


class System(ListMap):

    def __init__(self, components=None):

        print(__logo__)
        print("> System initialization", end=" ")

        data_dir = os.path.dirname(data_files.__file__)
        filepath = os.path.join(data_dir, "components_metadata.csv")
        meta_data = pd.read_csv(filepath)

        # If components are given, only use the relevant meta-data
        if components:
            meta_data = meta_data[meta_data["type"].isin(components)]

        super().__init__(groups=meta_data["type"])
        # Components are stored under their type and index ("inf_src", 1)
        self._hash = lambda c: (c.type, c.idx)

        # Create views by reading the tags of each active component
        for _, c_name, c_class, c_module, _ in meta_data.itertuples(name=None):
            class_module = importlib.import_module(c_module)
            component_class = getattr(class_module, c_class)
            tags = component_class.__dataclass_fields__["tags"].default_factory()
            for tag in tags:
                if tag in self.views:
                    self.views[tag].append(c_name)
                else:
                    self.add_view(tag, [c_name])

        self.add_view("generators", ["inf_src"])
        self.add_view("shunts", ["pa_rc"])
        self.add_view("branches", ["se_rl"])
        self.add_view("buses", ["bus"])
        print("... ok.")

    def add(self, value):
        group, idx = self._hash(value)
        self[group, idx] = value

    @classmethod
    def from_csv(cls, inputs_dir=None, components=None):

        # If no input directory is given, try using the working directory
        if not inputs_dir:
            inputs_directory = os.path.join(os.getcwd(), "inputs")

        self = cls(components=components)

        # Read components meta data to construct components
        data_dir = os.path.dirname(data_files.__file__)
        filepath = os.path.join(data_dir, "components_metadata.csv")
        meta_data = pd.read_csv(filepath).itertuples(name=None)

        print("> Load components via CSV files from:")

        for _, c_name, c_class, c_module, filename in meta_data:

            filepath = os.path.join(inputs_directory, filename)
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
            df["idx"] = list(range(1, len(df) + 1))

            # Create a component for each row (i.e., component) in the csv
            for row in df.itertuples(index=False):
                component = component_class(**row._asdict())
                # Add the component to the system
                self.add(component)

            print("... ok.")

        return self

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

    def create_zone(self, c_names):
        pass

    def permute(self, index):
        pass

    def interconnect(self):
        # Get components in order of generators, then shunts, then branches
        generators = self.view("generators", "ssm")
        shunts = self.view("shunts", "ssm")
        branches = self.view("branches", "ssm")

        models = list(generators) + list(shunts) + list(branches)

        # Then interconnect models
        return StateSpaceModel.from_interconnected(models, self.connections)

    def stack(self):
        return StateSpaceModel.from_stacked(self.components.all())

    def to_matlab_session(self, matlab_session_name=None):
        import matlab.engine
        
        current_matlab_sessions = matlab.engine.find_matlab()

        if not matlab_session_name in current_matlab_sessions:
            print('> Initiate Matlab session, as a session was not founded or entered.')
            eng = matlab.engine.start_matlab()
        else:
            eng = matlab.engine.connect_matlab(matlab_session_name)
            print(f'> Connect to Matlab session: {matlab_session_name} ... ok.')
    
        components_types = self.component_types
        for typ in components_types:
            components = getattr(self, typ)

            components_dict = [data_tools.convert_class_instance_to_dictionary(i) for i in components]

            eng.workspace[typ] = components_dict

        eng.quit()
