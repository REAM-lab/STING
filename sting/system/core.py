# Import standard and third-party packages
import pandas as pd
import importlib
import os
from typing import get_type_hints
from dataclasses import make_dataclass, field
#import matlab.engine

# Import src packages
# from sting.utils import data_tools
import sting
from sting import data_files
from sting.line.pi_model import decompose_lines 
#from sting.shunt.paralle_rc import combine_shunts
from sting.utils.graph_matrices import get_ccm_matrices
from sting.models import StateSpaceModel

# Need some kind of list tracking the global ordering of all components? Right,
# now this is implicitly controlled by the grid_components dataframe but we need
# more control... This could be done by making each Zone a "component"? self.to_zones()
# zone1 == [_, _, _, ]
# zone2 == [_, _]
# etc.... basically this is like a reindex option. Base is to index by c_name but 
# maybe we want to build an index on zone, or gen or something else entirely... 

class System: 

    def __init__(self, active_components=None):
        print(sting.__logo__)

        print("> System initialization", end=" ")

        # Read csv mapping each componet to an input file and python class
        data_dir = os.path.dirname(data_files.__file__)
        meta_data = pd.read_csv(os.path.join(data_dir, "components_metadata.csv"))
        
        # Filter meta_data by all components in the active components list
        if active_components:
            meta_data = meta_data[meta_data['components'].isin(active_components)]
            
        # Create a dataclass for all components, initialized to an empty list
        c_names = [
            (c_name, list, field(default_factory=list))
            for c_name in meta_data['component']]
        
        # Create methods for dataclass to return all components of a given type
        def get(s, group, flat):
            """Return a list of all components of a given type"""
            attr_list = [getattr(s, g) for g in group]
            if flat: # Return a single list (rather than a list of lists)
                return sum(attr_list, [])
            return attr_list
        
        # Method to return all components
        c_all = meta_data['component'].to_list()
        namespace = {
            "all": lambda s, flat=True, group=c_all.copy(): get(s, c_all, flat)
        }

        # Methods to return all components of a given type (e.g. shunts, branches, generators)
        for c_type in meta_data['type'].unique():
            # Get a list of all components of a given type
            mask = (meta_data['type'] == c_type)
            group = meta_data.loc[mask, 'component'].to_list()

            namespace[c_type] = (
                lambda s, flat=True, group=group.copy(): get(s, group, flat))
            
        Components = make_dataclass(
            "Components", c_names, slots=True, namespace=namespace)
        
        self.meta_data = meta_data
        self.components = Components()
        self.connections = None
        print("... ok.")
            

    @classmethod
    def from_csv(cls, inputs_directory=None, active_components=None):
        #TODO: Auto detect active components from input files
        if not inputs_directory:
            inputs_directory = os.path.join(os.getcwd(), 'inputs')

        self = cls()
        
        print("> Load components via CSV files from:")

        for(_, c_name, c_type, c_class, c_module, filename)  in self.meta_data.itertuples(name=None):
            
            filepath = os.path.join(inputs_directory, filename)
            # If no such file exits, continue
            if not os.path.exists(filepath):
                continue 

            # Import module, class, and expected data types
            class_module = importlib.import_module(c_module)
            component_class = getattr(class_module, c_class)
            parameters_dtypes = get_type_hints(component_class)
            parameters_dtypes = {
                key: value for key, value in parameters_dtypes.items() 
                if value.__module__ == 'builtins'}

            print(f"\t- '{filepath}'", end=' ')
            df = pd.read_csv(filepath, dtype=parameters_dtypes)
            n_components = len(df)

            # Check if there is at least one row aside from headers
            if n_components == 0: 
                print("... is empty.")  
                continue
                
            # Build index if one does not exits
            if 'idx' not in df.columns:
                df['idx'] = [c_name + str(i) for i in range(1, n_components+1)]

            # Create a component for each row (i.e., component) in the csv
            for row in df.itertuples(index=False): 
                component = component_class(**row._asdict())
                getattr(self.components, c_name).append(component)
                
            print("... ok.")

        
        return self
        
        
    def prep_powerflow(self):
        """
        Apply any component clean up needed prior to running power flow.
        """
        decompose_lines(self)
        # self.combine_shunts()
        
        
    def construct_ssm(self, pf_instance):
        """
        Create each components SSM given a power flow solution
        """
        # Build each components SSM
        self.apply('_load_power_flow_solution', pf_instance)
        self.apply('_calculate_emt_initial_conditions')
        self.apply('_build_small_signal_model')
        
        # Construct the component connection matrices for the system model
        self.connections = get_ccm_matrices(self)
        
        
        
    def apply(self, method, *args):
        """
        Apply a method to all components in the system.
        """
        for c in self.components.all():
            if hasattr(c, method):
                getattr(c, method)(*args) 
            
        
    def create_zone(self, c_names):
        pass
                   

    def permute(self, index):
        # assert len(set(index)) == len(self.components)
        pass


    def interconnect(self):
        return StateSpaceModel.from_interconnected(self.components.all(), self.connections)


    def stack(self):
        return StateSpaceModel.from_stacked(self.components.all())

    
    def to_matlab_session(self, session_name = None):
        pass
    
                

    # def export_components_data_as_matlab_file(self, matlab_session_name = None):

    #     current_matlab_sessions = None #matlab.engine.find_matlab()

    #     if not matlab_session_name in current_matlab_sessions:
    #         print('> Initiate Matlab session, as a session was not founded or entered.')
    #         eng = matlab.engine.start_matlab()
    #     else:
    #         eng = matlab.engine.connect_matlab(matlab_session_name)
    #         print(f'> Connect to Matlab session: {matlab_session_name} ... ok.')
    
    #     components_types = self.component_types
    #     for typ in components_types:
    #         components = getattr(self, typ)

    #         components_dict = [data_tools.convert_class_instance_to_dictionary(i) for i in components]

    #         eng.workspace[typ] = components_dict

    #     eng.quit()