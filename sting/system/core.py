import pandas as pd
import importlib
import os
from typing import get_type_hints
from sting import data_files

class System: 

    def __init__(self):

        color1 = "\033[93m"
        color2 = "\033[0m"

        logo1 = (
        r"   ______  " + color1 + r" __" + color2 + "\n"       
        r"  / __/ /_" + color1 + r"_/ /" + color2 + r"_  _____ _" + "\n"
        r" _\ \/ __" + color1 + r"/ _/" + color2 + r"/ \/ / _ `/" + "\n"
        r"/___/__/" + color1 + r"//  " + color2 + r"/_/\_/\_, /"  + "\n"
        r"       " + color1 + r"/   " + color2 + r"      /___/  ")

        #print("=" * 25)
        print(logo1)
        #print("\u26A1 STING 0.1.0 \u26A1")
        print("\nVersion: 0.1.0 (2025-11-05)")
        print("=" * 40)
        print('\n')
        print("> System initialization: \n")

        module_file_path = data_files.__file__
        filepath = os.path.join(os.path.dirname(module_file_path), 'grid_components.csv')
        self.grid_components_df = pd.read_csv(filepath) 
        
        df = self.grid_components_df.copy()
        self.generator_types_list =  df.loc[df['type'] == 'generator', 'component'].to_list()
        self.line_types_list =  df.loc[df['type'] == 'line', 'component'].to_list()
        self.branch_types_list =  df.loc[df['type'] == 'branch', 'component'].to_list()
        self.shunt_types_list =  df.loc[df['type'] == 'shunt', 'component'].to_list()
        self.component_types = dict(zip(self.grid_components_df['component'], self.grid_components_df['class']))

        print("    Empty lists created for the following components:")
        for _, row1 in self.grid_components_df.iterrows():

            component_name = row1['component'] # Name of component 
            component_class = row1['class']
            setattr(self, component_name, [])  # Add an empty list of components

            print(f"  - {component_name} : {component_class} ")
        print(' ')

    def load_components_via_input_csv_files(self):

        print("> Load components via CSV files from: \n")

        inputfolder_directory = os.path.join(os.getcwd(), 'inputs') 
        for(_, component, type, clas, module, input_csv)  in self.grid_components_df.itertuples(name=None):

            class_module = importlib.import_module(module)  # Import module
            component_class = getattr(class_module, clas) # Import class
            component_name = component # Name of component 
            parameters_dtypes = get_type_hints(component_class)
            parameters_dtypes = {key: value for key, value in parameters_dtypes.items() if value.__module__ == 'builtins'}

            
            components_list =  getattr(self, component_name) # Get list of components (empty or already created)
            
            filename = input_csv
            filepath = os.path.join(inputfolder_directory, filename) 

            if os.path.exists(filepath):
                print(f"  - '{filepath}'", end=' ')
                df = pd.read_csv(filepath, dtype=parameters_dtypes) # Import dataframe as csv
                n_components = len(df) # number of components 

                if n_components>0: # check if there is at least one row aside from headers
                    
                    if 'idx' not in df.columns:
                        df['idx'] = [component_name + str(i) for i in range(1, n_components+1)] # create index

                    for row2 in df.itertuples(index=False): # iterate over the row of each csv
                        component_data = row2._asdict() # transform row into dict
                        components_list.append(component_class(**component_data)) # create component and add to a list
                    print("... ok.")

                else:
                    print("... it is empty.")   
        print(' ')

    def dissect_lines_into_branches_and_shunts(self):

        from sting.branch.series_rl import Series_rl_branch
        from sting.shunt.parallel_rc import Parallel_rc_shunt

        print("> Add branches and shunts from dissecting lines: \n")

        print("  - Lines with no series compensation", end=' ')
        for line in self.line_ns:
            self.se_rl.append(Series_rl_branch( idx = 'from_' + line.idx, 
                                                type = 'branch', 
                                                from_bus = line.from_bus, 
                                                to_bus = line.to_bus,
                                                sbase = line.sbase,
                                                vbase = line.vbase,
                                                fbase = line.fbase,
                                                r = line.r,
                                                l = line.l ))
                
            self.pa_rc.append(Parallel_rc_shunt( idx = line.idx + '_frombus',
                                                 type = 'shunt', 
                                                 bus_idx = line.from_bus,
                                                 sbase = line.sbase,
                                                 vbase = line.vbase,
                                                 fbase = line.fbase,
                                                 r = 1/line.g,
                                                 c = 1/line.b,
                                                  ))
            
            self.pa_rc.append(Parallel_rc_shunt( idx = line.idx + '_tobus',
                                                 type = 'shunt', 
                                                 bus_idx = line.to_bus,
                                                 sbase = line.sbase,
                                                 vbase = line.vbase,
                                                 fbase = line.fbase,
                                                 r = 1/line.g,
                                                 c = 1/line.b,
                                                  ))
        print("... ok")
        print(" ")
            
        # TODO: Do the same for line with series compensation

    def reduce_shunts_to_one_per_bus(self):

        print("> Reduce shunts to have one shunt per bus: \n")

        from sting.shunt.parallel_rc import Parallel_rc_shunt

        bus_idx, g, b = [], [], [], []

        for shuntype in ['pa_rc']: # iterate over each shunt type
            shs = getattr(self, shuntype)
            bus_idx.extend([s.bus_idx for s in shs])
            g.extend([s.g for s in shs])
            b.extend([s.b for s in shs])

        shunt_df = pd.DataFrame({'bus_idx': bus_idx, 'g': g, 'b': b})    
        shunt_df = shunt_df.pivot_table(index='bus_idx', values=['g', 'b'], aggfunc ='sum')
        shunt_df['r'] = 1/shunt_df['g']
        shunt_df['c'] = 1/shunt_df['b']
        shunt_df['idx'] = ['shred' + str(i) for i in range(len(shunt_df))] 
        shunt_df.reset_index(inplace=True)

        pa_rc = [] # create new list of components "parallel rc shunts"

        for _, row in shunt_df.iterrows(): # iterate over the row of shunt_df
                component_data = row.to_dict() # transform row into dict
                pa_rc.append(Parallel_rc_shunt(**component_data)) # create component and add to a lis
 
        self.pa_rc = pa_rc # assign new list of components. The previous list will be deleted.
        print("  - New list of parallel rc components created ... ok \n")

    def transfer_power_flow_solution_to_components(self, power_flow_instance):

        for type in self.component_types:
            list_of_components = getattr(self, type)
            c0 = list_of_components[0] if list_of_components else None
            if hasattr(c0, '_load_power_flow_solution'):
                for c in list_of_components:
                    c._load_power_flow_solution(power_flow_instance)

    def calculate_emt_initial_condition_of_components(self):

        for type in self.component_types:
            list_of_components = getattr(self, type)
            c0 = list_of_components[0] if list_of_components else None
            if hasattr(c0, '_calculate_emt_initial_conditions'):
                for c in list_of_components:
                    c._calculate_emt_initial_conditions()

    def build_small_signal_model_of_components(self):

        for type in self.component_types:
            list_of_components = getattr(self, type)
            c0 = list_of_components[0] if list_of_components else None
            if hasattr(c0, '_build_small_signal_model'):
                for c in list_of_components:
                    c._build_small_signal_model()


