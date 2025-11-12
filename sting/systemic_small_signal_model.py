# Import standard python packages and third-party packages
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import os
import matlab.engine
from scipy.linalg import block_diag

# Import sting packages
from sting.system.core import System
from sting.utils import graph_matrices, linear_systems_tools, data_tools
from sting.utils.linear_systems_tools import State_space_model

@dataclass
class Composite_small_signal_model:

    input_system: System
    ''' Power grid, including set of generators, branches, shunts and respective small signal models of each component'''

    system: State_space_model= field(init=False)
    generators: State_space_model = field(init=False)
    shunts: State_space_model = field(init=False)
    branches: State_space_model= field(init=False)

    def __post_init__(self):
        self.build_generators_composite_model()
        self.build_shunts_composite_model()
        self.build_branches_composite_model()
        self.build_system_composite_model()

    def build_generators_composite_model(self):
        print("> Build composite model of generators", end=' ')
        
        # Get a list of generators types. For example, list_of_gentypes = [inf_src, gfli_a, gfli_b]
        list_of_gentypes = self.input_system.generator_types_list

        # Create empty lists to put away the models and transformations
        # The orders of the items in this list respect the order of list_of_gentypes
        ssm_of_gentypes = []
        T2_of_gentypes = []
        T3_of_gentypes = []

        for typ in list_of_gentypes: # iterate over the list of types
        
            # Get the list of generators, for example, list_of_gens = [inf_src[0], inf_src[1]]
            # Each item in this list is a class instance
            list_of_gens = getattr(self.input_system, typ) 

            if list_of_gens: # check if the list if empty.
            
                # Get state-space matrices and put them in a list, then
                # construct block diagonal matrices
                Astack = block_diag(*[g.ssm.A for g in list_of_gens])
                Bstack = block_diag(*[g.ssm.B for g in list_of_gens])
                Cstack = block_diag(*[g.ssm.C for g in list_of_gens])
                Dstack = block_diag(*[g.ssm.D for g in list_of_gens])

                n = len(list_of_gens) # number of generators
                
                # Take the first generator in the list.
                g0 = list_of_gens[0] 

                # nd: number of device side inputs for a generator
                # ng: number of grid side inputs for a generator
                # Note that as all of the generators in this list_of_gens are of the same class, 
                # then they must have the same nd and ng
                nd, ng = len(g0.ssm.device_side_inputs), len(g0.ssm.grid_side_inputs)

                # Build transformation (permutation) matrices
                matrix1 = np.kron(np.eye(n), 
                                  np.hstack( ( np.eye(nd), np.zeros((nd, ng)) ) ) )
                matrix2 = np.kron(np.eye(n), 
                                  np.hstack( ( np.zeros((ng, nd)), np.eye(ng) ) ) )

                T1 = np.vstack((matrix1, matrix2))
                
                # Construct new B and D matrices.
                # B and D are re-ordered to have their columns with order: first device_side_inputs and then grid_side_inputs
                Bnew = np.matmul(Bstack, np.linalg.inv(T1))
                Dnew = np.matmul(Dstack, np.linalg.inv(T1))
                
                # Collect device_side_inputs, grid_side_inputs, etc from all the generators in list_of_gens
                # Note that we use tuples that allow to identify to which generator the variable belongs to.
                dev_inputs = [(g.idx, ud) for g in list_of_gens for ud in g.ssm.device_side_inputs ]
                grid_inputs = [(g.idx, ug) for g in list_of_gens for ug in g.ssm.grid_side_inputs ]
                states = [(g.idx, x) for g in list_of_gens for x in g.ssm.states ]
                outputs = [(g.idx, y) for g in list_of_gens for y in g.ssm.outputs ]

                # Vertically stack states, etc.
                initial_states = np.vstack([g.ssm.initial_states for g in list_of_gens])
                initial_device_side_inputs = np.vstack([g.ssm.initial_device_side_inputs for g in list_of_gens])
                initial_grid_side_inputs = np.vstack([g.ssm.initial_grid_side_inputs for g in list_of_gens])
                initial_outputs = np.vstack([g.ssm.initial_outputs for g in list_of_gens])
                
                # Append the State_space_model object to the list 
                ssm_of_gentypes.append(State_space_model(   A = Astack,
                                                            B = Bnew,
                                                            C = Cstack,
                                                            D = Dnew,
                                                            device_side_inputs = dev_inputs,
                                                            grid_side_inputs = grid_inputs,
                                                            states= states,
                                                            outputs= outputs,
                                                            initial_states = initial_states,
                                                            initial_device_side_inputs=initial_device_side_inputs,
                                                            initial_grid_side_inputs=initial_grid_side_inputs,
                                                            initial_outputs=initial_outputs
                                                          ))
                
                # Also, append transformations that are used later
                T2_of_gentypes.append(np.hstack(( np.eye(n*nd), np.zeros((n*nd, n*ng)) )) )
                T3_of_gentypes.append(np.hstack(( np.zeros((n*ng, n*nd)), np.eye(n*ng) )) )

        # Construct block diagonal matrices of the ssms of the gentypes
        Astack = block_diag(*[ssm.A for ssm in ssm_of_gentypes])
        Bstack = block_diag(*[ssm.B for ssm in ssm_of_gentypes])
        Cstack = block_diag(*[ssm.C for ssm in ssm_of_gentypes])
        Dstack = block_diag(*[ssm.D for ssm in ssm_of_gentypes])

        # Build transformations
        T4 = block_diag(*T2_of_gentypes)
        T5 = block_diag(*T3_of_gentypes)
        T6 = np.vstack((T4, T5))
        
        # Bnew and Dnew have first device_side_inputs and then grid_side_inputs
        Bnew = Bstack @ np.linalg.inv(T6)
        Dnew = Dstack @ np.linalg.inv(T6)

        # Collect ...
        dev_inputs = [ud for ssm in ssm_of_gentypes for ud in ssm.device_side_inputs ]
        grid_inputs = [ug for ssm in ssm_of_gentypes for ug in ssm.grid_side_inputs ]
        states = [x for ssm in ssm_of_gentypes for x in ssm.states ]
        outputs = [y for ssm in ssm_of_gentypes for y in ssm.outputs ]

        initial_states = np.vstack([ssm.initial_states for ssm in ssm_of_gentypes])
        initial_device_side_inputs = np.vstack([ssm.initial_device_side_inputs for ssm in ssm_of_gentypes])
        initial_grid_side_inputs = np.vstack([ssm.initial_grid_side_inputs for ssm in ssm_of_gentypes])
        initial_outputs = np.vstack([ssm.initial_outputs for ssm in ssm_of_gentypes])

        self.generators = State_space_model(    A = Astack,
                                                B = Bnew,
                                                C = Cstack,
                                                D = Dnew,
                                                device_side_inputs = dev_inputs,
                                                grid_side_inputs = grid_inputs,
                                                states= states,
                                                outputs= outputs,
                                                initial_states=initial_states,
                                                initial_device_side_inputs=initial_device_side_inputs,
                                                initial_grid_side_inputs=initial_grid_side_inputs,
                                                initial_outputs=initial_outputs
                                              )
        
        print("... ok. \n")

        outputfolder_directory = os.path.join(os.getcwd(), 'outputs') 

        print("  > Creation of files with the small signal models: ")
        
        filename = 'generators_composite_ssm_A.csv'
        A_df = pd.DataFrame(Astack, index = states , columns=states)
        filepath = os.path.join(outputfolder_directory,  filename)
        A_df.to_csv(filepath)
        print(f'  - {filename} ')

        filename = 'generators_composite_ssm_B.csv'
        B_df = pd.DataFrame(Bnew, index = states , columns=dev_inputs + grid_inputs)
        filepath = os.path.join(outputfolder_directory, filename)
        B_df.to_csv(filepath)
        print(f'  - {filename} ')

        filename = 'generators_composite_ssm_C.csv'
        C_df = pd.DataFrame(Cstack, index = outputs , columns=states)
        filepath = os.path.join(outputfolder_directory, filename)
        C_df.to_csv(filepath)
        print(f'  - {filename} ')

        filename = 'generators_composite_ssm_D.csv'
        D_df = pd.DataFrame(Dnew, index = outputs , columns=dev_inputs + grid_inputs)
        filepath = os.path.join(outputfolder_directory, filename)
        D_df.to_csv(filepath)
        print(f'  - {filename} \n')


    def build_shunts_composite_model(self):

        print("> Build composite model of shunts", end=' ')
        
        # Get the list of shunts, for example, list_of_shunts = [pr_rc[0], pr_rc[1]]
        list_of_shunttypes = self.input_system.shunt_types_list

        # Create list that saves ssms of the shunts.
        ssm_of_shunttypes = []

        for typ in list_of_shunttypes: # iterate over the shunt types
            
            # Get list of shunts, for example, list of shunts 
            list_of_shunts = getattr(self.input_system, typ)

            if list_of_shunts: # check if the list is empty
                
                # Build block diagonal matrices by stacking the matrices of the elements.
                Astack = block_diag(*[s.ssm.A for s in list_of_shunts])
                Bstack = block_diag(*[s.ssm.B for s in list_of_shunts])
                Cstack = block_diag(*[s.ssm.C for s in list_of_shunts])
                Dstack = block_diag(*[s.ssm.D for s in list_of_shunts])

                # Collect ... (very similar to generators)
                grid_inputs = [(s.idx, ug) for s in list_of_shunts for ug in s.ssm.grid_side_inputs ]
                states = [(s.idx, x) for s in list_of_shunts for x in s.ssm.states ]
                outputs = [(s.idx, y) for s in list_of_shunts for y in s.ssm.outputs ]
                
                initial_states =  np.vstack([s.ssm.initial_states for s in list_of_shunts])
                initial_grid_inputs = np.vstack([s.ssm.initial_grid_side_inputs for s in list_of_shunts])
                initial_outputs = np.vstack([s.ssm.initial_outputs for s in list_of_shunts])

                ssm_of_shunttypes.append(State_space_model( A = Astack,
                                                            B = Bstack,
                                                            C = Cstack,
                                                            D = Dstack,
                                                            grid_side_inputs = grid_inputs,
                                                            states= states,
                                                            outputs= outputs,
                                                            initial_grid_side_inputs=initial_grid_inputs,
                                                            initial_states = initial_states,
                                                            initial_outputs = initial_outputs
                                                          ))

        Astack = block_diag(*[ssm.A for ssm in ssm_of_shunttypes])
        Bstack = block_diag(*[ssm.B for ssm in ssm_of_shunttypes])
        Cstack = block_diag(*[ssm.C for ssm in ssm_of_shunttypes])
        Dstack = block_diag(*[ssm.D for ssm in ssm_of_shunttypes])

        grid_inputs = [ug for ssm in ssm_of_shunttypes for ug in ssm.grid_side_inputs ]
        states = [x for ssm in ssm_of_shunttypes for x in ssm.states ]
        outputs = [y for ssm in ssm_of_shunttypes for y in ssm.outputs ]
        
        initial_states = np.vstack([ssm.initial_states for ssm in ssm_of_shunttypes])
        initial_grid_inputs = np.vstack([ssm.initial_grid_side_inputs for ssm in ssm_of_shunttypes])
        initial_outputs = np.vstack([ssm.initial_outputs for ssm in ssm_of_shunttypes])

        self.shunts = State_space_model(    A = Astack,
                                            B = Bstack,
                                            C = Cstack,
                                            D = Dstack,
                                            grid_side_inputs = grid_inputs,
                                            states= states,
                                            outputs= outputs,
                                            initial_states=initial_states,
                                            initial_grid_side_inputs=initial_grid_inputs,
                                            initial_outputs=initial_outputs
                                              )
        
        print("... ok. \n")
        
        outputfolder_directory = os.path.join(os.getcwd(), 'outputs') 
        
        print(f'  > Creation of files with the small signal model in {outputfolder_directory}: ')
        
        filename = 'shunts_composite_ssm_A.csv'
        A_df = pd.DataFrame(Astack, index = states , columns=states)
        filepath = os.path.join(outputfolder_directory, filename)
        A_df.to_csv(filepath)
        print(f'  - {filename} ')

        filename = 'shunts_composite_ssm_B.csv'
        B_df = pd.DataFrame(Bstack, index = states , columns=grid_inputs)
        filepath = os.path.join(outputfolder_directory, filename)
        B_df.to_csv(filepath)
        print(f'  - {filename} ')

        filename = 'shunts_composite_ssm_C.csv'
        C_df = pd.DataFrame(Cstack, index = outputs , columns=states)
        filepath = os.path.join(outputfolder_directory, filename)
        C_df.to_csv(filepath)
        print(f'  - {filename} ')

        filename = 'shunts_composite_ssm_D.csv'
        D_df = pd.DataFrame(Dstack, index = outputs , columns=grid_inputs)
        filepath = os.path.join(outputfolder_directory, filename)
        D_df.to_csv(filepath)
        print(f'  - {filename} \n')


    def build_branches_composite_model(self):

        print("> Build composite model of branches", end=' ')
        
        # The code here is very similar to the function that builds the shunt composite model.
        # Then, no comments in this method.
        list_of_branchtypes = self.input_system.branch_types_list

        ssm_of_branchtypes = []

        for typ in list_of_branchtypes:
            list_of_branches = getattr(self.input_system, typ)

            if list_of_branches:
                Astack = block_diag(*[s.ssm.A for s in list_of_branches])
                Bstack = block_diag(*[s.ssm.B for s in list_of_branches])
                Cstack = block_diag(*[s.ssm.C for s in list_of_branches])
                Dstack = block_diag(*[s.ssm.D for s in list_of_branches])

                grid_inputs = [(b.idx, ug) for b in list_of_branches for ug in b.ssm.grid_side_inputs ]
                states = [(b.idx, x) for b in list_of_branches for x in b.ssm.states ]
                outputs = [(b.idx, y) for b in list_of_branches for y in b.ssm.outputs ]
                
                initial_states =  np.vstack([b.ssm.initial_states for b in list_of_branches])
                initial_grid_inputs = np.vstack([b.ssm.initial_grid_side_inputs for b in list_of_branches])
                initial_outputs = np.vstack([b.ssm.initial_outputs for b in list_of_branches])

                ssm_of_branchtypes.append(State_space_model(    A = Astack,
                                                                B = Bstack,
                                                                C = Cstack,
                                                                D = Dstack,
                                                                grid_side_inputs = grid_inputs,
                                                                states= states,
                                                                outputs= outputs,
                                                                initial_grid_side_inputs=initial_grid_inputs,
                                                                initial_states=initial_states,
                                                                initial_outputs=initial_outputs
                                                          ))

        Astack = block_diag(*[ssm.A for ssm in ssm_of_branchtypes])
        Bstack = block_diag(*[ssm.B for ssm in ssm_of_branchtypes])
        Cstack = block_diag(*[ssm.C for ssm in ssm_of_branchtypes])
        Dstack = block_diag(*[ssm.D for ssm in ssm_of_branchtypes])

        grid_inputs = [ug for ssm in ssm_of_branchtypes for ug in ssm.grid_side_inputs ]
        states = [x for ssm in ssm_of_branchtypes for x in ssm.states ]
        outputs = [y for ssm in ssm_of_branchtypes for y in ssm.outputs ]
        
        initial_states = np.vstack([ssm.initial_states for ssm in ssm_of_branchtypes])
        initial_grid_inputs = np.vstack([ssm.initial_grid_side_inputs for ssm in ssm_of_branchtypes])
        initial_outputs = np.vstack([ssm.initial_outputs for ssm in ssm_of_branchtypes])

        
        self.branches = State_space_model(  A = Astack,
                                            B = Bstack,
                                            C = Cstack,
                                            D = Dstack,
                                            grid_side_inputs = grid_inputs,
                                            states= states,
                                            outputs= outputs,
                                            initial_grid_side_inputs=initial_grid_inputs,
                                            initial_states=initial_states,
                                            initial_outputs=initial_outputs
                                              )
        print("... ok. \n")

        outputfolder_directory = os.path.join(os.getcwd(), 'outputs') 
        
        print(f' > Creation of files with the small signal model in {outputfolder_directory}: ')
        
        filename = 'branches_composite_ssm_A.csv'
        A_df = pd.DataFrame(Astack, index = states , columns=states)
        filepath = os.path.join(outputfolder_directory, filename)
        A_df.to_csv(filepath)
        print(f'  - {filename} ')

        filename = 'branches_composite_ssm_B.csv'
        B_df = pd.DataFrame(Bstack, index = states , columns=grid_inputs)
        filepath = os.path.join(outputfolder_directory, filename)
        B_df.to_csv(filepath)
        print(f'  - {filename} ')

        filename = 'branches_composite_ssm_C.csv'
        C_df = pd.DataFrame(Cstack, index = outputs , columns=states)
        filepath = os.path.join(outputfolder_directory, filename)
        C_df.to_csv(filepath)
        print(f'  - {filename} ')

        filename = 'branches_composite_ssm_D.csv'
        D_df = pd.DataFrame(Dstack, index = outputs , columns=grid_inputs)
        filepath = os.path.join(outputfolder_directory, filename)
        D_df.to_csv(filepath)
        print(f'  - {filename} \n')


    def build_system_composite_model(self):
        print("> Build composite model of the full system", end=' ')

        # Get the number of buses of the full system
        num_buses = len(self.input_system.bus)

        # List that will contain the connecting bus of all the generators 
        # Note that the first item is the connecting bus of the first generator, 
        # the second item is the connecting bus of the second generator.
        # 'list_of_gens' is for example [inf_src[0], inf_src[1]]
        # Note that the order is respected. The list 'generator_types_list' has been used in previous methods
        gen_bus_connections = []
        list_of_gentypes = self.input_system.generator_types_list
        for typ in list_of_gentypes:
            list_of_gens = getattr(self.input_system, typ)
            gen_bus_connections.extend([g.bus_idx for g in list_of_gens])

        # Build generation connection matrix
        gen_cx = graph_matrices.build_generation_connection_matrix(num_buses, gen_bus_connections)
        
        
        # List that will contain the tuples (from_bus, to_bus) of the branches
        # 'list_of_branches' is for example [se_rl[0], se_rl[1]]
        # Note that the order is respected. The list 'branch_types_list' has been used in previous methods
        branch_frombus_tobus = []
        list_of_branchtypes = self.input_system.branch_types_list
        for typ in list_of_branchtypes:
            list_of_branches = getattr(self.input_system, typ)
            branch_frombus_tobus.extend([(b.from_bus, b.to_bus) for b in list_of_branches])

        # Build oriented incidence matrix
        or_inc = graph_matrices.build_oriented_incidence_matrix(num_buses, branch_frombus_tobus)

        # Build unoriented incidence matrix
        un_inc = abs(or_inc)

        d_gen = len(self.generators.device_side_inputs) # total number of device_side_inputs of generators
        g_gen = len(self.generators.grid_side_inputs) # total number of grid_side_inputs of generators
        y_gen = len(self.generators.outputs) # total number of outputs of the generators

        g_br = len(self.branches.grid_side_inputs) # total number of grid_side_inputs of branches
        y_br = len(self.branches.outputs) # total number of outputs of branches

        g_sh = len(self.shunts.grid_side_inputs) # total number of grid_side_inputs of generators
        y_sh = len(self.shunts.outputs) # total number of outputs of branches

        y = y_gen + y_sh + y_br # total number of outputs of the system

        # Construct matrix F. We first build its blocks.
        F11 = np.zeros( (d_gen, y_gen) )
        F12 = np.zeros( (d_gen, y_sh) )
        F13 = np.zeros( (d_gen, y_br) )

        F21 = np.zeros( (g_gen, y_gen) )
        F22 = np.kron( gen_cx, np.eye(2) )
        F23 = np.zeros( (g_gen, y_br) )

        F31 = np.kron( np.transpose(gen_cx), np.eye(2))
        F32 = np.zeros( (g_sh, y_sh) )
        F33 = np.kron( or_inc, np.eye(2) )

        F41 = np.zeros( (g_br, y_gen) )
        F42 = np.kron( 0.5*(np.kron( np.transpose(un_inc) , np.array([[1], [1]]) )
                            + np.kron( np.transpose(or_inc) , np.array([[-1], [1]]) ) ) 
                             , np.eye(2) )
        F43 = np.zeros( (g_br, y_br) )

        F = np.block( [[F11, F12, F13],
                       [F21, F22, F23],
                       [F31, F32, F33],
                       [F41, F42, F43] ])
        
        # Construct matrix G. We first build its blocks.
        G11 = np.eye(d_gen)
        G21 = np.zeros((g_gen, d_gen))
        G31 = np.zeros((g_sh, d_gen))
        G41 = np.zeros((g_br, d_gen))

        G = np.block( [[G11], 
                      [G21],
                      [G31],
                      [G41]])
        
        # Construct matrix H and L
        H = np.eye(y)
        L = np.zeros((y, d_gen))

        # Use CCM method to connect the composite systems of the generators, shunts and branches. 
        # The order matters: firt, gens, second shunts, and then branches.
        # The order matters because the interconnection matrices have been built considering that order.

        ssm = linear_systems_tools.connect_models_via_CCM(F, G, H, L,
                                                            [self.generators, 
                                                             self.shunts,
                                                             self.branches,
                                                            ])
        
        # The order is important when defining the states, etc. First, generators, then, shunts and finally branches
        ssm.device_side_inputs = self.generators.device_side_inputs
        ssm.inputs = ssm.device_side_inputs
        ssm.states  = self.generators.states  + self.shunts.states  + self.branches.states
        ssm.outputs = self.generators.outputs + self.shunts.outputs + self.branches.outputs
        
        ssm.initial_device_side_inputs = self.generators.initial_device_side_inputs
        ssm.initial_inputs = ssm.initial_device_side_inputs
        ssm.initial_states =    np.vstack((self.generators.initial_states, 
                                           self.shunts.initial_states, 
                                           self.branches.initial_states))
        ssm.initial_outputs =   np.vstack((self.generators.initial_outputs, 
                                          self.shunts.initial_outputs, 
                                          self.branches.initial_outputs))
        
        self.system = ssm

        print("... ok. \n")

        outputfolder_directory = os.path.join(os.getcwd(), 'outputs') 
        
        print(f' > Creation of files with the small signal model in {outputfolder_directory}: ')
        
        filename = 'system_composite_ssm_A.csv'
        A_df = pd.DataFrame(ssm.A, index = ssm.states , columns=ssm.states)
        filepath = os.path.join(outputfolder_directory, filename)
        A_df.to_csv(filepath)
        print(f'  - {filename} ')

        filename = 'system_composite_ssm_B.csv'
        B_df = pd.DataFrame(ssm.B, index = ssm.states , columns=ssm.inputs)
        filepath = os.path.join(outputfolder_directory, filename )
        B_df.to_csv(filepath)
        print(f'  - {filename} ')

        filename = 'system_composite_ssm_C.csv'
        C_df = pd.DataFrame(ssm.C, index = ssm.outputs , columns=ssm.states)
        filepath = os.path.join(outputfolder_directory, filename)
        C_df.to_csv(filepath)
        print(f'  - {filename} ')

        filename = 'system_composite_ssm_D.csv'
        D_df = pd.DataFrame(ssm.D, index = ssm.outputs , columns=ssm.inputs)
        filepath = os.path.join(outputfolder_directory, filename)
        D_df.to_csv(filepath)
        print(f'  - {filename} \n')


    def export_components_data_as_matlab_file(self, matlab_session_name = None):

        current_matlab_sessions = matlab.engine.find_matlab()

        if not matlab_session_name in current_matlab_sessions:
            print('> Initiate Matlab session, as a session was not founded or entered.')
            eng = matlab.engine.start_matlab()
        else:
            eng = matlab.engine.connect_matlab(matlab_session_name)
            print(f'> Connect to Matlab session: {matlab_session_name} ... ok.')
    
        attributes = ['system', 'generators', 'shunts', 'branches']
        for attr in attributes:
            component = getattr(self, attr)

            components_dict = data_tools.convert_class_instance_to_dictionary(component )

            eng.workspace[attr] = components_dict

        eng.quit()

