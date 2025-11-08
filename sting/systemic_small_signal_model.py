from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import os

from sting.system.core import System
from scipy.linalg import block_diag
from typing import NamedTuple, Optional
from sting.utils import graph_matrices, linear_systems_tools
from sting.utils.linear_systems_tools import State_space_model


@dataclass
class Composite_small_signal_model:

    input_system: System
    ''' Power grid, including set of generators, branches, shunts and respective small signal models of each component'''

    system: Optional[State_space_model] = None
    generators: Optional[State_space_model] = None
    shunts: Optional[State_space_model] = None
    branches: Optional[State_space_model] = None

    def __post_init__(self):
        self.build_generators_composite_model()
        self.build_shunts_composite_model()
        self.build_branches_composite_model()
        self.build_system_composite_model()

    def build_generators_composite_model(self):
        print("> Build composite model of generators", end=' ')
        list_of_gentypes = self.input_system.generator_types_list

        ssm_of_gentypes = []
        T2_of_gentypes = []
        T3_of_gentypes = []

        for typ in list_of_gentypes:
            list_of_gens = getattr(self.input_system, typ)

            if list_of_gens:
                A = [g.ssm.A for g in list_of_gens]
                B = [g.ssm.B for g in list_of_gens]
                C = [g.ssm.C for g in list_of_gens]
                D = [g.ssm.D for g in list_of_gens]

                Astack = block_diag(*A)
                Bstack = block_diag(*B)
                Cstack = block_diag(*C)
                Dstack = block_diag(*D)

                n = len(list_of_gens)
                g0 = list_of_gens[0]

                nd, ng = len(g0.ssm.device_side_inputs), len(g0.ssm.grid_side_inputs)

                matrix1 = np.kron(np.eye(n), 
                            np.hstack( ( np.eye(nd), np.zeros((nd, ng)) ) ) )
                matrix2 = np.kron(np.eye(n), 
                            np.hstack( ( np.zeros((ng, nd)), np.eye(ng) ) ) )

                T1 = np.vstack((matrix1, matrix2))

                dev_inputs = [(g.idx, input) for g in list_of_gens for input in g.ssm.device_side_inputs ]
                grid_inputs = [(g.idx, input) for g in list_of_gens for input in g.ssm.grid_side_inputs ]
                states = [(g.idx, state) for g in list_of_gens for state in g.ssm.states ]
                outputs = [(g.idx, output) for g in list_of_gens for output in g.ssm.outputs ]

                ssm_of_gentypes.append(State_space_model(  A = Astack,
                                                B = np.matmul(Bstack, np.linalg.inv(T1)),
                                                C = Cstack,
                                                D = np.matmul(Dstack, np.linalg.inv(T1)),
                                                device_side_inputs = dev_inputs,
                                                grid_side_inputs = grid_inputs,
                                                states= states,
                                                outputs= outputs
                                                          ))
                T2_of_gentypes.append(np.hstack(( np.eye(n*nd), np.zeros((n*nd, n*ng)) )) )
                T3_of_gentypes.append(np.hstack(( np.zeros((n*ng, n*nd)), np.eye(n*ng) )) )

        A = block_diag(*[ssm.A for ssm in ssm_of_gentypes])
        B = block_diag(*[ssm.B for ssm in ssm_of_gentypes])
        C = block_diag(*[ssm.C for ssm in ssm_of_gentypes])
        D = block_diag(*[ssm.D for ssm in ssm_of_gentypes])

        T4 = block_diag(*T2_of_gentypes)
        T5 = block_diag(*T3_of_gentypes)
        T6 = np.vstack((T4, T5))

        B = B @ np.linalg.inv(T6)
        D = D @ np.linalg.inv(T6)

        dev_inputs = [input for ssm in ssm_of_gentypes for input in ssm.device_side_inputs ]
        grid_inputs = [input for ssm in ssm_of_gentypes for input in ssm.grid_side_inputs ]
        states = [state for ssm in ssm_of_gentypes for state in ssm.states ]
        outputs = [output for ssm in ssm_of_gentypes for output in ssm.outputs ]

        self.generators = State_space_model(   A = A,
                                    B = B,
                                    C = C,
                                    D = D,
                                    device_side_inputs = dev_inputs,
                                    grid_side_inputs = grid_inputs,
                                    states= states,
                                    outputs= outputs
                                              )
        
        print("... ok. \n")

        outputfolder_directory = os.path.join(os.getcwd(), 'outputs') 

        print("  > Creation of files with the small signal models: ")
        A_df = pd.DataFrame(A, index = states , columns=states)
        filepath = os.path.join(outputfolder_directory, 'generators_composite_ssm_A.csv')
        A_df.to_csv(filepath)
        print(f'  - {filepath} ')

        B_df = pd.DataFrame(B, index = states , columns=dev_inputs + grid_inputs)
        filepath = os.path.join(outputfolder_directory, 'generators_composite_ssm_B.csv')
        B_df.to_csv(filepath)
        print(f'  - {filepath} ')

        C_df = pd.DataFrame(C, index = outputs , columns=states)
        filepath = os.path.join(outputfolder_directory, 'generators_composite_ssm_C.csv')
        C_df.to_csv(filepath)
        print(f'  - {filepath} ')

        D_df = pd.DataFrame(D, index = outputs , columns=dev_inputs + grid_inputs)
        filepath = os.path.join(outputfolder_directory, 'generators_composite_ssm_D.csv')
        D_df.to_csv(filepath)
        print(f'  - {filepath} \n')


    def build_shunts_composite_model(self):

        print("(*) Build composite model of shunts", end=' ')
        list_of_shunttypes = self.input_system.shunt_types_list

        ssm_of_shunttypes = []

        for typ in list_of_shunttypes:
            
            list_of_shunts = getattr(self.input_system, typ)

            if list_of_shunts:
                A = [s.ssm.A for s in list_of_shunts]
                B = [s.ssm.B for s in list_of_shunts]
                C = [s.ssm.C for s in list_of_shunts]
                D = [s.ssm.D for s in list_of_shunts]

                Astack = block_diag(*A)
                Bstack = block_diag(*B)
                Cstack = block_diag(*C)
                Dstack = block_diag(*D)

                n = len(list_of_shunts)
                s0 = list_of_shunts[0]

                grid_inputs = [(s.idx, input) for s in list_of_shunts for input in s.ssm.grid_side_inputs ]
                states = [(s.idx, state) for s in list_of_shunts for state in s.ssm.states ]
                outputs = [(s.idx, output) for s in list_of_shunts for output in s.ssm.outputs ]

                ssm_of_shunttypes.append(State_space_model(  A = Astack,
                                                B = Bstack,
                                                C = Cstack,
                                                D = Dstack,
                                                grid_side_inputs = grid_inputs,
                                                states= states,
                                                outputs= outputs
                                                          ))

        A = block_diag(*[ssm.A for ssm in ssm_of_shunttypes])
        B = block_diag(*[ssm.B for ssm in ssm_of_shunttypes])
        C = block_diag(*[ssm.C for ssm in ssm_of_shunttypes])
        D = block_diag(*[ssm.D for ssm in ssm_of_shunttypes])

        grid_inputs = [input for ssm in ssm_of_shunttypes for input in ssm.grid_side_inputs ]
        states = [state for ssm in ssm_of_shunttypes for state in ssm.states ]
        outputs = [output for ssm in ssm_of_shunttypes for output in ssm.outputs ]

        self.shunts = State_space_model(   A = A,
                                            B = B,
                                            C = C,
                                            D = D,
                                            grid_side_inputs = grid_inputs,
                                            states= states,
                                            outputs= outputs
                                              )
        
        print("... ok. \n")
        
        outputfolder_directory = os.path.join(os.getcwd(), 'outputs') 
        
        print("  > Creation of files with the small signal models: ")
        A_df = pd.DataFrame(A, index = states , columns=states)
        filepath = os.path.join(outputfolder_directory, 'shunts_composite_ssm_A.csv')
        A_df.to_csv(filepath)
        print(f'  - {filepath} ')

        B_df = pd.DataFrame(B, index = states , columns=grid_inputs)
        filepath = os.path.join(outputfolder_directory, 'shunts_composite_ssm_B.csv')
        B_df.to_csv(filepath)
        print(f'  - {filepath} ')

        C_df = pd.DataFrame(C, index = outputs , columns=states)
        filepath = os.path.join(outputfolder_directory, 'shunts_composite_ssm_C.csv')
        C_df.to_csv(filepath)
        print(f'  - {filepath} ')

        D_df = pd.DataFrame(D, index = outputs , columns=grid_inputs)
        filepath = os.path.join(outputfolder_directory, 'shunts_composite_ssm_D.csv')
        D_df.to_csv(filepath)
        print(f'  - {filepath} \n')

    def build_branches_composite_model(self):

        print("(*) Build composite model of branches", end=' ')
        list_of_branchtypes = self.input_system.branch_types_list

        ssm_of_branchtypes = []

        for typ in list_of_branchtypes:
            list_of_branches = getattr(self.input_system, typ)

            if list_of_branches:
                A = [s.ssm.A for s in list_of_branches]
                B = [s.ssm.B for s in list_of_branches]
                C = [s.ssm.C for s in list_of_branches]
                D = [s.ssm.D for s in list_of_branches]

                Astack = block_diag(*A)
                Bstack = block_diag(*B)
                Cstack = block_diag(*C)
                Dstack = block_diag(*D)

                n = len(list_of_branches)
                b0 = list_of_branches[0]

                grid_inputs = [(b.idx, input) for b in list_of_branches for input in b.ssm.grid_side_inputs ]
                states = [(b.idx, state) for b in list_of_branches for state in b.ssm.states ]
                outputs = [(b.idx, output) for b in list_of_branches for output in b.ssm.outputs ]

                ssm_of_branchtypes.append(State_space_model(  A = Astack,
                                                B = Bstack,
                                                C = Cstack,
                                                D = Dstack,
                                                grid_side_inputs = grid_inputs,
                                                states= states,
                                                outputs= outputs
                                                          ))

        A = block_diag(*[ssm.A for ssm in ssm_of_branchtypes])
        B = block_diag(*[ssm.B for ssm in ssm_of_branchtypes])
        C = block_diag(*[ssm.C for ssm in ssm_of_branchtypes])
        D = block_diag(*[ssm.D for ssm in ssm_of_branchtypes])

        grid_inputs = [input for ssm in ssm_of_branchtypes for input in ssm.grid_side_inputs ]
        states = [state for ssm in ssm_of_branchtypes for state in ssm.states ]
        outputs = [output for ssm in ssm_of_branchtypes for output in ssm.outputs ]

        self.branches = State_space_model(   A = A,
                                            B = B,
                                            C = C,
                                            D = D,
                                            grid_side_inputs = grid_inputs,
                                            states= states,
                                            outputs= outputs
                                              )
        print("... ok. \n")

        outputfolder_directory = os.path.join(os.getcwd(), 'outputs') 
        
        print("  > Creation of files with the small signal models: ")
        A_df = pd.DataFrame(A, index = states , columns=states)
        filepath = os.path.join(outputfolder_directory, 'branches_composite_ssm_A.csv')
        A_df.to_csv(filepath)
        print(f'  - {filepath} ')

        B_df = pd.DataFrame(B, index = states , columns=grid_inputs)
        filepath = os.path.join(outputfolder_directory, 'branches_composite_ssm_B.csv')
        B_df.to_csv(filepath)
        print(f'  - {filepath} ')

        C_df = pd.DataFrame(C, index = outputs , columns=states)
        filepath = os.path.join(outputfolder_directory, 'branches_composite_ssm_C.csv')
        C_df.to_csv(filepath)
        print(f'  - {filepath} ')

        D_df = pd.DataFrame(D, index = outputs , columns=grid_inputs)
        filepath = os.path.join(outputfolder_directory, 'branches_composite_ssm_D.csv')
        D_df.to_csv(filepath)
        print(f'  - {filepath} \n')

    def build_system_composite_model(self):
        print("> Build composite model of the full system", end=' ')

        num_buses = len(self.input_system.bus)

        gen_bus_connections = []

        list_of_gentypes = self.input_system.generator_types_list
        for typ in list_of_gentypes:
            list_of_gens = getattr(self.input_system, typ)
            gen_bus_connections.extend([g.bus_idx for g in list_of_gens])

        branch_frombus_tobus = []
        list_of_branchtypes = self.input_system.branch_types_list
        for typ in list_of_branchtypes:
            list_of_branches = getattr(self.input_system, typ)
            branch_frombus_tobus.extend([(b.from_bus, b.to_bus) for b in list_of_branches])

        gen_cx = graph_matrices.build_generation_connection_matrix(num_buses, gen_bus_connections)
        or_inc = graph_matrices.build_oriented_incidence_matrix(num_buses, branch_frombus_tobus)

        un_inc = abs(or_inc)

        d_gen = len(self.generators.device_side_inputs)
        g_gen = len(self.generators.grid_side_inputs)
        y_gen = len(self.generators.outputs)

        g_br = len(self.branches.grid_side_inputs)
        y_br = len(self.branches.outputs)

        g_sh = len(self.shunts.grid_side_inputs)
        y_sh = len(self.shunts.outputs)

        y = y_gen + y_sh + y_br

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

        F = np.block( [   [F11, F12, F13],
                            [F21, F22, F23],
                            [F31, F32, F33],
                            [F41, F42, F43] ])
        
        G11 = np.eye(g_gen)
        G21 = np.zeros((g_gen, d_gen))
        G31 = np.zeros((g_sh, d_gen))
        G41 = np.zeros((g_br, d_gen))

        G = np.block( [[G11], 
                      [G21],
                      [G31],
                      [G41]])
        
        H = np.eye(y)
        L = np.zeros((y, d_gen))

        # Use CCM method to connect the composite systems of the generators, shunts and branches. 
        # The order matters: firt, gens, second shunts, and then branches.
        # The order matters because the interconnection matrices have been built considering that order.
        inputs = self.generators.device_side_inputs
        outputs = self.generators.outputs +self.branches.outputs + self.shunts.outputs

        ssm = linear_systems_tools.connect_models_via_CCM(F, G, H, L,
                                                            [self.generators, 
                                                             self.shunts,
                                                             self.branches,
                                                            ],
                                                            inputs=inputs, 
                                                            outputs=outputs)
        
        self.system = ssm

        print("... ok. \n")

        outputfolder_directory = os.path.join(os.getcwd(), 'outputs') 
        
        print("  > Creation of files with the small signal models: ")
        A_df = pd.DataFrame(ssm.A, index = ssm.states , columns=ssm.states)
        filepath = os.path.join(outputfolder_directory, 'system_composite_ssm_A.csv')
        A_df.to_csv(filepath)
        print(f'  - {filepath} ')

        B_df = pd.DataFrame(ssm.B, index = ssm.states , columns=ssm.inputs)
        filepath = os.path.join(outputfolder_directory, 'system_composite_ssm_B.csv')
        B_df.to_csv(filepath)
        print(f'  - {filepath} ')

        C_df = pd.DataFrame(ssm.C, index = ssm.outputs , columns=ssm.states)
        filepath = os.path.join(outputfolder_directory, 'system_composite_ssm_C.csv')
        C_df.to_csv(filepath)
        print(f'  - {filepath} ')

        D_df = pd.DataFrame(ssm.D, index = ssm.outputs , columns=ssm.inputs)
        filepath = os.path.join(outputfolder_directory, 'system_composite_ssm_D.csv')
        D_df.to_csv(filepath)
        print(f'  - {filepath} \n')




