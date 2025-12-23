import numpy as np

def abc2dq0(x_a, x_b, x_c, theta):
    '''It returns value of axis d, q, and zero that results from applying the a standard
    park transformation to the three-phase quantities x_a, x_b, x_c
    
    Inputs:
    -------
    x_a (float),  x_b (float),  x_c (float): values of phase a, b, and c. They can be voltage or current.
    theta: angle applied to the standard Park transformation.
    
    Outputs:
    ------
    x_d (float), x_q (float), x_0 (float): values of phase d, q, and 0. 
    '''

    K = (2/3)*np.array([ [np.cos(theta),  np.cos(theta-2*np.pi/3), np.cos(theta+2*np.pi/3)], 
                         [ -np.sin(theta), -np.sin(theta-2*np.pi/3),-np.sin(theta+2*np.pi/3)], 
                         [ 1/2,   1/2,            1/2]])
        
    x_dq0 = np.matmul(K, np.array([ x_a, x_b, x_c ]))
    x_dq0 = 1/np.sqrt(2)*x_dq0
    x_d, x_q, x_0 = x_dq0[0], x_dq0[1], x_dq0[2]

    return x_d, x_q, x_0

def dq02abc(x_d, x_q, x_0, theta):
        
    K = np.array([ [np.cos(theta),              -np.sin(theta),                 1],
                   [np.cos(theta - 2*np.pi/3),  -np.sin(theta - 2*np.pi/3),     1],
                   [np.cos(theta + 2*np.pi/3),  -np.sin(theta + 2*np.pi/3),     1]])
    
    x_abc = np.matmul(K, np.array([ x_d, x_q, x_0 ]))
    x_abc = np.sqrt(2)*x_abc
    x_a, x_b, x_c = x_abc[0], x_abc[1], x_abc[2]

    return x_a, x_b, x_c