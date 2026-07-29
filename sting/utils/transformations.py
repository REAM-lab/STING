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

def dq2DQ(x_d: float, x_q: float, theta: float) -> tuple[float, float]:
    """
    Transforms dq coordinates to DQ coordinates.
    dq frame is usually considered as the device reference frame,
    while DQ frame is usually considered as the grid reference frame.
    
    Inputs:
    - x_d (float): value of quantity, e.g, voltage or current, of axis d.
    - x_q (float): value of quantity, e.g, voltage or current, of axis q.
    - theta (float): angle in radians applied to the transformation.

    Outputs:
    ------
    x_D (float): value of axis D.
    x_Q (float): value of axis Q.
    """

    R = np.array([
            [ np.cos(theta),-np.sin(theta) ],
            [ np.sin(theta), np.cos(theta) ]
    ])

    x_DQ = np.matmul(R, np.array([ x_d, x_q ]))
    x_D, x_Q = x_DQ[0], x_DQ[1]

    return x_D, x_Q

def DQ2dq(x_D: float, x_Q: float, theta: float) -> tuple[float, float]:
    """
    Transforms DQ coordinates to dq coordinates.
    DQ frame is usually considered as the grid reference frame,
    while dq frame is usually considered as the device reference frame.
    
    Inputs:
    - x_D (float): value of quantity, e.g, voltage or current, of axis D.
    - x_Q (float): value of quantity, e.g, voltage or current, of axis Q.
    - theta (float): angle in radians applied to the transformation.

    Outputs:
    ------
    x_d (float): value of axis d.
    x_q (float): value of axis q.
    """

    R = np.array([
            [ np.cos(theta), np.sin(theta) ],
            [-np.sin(theta), np.cos(theta) ]
    ])

    x_dq = np.matmul(R, np.array([ x_D, x_Q ]))
    x_d, x_q = x_dq[0], x_dq[1]

    return x_d, x_q



def d_dq2DQ_dangle(x_d: float, x_q: float, theta: float) -> np.ndarray:
    """
    Returns the derivatives of the transformation dq2DQ with respect to the angle theta.
    dq frame is usually considered as the device reference frame,
    while DQ frame is usually considered as the grid reference frame.

    Inputs:
    - x_d (float): initial condition of quantity of axis d.
    - x_q (float): initial condition of quantity of axis q.
    - theta (float): angle in radians.

    Returns:
    - d_dq2DQ_dangle (numpy.ndarray): derivatives of the transformation with respect to the angle theta.
    it is vector of size 2x1.
    """

    U =  np.array([
            [ -np.sin(theta),-np.cos(theta) ],
            [ np.cos(theta), -np.sin(theta) ]
    ])

    d_dq2DQ_dangle = np.matmul(U, np.array([ x_d, x_q ]))

    return d_dq2DQ_dangle

def d_DQ2dq_dangle(x_D: float, x_Q: float, theta: float) -> np.ndarray:
    """
    Returns the derivatives of the transformation DQ2dq with respect to the angle theta.
    DQ frame is usually considered as the grid reference frame,
    while dq frame is usually considered as the device reference frame.

    Inputs:
    - x_D (float): initial condition of quantity of axis D.
    - x_Q (float): initial condition of quantity of axis Q.
    - theta (float): angle in radians.

    Returns:
    - d_DQ2dq_dangle (numpy.ndarray): derivatives of the transformation with respect to the angle theta.
    it is vector of size 2x1.
    """

    U =  np.array([
            [ -np.sin(theta), np.cos(theta) ],
            [ -np.cos(theta),-np.sin(theta) ]
    ])

    d_DQ2dq_dangle = np.matmul(U, np.array([ x_D, x_Q ]))

    return d_DQ2dq_dangle