"""
    We define the true dynamics and the paremetric family of models

"""

#################################
# Dependencies

import numpy as np
import torch

###############################
# Parameters
a = 1.4
b = 0.3

###############################
# True dynamics
def true_dynamics(x: np.ndarray) -> np.ndarray:
    x_next = np.array([
        1 - a*x[0]**2 + x[1],
        b*x[0]
    ])

    return x_next

###############################
# Parametric family
def model(x:np.ndarray, theta:np.ndarray) -> np.ndarray:
    x_next = np.array([
        1 - theta[0]*x[0]**2 + x[1],
        theta[1]*x[0]
    ])

    return x_next

def model_derivative_matrix(x: np.ndarray, theta:np.ndarray) -> np.ndarray:
    C = np.array([
        [-x[0]**2, 0],
        [0, x[0]]
    ])

    return C

def model_derivative_matrix_tensor(x: torch.Tensor, theta:torch.Tensor) -> torch.Tensor:
    # Ensure that the operations on x and theta are differentiable and tracked by autograd
    c_00 = -x[0]**2
    c_11 = x[0] 

    # Create the matrix directly using the values computed from x and theta
    C = torch.stack([torch.cat([c_00.unsqueeze(0), torch.tensor([0.0])]),
                     torch.cat([torch.tensor([0.0]), c_11.unsqueeze(0)])])

    return C

    return C