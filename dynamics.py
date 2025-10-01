"""
    We define the true dynamics and the paremetric family of models

"""

#################################
# Dependencies

import numpy as np
import torch

###############################
# Parameters
A = np.array([[1., 2.], [3., 4.]])

###############################
# True dynamics
def true_dynamics(x: np.ndarray) -> np.ndarray:
    x_next = A@x

    return x_next

###############################
# Parametric family
def model(x:np.ndarray, theta:np.ndarray) -> np.ndarray:
    x_next = np.array([
        [theta[0], theta[1]],
        [theta[2], theta[3]]
    ])@x

    return x_next

def model_derivative_matrix(x: np.ndarray, theta:np.ndarray) -> np.ndarray:
    C = np.array([
        [x[0], x[1], 0., 0.],
        [0., 0., x[0], x[1]]
    ])

    return C

def model_derivative_matrix_tensor(x: torch.Tensor, theta:torch.Tensor) -> torch.Tensor:
    # Ensure that the operations on x and theta are differentiable and tracked by autograd
    c_00 = x[0]
    c_01 = x[1]
    c_10 = x[0]
    c_11 = x[1] 

    # Create the matrix directly using the values computed from x and theta
    C = torch.stack([torch.cat([c_00.unsqueeze(0), c_01.unsqueeze(0), torch.tensor([0.0]), torch.tensor([0.0])]),
                     torch.cat([torch.tensor([0.0]), torch.tensor([0.0]), c_10.unsqueeze(0), c_11.unsqueeze(0)])])

    return C