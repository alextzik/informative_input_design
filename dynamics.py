"""
    We define the true dynamics and the paremetric family of models

"""

#################################
# Dependencies

import numpy as np
import torch

###############################
# Parameters
Dt = 0.1
b = 1

###############################
# True dynamics
def true_dynamics(x: np.ndarray, u:np.ndarray) -> np.ndarray:
    x_next = np.array([
        x[0] + u[0]*Dt*np.cos(x[2]),
        b*x[1] + u[0]*Dt*np.sin(x[2]),
        x[2] + Dt*u[1]
    ])

    return x_next

###############################
# Parametric family
def model(x:np.ndarray, u:np.ndarray, theta:np.ndarray) -> np.ndarray:
    x_next = np.array([
        theta[1]*x[0]          + u[0]*theta[0]*np.cos(x[2]),
        theta[1]*x[1] + u[0]*theta[0]*np.sin(x[2]),
        x[2]          + theta[0]*u[1]
    ])
    return x_next

def model_derivative_matrix(x: np.ndarray, u:np.ndarray, theta:np.ndarray) -> np.ndarray:
    C = np.array([
        [u[0]*np.cos(x[2]), x[0]],
        [u[0]*np.sin(x[2]), x[1]],
        [u[1],              0]
    ])

    return C

def model_derivative_matrix_tensor(x: torch.Tensor, u:np.ndarray, theta:torch.Tensor) -> torch.Tensor:
    # Ensure that the operations on x and theta are differentiable and tracked by autograd
    c_00 = u[0]*torch.cos(x[2])
    c_10 = u[0]*torch.sin(x[2])
    c_11 = x[1]
    c_01 = x[0]
    c_20 = u[1]

    # Create the matrix directly using the values computed from x and theta
    C = torch.stack([torch.cat([c_00.unsqueeze(0), c_01.unsqueeze(0)]),
                     torch.cat([c_10.unsqueeze(0), c_11.unsqueeze(0)]),
                     torch.cat([c_20.unsqueeze(0), torch.tensor([0.0])])])

    return C