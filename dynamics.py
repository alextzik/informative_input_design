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
    x_next = theta.reshape(2,2) @ x

    return x_next

def model_derivative_matrix(x: np.ndarray, theta:np.ndarray) -> np.ndarray:
    # For 2x2 matrix theta [theta_1, theta_2; theta_3, theta_4]
    # The derivative with respect to each parameter forms the Jacobian
    # Each column corresponds to derivative w.r.t. one parameter
    C = np.array([
        [x[0], x[1], 0, 0],      # derivatives w.r.t. theta_1, theta_2 for first equation
        [0, 0, x[0], x[1]]       # derivatives w.r.t. theta_3, theta_4 for second equation
    ])

    return C

def model_derivative_matrix_tensor(x: torch.Tensor, theta:torch.Tensor) -> torch.Tensor:
    # Ensure that the operations on x and theta are differentiable and tracked by autograd
    # For 2x2 matrix theta [theta_1, theta_2; theta_3, theta_4]
    # Each column corresponds to derivative w.r.t. one parameter
    # Build using stack/cat to preserve gradient flow (avoid torch.tensor([...]) which detaches)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    row1 = torch.stack([x[0], x[1], zero, zero])
    row2 = torch.stack([zero, zero, x[0], x[1]])
    C = torch.stack([row1, row2])

    return C 

# def model(x:np.ndarray, theta:np.ndarray) -> np.ndarray:
#     x_next = np.vstack([theta.reshape(-1,), 
#                        theta.reshape(-1,)])@x

#     return x_next

# def model_derivative_matrix(x: np.ndarray, theta:np.ndarray) -> np.ndarray:
#     C = np.vstack([
#         x,
#         x
#     ])

#     return C

# def model_derivative_matrix_tensor(x: torch.Tensor, theta:torch.Tensor) -> torch.Tensor:
#     # Ensure that the operations on x and theta are differentiable and tracked by autograd

#     # Create the matrix directly using the values computed from x and theta
#     C = torch.stack([x,
#                      x])

#     return C 