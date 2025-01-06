"""
    The functions that define our approach
"""

###############################
# Dependencies
import numpy as np
import cvxpy as cp
import torch

from dynamics import true_dynamics, model, model_derivative_matrix, model_derivative_matrix_tensor

###############################
# Parameter Estimation
def compute_map_estimate(theta_est: np.ndarray, 
                         theta_prior: np.ndarray,
                         Sigma_prior: np.ndarray,
                         ys: list[np.ndarray],
                         xs: list[np.ndarray],
                         Sigmas_obs: list[np.ndarray],
                         delta: float) -> np.ndarray:

    theta = cp.Variable(theta_est.shape[0])
    prior_objective = [cp.matrix_frac(theta - theta_prior, Sigma_prior)]
    observation_objectives = [cp.matrix_frac(
                                    y - model(x, theta_est) - model_derivative_matrix(x, theta_est)@(theta-theta_est), 
                                    Sigma_obs)
                               for (x, y, Sigma_obs) in zip(xs, ys, Sigmas_obs)]
    
    constraints = [cp.norm(theta-theta_est, "inf") <= delta*np.linalg.norm(theta_est)]

    prob = cp.Problem(cp.Minimize(sum(prior_objective+observation_objectives)), constraints)
    prob.solve(solver=cp.SCS)

    return theta.value

###############################
# Informative Input Design
def compute_next_input(theta_est: np.ndarray,
                       Sigma_prior: np.ndarray,
                       xs: list[np.ndarray],
                       Sigmas_obs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    
    # Convert numpy arrays to torch tensors of type float32
    theta_est = torch.tensor(theta_est, dtype=torch.float32, requires_grad=True)  # Ensure theta_est has requires_grad=True
    Sigma_prior = torch.tensor(Sigma_prior, dtype=torch.float32)
    Sigmas_obs = [torch.tensor(Sigma_obs, dtype=torch.float32) for Sigma_obs in Sigmas_obs]
    
    # Inverse of Sigma_obs
    inv_Sigmas_obs = [torch.inverse(Sigma_obs) for Sigma_obs in Sigmas_obs]
    
    # Compute the sum over xs for the constant part
    sum_term = sum(
        model_derivative_matrix_tensor(torch.tensor(_x, dtype=torch.float32), theta_est).T @ inv_Sigma_obs @ model_derivative_matrix_tensor(torch.tensor(_x, dtype=torch.float32), theta_est)
        for (_x, inv_Sigma_obs) in zip(xs, inv_Sigmas_obs[:-1])
    )
    
    # Constant part: inverse of Sigma_prior + sum_term
    const = torch.inverse(Sigma_prior) + sum_term
    
    # Initialize x as a torch tensor of type float32 with requires_grad=True
    x = torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)

    # Optimizer (Gradient Descent)
    optimizer = torch.optim.Adam([x], lr=0.1)  # Using Adam optimizer to maximize log_det

    # Number of optimization iterations
    num_iterations = 500

    # Gradient ascent loop (to maximize log_det)
    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Zero the gradients from the previous iteration
        
        # Compute the log-det for the current x and theta
        loss = -torch.logdet(
            const + model_derivative_matrix_tensor(x, theta_est).T @ inv_Sigmas_obs[-1] @ model_derivative_matrix_tensor(x, theta_est)
        )  # We minimize the negative log-det to maximize it
        
        # Backpropagate gradients
        loss.backward(retain_graph=True)
        # Update x using the optimizer
        optimizer.step()
        
        # Print progress
    print(f"Iteration {iteration}, Loss: {loss.item()}, x: {x.detach().numpy()}")
    Sigma_post = np.linalg.inv(
        (const + model_derivative_matrix_tensor(x, theta_est).T @ inv_Sigmas_obs[-1] @ model_derivative_matrix_tensor(x, theta_est)).detach().numpy()
    )
    return x.detach().numpy(), Sigma_post




    

