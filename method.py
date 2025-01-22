"""
    The functions that define our approach
"""

###############################
# Dependencies
import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dynamics import model, model_derivative_matrix, model_derivative_matrix_tensor

###############################
# Parameter Estimation
def compute_map_estimate(theta_est: np.ndarray, 
                         theta_prior: np.ndarray,
                         Sigma_prior: np.ndarray,
                         ys: list[np.ndarray],
                         xs: list[np.ndarray],
                         Sigmas_obs: list[np.ndarray],
                         delta=0.3) -> np.ndarray:

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
        model_derivative_matrix_tensor(
            torch.tensor(_x, dtype=torch.float32), theta_est).T 
        @ inv_Sigma_obs 
        @ model_derivative_matrix_tensor(torch.tensor(_x, dtype=torch.float32), theta_est)
        for (_x, inv_Sigma_obs) in zip(xs, inv_Sigmas_obs[:-1])
    )
    
    # Constant part: inverse of Sigma_prior + sum_term
    const = torch.inverse(Sigma_prior) + sum_term
    
    # Initialize x as a torch tensor of type float32 with requires_grad=True
    x = torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)

    # Optimizer (Gradient Descent)
    optimizer = torch.optim.Adam([x], lr=0.1)  # Using Adam optimizer to maximize log_det

    # Number of optimization iterations
    num_iterations = 1000

    # Gradient ascent loop (to maximize log_det)
    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Zero the gradients from the previous iteration
        
        # Compute the log-det for the current x and theta
        loss = -torch.logdet(
            const + 
            model_derivative_matrix_tensor(x, theta_est).T 
            @ inv_Sigmas_obs[-1] 
            @ model_derivative_matrix_tensor(x, theta_est)
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


def fit_model(theta_prior: np.ndarray,
                xs: list[np.ndarray],
                ys: list[np.ndarray]) -> np.ndarray:
    xs = xs.copy()
    ys = ys.copy()
    xs = [x.reshape(1,-1) for x in xs]
    ys = [y.reshape(1,-1) for y in ys]

    X = torch.tensor(np.vstack(xs), dtype=torch.float32)
    Y = torch.tensor(np.vstack(ys), dtype=torch.float32)
    
    # Define the Hénon map model
    class HenonModel(nn.Module):
        def __init__(self):
            super(HenonModel, self).__init__()
            # Parameters a and b as learnable parameters
            self.a = nn.Parameter(torch.tensor(theta_prior[0]))  # Initial guess for a
            self.b = 1.0  # b is fixed
            self.c = nn.Parameter(torch.tensor(theta_prior[1])) 

        def forward(self, x, y):
            # Apply the Hénon map equations
            x_next = 1 - self.a * x**2 + self.b * y
            y_next = self.c * x
            return torch.stack([x_next, y_next], dim=1)
        
    # Instantiate the model, loss function, and optimizer
    model = HenonModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model without batching (using the entire dataset)
    epochs = 10000
    losses = []

    for epoch in range(epochs):
        model.train()
        
        # Forward pass using the entire dataset
        output = model(X[:, 0], X[:, 1])  # Separate x and y components
        loss = criterion(output, Y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Plot the training loss curve
    # plt.plot(losses)
    # plt.title('Training Loss Curve')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.show()
    
    return np.array([model.a.item(), model.c.item()])
    

