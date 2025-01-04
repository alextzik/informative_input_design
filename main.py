"""
    The main loop
"""

###############################
# Dependencies
import numpy as np
import cvxpy as cp
import torch 

import matplotlib.pyplot as plt

from dynamics import true_dynamics, a, b
from method import compute_map_estimate, compute_next_input
from helper_funcs import plot_confidence_ellipse

###############################
# Parameters
theta_prior = np.array([7., 4.])
Sigma_prior = 0.1*np.eye(2)

xs = [np.array([1., 1.])]
ys = [true_dynamics(x) for x in xs]

Sigma_obs = 0.5*np.eye(2)

num_timesteps = 20

###############################
# Methods
def run_proposed_method(theta_prior: np.ndarray,
                        Sigma_prior: np.ndarray,
                        xs: list[np.ndarray],
                        ys: list[np.ndarray],
                        Sigma_obs: np.ndarray) -> list[float]:
    
    theta_est = theta_prior.copy()
    dists = [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

    ax = plot_confidence_ellipse(theta_est, Sigma_prior)
    for timestep in range(num_timesteps):
        print(timestep)
        theta_est = compute_map_estimate(   theta_est=theta_est,
                                            theta_prior=theta_prior,
                                            Sigma_prior=Sigma_prior,
                                            ys=ys,
                                            xs=xs,
                                            Sigma_obs=Sigma_obs,
                                            delta=0.3)
        
        x_next, Sigma_post = compute_next_input(theta_est=theta_est,
                                    Sigma_prior=Sigma_prior,
                                    xs=xs,
                                    Sigma_obs=Sigma_obs)
        
        xs = xs + [x_next]
        ys = ys + [true_dynamics(x_next)]

        ax = plot_confidence_ellipse(theta_est, Sigma_post)

        dists += [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

    plt.show()

    return dists

def run_random_selection(theta_prior: np.ndarray,
                        Sigma_prior: np.ndarray,
                        xs: list[np.ndarray],
                        ys: list[np.ndarray],
                        Sigma_obs: np.ndarray) -> list[float]:
    dists = []
    theta_est = theta_prior.copy()
    for timestep in range(num_timesteps):
        print(timestep)
        theta_est = compute_map_estimate(   theta_est=theta_est,
                                            theta_prior=theta_prior,
                                            Sigma_prior=Sigma_prior,
                                            ys=ys,
                                            xs=xs,
                                            Sigma_obs=Sigma_obs,
                                            delta=0.3)
        
        x_next = 1*np.random.normal(size=xs[0].shape)
        
        xs = xs + [x_next]
        ys = ys + [true_dynamics(x_next)]

        dists += [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

    return dists

###############################
# Main loop
dists_proposed = run_proposed_method(theta_prior=theta_prior,
                                        Sigma_prior=Sigma_prior,
                                        xs=xs,
                                        ys=ys,
                                        Sigma_obs=Sigma_obs)
dists_random = run_random_selection(theta_prior=theta_prior,
                                        Sigma_prior=Sigma_prior,
                                        xs=xs,
                                        ys=ys,
                                        Sigma_obs=Sigma_obs)
plt.plot(range(len(dists_proposed)), dists_proposed, label="Proposed")
plt.plot(range(len(dists_random)), dists_random, label="Baseline")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("|θ_est - θ_true|")
plt.show()