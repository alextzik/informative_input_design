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

###############################
# Parameters
theta_prior = np.array([2., 2.])
Sigma_prior = 3*np.eye(2)
theta_est = theta_prior.copy()

xs = [np.array([1., 1.])]
ys = [true_dynamics(x) for x in xs]

Sigma_obs = 0.5*np.eye(2)

num_timesteps = 10

###############################
# Main loop
dists = []
for timestep in range(num_timesteps):
    print(timestep)
    theta_est = compute_map_estimate(theta_est=theta_est,
                                     theta_prior=theta_prior,
                                     Sigma_prior=Sigma_prior,
                                     ys=ys,
                                     xs=xs,
                                     Sigma_obs=Sigma_obs,
                                     delta=0.3)
    
    x_next = compute_next_input(theta_est=theta_est,
                                Sigma_prior=Sigma_prior,
                                xs=xs,
                                Sigma_obs=Sigma_obs)
    
    xs = xs + [x_next]
    ys = ys + [true_dynamics(x_next)]

    dists += [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

plt.plot(range(len(dists)), dists)
plt.show()