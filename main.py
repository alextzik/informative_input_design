"""
    The main loop
"""

###############################
# Dependencies
import numpy as np
import random 

import matplotlib.pyplot as plt

from dynamics import true_dynamics, Dt, b, model, model_derivative_matrix
from method import compute_map_estimate, compute_next_input
from helper_funcs import plot_confidence_ellipse, compute_log_det_Sigma
from copy import copy

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

###############################
# Parameters
# Parameters
theta_prior = np.array([0.4, 3.5])
Sigma_prior = 0.1*np.eye(2)

us = [np.array([0., 0.])]
xs = [np.array([0., 0, 0.])]
ys = [true_dynamics(x, u) for (x, u) in zip(xs, us)]

# Sigma = 0.2*np.eye(3)

# Sigma = 0.2*np.eye(2)
Sigmas = [(a/10)*np.eye(3) for a in range(2, 14, 2)]
# Sigmas = [0.2*np.eye(2)]
for i in range(5):
    A = np.random.randn(3,3)
    Sigmas += [A.T@A + 0.01*np.eye(3)]

DELTA = 0.3
num_timesteps = 30
num_iterations = 10
MAX_AMPL = 1.


###############################
# Methods
def run_method(theta_prior: np.ndarray,
                Sigma_prior: np.ndarray,
                xs: list[np.ndarray],
                us: list[np.ndarray],
                ys: list[np.ndarray],
                Sigma: np.ndarray,
                method: str,
                delta: float=DELTA,
                ) -> list[float]:
    
    delta = copy(delta)
    
    theta_prev = theta_prior.copy()
    dists = [np.linalg.norm(theta_prev - np.array([Dt, b]), ord=np.inf)]
    Sigmas_obs = [Sigma for _x in range(len(xs))]

    ax = plot_confidence_ellipse(theta_prev, Sigma_prior)

    for timestep in range(num_timesteps):
        print(timestep)

        # Loop in Algorithm 1
        if (method=="proposed" or method=="naive") and timestep>2:
            for iteration in range(num_iterations):
                theta_next = compute_map_estimate(  theta_est=theta_prev,
                                                    theta_prior=theta_prior,
                                                    Sigma_prior=Sigma_prior,
                                                    ys=ys,
                                                    xs=xs,
                                                    us=us,
                                                    Sigmas_obs=Sigmas_obs,
                                                    delta=delta
                                                    )
                # Compute posterior covariance
                Sigma_post = np.linalg.inv(
                    np.linalg.inv(Sigma_prior)  +
                    sum([model_derivative_matrix(_x, _u, theta_next).T 
                                @ np.linalg.inv(S) 
                                @ model_derivative_matrix(_x, _u, theta_next)
                        for (_x, _u, S) in zip(xs, us, Sigmas_obs)])
                )

                ax = plot_confidence_ellipse(theta_next, Sigma_post, ax)
                
            
                # Obtain model errors
                model_errors = []
                for _ in range(len(ys)):
                    model_errors += [(ys[_] - model(xs[_], us[_], theta_next)).reshape(-1,1)]
                model_errors = np.hstack(model_errors)

                # Obtain linearization errors
                linear_errors = []
                for _ in range(len(ys)):
                    linear_errors += [( model(xs[_], us[_], theta_next) - (model(xs[_], us[_], theta_prev) 
                                                                    + model_derivative_matrix(xs[_], us[_], theta_prev)@(theta_next-theta_prev))
                                        ).reshape(-1,1)]
                linear_errors = np.hstack(linear_errors)
                
                # Check if linearization errors are significant
                if (np.mean(np.linalg.norm(linear_errors, axis=0)) >= 0.5*np.mean(np.linalg.norm(model_errors, axis=0))):
                    delta = delta*0.8 # Reduce trust region

                else:
                    Sigma_obs = np.cov(linear_errors+model_errors)
                    L, Q = np.linalg.eig(Sigma_obs)
                    L[L <= np.max(L)*1e-1] = 1e-1*np.max(L)
                    Sigma_obs = Q @ np.diag(L) @ Q.T
                    Sigmas_obs = [Sigma_obs for _ in range(len(xs))]

                    Sigma_model_errors = np.cov(model_errors)

                    Sigmas_model_errors = [Sigma_model_errors for _ in range(len(xs)+1)]

                    theta_prev = theta_next.copy()
            
        else:
            Sigmas_obs = [Sigma for _x in range(len(xs))]
            theta_next = compute_map_estimate(  theta_est=theta_prev,
                                                    theta_prior=theta_prior,
                                                    Sigma_prior=Sigma_prior,
                                                    ys=ys,
                                                    xs=xs,
                                                    us=us,
                                                    Sigmas_obs=Sigmas_obs,
                                                    delta=delta
                                                    )
            # Compute posterior covariance
            Sigma_post = np.linalg.inv(
                    np.linalg.inv(Sigma_prior)  +
                    sum([model_derivative_matrix(_x, _u, theta_next).T 
                                @ np.linalg.inv(S) 
                                @ model_derivative_matrix(_x, _u, theta_next)
                        for (_x, _u, S) in zip(xs, us, Sigmas_obs)])
                )

            ax = plot_confidence_ellipse(theta_next, Sigma_post, ax)

            theta_prev = theta_next.copy()

            Sigmas_model_errors = [Sigma for _x in range(len(xs))]

        if method=="naive":
            u_next = np.random.rand(2,)
        else:
            u_next, Sigma_post = compute_next_input(theta_est=theta_next,
                                                    Sigma_prior=Sigma_prior,
                                                    xs=xs,
                                                    us=us, 
                                                    ys=ys,
                                                    Sigmas_obs=Sigmas_model_errors)
        
        logdet = compute_log_det_Sigma(theta_est=theta_next,
                                    Sigma_prior=Sigma_prior,
                                    xs=xs,
                                    us=us,
                                    Sigmas_obs=Sigmas_model_errors)
        
        if np.linalg.norm(u_next) >= MAX_AMPL:
            u_next = u_next/np.linalg.norm(u_next)*MAX_AMPL

        us = us + [u_next]
        xs = xs + [ys[-1]]
        ys = ys + [true_dynamics(xs[-1], u_next)]

        dists += [np.linalg.norm(theta_next - np.array([Dt, b]), ord=np.inf)]
    
    ax.set_aspect('equal', adjustable='box')  # Keep aspect ratio equal
    ax.grid(True)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    plt.show()

    return dists, xs, logdet


###############################
# Main loop
methods = ["proposed", "baseline"] 
results = {_: {} for _ in methods}
for _m in methods:
    results[_m]["dists"] = []
    results[_m]["xs"] = []
    results[_m]["logdet"] = []
    for Sigma in [Sigmas[0]]:
        r1, r2, r3 = run_method(theta_prior=theta_prior,
                                Sigma_prior=Sigma_prior,
                                xs=xs,
                                us=us,
                                ys=ys,
                                Sigma=Sigma,
                                method=_m)
        results[_m]["dists"] += [r1]
        results[_m]["xs"] += [r2]
        results[_m]["logdet"] += [r3]

    results[_m]["dists"] = np.vstack(results[_m]["dists"])
    results[_m]["logdet"] = np.mean(results[_m]["logdet"])
breakpoint()
# Plot error
for _ in methods:
    mean = np.mean(results[_]["dists"], axis=0)
    std = np.std(results[_]["dists"], axis=0)
    x = np.arange(results[_]["dists"].shape[1])
    
    plt.plot(range(results[_]["dists"].shape[1]), mean, label=_)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
plt.legend(loc='upper right')
plt.xlabel("Iteration")
plt.ylabel(r'$||\hat{\theta} - \theta_\mathrm{true}||_\infty$')
plt.show()

# Plot selected inputs
# Create two subplots arranged vertically (nrows=2, ncols=1)

# Plot on the first subplot
for _ in methods:
    xs = np.vstack(results[_]["xs"])
    plt.plot(xs[:, 0], xs[:, 1], label=_)
plt.xlabel(r'$x_{t, 2}$')
plt.ylabel(r'$x_{t, 1}$')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

for _ in methods:
    print(_, results[_]["logdet"])
    print()