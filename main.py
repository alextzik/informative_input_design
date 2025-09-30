"""
    The main loop
"""

###############################
# Dependencies
import numpy as np
import random 

import matplotlib.pyplot as plt

from dynamics_henon import true_dynamics, a, b, model, model_derivative_matrix
from method import compute_map_estimate, compute_next_input
from helper_funcs import plot_confidence_ellipse, compute_log_det_Sigma
from copy import copy

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25

###############################
# Parameters
theta_prior = np.array([7., 4.])
Sigma_prior = 0.1*np.eye(2)

xs = [np.array([1., 1.]), np.array([1.1, 1.2]), np.array([1.15, 1.2])]
ys = [true_dynamics(x) for x in xs]

# Sigma = 0.2*np.eye(2)
# Sigmas = [(a/10)*np.eye(2) for a in range(40, 140, 10)]
Sigmas = [0.2*np.eye(2)]
# for i in range(20):
#     A = np.random.randn(2,2)
#     Sigmas += [A.T@A + 0.01*np.eye(2)]

DELTA = 0.3
num_timesteps = 15
num_iterations = 2

###############################
# Methods
def run_method(theta_prior: np.ndarray,
                Sigma_prior: np.ndarray,
                xs: list[np.ndarray],
                ys: list[np.ndarray],
                Sigma: np.ndarray,
                method: str,
                delta: float=DELTA,
                ) -> list[float]:
    
    delta = copy(delta)
    
    theta_prev = theta_prior.copy()
    dists = [np.linalg.norm(theta_prev - np.array([a, b]), ord=np.inf)]
    Sigmas_obs = [Sigma for _x in range(len(xs))]

    ax = plot_confidence_ellipse(theta_prev, Sigma_prior)

    for timestep in range(num_timesteps):
        print(timestep)

        # Loop in Algorithm 1
        if method=="proposed" or method=="naive":
            for iteration in range(num_iterations):
                theta_next = compute_map_estimate(  theta_est=theta_prev,
                                                    theta_prior=theta_prior,
                                                    Sigma_prior=Sigma_prior,
                                                    ys=ys,
                                                    xs=xs,
                                                    Sigmas_obs=Sigmas_obs,
                                                    delta=delta
                                                    )
                # Compute posterior covariance
                Sigma_post = np.linalg.inv(
                    np.linalg.inv(Sigma_prior)  +
                    sum([model_derivative_matrix(_x, theta_next).T 
                                @ np.linalg.inv(S) 
                                @ model_derivative_matrix(_x, theta_next)
                        for (_x, S) in zip(xs, Sigmas_obs)])
                )

                ax = plot_confidence_ellipse(theta_next, Sigma_post, ax)
                
                # Obtain model errors
                model_errors = []
                for _ in range(len(ys)):
                    model_errors += [(ys[_] - model(xs[_], theta_next)).reshape(-1,1)]
                model_errors = np.hstack(model_errors)

                # Obtain linearization errors
                linear_errors = []
                for _ in range(len(ys)):
                    linear_errors += [( model(xs[_], theta_next) - (model(xs[_], theta_prev) 
                                                                    + model_derivative_matrix(xs[_], theta_prev)@(theta_next-theta_prev))
                                        ).reshape(-1,1)]
                linear_errors = np.hstack(linear_errors)

                # Check if linearization errors are significant
                if (np.mean(np.linalg.norm(linear_errors, axis=0)) >= 0.5*np.mean(np.linalg.norm(model_errors, axis=0))):
                    delta = delta*0.8 # Reduce trust region

                else:
                    Sigma_obs = np.cov(linear_errors+model_errors)
                    # make sure Sigma_obs is positive definite
                    Sigma_obs += 0.01*np.eye(Sigma_obs.shape[0])
                    Sigmas_obs = [Sigma_obs for _ in range(len(xs)+1)]

                    Sigma_model_errors = np.cov(model_errors)

                    Sigmas_model_errors = [Sigma_model_errors for _ in range(len(xs)+1)]

                    theta_prev = theta_next.copy()

        else:
            Sigmas_obs = [Sigma for _x in range(len(xs))]
            for iteration in range(num_iterations):
                theta_next = compute_map_estimate(  theta_est=theta_prev,
                                                        theta_prior=theta_prior,
                                                        Sigma_prior=Sigma_prior,
                                                        ys=ys,
                                                        xs=xs,
                                                        Sigmas_obs=Sigmas_obs,
                                                        delta=delta
                                                        )
                theta_prev = theta_next.copy()

            # Compute posterior covariance
            Sigma_post = np.linalg.inv(
                    np.linalg.inv(Sigma_prior)  +
                    sum([model_derivative_matrix(_x, theta_next).T 
                                @ np.linalg.inv(S) 
                                @ model_derivative_matrix(_x, theta_next)
                        for (_x, S) in zip(xs, Sigmas_obs)])
                )

            ax = plot_confidence_ellipse(theta_next, Sigma_post, ax)

            theta_prev = theta_next.copy()

            Sigmas_model_errors = [Sigma for _x in range(len(xs))]

        if method=="naive":
            x_next = np.random.rand(2,)
        else:
            x_next, Sigma_post, min_Sigma_post = compute_next_input(theta_est=theta_next,
                                                    Sigma_prior=Sigma_prior,
                                                    xs=xs,
                                                    Sigmas_obs=Sigmas_model_errors)
        
        logdet = compute_log_det_Sigma(theta_est=theta_next,
                                    Sigma_prior=Sigma_prior,
                                    xs=xs,
                                    Sigmas_obs=Sigmas_model_errors)

        xs = xs + [x_next]
        ys = ys + [true_dynamics(x_next)]

        dists += [np.linalg.norm(theta_next - np.array([a, b]), ord=np.inf)]
    
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
    for Sigma in Sigmas:
        r1, r2, r3 = run_method(theta_prior=theta_prior,
                                Sigma_prior=Sigma_prior,
                                xs=xs,
                                ys=ys,
                                Sigma=Sigma,
                                method=_m)
        results[_m]["dists"] += [r1]
        results[_m]["xs"] += [r2]
        results[_m]["logdet"] += [r3]

    results[_m]["dists"] = np.vstack(results[_m]["dists"])
    results[_m]["logdet"] = np.mean(results[_m]["logdet"])

# Plot error
plt.figure(figsize=(10, 6))
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
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

# Plot on the first subplot
for _ in methods:
    xs = np.vstack(results[_]["xs"])
    axs[0].plot(range(len(xs[:, 0])), xs[:, 0], label=_)
    axs[1].plot(range(len(xs[:, 1])), xs[:, 1], label=_)
axs[0].set_title('Input Coordinate 1')
axs[0].set_xlabel('timestep')
# axs[0].legend(loc='upper right')
axs[1].set_title('Input Coordinate 2')
axs[1].set_xlabel('timestep')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

for _ in methods:
    print(_, results[_]["logdet"])
    print()