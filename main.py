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
plt.rcParams['font.size'] = 25

###############################
# Parameters
thetas_prior = [np.array([0.4, 2.5])]
Sigmas_prior = [0.5*np.eye(2)]

us = [np.array([0., 0.]), np.array([0.3, 0.1]), np.array([0.2, 0.1])]
xs = [np.array([0., 0, 0.]), np.array([0.1, 0., 0.]), np.array([0.25, 0.05, 0.01])]
ys = [true_dynamics(x, u) for (x, u) in zip(xs, us)]

Sigmas = [0.2*np.eye(3)]
# Sigmas = [(a/10)*np.eye(2) for a in range(40, 140, 10)]
for i in range(30):
    A = np.random.randn(3,3)
    Sigmas += [A.T@A + 0.01*np.eye(3)]

    thetas_prior += [np.array([random.uniform(0,2), random.uniform(0,2)])]

    A = 3*np.random.randn(2,2)
    Sigmas_prior += [A.T@A + 0.01*np.eye(2)]

DELTA = 0.3
MAX_AMPL = 1.
num_timesteps = 30
num_iterations = 10

###############################
# Methods
def run_method(theta_prior: np.ndarray,
                Sigma_prior: np.ndarray,
                xs: list[np.ndarray],
                ys: list[np.ndarray],
                us: list[np.ndarray],
                Sigma: np.ndarray,
                method: str,
                delta: float=DELTA,
                ) -> list[float]:
    
    delta = copy(delta)
    
    theta_prev = theta_prior.copy()
    dists = [np.linalg.norm(theta_prev - np.array([Dt, b]), ord=np.inf)]
    errors_dict = {}
    errors_dict["model_errors"] = []
    errors_dict["linearization_errors"] = []
    log_dets = []
    Sigmas_obs = [Sigma for _x in range(len(xs))]

    # ax = plot_confidence_ellipse(theta_prev, Sigma_prior)

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
                                                    us=us,
                                                    Sigmas_obs=Sigmas_obs,
                                                    delta=delta
                                                    )
                # Compute posterior covariance
                Sigma_post = np.linalg.inv(
                    np.linalg.inv(Sigma_prior)  +
                    sum([model_derivative_matrix(_x, _u, theta_next).T 
                                @ np.linalg.inv(S) 
                                @ model_derivative_matrix(_x, _u,theta_next)
                        for (_x, _u, S) in zip(xs, us, Sigmas_obs)])
                )

                # ax = plot_confidence_ellipse(theta_next, Sigma_post, ax)
                
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
                                                    us=us,
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

            # ax = plot_confidence_ellipse(theta_next, Sigma_post, ax)

            theta_prev = theta_next.copy()

            Sigmas_model_errors = [Sigma for _x in range(len(xs))]

        if method=="naive":
            x_next = np.random.rand(2,)
        else:
            u_next, Sigma_post = compute_next_input(theta_est=theta_next,
                                                    Sigma_prior=Sigma_prior,
                                                    xs=xs,
                                                    us=us, 
                                                    ys=ys,
                                                    Sigmas_obs=Sigmas_model_errors)

        if np.abs(u_next[0]) > MAX_AMPL:
            u_next[0] = np.sign(u_next[0])*MAX_AMPL/np.abs(u_next[0])
        # wrap u_next[1] to be within [-pi, pi]
        u_next[1] = u_next[1]%(2*np.pi)

        us = us + [u_next]
        xs = xs + [ys[-1]]
        ys = ys + [true_dynamics(xs[-1], u_next)]

        dists += [np.linalg.norm(theta_next - np.array([Dt, b]), ord=np.inf)]
        errors_dict["model_errors"] += [np.mean(np.linalg.norm(model_errors, axis=0))]
        errors_dict["linearization_errors"] += [np.mean(np.linalg.norm(linear_errors, axis=0))]
        log_dets += [np.linalg.slogdet(Sigmas_model_errors[0])[1]]

    # ax.set_aspect('equal', adjustable='box')  # Keep aspect ratio equal
    # ax.grid(True)
    # ax.set_xlabel(r'$\theta_1$')
    # ax.set_ylabel(r'$\theta_2$')
    # plt.show()

    return dists, us, errors_dict, log_dets


###############################
# Main loop
methods = ["proposed"] 
results = {_: {} for _ in methods}
j = 0
for _m in methods:
    results[_m]["dists"] = []
    results[_m]["xs"] = []
    results[_m]["errors_dict"] = {}
    results[_m]["errors_dict"]["model_errors"] = []
    results[_m]["errors_dict"]["linearization_errors"] = []
    results[_m]["log_dets"] = []
    for Sigma, theta_prior, Sigma_prior in zip(Sigmas, thetas_prior, Sigmas_prior): 
        j += 1
        print(f"Run {j}")
        r1, r2, r3, r4 = run_method(theta_prior=theta_prior,
                                Sigma_prior=Sigma_prior,
                                xs=xs,
                                ys=ys,
                                us=us,
                                Sigma=Sigma,
                                method=_m)
        results[_m]["dists"] += [r1]
        results[_m]["xs"] += [r2]
        results[_m]["errors_dict"]["model_errors"] += [r3["model_errors"]]
        results[_m]["errors_dict"]["linearization_errors"] += [r3["linearization_errors"]]
        results[_m]["log_dets"] += [r4]

    results[_m]["dists"] = np.vstack(results[_m]["dists"])
    results[_m]["errors_dict"]["model_errors"] = np.vstack(results[_m]["errors_dict"]["model_errors"])
    results[_m]["errors_dict"]["linearization_errors"] = np.vstack(results[_m]["errors_dict"]["linearization_errors"])
    results[_m]["log_dets"] = np.vstack(results[_m]["log_dets"])

# Plot error
plt.figure(figsize=(10, 6))
for _ in methods:
    mean = np.mean(results[_]["dists"], axis=0)
    std = np.std(results[_]["dists"], axis=0)
    x = np.arange(results[_]["dists"].shape[1])
    
    plt.plot(range(results[_]["dists"].shape[1]), mean, label=_)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
# plt.legend(loc='upper right')
plt.xlabel("Iteration")
plt.ylabel(r'$||\hat{\theta} - \theta_\mathrm{true}||_\infty$')
plt.tight_layout()
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
axs[0].set_xlabel('Time')
# axs[0].legend(loc='upper right')
axs[1].set_title('Input Coordinate 2')
axs[1].set_xlabel('Time')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# plot logdet
plt.figure(figsize=(10, 6))
for _ in methods:
    mean = np.mean(results[_]["errors_dict"]["model_errors"], axis=0)
    std = np.std(results[_]["errors_dict"]["model_errors"], axis=0)
    x = np.arange(mean.shape[0])
    plt.plot(x, mean, label="Model")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    mean = np.mean(results[_]["errors_dict"]["linearization_errors"], axis=0)
    std = np.std(results[_]["errors_dict"]["linearization_errors"], axis=0)
    plt.plot(x, mean, label="Linearization")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)

plt.xlabel("Time")
plt.ylabel("Average Error Norm")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# plot logdet
plt.figure(figsize=(10, 6))
for _ in methods:
    mean = np.mean(results[_]["log_dets"], axis=0)
    std = np.std(results[_]["log_dets"], axis=0)
    x = np.arange(mean.shape[0])
    plt.plot(x, mean, label="Model Error Covariance")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)

plt.xlabel("Time")
plt.ylabel("Log Determinant")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
