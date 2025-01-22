"""
    The main loop
"""

###############################
# Dependencies
import numpy as np
import random 

import matplotlib.pyplot as plt

from dynamics import true_dynamics, a, b, model
from method import compute_map_estimate, compute_next_input
from helper_funcs import plot_confidence_ellipse, compute_log_det_Sigma

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15

###############################
# Parameters
theta_prior = np.array([3., 1.])
Sigma_prior = 0.1*np.eye(2)

xs = [np.array([1., 1.])]
ys = [true_dynamics(x) for x in xs]

Sigma_obs = 10*np.eye(2)

num_timesteps = 50
NUMBER_OF_SINS = 30
MAX_AMPL = 7.

###############################
# Methods
def run_proposed_method(theta_prior: np.ndarray,
                        Sigma_prior: np.ndarray,
                        xs: list[np.ndarray],
                        ys: list[np.ndarray],
                        Sigma_obs: np.ndarray) -> list[float]:
    
    theta_est = theta_prior.copy()
    dists = [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]
    Sigmas_obs = [Sigma_obs for _x in range(len(xs))]

    ax = plot_confidence_ellipse(theta_est, Sigma_prior)

    for timestep in range(num_timesteps):
        print(timestep)
        theta_est = compute_map_estimate(   theta_est=theta_est,
                                            theta_prior=theta_prior,
                                            Sigma_prior=Sigma_prior,
                                            ys=ys,
                                            xs=xs,
                                            Sigmas_obs=[Sigma_obs for _x in range(len(xs))],#Sigmas_obs,
                                            )

        delta_ys = []
        for _ in range(len(ys)):
            delta_ys += [(model(xs[_], theta_est) - ys[_]).reshape(-1,1)]
        Sigmas_obs = [0.8*1/len(delta_ys) * np.sum([_y @ _y.T for _y in delta_ys], axis=0) + 0.2*Sigma_obs 
                        for _ in range(len(xs)+1)]

        x_next, Sigma_post = compute_next_input(theta_est=theta_est,
                                    Sigma_prior=Sigma_prior,
                                    xs=xs,
                                    Sigmas_obs=Sigmas_obs)
        
        if np.linalg.norm(x_next) >= MAX_AMPL:
            x_next = x_next/np.linalg.norm(x_next)*MAX_AMPL
        
        xs = xs + [x_next]
        ys = ys + [true_dynamics(x_next)]

        ax = plot_confidence_ellipse(theta_est, Sigma_post, ax)

        dists += [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]
    
    ax.set_aspect('equal', adjustable='box')  # Keep aspect ratio equal
    ax.grid(True)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    plt.show()

    logdet = compute_log_det_Sigma(theta_est=theta_est,
                                   Sigma_prior=Sigma_prior,
                                   xs=xs,
                                   Sigmas_obs=[Sigma_obs for _x in range(len(xs))])
    return dists, xs, logdet


def run_proposed_method_no_Sigma_update(theta_prior: np.ndarray,
                        Sigma_prior: np.ndarray,
                        xs: list[np.ndarray],
                        ys: list[np.ndarray],
                        Sigma_obs: np.ndarray) -> list[float]:
    
    theta_est = theta_prior.copy()
    dists = [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

    for timestep in range(num_timesteps):
        print(timestep)
        theta_est = compute_map_estimate(   theta_est=theta_est,
                                            theta_prior=theta_prior,
                                            Sigma_prior=Sigma_prior,
                                            ys=ys,
                                            xs=xs,
                                            Sigmas_obs=[Sigma_obs for _x in range(len(xs))],
                                            )
        
        x_next, Sigma_post = compute_next_input(theta_est=theta_est,
                                    Sigma_prior=Sigma_prior,
                                    xs=xs,
                                    Sigmas_obs=[Sigma_obs for _x in range(len(xs)+1)])
        
        if np.linalg.norm(x_next) >= MAX_AMPL:
            x_next = x_next/np.linalg.norm(x_next)*MAX_AMPL
        
        xs = xs + [x_next]
        ys = ys + [true_dynamics(x_next)]

        dists += [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

    logdet = compute_log_det_Sigma(theta_est=theta_est,
                                   Sigma_prior=Sigma_prior,
                                   xs=xs,
                                   Sigmas_obs=[Sigma_obs for _x in range(len(xs))])

    return dists, xs, logdet


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
                                            Sigmas_obs=[Sigma_obs for _x in range(len(xs))],
                                            )
        
        x_next = 1*np.random.normal(size=xs[0].shape)

        if np.linalg.norm(x_next) >= MAX_AMPL:
            x_next = x_next/np.linalg.norm(x_next)*MAX_AMPL
        
        xs = xs + [x_next]
        ys = ys + [true_dynamics(x_next)]

        dists += [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

    logdet = compute_log_det_Sigma(theta_est=theta_est,
                                   Sigma_prior=Sigma_prior,
                                   xs=xs,
                                   Sigmas_obs=[Sigma_obs for _x in range(len(xs))])

    return dists, xs, logdet

def run_prbs(theta_prior: np.ndarray,
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
                                            Sigmas_obs=[Sigma_obs for _x in range(len(xs))],
                                            )
        
        x_next = np.array([random.choice([0., 1.]) for _ in range(len(xs[0]))]).reshape(xs[0].shape)

        if np.linalg.norm(x_next) >= MAX_AMPL:
            x_next = x_next/np.linalg.norm(x_next)*MAX_AMPL
        
        xs = xs + [x_next]
        ys = ys + [true_dynamics(x_next)]

        dists += [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

    logdet = compute_log_det_Sigma(theta_est=theta_est,
                                   Sigma_prior=Sigma_prior,
                                   xs=xs,
                                   Sigmas_obs=[Sigma_obs for _x in range(len(xs))])

    return dists, xs, logdet

def run_multisine(theta_prior: np.ndarray,
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
                                            Sigmas_obs=[Sigma_obs for _x in range(len(xs))],
                                            )
        
        x_next = np.array([
            np.sum([np.random.uniform(-1,1)*np.sin(2*np.pi*(2*j+0)*(timestep+1)/num_timesteps + np.random.uniform(0, 2*np.pi)) for j in range(NUMBER_OF_SINS)]),
            np.sum([np.random.uniform(-1,1)*np.sin(2*np.pi*(2*j+1)*(timestep+1)/num_timesteps + np.random.uniform(0, 2*np.pi)) for j in range(NUMBER_OF_SINS)])
        ]).reshape(xs[0].shape)

        if np.linalg.norm(x_next) >= MAX_AMPL:
            x_next = x_next/np.linalg.norm(x_next)*MAX_AMPL
        
        xs = xs + [x_next]
        ys = ys + [true_dynamics(x_next)]

        dists += [np.linalg.norm(theta_est - np.array([a, b]), ord=np.inf)]

    logdet = compute_log_det_Sigma(theta_est=theta_est,
                                   Sigma_prior=Sigma_prior,
                                   xs=xs,
                                   Sigmas_obs=[Sigma_obs for _x in range(len(xs))])

    return dists, xs, logdet

###############################
# Main loop
methods = ["proposed", "random", "PRBS", "multisine"]
results = {_: {} for _ in methods}
results["proposed"]["dists"], results["proposed"]["xs"], results["proposed"]["logdet"] = run_proposed_method(theta_prior=theta_prior,
                                                                    Sigma_prior=Sigma_prior,
                                                                    xs=xs,
                                                                    ys=ys,
                                                                    Sigma_obs=Sigma_obs)
# dists_proposed_no_Sigma_update = run_proposed_method_no_Sigma_update(theta_prior=theta_prior,
#                                         Sigma_prior=Sigma_prior,
#                                         xs=xs,
#                                         ys=ys,
#                                         Sigma_obs=Sigma_obs)
results["random"]["dists"], results["random"]["xs"], results["random"]["logdet"] = run_random_selection(theta_prior=theta_prior,
                                                                    Sigma_prior=Sigma_prior,
                                                                    xs=xs,
                                                                    ys=ys,
                                                                    Sigma_obs=Sigma_obs)
results["PRBS"]["dists"], results["PRBS"]["xs"], results["PRBS"]["logdet"] = run_prbs(theta_prior=theta_prior,
                                                                    Sigma_prior=Sigma_prior,
                                                                    xs=xs,
                                                                    ys=ys,
                                                                    Sigma_obs=Sigma_obs)
results["multisine"]["dists"], results["multisine"]["xs"], results["multisine"]["logdet"] = run_multisine(theta_prior=theta_prior,
                                                                    Sigma_prior=Sigma_prior,
                                                                    xs=xs,
                                                                    ys=ys,
                                                                    Sigma_obs=Sigma_obs)

# Plot error
for _ in methods:
    plt.plot(range(len(results[_]["dists"])), results[_]["dists"], label=_)
plt.legend(loc='upper right')
plt.xlabel("Iteration")
plt.ylabel(r'$||\hat{\theta} - \theta_\mathrm{true}||_\infty$')
plt.show()

# Plot selected inputs
# Create two subplots arranged vertically (nrows=2, ncols=1)
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(3, 4))

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